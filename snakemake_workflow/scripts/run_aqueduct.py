"""Run the flood model for a single tile, return period and SLR scenario.

Engine selected by `simulation.engine` (config.yml): "python" (the
validated dense+3-sweep port, `flood_model.flood_depth_dense` - default) or
"julia" (the original compiled `aqueduct.exe`, kept as a fallback/comparison
option). See `aqueduct_runner.py` for the two engines' shared helpers, and
`scripts/run_aqueduct_cli.py` for the equivalent HPC/sbatch entry point -
keep the two in sync.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct_runner import (  # noqa: E402
    NO_STATIONS_REASON,
    is_oom_error,
    log_run_timing,
    log_skipped_tile,
    mark_tile_oom,
    print_simulation_progress,
    run_aqueduct,
    run_aqueduct_python,
    tile_marked_oom,
    tile_output_complete,
)
from config_utils import retry_transient_io  # noqa: E402
from rasters import save_nodata_raster  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

model_outputs = snakemake.params.model_outputs  # noqa: F821
skipped_dir = os.path.join(model_outputs, "skipped_tiles")
oom_dir = os.path.join(model_outputs, "oom_tiles")
timings_dir = os.path.join(model_outputs, "run_timings")
tile_id = snakemake.wildcards.tile_id  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_name = f"{return_period}_{waterlevel_name}"
engine = snakemake.params.engine  # noqa: F821

boundaries = retry_transient_io(gpd.read_file, snakemake.input.boundaries)  # noqa: F821

with open(snakemake.input.crop_info, encoding="utf-8") as f:  # noqa: F821
    crop_info = json.load(f)

if boundaries.empty:
    log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=NO_STATIONS_REASON)
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
elif crop_info["empty_candidate"]:
    log_skipped_tile(
        skipped_dir,
        tile_id,
        scenario_name,
        reason="flood_extent_crop: no cell in this tile can flood in any scenario (elevation exceeds the highest boundary water level everywhere)",
    )
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
elif tile_marked_oom(oom_dir, tile_id):
    log_skipped_tile(
        skipped_dir,
        tile_id,
        scenario_name,
        reason=f"tile too large for the {engine} engine (out-of-memory on a previous return period/SLR scenario for this tile)",
    )
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
else:
    start = time.perf_counter()
    succeeded = False

    if engine == "python":
        flooding_config = snakemake.params.flooding_config  # noqa: F821
        try:
            run_aqueduct_python(
                snakemake.input.dem, snakemake.input.mask, snakemake.input.friction,  # noqa: F821
                snakemake.input.boundaries, snakemake.output.waterdepth,  # noqa: F821
                resolution=flooding_config["resolution"], k=flooding_config["knn"],
                variable=waterlevel_name,
            )
        except MemoryError:
            mark_tile_oom(oom_dir, tile_id, reason="MemoryError in flood_depth_dense - tile too large for the python engine")
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason="MemoryError - tile too large for the python engine")
            save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
        else:
            succeeded = True
    else:  # engine == "julia"
        try:
            run_aqueduct(snakemake.params.aqueduct_executable, snakemake.input.toml)  # noqa: F821
        except subprocess.CalledProcessError as e:
            if not is_oom_error(e):
                raise
            mark_tile_oom(oom_dir, tile_id, reason="OutOfMemoryError in component_indices (core.jl) - tile too large for Aqueduct")
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason="OutOfMemoryError - tile too large for Aqueduct")
            save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
        else:
            succeeded = True

    if succeeded:
        log_run_timing(
            timings_dir, tile_id, return_period, waterlevel_name,
            elapsed_s=time.perf_counter() - start,
            cropped_pixels=crop_info["cropped_pixels"],
            original_pixels=crop_info["original_pixels"],
            crop_enabled=crop_info["enabled"],
        )

# Only a per-tile milestone (not printed for every scenario job): once this
# tile's own output count reaches n_scenarios_per_tile, do the one O(n_tiles)
# scan across the whole grid and report the running totals.
n_scenarios_per_tile = snakemake.params.n_scenarios_per_tile  # noqa: F821
if tile_output_complete(model_outputs, tile_id, n_scenarios_per_tile):
    tile_ids = load_tile_grid(snakemake.params.tile_grid_path)["tile_id"].astype(int).tolist()  # noqa: F821
    print_simulation_progress(model_outputs, oom_dir, skipped_dir, tile_ids, n_scenarios_per_tile)
