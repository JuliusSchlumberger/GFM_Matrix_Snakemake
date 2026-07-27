"""Run the Aqueduct flood model for a single tile, return period and SLR scenario."""

import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct_runner import (  # noqa: E402
    NO_STATIONS_REASON,
    is_oom_error,
    log_skipped_tile,
    mark_tile_oom,
    print_simulation_progress,
    run_aqueduct,
    tile_marked_oom,
    tile_output_complete,
)
from config_utils import retry_transient_io  # noqa: E402
from rasters import save_nodata_raster  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

model_outputs = snakemake.params.model_outputs  # noqa: F821
skipped_dir = os.path.join(model_outputs, "skipped_tiles")
oom_dir = os.path.join(model_outputs, "oom_tiles")
tile_id = snakemake.wildcards.tile_id  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_name = f"{return_period}_{waterlevel_name}"

boundaries = retry_transient_io(gpd.read_file, snakemake.input.boundaries)  # noqa: F821

if boundaries.empty:
    log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=NO_STATIONS_REASON)
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
elif tile_marked_oom(oom_dir, tile_id):
    log_skipped_tile(
        skipped_dir,
        tile_id,
        scenario_name,
        reason="tile too large for Aqueduct (OutOfMemoryError on a previous return period/SLR scenario for this tile)",
    )
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
else:
    try:
        run_aqueduct(snakemake.params.aqueduct_executable, snakemake.input.toml)  # noqa: F821
    except subprocess.CalledProcessError as e:
        if not is_oom_error(e):
            raise
        mark_tile_oom(oom_dir, tile_id, reason="OutOfMemoryError in component_indices (core.jl) - tile too large for Aqueduct")
        log_skipped_tile(skipped_dir, tile_id, scenario_name, reason="OutOfMemoryError - tile too large for Aqueduct")
        save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821

# Only a per-tile milestone (not printed for every scenario job): once this
# tile's own output count reaches n_scenarios_per_tile, do the one O(n_tiles)
# scan across the whole grid and report the running totals.
n_scenarios_per_tile = snakemake.params.n_scenarios_per_tile  # noqa: F821
if tile_output_complete(model_outputs, tile_id, n_scenarios_per_tile):
    tile_ids = load_tile_grid(snakemake.params.tile_grid_path)["tile_id"].astype(int).tolist()  # noqa: F821
    print_simulation_progress(model_outputs, oom_dir, skipped_dir, tile_ids, n_scenarios_per_tile)
