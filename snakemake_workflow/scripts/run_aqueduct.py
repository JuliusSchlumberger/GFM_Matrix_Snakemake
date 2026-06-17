"""Run the Aqueduct flood model for a single tile and SLR scenario."""

import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct_runner import is_oom_error, log_skipped_tile, mark_tile_oom, run_aqueduct, tile_marked_oom  # noqa: E402
from rasters import save_nodata_raster  # noqa: E402

model_outputs = snakemake.config["simulation"]["model_outputs"]  # noqa: F821
skipped_dir = os.path.join(model_outputs, "skipped_tiles")
oom_dir = os.path.join(model_outputs, "oom_tiles")
tile_id = snakemake.wildcards.tile_id  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

boundaries = gpd.read_file(snakemake.input.boundaries)  # noqa: F821

if boundaries.empty:
    log_skipped_tile(skipped_dir, tile_id, waterlevel_name, reason="no water level boundary stations within tile")
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.config["simulation"]["input_raster"])  # noqa: F821
elif tile_marked_oom(oom_dir, tile_id):
    log_skipped_tile(
        skipped_dir,
        tile_id,
        waterlevel_name,
        reason="tile too large for Aqueduct (OutOfMemoryError on a previous SLR scenario for this tile)",
    )
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.config["simulation"]["input_raster"])  # noqa: F821
else:
    try:
        run_aqueduct(snakemake.config["simulation"]["aqueduct_executable"], snakemake.input.toml)  # noqa: F821
    except subprocess.CalledProcessError as e:
        if not is_oom_error(e):
            raise
        mark_tile_oom(oom_dir, tile_id, reason="OutOfMemoryError in component_indices (core.jl) - tile too large for Aqueduct")
        log_skipped_tile(skipped_dir, tile_id, waterlevel_name, reason="OutOfMemoryError - tile too large for Aqueduct")
        save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.config["simulation"]["input_raster"])  # noqa: F821
