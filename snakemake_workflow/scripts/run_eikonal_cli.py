"""Run the flood model for one tile/return_period/waterlevel_name from a plain CLI call.

HPC entry point for the sbatch scripts from generate_eikonal_jobs.py; logs a
run-timing record for every outcome.

Usage:
    python run_eikonal_cli.py --config <resolved_config.yml>
        --tile-id <id> --return-period <RP..> --waterlevel-name <SLR_..>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eikonal_runner import log_run_timing, run_tile_scenario  # noqa: E402
from config_utils import path_ready  # noqa: E402


def _output_already_done(output_path: str) -> bool:
    # Readable-raster check, not existence: a job killed mid-write leaves a
    # truncated file; reading one pixel forces GDAL to validate structure.
    if not path_ready(output_path):
        return False
    try:
        with rasterio.open(output_path) as src:
            src.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def main() -> None:
    """Solve, skip or write a placeholder for one (tile, rp, slr) scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a fully-resolved config.yml (e.g. resolved_config.yml)")
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--return-period", required=True)
    parser.add_argument("--waterlevel-name", required=True)
    args = parser.parse_args()
    start = time.perf_counter()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_outputs = config["simulation"]["model_outputs"]
    # Group-shared preprocessing inputs for calibration sweeps; see preprocessing.smk.
    preprocessing_inputs_dir = config["simulation"].get("preprocessing_inputs_dir") or model_outputs
    raster_config = config["raster_format"]
    flooding_config = config["simulation"]["flooding"]
    ocean_code = config["tile_generation"]["ocean_code"]
    river_code = config["tile_generation"]["river_code"]
    tile_grid_path = config["tile_grid"]["path"]

    tile_id = args.tile_id
    scenario_name = f"{args.return_period}_{args.waterlevel_name}"
    tile_dir = os.path.join(model_outputs, tile_id)
    inputs_dir = os.path.join(preprocessing_inputs_dir, tile_id, "inputs")
    dem_path = os.path.join(inputs_dir, "dem.tif")
    mask_path = os.path.join(inputs_dir, "mask.tif")
    friction_path = os.path.join(inputs_dir, "friction.tif")
    boundaries_path = os.path.join(inputs_dir, f"boundaries_{scenario_name}.gpkg")
    output_path = os.path.join(tile_dir, "results", f"waterdepth_{scenario_name}.tif")

    # Idempotency for resumed/rebalanced dispatches.
    timings_dir = os.path.join(model_outputs, "run_timings")

    if _output_already_done(output_path):
        # No timing record: the earlier solve already wrote the real one.
        return

    result = run_tile_scenario(
        tile_id, args.return_period, args.waterlevel_name,
        dem_path=dem_path, mask_path=mask_path, friction_path=friction_path,
        boundaries_path=boundaries_path, output_path=output_path,
        model_outputs=model_outputs, preprocessing_inputs_dir=preprocessing_inputs_dir,
        tile_grid_path=tile_grid_path, raster_config=raster_config, flooding_config=flooding_config,
        ocean_code=ocean_code, river_code=river_code,
        # Staged to node-local scratch: GDAL/OGR's SQLite lock on a gpkg can
        # hang indefinitely on P:\ from a Hydrax compute node.
        stage_boundaries=True,
    )
    # elapsed = this whole invocation.
    if result["outcome"] == "solved":
        record = {"outcome": "solved", **result["diagnostics"]}
    else:
        record = {k: v for k, v in result.items() if k in ("outcome", "skip_reason")}
    log_run_timing(
        timings_dir, tile_id, args.return_period, args.waterlevel_name,
        elapsed_s=time.perf_counter() - start,
        obstacle_coupling_diagnostics=record,
    )


if __name__ == "__main__":
    main()
