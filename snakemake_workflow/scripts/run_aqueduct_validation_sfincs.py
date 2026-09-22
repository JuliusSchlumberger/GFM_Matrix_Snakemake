"""Run the production eikonal flood solver for ONE tile, reading its inputs
from validation_sfincs/{tile_id}/inputs/ instead of the normal
model_outputs/{tile_id}/inputs/ - a standalone CLI purpose-built for the
258-tile SFINCS validation batch, NOT a replacement for run_aqueduct_cli.py.

Why this exists rather than reusing run_aqueduct_cli.py directly: that
script is tightly coupled to the production model_outputs/ + tile_grid.path
(hop_distance lookup) conventions. Every tile in this validation batch was
deliberately selected as hop_distance=0 (own real ocean boundary - see
sfincs_tiles' own plan doc), so there is no neighbour-seeding path to
support here, and reading from validation_sfincs/ instead of model_outputs/
would mean monkeypatching that script's own path construction throughout.
This script is the same underlying call (aqueduct_runner.run_aqueduct_python)
with none of that unneeded complexity.

Deliberately uses the ALREADY-ARCHIVED (pre-MDT-fix) boundaries_RP100_SLR_0.gpkg
sitting in validation_sfincs/{tile_id}/inputs/ - NOT freshly regenerated
ones - so this run isolates the effect of the OTHER fix
(simulation.flooding.friction_scale_factor, now 30.0) from the MDT sign
fix, since SFINCS's own validation run this session used those exact same
(old, stale) boundaries as its own forcing baseline - an apples-to-apples
comparison against the existing SFINCS results depends on that.

Usage:
    python run_aqueduct_validation_sfincs.py --config <resolved_config.yml> --tile-id 37
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import rasterio
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aqueduct_runner import run_aqueduct_python  # noqa: E402
from config_utils import path_ready, retry_transient_io  # noqa: E402

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"


def _output_already_done(output_path: str) -> bool:
    """Same real-readability check as run_aqueduct_cli.py's own - a job
    killed mid-write (SLURM time limit, node failure) can leave a truncated
    file behind that a bare exists() check would wrongly treat as done."""
    if not path_ready(output_path):
        return False
    try:
        with rasterio.open(output_path) as src:
            src.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a fully-resolved config.yml (e.g. resolved_config.yml)")
    parser.add_argument("--tile-id", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root = config["paths"]["root"]
    flooding_config = config["simulation"]["flooding"]
    ocean_code = config["tile_generation"]["ocean_code"]
    river_code = config["tile_generation"]["river_code"]

    tile_id = args.tile_id
    tile_dir = os.path.join(root, "validation_sfincs", tile_id, "inputs")
    out_dir = os.path.join(root, "validation_sfincs", tile_id, "outputs")
    dem_path = os.path.join(tile_dir, "dem.tif")
    mask_path = os.path.join(tile_dir, "mask.tif")
    friction_path = os.path.join(tile_dir, "friction.tif")
    boundaries_path = os.path.join(tile_dir, f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    output_path = os.path.join(out_dir, f"eikonal_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif")

    if _output_already_done(output_path):
        print(f"tile {tile_id}: already done, skipping")
        return

    if not path_ready(boundaries_path):
        print(f"tile {tile_id}: SKIP - no {boundaries_path}")
        return
    boundaries = retry_transient_io(gpd.read_file, boundaries_path)
    if boundaries.empty:
        print(f"tile {tile_id}: SKIP - boundaries file is empty (no station for this tile)")
        return

    retry_transient_io(Path(out_dir).mkdir, parents=True, exist_ok=True)

    diagnostics = run_aqueduct_python(
        dem_path, mask_path, friction_path, output_path,
        resolution=flooding_config["resolution"], k=flooding_config["knn"],
        variable=WATERLEVEL_NAME, boundaries_path=boundaries_path,
        ocean_code=ocean_code, river_code=river_code,
        obstacle_coupling=flooding_config.get("obstacle_coupling", {}).get("enabled", False),
        max_outer_iterations=flooding_config.get("obstacle_coupling", {}).get("max_outer_iterations", 5),
        max_rounds=flooding_config["max_rounds"],
        outer_convergence_pct=flooding_config.get("obstacle_coupling", {}).get("outer_convergence_pct", 0.01),
        waterlevel_epsilon_m=flooding_config["waterlevel_epsilon_m"],
        friction_scale_factor=flooding_config.get("friction_scale_factor", 1.0),
    )
    print(f"tile {tile_id}: done - {diagnostics}")


if __name__ == "__main__":
    main()
