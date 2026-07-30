"""Batch version of time_full_pipeline.py: full-pipeline, stage-by-stage
Python timing (read -> dem/mask prep -> coastline mask -> KNN+IDW -> sweep
(solve_eikonal_dense, sweep_budget=3) -> flood classification + connected-
component pruning) across the same 20 tiles used in run_20tile_report.py -
the true, fair counterpart to Julia's full-pipeline timing (Europe_West.log
aggregate + the second machine's tile-2660 stage breakdown), since every
earlier Python timing in this investigation measured only the sweep step.

Usage:
    python time_full_pipeline_batch.py <output_csv>
"""
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import solve_eikonal_dense  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402

MODEL_OUTPUTS = Path("D:/GFM/model_outputs")
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# same 20 tiles as run_20tile_report.py
TILES = [
    1992, 2330, 2331, 2341, 2342, 2343, 2353, 2640, 2643, 2650,
    2652, 2661, 2662, 2670, 2863, 2870, 2872, 2873, 2882, 3072,
]


def run_one(tile_id: int) -> dict:
    tile_dir = MODEL_OUTPUTS / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"

    t0 = time.perf_counter()

    t_read0 = time.perf_counter()
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
    resolution = toml_cfg["flooding"]["resolution"]
    knn = toml_cfg["waterlevels"]["knn"]
    variable = toml_cfg["waterlevels"]["name"]

    with rasterio.open(inputs / "dem.tif") as src:
        dem = src.read(1)
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        mask = src.read(1).astype(np.int64)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = src.read(1)
    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")
    t_read = time.perf_counter() - t_read0

    t_dem0 = time.perf_counter()
    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    t_dem = time.perf_counter() - t_dem0

    t_coast0 = time.perf_counter()
    coastline = coastline_mask(mask, ocean_code=1)
    coastline_rows, coastline_cols = np.nonzero(coastline)
    t_coast = time.perf_counter() - t_coast0

    t_idw0 = time.perf_counter()
    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    station_values = boundaries[variable].to_numpy()
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)),
    )
    t_idw = time.perf_counter() - t_idw0

    epsilon = float(friction.min()) / (resolution * 10.0)
    t_sweep0 = time.perf_counter()
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon, sweep_budget=3,
    )
    t_sweep = time.perf_counter() - t_sweep0

    t_prune0 = time.perf_counter()
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)
    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]
    t_prune = time.perf_counter() - t_prune0

    t_total = time.perf_counter() - t0

    return {
        "tile": tile_id,
        "scenario": scenario,
        "n_cells": int(dem.size),
        "t_read": round(t_read, 3),
        "t_dem": round(t_dem, 3),
        "t_coast": round(t_coast, 3),
        "t_idw": round(t_idw, 3),
        "t_sweep": round(t_sweep, 3),
        "t_prune": round(t_prune, 3),
        "t_total": round(t_total, 3),
        "status": "ok",
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    fieldnames = [
        "tile", "scenario", "n_cells", "t_read", "t_dem", "t_coast",
        "t_idw", "t_sweep", "t_prune", "t_total", "status",
    ]

    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for tile_id in TILES:
            print(f"=== tile {tile_id} ===", flush=True)
            try:
                row = run_one(tile_id)
                print(f"  total={row['t_total']:.2f}s  "
                      f"(read={row['t_read']:.2f} dem={row['t_dem']:.2f} coast={row['t_coast']:.2f} "
                      f"idw={row['t_idw']:.2f} sweep={row['t_sweep']:.2f} prune={row['t_prune']:.2f})",
                      flush=True)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
                row = {k: "" for k in fieldnames}
                row["tile"] = tile_id
                row["status"] = f"error: {exc}"
            writer.writerow(row)
            f.flush()

    print(f"\nDone. Report written to {out_path}")


if __name__ == "__main__":
    main()
