"""Batch accuracy report: run the (now bit-exact, per the orthant-order +
discriminant fixes in eikonal.py) dense+3-sweep Python port against real
Julia `aqueduct.exe` output, across a sample of tiles not used in the
original fix/validation set (1962, 1963, 1981, 1990, 2660, 26729).

For each tile, reports:
  - Jaccard (flood/no-flood extent agreement)
  - only_python / only_julia (unique flooded-cell counts, each side)
  - RMSE, ME (mean signed diff, python-julia), q90 (90th pct of |diff|),
    max|diff| of water depth - all computed over BOTH-FLOODED cells only
    (isolates depth accuracy from extent disagreement)

Writes one row per tile to a CSV, printing progress as it goes so a
partial run is still useful if interrupted.

Usage:
    python run_20tile_report.py <output_csv>
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

# 20 of the 38 tiles not used in the original discriminant/orthant-order fix
# validation set (1962, 1963, 1981, 1990, 2660, 26729), chosen via
# `random.Random(42).shuffle(...)` over the sorted list of untested tiles
# with confirmed RP100_SLR_0 inputs+results on disk.
TILES = [
    1992, 2330, 2331, 2341, 2342, 2343, 2353, 2640, 2643, 2650,
    2652, 2661, 2662, 2670, 2863, 2870, 2872, 2873, 2882, 3072,
]


def run_one(tile_id: int) -> dict:
    tile_dir = MODEL_OUTPUTS / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"
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

    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))

    coastline = coastline_mask(mask, ocean_code=1)
    coastline_rows, coastline_cols = np.nonzero(coastline)

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    station_values = boundaries[variable].to_numpy()
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)),
    )

    epsilon = float(friction.min()) / (resolution * 10.0)

    start = time.perf_counter()
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon, sweep_budget=3,
    )
    elapsed = time.perf_counter() - start

    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)

    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]

    julia_path = tile_dir / "results" / f"waterdepth_{scenario}.tif"
    with rasterio.open(julia_path) as src:
        julia_waterdepth = src.read(1).astype(np.float64)

    julia_valid = julia_waterdepth < np.finfo(np.float32).max
    py = waterdepth[julia_valid]
    jl = julia_waterdepth[julia_valid]
    py_flood = py > 0
    jl_flood = jl > 0
    both = py_flood & jl_flood
    either = py_flood | jl_flood
    only_py = py_flood & ~jl_flood
    only_jl = jl_flood & ~py_flood

    jaccard = 100 * both.sum() / either.sum() if either.any() else float("nan")

    diff_both = (py - jl)[both]
    if diff_both.size:
        rmse = float(np.sqrt((diff_both ** 2).mean()))
        me = float(diff_both.mean())
        q90 = float(np.percentile(np.abs(diff_both), 90))
        maxdiff = float(np.abs(diff_both).max())
    else:
        rmse = me = q90 = maxdiff = float("nan")

    return {
        "tile": tile_id,
        "scenario": scenario,
        "n_cells": int(julia_valid.sum()),
        "time_s": round(elapsed, 1),
        "jaccard_pct": round(jaccard, 4),
        "only_python": int(only_py.sum()),
        "only_julia": int(only_jl.sum()),
        "n_both_flooded": int(both.sum()),
        "rmse_m": round(rmse, 6),
        "me_m": round(me, 6),
        "q90_abs_diff_m": round(q90, 6),
        "max_abs_diff_m": round(maxdiff, 6),
        "status": "ok",
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    fieldnames = [
        "tile", "scenario", "n_cells", "time_s", "jaccard_pct",
        "only_python", "only_julia", "n_both_flooded",
        "rmse_m", "me_m", "q90_abs_diff_m", "max_abs_diff_m", "status",
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
                print(f"  jaccard={row['jaccard_pct']:.3f}%  "
                      f"only_py={row['only_python']:,}  only_jl={row['only_julia']:,}  "
                      f"RMSE={row['rmse_m']:.4f}m  ME={row['me_m']:+.4f}m  "
                      f"q90={row['q90_abs_diff_m']:.4f}m  max={row['max_abs_diff_m']:.4f}m  "
                      f"({row['time_s']}s)", flush=True)
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
