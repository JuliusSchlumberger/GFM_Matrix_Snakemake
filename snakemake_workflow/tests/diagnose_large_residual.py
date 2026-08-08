"""Diagnose the implausible 2-6m single-cell residuals seen in the overnight
calibration run (tiles 2330, 2340, 2341, 2642, 2871, 2873, 3053 all showed
last_max_change of 1-6m even after 100 sweeps / 25 rounds, with the SAME
few odd values - e.g. 2.20689845 and 2.21949720 - recurring across
unrelated tiles, which is far too coincidental to be "genuinely slow
geography" and smells like a specific numerical artifact instead).

For each target tile, replays the obstacle-coupling outer loop's static
pre-filter + outer-1 inner solve, but with PER-ROUND diagnostics: the
(row, col) of the single largest-magnitude cell update each round, its
before/after T value, and the friction/dem context there. If the same cell
keeps flipping between the same 2 (or few) values round after round, that's
a genuine oscillation/limit-cycle, not slow convergence.

Usage:
    python diagnose_large_residual.py <tile_id> [<tile_id> ...]
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask  # noqa: E402
from rasters import decode_dem_cm, decode_friction_int16  # noqa: E402

MODEL_OUTPUTS = Path("D:/GFM/model_outputs")
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
BLOCK_FRICTION = 100.0
INNER_MAX_ROUNDS = 25
EPSILON = 0.03  # matches the new production WATERLEVEL_EPSILON_M


def load_tile(tile_id: int):
    tile_dir = MODEL_OUTPUTS / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
    knn = toml_cfg["waterlevels"]["knn"]
    variable = toml_cfg["waterlevels"]["name"]

    with rasterio.open(inputs / "dem.tif") as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        mask = src.read(1).astype(np.int64)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1))
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
        min(knn, len(station_values)), mask, 1,
    )
    max_waterlevel = float(station_values.max())
    return dem, mask, friction, coastline_rows, coastline_cols, initial, max_waterlevel


def diagnose(tile_id: int, n_rounds=INNER_MAX_ROUNDS):
    print(f"\n=== tile {tile_id} ===", flush=True)
    dem, mask, friction, seed_rows, seed_cols, initial, max_waterlevel = load_tile(tile_id)
    seed_values = -initial
    dtype = friction.dtype

    static_blocked = dem > max_waterlevel
    friction_b = np.where(static_blocked, dtype.type(BLOCK_FRICTION), friction)

    m, n = friction_b.shape
    t = np.zeros((m + 1, n + 1), dtype=dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    history = []
    prev = t[1:, 1:].copy()
    for r in range(n_rounds):
        prev = t[1:, 1:].copy()
        round_max = 0.0
        for orthant in _ORTHANT_ORDER:
            round_max = max(round_max, _dense_sweep(t, friction_b, orthant, neg_two, eight, four))
        cur = t[1:, 1:]
        diff = np.abs(cur - prev)
        row, col = np.unravel_index(np.argmax(diff), diff.shape)
        row, col = int(row), int(col)
        entry = {
            "round": r + 1,
            "row": row, "col": col,
            "round_max": round_max,
            "diff_at_argmax": float(diff[row, col]),
            "t_before": float(prev[row, col]),
            "t_after": float(cur[row, col]),
            "friction_here": float(friction_b[row, col]),
            "dem_here": float(dem[row, col]),
            "static_blocked_here": bool(static_blocked[row, col]),
        }
        history.append(entry)
        print(f"  round {r+1:2d}: round_max={round_max:.6f}  argmax=({row},{col})  "
              f"friction={entry['friction_here']:.4f}  dem={entry['dem_here']:.4f}  "
              f"static_blocked={entry['static_blocked_here']}  "
              f"t: {entry['t_before']:.6f} -> {entry['t_after']:.6f}", flush=True)
        if round_max <= EPSILON:
            print(f"  converged at round {r+1} (round_max <= epsilon={EPSILON})", flush=True)
            break

    # Check whether the argmax location repeats (oscillation) vs wanders.
    locs = [(h["row"], h["col"]) for h in history]
    unique_locs = set(locs)
    print(f"  {len(history)} rounds run, {len(unique_locs)} unique argmax location(s): "
          f"{sorted(unique_locs)[:10]}{' ...' if len(unique_locs) > 10 else ''}", flush=True)
    if len(unique_locs) <= 3 and len(history) >= 4:
        # print the full t-value trajectory at the most common location
        from collections import Counter
        common_loc, count = Counter(locs).most_common(1)[0]
        vals = [h["t_after"] for h in history if (h["row"], h["col"]) == common_loc]
        print(f"  cell {common_loc} was the argmax {count}/{len(history)} times; "
              f"its t_after trajectory when so: {[round(v, 5) for v in vals]}", flush=True)
        r, c = common_loc
        print(f"  neighborhood of {common_loc}: dem={dem[r, c]:.4f}  "
              f"friction_b 3x3 block:\n{friction_b[max(r-1,0):r+2, max(c-1,0):c+2]}", flush=True)
        print(f"  dem 3x3 block:\n{dem[max(r-1,0):r+2, max(c-1,0):c+2]}", flush=True)
        print(f"  static_blocked 3x3 block:\n{static_blocked[max(r-1,0):r+2, max(c-1,0):c+2]}", flush=True)
    return history


def main():
    tile_ids = [int(a) for a in sys.argv[1:]]
    for tile_id in tile_ids:
        diagnose(tile_id)


if __name__ == "__main__":
    main()
