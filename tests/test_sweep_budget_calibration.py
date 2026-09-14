"""Calibration run of the production, non-obstacle-coupling sweep count
(`flood_model.flood_depth_dense`'s `sweep_budget`/round-based `max_rounds`),
across a representative pool of wave-0 tiles (2026-08 - see
`C:\\Users\\Schlu005\\.claude\\plans\\smooth-wandering-map.md` for the
scale-up-to-40-tiles methodology; tile IDs come from
`select_calibration_tiles.py`'s `candidate_tiles.txt`).

For each tile, seeds exactly like production (`coastline_mask` +
`_idw_seed_values`), then runs ONE continuous sequence of individual
`_dense_sweep` calls in `_ORTHANT_ORDER[i % 4]` order - exactly
`solve_eikonal_dense`'s own `sweep_budget` semantics, so results are
directly comparable to the real production code path rather than a
reimplementation that could drift - reporting EVERY individual sweep from 1
to MAX_SWEEPS (6 rounds x 4 sweeps - tight enough now that exhaustive
per-sweep reporting is tractable, unlike the wider pilot run's sparse
checkpoint list). Doing this as a single continuous pass (not N independent
from-scratch solves) avoids O(N^2) redundant work on the largest tiles.

Dry-tile handling: if sweep 1 already shows zero flooding, the tile is
logged DRY and the remaining sweeps are skipped for it (no point spending
23 more sweeps on a tile already known uninformative) - this is how
"replace dry tiles" is implemented for the 40-tile study: the candidate
pool is deliberately over-provisioned (~55 tiles for 40 needed), and this
script writes `wet_tiles_selected.txt` (the first N_TILES_WANTED
confirmed-wet tile_ids, in the order given) for
`test_obstacle_coupling_calibration.py` to consume directly.

At each sweep, compares the full tile-shaped depth array (zeros where dry)
to the previous sweep's (sweep 1 compared against an implicit all-dry
"sweep 0"): max absolute depth change (whole array - well-defined
regardless), median/mean absolute depth change (restricted to cells that
actually changed - a whole-array median/mean would trivially read ~0 since
flooding covers a small minority of any tile), count of cells newly
flooded (depth 0 -> >0 - no "un-flooded" count needed: within one
un-blocked sweep sequence, Fast Sweeping's relaxation only ever lowers each
cell's eikonal travel-time cost, so flooding can only grow, never shrink),
and count/percent of all cells whose depth changed at all (broader than
"newly flooded" - includes already-flooded cells still settling deeper/
shallower).

Output: ONE CSV per tile, written incrementally, in CSV_DIR, plus a shared
progress log on stdout.

Usage:
    python test_sweep_budget_calibration.py <output_dir> [tile_id ...]
"""
import csv
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
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402
from rasters import decode_dem_cm, decode_friction_int16, decode_waterlevel_cm  # noqa: E402

MODEL_OUTPUTS = Path("D:/GFM/model_outputs")
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# tile_generation.river_code / ocean_code in config.yml - hardcoded here,
# matching test_obstacle_coupling_calibration.py's own convention.
OCEAN_CODE = 1
RIVER_CODE = 3

# Fallback for a quick standalone smoke test if no tile_ids are given on
# the command line - the real 40-tile study always passes an explicit list
# (see module docstring).
TILES = [1826, 711, 1722, 548, 1424, 1463, 497]

MAX_ROUNDS = 20  # 100-tile study (was 6 for the 40-tile study, 25 in the 7-tile pilot)
MAX_SWEEPS = MAX_ROUNDS * 4  # every individual sweep 1..80 is reported (see module docstring)
N_TILES_WANTED = 100  # size of the final wet_tiles_selected.txt list


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
        # int8, not int64: mask only ever holds a handful of small codes,
        # every downstream use is a plain equality/inequality comparison -
        # int64 was 8x more memory than this array ever needed (found
        # 2026-08 investigating OOM failures on large real tiles).
        mask = src.read(1).astype(np.int8)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1))
    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")

    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    coastline = coastline_mask(mask, ocean_code=OCEAN_CODE, river_code=RIVER_CODE)
    coastline_rows, coastline_cols = np.nonzero(coastline)

    # boundaries.gpkg stores int16-centimetre-encoded water levels
    # (rasters.encode_waterlevel_cm, 2026-08) - decode before use. An empty
    # boundaries file (no COAST-RP station found for this tile/scenario -
    # see extract_boundaries.py) is a real, expected outcome for small/
    # isolated tiles, matching production's own NO_STATIONS_REASON skip -
    # _idw_seed_values can't build a BallTree from zero stations, so skip
    # it entirely and seed nothing, same as a genuinely dry tile (empty
    # seed arrays -> t stays all-zero -> n_inundated correctly comes out 0
    # at sweep 1, triggering the same DRY handling as zero-flooding tiles).
    station_values = decode_waterlevel_cm(boundaries[variable].to_numpy())
    if len(station_values) == 0:
        coastline_rows = np.array([], dtype=coastline_rows.dtype)
        coastline_cols = np.array([], dtype=coastline_cols.dtype)
        initial = np.array([], dtype=np.float64)
        return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)), mask, OCEAN_CODE,
    )
    return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial


def _full_depth_array(t, dem, mask, coastline) -> np.ndarray:
    """Identical math to flood_depth_dense's own non-coupling path, but
    returns the FULL tile-shaped depth array (zeros where dry) rather than
    just flooded-cell aggregate stats - needed to diff consecutive sweeps
    cell-by-cell.
    """
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != OCEAN_CODE)
    flood = prune_to_coast_connected(flood, coastline)
    depth = np.zeros_like(dem)
    depth[flood] = waterlevel[flood] - dem[flood]
    return depth


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray, n_cells: int) -> dict:
    """Per-sweep change metrics comparing depth_curr to depth_prev - see
    module docstring for why median/mean are restricted to changed cells
    while max and the newly-flooded/changed counts use the whole array.
    """
    abs_diff = np.abs(depth_curr - depth_prev)
    changed = abs_diff > 0
    n_changed = int(changed.sum())
    changed_vals = abs_diff[changed]
    n_newly_flooded = int(((depth_prev == 0) & (depth_curr > 0)).sum())
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "median_depth_change_abs": round(float(np.median(changed_vals)), 4) if changed_vals.size else 0.0,
        "mean_depth_change_abs": round(float(changed_vals.mean()), 4) if changed_vals.size else 0.0,
        "n_newly_flooded": n_newly_flooded,
        "n_cells_changed": n_changed,
        "pct_cells_changed": round(100.0 * n_changed / n_cells, 6),
    }


def run_tile_sweep_trace(tile_id: int, max_sweeps: int) -> tuple[list[dict], bool]:
    """Returns (rows, is_wet). is_wet is False iff sweep 1 already shows
    zero flooding - callers should skip such a tile for wet_tiles_selected.txt.
    """
    dem, mask, friction, coastline, seed_rows, seed_cols, initial = load_tile(tile_id)
    seed_values = -initial
    dtype = friction.dtype
    n_cells = dem.size

    m, n = friction.shape
    # t defaults to +99 (waterlevel=-t=-99m), not 0 - matches src/eikonal.py's
    # solve_eikonal_dense (2026-08 fix): a cell no seed's influence ever
    # reaches should read as "never flooded", not "flooded at exactly sea
    # level". This script reimplements the sweep loop directly (bypassing
    # solve_eikonal_dense) for per-sweep instrumentation, so needs the same
    # fix applied here explicitly to stay consistent with production.
    t = np.full((m + 1, n + 1), 99.0, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    rows = []
    prev_depth = np.zeros_like(dem)  # implicit all-dry "sweep 0" reference for sweep 1's diff
    t0 = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        _dense_sweep(t, friction, _ORTHANT_ORDER[(sweep - 1) % 4], neg_two, eight, four)
        depth = _full_depth_array(t, dem, mask, coastline)
        n_inundated = int((depth > 0).sum())
        change = _depth_change_metrics(prev_depth, depth, n_cells)

        row = {
            "tile": tile_id,
            "n_cells": n_cells,
            "sweep_count": sweep,
            "cum_time_s": round(time.perf_counter() - t0, 3),
            "n_inundated": n_inundated,
            "depth_mean": round(float(depth[depth > 0].mean()), 4) if n_inundated else 0.0,
            "depth_sum": round(float(depth.sum()), 2),
            "depth_max": round(float(depth.max()), 4),
            "status": "ok",
        }
        row.update(change)
        rows.append(row)

        if sweep == 1 and n_inundated == 0:
            return rows, False  # dry - caller stops here, doesn't burn the remaining sweeps

        prev_depth = depth

    return rows, True


FIELDNAMES = [
    "tile", "n_cells", "sweep_count", "cum_time_s", "n_inundated", "depth_mean",
    "depth_sum", "depth_max", "max_depth_change_abs", "median_depth_change_abs",
    "mean_depth_change_abs", "n_newly_flooded", "n_cells_changed", "pct_cells_changed",
    "status",
]


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = [int(t) for t in sys.argv[2:]] if len(sys.argv) > 2 else TILES

    wet_tiles: list[int] = []

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows, is_wet = run_tile_sweep_trace(tile_id, MAX_SWEEPS)
                for row in tile_rows:
                    print(f"  sweeps={row['sweep_count']:3d}  cum_time={row['cum_time_s']}s  "
                          f"n_inundated={row['n_inundated']:,}  depth_sum={row['depth_sum']}  "
                          f"max_change={row['max_depth_change_abs']}  "
                          f"median_change={row['median_depth_change_abs']}  "
                          f"mean_change={row['mean_depth_change_abs']}  "
                          f"n_newly_flooded={row['n_newly_flooded']:,}  "
                          f"n_changed={row['n_cells_changed']:,} ({row['pct_cells_changed']}%)", flush=True)
                    writer.writerow(row)
                    f.flush()
                if is_wet:
                    wet_tiles.append(tile_id)
                    print(f"  SUMMARY: wet ({len(wet_tiles)}/{N_TILES_WANTED} selected so far)", flush=True)
                else:
                    print("  SUMMARY: DRY at sweep 1 - excluded", flush=True)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
                err_row = {k: "" for k in FIELDNAMES}
                err_row["tile"] = tile_id
                err_row["status"] = f"error: {exc}"
                writer.writerow(err_row)

        if len(wet_tiles) >= N_TILES_WANTED:
            print(f"\n{N_TILES_WANTED} wet tiles found - stopping early, "
                  f"{len(tiles) - tiles.index(tile_id) - 1} remaining candidate(s) not needed", flush=True)
            break

    selected = wet_tiles[:N_TILES_WANTED]
    wet_file = out_dir / "wet_tiles_selected.txt"
    with open(wet_file, "w") as f:
        for tile_id in selected:
            f.write(f"{tile_id}\n")
    print(f"\nDone. Per-tile CSVs written to {out_dir}")
    print(f"{len(selected)}/{N_TILES_WANTED} wet tiles selected, written to {wet_file}")
    if len(selected) < N_TILES_WANTED:
        print(f"WARNING: only found {len(selected)} wet tiles out of {len(tiles)} candidates - "
              f"need more candidates from select_calibration_tiles.py", flush=True)


if __name__ == "__main__":
    main()
