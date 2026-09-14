"""Calibration run of obstacle-coupling across a representative pool of
wave-0 tiles (2026-08 - scaled up from a 7-tile pilot to 40 tiles; see
`C:\\Users\\Schlu005\\.claude\\plans\\smooth-wandering-map.md`). Tile IDs
come from `test_sweep_budget_calibration.py`'s `wet_tiles_selected.txt`
(that script's sweep=1 result already confirmed each of these tiles has
real flooding for this scenario - no separate dry-check needed here).

Per outer iteration, the inner solve now runs up to INNER_MAX_ROUNDS full
rounds (4 sweeps each, Julia's real Gray-code order), stopping early once
its own round-level max_change drops to/below epsilon - this is the
standard solve_eikonal_dense(max_rounds=...) semantics, NOT a fixed
single-round call.

Per-cell change diagnostics for the inner solve's own stopping point,
computed from the exact elementwise |after - before| diff of the LAST
round it performed:
  - pct_cells_still_changing: % of all cells in the tile where that diff
    exceeds epsilon (relative measure of how "unsettled" the inner solve
    still is when it stops).
  - mean/min/max_cell_change: mean/min/max of that diff, over all cells.

An explicit n_outer=0 row is written first for each tile: the plain,
un-blocked single solve (no static pre-filter, no obstacle_coupling at
all) - the "no blocking whatsoever" baseline the outer-loop trace is
compared against. From n_outer=1 onward, both configs still use:
  - the static dem > max_waterlevel pre-filter.
  - the tolerant outer-loop stopping criterion: stop once pct_newly_blocked
    (relative to total tile cells) drops below OUTER_CONVERGENCE_PCT.

Per outer round (round 0/baseline is the implicit reference for round 1's
diff, then round N vs round N-1 for N>=2 - matches pct_newly_blocked's own
existing convention): pct_blocked_cumulative (running total, "basis"),
pct_newly_blocked (round-over-round delta, "change per round"),
max_depth_change_abs (max absolute per-cell depth diff vs the previous
round, whole array), and n_newly_flooded/n_no_longer_flooded (unlike the
sweep-level report, BOTH directions matter here - re-blocking a cell can
genuinely un-flood it, that's the entire point of obstacle_coupling).
Baseline (n_outer=0) has no "previous round" to diff against, so these get
0 placeholders, same convention as pct_newly_blocked already used.

Each tile's CSV ends with one extra summary row (status="summary") giving
pct_removed_by_static_filter_alone (comparing n_outer=0 -> n_outer=1) and
pct_additional_removed_by_full_outer_loop (n_outer=1 -> the final,
converged row) - the direct answer to "is the free static pre-filter
already most of the benefit, or does the iterative re-blocking loop matter
much more."

Output: ONE CSV per tile (not a single combined file), written
incrementally, in CSV_DIR, plus a shared progress log on stdout.

Usage:
    python test_obstacle_coupling_calibration.py <output_dir> [tile_id ...]
"""
import csv
import gc
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
OCEAN_CODE = 1
BLOCK_FRICTION = 100.0
OUTER_CONVERGENCE_PCT = 0.01
MAX_OUTER = 5  # 400-tile study (was 3 for the 40-tile study, 20 in the 7-tile pilot) - matches production's obstacle_coupling.max_outer_iterations default
INNER_MAX_ROUNDS = 20  # 400-tile study (was 6 for the 40-tile study, 25 in the 7-tile pilot)

# tile_generation.river_code in config.yml - hardcoded here rather than
# reading config.yml, matching this script's existing ocean_code convention.
RIVER_CODE = 3

# Fallback for a quick standalone smoke test if no tile_ids are given on
# the command line - the real 40-tile study always passes an explicit list
# read from test_sweep_budget_calibration.py's wet_tiles_selected.txt (see
# module docstring).
TILES = [1826, 711, 1722, 548, 1424, 1463, 497]


def load_tile(tile_id: int):
    tile_dir = MODEL_OUTPUTS / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
    resolution = toml_cfg["flooding"]["resolution"]
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
    # (rasters.encode_waterlevel_cm, 2026-08) - decode before use. This
    # script's tile list is always drawn from test_sweep_budget_
    # calibration.py's wet_tiles_selected.txt, which already excludes any
    # tile with zero COAST-RP stations (that script's own DRY handling) -
    # so this should never actually be empty in practice. Fail loudly and
    # clearly here rather than let it crash inside _idw_seed_values'
    # BallTree construction (zero samples) or station_values.max() below
    # with a confusing scikit-learn traceback, in case that invariant is
    # ever violated (e.g. a hand-picked tile_id passed directly on the CLI).
    station_values = decode_waterlevel_cm(boundaries[variable].to_numpy())
    if len(station_values) == 0:
        raise ValueError(
            f"tile {tile_id} has zero boundary stations - should have been excluded "
            "from wet_tiles_selected.txt by test_sweep_budget_calibration.py"
        )

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)), mask, OCEAN_CODE,
    )
    epsilon = float(friction.min()) / (resolution * 10.0)
    max_waterlevel = float(station_values.max())
    return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial, epsilon, max_waterlevel


def solve_inner(friction, seed_rows, seed_cols, seed_values, dtype, epsilon, max_rounds):
    """Up to max_rounds full rounds (4 sweeps each), epsilon early-exit -
    same semantics as solve_eikonal_dense(max_rounds=...). Also returns
    exact per-cell change diagnostics from the LAST round performed.
    """
    m, n = friction.shape
    # t defaults to +99 (waterlevel=-t=-99m), not 0 - matches src/eikonal.py's
    # solve_eikonal_dense (2026-08 fix): a cell no seed's influence ever
    # reaches should read as "never flooded", not "flooded at exactly sea
    # level". This script reimplements the sweep loop directly (bypassing
    # solve_eikonal_dense) for per-round instrumentation, so needs the same
    # fix applied here explicitly to stay consistent with production.
    t = np.full((m + 1, n + 1), 99.0, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)
    round_max_change = 0.0
    n_rounds_used = 0
    prev_domain = t[1:, 1:].copy()
    for r in range(max_rounds):
        prev_domain = t[1:, 1:].copy()
        round_max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            round_max_change = max(round_max_change, _dense_sweep(t, friction, orthant, neg_two, eight, four))
        n_rounds_used = r + 1
        if round_max_change <= epsilon:
            break

    # In-place, native-dtype diff - avoids materializing several extra
    # float64-upcast full-tile temporaries (was ~4x2.67GB on the largest
    # tile, enough to OOM even though the sweeps themselves fit in memory).
    diff = t[1:, 1:] - prev_domain
    np.abs(diff, out=diff)
    n_cells = diff.size
    change_stats = {
        "n_rounds_used": n_rounds_used,
        "pct_cells_still_changing": round(100.0 * float(np.count_nonzero(diff > epsilon)) / n_cells, 6),
        "mean_cell_change": round(float(diff.mean(dtype=np.float64)), 8),
        "min_cell_change": round(float(diff.min()), 8),
        "max_cell_change": round(float(diff.max()), 8),
    }
    return t, round_max_change, change_stats


def _full_depth_array(waterlevel, dem, mask, coastline) -> np.ndarray:
    """Identical math to flood_depth_dense's own non-coupling path, but
    returns the FULL tile-shaped depth array (zeros where dry) rather than
    just flooded-cell aggregate stats - needed to diff consecutive rounds
    cell-by-cell, and computed once here (not duplicated inside
    _snapshot_from_depth) to avoid a second prune_to_coast_connected pass.
    Takes the already-computed `waterlevel` array (not `t_b`) so callers can
    release `t_b` (the dominant allocation) before calling this.
    """
    flood = (waterlevel > dem) & (mask != OCEAN_CODE)
    flood = prune_to_coast_connected(flood, coastline)
    depth = np.zeros_like(dem)
    depth[flood] = waterlevel[flood] - dem[flood]
    return depth


def _snapshot_from_depth(depth, n_outer, stopped_early, last_max_change, epsilon,
                          cum_elapsed, iter_elapsed, change_stats) -> dict:
    flood = depth > 0
    n_inundated = int(flood.sum())
    row = {
        "n_outer": n_outer,
        "cum_time_s": round(cum_elapsed, 3),
        "iter_time_s": round(iter_elapsed, 3),
        "outer_stopped_early": stopped_early,
        "last_max_change": round(float(last_max_change), 8),
        "epsilon": epsilon,
        "inner_converged": bool(last_max_change <= epsilon),
        "n_inundated": n_inundated,
    }
    row.update(change_stats)
    if n_inundated:
        depth_vals = depth[flood]
        row["depth_mean"] = round(float(depth_vals.mean()), 4)
        row["depth_min"] = round(float(depth_vals.min()), 4)
        row["depth_max"] = round(float(depth_vals.max()), 4)
        row["depth_q20"] = round(float(np.percentile(depth_vals, 20)), 4)
        row["depth_q80"] = round(float(np.percentile(depth_vals, 80)), 4)
    else:
        row["depth_mean"] = row["depth_min"] = row["depth_max"] = row["depth_q20"] = row["depth_q80"] = 0.0
    return row


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray) -> dict:
    """Per-round change metrics vs. the previous round - both directions
    matter here (unlike the sweep-level report): re-blocking a cell can
    genuinely un-flood it.
    """
    abs_diff = np.abs(depth_curr - depth_prev)
    n_newly_flooded = int(((depth_prev == 0) & (depth_curr > 0)).sum())
    n_no_longer_flooded = int(((depth_prev > 0) & (depth_curr == 0)).sum())
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "n_newly_flooded": n_newly_flooded,
        "n_no_longer_flooded": n_no_longer_flooded,
    }


def run_tile_trace(tile_id: int, max_outer: int) -> list[dict]:
    dem, mask, friction, coastline, seed_rows, seed_cols, initial, epsilon, max_waterlevel = load_tile(tile_id)
    seed_values = -initial
    dtype = friction.dtype
    n_cells = dem.size

    t0 = time.perf_counter()

    # n_outer=0 baseline: the plain, entirely unblocked solve (no static
    # pre-filter, no obstacle_coupling at all) - the "no blocking
    # whatsoever" reference point the rest of this trace is measured
    # against (see module docstring's summary-row explanation).
    iter_t0 = time.perf_counter()
    t0_b, max_change0, change_stats0 = solve_inner(
        friction, seed_rows, seed_cols, seed_values, dtype, epsilon, INNER_MAX_ROUNDS,
    )
    wl0 = -t0_b[1:, 1:]
    # t0_b (the dominant allocation, e.g. ~830MB on the largest tile at
    # float32) is never needed again once wl0 is extracted - without this,
    # it stays bound to a local variable (and therefore alive) for the
    # ENTIRE rest of the tile's outer-loop processing, on top of every
    # outer iteration's own t_b (found 2026-08 investigating OOM failures).
    del t0_b
    gc.collect()
    prev_depth = _full_depth_array(wl0, dem, mask, coastline)
    del wl0
    baseline_snap = _snapshot_from_depth(prev_depth, 0, False, max_change0, epsilon,
                                          time.perf_counter() - t0, time.perf_counter() - iter_t0, change_stats0)
    baseline_snap["tile"] = tile_id
    baseline_snap["n_cells"] = n_cells
    baseline_snap["pct_newly_blocked"] = 0.0
    baseline_snap["pct_blocked_cumulative"] = 0.0
    # No "previous round" to diff baseline against - 0 placeholders, same
    # convention as pct_newly_blocked above.
    baseline_snap["max_depth_change_abs"] = 0.0
    baseline_snap["n_newly_flooded"] = 0
    baseline_snap["n_no_longer_flooded"] = 0
    baseline_snap["status"] = "ok"
    rows = [baseline_snap]

    static_blocked = dem > max_waterlevel
    friction_b = np.where(static_blocked, dtype.type(BLOCK_FRICTION), friction)
    prev_blocked = None

    for outer in range(max_outer):
        n_outer = outer + 1
        iter_t0 = time.perf_counter()
        t_b, max_change, change_stats = solve_inner(
            friction_b, seed_rows, seed_cols, seed_values, dtype, epsilon, INNER_MAX_ROUNDS,
        )
        iter_elapsed = time.perf_counter() - iter_t0
        wl_b = -t_b[1:, 1:]
        del t_b  # release the dominant allocation now - see baseline's own comment above for why
        gc.collect()

        blocked = (wl_b <= dem) | static_blocked
        blocked[seed_rows, seed_cols] = False
        depth = _full_depth_array(wl_b, dem, mask, coastline)
        del wl_b

        n_newly = int(blocked.sum()) if prev_blocked is None else int((blocked & ~prev_blocked).sum())
        pct_newly = 100.0 * n_newly / n_cells
        stopped_early = prev_blocked is not None and pct_newly < OUTER_CONVERGENCE_PCT

        change = _depth_change_metrics(prev_depth, depth)
        snap = _snapshot_from_depth(depth, n_outer, stopped_early, max_change, epsilon,
                                     time.perf_counter() - t0, iter_elapsed, change_stats)
        snap.update(change)
        snap["tile"] = tile_id
        snap["n_cells"] = n_cells
        snap["pct_newly_blocked"] = round(pct_newly, 6)
        snap["pct_blocked_cumulative"] = round(100.0 * int(blocked.sum()) / n_cells, 6)
        snap["status"] = "ok"
        rows.append(snap)

        prev_depth = depth
        if stopped_early:
            break
        friction_b = np.where(blocked, dtype.type(BLOCK_FRICTION), friction)
        prev_blocked = blocked

    return rows


FIELDNAMES = [
    "tile", "n_outer", "n_cells", "cum_time_s", "iter_time_s", "n_rounds_used",
    "outer_stopped_early", "pct_blocked_cumulative", "pct_newly_blocked",
    "last_max_change", "epsilon", "inner_converged", "pct_cells_still_changing",
    "mean_cell_change", "min_cell_change", "max_cell_change", "max_depth_change_abs",
    "n_newly_flooded", "n_no_longer_flooded", "n_inundated", "depth_mean",
    "depth_min", "depth_max", "depth_q20", "depth_q80", "status",
    "pct_removed_by_static_filter_alone", "pct_additional_removed_by_full_outer_loop",
    "n_outer_iterations_to_convergence",
]


def build_summary_row(rows: list[dict], tile_id: int) -> dict:
    """One extra CSV row (status='summary') answering: is the free static
    dem > max_waterlevel pre-filter (n_outer=0 -> n_outer=1) already most of
    the benefit, or does the iterative re-blocking outer loop (n_outer=1 ->
    final/converged) remove meaningfully more flooding beyond that?
    """
    baseline = rows[0]  # n_outer=0: no blocking at all
    static_only = rows[1]  # n_outer=1: static pre-filter + one inner solve, zero re-blocking
    final = rows[-1]  # converged (or MAX_OUTER-capped) row

    n0 = baseline["n_inundated"]
    row = {k: "" for k in FIELDNAMES}
    row["tile"] = tile_id
    row["status"] = "summary"
    if n0 > 0:
        row["pct_removed_by_static_filter_alone"] = round(100.0 * (n0 - static_only["n_inundated"]) / n0, 4)
        row["pct_additional_removed_by_full_outer_loop"] = round(
            100.0 * (static_only["n_inundated"] - final["n_inundated"]) / n0, 4,
        )
    else:
        row["pct_removed_by_static_filter_alone"] = 0.0
        row["pct_additional_removed_by_full_outer_loop"] = 0.0
    row["n_outer_iterations_to_convergence"] = final["n_outer"]
    return row


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = [int(t) for t in sys.argv[2:]] if len(sys.argv) > 2 else TILES

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows = run_tile_trace(tile_id, MAX_OUTER)
                for row in tile_rows:
                    label = "baseline (no blocking)" if row["n_outer"] == 0 else f"outer {row['n_outer']}"
                    print(f"  {label}: cum_time={row['cum_time_s']}s  "
                          f"iter_time={row['iter_time_s']}s  n_rounds_used={row['n_rounds_used']}"
                          f"{'  (stopped early)' if row['outer_stopped_early'] else ''}  "
                          f"pct_blocked_cumulative={row['pct_blocked_cumulative']}%  "
                          f"pct_newly_blocked={row['pct_newly_blocked']}%  "
                          f"last_max_change={row['last_max_change']:.8f}  "
                          f"inner_converged={row['inner_converged']}", flush=True)
                    print(f"    pct_cells_still_changing={row['pct_cells_still_changing']}%  "
                          f"mean_change={row['mean_cell_change']}  min_change={row['min_cell_change']}  "
                          f"max_change={row['max_cell_change']}", flush=True)
                    print(f"    n_inundated={row['n_inundated']:,}  "
                          f"depth mean={row['depth_mean']}  min={row['depth_min']}  max={row['depth_max']}  "
                          f"q20={row['depth_q20']}  q80={row['depth_q80']}", flush=True)
                    print(f"    max_depth_change_abs={row['max_depth_change_abs']}  "
                          f"n_newly_flooded={row['n_newly_flooded']:,}  "
                          f"n_no_longer_flooded={row['n_no_longer_flooded']:,}", flush=True)
                    writer.writerow(row)
                    f.flush()

                summary = build_summary_row(tile_rows, tile_id)
                print(f"  SUMMARY: static filter alone removed "
                      f"{summary['pct_removed_by_static_filter_alone']}% of baseline flooding; "
                      f"full outer loop removed a further "
                      f"{summary['pct_additional_removed_by_full_outer_loop']}% "
                      f"({summary['n_outer_iterations_to_convergence']} outer iteration(s) to convergence)",
                      flush=True)
                writer.writerow(summary)
                f.flush()
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
                err_row = {k: "" for k in FIELDNAMES}
                err_row["tile"] = tile_id
                err_row["status"] = f"error: {exc}"
                writer.writerow(err_row)

    print(f"\nDone. Per-tile CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
