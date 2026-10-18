"""Trace obstacle-coupling outer iterations on the wet calibration tiles.

Tiles come from run_sweep_budget_study.py's wet_tiles_selected.txt. Writes one
CSV per tile: an n_outer=0 unblocked baseline, one row per outer iteration
(static dem > max_waterlevel pre-filter, stop when pct_newly_blocked <
--outer-convergence-pct), and a status="summary" row splitting flooding removed
by the static filter alone vs the full outer loop.

Each inner solve runs up to --max-rounds rounds (4 sweeps, Julia's
Gray-code order) with epsilon early exit; blocking uses
flood_model.OBSTACLE_BLOCK_FRICTION. Unlike production, dynamic blocking is
restricted to cells with t < 99 (reached): unreached cells read as
waterlevel -99 m and a block is never undone. pct_unreached is logged.

Usage:
    python run_obstacle_coupling_study.py --out-dir <output_dir> [--config <config.yml>]
        [--tile-ids id [id ...]] [--max-outer-iterations N] [--max-rounds N] [--epsilon M]

The flag names mirror the config keys (obstacle_coupling.max_outer_iterations,
flooding.max_rounds).
"""
import argparse
import csv
import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_model import OBSTACLE_BLOCK_FRICTION as BLOCK_FRICTION  # noqa: E402
from flood_model import prune_to_coast_connected  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep_budget_study import OCEAN_CODE, TILES, load_tile  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTER_EPSILON_M = 0.03  # = simulation.flooding.waterlevel_epsilon_m
DEFAULT_MAX_OUTER = 15  # beyond production's default, to find where tiles stop needing more
DEFAULT_INNER_MAX_ROUNDS = 40  # methods_03 §2: most tiles converge or freeze extent by round 40
# Not config-driven on purpose; pass --max-outer-iterations/--max-rounds to match config.yml.

# t only decreases from the unseeded 99.0 (eikonal.solve_eikonal_dense) once seeds reach a cell.
UNREACHED_SENTINEL = 99.0


def solve_inner(friction, seed_rows, seed_cols, seed_values, dtype, epsilon, max_rounds):
    """Run up to max_rounds 4-sweep rounds with epsilon early exit.

    Mirrors solve_eikonal_dense(max_rounds=...); also returns per-cell change
    stats of the last round.
    """
    m, n = friction.shape
    # Unseeded t=+99 (waterlevel -99 m) = never flooded, as in solve_eikonal_dense.
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

    # In-place native-dtype diff: float64 temporaries OOM on the largest tile.
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
    """Return tile-shaped depth (0 where dry), same math as flood_depth_dense without coupling."""
    flood = (waterlevel > dem) & (mask != OCEAN_CODE)
    flood = prune_to_coast_connected(flood, coastline)
    depth = np.zeros_like(dem)
    depth[flood] = waterlevel[flood] - dem[flood]
    return depth


def _snapshot_from_depth(depth, n_outer, stopped_early, last_max_change, epsilon,
                          cum_elapsed, iter_elapsed, change_stats, pct_unreached) -> dict:
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
        "pct_unreached": round(pct_unreached, 6),
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
    """Count both directions: re-blocking a cell can un-flood it."""
    abs_diff = np.abs(depth_curr - depth_prev)
    n_newly_flooded = int(((depth_prev == 0) & (depth_curr > 0)).sum())
    n_no_longer_flooded = int(((depth_prev > 0) & (depth_curr == 0)).sum())
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "n_newly_flooded": n_newly_flooded,
        "n_no_longer_flooded": n_no_longer_flooded,
    }


def run_tile_trace(
    tile_id: int, model_outputs: Path, knn: int, friction_scale_factor: float,
    max_outer: int, inner_max_rounds: int, epsilon: float, outer_convergence_pct: float,
) -> list[dict]:
    """Trace the unblocked baseline and each outer iteration; one row per solve."""
    dem, mask, friction, coastline, seed_rows, seed_cols, initial, max_waterlevel = load_tile(
        tile_id, model_outputs, knn, friction_scale_factor, on_no_stations="raise",
    )
    seed_values = -initial
    dtype = friction.dtype
    n_cells = dem.size

    t0 = time.perf_counter()

    # n_outer=0 baseline: unblocked solve, no static pre-filter.
    iter_t0 = time.perf_counter()
    t0_b, max_change0, change_stats0 = solve_inner(
        friction, seed_rows, seed_cols, seed_values, dtype, epsilon, inner_max_rounds,
    )
    wl0 = -t0_b[1:, 1:]
    pct_unreached0 = 100.0 * float(np.count_nonzero(t0_b[1:, 1:] >= UNREACHED_SENTINEL)) / n_cells
    # Free t0_b (~830MB float32 on the largest tile) before the outer loop (OOM otherwise).
    del t0_b
    gc.collect()
    prev_depth = _full_depth_array(wl0, dem, mask, coastline)
    del wl0
    baseline_snap = _snapshot_from_depth(prev_depth, 0, False, max_change0, epsilon,
                                          time.perf_counter() - t0, time.perf_counter() - iter_t0,
                                          change_stats0, pct_unreached0)
    baseline_snap["tile"] = tile_id
    baseline_snap["n_cells"] = n_cells
    baseline_snap["pct_newly_blocked"] = 0.0
    baseline_snap["pct_blocked_cumulative"] = 0.0
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
            friction_b, seed_rows, seed_cols, seed_values, dtype, epsilon, inner_max_rounds,
        )
        iter_elapsed = time.perf_counter() - iter_t0
        wl_b = -t_b[1:, 1:]
        # Unreached cells (t=99 sentinel, waterlevel -99 m) must not be blocked:
        # blocks persist through every later outer iteration.
        reached = t_b[1:, 1:] < UNREACHED_SENTINEL
        pct_unreached = 100.0 * float(np.count_nonzero(~reached)) / n_cells
        del t_b  # free the dominant allocation before the next solve (OOM otherwise)
        gc.collect()

        blocked = ((wl_b <= dem) & reached) | static_blocked
        if prev_blocked is not None:
            # Monotonic union (as in flood_model): otherwise blocked cells can reopen,
            # giving an undamped period-2 oscillation.
            blocked = blocked | prev_blocked
        blocked[seed_rows, seed_cols] = False
        depth = _full_depth_array(wl_b, dem, mask, coastline)
        del wl_b

        n_newly = int(blocked.sum()) if prev_blocked is None else int((blocked & ~prev_blocked).sum())
        pct_newly = 100.0 * n_newly / n_cells
        stopped_early = prev_blocked is not None and pct_newly < outer_convergence_pct

        change = _depth_change_metrics(prev_depth, depth)
        snap = _snapshot_from_depth(depth, n_outer, stopped_early, max_change, epsilon,
                                     time.perf_counter() - t0, iter_elapsed, change_stats, pct_unreached)
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
    "last_max_change", "epsilon", "inner_converged", "pct_unreached", "pct_cells_still_changing",
    "mean_cell_change", "min_cell_change", "max_cell_change", "max_depth_change_abs",
    "n_newly_flooded", "n_no_longer_flooded", "n_inundated", "depth_mean",
    "depth_min", "depth_max", "depth_q20", "depth_q80", "status",
    "pct_removed_by_static_filter_alone", "pct_additional_removed_by_full_outer_loop",
    "n_outer_iterations_to_convergence",
]


def build_summary_row(rows: list[dict], tile_id: int) -> dict:
    """Summarize % of baseline flooding removed by static pre-filter vs full outer loop."""
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
    # allow_abbrev=False: a removed flag that is a prefix of its replacement must be an error.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-outer-iterations", type=int, default=None, help=f"default: {DEFAULT_MAX_OUTER} (production uses config simulation.flooding.obstacle_coupling.max_outer_iterations)")
    parser.add_argument("--max-rounds", type=int, default=None, help=f"default: {DEFAULT_INNER_MAX_ROUNDS} (production uses config simulation.flooding.max_rounds)")
    parser.add_argument("--epsilon", type=float, default=None, help=f"default: {DEFAULT_OUTER_EPSILON_M} (matches production's own config simulation.flooding.waterlevel_epsilon_m)")
    parser.add_argument("--outer-convergence-pct", type=float, default=None, help="default: config simulation.flooding.obstacle_coupling.outer_convergence_pct")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = args.tile_ids if args.tile_ids else TILES

    cfg = load_config(args.config)
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    flooding_cfg = cfg["simulation"]["flooding"]
    knn = int(flooding_cfg["knn"])
    friction_scale_factor = float(flooding_cfg["friction_scale_factor"])
    max_outer = args.max_outer_iterations if args.max_outer_iterations is not None else DEFAULT_MAX_OUTER
    inner_max_rounds = args.max_rounds if args.max_rounds is not None else DEFAULT_INNER_MAX_ROUNDS
    epsilon = args.epsilon if args.epsilon is not None else DEFAULT_OUTER_EPSILON_M
    outer_convergence_pct = args.outer_convergence_pct if args.outer_convergence_pct is not None else float(flooding_cfg["obstacle_coupling"]["outer_convergence_pct"])
    print(f"model_outputs={model_outputs}  knn={knn}  friction_scale_factor={friction_scale_factor}  "
          f"max_outer={max_outer}  inner_max_rounds={inner_max_rounds}  epsilon={epsilon}  "
          f"outer_convergence_pct={outer_convergence_pct}", flush=True)

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows = run_tile_trace(
                    tile_id, model_outputs, knn, friction_scale_factor,
                    max_outer, inner_max_rounds, epsilon, outer_convergence_pct,
                )
                for row in tile_rows:
                    label = "baseline (no blocking)" if row["n_outer"] == 0 else f"outer {row['n_outer']}"
                    print(f"  {label}: cum_time={row['cum_time_s']}s  "
                          f"iter_time={row['iter_time_s']}s  n_rounds_used={row['n_rounds_used']}"
                          f"{'  (stopped early)' if row['outer_stopped_early'] else ''}  "
                          f"pct_blocked_cumulative={row['pct_blocked_cumulative']}%  "
                          f"pct_newly_blocked={row['pct_newly_blocked']}%  "
                          f"last_max_change={row['last_max_change']:.8f}  "
                          f"inner_converged={row['inner_converged']}  "
                          f"pct_unreached={row['pct_unreached']}%", flush=True)
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
