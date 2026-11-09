"""Trace obstacle-coupling outer iterations on the wet calibration tiles.

Tiles come from run_sweep_budget_study.py's wet_tiles_selected.txt. Each tile is solved by
production's own path (eikonal_runner.solve_wave0_tile: inputs, seeds, two-component static
level, duration correction B, block sweeps, flood_model's outer loop), once without coupling
(the n_outer=0 baseline) and once coupled with up to --max-outer-iterations outer iterations.
With correction B production solves twice per run (solve_pass "first", then "duration"); every
outer iteration of both passes gets a row, and a status="summary" row splits the last pass's
flooding removed by the static pre-filter alone vs the full outer loop.

Usage:
    python run_obstacle_coupling_study.py --out-dir <output_dir> [--config <config.yml>]
        [--tile-ids id [id ...]] [--max-outer-iterations N] [--max-rounds N] [--epsilon M]

The flag names mirror the config keys (obstacle_coupling.max_outer_iterations,
flooding.max_rounds).
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config  # noqa: E402
from eikonal import UNREACHED_T  # noqa: E402
from eikonal_runner import solve_wave0_tile  # noqa: E402
from flood_model import SolveTrace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep_budget_study import RETURN_PERIOD, TILES, WATERLEVEL_NAME, study_overrides  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_OUTER = 15  # beyond production's default, to find where tiles stop needing more
# Production's config.yml max_rounds; the calibration config's 100 is the sweep study's ceiling.
DEFAULT_INNER_ROUNDS = 40


def _depth_row(depth: np.ndarray, waterlevel: np.ndarray | None) -> dict:
    flood = depth > 0
    n_inundated = int(flood.sum())
    row = {"n_inundated": n_inundated,
           "pct_unreached": "" if waterlevel is None else
           round(100.0 * float(np.count_nonzero(waterlevel <= -UNREACHED_T)) / depth.size, 6)}
    if n_inundated:
        v = depth[flood]
        row.update(depth_mean=round(float(v.mean()), 4), depth_min=round(float(v.min()), 4),
                   depth_max=round(float(v.max()), 4), depth_q20=round(float(np.percentile(v, 20)), 4),
                   depth_q80=round(float(np.percentile(v, 80)), 4))
    else:
        row.update(depth_mean=0.0, depth_min=0.0, depth_max=0.0, depth_q20=0.0, depth_q80=0.0)
    return row


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray) -> dict:
    """Count both directions: re-blocking a cell can un-flood it."""
    abs_diff = np.abs(depth_curr - depth_prev)
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "n_newly_flooded": int(((depth_prev == 0) & (depth_curr > 0)).sum()),
        "n_no_longer_flooded": int(((depth_prev > 0) & (depth_curr == 0)).sum()),
    }


class OuterTrace(SolveTrace):
    """Record one row per outer iteration of every solve pass, and the last uncoupled solve."""

    def __init__(self, tile_id: int, epsilon: float, outer_convergence_pct: float):
        self.tile_id, self.epsilon, self.outer_convergence_pct = tile_id, epsilon, outer_convergence_pct
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()
        self.prev_depth = None

    def solve_started(self, solve_pass, outer, depth_of) -> None:
        self.solve_pass, self.depth_of, self.iter_t0 = solve_pass, depth_of, time.perf_counter()
        self.n_sweeps, self.round_change, self.last_round_change = 0, 0.0, 0.0
        if outer <= 1:
            self.prev_depth = None

    def sweep(self, t, change) -> None:
        self.n_sweeps += 1
        self.round_change = max(self.round_change, change)
        if self.n_sweeps % 4 == 0:
            self.last_round_change, self.round_change = self.round_change, 0.0

    def outer_done(self, outer, waterlevel, blocked, pct_newly_blocked, inner) -> None:
        depth = self.depth_of(waterlevel)
        prev = self.prev_depth if self.prev_depth is not None else np.zeros_like(depth)
        row = {"tile": self.tile_id, "solve_pass": self.solve_pass, "n_outer": outer, "n_cells": depth.size,
               "cum_time_s": round(time.perf_counter() - self.t0, 3),
               "iter_time_s": round(time.perf_counter() - self.iter_t0, 3),
               "n_rounds_used": inner.get("n_rounds_used"),
               "outer_stopped_early": outer > 1 and pct_newly_blocked < self.outer_convergence_pct,
               "pct_blocked_cumulative": round(100.0 * int(blocked.sum()) / depth.size, 6),
               "pct_newly_blocked": round(pct_newly_blocked, 6),
               "last_max_change": round(float(self.last_round_change), 8), "epsilon": self.epsilon,
               "inner_converged": bool(inner.get("converged")), "status": "ok",
               **_depth_row(depth, waterlevel), **_depth_change_metrics(prev, depth)}
        self.rows.append(row)
        self.prev_depth = depth


class BaselineTrace(SolveTrace):
    """Keep the round count and last round change of the last uncoupled solve pass."""

    def solve_started(self, solve_pass, outer, depth_of) -> None:
        self.n_sweeps, self.round_change, self.last_round_change = 0, 0.0, 0.0

    def sweep(self, t, change) -> None:
        self.n_sweeps += 1
        self.round_change = max(self.round_change, change)
        if self.n_sweeps % 4 == 0:
            self.last_round_change, self.round_change = self.round_change, 0.0


def run_tile_trace(config: dict, tile_id: int, overrides: dict, max_outer: int, inner_max_rounds: int,
                   epsilon: float, outer_convergence_pct: float) -> list[dict]:
    """Trace the uncoupled baseline and each outer iteration; one row per solve."""
    settings = {"max_rounds": inner_max_rounds, "waterlevel_epsilon_m": epsilon, **overrides}
    base = BaselineTrace()
    t0 = time.perf_counter()
    depth0, _ = solve_wave0_tile(config, tile_id, RETURN_PERIOD, WATERLEVEL_NAME, _REPO_ROOT, trace=base,
                                 obstacle_coupling=False, **settings)
    baseline = {"tile": tile_id, "solve_pass": "baseline", "n_outer": 0, "n_cells": depth0.size,
                "cum_time_s": round(time.perf_counter() - t0, 3), "iter_time_s": round(time.perf_counter() - t0, 3),
                "n_rounds_used": -(-base.n_sweeps // 4), "outer_stopped_early": False,
                "pct_blocked_cumulative": 0.0, "pct_newly_blocked": 0.0,
                "last_max_change": round(float(base.last_round_change), 8), "epsilon": epsilon,
                "inner_converged": bool(base.last_round_change <= epsilon), "status": "ok",
                **_depth_row(depth0, None),
                "max_depth_change_abs": 0.0, "n_newly_flooded": 0, "n_no_longer_flooded": 0}
    trace = OuterTrace(tile_id, epsilon, outer_convergence_pct)
    solve_wave0_tile(config, tile_id, RETURN_PERIOD, WATERLEVEL_NAME, _REPO_ROOT, trace=trace,
                     obstacle_coupling=True, max_outer_iterations=max_outer,
                     outer_convergence_pct=outer_convergence_pct, **settings)
    return [baseline, *trace.rows]


FIELDNAMES = [
    "tile", "solve_pass", "n_outer", "n_cells", "cum_time_s", "iter_time_s", "n_rounds_used",
    "outer_stopped_early", "pct_blocked_cumulative", "pct_newly_blocked",
    "last_max_change", "epsilon", "inner_converged", "pct_unreached", "max_depth_change_abs",
    "n_newly_flooded", "n_no_longer_flooded", "n_inundated", "depth_mean",
    "depth_min", "depth_max", "depth_q20", "depth_q80", "status",
    "pct_removed_by_static_filter_alone", "pct_additional_removed_by_full_outer_loop",
    "n_outer_iterations_to_convergence",
]


def build_summary_row(rows: list[dict], tile_id: int) -> dict:
    """Summarize % of baseline flooding removed by static pre-filter vs full outer loop (last pass)."""
    baseline = rows[0]  # n_outer=0: no blocking at all
    last = [r for r in rows[1:] if r["solve_pass"] == rows[-1]["solve_pass"]]
    static_only = last[0]  # n_outer=1: static pre-filter + one inner solve, zero re-blocking
    final = last[-1]  # converged (or capped) row

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
    parser.add_argument("--friction-source", choices=["landcover", "uniform"], default=None,
                        help="default: config simulation.flooding.friction_source")
    parser.add_argument("--uniform-attenuation-m-per-km", type=float, default=None,
                        help="uniform rate (default: config simulation.flooding.uniform_attenuation_m_per_km)")
    parser.add_argument("--max-outer-iterations", type=int, default=DEFAULT_MAX_OUTER,
                        help="default: %(default)s (production: config obstacle_coupling.max_outer_iterations)")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_INNER_ROUNDS,
                        help="inner solve cap (default: %(default)s, production's)")
    parser.add_argument("--epsilon", type=float, default=None, help="default: config simulation.flooding.waterlevel_epsilon_m")
    parser.add_argument("--outer-convergence-pct", type=float, default=None, help="default: config simulation.flooding.obstacle_coupling.outer_convergence_pct")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = args.tile_ids if args.tile_ids else TILES

    cfg = load_config(args.config)
    flooding_cfg = cfg["simulation"]["flooding"]
    overrides = study_overrides(args.friction_source, args.uniform_attenuation_m_per_km)
    epsilon = args.epsilon if args.epsilon is not None else float(flooding_cfg["waterlevel_epsilon_m"])
    outer_convergence_pct = (args.outer_convergence_pct if args.outer_convergence_pct is not None
                             else float(flooding_cfg["obstacle_coupling"]["outer_convergence_pct"]))
    print(f"model_outputs={cfg['simulation']['model_outputs']}  overrides={overrides}  "
          f"max_outer={args.max_outer_iterations}  inner_max_rounds={args.max_rounds}  epsilon={epsilon}  "
          f"outer_convergence_pct={outer_convergence_pct}", flush=True)

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows = run_tile_trace(cfg, tile_id, overrides, args.max_outer_iterations, args.max_rounds,
                                           epsilon, outer_convergence_pct)
                for row in tile_rows:
                    print(f"  {row['solve_pass']:>8} outer {row['n_outer']}: iter_time={row['iter_time_s']}s  "
                          f"n_rounds_used={row['n_rounds_used']}"
                          f"{'  (stopped early)' if row['outer_stopped_early'] else ''}  "
                          f"pct_blocked_cumulative={row['pct_blocked_cumulative']}%  "
                          f"pct_newly_blocked={row['pct_newly_blocked']}%  "
                          f"inner_converged={row['inner_converged']}  n_inundated={row['n_inundated']:,}  "
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
