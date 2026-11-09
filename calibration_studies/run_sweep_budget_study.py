"""Trace the non-coupling eikonal solve sweep by sweep on calibration tiles.

Tiles come from select_study_candidate_tiles.py's candidate_tiles.txt. Each tile is solved by
production's own path (eikonal_runner.solve_wave0_tile: inputs, seeds, two-component static level,
duration correction B, block sweeps) with obstacle coupling off and up to MAX_ROUNDS_CEILING
rounds; the solver stops once a round's max change <= --epsilon. With correction B production
solves twice (solve_pass "first", then "duration" with the corrected friction); both are traced.
A tile dry at the first sweep is logged DRY and stopped (the candidate pool is over-provisioned).
Writes one CSV per tile with per-sweep depth-change metrics.

Usage:
    python run_sweep_budget_study.py --out-dir <output_dir> [--config <config.yml>] [--tile-ids id [id ...]]
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
from eikonal_runner import solve_wave0_tile  # noqa: E402
from flood_model import SolveTrace  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# Smoke-test fallback when no --tile-ids are given.
TILES = [2083, 2514, 1421, 2569, 2858, 2314, 753]

SWEEPS_PER_ROUND = 4  # one full cycle of eikonal._ORTHANT_ORDER
MAX_ROUNDS_CEILING = 100
N_TILES_WANTED_DEFAULT = 500  # only reachable in a single sequential run, not per HPC node


def study_overrides(friction_source: str | None, uniform_m_per_km: float | None) -> dict:
    """Return solve_wave0_tile overrides for a friction source (None: the config's)."""
    out = {}
    if friction_source is not None:
        out["friction_source"] = friction_source
    if uniform_m_per_km is not None:
        out["uniform_attenuation_m_per_km"] = uniform_m_per_km
    return out


class DryTile(Exception):
    pass


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray, n_cells: int) -> dict:
    """Median/mean use changed cells only (flooding is a small minority of a tile).

    No un-flooded count: without blocking, sweeps only lower t, so flooding only grows.
    """
    abs_diff = np.abs(depth_curr - depth_prev)
    changed = abs_diff > 0
    n_changed = int(changed.sum())
    changed_vals = abs_diff[changed]
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "median_depth_change_abs": round(float(np.median(changed_vals)), 4) if changed_vals.size else 0.0,
        "mean_depth_change_abs": round(float(changed_vals.mean()), 4) if changed_vals.size else 0.0,
        "n_newly_flooded": int(((depth_prev == 0) & (depth_curr > 0)).sum()),
        "n_cells_changed": n_changed,
        "pct_cells_changed": round(100.0 * n_changed / n_cells, 6),
    }


class SweepTrace(SolveTrace):
    """Record one row per sweep of every solve pass."""

    def __init__(self, tile_id: int):
        self.tile_id = tile_id
        self.rows: list[dict] = []

    def solve_started(self, solve_pass, outer, depth_of) -> None:
        self.solve_pass, self.depth_of = solve_pass, depth_of
        self.sweep_count, self.prev_depth, self.t0 = 0, None, time.perf_counter()

    def sweep(self, t, change) -> None:
        self.sweep_count += 1
        depth = self.depth_of(-t[1:, 1:])
        if self.prev_depth is None:
            self.prev_depth = np.zeros_like(depth)  # implicit all-dry "sweep 0"
        n_inundated = int((depth > 0).sum())
        if self.solve_pass == "first" and self.sweep_count == 1 and n_inundated == 0:
            raise DryTile
        row = {"tile": self.tile_id, "solve_pass": self.solve_pass, "n_cells": depth.size,
               "sweep_count": self.sweep_count, "cum_time_s": round(time.perf_counter() - self.t0, 3),
               "sweep_max_change_raw": round(float(change), 8), "n_inundated": n_inundated,
               "depth_mean": round(float(depth[depth > 0].mean()), 4) if n_inundated else 0.0,
               "depth_sum": round(float(depth.sum()), 2), "depth_max": round(float(depth.max()), 4),
               "status": "ok", **_depth_change_metrics(self.prev_depth, depth, depth.size)}
        self.rows.append(row)
        self.prev_depth = depth


def run_tile_sweep_trace(config: dict, tile_id: int, overrides: dict, epsilon: float) -> tuple[list[dict], bool, bool]:
    """Trace one tile's production solve; return (rows, is_wet, round_converged of the last pass)."""
    trace = SweepTrace(tile_id)
    try:
        solve_wave0_tile(config, tile_id, RETURN_PERIOD, WATERLEVEL_NAME, _REPO_ROOT, trace=trace,
                         obstacle_coupling=False, max_rounds=MAX_ROUNDS_CEILING, waterlevel_epsilon_m=epsilon,
                         **overrides)
    except DryTile:
        return trace.rows, False, False
    last = [r for r in trace.rows if r["solve_pass"] == trace.rows[-1]["solve_pass"]]
    converged = max(r["sweep_max_change_raw"] for r in last[-SWEEPS_PER_ROUND:]) <= epsilon
    return trace.rows, True, converged


FIELDNAMES = [
    "tile", "solve_pass", "n_cells", "sweep_count", "cum_time_s", "sweep_max_change_raw", "n_inundated",
    "depth_mean", "depth_sum", "depth_max", "max_depth_change_abs", "median_depth_change_abs",
    "mean_depth_change_abs", "n_newly_flooded", "n_cells_changed", "pct_cells_changed", "status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids", type=int, nargs="*", default=None)
    parser.add_argument("--friction-source", choices=["landcover", "uniform"], default=None,
                        help="default: config simulation.flooding.friction_source")
    parser.add_argument("--uniform-attenuation-m-per-km", type=float, default=None,
                        help="uniform rate (default: config simulation.flooding.uniform_attenuation_m_per_km)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="round-level convergence threshold (default: simulation.flooding.waterlevel_epsilon_m)")
    parser.add_argument("--n-tiles-wanted", type=int, default=N_TILES_WANTED_DEFAULT, help=f"stop early once this many wet tiles are found, in this process's own single-process run only (default: {N_TILES_WANTED_DEFAULT})")
    parser.add_argument(
        "--write-wet-tiles-summary", action="store_true",
        help="write wet_tiles_selected.txt from THIS process's own tiles only - only safe for a "
             "single sequential run that sees every candidate tile. Off by default: under any "
             "parallel HPC split (array job OR N-node batch job), every concurrent "
             "process would otherwise race to overwrite the same file with just its own tiny "
             "slice. Use aggregate_wet_tiles.py after a parallel run instead - it scans every "
             "tile's own CSV post-hoc and is safe regardless of how the work was split.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = args.tile_ids if args.tile_ids else TILES
    n_tiles_wanted = args.n_tiles_wanted

    cfg = load_config(args.config)
    overrides = study_overrides(args.friction_source, args.uniform_attenuation_m_per_km)
    if args.epsilon is None:
        args.epsilon = float(cfg["simulation"]["flooding"]["waterlevel_epsilon_m"])
    print(f"model_outputs={cfg['simulation']['model_outputs']}  overrides={overrides}  "
          f"max_rounds={MAX_ROUNDS_CEILING}  epsilon={args.epsilon}  n_tiles_wanted={n_tiles_wanted}", flush=True)

    wet_tiles: list[int] = []
    errored_tiles: list[int] = []

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows, is_wet, round_converged = run_tile_sweep_trace(cfg, tile_id, overrides, args.epsilon)
                for row in tile_rows:
                    print(f"  {row['solve_pass']:>8} sweeps={row['sweep_count']:3d}  cum_time={row['cum_time_s']}s  "
                          f"n_inundated={row['n_inundated']:,}  depth_sum={row['depth_sum']}  "
                          f"max_change={row['max_depth_change_abs']}  "
                          f"n_newly_flooded={row['n_newly_flooded']:,}  "
                          f"n_changed={row['n_cells_changed']:,} ({row['pct_cells_changed']}%)", flush=True)
                    writer.writerow(row)
                    f.flush()
                if is_wet:
                    wet_tiles.append(tile_id)
                    n_rounds = sum(r["solve_pass"] == tile_rows[-1]["solve_pass"] for r in tile_rows) // SWEEPS_PER_ROUND
                    conv_label = f"round-converged at round {n_rounds}" if round_converged else \
                        f"did NOT converge within {MAX_ROUNDS_CEILING} rounds"
                    print(f"  SUMMARY: wet, last pass {conv_label} ({len(wet_tiles)}/{n_tiles_wanted} selected so far)",
                          flush=True)
                else:
                    print("  SUMMARY: DRY at sweep 1 - excluded", flush=True)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
                err_row = {k: "" for k in FIELDNAMES}
                err_row["tile"] = tile_id
                err_row["status"] = f"error: {exc}"
                writer.writerow(err_row)
                errored_tiles.append(tile_id)

        if len(wet_tiles) >= n_tiles_wanted:
            print(f"\n{n_tiles_wanted} wet tiles found - stopping early, "
                  f"{len(tiles) - tiles.index(tile_id) - 1} remaining candidate(s) not needed", flush=True)
            break

    print(f"\nDone. Per-tile CSVs written to {out_dir}")
    if errored_tiles:
        print(f"WARNING: {len(errored_tiles)} tile(s) errored (status 'error' row, neither wet nor dry): "
              f"{errored_tiles}", flush=True)
    # Off by default: parallel processes would race on this file (use aggregate_wet_tiles.py).
    if args.write_wet_tiles_summary:
        selected = wet_tiles[:n_tiles_wanted]
        wet_file = out_dir / "wet_tiles_selected.txt"
        with open(wet_file, "w") as f:
            for tile_id in selected:
                f.write(f"{tile_id}\n")
        print(f"{len(selected)}/{n_tiles_wanted} wet tiles selected, written to {wet_file}")
        if len(selected) < n_tiles_wanted:
            print(f"WARNING: only found {len(selected)} wet tiles out of {len(tiles)} candidates - "
                  f"need more candidates from select_study_candidate_tiles.py", flush=True)


if __name__ == "__main__":
    main()
