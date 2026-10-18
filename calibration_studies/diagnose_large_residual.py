"""Diagnose large single-cell eikonal residuals (oscillation vs slow convergence).

Replays the static pre-filter + first outer inner solve and prints the
largest-update cell per round; one cell flipping between a few values is a
limit cycle, not slow convergence.

Usage:
    python diagnose_large_residual.py <tile_id> [<tile_id> ...] [--config <config.yml>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_model import OBSTACLE_BLOCK_FRICTION as BLOCK_FRICTION  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep_budget_study import load_tile  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
INNER_MAX_ROUNDS = 25
EPSILON = 0.03  # matches production's simulation.flooding.waterlevel_epsilon_m


def diagnose(tile_id: int, model_outputs: Path, knn: int, friction_scale_factor: float, n_rounds=INNER_MAX_ROUNDS):
    """Run inner sweep rounds and print the largest-update cell per round."""
    print(f"\n=== tile {tile_id} ===", flush=True)
    dem, mask, friction, _, seed_rows, seed_cols, initial, max_waterlevel = load_tile(
        tile_id, model_outputs, knn, friction_scale_factor, on_no_stations="raise",
    )
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

    locs = [(h["row"], h["col"]) for h in history]
    unique_locs = set(locs)
    print(f"  {len(history)} rounds run, {len(unique_locs)} unique argmax location(s): "
          f"{sorted(unique_locs)[:10]}{' ...' if len(unique_locs) > 10 else ''}", flush=True)
    if len(unique_locs) <= 3 and len(history) >= 4:
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tile_ids", type=int, nargs="+")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    knn = int(cfg["simulation"]["flooding"]["knn"])
    friction_scale_factor = float(cfg["simulation"]["flooding"]["friction_scale_factor"])

    for tile_id in args.tile_ids:
        diagnose(tile_id, model_outputs, knn, friction_scale_factor)


if __name__ == "__main__":
    main()
