"""Plot the 4-panel sweep-budget calibration figure (docs/methods_03 §2).

(a) ECDF of rounds to round_max_change <= epsilon; per-round bands over wet
tiles of (b) raw solver residual [m], (c) connectivity-pruned max depth change
[m], (d) newly flooded cells. Tiles drop out once converged; no carry-forward.
Rebuilds wet_tiles_selected.txt first (overwrites it).

Usage:
    python plot_sweep_calibration_bands.py <sweep_budget_dir> --fig-dir <figures_dir> [--max-rounds 40] [--n-tiles-wanted 500]

<figures_dir> is also accepted as a second positional argument.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aggregate_wet_tiles import aggregate_wet_tiles  # noqa: E402
from cli_args import add_option_with_positional, require  # noqa: E402
from config_utils import load_config  # noqa: E402
from plot_sweep_budget_convergence import (  # noqa: E402
    N_COMPLETE_ROUNDS, SWEEPS_PER_ROUND, band_stats, collect as collect_convergence, plot_band, read_trace_csv,
)

N_TILES_WANTED_DEFAULT = 500  # matches aggregate_wet_tiles.py's own default

_REPO_ROOT = Path(__file__).resolve().parent.parent
FLOOR = 1e-4  # log-scale display floor for panels (b)/(c)
FLOOR_COUNT = 0.2  # log-scale display floor for panel (d)'s raw cell counts


def _aggregate_to_rounds(csv_path: Path) -> pd.DataFrame | None:
    # Per 4-sweep round: max of change columns, sum of newly flooded, last n_inundated.
    df = read_trace_csv(csv_path, "sweep_max_change_raw")
    if df is None or len(df) < SWEEPS_PER_ROUND:
        return None  # unreadable, or a dry tile

    tile_id = int(df["tile"].iloc[0])
    n_rounds = len(df) // SWEEPS_PER_ROUND
    records = []
    for r in range(1, n_rounds + 1):
        chunk = df.iloc[(r - 1) * SWEEPS_PER_ROUND: r * SWEEPS_PER_ROUND]
        records.append({
            "tile": tile_id,
            "round": r,
            "round_max_change": chunk["sweep_max_change_raw"].max(),
            "round_max_depth_change_abs": chunk["max_depth_change_abs"].max(),
            "round_n_newly_flooded": chunk["n_newly_flooded"].sum(),
            "round_n_inundated": chunk["n_inundated"].iloc[-1],
        })
    return pd.DataFrame.from_records(records)


def _plot_cdf(ax, cdf_df: pd.DataFrame, max_rounds: int) -> None:
    converged = cdf_df[cdf_df["converged"]].sort_values("n_rounds_used")
    n_censored = int((~cdf_df["converged"]).sum())
    n_total = len(cdf_df)

    xs = converged["n_rounds_used"].to_numpy()
    ys = np.arange(1, len(xs) + 1) / n_total  # censored tiles show as the gap to 1.0

    real_max = int(xs.max()) if len(xs) else 1
    censored_x = real_max + max(3, real_max // 5)

    ax.step(xs, ys, where="post", linewidth=2, color="#1f77b4", label=f"converged (n={n_total - n_censored})")
    if n_censored:
        ax.hlines(ys[-1] if len(ys) else 0.0, xs[-1] if len(xs) else 0, censored_x,
                   color="#1f77b4", linewidth=2, linestyle="--")
        ax.scatter([censored_x], [ys[-1] if len(ys) else 0.0], marker="x", color="#d62728", s=80, zorder=5,
                   label=f"not converged within {N_COMPLETE_ROUNDS} rounds (n={n_censored})")
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_xlabel("rounds to converge (round_max_change <= epsilon)")
    ax.set_ylabel("cumulative fraction of tiles")
    ax.set_xlim(0, censored_x + 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)


def make_figure(long_df: pd.DataFrame, cdf_df: pd.DataFrame, epsilon: float, max_rounds: int, out_path: Path) -> None:
    """Draw the 4-panel figure to out_path."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))

    ax = axes[0, 0]
    _plot_cdf(ax, cdf_df, max_rounds)
    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=13)

    ax = axes[0, 1]
    plot_band(ax, band_stats(long_df, "round_max_change", "round"), "#d62728", "max_change (raw solver residual)", floor=FLOOR)
    ax.axhline(epsilon, color="grey", linestyle=":", linewidth=1, label=f"epsilon ({epsilon}m)")
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("metres")
    ax.set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    plot_band(ax, band_stats(long_df, "round_max_depth_change_abs", "round"), "#ff7f0e", "max_depth_change_abs (flood extent only)", floor=FLOOR)
    ax.axhline(epsilon, color="grey", linestyle=":", linewidth=1, label=f"epsilon ({epsilon}m)")
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("metres")
    ax.set_title("(c)", loc="left", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    plot_band(ax, band_stats(long_df, "round_n_newly_flooded", "round"), "#1f77b4", "newly flooded cells", floor=FLOOR_COUNT)
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("newly flooded cells (this round, log)")
    ax.set_title("(d)", loc="left", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_budget_dir")
    add_option_with_positional(parser, "--fig-dir", "figures_dir")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--epsilon", type=float, default=None, help="default: config simulation.flooding.waterlevel_epsilon_m")
    parser.add_argument("--max-rounds", type=int, default=None, help="default: config simulation.flooding.max_rounds")
    parser.add_argument(
        "--n-tiles-wanted", type=int, default=N_TILES_WANTED_DEFAULT,
        help=f"forwarded to aggregate_wet_tiles.py's own selection, run as this script's first "
             f"step (default: {N_TILES_WANTED_DEFAULT})",
    )
    args = parser.parse_args()
    require(parser, args, "figures_dir", "--fig-dir")

    cfg = load_config(args.config)
    epsilon = args.epsilon if args.epsilon is not None else float(cfg["simulation"]["flooding"]["waterlevel_epsilon_m"])
    max_rounds = args.max_rounds if args.max_rounds is not None else int(cfg["simulation"]["flooding"]["max_rounds"])

    sweep_budget_dir = Path(args.sweep_budget_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tile_ids = aggregate_wet_tiles(sweep_budget_dir, args.n_tiles_wanted)
    print(f"{len(tile_ids)} wet tile(s) selected (wet_tiles_selected.txt rebuilt fresh)")

    records = []
    for tid in tile_ids:
        csv_path = sweep_budget_dir / f"{tid}.csv"
        if not csv_path.exists():
            continue
        rounds_df = _aggregate_to_rounds(csv_path)
        if rounds_df is not None:
            records.append(rounds_df)
    long_df = pd.concat(records, ignore_index=True)
    n_tiles = long_df["tile"].nunique()
    print(f"{n_tiles} tile(s) with a usable sweep trace, {len(long_df)} tile-round row(s)")

    # Full N_COMPLETE_ROUNDS ceiling, not just max_rounds.
    cdf_df = collect_convergence(sweep_budget_dir, epsilon)
    cdf_df = cdf_df[cdf_df["tile"].isin(tile_ids)]
    print(f"{len(cdf_df)} tile(s) in the CDF panel ({int(cdf_df['converged'].sum())} converged "
          f"within {N_COMPLETE_ROUNDS} rounds)")

    make_figure(long_df, cdf_df, epsilon, max_rounds, figures_dir / "sweep_calibration_bands.png")


if __name__ == "__main__":
    main()
