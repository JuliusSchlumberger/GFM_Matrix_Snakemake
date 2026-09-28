"""Single 4-panel summary figure for the sweep-budget calibration study
(see tests/test_sweep_budget_calibration.py and
docs/methods_03_calibration_sensitivity.md §2).

Panels (b)-(d) are median/98th-percentile/min-max bands across the wet
calibration tiles (`wet_tiles_selected.txt`) that still have a real
observation at that round, plotted against round. Reads
test_sweep_budget_calibration.py's per-SWEEP raw CSVs directly (not
sweep_convergence_summary.csv, which only has one row per tile) and
aggregates to per-ROUND values - see _aggregate_to_rounds for the exact
rule per column. A tile's own trace ends once production's round-level
early-exit criterion (round_max_change <= epsilon) is satisfied, so the
number of tiles contributing to a given round's band shrinks at higher
round numbers as tiles finish - real (raw, unmodified) data throughout,
no imputation/carry-forward for rounds past a tile's own trace.

Panel (a) is the ECDF of rounds-to-converge, reusing
plot_sweep_budget_convergence.collect() (restricted to the same wet-tile
population as (b)-(d)) so both views of the same underlying convergence
event are in one figure - this is the more direct evidence for the
max_rounds cutoff itself (fraction of tiles converged by round R), with
(b)-(d) as supporting context on what the residual/depth/extent actually
look like at that point.

Panels:
  (a) ECDF of rounds-to-converge (round_max_change <= epsilon) - fraction
      of tiles converged by round R.
  (b) max_change (raw solver residual, sweep_max_change_raw) - solver
      stability, the exact quantity production's real convergence check
      uses.
  (c) max_depth_change_abs (connectivity-pruned) - the actual max depth
      change within the real flood extent, contrasted against (b) in
      docs/methods_03_calibration_sensitivity.md's own discussion of why
      they diverge.
  (d) newly flooded cells per round (raw counts, not cumulative, not
      normalized by tile size - log scale).

Usage:
    python plot_sweep_calibration_bands.py <sweep_budget_dir> <figures_dir> [--max-rounds 40]
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
from config_utils import load_config  # noqa: E402
from plot_sweep_budget_convergence import N_COMPLETE_ROUNDS, collect as collect_convergence  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEPS_PER_ROUND = 4
FLOOR = 1e-4  # log-scale display floor for panels (b)/(c)
FLOOR_COUNT = 0.2  # log-scale display floor for panel (d)'s raw cell counts


def _aggregate_to_rounds(csv_path: Path) -> pd.DataFrame | None:
    """One row per complete round (4 sweeps) for one tile:
    round_max_change (max of sweep_max_change_raw - the exact quantity
    solve_eikonal_dense's own round loop compares to epsilon),
    round_max_depth_change_abs (max of max_depth_change_abs),
    round_n_newly_flooded (SUM of n_newly_flooded - a true per-round
    total, not a max), round_n_inundated (value at the round's LAST sweep
    - the flooded-cell count as of the end of that round).
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "sweep_max_change_raw" not in df.columns:
        return None
    if len(df) < SWEEPS_PER_ROUND:
        return None  # dry tile or otherwise too short to form a full round

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


def _band_stats(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = long_df.groupby("round")[value_col]
    return pd.DataFrame({
        "median": g.median(), "p02": g.quantile(0.02), "p98": g.quantile(0.98),
        "min": g.min(), "max": g.max(), "n": g.count(),
    })


def _plot_band(ax, s: pd.DataFrame, color: str, label: str, floor: float | None = None, show_p02: bool = False) -> None:
    lo, hi, med, p98 = s["min"], s["max"], s["median"], s["p98"]
    if floor is not None:
        lo, hi, med, p98 = (x.clip(lower=floor) for x in (lo, hi, med, p98))
    ax.fill_between(s.index, lo, hi, alpha=0.15, color=color)
    ax.plot(s.index, med, color=color, linewidth=2, label=f"{label} - median")
    ax.plot(s.index, p98, color=color, linewidth=1.3, linestyle="--", label=f"{label} - 98th percentile")
    if show_p02:
        p02 = s["p02"].clip(lower=floor) if floor is not None else s["p02"]
        ax.plot(s.index, p02, color=color, linewidth=1.3, linestyle="-.", label=f"{label} - 2nd percentile")


def _plot_cdf(ax, cdf_df: pd.DataFrame, max_rounds: int) -> None:
    converged = cdf_df[cdf_df["converged"]].sort_values("n_rounds_used")
    n_censored = int((~cdf_df["converged"]).sum())
    n_total = len(cdf_df)

    xs = converged["n_rounds_used"].to_numpy()
    ys = np.arange(1, len(xs) + 1) / n_total  # denominator = ALL tiles, so censored tiles show as the gap to 1.0

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
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))

    # (a) ECDF of rounds-to-converge
    ax = axes[0, 0]
    _plot_cdf(ax, cdf_df, max_rounds)
    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=13)

    # (b) raw solver residual - solver stability
    ax = axes[0, 1]
    _plot_band(ax, _band_stats(long_df, "round_max_change"), "#d62728", "max_change (raw solver residual)", floor=FLOOR)
    ax.axhline(epsilon, color="grey", linestyle=":", linewidth=1, label=f"epsilon ({epsilon}m)")
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("metres")
    ax.set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) max depth change among actually-flooded cells (connectivity-pruned)
    ax = axes[1, 0]
    _plot_band(ax, _band_stats(long_df, "round_max_depth_change_abs"), "#ff7f0e", "max_depth_change_abs (flood extent only)", floor=FLOOR)
    ax.axhline(epsilon, color="grey", linestyle=":", linewidth=1, label=f"epsilon ({epsilon}m)")
    ax.axvline(max_rounds, color="black", linestyle="--", linewidth=1.3, label=f"max_rounds={max_rounds}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("metres")
    ax.set_title("(c)", loc="left", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) newly flooded cells per round (raw counts, log)
    ax = axes[1, 1]
    _plot_band(ax, _band_stats(long_df, "round_n_newly_flooded"), "#1f77b4", "newly flooded cells", floor=FLOOR_COUNT)
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
    parser.add_argument("figures_dir")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--epsilon", type=float, default=None, help="default: config simulation.flooding.waterlevel_epsilon_m")
    parser.add_argument("--max-rounds", type=int, default=None, help="default: config simulation.flooding.max_rounds")
    args = parser.parse_args()

    cfg = load_config(args.config)
    epsilon = args.epsilon if args.epsilon is not None else float(cfg["simulation"]["flooding"]["waterlevel_epsilon_m"])
    max_rounds = args.max_rounds if args.max_rounds is not None else int(cfg["simulation"]["flooding"]["max_rounds"])

    sweep_budget_dir = Path(args.sweep_budget_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    wet_tiles_path = sweep_budget_dir / "wet_tiles_selected.txt"
    tile_ids = [int(line.strip()) for line in wet_tiles_path.read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} wet tile(s) from {wet_tiles_path}")

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

    # Same wet-tile population as long_df, via plot_sweep_budget_convergence's own
    # tile-convergence detector (full N_COMPLETE_ROUNDS ceiling, not just max_rounds).
    cdf_df = collect_convergence(sweep_budget_dir, epsilon)
    cdf_df = cdf_df[cdf_df["tile"].isin(tile_ids)]
    print(f"{len(cdf_df)} tile(s) in the CDF panel ({int(cdf_df['converged'].sum())} converged "
          f"within {N_COMPLETE_ROUNDS} rounds)")

    make_figure(long_df, cdf_df, epsilon, max_rounds, figures_dir / "sweep_calibration_bands.png")


if __name__ == "__main__":
    main()
