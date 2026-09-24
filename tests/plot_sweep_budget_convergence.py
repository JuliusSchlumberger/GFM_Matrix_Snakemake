"""Two figures from test_sweep_budget_calibration.py's per-tile sweep traces
(2026-09-24, 260-tile study):

  1. calibration_sweep_convergence_cdf.png - ECDF of how many ROUNDS (4
     sweeps each, production's real convergence-check unit - see the
     conversation this was built in for why sweeps != rounds) each tile
     needed before production's own round-level early-exit check
     (round_max_change <= waterlevel_epsilon_m) would have stopped it.
  2. calibration_sweep_convergence_vs_time.png - scatter of rounds-to-
     converge vs. wall-clock time to reach that round, point size/colour
     encoding tile size (n_cells, log scale) - does a tile needing more
     rounds also cost proportionally more time, or are there disagreements
     (e.g. a small tile that's slow to converge, or a huge tile that
     converges fast)?

Reconstructs round-level convergence from the continuous per-SWEEP trace
already collected (`sweep_max_change_raw` - `_dense_sweep`'s own return
value, the same raw per-cell update magnitude `solve_eikonal_dense`'s round
loop maxes over each 4-sweep group and compares to epsilon) - no second,
separate round-based solve needed. Each tile's 50 sweeps give 12 COMPLETE
rounds (sweeps 1-48); the trailing 2 sweeps (49-50) don't form a complete
round and are ignored for this purpose. A tile whose round_max_change never
drops to/below epsilon within those 12 rounds is right-censored ("not
converged within 12 rounds") - plotted at the ECDF's right edge, excluded
from the scatter (no well-defined convergence time).

Usage:
    python plot_sweep_budget_convergence.py <sweep_budget_dir> <figures_dir> [--epsilon 0.03]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEPS_PER_ROUND = 4
N_COMPLETE_ROUNDS = 100  # matches test_sweep_budget_calibration.py's own MAX_ROUNDS_CEILING
# (2026-09-24) - the true ceiling every tile's trace is capped at; the plots below use the
# ACTUAL observed max round among converged tiles for their axis range, not this ceiling
# directly, so a study where most tiles converge well under 100 rounds doesn't render as a
# mostly-empty 0-100 axis - see plot_cdf/plot_rounds_vs_time's own comments.


def _tile_convergence(csv_path: Path, epsilon: float) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None  # a tile the run is still writing right now (header not flushed yet) - not an error
    if df.empty or "sweep_max_change_raw" not in df.columns:
        return None
    if len(df) < SWEEPS_PER_ROUND:
        return None  # dry tile (1 row) or otherwise too short to form a full round

    n_cells = int(df["n_cells"].iloc[0])
    tile_id = int(df["tile"].iloc[0])

    max_change = df["sweep_max_change_raw"].to_numpy()
    cum_time = df["cum_time_s"].to_numpy()
    n_rounds_available = min(N_COMPLETE_ROUNDS, len(max_change) // SWEEPS_PER_ROUND)

    n_rounds_used = None
    time_to_converge_s = None
    for r in range(1, n_rounds_available + 1):
        round_max_change = max_change[(r - 1) * SWEEPS_PER_ROUND: r * SWEEPS_PER_ROUND].max()
        if round_max_change <= epsilon:
            n_rounds_used = r
            time_to_converge_s = float(cum_time[r * SWEEPS_PER_ROUND - 1])
            break

    return {
        "tile": tile_id,
        "n_cells": n_cells,
        "converged": n_rounds_used is not None,
        "n_rounds_used": n_rounds_used,
        "time_to_converge_s": time_to_converge_s,
        "n_rounds_available": n_rounds_available,
    }


def collect(sweep_budget_dir: Path, epsilon: float) -> pd.DataFrame:
    records = []
    csvs = sorted(sweep_budget_dir.glob("*.csv"))
    for p in csvs:
        rec = _tile_convergence(p, epsilon)
        if rec is not None:
            records.append(rec)
    return pd.DataFrame.from_records(records)


def plot_cdf(df: pd.DataFrame, epsilon: float, out_path: Path) -> None:
    converged = df[df["converged"]].sort_values("n_rounds_used")
    n_censored = int((~df["converged"]).sum())
    n_total = len(df)

    xs = converged["n_rounds_used"].to_numpy()
    ys = np.arange(1, len(xs) + 1) / n_total  # denominator = ALL tiles, so censored tiles show as the gap to 1.0

    # Axis range driven by the ACTUAL observed max round among converged tiles, not the
    # N_COMPLETE_ROUNDS=100 ceiling directly - a study where most tiles converge in, say,
    # 20 rounds shouldn't render as a mostly-empty 0-100 axis. The censored marker (if any)
    # is placed at a fixed, clearly-labelled offset past the real data, not literally at
    # round 101 (which would misleadingly suggest "just one round further" rather than
    # "never converged even by round 100").
    real_max = int(xs.max()) if len(xs) else 1
    censored_x = real_max + max(3, real_max // 5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(xs, ys, where="post", linewidth=2, color="#1f77b4")
    if n_censored:
        ax.hlines(ys[-1] if len(ys) else 0.0, xs[-1] if len(xs) else 0, censored_x,
                   color="#1f77b4", linewidth=2, linestyle="--")
        ax.scatter([censored_x], [ys[-1] if len(ys) else 0.0], marker="x", color="#d62728", s=80, zorder=5,
                   label=f"not converged within {N_COMPLETE_ROUNDS} rounds (n={n_censored})")
        ax.legend(loc="lower right")

    ax.set_xlabel("rounds to converge (4 sweeps/round, round_max_change <= epsilon)")
    ax.set_ylabel("cumulative fraction of tiles")
    ax.set_xlim(0, censored_x + 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Rounds to convergence, {n_total} tiles (epsilon={epsilon}m)\n"
        f"{n_total - n_censored} converged within {N_COMPLETE_ROUNDS} rounds, {n_censored} did not",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_rounds_vs_time(df: pd.DataFrame, epsilon: float, out_path: Path) -> None:
    converged = df[df["converged"]].copy()
    n_censored = int((~df["converged"]).sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = 20 + 60 * (np.log10(converged["n_cells"]) - np.log10(converged["n_cells"]).min()) / max(
        np.log10(converged["n_cells"]).max() - np.log10(converged["n_cells"]).min(), 1e-9
    )
    # jitter integer round counts slightly so overlapping tiles at the same
    # round are visually distinguishable rather than stacking exactly.
    rng = np.random.default_rng(0)
    x_jitter = converged["n_rounds_used"] + rng.uniform(-0.15, 0.15, size=len(converged))
    sc = ax.scatter(
        x_jitter, converged["time_to_converge_s"], c=np.log10(converged["n_cells"]),
        s=sizes, cmap="viridis", alpha=0.75, edgecolors="black", linewidths=0.3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("rounds to converge (jittered)")
    ax.set_ylabel("wall-clock time to reach that round (s, log scale)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(n_cells)")
    ax.set_title(
        f"Rounds to converge vs. time to converge, {len(converged)} converged tiles "
        f"(epsilon={epsilon}m, {n_censored} not-converged tile(s) excluded)",
        fontsize=12,
    )
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
    args = parser.parse_args()

    epsilon = args.epsilon
    if epsilon is None:
        cfg = load_config(args.config)
        epsilon = float(cfg["simulation"]["flooding"]["waterlevel_epsilon_m"])

    sweep_budget_dir = Path(args.sweep_budget_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = collect(sweep_budget_dir, epsilon)
    print(f"{len(df)} tile(s) with a usable sweep trace (epsilon={epsilon}m)")
    if df.empty:
        print("Nothing to plot.")
        return

    n_converged = int(df["converged"].sum())
    print(f"{n_converged} of {len(df)} converged within {N_COMPLETE_ROUNDS} rounds")
    df.to_csv(figures_dir / "sweep_convergence_summary.csv", index=False)
    print(f"Wrote {figures_dir / 'sweep_convergence_summary.csv'}")

    plot_cdf(df, epsilon, figures_dir / "calibration_sweep_convergence_cdf.png")
    plot_rounds_vs_time(df, epsilon, figures_dir / "calibration_sweep_convergence_vs_time.png")


if __name__ == "__main__":
    main()
