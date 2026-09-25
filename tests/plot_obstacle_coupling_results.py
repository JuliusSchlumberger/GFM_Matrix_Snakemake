"""Result visualizations for the 260-tile obstacle-coupling calibration study
(2026-09-25) - mirrors plot_sweep_budget_convergence.py's own CDF/scatter
pattern, adapted to the outer-loop trace's own columns.

Reads every <tile_id>.csv in the obstacle_coupling output dir. Each tile's
CSV is: one baseline row (n_outer=0, completely unblocked), one row per
outer iteration actually run (n_outer=1..N, N capped at --max-outer), and
one status="summary" row (pct_removed_by_static_filter_alone,
pct_additional_removed_by_full_outer_loop, n_outer_iterations_to_convergence).

Produces:
  - obstacle_coupling_outer_cdf.png: ECDF of outer iterations to convergence
    (a tile that never satisfies outer_convergence_pct within max_outer
    iterations is right-censored, same convention as the sweep-budget CDF).
  - obstacle_coupling_static_vs_loop.png: paired comparison of
    pct_removed_by_static_filter_alone vs. pct_additional_removed_by_full_
    outer_loop - the production-relevant question this whole study exists
    to answer (is the free static pre-filter already most of the benefit?).
  - obstacle_coupling_outer_vs_time.png: outer iterations to converge vs.
    wall-clock time, point size/colour = tile size (log n_cells) - same
    treatment as the sweep-budget study's own rounds-vs-time scatter.

Usage:
    python plot_obstacle_coupling_results.py <obstacle_coupling_dir> <figures_dir>
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


def collect(obstacle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary_df, baseline_df) - one row per tile each.

    The status="summary" row only ever has 4 real fields set (tile, status,
    pct_removed_by_static_filter_alone, pct_additional_removed_by_full_
    outer_loop, n_outer_iterations_to_convergence) - build_summary_row()
    initializes every OTHER column to "" (see test_obstacle_coupling_
    calibration.py), so cum_time_s/n_cells/etc. are blank there, not real
    values. This merges in the actual FINAL outer-iteration row (the one
    whose own n_outer equals n_outer_iterations_to_convergence) for those -
    a real, populated row - so cum_time_s/n_cells etc. are usable downstream.
    """
    summary_records = []
    baseline_records = []
    for p in sorted(obstacle_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        summary_rows = df[df["status"] == "summary"]
        baseline_rows = df[df["n_outer"] == 0]
        if summary_rows.empty or baseline_rows.empty:
            continue
        summary_row = summary_rows.iloc[0].to_dict()

        final_n_outer = summary_row["n_outer_iterations_to_convergence"]
        final_rows = df[(df["status"] == "ok") & (df["n_outer"] == final_n_outer)]
        if final_rows.empty:
            continue
        final_row = final_rows.iloc[0]
        for col in ("cum_time_s", "n_cells", "n_inundated"):
            summary_row[col] = final_row[col]

        summary_records.append(summary_row)
        baseline_records.append(baseline_rows.iloc[0].to_dict())
    return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(baseline_records)


def plot_outer_cdf(summary: pd.DataFrame, max_outer: int, out_path: Path) -> None:
    """Bar chart, not an ECDF (2026-09-25, found live): the real distribution
    of n_outer_iterations_to_convergence turned out almost entirely
    2-valued (166/260 tiles converge at EXACTLY outer=2, only 4 more spread
    across 4/6) plus the max_outer=15 censored bucket - an ECDF step plot
    over only 3-4 distinct x values is visually misleading (looks like a
    smooth-ish curve when it's really two spikes), a bar chart shows the
    real, near-bimodal shape honestly."""
    n_iter = summary["n_outer_iterations_to_convergence"].to_numpy()
    n_total = len(n_iter)
    censored = n_iter >= max_outer
    n_censored = int(censored.sum())

    converged = n_iter[~censored].astype(int)
    values, counts = np.unique(converged, return_counts=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(values, counts, color="#1f77b4", width=0.8, label="converged")
    if n_censored:
        censored_x = (values.max() if len(values) else 0) + 2
        ax.bar([censored_x], [n_censored], color="#d62728", width=0.8,
               label=f"hit max_outer={max_outer} without converging")
        ax.set_xticks(list(values) + [censored_x])
        ax.set_xticklabels([str(v) for v in values] + [f">={max_outer}"])
    for x, c in zip(values, counts):
        ax.text(x, c + n_total * 0.01, str(c), ha="center", fontsize=9)
    if n_censored:
        ax.text(censored_x, n_censored + n_total * 0.01, str(n_censored), ha="center", fontsize=9)

    ax.set_xlabel("outer iterations to converge (pct_newly_blocked < outer_convergence_pct)")
    ax.set_ylabel("number of tiles")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"Outer-loop convergence, {n_total} tiles\n{n_total - n_censored} converged within {max_outer} outer iterations, {n_censored} did not", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_static_vs_loop(summary: pd.DataFrame, out_path: Path) -> None:
    static_pct = summary["pct_removed_by_static_filter_alone"].to_numpy()
    loop_pct = summary["pct_additional_removed_by_full_outer_loop"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    bins = np.linspace(0, max(np.nanmax(static_pct), np.nanmax(loop_pct), 1), 30)
    ax.hist(static_pct, bins=bins, alpha=0.65, label=f"static filter alone (median={np.nanmedian(static_pct):.2f}%)", color="#1f77b4")
    ax.hist(loop_pct, bins=bins, alpha=0.65, label=f"+ full outer loop (median={np.nanmedian(loop_pct):.2f}%)", color="#d62728")
    ax.set_xlabel("% of baseline flooded cells removed")
    ax.set_ylabel("number of tiles")
    ax.legend(fontsize=9)
    ax.set_title("Static pre-filter alone vs. additional benefit\nof the full iterative outer loop")

    ax = axes[1]
    ax.scatter(static_pct, loop_pct, s=30, alpha=0.6, edgecolors="black", linewidths=0.3)
    lim = max(np.nanmax(static_pct), np.nanmax(loop_pct), 1) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("% removed by static filter alone")
    ax.set_ylabel("% additionally removed by full outer loop")
    ax.set_title("Per-tile: static-filter benefit vs.\nadditional outer-loop benefit")

    fig.suptitle(f"Static pre-filter vs. iterative outer loop, {len(summary)} tiles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_outer_vs_time(summary: pd.DataFrame, max_outer: int, out_path: Path) -> None:
    n_iter = summary["n_outer_iterations_to_convergence"].to_numpy()
    converged_mask = n_iter < max_outer
    conv = summary[converged_mask].copy()
    n_censored = int((~converged_mask).sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    log_cells = np.log10(conv["n_cells"])
    sizes = 20 + 60 * (log_cells - log_cells.min()) / max(log_cells.max() - log_cells.min(), 1e-9)
    rng = np.random.default_rng(0)
    x_jitter = conv["n_outer_iterations_to_convergence"] + rng.uniform(-0.15, 0.15, size=len(conv))
    sc = ax.scatter(x_jitter, conv["cum_time_s"], c=log_cells, s=sizes, cmap="viridis", alpha=0.75, edgecolors="black", linewidths=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("outer iterations to converge (jittered)")
    ax.set_ylabel("wall-clock time to reach convergence (s, log scale)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(n_cells)")
    ax.set_title(f"Outer iterations vs. time to converge, {len(conv)} converged tiles ({n_censored} not-converged excluded)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("obstacle_dir")
    parser.add_argument("figures_dir")
    parser.add_argument("--max-outer", type=int, default=15)
    args = parser.parse_args()

    obstacle_dir = Path(args.obstacle_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary, baseline = collect(obstacle_dir)
    print(f"{len(summary)} tile(s) with a usable summary row")
    if summary.empty:
        print("Nothing to plot.")
        return
    summary.to_csv(figures_dir / "obstacle_coupling_summary.csv", index=False)
    print(f"Wrote {figures_dir / 'obstacle_coupling_summary.csv'}")

    n_converged = int((summary["n_outer_iterations_to_convergence"] < args.max_outer).sum())
    print(f"{n_converged} of {len(summary)} converged within {args.max_outer} outer iterations")
    print(f"pct_removed_by_static_filter_alone: median={summary['pct_removed_by_static_filter_alone'].median():.2f}%")
    print(f"pct_additional_removed_by_full_outer_loop: median={summary['pct_additional_removed_by_full_outer_loop'].median():.2f}%")

    plot_outer_cdf(summary, args.max_outer, figures_dir / "obstacle_coupling_outer_cdf.png")
    plot_static_vs_loop(summary, figures_dir / "obstacle_coupling_static_vs_loop.png")
    plot_outer_vs_time(summary, args.max_outer, figures_dir / "obstacle_coupling_outer_vs_time.png")


if __name__ == "__main__":
    main()
