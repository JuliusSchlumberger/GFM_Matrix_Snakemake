"""Single 2-panel summary figure for the obstacle-coupling calibration
study (see calibration_studies/test_obstacle_coupling_calibration.py and
docs/methods_03_calibration_sensitivity.md §3): each panel is a
median/2nd/98th-percentile/min-max band across the wet calibration tiles
that still have a real observation at that outer iteration, plotted
against outer iteration, with a vertical line at `max_outer_iterations`
marking the production cap that was actually chosen.

Reads test_obstacle_coupling_calibration.py's per-tile raw CSVs directly:
the n_outer=0 baseline row and one row per outer iteration actually run
(status="ok"). A tile's own trace ends once outer_convergence_pct is
satisfied, so the number of tiles contributing to a given outer
iteration's band shrinks at higher iteration numbers as tiles finish -
real (raw, unmodified) data throughout, no imputation/carry-forward for
iterations past a tile's own trace.

Panels:
  (a) converged-vs-still-active tile counts per outer iteration: a stacked
      bar per iteration splitting all tiles into those that already
      satisfied outer_convergence_pct and stopped by that iteration vs
      those still actively iterating past it - the discrete-count version
      of the shrinking sample size behind every band in this figure.
  (b) pct_newly_blocked (the literal outer-loop stopping-criterion metric -
      % of the tile's cells newly blocked this iteration) vs outer
      iteration, log scale, with a horizontal reference line at
      outer_convergence_pct - shows how tiles approach/cross the actual
      cutoff, not just an absolute cell count.

Usage:
    python plot_obstacle_coupling_calibration_bands.py <obstacle_coupling_dir> <figures_dir> [--max-outer-iterations 3]
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
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
FLOOR_PCT = 1e-4  # log-scale display floor for panel (b)'s pct_newly_blocked


def _load_tile_trace(csv_path: Path) -> pd.DataFrame | None:
    """Per-iteration rows (n_outer >= 1) for one tile: tile id, outer
    iteration, and pct_newly_blocked (the literal outer-loop stopping-
    criterion metric). A tile's own baseline (n_outer=0, no blocking at
    all) must show some flooding for the tile to be included - matches
    every other figure drawn from this same calibration-tile pool."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    baseline_rows = df[df["n_outer"] == 0]
    if baseline_rows.empty:
        return None
    tile_id = int(baseline_rows.iloc[0]["tile"])
    baseline_n_inundated = float(baseline_rows.iloc[0]["n_inundated"])
    if baseline_n_inundated <= 0:
        return None

    iters = df[(df["status"] == "ok") & (df["n_outer"] >= 1)].copy()
    if iters.empty:
        return None
    iters["tile"] = tile_id
    return iters[["tile", "n_outer", "pct_newly_blocked"]]


def _band_stats(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = long_df.groupby("n_outer")[value_col]
    return pd.DataFrame({
        "median": g.median(), "p02": g.quantile(0.02), "p98": g.quantile(0.98),
        "min": g.min(), "max": g.max(), "n": g.count(),
    })


def _plot_band(ax, s: pd.DataFrame, color: str, label: str, floor: float | None = None) -> None:
    lo, hi, med, p98, p02 = s["min"], s["max"], s["median"], s["p98"], s["p02"]
    if floor is not None:
        lo, hi, med, p98, p02 = (x.clip(lower=floor) for x in (lo, hi, med, p98, p02))
    ax.fill_between(s.index, lo, hi, alpha=0.15, color=color, label="full range (min-max) across tiles")
    ax.plot(s.index, med, color=color, linewidth=2, label=f"{label} - median")
    ax.plot(s.index, p98, color=color, linewidth=1.3, linestyle="--", label=f"{label} - 98th percentile")
    ax.plot(s.index, p02, color=color, linewidth=1.3, linestyle="-.", label=f"{label} - 2nd percentile")


def _plot_convergence_bars(ax, final_iter: pd.Series, x_max: int) -> None:
    """Stacked bar per outer iteration x: tiles already converged and
    stopped by x (final_iter <= x) vs tiles still actively iterating past x
    (final_iter > x) - the discrete-count version of the shrinking sample
    size behind every band in this figure."""
    n_tiles = len(final_iter)
    xs = np.arange(1, x_max + 1)
    converged = np.array([(final_iter <= x).sum() for x in xs])
    still_active = n_tiles - converged
    ax.bar(xs, converged, color="#2ca02c", label="converged (stopped by this iteration)")
    ax.bar(xs, still_active, bottom=converged, color="#d62728", alpha=0.7,
           hatch="///", edgecolor="white", linewidth=0.5, label="still actively iterating")
    ax.set_ylabel(f"tile count (n={n_tiles})")


def make_figure(long_df: pd.DataFrame, max_outer_iterations: int, outer_convergence_pct: float,
                 x_max: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    ax = axes[0]
    final_iter = long_df.groupby("tile")["n_outer"].max()
    _plot_convergence_bars(ax, final_iter, x_max)
    ax.axvline(max_outer_iterations, color="black", linestyle="--", linewidth=1.3,
               label=f"max_outer_iterations={max_outer_iterations}")
    ax.set_xlabel("outer iteration")
    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=13)
    ax.set_xlim(0.5, x_max + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="center left")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    _plot_band(ax, _band_stats(long_df, "pct_newly_blocked"), "#ff7f0e", "newly blocked (% of tile)", floor=FLOOR_PCT)
    ax.axhline(outer_convergence_pct, color="black", linestyle=":", linewidth=1.3,
               label=f"outer_convergence_pct={outer_convergence_pct}")
    ax.axvline(max_outer_iterations, color="black", linestyle="--", linewidth=1.3,
               label=f"max_outer_iterations={max_outer_iterations}")
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("newly-blocked increment proposed for the next iteration\n(% of tile's cells, log)")
    ax.set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    ax.set_xlim(1, x_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("obstacle_dir")
    parser.add_argument("figures_dir")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--max-outer-iterations", type=int, default=None,
                         help="default: config simulation.flooding.obstacle_coupling.max_outer_iterations")
    parser.add_argument("--x-max", type=int, default=6,
                         help="x-axis upper limit (outer iteration) - independent of the max outer "
                              "iteration actually observed in the data, for consistent axis framing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    obstacle_cfg = cfg["simulation"]["flooding"]["obstacle_coupling"]
    max_outer_iterations = args.max_outer_iterations
    if max_outer_iterations is None:
        max_outer_iterations = int(obstacle_cfg["max_outer_iterations"])
    outer_convergence_pct = float(obstacle_cfg["outer_convergence_pct"])

    obstacle_dir = Path(args.obstacle_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted(obstacle_dir.glob("*.csv")):
        iters = _load_tile_trace(p)
        if iters is not None:
            records.append(iters)
    long_df = pd.concat(records, ignore_index=True)
    n_tiles = long_df["tile"].nunique()
    print(f"{n_tiles} tile(s) with a usable outer-iteration trace, {len(long_df)} tile-iteration row(s)")

    make_figure(long_df, max_outer_iterations, outer_convergence_pct, args.x_max,
                figures_dir / "obstacle_coupling_calibration_bands.png")


if __name__ == "__main__":
    main()
