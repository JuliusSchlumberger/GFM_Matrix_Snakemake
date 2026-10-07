"""Temporal development of % flooded area across all deltas, one subplot per
ssp_scenario - from `compute_delta_flood_extent.py`'s own output CSV, combined
with the real AR6 median SLR trajectory (`visualization.slr_trajectories_csv`).

Layout follows `analysis/plot_timeseries.py`'s own established convention
(`1 x n_ssp` subplots, per-panel SSP-colored title, one shared legend below
all panels) - see that script for the precedent this adapts. `sharex=True`
and `sharey=True`: since the flood simulation forces a fixed global-mean SLR
magnitude regardless of ssp_scenario, `pct_flooded_of_modeled_area` for a
given slr_scenario is identical across ssp panels - the ONLY ssp-dependent
quantity is which year that magnitude lands on (via each ssp's own AR6
trajectory). With independent (unshared) x-axes, each panel autoscales to
its own year range and hides that real difference in rate; sharing the axis
makes it visible instead.

Double-interpolation, no extrapolation needed:
  1. year -> SLR magnitude (mm), via `np.interp` on the real AR6 median
     trajectory's own decadal points (2020-2150) - never extrapolated,
     since every year we plot (<=2150) is within the trajectory's own
     range.
  2. SLR magnitude -> pct_flooded, via `np.interp` on each delta's own 5
     REAL simulated flood-extent points (SLR_0/500/1000/1500/2000 mm,
     already in the CSV, identical across ssp_scenario - see above). Valid
     without extrapolation too: under the median trajectory neither ssp
     exceeds ~870mm by 2150, well inside the simulated 0-2000mm range.
Chaining these two real interpolations gives a smooth, fully-real
pct_flooded(year) curve for every delta out to 2150 - an earlier version of
this script instead linearly extrapolated pct-vs-year from just the
SLR_0->SLR_500 slope once year_slr_reached_median ran out (NaN beyond
SLR_500 - see add_slr_year_column.py's own docstring), which is strictly
worse: it ignores the real SLR_1000/1500 simulated points and assumes a
straight-line pct/SLR relationship instead of actually interpolating it.

Colour choice: with 48 deltas and no existing per-delta palette anywhere in
this codebase, drawing 48 distinct hues would be the exact anti-pattern
`map_style.py`'s own categorical-palette convention warns against (a
palette this big always degenerates to indistinguishable colours). Instead
every delta's own line is drawn thin and semi-transparent in the panel's
own SSP color (a standard "spaghetti plot" - the overlapping density itself
shows the spread), with a single bold median-across-deltas line on top.

Usage:
    python plot_delta_affected_fraction_timeseries.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
SSP_COLORS_DEFAULT = {"SSP1": "#1f77b4", "SSP2": "#2ca02c", "SSP5": "#d62728"}
SSP_RCP_CODES_DEFAULT = {"SSP1": 126, "SSP2": 245, "SSP5": 585}
TRAJECTORY_START_YEAR = 2020
TRAJECTORY_END_YEAR = 2150  # last year in slr_trajectories_global_median.csv


def _ssp_label(ssp_scenario: str, ssp_rcp_codes: dict) -> str | None:
    code = int(ssp_scenario.replace("ssp", ""))
    matches = [label for label, c in ssp_rcp_codes.items() if c == code]
    return matches[0] if matches else None


def _ssp_color(ssp_scenario: str, ssp_colors: dict, ssp_rcp_codes: dict) -> str:
    label = _ssp_label(ssp_scenario, ssp_rcp_codes)
    return ssp_colors.get(label, "#444444") if label else "#444444"


def _rcp_suffix(ssp_scenario: str, ssp_rcp_codes: dict) -> str:
    code = int(ssp_scenario.replace("ssp", ""))
    return f"-RCP{(code % 100) / 10.0:.1f}"


def plot_affected_fraction_timeseries(
    df: pd.DataFrame, traj_df: pd.DataFrame, ssp_colors: dict, ssp_rcp_codes: dict, out_path: Path,
) -> None:
    ssp_scenarios = sorted(df["ssp_scenario"].unique())
    year_grid = np.linspace(TRAJECTORY_START_YEAR, TRAJECTORY_END_YEAR, 261)
    traj_years = traj_df.index.to_numpy(dtype=float)

    fig, axes = plt.subplots(1, len(ssp_scenarios), figsize=(6 * len(ssp_scenarios), 5), sharey=True, sharex=True)
    if len(ssp_scenarios) == 1:
        axes = [axes]

    for ax, ssp in zip(axes, ssp_scenarios):
        color = _ssp_color(ssp, ssp_colors, ssp_rcp_codes)
        label = _ssp_label(ssp, ssp_rcp_codes)
        traj_mm = traj_df[f"{label}_p50"].to_numpy(dtype=float)
        slr_mm_of_year = np.interp(year_grid, traj_years, traj_mm)

        sub = df[df["ssp_scenario"] == ssp].copy()
        sub["slr_mm"] = sub["slr_scenario"].str.split("_").str[1].astype(float)

        pct_curves = []
        for delta_id, grp in sub.groupby("delta_id"):
            grp = grp.sort_values("slr_mm")
            pct_of_year = np.interp(slr_mm_of_year, grp["slr_mm"].to_numpy(), grp["pct_flooded_of_modeled_area"].to_numpy())
            ax.plot(year_grid, pct_of_year, color=color, alpha=0.18, linewidth=1.0, zorder=2)
            pct_curves.append(pct_of_year)

        median_pct = np.median(np.vstack(pct_curves), axis=0)
        ax.plot(year_grid, median_pct, color=color, alpha=1.0, linewidth=2.5, zorder=3,
                 label=f"median across deltas (n={len(pct_curves)})")

        ax.set_title(f"{ssp}{_rcp_suffix(ssp, ssp_rcp_codes)}", color=color, fontsize=11)
        ax.set_xlabel("Year (median AR6 SLR trajectory)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_xlim(TRAJECTORY_START_YEAR, TRAJECTORY_END_YEAR)
    axes[0].set_ylabel("% of delta's modeled area flooded", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=9, bbox_to_anchor=(0.5, 0.02), ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--csv", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--slr-trajectories-csv", default=None, help="default: visualization.slr_trajectories_csv from config")
    parser.add_argument("--out", default=None, help="default: {root}/deltas_floodmaps/figures/delta_affected_fraction_timeseries.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    viz = cfg.get("visualization", {})
    csv_path = Path(args.csv) if args.csv else root / "deltas_floodmaps" / "delta_flood_extent.csv"
    traj_path = Path(args.slr_trajectories_csv) if args.slr_trajectories_csv else Path(
        viz.get("slr_trajectories_csv", root / "processed_inputs" / "slr_trajectories_global_median.csv")
    )
    out_path = Path(args.out) if args.out else root / "deltas_floodmaps" / "figures" / "delta_affected_fraction_timeseries.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    traj_df = pd.read_csv(traj_path, index_col="year")
    print(f"{len(df)} row(s) from {csv_path}")
    print(f"SLR trajectory data: {traj_path} (years {traj_df.index.min()}-{traj_df.index.max()})")

    ssp_colors = viz.get("ssp_colors", SSP_COLORS_DEFAULT)
    ssp_rcp_codes = viz.get("ssp_rcp_codes", SSP_RCP_CODES_DEFAULT)

    plot_affected_fraction_timeseries(df, traj_df, ssp_colors, ssp_rcp_codes, out_path)


if __name__ == "__main__":
    main()
