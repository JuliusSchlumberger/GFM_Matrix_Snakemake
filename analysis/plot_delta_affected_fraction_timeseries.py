"""Plot % flooded delta area over time, one panel per ssp_scenario.

pct_flooded(year) chains two interpolations without extrapolation: year -> SLR (mm)
on the AR6 median trajectory (2020-2150), then SLR -> pct_flooded on each delta's
simulated SLR_0..2000 points (medians stay below ~870 mm by 2150). pct_flooded per SLR
is SSP-independent; only the year differs, so x and y axes are shared to show the rate
difference. 48 deltas: thin per-delta lines plus a bold median, not 48 hues.

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
TRAJECTORY_START_YEAR = 2020
TRAJECTORY_END_YEAR = 2150  # last year of the AR6 trajectory CSV


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
    """Plot per-delta pct_flooded(year) curves plus the cross-delta median, per SSP panel."""
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
        sub["slr_mm"] = sub["waterlevel_name"].str.split("_").str[1].astype(float)

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
    viz = cfg["visualization"]
    csv_path = Path(args.csv) if args.csv else root / "deltas_floodmaps" / "delta_flood_extent.csv"
    traj_path = Path(args.slr_trajectories_csv) if args.slr_trajectories_csv else Path(viz["slr_trajectories_csv"])
    out_path = Path(args.out) if args.out else root / "deltas_floodmaps" / "figures" / "delta_affected_fraction_timeseries.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "waterlevel_name" not in df.columns:  # older CSVs name the SLR level column slr_scenario
        df = df.rename(columns={"slr_scenario": "waterlevel_name"})
    traj_df = pd.read_csv(traj_path, index_col="year")
    print(f"{len(df)} row(s) from {csv_path}")
    print(f"SLR trajectory data: {traj_path} (years {traj_df.index.min()}-{traj_df.index.max()})")

    ssp_colors = viz["ssp_colors"]
    ssp_rcp_codes = viz["ssp_rcp_codes"]

    plot_affected_fraction_timeseries(df, traj_df, ssp_colors, ssp_rcp_codes, out_path)


if __name__ == "__main__":
    main()
