"""Map river deltas colored by % of modeled area flooded, one figure per ssp_scenario.

One panel per slr_scenario from compute_delta_flood_extent.py's CSV. Ocean is left
white; water color is reserved for raster masks.

Usage:
    python plot_delta_flood_map.py
    python plot_delta_flood_map.py --csv path/to/delta_flood_extent.csv
"""

from __future__ import annotations

import argparse
import math
import string
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from map_style import LAND_COLOR  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
COAST_COLOR = "#b8b8b3"
PROTECTED_COLOR = "#8c8c8c"  # FLOPROS protection >= 100 yr, not failed: all land dry
FAILED_EDGE = "#222222"  # a >= 100 yr protection class fails at this SLR


def _slr_label(slr_scenario: str) -> str:
    mm = int(slr_scenario.split("_")[1])
    return f"SLR +{mm / 1000:.1f} m"


def plot_delta_flood_map(df: pd.DataFrame, ssp_scenario: str, out_path: Path) -> None:
    """Plot one ssp_scenario's delta points, one EqualEarth panel per slr_scenario."""
    sub_ssp = df[df["ssp_scenario"] == ssp_scenario]
    slr_scenarios = sorted(sub_ssp["slr_scenario"].unique(), key=lambda s: int(s.split("_")[1]))

    ncols = math.ceil(math.sqrt(len(slr_scenarios)))
    nrows = math.ceil(len(slr_scenarios) / ncols)

    proj = ccrs.EqualEarth()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 2.85 * nrows),
        subplot_kw={"projection": proj}, facecolor="white",
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.12)
    axes = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]

    sc = None
    for ax, slr_scenario, letter in zip(axes, slr_scenarios, string.ascii_lowercase):
        sub = sub_ssp[(sub_ssp["slr_scenario"] == slr_scenario) & (sub_ssp["n_tiles"] > 0)]
        status = sub["protection_status"] if "protection_status" in sub.columns else pd.Series("", index=sub.index)
        prot = sub[status == "protected"]
        sub = sub[status != "protected"]
        failed = (status[sub.index] == "protection_failed").to_numpy()
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
        ax.scatter(prot["centroid_lon"], prot["centroid_lat"], transform=ccrs.PlateCarree(),
                   color=PROTECTED_COLOR, s=30, alpha=0.9, linewidths=0.4, edgecolors="white", zorder=3)
        sc = ax.scatter(
            sub["centroid_lon"], sub["centroid_lat"], transform=ccrs.PlateCarree(),
            c=sub["pct_flooded_of_modeled_area"], cmap="YlOrRd", vmin=0, vmax=100,
            s=30, alpha=0.9, linewidths=np.where(failed, 1.0, 0.4),
            edgecolors=[FAILED_EDGE if f else "white" for f in failed], zorder=3,
        )
        ax.spines["geo"].set_edgecolor(COAST_COLOR)
        ax.spines["geo"].set_linewidth(0.6)
        ax.text(0.0, 1.06, f"({letter}) {_slr_label(slr_scenario)}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10, fontweight="bold")

    for ax in axes[len(slr_scenarios):]:
        ax.axis("off")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label("% of delta's modeled area flooded")
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=PROTECTED_COLOR, mec="white",
                      label="protected (>= 100 yr)"),
               Line2D([], [], ls="", marker="o", ms=6, color="#fd8d3c", mec=FAILED_EDGE, mew=1.0,
                      label="protection failed")]
    if not (sub_ssp.get("protection_status", pd.Series(dtype=str)) == "protection_failed").any():
        handles = handles[:1]
    if "protection_status" in sub_ssp.columns:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "deltas_ssp126_materialized.yml"))
    parser.add_argument("--csv", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--out-dir", default=None, help="default: {root}/deltas_floodmaps/figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    csv_path = Path(args.csv) if args.csv else root / "deltas_floodmaps" / "delta_flood_extent.csv"
    out_dir = Path(args.out_dir) if args.out_dir else root / "deltas_floodmaps" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"{len(df)} row(s) from {csv_path}")

    for ssp_scenario in sorted(df["ssp_scenario"].unique()):
        out_path = out_dir / f"delta_flood_pct_{ssp_scenario}.png"
        plot_delta_flood_map(df, ssp_scenario, out_path)


if __name__ == "__main__":
    main()
