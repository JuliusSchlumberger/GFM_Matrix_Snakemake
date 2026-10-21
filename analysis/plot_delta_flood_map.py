"""Map river deltas colored by % of modeled area flooded, one figure per ssp_scenario.

One panel per waterlevel_name (SLR level) from compute_delta_flood_extent.py's CSV; each delta is
a point at its centroid, coloured by pct_flooded_of_modeled_area (flood hazard only, no protection).
Deltas without tile coverage (n_tiles == 0) are left out. Ocean is left white; water color is
reserved for raster masks.

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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from map_style import LAND_COLOR  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
COAST_COLOR = "#b8b8b3"


def _slr_label(waterlevel_name: str) -> str:
    mm = int(waterlevel_name.split("_")[1])
    return f"SLR +{mm / 1000:.1f} m"


def plot_delta_flood_map(df: pd.DataFrame, ssp_scenario: str, out_path: Path) -> None:
    """Plot one ssp_scenario's delta points, one EqualEarth panel per waterlevel_name."""
    sub_ssp = df[df["ssp_scenario"] == ssp_scenario]
    waterlevel_names = sorted(sub_ssp["waterlevel_name"].unique(), key=lambda s: int(s.split("_")[1]))

    ncols = math.ceil(math.sqrt(len(waterlevel_names)))
    nrows = math.ceil(len(waterlevel_names) / ncols)

    proj = ccrs.EqualEarth()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 2.85 * nrows),
        subplot_kw={"projection": proj}, facecolor="white",
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.12)
    axes = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]

    sc = None
    for ax, waterlevel_name, letter in zip(axes, waterlevel_names, string.ascii_lowercase):
        sub = sub_ssp[(sub_ssp["waterlevel_name"] == waterlevel_name) & (sub_ssp["n_tiles"] > 0)]
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
        sc = ax.scatter(
            sub["centroid_lon"], sub["centroid_lat"], transform=ccrs.PlateCarree(),
            c=sub["pct_flooded_of_modeled_area"], cmap="YlOrRd", vmin=0, vmax=100,
            s=30, alpha=0.9, linewidths=0.4, edgecolors="white", zorder=3,
        )
        ax.spines["geo"].set_edgecolor(COAST_COLOR)
        ax.spines["geo"].set_linewidth(0.6)
        ax.text(0.0, 1.06, f"({letter}) {_slr_label(waterlevel_name)}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10, fontweight="bold")

    for ax in axes[len(waterlevel_names):]:
        ax.axis("off")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label("% of delta's modeled area flooded")
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
