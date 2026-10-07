"""Global map of tiles EXCLUDED from a friction-sweep calibration study
because their SFINCS model was never successfully built - colored by why
(tile_status.py's own categories: no_station, no_boundary_cells,
antimeridian, other_error, or "not_yet_attempted" for a tile with no
tile_status.json yet and no SFINCS model either).

Reads calibration_tile_status.csv (report_calibration_tile_status.py's own
output - rerun that first for a fresh picture) joined against the tile
grid's own real geometry for centroids (NOT tile_selection_metadata.csv,
which only covers tiles that reached postprocessing - a tile excluded
before ever getting that far wouldn't be in it). White ocean background,
same convention as every other world map in this repo (a plain background
is not a real raster mask - see plot_delta_flood_map.py's own module
docstring for the full reasoning; this was also just fixed in
plot_validation_results.py/select_validation_tiles.py, which had the same
blue-background bug).

Usage:
    python plot_excluded_tiles_map.py --base-dir-name sfincs_calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import LAND_COLOR, draw_caption_box  # noqa: E402

COAST_COLOR = "#b8b8b3"

CATEGORY_COLORS = {
    # Okabe-Ito colorblind-safe set, picked to keep no_boundary_cells and
    # other_error clearly apart (an earlier orange-red/red pairing here was
    # nearly indistinguishable at marker size - 2026-10-07 fix).
    "no_station": "#999999",       # grey - neutral: a data-availability exclusion, not a failure
    "no_boundary_cells": "#0072B2",  # blue
    "antimeridian": "#CC79A7",       # reddish-purple
    "other_error": "#D55E00",        # vermillion
    "not_yet_attempted": "#009E73",  # green
}
CATEGORY_LABELS = {
    "no_station": "no COAST-RP station",
    "no_boundary_cells": "0 waterlevel-boundary cells",
    "antimeridian": "antimeridian-crossing",
    "other_error": "other error",
    "not_yet_attempted": "not yet attempted",
}


def _category(row: pd.Series) -> str:
    if not row["has_station"]:
        return "no_station"
    if pd.isna(row["last_status"]):
        return "not_yet_attempted"
    status = row["last_status"]
    return status if status in CATEGORY_COLORS else "other_error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--status-csv", default=None, help="default: {base-dir-name}/calibration_tile_status.csv")
    parser.add_argument("--tile-grid", default=None, help="default: {base-dir-name}/{base-dir-name}_grid.gpkg")
    parser.add_argument("--out", default=None, help="default: {base-dir-name}/figures/excluded_tiles_map.png")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    status_csv = Path(args.status_csv) if args.status_csv else base_dir / "calibration_tile_status.csv"
    tile_grid_path = Path(args.tile_grid) if args.tile_grid else base_dir / f"{args.base_dir_name}_grid.gpkg"
    out_path = Path(args.out) if args.out else base_dir / "figures" / "excluded_tiles_map.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status_df = pd.read_csv(status_csv)
    status_df["tile_id"] = status_df["tile_id"].astype(str)
    print(f"{len(status_df)} tile(s) from {status_csv}")

    grid = gpd.read_file(tile_grid_path)
    grid["tile_id"] = grid["tile_id"].astype(str)
    centroids = grid.geometry.centroid
    grid = grid.assign(lon=centroids.x, lat=centroids.y)[["tile_id", "lon", "lat"]]

    df = status_df.merge(grid, on="tile_id", how="left")
    excluded = df[~df["sfincs_ready"]].copy()
    excluded["category"] = excluded.apply(_category, axis=1)
    print(f"{len(excluded)}/{len(df)} tile(s) excluded (SFINCS not built):")
    print(excluded["category"].value_counts().to_string())

    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)

    for category, color in CATEGORY_COLORS.items():
        sub = excluded[excluded["category"] == category]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"], transform=ccrs.PlateCarree(),
            c=color, s=26, alpha=0.9, linewidths=0.4, edgecolors="white", zorder=3,
            label=f"{CATEGORY_LABELS[category]} (n={len(sub)})",
        )

    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    draw_caption_box(ax, [
        f"n={len(excluded)} tile(s) excluded of {len(df)} total",
        f"{status_csv.name}",
    ])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
