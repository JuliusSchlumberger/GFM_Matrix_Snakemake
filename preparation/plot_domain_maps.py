"""Global Equal Earth map(s) of the connectivity-first tiling pipeline's
own domain set (preparation/build_tile_manifest.py / src/connectivity_tiling.py):
the FINAL domains ("domain_final" figure) and, when available, the domains
DROPPED for having no connectivity path back to open water within their
own component (see docs/methods_01_tile_processing_and_waterlevels.md
section 3's own description of this drop rule).

plot_final_domains_map() reads straight from tile_grid.path
(domain_tiles_global.gpkg) - the pipeline's own authoritative, always-
current output. plot_dropped_unreachable_map() has no equivalent
standalone source (dropped domains aren't part of the final grid), so it
only runs from build_tile_manifest.py's own run() below, where they're
still in memory.

Folded into the tiling pipeline itself (2026-10-07): build_tile_manifest.py's
own run() now calls both plot functions directly at the end, using
already-in-memory GeoDataFrames - no extra disk read, always in sync with
whatever tile_grid.path that run just wrote.

Usage (standalone - e.g. to refresh domains_final.png without re-running
tile generation):
    python plot_domain_maps.py
    python plot_domain_maps.py --tile-grid path/to/domain_tiles_global.gpkg --out-dir DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAND_COLOR = "#d8d8d4"
COAST_COLOR = "#b8b8b3"
DOMAIN_COLOR = "#2a78d6"
DROPPED_COLOR = "#d6572a"

DEFAULT_TILE_GRID = Path(
    r"P:\11212688-004-global-floodmaps\modelling\processed_inputs\mask\domain_tiles_global.gpkg"
)
DEFAULT_OUT_DIR = Path(r"P:\11212688-004-global-floodmaps\modelling\processed_inputs\mask")


def plot_domains_map(gdf: gpd.GeoDataFrame, out_path: Path, color: str, label: str) -> None:
    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    ax.add_geometries(
        gdf.geometry, crs=ccrs.PlateCarree(),
        facecolor=color, edgecolor=color, linewidth=0.2, alpha=0.75, zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path} ({len(gdf)} domain(s) - {label})")


def plot_final_domains_map(final_gdf: gpd.GeoDataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domains_final.png"
    plot_domains_map(final_gdf, out_path, DOMAIN_COLOR, "final simulation domains")
    return out_path


def plot_dropped_unreachable_map(dropped_gdf: gpd.GeoDataFrame, out_dir: Path) -> Path | None:
    """None (writes nothing) if there's nothing to plot - an empty map is
    not a useful artifact, and a 0-domain scatter would just be a blank
    globe."""
    if len(dropped_gdf) == 0:
        print("No unreachable-dropped domains this run - skipping domains_dropped_unreachable.png")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domains_dropped_unreachable.png"
    plot_domains_map(
        dropped_gdf, out_path, DROPPED_COLOR,
        "domains dropped (no connectivity path to open water)",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-grid", default=str(DEFAULT_TILE_GRID))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    final_gdf = gpd.read_file(args.tile_grid)
    print(f"{len(final_gdf)} domain(s) from {args.tile_grid}")
    plot_final_domains_map(final_gdf, Path(args.out_dir))


if __name__ == "__main__":
    main()
