"""Global Equal Earth maps of simulation domains, for the paper's methods
figures: the final domain set, and the domains dropped for lack of
population exposure. Reads the debug GeoPackages tile_generation writes
when tile_generation.write_debug_gpkg=true (see src/tile_chunking.py).

Usage:
    python plot_domain_maps.py [--debug-dir DIR] [--out-dir DIR]
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
OCEAN_COLOR = "#fcfcfb"
COAST_COLOR = "#b8b8b3"
DOMAIN_COLOR = "#2a78d6"
DROPPED_COLOR = "#d6572a"

DEFAULT_DEBUG_DIR = Path(
    r"P:\11212688-004-global-floodmaps\modelling\processed_inputs\mask\tile_generation_debug"
)
DEFAULT_OUT_DIR = Path(r"P:\11212688-004-global-floodmaps\modelling\processed_inputs\mask")


def plot_domains_map(gdf: gpd.GeoDataFrame, out_path: Path, color: str, label: str) -> None:
    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor=OCEAN_COLOR)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)

    ax.add_geometries(
        gdf.geometry, crs=ccrs.PlateCarree(),
        facecolor=color, edgecolor=color, linewidth=0.2, alpha=0.75, zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=OCEAN_COLOR)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(gdf)} domain(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_domains = gpd.read_file(debug_dir / "13_chunks_ordered.gpkg")
    plot_domains_map(
        final_domains, out_dir / "domains_final.png", DOMAIN_COLOR, "final simulation domains",
    )

    dropped_no_exposure = gpd.read_file(debug_dir / "07_dropped_no_exposure_or_shave_empty.gpkg")
    dropped_no_exposure = dropped_no_exposure[dropped_no_exposure["reason"] == "no_population_exposure"]
    plot_domains_map(
        dropped_no_exposure, out_dir / "domains_dropped_no_exposure.png", DROPPED_COLOR,
        "domains dropped (no population exposure)",
    )


if __name__ == "__main__":
    main()
