"""Plot global Equal Earth maps of the final and dropped-unreachable tile domains.

build_tile_manifest.run() calls these directly; standalone use re-plots
domains_final.png from tile_grid.path. --mark-no-stations highlights domains the
simulation skipped for no COAST-RP station (eikonal_runner.NO_STATIONS_REASON).

Usage:
    python plot_domain_maps.py
    python plot_domain_maps.py --tile-grid path/to/domain_tiles_global.gpkg --out-dir DIR
    python plot_domain_maps.py --mark-no-stations [--skipped-dir <model_outputs>/skipped_tiles]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

LAND_COLOR = "#d8d8d4"
COAST_COLOR = "#b8b8b3"
DOMAIN_COLOR = "#2a78d6"
INLAND_DOMAIN_COLOR = "#8db9ea"  # hop_distance > 0
DROPPED_COLOR = "#d6572a"
NO_STATIONS_COLOR = "#ef8a7e"


def no_station_tile_ids(skipped_dir: Path) -> set[int]:
    """Return tile ids skipped for no COAST-RP station (one marker per tile is read)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eikonal_runner import NO_STATIONS_REASON

    first: dict[str, Path] = {}
    for f in sorted(Path(skipped_dir).glob("*.txt")):
        first.setdefault(f.name.split("_", 1)[0], f)
    return {int(t) for t, f in first.items() if t.isdigit()
            and NO_STATIONS_REASON in f.read_text(errors="replace")}


def plot_domains_map(gdf: gpd.GeoDataFrame, out_path: Path, color: str, label: str,
                     inland_color: str | None = None, highlight: set[int] | None = None,
                     highlight_color: str = NO_STATIONS_COLOR) -> None:
    """Plot domains on a global Equal Earth map.

    inland_color: hop_distance > 0 domains, drawn under hop 0 ones.
    highlight: tile ids drawn on top in highlight_color.
    """
    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    layers = [(gdf, color, 3)]
    if inland_color is not None and "hop_distance" in gdf.columns:
        inland = gdf["hop_distance"] > 0
        layers = [(gdf[inland], inland_color, 3), (gdf[~inland], color, 4)]
    if highlight:
        marked = gdf["tile_id"].astype(int).isin(highlight)
        layers = [(sub[~marked.loc[sub.index]], c, z) for sub, c, z in layers] + [(gdf[marked], highlight_color, 5)]
    for sub, c, z in layers:
        if len(sub):
            # wider outline keeps small highlighted domains visible at global scale
            ax.add_geometries(sub.geometry, crs=ccrs.PlateCarree(), facecolor=c, edgecolor=c,
                              linewidth=0.6 if z == 5 else 0.2, alpha=0.9 if z == 5 else 0.75, zorder=z)
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path} ({len(gdf)} domain(s) - {label})")


def plot_final_domains_map(final_gdf: gpd.GeoDataFrame, out_dir: Path,
                           no_station_ids: set[int] | None = None) -> Path:
    """Plot final domains; with no_station_ids, write domains_final_no_stations.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_station_ids:
        out_path = out_dir / "domains_final_no_stations.png"
        n = int(final_gdf["tile_id"].astype(int).isin(no_station_ids).sum())
        plot_domains_map(final_gdf, out_path, DOMAIN_COLOR,
                         f"final simulation domains, {n} skipped for no COAST-RP station in light red",
                         inland_color=INLAND_DOMAIN_COLOR, highlight=no_station_ids)
        return out_path
    out_path = out_dir / "domains_final.png"
    plot_domains_map(final_gdf, out_path, DOMAIN_COLOR, "final simulation domains (hop > 0 lighter)",
                     inland_color=INLAND_DOMAIN_COLOR)
    return out_path


def plot_dropped_unreachable_map(dropped_gdf: gpd.GeoDataFrame, out_dir: Path) -> Path | None:
    """Plot domains dropped as unreachable; write nothing and return None if empty."""
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
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-grid", default=None, help="defaults to tile_grid.path from --config")
    parser.add_argument("--out-dir", default=None,
                        help="defaults to <processed_inputs_dir>/mask from --config")
    parser.add_argument("--mark-no-stations", action="store_true",
                        help="write domains_final_no_stations.png: domains skipped for having no COAST-RP "
                             "station in light red (needs the simulation's skipped_tiles markers)")
    parser.add_argument("--skipped-dir", default=None,
                        help="<model_outputs>/skipped_tiles (only with --mark-no-stations); defaults to "
                             "simulation.model_outputs/skipped_tiles from --config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_grid = args.tile_grid or cfg["tile_grid"]["path"]
    out_dir = args.out_dir or str(Path(cfg["paths"]["processed_inputs_dir"]) / "mask")
    skipped_dir = args.skipped_dir or str(Path(cfg["simulation"]["model_outputs"]) / "skipped_tiles")

    final_gdf = gpd.read_file(tile_grid)
    print(f"{len(final_gdf)} domain(s) from {tile_grid}")
    ids = None
    if args.mark_no_stations:
        ids = no_station_tile_ids(Path(skipped_dir))
        print(f"{len(ids)} domain(s) skipped for no COAST-RP station ({skipped_dir})")
    plot_final_domains_map(final_gdf, Path(out_dir), no_station_ids=ids)


if __name__ == "__main__":
    main()
