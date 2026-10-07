"""Diagnostic figure, one per tile, for every tile under a calibration/
validation run (default `sfincs_calibration`) that has no COAST-RP boundary
station for the given scenario - the exact failure mode
`build_boundary_forcing.py`/`build_sfincs_tile.py` reports as "boundaries_
{RP}_{SLR}.gpkg is empty (no COAST-RP station for this tile)" and drops the
tile for.

A tile matches iff its own `inputs/boundaries_{return_period}_
{waterlevel_name}.gpkg` exists and has zero rows - that file's presence and
emptiness is read directly off disk, not parsed from log text, so this is
accurate regardless of how long ago the run happened or whether its logs
are still around.

For each matching tile, writes `{tile_id}.png` to a new subdirectory
(`--out-subdir`, default `no_coastrp_station_figures`) under the run's own
directory:
  - Main panel: a view wide enough to show the tile's own footprint AND
    its `--n-nearest` closest raw COAST-RP stations together (padded union
    of the tile bbox and those stations' locations) - land/water drawn from
    the global DeltaDTM mask VRT (not the tile's own small `inputs/
    mask.tif`, which rarely covers ground anywhere near a station hundreds
    of km away). The tile's own bbox is drawn as an outlined rectangle so
    it stays identifiable even when the view has to zoom out a long way to
    reach the nearest real station - itself a useful diagnostic: how
    isolated this tile actually is.
  - Figure title: tile ID and the exact distance to the single nearest
    COAST-RP station (geodesic, WGS84).
  - Top-right inset: a small global Equal Earth locator map marking the
    tile's own location, for geographic context at a glance.

Distances are computed against every RAW COAST-RP station (`coast_rp`
catalog entry, ~23k stations globally) - before any per-tile ocean-
connectivity filtering - since "no COAST-RP station" is itself a
pre-connectivity-filter emptiness (see module docstring above), not a
connectivity rejection.

Run under `gfm_python_preprocessing`, like every other sfincs_tiles/
plotting script (hydromt-sfincs-dev's matplotlib crashes on savefig).

Usage:
    python plot_no_coastrp_tiles.py
    python plot_no_coastrp_tiles.py --base-dir-name sfincs_calibration --n-nearest 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root, resolve_catalog_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import LAND_COLOR, LAND_LABEL, WATER_COLOR, WATER_LABEL, draw_caption_box  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
NEAREST_COLOR = "#d62728"  # same alert-red family as map_style's B_ONLY_COLOR
TILE_OUTLINE_COLOR = "#222222"
N_NEAREST_DEFAULT = 5
VIEW_PAD_FRAC = 0.15  # fraction of the tile-plus-stations span added as margin
VIEW_PAD_MIN_DEG = 0.05  # floor, so a degenerate (near-zero-span) view never happens
MAX_READ_PX = 3000  # cap on the land/water raster read's longer side, see _land_water_background
DELTADTM_LAND_CODE = 0  # inputs/DeltaDTM_masks/deltadtm_mask.vrt convention: 0=land, 1=ocean, 2=lake, 3=river
DELTADTM_OCEAN_CODE = 1
GEOD = Geod(ellps="WGS84")


def find_no_coastrp_tiles(base_dir: Path, return_period: str, waterlevel_name: str) -> list[int]:
    """Tile IDs under base_dir whose own boundaries gpkg for this scenario
    exists and is empty - see module docstring for why this, not log text,
    is the ground truth."""
    matches = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        bpath = d / "inputs" / f"boundaries_{return_period}_{waterlevel_name}.gpkg"
        if not bpath.exists():
            continue
        if gpd.read_file(bpath).empty:
            matches.append(int(d.name))
    return matches


def load_all_coastrp_stations(catalog_path: Path, root: Path) -> gpd.GeoDataFrame:
    """Every raw COAST-RP station (lon/lat only - no per-scenario water
    level needed for a nearest-distance diagnostic)."""
    nc_path = resolve_catalog_path(catalog_path, root, "coast_rp")
    with xr.open_dataset(nc_path) as ds:
        lon = ds["station_x_coordinate"].values.astype(float)
        lat = ds["station_y_coordinate"].values.astype(float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    return gpd.GeoDataFrame(
        {"lon": lon[valid], "lat": lat[valid]},
        geometry=gpd.points_from_xy(lon[valid], lat[valid]), crs="EPSG:4326",
    )


def nearest_stations(tile_lon: float, tile_lat: float, stations: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """The n closest stations to (tile_lon, tile_lat), geodesic distance
    (metres, WGS84), ascending."""
    _, _, dist_m = GEOD.inv(
        np.full(len(stations), tile_lon), np.full(len(stations), tile_lat),
        stations["lon"].to_numpy(), stations["lat"].to_numpy(),
    )
    out = stations.copy()
    out["dist_km"] = dist_m / 1000.0
    return out.sort_values("dist_km").head(n).reset_index(drop=True)


def _land_water_background(mask_vrt: Path, bbox: list[float]):
    """Windowed read of the global DeltaDTM mask VRT, split into two
    independently-transparent layers: land (code 0) and ocean (code 1)
    only - lake/river (2/3) and anywhere outside the VRT's own coverage
    stay transparent, showing the figure's plain white background through,
    rather than being swept into a blanket water fill (a prior version of
    this function painted the whole axes facecolor WATER_COLOR, which
    wrongly colored lakes/rivers and genuine no-data gaps the same blue as
    real ocean). Same windowed-read recipe as
    analysis/plot_coastrp_stations_map.py's own `_land_background`, plus a
    read-resolution cap: an isolated tile's view can span tens of degrees
    to reach its nearest station, and the VRT's native ~1-arcsecond
    resolution over that large an area is gigapixels - more than this
    figure's own output size could ever show and large enough to exhaust
    memory outright. `out_shape` downsamples (nearest, since this is a
    categorical land/water read) to at most MAX_READ_PX per side."""
    with rasterio.open(mask_vrt) as src:
        b = src.bounds
        cb = (max(bbox[0], b.left), max(bbox[1], b.bottom), min(bbox[2], b.right), min(bbox[3], b.top))
        window = from_bounds(*cb, transform=src.transform).round_offsets().round_lengths()
        scale = min(1.0, MAX_READ_PX / max(window.height, window.width, 1))
        out_shape = (max(1, round(window.height * scale)), max(1, round(window.width * scale)))
        arr = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.nearest)
        transform = src.window_transform(window) * Affine.scale(
            window.width / out_shape[1], window.height / out_shape[0],
        )
    ext = [transform.c, transform.c + arr.shape[1] * transform.a,
           transform.f + arr.shape[0] * transform.e, transform.f]
    land = np.where(arr == DELTADTM_LAND_CODE, np.float32(1.0), np.float32(np.nan))
    water = np.where(arr == DELTADTM_OCEAN_CODE, np.float32(1.0), np.float32(np.nan))
    return land, water, ext


def _draw_locator_inset(fig, tile_lon: float, tile_lat: float) -> None:
    """Small top-right Equal Earth world map marking the tile's own
    location - geographic context at a glance, independent of the main
    panel's own (possibly very zoomed-out) view. Ocean left white (no
    facecolor fill) - land is the only thing this inset needs to show."""
    ax = fig.add_axes([0.70, 0.68, 0.28, 0.28], projection=ccrs.EqualEarth())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=1)
    ax.scatter([tile_lon], [tile_lat], transform=ccrs.PlateCarree(), s=40, marker="*",
               c=NEAREST_COLOR, edgecolors="white", linewidths=0.6, zorder=3)
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)


def make_figure(tile_id: int, tile_bbox: list[float], shown: gpd.GeoDataFrame,
                mask_vrt: Path, out_path: Path) -> None:
    """`shown` is the caller's already-sorted, already-truncated n-nearest
    stations (see `nearest_stations`) - this function only renders."""
    tile_lon = (tile_bbox[0] + tile_bbox[2]) / 2.0
    tile_lat = (tile_bbox[1] + tile_bbox[3]) / 2.0

    view_minx = min(tile_bbox[0], shown["lon"].min())
    view_miny = min(tile_bbox[1], shown["lat"].min())
    view_maxx = max(tile_bbox[2], shown["lon"].max())
    view_maxy = max(tile_bbox[3], shown["lat"].max())
    pad = max((view_maxx - view_minx) * VIEW_PAD_FRAC, (view_maxy - view_miny) * VIEW_PAD_FRAC, VIEW_PAD_MIN_DEG)
    view_bbox = [view_minx - pad, view_miny - pad, view_maxx + pad, view_maxy + pad]

    land, water, bg_ext = _land_water_background(mask_vrt, view_bbox)

    fig, ax = plt.subplots(figsize=(11, 11 * (view_bbox[3] - view_bbox[1]) / max(view_bbox[2] - view_bbox[0], 1e-6)))
    ax.set_facecolor("white")
    ax.imshow(land, extent=bg_ext, origin="upper", cmap=ListedColormap([LAND_COLOR]),
              interpolation="nearest", zorder=0)
    ax.imshow(water, extent=bg_ext, origin="upper", cmap=ListedColormap([WATER_COLOR]),
              interpolation="nearest", zorder=0)

    ax.add_patch(Rectangle(
        (tile_bbox[0], tile_bbox[1]), tile_bbox[2] - tile_bbox[0], tile_bbox[3] - tile_bbox[1],
        fill=False, edgecolor=TILE_OUTLINE_COLOR, linewidth=1.8, zorder=4,
    ))
    ax.scatter(shown["lon"], shown["lat"], c=NEAREST_COLOR, s=50, edgecolors="white",
               linewidths=0.8, zorder=5)
    for _, row in shown.iterrows():
        ax.annotate(f"{row['dist_km']:.0f} km", (row["lon"], row["lat"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=NEAREST_COLOR, zorder=6)

    nearest_km = float(shown["dist_km"].iloc[0])
    ax.set_xlim(view_bbox[0], view_bbox[2])
    ax.set_ylim(view_bbox[1], view_bbox[3])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(handles=[
        Patch(facecolor=LAND_COLOR, edgecolor="black", linewidth=0.3, label=LAND_LABEL),
        Patch(facecolor=WATER_COLOR, edgecolor="black", linewidth=0.3, label=WATER_LABEL),
        Patch(facecolor=TILE_OUTLINE_COLOR, label=f"tile {tile_id} footprint"),
        Patch(facecolor=NEAREST_COLOR, label=f"{len(shown)} nearest COAST-RP station(s)"),
    ], loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="grey")
    draw_caption_box(ax, [f"tile {tile_id}", f"no COAST-RP station within this tile"], loc="upper left")

    fig.suptitle(f"Tile {tile_id} - no COAST-RP boundary station "
                 f"(nearest: {nearest_km:.1f} km away)", fontsize=12)
    _draw_locator_inset(fig, tile_lon, tile_lat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path} (nearest station {nearest_km:.1f} km away)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--data-catalog", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"))
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--waterlevel-name", default="SLR_0")
    parser.add_argument("--out-subdir", default="no_coastrp_station_figures")
    parser.add_argument("--n-nearest", type=int, default=N_NEAREST_DEFAULT,
                         help="how many of the closest COAST-RP stations to plot/label per tile")
    args = parser.parse_args()

    config_path = Path(args.config)
    root = read_root(config_path)
    base_dir = root / args.base_dir_name
    mask_vrt = root / "inputs" / "DeltaDTM_masks" / "deltadtm_mask.vrt"

    tile_ids = find_no_coastrp_tiles(base_dir, args.return_period, args.waterlevel_name)
    print(f"{len(tile_ids)} tile(s) with no COAST-RP station "
          f"({args.return_period}/{args.waterlevel_name}): {tile_ids}")
    if not tile_ids:
        return

    stations = load_all_coastrp_stations(Path(args.data_catalog), root)
    print(f"{len(stations)} raw COAST-RP station(s) loaded for distance lookup")

    out_dir = base_dir / args.out_subdir
    for tile_id in tile_ids:
        with open(base_dir / str(tile_id) / "inputs" / "model_bbox.json") as f:
            tile_bbox = json.load(f)
        tile_lon = (tile_bbox[0] + tile_bbox[2]) / 2.0
        tile_lat = (tile_bbox[1] + tile_bbox[3]) / 2.0
        shown = nearest_stations(tile_lon, tile_lat, stations, n=args.n_nearest)
        make_figure(tile_id, tile_bbox, shown, mask_vrt, out_dir / f"{tile_id}.png")

    print(f"\nDone. {len(tile_ids)} figure(s) written to {out_dir}")


if __name__ == "__main__":
    main()
