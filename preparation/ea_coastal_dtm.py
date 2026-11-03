#!/usr/bin/env python3
"""
Download the EA LIDAR Composite DTM 1m (2022) for coastal England.

  1. build the OS 5 km tile grid and keep tiles within --buffer m of the coast
  2. fetch each tile via the Defra Data Services Platform's WCS 2.0.1 service
     (GetCoverage, one request per tile, LZW-compressed GeoTIFF)
  3. write a VRT mosaic

Usage (standalone CLI, staged):
  python ea_coastal_dtm.py tiles --land england.gpkg --buffer 30000 --out tiles.gpkg
  python ea_coastal_dtm.py download --tiles tiles.gpkg --outdir dtm
  python ea_coastal_dtm.py mosaic --outdir dtm

Usage (repo pipeline, all three stages into inputs/validation/England/new_inputs/lidar_dem):
  python run_preparation.py ea_coastal_dtm --config snakemake_workflow/config/config.yml

Requires: geopandas, shapely, requests; GDAL python bindings for `mosaic`.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import retry_transient_io  # noqa: E402

# Shared defaults - both run(config) and the standalone CLI's own --buffer flag
# below read this, so they can't drift apart.
#
# 30 km, not the original 7.5 km: checked against the EA Flood Zone 3 (sea)
# validation benchmark (inputs/validation/England/fz3_sea_modelled.gpkg) - at
# 7.5 km, 13.7% of its 7,071 km^2 (969 km^2) fell outside the tile selection,
# concentrated in estuaries a straight-line coastal buffer can't follow (Humber,
# the Wash, Severn, Suffolk's Deben/Orwell - tidal flooding travels far inland
# along the river there). 30 km is the buffer at which 0% of the benchmark is
# left uncovered (checked at 2.5 km steps to 15 km, then 20/25/30 km).
_BUFFER_M = 30000

BNG = "EPSG:27700"
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
TILE = 5000

# The script originally shipped with two download methods (a Defra ArcGIS GP
# "portal" job-submission service, and a separate ArcGIS ImageServer
# exportImage endpoint) - both confirmed dead 2026-11-03: every path under
# environment.data.gov.uk/arcgis/rest/services and /image/rest/services now
# returns a generic {"error":{"code":400,"message":"Invalid URL"}} body (200
# status) or an outright 403 from the Azure gateway fronting it, for the
# service root as much as for any specific service - EA decommissioned that
# whole API generation, not just these two endpoints. The replacement is the
# Defra Data Services Platform's WCS 2.0.1 service (found via the dataset's
# "Survey Data - Download" resource on data.gov.uk, which now points at an
# interactive-only app, plus its sibling WCS resource, which is live and
# standard OGC): https://environment.data.gov.uk/spatialdata/
# lidar-composite-digital-terrain-model-dtm-1m/wcs - confirmed working
# end-to-end (GetCapabilities, DescribeCoverage, a real GetCoverage request
# verified byte-identical with/without geotiff:compression=LZW, ~5x smaller
# with it). One coverage spans all of England at native 1 m resolution, so a
# GetCoverage subset request per tile is simpler than the old job-queue
# methods - no polling, no AOI-batching, no result-URL harvesting.
WCS_SERVER = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
WCS_COVERAGE_ID = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"


# ---------------------------------------------------------------- OS grid --

def os_tile_name(e, n):
    """5 km OS grid name (e.g. TM33nw) for the tile whose SW corner is (e, n)."""
    E, N = e + 1_000_000, n + 500_000
    l1 = LETTERS[(4 - N // 500_000) * 5 + E // 500_000]
    l2 = LETTERS[(4 - (N % 500_000) // 100_000) * 5 + (E % 500_000) // 100_000]
    e10, n10 = (e % 100_000) // 10_000, (n % 100_000) // 10_000
    q = ("n" if (n % 10_000) >= TILE else "s") + ("e" if (e % 10_000) >= TILE else "w")
    return f"{l1}{l2}{e10}{n10}{q}"


def build_tiles(land, buffer_m):
    """Tiles intersecting the land boundary buffered by buffer_m."""
    land = land.to_crs(BNG)
    # Dissolve into one outline BEFORE taking .boundary: `land` is 152 separate
    # county/unitary-authority polygons (Natural Earth's England admin-1 layer is
    # subdivided that way, not one outline for the whole country), so a per-row
    # .boundary includes every inland county-to-county border, not just the real
    # coastline - buffering those too blanketed nearly the whole country (212,350
    # km^2 at 30 km, 163% of England's own 130,279 km^2 land area, and genuinely
    # included Birmingham/Leicester/Nottingham). Dissolving first drops the
    # internal borders, leaving only the true outer perimeter to buffer.
    dissolved = land.union_all() if hasattr(land, "union_all") else land.unary_union
    land = gpd.GeoDataFrame(geometry=[dissolved], crs=BNG)
    coast = land.boundary.buffer(buffer_m).union_all() if hasattr(land, "union_all") \
        else land.boundary.buffer(buffer_m).unary_union
    minx, miny, maxx, maxy = coast.bounds
    e0, n0 = int(minx // TILE) * TILE, int(miny // TILE) * TILE
    names, geoms = [], []
    for e in range(max(e0, 0), int(maxx) + TILE, TILE):
        for n in range(max(n0, 0), int(maxy) + TILE, TILE):
            g = box(e, n, e + TILE, n + TILE)
            if g.intersects(coast):
                names.append(os_tile_name(e, n))
                geoms.append(g)
    return gpd.GeoDataFrame({"tile": names}, geometry=geoms, crs=BNG)


def load_land(path):
    if path:
        return gpd.read_file(path)
    # fallback: Natural Earth 10 m admin-1, England only (coarse but fine for 5 km tiles)
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
    ne = gpd.read_file(url)
    return ne[(ne["admin"] == "United Kingdom") & (ne["geonunit"] == "England")]


# -------------------------------------------------------------- WCS method --

def prune_stale_tiles(tiles, outdir) -> int:
    """Delete any {tile}.tif in outdir not in the current tile selection.

    download_wcs already names files by tile ID and skips ones already on disk,
    so re-running after the tile selection changes (a different --buffer, or the
    2026-11-03 county-boundary fix in build_tiles) downloads only what's newly
    missing - but anything downloaded under the OLD selection that's no longer
    wanted would otherwise sit there forever. Returns the number removed.
    """
    valid_names = set(tiles["tile"])
    removed = 0
    for p in outdir.glob("*.tif"):
        if p.stem not in valid_names:
            p.unlink()
            removed += 1
    if removed:
        print(f"removed {removed} stale tile(s) no longer in the coastal selection")
    return removed


def download_wcs(tiles, outdir):
    """One WCS GetCoverage request per tile, its full 5 km extent in one shot."""
    session = requests.Session()
    for _, t in tiles.iterrows():
        dst = outdir / f"{t.tile}.tif"
        if dst.exists():
            continue
        minx, miny, maxx, maxy = t.geometry.bounds
        params = {
            "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
            "coverageId": WCS_COVERAGE_ID,
            "subset": [f"E({minx},{maxx})", f"N({miny},{maxy})"],
            "format": "image/tiff",
            "geotiff:compression": "LZW",
        }
        try:
            r = session.get(WCS_SERVER, params=params, timeout=300)
        except requests.RequestException as ex:
            print("skip", dst.name, f"request failed: {ex}", file=sys.stderr)
            continue
        if r.status_code != 200 or not r.content.startswith((b"II*\x00", b"MM\x00*")):
            print("skip", dst.name, r.status_code, r.content[:200], file=sys.stderr)
            continue
        # retry_transient_io: an SMB blip on the shared P:\ root raises
        # FileNotFoundError/OSError here even though outdir still exists -
        # confirmed in a real 29289s run that died on tile TR34ne this way,
        # losing every tile still queued behind it.
        retry_transient_io(dst.write_bytes, r.content)
        print("got", dst.name)


# ----------------------------------------------------------------- mosaic --

def mosaic(outdir):
    from osgeo import gdal
    tifs = sorted(str(p) for p in outdir.rglob("*.tif"))
    # No srcNodata/VRTNodata override: each downloaded tile already carries its
    # own correct nodata tag from the WCS service (-3.4028235e+38, float32's
    # sentinel min - NOT the old ArcGIS exportImage methods' -9999), so GDAL
    # picks it up from the source files directly.
    gdal.BuildVRT(str(outdir / "dtm_1m_2022_coastal.vrt"), tifs)
    print(f"{len(tifs)} rasters -> {outdir / 'dtm_1m_2022_coastal.vrt'}")


# ------------------------------------------------------------- run(config) --

def run(config: dict) -> None:
    """Build the coastal tile grid, download EA LIDAR DTM tiles, mosaic to a VRT.

    All three stages write flat into inputs/validation/England/new_inputs/lidar_dem
    under paths.root, alongside this repo's other England validation inputs. Each stage
    is resumable: coastal_tiles.gpkg is rebuilt only if missing, prune_stale_tiles
    removes any already-downloaded tile no longer in the current selection (e.g.
    after a --buffer change or a build_tiles fix), download_wcs then skips
    whatever's still on disk and fetches only what's newly missing, and mosaic
    always rescans and rewrites the VRT (cheap - just a rglob + BuildVRT).
    """
    out_root = Path(config["paths"]["root"]) / "inputs" / "validation" / "England" / "new_inputs" / "lidar_dem"
    retry_transient_io(out_root.mkdir, parents=True, exist_ok=True)

    tiles_path = out_root / "coastal_tiles.gpkg"
    if tiles_path.exists():
        print(f"{tiles_path.name} already exists - skipping tile-grid build.")
    else:
        tiles = build_tiles(load_land(None), _BUFFER_M)
        tiles.to_file(tiles_path, driver="GPKG")
        print(f"{len(tiles)} coastal tiles -> {tiles_path}")

    tiles = gpd.read_file(tiles_path).to_crs(BNG)
    prune_stale_tiles(tiles, out_root)
    download_wcs(tiles, out_root)
    mosaic(out_root)


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("tiles")
    a.add_argument("--land", help="land polygon (any CRS); default Natural Earth England")
    a.add_argument("--buffer", type=float, default=_BUFFER_M, help="coastal strip half-width, m")
    a.add_argument("--out", default="coastal_tiles.gpkg")

    b = sub.add_parser("download")
    b.add_argument("--tiles", default="coastal_tiles.gpkg")
    b.add_argument("--outdir", default="dtm")

    c = sub.add_parser("mosaic")
    c.add_argument("--outdir", default="dtm")

    args = ap.parse_args()
    if args.cmd == "tiles":
        tiles = build_tiles(load_land(args.land), args.buffer)
        tiles.to_file(args.out, driver="GPKG")
        print(f"{len(tiles)} coastal tiles -> {args.out}")
    elif args.cmd == "download":
        tiles = gpd.read_file(args.tiles).to_crs(BNG)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        prune_stale_tiles(tiles, outdir)
        download_wcs(tiles, outdir)
    else:
        mosaic(Path(args.outdir))


if __name__ == "__main__":
    main()
