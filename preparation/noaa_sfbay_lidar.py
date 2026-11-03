#!/usr/bin/env python3
"""
Download the 2025 San Francisco Bay–Delta lidar from NOAA's public S3 bucket.

NOAA IDs: 14675 = 1 ft hydro-flattened DEM (GeoTIFF), 14673 = classified point cloud (LAZ).
No credentials needed. Tiles are CA State Plane Zone 2/3, NAD83(2011), US survey feet,
NAVD88 ft (GEOID18).

Usage (standalone CLI):
  python noaa_sfbay_lidar.py list --product dem
  python noaa_sfbay_lidar.py download --product dem --bbox -122.55 37.7 -122.3 37.9 --outdir sf_dem
  python noaa_sfbay_lidar.py download --product laz --aoi my_area.gpkg --outdir sf_laz
  python noaa_sfbay_lidar.py download --product dem --outdir sf_dem            # everything (~large)
  python noaa_sfbay_lidar.py tometres --indir sf_dem --epsg 26910 --out sf_dem_m.tif

Usage (repo pipeline, DEM clipped to the CoSMoS SF Bay benchmark's own flood
extent + a 3 km buffer, converted to metres, into
inputs/validation/California/new_inputs/lidar_dem):
  python run_preparation.py noaa_sfbay_lidar --config snakemake_workflow/config/config.yml

Requires: requests; geopandas + shapely for --bbox/--aoi; rasterio + GDAL for `tometres`.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import retry_transient_io  # noqa: E402

BUCKET = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
PRODUCTS = {
    "dem": {"prefix": "dem/CA_SFBay_Hyfltnd_DEM_2025_14675/", "ext": ".tif"},
    "laz": {"prefix": "laz/geoid18/14673/", "ext": ".laz"},
}
US_FT = 0.304800609601  # US survey foot -> metre
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


# -------------------------------------------------------------- S3 listing --

def list_keys(prefix, session=None):
    """All (key, size) under a prefix via unsigned ListObjectsV2."""
    session = session or requests.Session()
    keys, token, page = [], None, 0
    print(f"listing s3://{prefix} ...", file=sys.stderr, flush=True)
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", NS):
            keys.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        page += 1
        print(f"  ...page {page}, {len(keys)} keys so far", file=sys.stderr, flush=True)
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            print(f"listing done: {len(keys)} keys", file=sys.stderr, flush=True)
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def tile_keys(keys, ext):
    return [(k, s) for k, s in keys if k.lower().endswith(ext) and "/supplemental/" not in k]


# ----------------------------------------------------------- AOI selection --

def select_by_aoi(keys, ext, tileindex_key, aoi, outdir, session):
    """Keep tiles whose footprint (from NOAA's tile index shapefile) intersects the AOI."""
    import geopandas as gpd

    zp = outdir / Path(tileindex_key).name
    if not zp.exists():
        print(f"fetching tile index {Path(tileindex_key).name} ...", file=sys.stderr, flush=True)
        fetch(session, tileindex_key, zp)
    print("matching tile index against AOI ...", file=sys.stderr, flush=True)
    with zipfile.ZipFile(zp) as z:
        shp = [n for n in z.namelist() if n.lower().endswith(".shp")][0]
    idx = gpd.read_file(f"zip://{zp}!{shp}")
    aoi = aoi.to_crs(idx.crs)
    hit = idx[idx.intersects(aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union)]
    # tile index rows carry a url/name column; match on the file stem
    col = next(c for c in idx.columns if c.lower() in ("url", "name", "location", "filename", "tile"))
    stems = {Path(str(v)).stem.lower() for v in hit[col]}
    sel = [(k, s) for k, s in keys if Path(k).stem.lower() in stems]
    print(f"{len(hit)} index tiles intersect AOI -> {len(sel)} {ext} keys", file=sys.stderr, flush=True)
    return sel


def load_aoi(args):
    import geopandas as gpd
    from shapely.geometry import box

    if args.aoi:
        return gpd.read_file(args.aoi)
    if args.bbox:
        w, s, e, n = args.bbox
        return gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326")
    return None


# --------------------------------------------------------------- download --

def fetch(session, key, dst, size=None):
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return "skip"
    with session.get(f"{BUCKET}/{key}", stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
        tmp.rename(dst)
    return "ok"


def download(sel, outdir, workers):
    outdir.mkdir(parents=True, exist_ok=True)
    total = sum(s for _, s in sel) / 1e9
    print(f"{len(sel)} files, {total:.1f} GB -> {outdir}", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, requests.Session(), k, outdir / Path(k).name, s): k for k, s in sel}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(sel)}] {f.result()} {Path(futs[f]).name}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(sel)}] FAIL {futs[f]}: {e}", file=sys.stderr, flush=True)


# ------------------------------------------------- ft -> m, reproject, VRT --

def tometres(indir, epsg, out, res):
    """Mosaic DEM tiles, reproject to EPSG, convert ftUS -> m (horizontal via GDAL, vertical by scaling)."""
    from osgeo import gdal

    gdal.UseExceptions()
    tifs = sorted(str(p) for p in indir.glob("*.tif"))
    print(f"building VRT from {len(tifs)} tiles ...", file=sys.stderr, flush=True)
    vrt = str(indir / "sfbay_dem_ft.vrt")
    gdal.BuildVRT(vrt, tifs)
    print(f"warping to EPSG:{epsg} @ {res} m ... (this is the slow step - a single blocking "
          f"GDAL call over all {len(tifs)} tiles; % progress below comes from GDAL itself)",
          file=sys.stderr, flush=True)
    warped = str(indir / "sfbay_dem_ft_proj.tif")
    gdal.Warp(warped, vrt, dstSRS=f"EPSG:{epsg}", xRes=res, yRes=res, resampleAlg="bilinear",
              dstNodata=-9999, creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
              callback=gdal.TermProgress_nocb)
    print("converting ftUS -> m ...", file=sys.stderr, flush=True)
    import rasterio
    with rasterio.open(warped) as src:
        prof = src.profile
        windows = list(src.block_windows(1))
        n = len(windows)
        step = max(1, n // 20)
        with rasterio.open(out, "w", **prof) as dst:
            for i, (_, win) in enumerate(windows, 1):
                a = src.read(1, window=win)
                m = a != src.nodata
                a[m] = a[m] * US_FT
                dst.write(a, 1, window=win)
                if i % step == 0 or i == n:
                    print(f"  ...{i}/{n} blocks", file=sys.stderr, flush=True)
    print(f"-> {out}  (EPSG:{epsg}, {res} m, NAVD88 metres)", flush=True)


# --------------------------------------------------------- flood-extent AOI --

_AOI_BUFFER_M = 3000
# ~20 m effective resolution reading the CoSMoS raster (2 m, 34,750 x 49,500 px -
# ~1.7 billion px full-res): only the extent's rough shape matters here, not
# per-pixel precision at a multi-km buffer scale, so a decimated read is both
# far cheaper and sufficient.
_AOI_DECIMATE = 10


def _benchmark_flood_extent_aoi(benchmark_path: Path, buffer_m: float):
    """CoSMoS SF Bay benchmark's own flood extent, dissolved and buffered.

    "Extent" = the raster's valid (non-nodata) footprint: per validate_sf_bay.py's
    own docstring, "CoSMoS only has values inside the flood extent" - nodata
    outside it is the product's own convention for dry/unsurveyed, not missing
    data to fill in. 234 km^2 raw, 2,654 km^2 with the 3 km buffer - vs. the
    30,247 km^2 sf_bay_county_bbox_union.gpkg (union of 9 counties' *rectangular*
    extents, mostly dry inland land nothing here needs 1 m DTM for).
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape

    print(f"reading {benchmark_path.name} (decimated {_AOI_DECIMATE}x) ...", file=sys.stderr, flush=True)
    with rasterio.open(benchmark_path) as src:
        out_h, out_w = src.height // _AOI_DECIMATE, src.width // _AOI_DECIMATE
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        crs = src.crs
    mask = ~np.isnan(data)
    print(f"vectorizing flood extent ({mask.sum()} of {mask.size} decimated cells wet) ...",
          file=sys.stderr, flush=True)
    polys = [shapely_shape(geom) for geom, _ in shapes(mask.astype("uint8"), mask=mask, transform=transform)]
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    dissolved = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    print(f"dissolved flood extent: {dissolved.area / 1e6:.1f} km^2, buffering {buffer_m:.0f} m ...",
          file=sys.stderr, flush=True)
    buffered = dissolved.buffer(buffer_m)
    print(f"buffered AOI: {buffered.area / 1e6:.1f} km^2", file=sys.stderr, flush=True)
    return gpd.GeoDataFrame(geometry=[buffered], crs=crs)


# ------------------------------------------------------------- run(config) --

def run(config: dict) -> None:
    """Download the 2025 SF Bay DEM clipped to the CoSMoS flood extent, to metres.

    Product is always "dem" - "laz" (classified point cloud) is ~3.5x bigger and
    nothing in this repo reads point clouds. AOI is the CoSMoS SF Bay benchmark's
    own flood extent (sf_bay_flddepth_slr000_w100_mosaic.tif, the same benchmark
    validate_sf_bay.py scores against) plus a 3 km buffer - not
    sf_bay_county_bbox_union.gpkg, whose 30,247 km^2 of rectangular county
    extents is mostly dry inland land this LIDAR has no reason to cover.
    Output: inputs/validation/California/new_inputs/lidar_dem/sfbay_dem_m.tif
    (EPSG:26910, 1 m, NAVD88 metres), alongside the existing California/flooding/
    benchmark data, plus the AOI polygon itself (flood_extent_aoi_3km_buffer.gpkg)
    for inspection.
    Resumable: fetch() skips a file already on disk at its expected size, and
    tometres() is cheap to just rerun (one VRT build + one warp + one unit scale).
    """
    benchmark_root = Path(config["validation"]["benchmark_root"])
    benchmark_path = benchmark_root / "California" / "flooding" / "sf_bay_flddepth_slr000_w100_mosaic.tif"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"{benchmark_path} not found - this is the CoSMoS SF Bay benchmark "
            "validate_sf_bay.py already scores against (docs/data.md); without it "
            "there's no principled AOI to restrict the ~167 GB DEM bucket to, so "
            "this step refuses to just download everything."
        )

    out_root = benchmark_root / "California" / "new_inputs" / "lidar_dem"
    retry_transient_io(out_root.mkdir, parents=True, exist_ok=True)

    print("=== building flood-extent AOI ===", file=sys.stderr, flush=True)
    aoi = _benchmark_flood_extent_aoi(benchmark_path, _AOI_BUFFER_M)
    aoi.to_file(out_root / "flood_extent_aoi_3km_buffer.gpkg", driver="GPKG")

    print("=== listing NOAA bucket + selecting tiles ===", file=sys.stderr, flush=True)
    session = requests.Session()
    prod = PRODUCTS["dem"]
    keys = list_keys(prod["prefix"], session)
    tiles = tile_keys(keys, prod["ext"])
    tidx = [k for k, _ in keys if "tileindex" in k.lower() and k.lower().endswith(".zip")]
    if not tidx:
        raise RuntimeError("no tile index found in the NOAA bucket listing - check prefix/bucket layout")
    sel = select_by_aoi(tiles, prod["ext"], tidx[0], aoi, out_root, session)

    print("=== downloading tiles ===", file=sys.stderr, flush=True)
    download(sel, out_root, workers=6)
    print("=== mosaicking + converting to metres ===", file=sys.stderr, flush=True)
    tometres(out_root, 26910, str(out_root / "sfbay_dem_m.tif"), 1.0)


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list")
    l.add_argument("--product", choices=PRODUCTS, default="dem")

    d = sub.add_parser("download")
    d.add_argument("--product", choices=PRODUCTS, default="dem")
    d.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="lon/lat bbox")
    d.add_argument("--aoi", help="polygon file (any CRS)")
    d.add_argument("--outdir", default="sfbay")
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--dry", action="store_true", help="list selection only")

    m = sub.add_parser("tometres")
    m.add_argument("--indir", default="sfbay")
    m.add_argument("--epsg", type=int, default=26910, help="target CRS (26910 = UTM 10N NAD83)")
    m.add_argument("--res", type=float, default=1.0, help="output cell size, m")
    m.add_argument("--out", default="sfbay_dem_m.tif")

    args = ap.parse_args()
    if args.cmd == "tometres":
        tometres(Path(args.indir), args.epsg, args.out, args.res)
        return

    prod = PRODUCTS[args.product]
    session = requests.Session()
    keys = list_keys(prod["prefix"], session)
    tiles = tile_keys(keys, prod["ext"])
    if args.cmd == "list":
        for k, s in keys:
            print(f"{s/1e6:9.1f} MB  {k}")
        print(f"\n{len(tiles)} {prod['ext']} tiles, {sum(s for _, s in tiles)/1e9:.1f} GB", file=sys.stderr)
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    aoi = load_aoi(args)
    sel = tiles
    if aoi is not None:
        tidx = [k for k, _ in keys if "tileindex" in k.lower() and k.lower().endswith(".zip")]
        if not tidx:
            sys.exit("no tile index found in bucket listing; run `list` and check")
        sel = select_by_aoi(tiles, prod["ext"], tidx[0], aoi, outdir, session)
    if args.dry:
        for k, s in sel:
            print(f"{s/1e6:9.1f} MB  {k}")
        return
    download(sel, outdir, args.workers)


if __name__ == "__main__":
    main()
