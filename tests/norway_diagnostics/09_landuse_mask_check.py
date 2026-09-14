"""Does the pipeline's existing permanent-water mask already remove the sea from Norway's polygons?

07_land_inundation_check.py proved the raw ``stormflo*_klimaarna`` polygons are
"everything below level X" INCLUDING the permanently-wet sea (97.6-99.3% of raw
polygon area at two AOIs).  The obvious fix was an expensive vector-level
dissolve+difference against ``middelhoyvann_klimaarna`` at conversion time.

But validation/validate_country.py ALREADY drops permanent water from the
evaluation domain for every country: validation.read_permanent_water_mask() /
permanent_water_mask() flag Copernicus Global Land Cover codes 80 ("Permanent
water bodies") and 200 ("Open sea"), and validate_country.py applies the
resulting ``not_water`` to the benchmark-wet area, the model-wet area AND the
domain.  If Copernicus classifies Norway's sea/fjords correctly, that generic
mask removes the sea for free and no vector preprocessing is needed.

Norway's coastline is far more complex than Spain's or France's - narrow fjords,
thousands of small islands - so 100 m Copernicus may resolve it worse.  This
script checks it for real, on five AOIs spanning the coast:

    sea_region = dissolve(stormflo200ar_klimaarna) INTERSECT dissolve(middelhoyvann_klimaarna)

i.e. exactly the part of the raw benchmark polygon that 07's method calls sea,
then rasterizes that region onto a lon/lat grid at the validation pipeline's own
resolution and reports what fraction of it Copernicus calls permanent water -
reprojected with the SAME nearest-neighbour path read_permanent_water_mask()
uses, so this measures the mask the pipeline would really apply, not an
idealised one.

High fraction (>90%) => the simple design is fine: load the RAW polygons as an
ordinary extent benchmark and let the existing mask strip the sea.
Meaningfully lower => vector-level sea subtraction is still needed.

Phase 1 (``--extract``) streams the AOI subsets of middelhoyvann_klimaarna out
of the 26 GiB dump once and caches them; phase 2 (default) does the geometry and
raster work and reuses that cache.

Usage:
    python 09_landuse_mask_check.py --extract     # once, ~5 min (streams the dump)
    python 09_landuse_mask_check.py               # the actual check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box

from nor_dump import block_for, columns, iter_raw_lines, load_index, unescape

HERE = Path(__file__).parent
NOR = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\validation\NOR")
LANDUSE = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\Copernicus\Copernicus_LandUse.tif")
SURGE_GPKG = NOR / "stormflo200ar_klimaarna.gpkg"
SURGE_LAYER = "stormflo200ar_klimaarna"
SEA_TABLE = "middelhoyvann_klimaarna"
CACHE = HERE / "norway_landusecheck_mhw.gpkg"

# Copernicus codes the pipeline treats as permanent water (validation.py).
PERMANENT_WATER_CODES = (80, 200)

# Validation grid resolution. validate_country.py builds its grid from the model
# waterdepth rasters themselves (_mosaic_read), which are EPSG:4326 at the
# configured flooding.resolution of 30 m: exactly 1 arc-second in latitude, and
# 1/cos(lat) arc-seconds in longitude so cells stay ~30 m square. Verified
# against a real output (model_outputs/2254/results/waterdepth_RP100_SLR_0.tif:
# res 1.0000" lat, 1.6779" lon at lat 6.9). The permanent-water mask is
# nearest-neighbour reprojected onto that grid, so sample there - not at the
# land-cover raster's own coarser 3.57" (~100 m) spacing, which would understate
# how badly a 100 m product resolves a 200 m-wide fjord.
RES_Y_DEG = 1.0 / 3600.0

# AOIs in lon/lat, spanning the range of Norwegian coastal morphology.
AOIS = {
    "oslofjord":   (10.55, 59.80, 10.90, 59.98),   # wide, sheltered SE fjord
    "bergen":      (5.20, 60.33, 5.42, 60.45),     # west-coast city + islands
    "naeroyfjord": (6.80, 60.85, 7.30, 61.15),     # narrow steep inner fjord
    "lofoten":     (13.80, 68.10, 14.40, 68.40),   # island maze, above Arctic Circle
    "hammerfest":  (23.40, 70.50, 24.00, 70.80),   # Finnmark, far north
}

_TO_UTM = Transformer.from_crs(4326, 25833, always_xy=True)


def utm_box(bb: tuple[float, float, float, float]):
    x0, y0 = _TO_UTM.transform(bb[0], bb[1])
    x1, y1 = _TO_UTM.transform(bb[2], bb[3])
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# ── Phase 1: pull the mean-high-water AOI subsets out of the dump ────────────

def extract_sea_layer() -> None:
    """Stream middelhoyvann_klimaarna once, keeping only polygons hitting an AOI."""
    index = load_index()
    dump = Path(index["dump"])
    block = block_for(index, SEA_TABLE)
    cols = columns(block)
    gi = cols.index("omrade")
    boxes = {name: utm_box(bb).bounds for name, bb in AOIS.items()}
    keep: dict[str, list] = {name: [] for name in AOIS}

    print(f"Streaming {SEA_TABLE} ({block['rows']:,} rows) for {len(AOIS)} AOIs...")
    t0 = time.time()
    n = 0
    batch: list[list[str]] = []

    def flush() -> None:
        nonlocal n
        if not batch:
            return
        geoms = shapely.from_wkb([bytes.fromhex(r[gi]) for r in batch])
        b = shapely.bounds(geoms)
        for name, bb in boxes.items():
            hit = np.where((b[:, 0] <= bb[2]) & (b[:, 2] >= bb[0])
                           & (b[:, 1] <= bb[3]) & (b[:, 3] >= bb[1]))[0]
            keep[name].extend(geoms[i] for i in hit)
        n += len(batch)
        batch.clear()

    for line in iter_raw_lines(dump, block):
        batch.append(line.decode("utf-8").split("\t"))
        if len(batch) >= 2000:
            flush()
    flush()
    print(f"  scanned {n:,} rows in {time.time() - t0:.0f}s")

    if CACHE.exists():
        CACHE.unlink()
    for name, geoms in keep.items():
        print(f"  {name:<12} {len(geoms):>6} mean-high-water polygons")
        gpd.GeoDataFrame(geometry=geoms or [shapely.Polygon()], crs=25833).to_file(
            CACHE, layer=name, driver="GPKG")
    print(f"  -> {CACHE}")


# ── Phase 2: the real check ─────────────────────────────────────────────────

def landuse_on_grid(bb, transform, shape) -> np.ndarray:
    """Copernicus land use on the target lon/lat grid, nearest-neighbour.

    Mirrors validation.read_permanent_water_mask(): windowed read of the source
    raster, then rasterio.warp.reproject with Resampling.nearest (never
    interpolate a categorical raster).
    """
    with rasterio.open(LANDUSE) as src:
        win = rasterio.windows.from_bounds(*bb, transform=src.transform)
        win = win.round_offsets().round_lengths()
        pad = 2
        win = rasterio.windows.Window(
            max(win.col_off - pad, 0), max(win.row_off - pad, 0),
            win.width + 2 * pad, win.height + 2 * pad)
        src_nodata = src.nodata if src.nodata is not None else 255
        arr = src.read(1, window=win, boundless=True,
                       fill_value=src_nodata).astype("float64")
        src_transform = src.window_transform(win)
        src_crs = src.crs

    dst = np.full(shape, -1.0, dtype="float64")
    reproject(source=arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs="EPSG:4326",
              src_nodata=src_nodata, dst_nodata=-1.0,
              resampling=Resampling.nearest)
    return dst


def check() -> None:
    if not SURGE_GPKG.exists():
        raise SystemExit(f"missing {SURGE_GPKG} - run preparation/convert_norway_stormflo.py first")
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} - run this script with --extract first")

    print(f"Copernicus permanent-water codes: {PERMANENT_WATER_CODES}")
    print(f"Grid: EPSG:4326 @ {RES_Y_DEG * 3600:.0f}\" lat x 1\"/cos(lat) lon "
          f"(~{RES_Y_DEG * 111320:.0f} m, the model waterdepth grid)\n")
    rows = []

    for name, bb in AOIS.items():
        clip = utm_box(bb)
        print("=" * 96)
        print(f"AOI {name}   lon/lat {bb}")
        print("=" * 96)

        surge = gpd.read_file(SURGE_GPKG, layer=SURGE_LAYER, bbox=clip.bounds).clip(clip)
        sea = gpd.read_file(CACHE, layer=name).clip(clip)
        if surge.empty:
            print("  no surge polygons here - skipped\n")
            continue

        surge_u = surge.union_all() if hasattr(surge, "union_all") else surge.unary_union
        sea_u = (sea.union_all() if hasattr(sea, "union_all") else sea.unary_union) \
            if not sea.empty else shapely.Polygon()

        sea_part = surge_u.intersection(sea_u)      # the "sea" inside the raw polygon
        land_part = surge_u.difference(sea_u)       # genuine land inundation
        print(f"  raw stormflo200ar polygon : {surge_u.area / 1e6:9.2f} km2 "
              f"({len(surge):,} features)")
        print(f"    of which sea (per MHW)  : {sea_part.area / 1e6:9.2f} km2 "
              f"({sea_part.area / surge_u.area * 100:5.2f}%)")
        print(f"    of which land           : {land_part.area / 1e6:9.2f} km2 "
              f"({land_part.area / surge_u.area * 100:5.2f}%)")

        # Target grid in EPSG:4326 over the AOI, at the model's ~30 m spacing:
        # 1" in latitude, 1"/cos(lat) in longitude (see RES_Y_DEG).
        lat_mid = (bb[1] + bb[3]) / 2.0
        res_x = RES_Y_DEG / np.cos(np.radians(lat_mid))
        w = int(np.ceil((bb[2] - bb[0]) / res_x))
        h = int(np.ceil((bb[3] - bb[1]) / RES_Y_DEG))
        transform = from_origin(bb[0], bb[3], res_x, RES_Y_DEG)
        grid_bb = (bb[0], bb[3] - h * RES_Y_DEG, bb[0] + w * res_x, bb[3])

        to_wgs = gpd.GeoSeries([sea_part, land_part, surge_u], crs=25833).to_crs(4326)
        sea_mask = rasterize([(to_wgs.iloc[0], 1)], out_shape=(h, w), transform=transform,
                             fill=0, all_touched=False).astype(bool)
        land_mask = rasterize([(to_wgs.iloc[1], 1)], out_shape=(h, w), transform=transform,
                              fill=0, all_touched=False).astype(bool)

        lu = landuse_on_grid(grid_bb, transform, (h, w))
        water = np.isin(lu, np.asarray(PERMANENT_WATER_CODES))

        n_sea = int(sea_mask.sum())
        n_land = int(land_mask.sum())
        sea_masked = int((sea_mask & water).sum())
        land_masked = int((land_mask & water).sum())
        pct_sea = sea_masked / n_sea * 100 if n_sea else float("nan")
        pct_land = land_masked / n_land * 100 if n_land else float("nan")

        print(f"  grid {h} x {w} px")
        print(f"    SEA cells   : {n_sea:>8,}  Copernicus permanent water: "
              f"{sea_masked:>8,}  -> {pct_sea:6.2f}%   (want HIGH)")
        print(f"    LAND cells  : {n_land:>8,}  Copernicus permanent water: "
              f"{land_masked:>8,}  -> {pct_land:6.2f}%   (want LOW - this is real")
        print(f"                                                                  "
              f"          benchmark area the mask would wrongly discard)")

        codes, counts = np.unique(lu[sea_mask], return_counts=True)
        top = sorted(zip(counts, codes), reverse=True)[:5]
        print("    sea-cell land-cover codes: "
              + ", ".join(f"{int(c)}={n / max(n_sea,1) * 100:.1f}%" for n, c in top))
        print()

        rows.append((name, surge_u.area / 1e6, sea_part.area / 1e6, land_part.area / 1e6,
                     n_sea, pct_sea, n_land, pct_land))

    print("=" * 96)
    print("SUMMARY - fraction of the raw polygon's SEA that Copernicus also calls permanent water")
    print("=" * 96)
    print(f"{'AOI':<13}{'raw km2':>10}{'sea km2':>10}{'land km2':>10}"
          f"{'sea px':>9}{'sea masked':>12}{'land px':>9}{'land masked':>13}")
    for r in rows:
        print(f"{r[0]:<13}{r[1]:>10.2f}{r[2]:>10.2f}{r[3]:>10.2f}"
              f"{r[4]:>9,}{r[5]:>11.2f}%{r[6]:>9,}{r[7]:>12.2f}%")
    if rows:
        tot_sea = sum(r[4] for r in rows)
        tot_sea_m = sum(r[4] * r[5] / 100 for r in rows)
        tot_land = sum(r[6] for r in rows)
        tot_land_m = sum(r[6] * r[7] / 100 for r in rows)
        print(f"\n  ALL AOIs: {tot_sea_m / tot_sea * 100:.2f}% of sea cells masked, "
              f"{tot_land_m / tot_land * 100:.2f}% of land cells masked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="phase 1: pull AOI subsets from the dump")
    a = ap.parse_args()
    if a.extract:
        extract_sea_layer()
    else:
        check()


if __name__ == "__main__":
    main()
