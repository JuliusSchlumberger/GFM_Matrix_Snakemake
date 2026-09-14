"""Would DeltaDTM's own land/ocean/lake/river mask beat Copernicus 100 m as the
permanent-water mask for flood-extent validation in Norway?

09_landuse_mask_check.py measured the real problem: validation.py's
``read_permanent_water_mask`` flags Copernicus Global Land Cover codes 80/200 as
permanent water and drops those cells from the evaluation domain entirely - but
Copernicus is ~3.57" (~110 m N-S) while the validation grid is ~1" (~30 m), so a
single coarse cell straddling shoreline kills every 30 m cell inside it.  On the
five Norwegian AOIs that discarded roughly half of the genuinely-dry benchmark
inundation strip.

DeltaDTM - the DEM this whole pipeline is built from - ships its OWN categorical
mask (0=land, 1=ocean, 2=lake, 3=river; see config.yml tile_generation.ocean_code
/river_code and preparation/build_deltadtm_vrt.py) at 1" N-S and 1-3" E-W
depending on latitude, i.e. at or near the validation grid's own spacing.  This
script measures, on the SAME AOIs / SAME grid / SAME nearest-neighbour reproject
path as 09, how the two sources actually compare on the two metrics that matter:

    SEA  cells masked  (want HIGH - the whole point is to drop permanent water)
    LAND cells masked  (want LOW  - this is real benchmark area wrongly discarded)

Three mask sources are compared, plus their union/intersection:

    copernicus      land_use codes (80, 200), exactly as validation.py does it
    deltadtm_vrt    the production inputs/DeltaDTM_masks/deltadtm_mask.vrt, i.e.
                    what a new catalog entry would really read today
    deltadtm_tiles  an in-memory VRT of just this AOI's native tiles built with
                    resolution="highest" - what build_deltadtm_vrt.py's current
                    (fixed) code produces, as an upper bound on the VRT path

Reuses 09's cached mean-high-water extract (norway_landusecheck_mhw.gpkg), so run
``python 09_landuse_mask_check.py --extract`` first if that file is missing.

Usage:
    python 10_deltadtm_mask_vs_landuse.py
    python 10_deltadtm_mask_vs_landuse.py --aoi bergen --aoi naeroyfjord
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
import shapely
from osgeo import gdal
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

gdal.UseExceptions()

HERE = Path(__file__).parent
NOR = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\validation\NOR")
LANDUSE = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\Copernicus\Copernicus_LandUse.tif")
DDTM_MASK_DIR = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\DeltaDTM_masks")
DDTM_MASK_VRT = DDTM_MASK_DIR / "deltadtm_mask.vrt"
SURGE_GPKG = NOR / "stormflo200ar_klimaarna.gpkg"
SURGE_LAYER = "stormflo200ar_klimaarna"
CACHE = HERE / "norway_landusecheck_mhw.gpkg"
OUT_JSON = HERE / "norway_deltadtm_vs_landuse.json"

PERMANENT_WATER_CODES = (80, 200)          # Copernicus, validation.py
DDTM_WATER_CODES = (1, 2, 3)               # ocean, lake, river
DDTM_NODATA = 255

RES_Y_DEG = 1.0 / 3600.0                   # validation grid: 1" lat (see 09)

AOIS = {
    "oslofjord":   (10.55, 59.80, 10.90, 59.98),
    "bergen":      (5.20, 60.33, 5.42, 60.45),
    "naeroyfjord": (6.80, 60.85, 7.30, 61.15),
    "lofoten":     (13.80, 68.10, 14.40, 68.40),
    "hammerfest":  (23.40, 70.50, 24.00, 70.80),
}

_TO_UTM = Transformer.from_crs(4326, 25833, always_xy=True)


def utm_box(bb):
    x0, y0 = _TO_UTM.transform(bb[0], bb[1])
    x1, y1 = _TO_UTM.transform(bb[2], bb[3])
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# ── raster reads (all mirror validation.read_permanent_water_mask) ───────────

def _windowed_nearest(path, bb, transform, shape, fallback_nodata):
    """Windowed read of `path` over `bb`, nearest-neighbour reprojected onto the
    target grid - the same two-step path read_permanent_water_mask uses (bbox
    subset via the catalog, then rasterio.warp.reproject with Resampling.nearest).
    """
    with rasterio.open(path) as src:
        win = rasterio.windows.from_bounds(*bb, transform=src.transform)
        win = win.round_offsets().round_lengths()
        pad = 2
        win = rasterio.windows.Window(
            max(win.col_off - pad, 0), max(win.row_off - pad, 0),
            win.width + 2 * pad, win.height + 2 * pad)
        nodata = src.nodata if src.nodata is not None else fallback_nodata
        arr = src.read(1, window=win, boundless=True, fill_value=nodata).astype("float64")
        src_transform = src.window_transform(win)
        src_crs = src.crs
        native_res = src.res

    dst = np.full(shape, -1.0, dtype="float64")
    reproject(source=arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs="EPSG:4326",
              src_nodata=nodata, dst_nodata=-1.0,
              resampling=Resampling.nearest)
    return dst, native_res


def _tiles_for(bb):
    """Native DeltaDTM mask tiles (1x1 deg, DeltaDTM_v1_1_NxxEyyy.tif) covering bb."""
    paths = []
    for lat in range(int(np.floor(bb[1])), int(np.ceil(bb[3]))):
        for lon in range(int(np.floor(bb[0])), int(np.ceil(bb[2]))):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            p = DDTM_MASK_DIR / f"DeltaDTM_v1_1_{ns}{ew}.tif"
            if p.exists():
                paths.append(str(p))
    return paths


def _native_vrt(bb, tmp_path):
    """In-memory-ish VRT of just this AOI's native tiles at resolution='highest',
    i.e. exactly what the current (fixed) build_deltadtm_vrt.py produces locally.
    """
    tifs = _tiles_for(bb)
    if not tifs:
        return None, []
    opts = gdal.BuildVRTOptions(resampleAlg="nearest", resolution="highest", VRTNodata=255)
    ds = gdal.BuildVRT(str(tmp_path), tifs, options=opts)
    ds.FlushCache()
    ds = None
    return str(tmp_path), tifs


# ── metrics ─────────────────────────────────────────────────────────────────

def _score(water, sea_mask, land_mask):
    n_sea, n_land = int(sea_mask.sum()), int(land_mask.sum())
    sm = int((sea_mask & water).sum())
    lm = int((land_mask & water).sum())
    return {
        "sea_px": n_sea, "sea_masked": sm,
        "sea_masked_pct": (sm / n_sea * 100) if n_sea else float("nan"),
        "land_px": n_land, "land_masked": lm,
        "land_masked_pct": (lm / n_land * 100) if n_land else float("nan"),
    }


def run(selected):
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} - run 09_landuse_mask_check.py --extract first")

    print(f"Copernicus permanent-water codes : {PERMANENT_WATER_CODES}")
    print(f"DeltaDTM mask water codes        : {DDTM_WATER_CODES} (0=land, 255=nodata)")
    print(f"Grid: EPSG:4326 @ 1\" lat x 1\"/cos(lat) lon (~30 m, the model waterdepth grid)\n")

    results = {}
    for name, bb in AOIS.items():
        if selected and name not in selected:
            continue
        clip = utm_box(bb)
        print("=" * 100)
        print(f"AOI {name}   lon/lat {bb}")
        print("=" * 100)

        surge = gpd.read_file(SURGE_GPKG, layer=SURGE_LAYER, bbox=clip.bounds).clip(clip)
        sea = gpd.read_file(CACHE, layer=name).clip(clip)
        if surge.empty:
            print("  no surge polygons here - skipped\n")
            continue
        surge_u = surge.union_all() if hasattr(surge, "union_all") else surge.unary_union
        sea_u = (sea.union_all() if hasattr(sea, "union_all") else sea.unary_union) if not sea.empty else shapely.Polygon()
        sea_part = surge_u.intersection(sea_u)
        land_part = surge_u.difference(sea_u)

        lat_mid = (bb[1] + bb[3]) / 2.0
        res_x = RES_Y_DEG / np.cos(np.radians(lat_mid))
        w = int(np.ceil((bb[2] - bb[0]) / res_x))
        h = int(np.ceil((bb[3] - bb[1]) / RES_Y_DEG))
        transform = from_origin(bb[0], bb[3], res_x, RES_Y_DEG)
        grid_bb = (bb[0], bb[3] - h * RES_Y_DEG, bb[0] + w * res_x, bb[3])

        to_wgs = gpd.GeoSeries([sea_part, land_part], crs=25833).to_crs(4326)
        sea_mask = rasterize([(to_wgs.iloc[0], 1)], out_shape=(h, w), transform=transform,
                             fill=0, all_touched=False).astype(bool)
        land_mask = rasterize([(to_wgs.iloc[1], 1)], out_shape=(h, w), transform=transform,
                              fill=0, all_touched=False).astype(bool)

        # cell area on this grid, for km2 reporting
        cell_m2 = (res_x * 111320.0 * np.cos(np.radians(lat_mid))) * (RES_Y_DEG * 110574.0)

        print(f"  grid {h} x {w} px   cell ~{cell_m2:.0f} m2")
        print(f"  raw stormflo200ar : {surge_u.area/1e6:8.2f} km2  "
              f"sea {sea_part.area/1e6:7.2f}  land {land_part.area/1e6:7.2f}")

        lu, lu_res = _windowed_nearest(LANDUSE, grid_bb, transform, (h, w), 255)
        cop_water = np.isin(lu, np.asarray(PERMANENT_WATER_CODES))

        dv, dv_res = _windowed_nearest(DDTM_MASK_VRT, grid_bb, transform, (h, w), DDTM_NODATA)
        ddtm_vrt_water = np.isin(dv, np.asarray(DDTM_WATER_CODES))

        tmp = HERE / f"_tmp_{name}_native.vrt"
        nvrt, tifs = _native_vrt(grid_bb, tmp)
        if nvrt:
            dn, dn_res = _windowed_nearest(nvrt, grid_bb, transform, (h, w), DDTM_NODATA)
            tmp.unlink(missing_ok=True)
        else:
            dn, dn_res = np.full((h, w), -1.0), (float("nan"), float("nan"))
        ddtm_tile_water = np.isin(dn, np.asarray(DDTM_WATER_CODES))
        ddtm_ocean_only = (dn == 1)

        native_res = []
        for t in tifs:
            with rasterio.open(t) as s:
                native_res.append((Path(t).stem[-7:], round(s.res[0] * 3600, 3), round(s.res[1] * 3600, 3)))

        print(f"  source native res (arcsec): copernicus x={lu_res[0]*3600:.4f} y={lu_res[1]*3600:.4f}"
              f"  |  mask VRT x={dv_res[0]*3600:.4f} y={dv_res[1]*3600:.4f}"
              f"  |  mask tiles {native_res}")

        # DeltaDTM nodata / coverage inside the benchmark footprint
        nod = int(((dn == DDTM_NODATA) | (dn < 0)) [sea_mask | land_mask].sum())
        print(f"  DeltaDTM mask nodata inside benchmark footprint: {nod:,} px "
              f"({nod / max(int((sea_mask|land_mask).sum()),1)*100:.2f}%)")

        variants = {
            "copernicus_80_200": cop_water,
            "deltadtm_vrt_123": ddtm_vrt_water,
            "deltadtm_tiles_123": ddtm_tile_water,
            "deltadtm_tiles_ocean_only_1": ddtm_ocean_only,
            "union_cop_or_ddtm": cop_water | ddtm_tile_water,
            "intersection_cop_and_ddtm": cop_water & ddtm_tile_water,
        }
        aoi_res = {"bbox": bb, "grid": [h, w], "cell_m2": cell_m2,
                   "native_res_arcsec": {"copernicus": [lu_res[0]*3600, lu_res[1]*3600],
                                         "deltadtm_vrt": [dv_res[0]*3600, dv_res[1]*3600],
                                         "deltadtm_tiles": native_res},
                   "ddtm_nodata_px_in_footprint": nod,
                   "variants": {}}

        print(f"  {'variant':<30}{'sea masked':>14}{'land masked':>16}{'land kept km2':>16}")
        for k, m in variants.items():
            s = _score(m, sea_mask, land_mask)
            aoi_res["variants"][k] = s
            kept_km2 = (s["land_px"] - s["land_masked"]) * cell_m2 / 1e6
            print(f"  {k:<30}{s['sea_masked_pct']:>13.2f}%{s['land_masked_pct']:>15.2f}%"
                  f"{kept_km2:>16.3f}")
        print()
        results[name] = aoi_res

    # ── overall ─────────────────────────────────────────────────────────────
    if results:
        print("=" * 100)
        print("ALL AOIs COMBINED (pixel-weighted)")
        print("=" * 100)
        keys = list(next(iter(results.values()))["variants"].keys())
        print(f"  {'variant':<30}{'sea masked':>14}{'land masked':>16}"
              f"{'land kept km2':>16}{'vs copernicus':>16}")
        base_kept = None
        for k in keys:
            sp = sum(r["variants"][k]["sea_px"] for r in results.values())
            sm = sum(r["variants"][k]["sea_masked"] for r in results.values())
            lp = sum(r["variants"][k]["land_px"] for r in results.values())
            lm = sum(r["variants"][k]["land_masked"] for r in results.values())
            kept = sum((r["variants"][k]["land_px"] - r["variants"][k]["land_masked"])
                       * r["cell_m2"] / 1e6 for r in results.values())
            if base_kept is None:
                base_kept = kept
            print(f"  {k:<30}{sm/sp*100:>13.2f}%{lm/lp*100:>15.2f}%{kept:>16.3f}"
                  f"{(kept - base_kept):>+15.3f}")
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"\n  -> {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", action="append", default=[])
    a = ap.parse_args()
    run(set(a.aoi))


if __name__ == "__main__":
    main()
