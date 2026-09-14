"""Is the permanent-water mask's over-reach coarse-cell ALIASING, or a genuine
waterline disagreement?  Distance-from-shoreline profile, Copernicus vs DeltaDTM.

10_deltadtm_mask_vs_landuse.py measured the headline numbers: on five Norwegian
AOIs the Copernicus-based mask wrongly discards 51.2% of the benchmark's
genuinely-dry inundation strip, DeltaDTM's own ocean/lake/river mask 39.0%, and
DeltaDTM ocean-only 22.3%.  That says DeltaDTM is better, but not WHY - and the
"why" decides whether the remaining error is fixable at all.

Two competing explanations:

  (a) ALIASING.  A ~110 m Copernicus cell straddling the shoreline is labelled
      "open sea", so every 30 m validation cell inside it dies - including the
      dry half.  The damage should then be concentrated within ~one source cell
      of the real waterline and fall off sharply beyond it, and DeltaDTM (whose
      cells are ~25-31 m here, i.e. the validation grid's own size) should show
      the same shape but confined to a much narrower band.

  (b) WATERLINE DISAGREEMENT.  The source simply draws land/sea somewhere else
      than Norway's middelhoyvann (mean high water) reference - a datum/epoch
      difference, not a resolution one.  Then the masked fraction would stay
      high well inland and no amount of extra resolution would help.

This bins the benchmark's dry strip by euclidean distance from the mean-high-
water sea boundary and reports the wrongly-masked fraction per band, for both
sources.  It also writes a per-AOI diagnostic GeoTIFF (categorical: which source
masked each dry cell) for direct QGIS inspection.

Same AOIs / grid / nearest-neighbour reproject path as 09 and 10.

Usage:
    python 11_permwater_shoreline_profile.py
    python 11_permwater_shoreline_profile.py --aoi lofoten --write-raster
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location("cmp10", HERE / "10_deltadtm_mask_vs_landuse.py")
c10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c10)

# Distance bands in metres from the mean-high-water sea boundary.  Chosen around
# the two source cell sizes at these latitudes: DeltaDTM ~25-31 m, Copernicus
# ~57 m (E-W) x ~110 m (N-S), so its diagonal reach is ~124 m.
BANDS = [(0, 30), (30, 60), (60, 120), (120, 240), (240, 1e9)]


def run(selected, write_raster):
    print("Wrongly-masked fraction of the benchmark's DRY strip, by distance from")
    print("the middelhoyvann (mean-high-water) sea boundary.\n")
    print("Source cell sizes at these latitudes (ground m, from the real rasters):")
    print("  Copernicus 3.5714\" x 3.5714\"  ->  ~57 m E-W x ~110 m N-S at 60N, ~37 x 110 m at 70N")
    print("  DeltaDTM   1.5-3\" x 1\"        ->  ~24-31 m E-W x ~31 m N-S (deliberately ~30 m everywhere)\n")

    totals = {b: {"n": 0, "cop": 0, "ddtm": 0, "ddtm_ocean": 0} for b in BANDS}

    for name, bb in c10.AOIS.items():
        if selected and name not in selected:
            continue
        clip = c10.utm_box(bb)
        surge = gpd.read_file(c10.SURGE_GPKG, layer=c10.SURGE_LAYER, bbox=clip.bounds).clip(clip)
        sea = gpd.read_file(c10.CACHE, layer=name).clip(clip)
        if surge.empty:
            continue
        surge_u = surge.union_all() if hasattr(surge, "union_all") else surge.unary_union
        sea_u = (sea.union_all() if hasattr(sea, "union_all") else sea.unary_union) \
            if not sea.empty else shapely.Polygon()
        sea_part = surge_u.intersection(sea_u)
        land_part = surge_u.difference(sea_u)

        lat_mid = (bb[1] + bb[3]) / 2.0
        res_x = c10.RES_Y_DEG / np.cos(np.radians(lat_mid))
        w = int(np.ceil((bb[2] - bb[0]) / res_x))
        h = int(np.ceil((bb[3] - bb[1]) / c10.RES_Y_DEG))
        transform = from_origin(bb[0], bb[3], res_x, c10.RES_Y_DEG)
        grid_bb = (bb[0], bb[3] - h * c10.RES_Y_DEG, bb[0] + w * res_x, bb[3])

        to_wgs = gpd.GeoSeries([sea_part, land_part], crs=25833).to_crs(4326)
        sea_mask = rasterize([(to_wgs.iloc[0], 1)], out_shape=(h, w), transform=transform,
                             fill=0, all_touched=False).astype(bool)
        land_mask = rasterize([(to_wgs.iloc[1], 1)], out_shape=(h, w), transform=transform,
                              fill=0, all_touched=False).astype(bool)
        if not land_mask.any():
            continue

        # Ground cell size (metres) for an anisotropy-correct distance transform.
        mx = res_x * 111320.0 * np.cos(np.radians(lat_mid))
        my = c10.RES_Y_DEG * 110574.0
        dist = distance_transform_edt(~sea_mask, sampling=(my, mx))

        lu, _ = c10._windowed_nearest(c10.LANDUSE, grid_bb, transform, (h, w), 255)
        cop = np.isin(lu, np.asarray(c10.PERMANENT_WATER_CODES))
        tmp = HERE / f"_tmp_{name}_prof.vrt"
        nvrt, _ = c10._native_vrt(grid_bb, tmp)
        dn, _ = c10._windowed_nearest(nvrt, grid_bb, transform, (h, w), c10.DDTM_NODATA)
        tmp.unlink(missing_ok=True)
        ddtm = np.isin(dn, np.asarray(c10.DDTM_WATER_CODES))
        ddtm_o = (dn == 1)

        print("=" * 92)
        print(f"AOI {name}   dry-strip cells {int(land_mask.sum()):,}   "
              f"median distance from shoreline {np.median(dist[land_mask]):.0f} m   "
              f"90th pct {np.percentile(dist[land_mask], 90):.0f} m")
        print(f"  {'band (m from sea)':<22}{'cells':>10}{'copernicus':>13}"
              f"{'deltadtm':>12}{'ddtm ocean-only':>18}")
        for lo, hi in BANDS:
            b = land_mask & (dist >= lo) & (dist < hi)
            n = int(b.sum())
            totals[(lo, hi)]["n"] += n
            label = f"{lo}-{hi}" if hi < 1e8 else f"{lo}+"
            if not n:
                print(f"  {label:<22}{0:>10}")
                continue
            nc, nd, no = int((b & cop).sum()), int((b & ddtm).sum()), int((b & ddtm_o).sum())
            totals[(lo, hi)]["cop"] += nc
            totals[(lo, hi)]["ddtm"] += nd
            totals[(lo, hi)]["ddtm_ocean"] += no
            print(f"  {label:<22}{n:>10,}{nc/n*100:>12.1f}%{nd/n*100:>11.1f}%{no/n*100:>17.1f}%")
        print()

        if write_raster:
            # 0 = kept by both, 1 = copernicus only, 2 = deltadtm only, 3 = both,
            # 255 = not part of the dry strip.
            cat = np.full((h, w), 255, dtype="uint8")
            cat[land_mask] = (cop[land_mask].astype("uint8")
                              + 2 * ddtm[land_mask].astype("uint8"))
            out = HERE / f"norway_permwater_diff_{name}.tif"
            with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1,
                               dtype="uint8", crs="EPSG:4326", transform=transform,
                               nodata=255, compress="deflate") as dst:
                dst.write(cat, 1)
                dst.update_tags(
                    legend="0=kept by both, 1=masked by Copernicus only, "
                           "2=masked by DeltaDTM only, 3=masked by both, 255=outside dry strip")
            print(f"  -> {out}")

    print("=" * 92)
    print("ALL AOIs COMBINED")
    print(f"  {'band (m from sea)':<22}{'cells':>10}{'copernicus':>13}{'deltadtm':>12}{'ddtm ocean-only':>18}")
    for (lo, hi), t in totals.items():
        if not t["n"]:
            continue
        label = f"{lo}-{hi}" if hi < 1e8 else f"{lo}+"
        print(f"  {label:<22}{t['n']:>10,}{t['cop']/t['n']*100:>12.1f}%"
              f"{t['ddtm']/t['n']*100:>11.1f}%{t['ddtm_ocean']/t['n']*100:>17.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", action="append", default=[])
    ap.add_argument("--write-raster", action="store_true")
    a = ap.parse_args()
    run(set(a.aoi), a.write_raster)


if __name__ == "__main__":
    main()
