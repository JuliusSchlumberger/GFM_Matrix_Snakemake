"""Step 3: inspect the REAL DeltaDTM elevation around Fort-de-France / Lamentin
(the FRJ_TRI_LAMENTIN benchmark area) and the per-tile DEM the model actually ran on.

Checks: coverage, resolution, nodata, value range, and the elevation distribution
INSIDE the benchmark polygon (which directly determines whether a ~0.4 m COAST-RP
forcing could ever flood it).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
pd.set_option("display.width", 220)

BOX = (-61.15, 14.52, -60.93, 14.68)   # Fort-de-France bay + Lamentin plain

print("=== raw DeltaDTM source tiles covering Martinique ===")
for name in ["DeltaDTM_v1_1_N14W061.tif", "DeltaDTM_v1_1_N14W062.tif",
             "DeltaDTM_v1_1_N15W062.tif", "DeltaDTM_v1_1_N15W064.tif"]:
    p = ROOT / "inputs/DeltaDTM" / name
    if not p.exists():
        print(f"  {name}: MISSING"); continue
    with rasterio.open(p) as s:
        a = s.read(1)
        valid = a[a != s.nodata] if s.nodata is not None else a.ravel()
        print(f"  {name}: bounds={tuple(round(v,4) for v in s.bounds)} shape={a.shape} "
              f"res=({s.res[0]:.8f},{s.res[1]:.8f}) nodata={s.nodata} dtype={a.dtype}")
        print(f"      valid px={valid.size} ({100*valid.size/a.size:.2f}%) "
              f"min={float(valid.min()) if valid.size else float('nan'):.3f} "
              f"max={float(valid.max()) if valid.size else float('nan'):.3f}")

print("\n=== DeltaDTM VRT mosaic ===")
vrt = ROOT / "inputs/DeltaDTM/deltadtm.vrt"
with rasterio.open(vrt) as s:
    print(f"  {vrt}\n  bounds={s.bounds}\n  res={s.res} nodata={s.nodata} dtype={s.dtypes}")
    w = from_bounds(*BOX, transform=s.transform)
    dem_vrt = s.read(1, window=w).astype("float64")
    vrt_transform = s.window_transform(w)
    vrt_nodata = s.nodata
print(f"  window over Fort-de-France box {BOX}: shape={dem_vrt.shape}")
valid = dem_vrt[dem_vrt != vrt_nodata]
print(f"  valid={valid.size}/{dem_vrt.size} ({100*valid.size/dem_vrt.size:.1f}%) "
      f"min={valid.min():.3f} max={valid.max():.3f} median={np.median(valid):.3f}")

# --- per-tile model DEM (what the simulation actually used) ------------------
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "src"))
from rasters import decode_dem_cm  # noqa: E402
from merge import decode_waterdepth_array, AQUEDUCT_NODATA  # noqa: E402

for tid in (2022, 2031):
    p = ROOT / "model_outputs" / str(tid) / "inputs" / "dem.tif"
    with rasterio.open(p) as s:
        print(f"\n=== tile {tid} model DEM: {p}")
        print(f"  bounds={tuple(round(v,4) for v in s.bounds)} shape=({s.height},{s.width}) "
              f"res=({s.res[0]:.8f},{s.res[1]:.8f}) crs={s.crs} nodata={s.nodata} dtype={s.dtypes[0]}")
        w = from_bounds(*BOX, transform=s.transform)
        raw = s.read(1, window=w)
        nod = s.nodata
    dem = decode_dem_cm(raw.astype("float64"))
    nodmask = raw == nod
    v = dem[~nodmask]
    print(f"  Fort-de-France window: shape={raw.shape} nodata px={int(nodmask.sum())} "
          f"({100*nodmask.mean():.1f}%)")
    if v.size:
        print(f"  elevation (m): min={v.min():.3f} p1={np.percentile(v,1):.3f} "
              f"median={np.median(v):.3f} max={v.max():.3f}")
        for t in (0.0, 0.3, 0.4, 0.5, 1.0, 2.0):
            print(f"    fraction of valid cells with elev <= {t:4.1f} m: {(v <= t).mean():.4f}")

    mp = ROOT / "model_outputs" / str(tid) / "inputs" / "mask.tif"
    with rasterio.open(mp) as s:
        mw = s.read(1, window=from_bounds(*BOX, transform=s.transform))
    vals, cnt = np.unique(mw, return_counts=True)
    print(f"  mask.tif values in window: {dict(zip(vals.tolist(), cnt.tolist()))}")

# --- elevation INSIDE the benchmark polygon ----------------------------------
bench = gpd.read_file(HERE / "benchmark_FRJ_martinique.gpkg")
geom = [g for g in bench.geometry.buffer(0)]
print("\n=== DeltaDTM elevation INSIDE the FRJ_TRI_LAMENTIN benchmark polygons ===")
with rasterio.open(vrt) as s:
    arr, tr = rio_mask(s, geom, crop=True, filled=True, nodata=s.nodata)
    nod = s.nodata
a = arr[0].astype("float64")
inpoly = a[a != nod]
print(f"  cells inside polygon: total={a.size} valid={inpoly.size} "
      f"nodata={a.size - inpoly.size} ({100*(a.size-inpoly.size)/a.size:.1f}%)")
print(f"  elevation (m): min={inpoly.min():.3f} p5={np.percentile(inpoly,5):.3f} "
      f"p25={np.percentile(inpoly,25):.3f} median={np.median(inpoly):.3f} "
      f"p75={np.percentile(inpoly,75):.3f} max={inpoly.max():.3f}")
rows = []
for t in (0.0, 0.2, 0.3, 0.334, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
    rows.append({"elev_threshold_m": t, "frac_benchmark_cells_below": round(float((inpoly <= t).mean()), 4)})
ecdf = pd.DataFrame(rows)
print(ecdf.to_string(index=False))
ecdf.to_csv(HERE / "benchmark_polygon_elevation_ecdf.csv", index=False)
np.save(HERE / "_benchmark_elevations.npy", inpoly)

# --- model water depth inside the benchmark polygon --------------------------
print("\n=== model RP100/SLR_0 water depth inside the benchmark polygon ===")
for tid in (2022, 2031):
    wp = ROOT / "model_outputs" / str(tid) / "results" / "waterdepth_RP100_SLR_0.tif"
    with rasterio.open(wp) as s:
        try:
            arr, _ = rio_mask(s, geom, crop=True, filled=True, nodata=s.nodata)
        except ValueError as e:
            print(f"  tile {tid}: {e}"); continue
        nod = s.nodata
    d = decode_waterdepth_array(arr[0].astype("float64"))
    fin = d[(d != AQUEDUCT_NODATA) & np.isfinite(d)]
    print(f"  tile {tid}: cells={d.size} computed={fin.size} "
          f"nodata/uncomputed={d.size - fin.size}")
    if fin.size:
        print(f"    depth (m): min={fin.min():.4f} median={np.median(fin):.4f} "
              f"max={fin.max():.4f}; frac>0.10m = {(fin > 0.10).mean():.5f}; "
              f"n>0.10m = {int((fin > 0.10).sum())}")

# --- merged chunk raster inside the benchmark polygon ------------------------
mch = ROOT / "merged_results/chunks/waterdepth_N10W065_RP100_SLR_0.tif"
print(f"\n=== merged chunk {mch.name} inside the benchmark polygon ===")
with rasterio.open(mch) as s:
    print(f"  bounds={tuple(round(v,4) for v in s.bounds)} shape=({s.height},{s.width}) "
          f"res={s.res} nodata={s.nodata} dtype={s.dtypes[0]}")
    arr, _ = rio_mask(s, geom, crop=True, filled=True, nodata=s.nodata)
    nod = s.nodata
d = decode_waterdepth_array(arr[0].astype("float64"))
fin = d[(d != AQUEDUCT_NODATA) & np.isfinite(d)]
print(f"  cells={d.size} computed={fin.size} uncomputed={d.size - fin.size}")
if fin.size:
    print(f"  depth (m): min={fin.min():.4f} median={np.median(fin):.4f} max={fin.max():.4f}; "
          f"n>0.10m={int((fin > 0.10).sum())} ({(fin > 0.10).mean():.5f})")
