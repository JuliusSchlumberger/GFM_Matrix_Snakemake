"""Step 5b: two remaining cross-checks.

(1) The whole COAST-RP return-period curve at Martinique vs Guadeloupe vs France -
    a flat RP2..RP1000 curve is the signature of a tide-only / surge-free signal.
(2) Vertical datum: does the tile's own dem.tif differ from the raw DeltaDTM VRT
    (i.e. how big is the EGM2008 -> GOCO06s geoid correction that was applied at
    Martinique)?  A wrongly-signed or oversized correction would bias the DEM high
    and suppress flooding independently of the forcing.
(3) What uniform still-water level would the benchmark polygon imply?
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from rasters import decode_dem_cm  # noqa: E402

ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

# ---------------------------------------------------------------- (1) RP curve
raw = xr.open_dataset(ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc")
x = raw.station_x_coordinate.values; y = raw.station_y_coordinate.values
RPS = ["storm_tide_rp_0001", "storm_tide_rp_0002", "storm_tide_rp_0005",
       "storm_tide_rp_0010", "storm_tide_rp_0025", "storm_tide_rp_0050",
       "storm_tide_rp_0100", "storm_tide_rp_0250", "storm_tide_rp_0500",
       "storm_tide_rp_1000"]
BOXES = {"Martinique": (-61.35, -60.75, 14.35, 14.95),
         "Guadeloupe": (-61.85, -61.00, 15.80, 16.55),
         "St Lucia": (-61.10, -60.85, 13.70, 14.15),
         "FR Gironde": (-1.60, -0.90, 45.20, 45.90),
         "FR Normandy": (-1.90, -1.10, 49.10, 49.80)}
rows = []
for name, (a, b, c, d) in BOXES.items():
    m = (x >= a) & (x <= b) & (y >= c) & (y <= d)
    rec = {"region": name, "n_stations": int(m.sum())}
    for v in RPS:
        vals = raw[v].values[m]
        rec[v.replace("storm_tide_rp_", "RP")] = round(float(np.nanmedian(vals)), 3)
    rec["RP1000_minus_RP2"] = round(rec["RP1000"] - rec["RP0002"], 3)
    rows.append(rec)
curve = pd.DataFrame(rows)
print("=== median raw COAST-RP storm-tide return-period curve (m above local MSL) ===")
print(curve.to_string(index=False))
curve.to_csv(HERE / "coastrp_return_period_curves.csv", index=False)

sid = raw.station_id.values
m = (x >= -61.35) & (x <= -60.75) & (y >= 14.35) & (y <= 14.95)
print(f"\nCOAST-RP station_id values around Martinique ({int(m.sum())}):")
for s_, xx, yy, v in zip(sid[m], x[m], y[m], raw.storm_tide_rp_0100.values[m]):
    print(f"  {s_}  ({xx:.3f},{yy:.3f})  RP100={v:.3f} m")

# ------------------------------------------------------- (2) vertical datum
BOX = (-61.16, 14.52, -60.92, 14.70)
with rasterio.open(ROOT / "inputs/DeltaDTM/deltadtm.vrt") as s:
    w = from_bounds(*BOX, transform=s.transform).round_offsets().round_lengths()
    vrt = s.read(1, window=w).astype(np.float64)
    vrt_tr = s.window_transform(w); vrt_nod = s.nodata
with rasterio.open(ROOT / "model_outputs/2022/inputs/dem.tif") as s:
    w2 = from_bounds(*BOX, transform=s.transform).round_offsets().round_lengths()
    tile = decode_dem_cm(s.read(1, window=w2).astype(np.float64))
    tile_raw = s.read(1, window=w2); tile_nod = s.nodata
    tile_tr = s.window_transform(w2)
print(f"\n=== DEM comparison over {BOX} ===")
print(f"  raw DeltaDTM VRT window: shape={vrt.shape} transform_origin=({vrt_tr.c:.6f},{vrt_tr.f:.6f})")
print(f"  tile 2022 dem.tif window: shape={tile.shape} transform_origin=({tile_tr.c:.6f},{tile_tr.f:.6f})")
if vrt.shape == tile.shape:
    both = (vrt != vrt_nod) & (tile_raw != tile_nod)
    diff = tile[both] - vrt[both]
    print(f"  cells compared: {int(both.sum())}")
    print(f"  tile_dem - raw_DeltaDTM (m): min={diff.min():.4f} p5={np.percentile(diff,5):.4f} "
          f"median={np.median(diff):.4f} p95={np.percentile(diff,95):.4f} max={diff.max():.4f}")
    print("  (this is the applied EGM2008 -> GOCO06s geoid correction, plus cm rounding)")
    # also: cells present in raw but nodata in tile (gap-filled?) and vice versa
    print(f"  raw-valid & tile-nodata: {int(((vrt != vrt_nod) & (tile_raw == tile_nod)).sum())}")
    print(f"  raw-nodata & tile-valid: {int(((vrt == vrt_nod) & (tile_raw != tile_nod)).sum())}"
          "   (DEM gap-fill / ocean zeroing)")
else:
    print("  window shapes differ - grids not directly comparable")

# --------------------------------------- (3) still-water level the benchmark implies
bench = gpd.read_file(HERE / "benchmark_FRJ_martinique.gpkg")
geoms = list(bench.geometry.buffer(0))
with rasterio.open(ROOT / "inputs/DeltaDTM/deltadtm.vrt") as s:
    bb = bench.total_bounds
    w = from_bounds(bb[0]-0.005, bb[1]-0.005, bb[2]+0.005, bb[3]+0.005,
                    transform=s.transform).round_offsets().round_lengths()
    a = s.read(1, window=w).astype(np.float64); tr = s.window_transform(w); nod = s.nodata
inside = ~geometry_mask(geoms, out_shape=a.shape, transform=tr, invert=False)
elev = a[inside & (a != nod)]
print("\n=== still-water level implied by the FRJ_TRI_LAMENTIN benchmark ===")
print(f"  benchmark cells (30 m): {elev.size}")
for q in (0.5, 0.75, 0.9, 0.95):
    print(f"  to cover {q*100:.0f}% of the benchmark you need a still-water level of "
          f"{np.quantile(elev, q):.2f} m")
print(f"  the model was given 0.33-0.52 m (median 0.40 m)")
print(f"  fraction of benchmark below 0.52 m: {(elev <= 0.52).mean():.4f}")
