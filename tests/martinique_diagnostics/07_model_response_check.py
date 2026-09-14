"""Step 5: is the model's near-zero flooding on Martinique consistent with the
0.33-0.52 m forcing it was given, or is something else suppressing it?

For each of several tiles, compares:
  - how much LAND (mask==0) sits below the tile's own max station water level
  - how much the model actually flooded
so an "elevation says it should flood but the model didn't" gap would show up.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from merge import AQUEDUCT_NODATA, decode_waterdepth_array  # noqa: E402
from rasters import decode_dem_cm, decode_waterlevel_cm  # noqa: E402

ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
MODEL = ROOT / "model_outputs"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

TILES = {2022: "Martinique A", 2031: "Martinique B", 2429: "St Lucia",
         2185: "St Vincent", 1515: "Guadeloupe A", 1958: "Guadeloupe B",
         516: "FR Gironde", 1635: "FR Normandy"}

rows = []
for tid, label in TILES.items():
    d = MODEL / str(tid)
    wl_path = d / "inputs/boundaries_RP100_SLR_0.gpkg"
    wd_path = d / "results/waterdepth_RP100_SLR_0.tif"
    if not wd_path.exists():
        print(f"tile {tid}: no RP100/SLR_0 result"); continue
    st = gpd.read_file(wl_path)
    wl = decode_waterlevel_cm(st["SLR_0"].to_numpy()) if len(st) else np.array([np.nan])

    with rasterio.open(d / "inputs/dem.tif") as s:
        dem = decode_dem_cm(s.read(1).astype(np.float64)); dem_nod = s.nodata
        raw_dem = s.read(1)
        cell_km2 = abs(s.res[0] * s.res[1]) * (111.32 ** 2) * np.cos(np.radians(
            (s.bounds.bottom + s.bounds.top) / 2))
    with rasterio.open(d / "inputs/mask.tif") as s:
        msk = s.read(1)
    with rasterio.open(wd_path) as s:
        wd = decode_waterdepth_array(s.read(1))

    valid = raw_dem != dem_nod
    land = valid & (msk == 0)
    computed = (wd != AQUEDUCT_NODATA) & np.isfinite(wd)
    wl_max = float(np.nanmax(wl)); wl_med = float(np.nanmedian(wl))

    land_below = land & (dem <= wl_max)
    land_below_med = land & (dem <= wl_med)
    wet10 = computed & (wd > 0.10) & land
    wet0 = computed & (wd > 0) & land

    rows.append({
        "tile": tid, "label": label,
        "n_stations": len(st), "wl_med_m": round(wl_med, 3), "wl_max_m": round(wl_max, 3),
        "land_cells": int(land.sum()),
        "land_km2": round(float(land.sum()) * cell_km2, 1),
        "land_below_wl_max_km2": round(float(land_below.sum()) * cell_km2, 2),
        "land_below_wl_med_km2": round(float(land_below_med.sum()) * cell_km2, 2),
        "model_wet_gt0_km2": round(float(wet0.sum()) * cell_km2, 2),
        "model_wet_gt0.10_km2": round(float(wet10.sum()) * cell_km2, 2),
        "wet/below_wl_max": round(float(wet10.sum()) / max(float(land_below.sum()), 1), 3),
        "max_depth_m": round(float(wd[computed].max()) if computed.any() else np.nan, 3),
        "computed_frac_of_valid": round(float(computed[valid].mean()), 4),
    })
    print(f"tile {tid} ({label}) done")

df = pd.DataFrame(rows)
print("\n================= per-tile model response, RP100 / SLR_0 =================")
print(df.to_string(index=False))
df.to_csv(HERE / "model_response_per_tile.csv", index=False)

print("\nInterpretation key:")
print("  land_below_wl_max_km2  = land area whose DeltaDTM elevation is at or below the tile's")
print("                           HIGHEST forcing station - the absolute upper bound on flooding")
print("  model_wet_gt0.10_km2   = land area the model actually flooded above the 0.10 m threshold")
print("  wet/below_wl_max       = how much of that theoretical maximum the hydraulics delivered")
