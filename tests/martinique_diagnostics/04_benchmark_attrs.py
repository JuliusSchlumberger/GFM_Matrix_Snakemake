"""Step 2c/4a: what does the FRJ_TRI_LAMENTIN benchmark actually represent
(marine submersion vs river overflow), and is the MDT correction a no-op globally?
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
BENCH = ROOT / "inputs/validation/FRA/n_inondable_03_02moy_s.shp"

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

b = gpd.read_file(BENCH)
frj = b[b.id_tri.astype(str).str.startswith("FRJ")].copy()
print(f"FRJ rows: {len(frj)}")
for col in ["typ_inond", "typ_inond2", "scenario", "cours_deau", "est_ref", "id_s_inond"]:
    if col in frj.columns:
        print(f"\n--- {col} value counts (FRJ / Martinique) ---")
        print(frj[col].value_counts(dropna=False).to_string())

print("\n--- typ_inond value counts for the WHOLE France benchmark (context) ---")
print(b.typ_inond.value_counts(dropna=False).to_string())

# area per typ_inond for Martinique, in km2 (equal-area projection)
frj_ea = frj.to_crs(6933)
frj_ea["area_km2"] = frj_ea.geometry.area / 1e6
print("\n--- FRJ area by typ_inond (km2, EPSG:6933) ---")
print(frj_ea.groupby("typ_inond")["area_km2"].agg(["count", "sum"]).round(3).to_string())
frj_ea.drop(columns="geometry").to_csv(HERE / "benchmark_FRJ_attributes.csv", index=False)

# metropole comparison: which typ_inond dominates there
met = b[~b.id_tri.astype(str).str.startswith(("FRJ", "FRG", "FRD", "FRY", "FRM"))]
print(f"\nmetropole-ish rows: {len(met)}")
print(met.typ_inond.value_counts(dropna=False).to_string())

# ---- is the MDT correction a global no-op? ----------------------------------
raw = xr.open_dataset(ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc")
proc = xr.open_dataset(ROOT / "processed_inputs/WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc")
rdf = pd.DataFrame({
    "lon": np.round(raw.station_x_coordinate.values, 4),
    "lat": np.round(raw.station_y_coordinate.values, 4),
    "raw": raw.storm_tide_rp_0100.values.astype(float),
})
pdf = pd.DataFrame({
    "lon": np.round(proc.station_x_coordinate.values, 4),
    "lat": np.round(proc.station_y_coordinate.values, 4),
    "proc": proc["COAST-RP_EWL_RP100_SLR_0"].values.astype(float),
})
j = rdf.merge(pdf, on=["lon", "lat"], how="inner")
j["diff"] = j.raw - j.proc
print(f"\n=== global raw-minus-processed RP100 over {len(j)} matched stations ===")
print(j["diff"].describe().round(5).to_string())
print(f"  n |diff| > 0.01 m: {(j['diff'].abs() > 0.01).sum()}")
print(f"  n |diff| > 0.001 m: {(j['diff'].abs() > 0.001).sum()}")
