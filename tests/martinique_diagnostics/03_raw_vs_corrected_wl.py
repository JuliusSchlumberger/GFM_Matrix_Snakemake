"""Step 2b: is Martinique's ~0.4 m RP100 forcing what raw COAST-RP says, or is it
an artefact of the MDT (mean-dynamic-topography) vertical-datum correction?

Compares the RAW COAST-RP dataset's own RP100 values near Martinique against the
processed WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc values the pipeline actually
forces with, for the same stations - and does the same for a metropolitan-France
control region.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
RAW = ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc"
PROC = ROOT / "processed_inputs/WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc"

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

print("=== RAW COAST-RP ===", RAW)
raw = xr.open_dataset(RAW)
print(raw)

print("\n=== PROCESSED RP100 / SLR_0 ===", PROC)
proc = xr.open_dataset(PROC)
print(proc)

def region(ds, xname, yname, lon0, lon1, lat0, lat1):
    x = ds[xname].values; y = ds[yname].values
    return (x >= lon0) & (x <= lon1) & (y >= lat0) & (y <= lat1)

BOXES = {
    "Martinique":  (-61.35, -60.75, 14.35, 14.95),
    "Guadeloupe":  (-61.85, -61.00, 15.80, 16.55),
    "FR_Gironde":  (-1.60, -0.90, 45.20, 45.90),
    "FR_Channel":  (-1.90, -1.10, 49.10, 49.80),
}

# locate the RP100 variable in the raw file
rp_vars = [v for v in raw.data_vars if "100" in v]
print("\nraw vars containing '100':", rp_vars)
xr_name = [v for v in raw.variables if "x_coord" in v or v == "lon"][0]
yr_name = [v for v in raw.variables if "y_coord" in v or v == "lat"][0]
print("raw coord vars:", xr_name, yr_name)

rows = []
for name, (a, b, c, d) in BOXES.items():
    mr = region(raw, xr_name, yr_name, a, b, c, d)
    mp = region(proc, "station_x_coordinate", "station_y_coordinate", a, b, c, d)
    rec = {"region": name, "n_raw": int(mr.sum()), "n_proc": int(mp.sum())}
    for v in rp_vars:
        vals = raw[v].values[mr]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            rec[f"raw_{v}_min"] = round(float(vals.min()), 3)
            rec[f"raw_{v}_med"] = round(float(np.median(vals)), 3)
            rec[f"raw_{v}_max"] = round(float(vals.max()), 3)
    pv = proc["COAST-RP_EWL_RP100_SLR_0"].values[mp]
    pv = pv[np.isfinite(pv)]
    if len(pv):
        rec["proc_min"] = round(float(pv.min()), 3)
        rec["proc_med"] = round(float(np.median(pv)), 3)
        rec["proc_max"] = round(float(pv.max()), 3)
    rows.append(rec)

df = pd.DataFrame(rows)
print("\n================ RAW vs PROCESSED RP100 (metres) ================")
print(df.to_string(index=False))
df.to_csv(HERE / "raw_vs_processed_waterlevels.csv", index=False)

# --- per-station join so the exact MDT offset applied is visible --------------
mr = region(raw, xr_name, yr_name, *BOXES["Martinique"])
mp = region(proc, "station_x_coordinate", "station_y_coordinate", *BOXES["Martinique"])
rp100 = [v for v in rp_vars if "rp" in v.lower() or "100" in v][0]
rawdf = pd.DataFrame({
    "lon": np.round(raw[xr_name].values[mr], 4),
    "lat": np.round(raw[yr_name].values[mr], 4),
    "raw_rp100": np.round(raw[rp100].values[mr], 4),
})
procdf = pd.DataFrame({
    "lon": np.round(proc["station_x_coordinate"].values[mp], 4),
    "lat": np.round(proc["station_y_coordinate"].values[mp], 4),
    "proc_rp100": np.round(proc["COAST-RP_EWL_RP100_SLR_0"].values[mp], 4),
})
j = rawdf.merge(procdf, on=["lon", "lat"], how="outer")
j["mdt_applied_m"] = (j.raw_rp100 - j.proc_rp100).round(4)
print("\n=== per-station RAW vs PROCESSED around Martinique ===")
print(j.sort_values("lat").to_string(index=False))
j.to_csv(HERE / "martinique_station_raw_vs_processed.csv", index=False)
print(f"\nMDT offset applied around Martinique: median={j.mdt_applied_m.median():.4f} m "
      f"min={j.mdt_applied_m.min():.4f} max={j.mdt_applied_m.max():.4f}")
