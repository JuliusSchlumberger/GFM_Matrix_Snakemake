"""Step 5c: audit the two halves of the vertical-datum design, region by region.

The pipeline's stated contract (config.yml / src/vertical_datum.py /
preparation/prepare_boundary_conditions.py) is:
    DEM      : DeltaDTM(EGM2008) + (N_EGM2008 - N_GOCO06s)      -> GOCO06s
    forcing  : COAST-RP storm tide (local MSL) - MDT            -> GOCO06s
so that DEM and forcing share one vertical reference.

This script measures BOTH terms as they actually are on disk, for Martinique,
Guadeloupe and metropolitan France.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
OFFSET = ROOT / "inputs/ICGM/geoid_offset_egm2008_goco06s.tif"
MDT = ROOT / "inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc"
pd.set_option("display.width", 240)

PTS = {
    "Martinique (Fort-de-France)": (-61.05, 14.60),
    "Martinique (Lamentin)":       (-61.00, 14.60),
    "Guadeloupe (Pointe-a-Pitre)": (-61.53, 16.24),
    "Guyane (Cayenne)":            (-52.33, 4.93),
    "Mayotte":                     (45.16, -12.78),
    "Reunion":                     (55.45, -20.88),
    "FR Gironde (Arcachon)":       (-1.16, 44.66),
    "FR Normandy (Cherbourg)":     (-1.62, 49.64),
    "FR Dunkerque":                (2.37, 51.04),
}

rows = []
print(f"geoid offset raster exists: {OFFSET.exists()}  -> {OFFSET}")
if OFFSET.exists():
    with rasterio.open(OFFSET) as s:
        print(f"  bounds={s.bounds} res={s.res} shape=({s.height},{s.width}) nodata={s.nodata}")
        offs = {k: float(list(s.sample([(x, y)]))[0][0]) for k, (x, y) in PTS.items()}
else:
    offs = {k: np.nan for k in PTS}

print(f"\nAVISO MDT exists: {MDT.exists()}  -> {MDT}")
mdts = {k: np.nan for k in PTS}
if MDT.exists():
    ds = xr.open_dataset(MDT)
    print(f"  variables: {list(ds.data_vars)}")
    var = "mdt" if "mdt" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var].squeeze()
    latd = next(d for d in da.dims if "lat" in d.lower())
    lond = next(d for d in da.dims if "lon" in d.lower())
    lonvals = da[lond].values
    lon0to360 = float(lonvals.min()) >= 0
    print(f"  using variable '{var}', dims {da.dims}, lon range "
          f"[{lonvals.min():.2f},{lonvals.max():.2f}]")
    for k, (x, y) in PTS.items():
        xx = (x % 360) if lon0to360 else x
        # nearest valid cell within +/- 1 deg (same idea as _nearest_valid_grid)
        sub = da.sel({lond: slice(xx - 1.0, xx + 1.0), latd: slice(y - 1.0, y + 1.0)})
        if sub.size == 0:
            sub = da.sel({lond: slice(xx - 1.0, xx + 1.0), latd: slice(y + 1.0, y - 1.0)})
        v = sub.values.astype(float)
        v = v[np.isfinite(v)]
        pt = da.sel({lond: xx, latd: y}, method="nearest").values
        mdts[k] = float(pt) if np.isfinite(pt) else (float(np.median(v)) if v.size else np.nan)

for k in PTS:
    rows.append({
        "location": k, "lon": PTS[k][0], "lat": PTS[k][1],
        "geoid_offset_added_to_DEM_m": round(offs[k], 3) if np.isfinite(offs[k]) else None,
        "AVISO_MDT_m": round(mdts[k], 3) if np.isfinite(mdts[k]) else None,
    })
df = pd.DataFrame(rows)
df["MDT_actually_subtracted_from_forcing_m"] = 0.0   # measured in 04_benchmark_attrs.py
df["datum_gap_DEM_minus_forcing_m"] = (
    df["geoid_offset_added_to_DEM_m"] + df["AVISO_MDT_m"]
).round(3)
print("\n================ vertical-datum audit ================")
print(df.to_string(index=False))
df.to_csv(HERE / "vertical_datum_audit.csv", index=False)
print("""
Columns:
  geoid_offset_added_to_DEM_m  N_EGM2008 - N_GOCO06s, ADDED to every DEM pixel by
                               src/vertical_datum.py (measured from the cached raster).
  AVISO_MDT_m                  the MDT the design says should have been removed from
                               COAST-RP; measured to be 0.000 m in the WL_scenarios
                               files actually on disk (04_benchmark_attrs.py).
  datum_gap_DEM_minus_forcing  how much higher the DEM sits relative to the forcing than
                               a consistent GOCO06s pairing would put it, IF MDT should be
                               ADDED to COAST-RP to reach the geoid frame.  Sign of the MDT
                               term is exactly the open question flagged in the report.
""")
