"""Step 11: characterise the COAST-RP TC-only / ETC-only dataset variants and
compare their return-period curves against the combined COAST-RP.nc actually
used as boundary forcing.

Motivation
----------
Step 08 (08_forcing_curve_and_datum.py) found the COAST-RP return-period curve
at Martinique to be nearly flat (RP1 0.294 m -> RP1000 0.538 m), which looks
like a tide-only signal with no tropical-cyclone (TC) surge.  A `COAST-RP_TC.nc`
file has since surfaced under modelling/validation/.  This script answers:

  1. What is COAST-RP_TC.nc?  Is it a *new* dataset or a decomposition of the
     one already in use?  (variables, stations, coverage, checksum vs the copy
     in inputs/COAST-RP_dataset/)
  2. At the SAME 27 boundary stations selected for Martinique's tile 2022
     (fig1_station_table.csv / boundary_stations_detail.csv), what is the
     RP curve in the combined file vs the TC-only vs the ETC-only file?
  3. Is the TC-only curve steep (real TC surge) or also flat?

Outputs
-------
  coastrp_variant_inventory.csv
  coastrp_tc_vs_combined_martinique.csv     (per-station, all RPs, all variants)
  coastrp_tc_vs_combined_curves.csv         (median curve per region x variant)
  fig8_coastrp_tc_vs_combined.png
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
SRC_DIR = ROOT / "inputs/COAST-RP_dataset"
VAL_DIR = ROOT / "validation"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

RPS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
RPVARS = [f"storm_tide_rp_{r:04d}" for r in RPS]

VARIANTS = {
    "combined": SRC_DIR / "COAST-RP.nc",      # the file the pipeline forces with
    "TC_only": SRC_DIR / "COAST-RP_TC.nc",
    "ETC_only": SRC_DIR / "COAST-RP_ETC.nc",
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------- 1. inventory
print("=" * 100)
print("1. INVENTORY of COAST-RP variants on disk")
print("=" * 100)

inv_rows = []
for path in sorted(list(SRC_DIR.glob("COAST-RP*.nc")) + list(VAL_DIR.glob("COAST-RP*.nc"))):
    ds = xr.open_dataset(path)
    inv_rows.append({
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime": pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "md5": md5(path),
        "n_stations": int(ds.dims["stations"]),
        "lat_min": round(float(ds.station_y_coordinate.min()), 3),
        "lat_max": round(float(ds.station_y_coordinate.max()), 3),
        "n_rp_vars": len([v for v in ds.data_vars if v.startswith("storm_tide_rp_")]),
        "title": str(ds.attrs.get("title", "")),
        "summary": str(ds.attrs.get("summary", ""))[:110],
    })
    ds.close()
inv = pd.DataFrame(inv_rows)
print(inv[["path", "bytes", "mtime", "n_stations", "lat_min", "lat_max", "md5"]].to_string(index=False))
print()
for r in inv_rows:
    print(f"  {Path(r['path']).name:22s} summary: {r['summary']}")
inv.to_csv(HERE / "coastrp_variant_inventory.csv", index=False)

# is validation/COAST-RP_TC.nc identical to inputs/COAST-RP_dataset/COAST-RP_TC.nc?
tc_src = SRC_DIR / "COAST-RP_TC.nc"
tc_val = VAL_DIR / "COAST-RP_TC.nc"
if tc_src.exists() and tc_val.exists():
    same = md5(tc_src) == md5(tc_val)
    print(f"\n  validation/COAST-RP_TC.nc identical to inputs/COAST-RP_dataset/COAST-RP_TC.nc? {same}")

# ------------------------------------------------- 2. per-station comparison
print()
print("=" * 100)
print("2. RP curves at Martinique tile-2022 boundary stations (combined vs TC-only vs ETC-only)")
print("=" * 100)

det = pd.read_csv(HERE / "boundary_stations_detail.csv")
mart = det[det.tile_id == 2022].reset_index(drop=True)
print(f"  {len(mart)} boundary stations for tile 2022 (from boundary_stations_detail.csv)")

dsets = {k: xr.open_dataset(p) for k, p in VARIANTS.items()}


def match(ds: xr.Dataset, lon: float, lat: float, tol: float = 0.02):
    """Index of the station in `ds` nearest (lon, lat); None if > tol degrees."""
    dx = ds.station_x_coordinate.values - lon
    dy = ds.station_y_coordinate.values - lat
    d2 = dx * dx + dy * dy
    i = int(np.argmin(d2))
    return (i, float(np.sqrt(d2[i]))) if np.sqrt(d2[i]) <= tol else (None, float(np.sqrt(d2[i])))


rows = []
for _, st in mart.iterrows():
    rec = {"lon": st.lon, "lat": st.lat, "dist_to_FdF_km": st.dist_to_FortDeFrance_km}
    for name, ds in dsets.items():
        i, dd = match(ds, st.lon, st.lat)
        rec[f"{name}_station_id"] = str(ds.station_id.values[i]) if i is not None else ""
        rec[f"{name}_match_deg"] = round(dd, 4)
        for r, v in zip(RPS, RPVARS):
            rec[f"{name}_RP{r:04d}"] = round(float(ds[v].values[i]), 4) if i is not None else np.nan
    rows.append(rec)
per_station = pd.DataFrame(rows)
per_station.to_csv(HERE / "coastrp_tc_vs_combined_martinique.csv", index=False)

n_tc_matched = int(per_station["TC_only_RP0100"].notna().sum())
print(f"  stations matched in TC-only file: {n_tc_matched}/{len(per_station)}")
print(f"  stations matched in ETC-only file: {int(per_station['ETC_only_RP0100'].notna().sum())}/{len(per_station)}")

show = ["lon", "lat", "dist_to_FdF_km",
        "combined_RP0002", "combined_RP0100", "combined_RP1000",
        "TC_only_RP0002", "TC_only_RP0100", "TC_only_RP1000",
        "ETC_only_RP0002", "ETC_only_RP0100", "ETC_only_RP1000"]
print("\n--- per-station (m above local MSL) ---")
print(per_station[show].to_string(index=False))

print("\n--- median RP curve over the 27 tile-2022 stations ---")
curve_rows = []
for name in VARIANTS:
    rec = {"variant": name, "n_stations": int(per_station[f"{name}_RP0100"].notna().sum())}
    for r in RPS:
        rec[f"RP{r:04d}"] = round(float(np.nanmedian(per_station[f"{name}_RP{r:04d}"])), 3)
    rec["RP1000_minus_RP0002"] = round(rec["RP1000"] - rec["RP0002"], 3)
    rec["RP1000_minus_RP0001"] = round(rec["RP1000"] - rec["RP0001"], 3)
    curve_rows.append(rec)
mart_curves = pd.DataFrame(curve_rows)
print(mart_curves.to_string(index=False))

# ---------------------------------- 3. same curves for the other benchmark regions
print()
print("=" * 100)
print("3. Regional median curves per variant (same boxes as 08_forcing_curve_and_datum.py)")
print("=" * 100)
BOXES = {
    "Martinique": (-61.35, -60.75, 14.35, 14.95),
    "Guadeloupe": (-61.85, -61.00, 15.80, 16.55),
    "St Lucia": (-61.10, -60.85, 13.70, 14.15),
    "FR Gironde": (-1.60, -0.90, 45.20, 45.90),
    "FR Normandy": (-1.90, -1.10, 49.10, 49.80),
}
reg_rows = []
for reg, (a, b, c, d) in BOXES.items():
    for name, ds in dsets.items():
        x = ds.station_x_coordinate.values
        y = ds.station_y_coordinate.values
        m = (x >= a) & (x <= b) & (y >= c) & (y <= d)
        rec = {"region": reg, "variant": name, "n_stations": int(m.sum())}
        for r, v in zip(RPS, RPVARS):
            rec[f"RP{r:04d}"] = (round(float(np.nanmedian(ds[v].values[m])), 3)
                                 if m.sum() else np.nan)
        rec["RP1000_minus_RP0002"] = (round(rec["RP1000"] - rec["RP0002"], 3)
                                      if m.sum() else np.nan)
        reg_rows.append(rec)
regional = pd.DataFrame(reg_rows)
print(regional.to_string(index=False))

pd.concat([
    mart_curves.assign(region="Martinique tile-2022 stations"),
    regional,
], ignore_index=True).to_csv(HERE / "coastrp_tc_vs_combined_curves.csv", index=False)

# ------------------------------------------------------------------- 4. figure
COL = {"combined": "#1f4e79", "TC_only": "#c0392b", "ETC_only": "#7f8c8d"}
LS = {"combined": "-", "TC_only": "-", "ETC_only": "--"}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

# (a) Martinique tile-2022 stations: median + spread
ax = axes[0]
for name in VARIANTS:
    med = [np.nanmedian(per_station[f"{name}_RP{r:04d}"]) for r in RPS]
    lo = [np.nanpercentile(per_station[f"{name}_RP{r:04d}"], 10) for r in RPS]
    hi = [np.nanpercentile(per_station[f"{name}_RP{r:04d}"], 90) for r in RPS]
    if np.all(np.isnan(med)):
        continue
    ax.plot(RPS, med, LS[name], color=COL[name], lw=2.2, marker="o", ms=4.5,
            label=f"{name} (n={int(per_station[f'{name}_RP0100'].notna().sum())})")
    ax.fill_between(RPS, lo, hi, color=COL[name], alpha=0.13, lw=0)
ax.set_xscale("log")
ax.set_xlabel("return period (years)")
ax.set_ylabel("storm tide (m above local MSL)")
ax.set_title("(a) Martinique — the 27 boundary stations\nactually used to force tile 2022")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8.5)

# (b) all regions, combined vs TC-only
ax = axes[1]
reg_col = {"Martinique": "#c0392b", "Guadeloupe": "#e67e22", "St Lucia": "#f1c40f",
           "FR Gironde": "#2980b9", "FR Normandy": "#16a085"}
for reg in BOXES:
    for name, ls, alpha in (("combined", "-", 1.0), ("TC_only", ":", 0.9)):
        sub = regional[(regional.region == reg) & (regional.variant == name)]
        if sub.empty or sub.n_stations.iloc[0] == 0:
            continue
        vals = [sub[f"RP{r:04d}"].iloc[0] for r in RPS]
        ax.plot(RPS, vals, ls, color=reg_col[reg], lw=2.0, alpha=alpha,
                label=f"{reg} {name}")
ax.set_xscale("log")
ax.set_xlabel("return period (years)")
ax.set_ylabel("storm tide (m above local MSL)")
ax.set_title("(b) regional median curves\nsolid = combined COAST-RP, dotted = TC-only")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, ncol=2)

# (c) the actual issue: forcing level vs what the benchmark needs
ax = axes[2]
med_comb = [np.nanmedian(per_station[f"combined_RP{r:04d}"]) for r in RPS]
med_tc = [np.nanmedian(per_station[f"TC_only_RP{r:04d}"]) for r in RPS]
ax.plot(RPS, med_comb, "-o", color=COL["combined"], lw=2.2, ms=4.5, label="combined COAST-RP (as forced)")
if not np.all(np.isnan(med_tc)):
    ax.plot(RPS, med_tc, "-o", color=COL["TC_only"], lw=2.2, ms=4.5, label="TC-only COAST-RP")
mdt = 0.611  # AVISO MDT at Fort-de-France, vertical_datum_audit.csv
ax.plot(RPS, [v + mdt for v in med_comb], "--", color=COL["combined"], lw=1.8,
        label=f"combined + MDT ({mdt:+.3f} m)")
ax.axhline(1.91, color="k", ls="-.", lw=1.6,
           label="SWL needed for HR=0.50 (1.91 m)\n(martinique_hit_rate_ceiling_scenarios.csv)")
ax.axhline(0.529, color="green", ls=":", lw=1.6, label="DEM geoid offset +0.529 m")
ax.set_xscale("log")
ax.set_xlabel("return period (years)")
ax.set_ylabel("water level (m)")
ax.set_title("(c) Martinique: forcing magnitude vs\nthe level the benchmark implies")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("COAST-RP variants at Martinique: does the TC-only dataset add the missing surge?",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(HERE / "fig8_coastrp_tc_vs_combined.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> {HERE / 'fig8_coastrp_tc_vs_combined.png'}")

# --------------------------------------------------------------- 5. verdict aid
print()
print("=" * 100)
print("5. Is the combined file already >= the TC-only file? (i.e. does COAST-RP.nc already contain TC?)")
print("=" * 100)
ok = per_station["TC_only_RP0100"].notna()
if ok.any():
    for r in (2, 100, 1000):
        c = per_station.loc[ok, f"combined_RP{r:04d}"]
        t = per_station.loc[ok, f"TC_only_RP{r:04d}"]
        e = per_station.loc[ok, f"ETC_only_RP{r:04d}"]
        print(f"  RP{r:<5d} combined median={c.median():.3f}  TC-only median={t.median():.3f}  "
              f"ETC-only median={e.median():.3f}  | combined>=TC for {int((c >= t - 1e-9).sum())}/{int(ok.sum())} stations"
              f"  | max(TC-combined)={float((t - c).max()):+.3f} m")

for ds in dsets.values():
    ds.close()
print("\nDone.")
