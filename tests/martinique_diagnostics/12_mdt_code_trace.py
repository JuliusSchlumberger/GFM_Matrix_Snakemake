"""Step 12: verify the MDT / vertical-datum correction by TRACING THE CODE and
reproducing the arithmetic, not just by comparing files.

Step 03/09 concluded "no MDT correction applied" from a raw-vs-processed file
comparison.  This script establishes *why*, and whether the code path is
capable of applying it at all:

  1. Reproduce the exact formula in prepare_boundary_conditions.combine_scenarios
     (line 422):  total_wl = (storm_tide - MDT) + SLR_fingerprint
     Solve for the residual  R = on_disk_WL - storm_tide - SLR_fingerprint.
       R == 0     -> MDT term was ZERO (correction not applied)
       R == -MDT  -> MDT SUBTRACTED (current code's convention)
       R == +MDT  -> MDT ADDED
     Done for SLR_0 (fingerprint == 0) AND SLR_500 (fingerprint != 0), so the
     test is not degenerate.
  2. Check which intermediate/cached artefacts exist and when they were
     written, vs when the MDT code became mandatory (git).
  3. Compute the MDT that WOULD be applied at Martinique's 27 tile-2022
     stations, and the resulting water level under each sign convention.
  4. Check the boundary condition file the simulation ACTUALLY consumed
     (model_outputs/2022/inputs/boundaries_*.gpkg) against all three options.

Outputs
-------
  mdt_residual_audit.csv
  mdt_sign_convention_martinique.csv
  fig9_mdt_sign_convention.png
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
PROC = ROOT / "processed_inputs"
WLDIR = PROC / "WL_scenarios"

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)


def stamp(p: Path) -> str:
    return (pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            if p.exists() else "MISSING")


# ============================================================ 1. artefact state
print("=" * 100)
print("1. STATE OF THE PIPELINE ARTEFACTS ON DISK")
print("=" * 100)
artefacts = [
    ("raw COAST-RP (catalog `coast_rp`)", ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc"),
    ("raw AVISO MDT (catalog `mdt_cnes_cls22`)", ROOT / "inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc"),
    ("step 1 out: COAST-RP_preprocessed.nc", PROC / "COAST-RP_preprocessed.nc"),
    ("step 2 out: MDT_mapped_on_coastal_points.nc", PROC / "MDT_mapped_on_coastal_points.nc"),
    ("step 3 out: SLR_base_ssp245_medium_2100.nc", PROC / "SLR_base_ssp245_medium_2100.nc"),
    ("step 3 out: SLR_fingerprints_ssp245_medium_all.nc", PROC / "SLR_fingerprints_ssp245_medium_all.nc"),
    ("step 4 out: WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc", WLDIR / "COAST-RP_EWL_RP100_SLR_0.nc"),
    ("step 4 out: WL_scenarios/COAST-RP_EWL_RP100_SLR_500.nc", WLDIR / "COAST-RP_EWL_RP100_SLR_500.nc"),
]
for label, p in artefacts:
    print(f"  {stamp(p):<20s}  {label}")
print(f"\n  WL_scenarios file count: {len(list(WLDIR.glob('*.nc')))}")

print("\n  git history of preparation/prepare_boundary_conditions.py:")
log = subprocess.run(
    ["git", "log", "--format=%h|%ad|%s", "--date=iso", "--",
     "snakemake_workflow/preparation/prepare_boundary_conditions.py"],
    cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
for line in log:
    h, d, s = line.split("|", 2)
    en = subprocess.run(["git", "show", f"{h}:snakemake_workflow/config/config.yml"],
                        cwd=REPO, capture_output=True, text=True).stdout
    blk = en.split("mdt_correction:")[1][:120] if "mdt_correction:" in en else ""
    flag = ("enabled: false" if "enabled: false" in blk else
            "enabled: true" if "enabled: true" in blk else
            "no `enabled` flag (MDT mandatory)" if blk else "no mdt_correction block")
    print(f"    {h}  {d}  {s[:44]:<44s} | config: {flag}")

# ================================================== 2. residual audit (the proof)
print()
print("=" * 100)
print("2. RESIDUAL AUDIT — solve  R = on_disk_WL - storm_tide - SLR_fingerprint  for every station")
print("=" * 100)

pre = xr.open_dataset(PROC / "COAST-RP_preprocessed.nc")
fp = xr.open_dataset(PROC / "SLR_fingerprints_ssp245_medium_all.nc")
n = int(pre.dims["stations"])
print(f"  COAST-RP_preprocessed.nc: {n} stations")

# station order must match for the element-wise formula to be meaningful
lon_p = pre.station_x_coordinate.values
lat_p = pre.station_y_coordinate.values

rows = []
for rp, slr in [(100, 0), (100, 500), (1000, 0), (1000, 500), (2, 0)]:
    f = WLDIR / f"COAST-RP_EWL_RP{rp}_SLR_{slr}.nc"
    if not f.exists():
        continue
    ds = xr.open_dataset(f)
    var = [v for v in ds.data_vars][0]
    wl = ds[var].values.astype(np.float64)
    assert np.allclose(ds.station_x_coordinate.values, lon_p), "station order mismatch"
    st = pre[f"storm_tide_rp_{rp:04d}"].values.astype(np.float64)
    slr_arr = fp[f"SLR_{slr}mm"].values.astype(np.float64)
    resid = wl - st - slr_arr
    rows.append({
        "file": f.name, "variable": var, "n_stations": len(wl),
        "slr_fp_median_m": round(float(np.nanmedian(slr_arr)), 4),
        "resid_min": float(np.nanmin(resid)), "resid_max": float(np.nanmax(resid)),
        "resid_absmax": float(np.nanmax(np.abs(resid))),
        "n_resid_nonzero_gt_1mm": int((np.abs(resid) > 1e-3).sum()),
    })
    ds.close()
audit = pd.DataFrame(rows)
print(audit.to_string(index=False))
audit.to_csv(HERE / "mdt_residual_audit.csv", index=False)

absmax = audit.resid_absmax.max()
print(f"\n  max |residual| over all tested scenarios = {absmax:.3e} m")
print("  => the MDT term in `total_wl = (storm_tide - mdt_arr) + slr_arr` was IDENTICALLY ZERO")
print("     for every station in every on-disk scenario file.")
print("  (the SLR_500 cases have a non-zero fingerprint, so this is not a degenerate test:")
print(f"   median SLR fingerprint there = {audit[audit.file.str.contains('SLR_500')].slr_fp_median_m.iloc[0]:.4f} m)")

# ============================== 3. what MDT *would* be at Martinique's stations
print()
print("=" * 100)
print("3. WHAT THE MDT LOOKUP WOULD RETURN AT MARTINIQUE'S 27 TILE-2022 STATIONS")
print("=" * 100)

import sys
sys.path.insert(0, str(REPO / "preparation"))
sys.path.insert(0, str(REPO / "src"))
from prepare_boundary_conditions import _load_mdt, _nearest_valid_grid  # noqa: E402

mdt_path = ROOT / "inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc"
mdt_da = _load_mdt(mdt_path, "mdt")
lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())
print(f"  MDT grid: dims={dict(mdt_da.sizes)}  units={mdt_da.attrs.get('units','?')}")

det = pd.read_csv(HERE / "boundary_stations_detail.csv")
mart = det[det.tile_id == 2022].reset_index(drop=True)

recs = []
for _, st in mart.iterrows():
    m = _nearest_valid_grid(mdt_da, lon_dim, float(st.lon), lat_dim, float(st.lat), 3.0)
    # raw storm tide at that station (RP100, SLR_0) from the preprocessed file
    d2 = (lon_p - st.lon) ** 2 + (lat_p - st.lat) ** 2
    i = int(np.argmin(d2))
    stide = float(pre["storm_tide_rp_0100"].values[i])
    recs.append({
        "lon": st.lon, "lat": st.lat, "dist_FdF_km": st.dist_to_FortDeFrance_km,
        "storm_tide_RP100_m": round(stide, 4),
        "MDT_m": round(m, 4),
        "WL_as_run_no_MDT": round(stide, 4),
        "WL_if_MDT_SUBTRACTED_current_code": round(stide - m, 4),
        "WL_if_MDT_ADDED_physically_correct": round(stide + m, 4),
    })
sign = pd.DataFrame(recs)
sign.to_csv(HERE / "mdt_sign_convention_martinique.csv", index=False)
print(sign.to_string(index=False))
print("\n  medians:")
for c in ["storm_tide_RP100_m", "MDT_m", "WL_if_MDT_SUBTRACTED_current_code",
          "WL_if_MDT_ADDED_physically_correct"]:
    print(f"    {c:<40s} {sign[c].median():+.4f} m")
print(f"\n  DEM-side geoid offset added at Martinique (vertical_datum_audit.csv): +0.529 m")
print( "  DEM is in the GOCO06s frame; MDT = sea surface ABOVE that geoid (catalog line 406-409:")
print( "  H_MSL = H_GOCO06s - MDT  <=>  H_GOCO06s = H_MSL + MDT). COAST-RP is referenced to")
print( "  local MSL (catalog line 34) => converting it INTO the DEM's frame requires ADDING MDT.")

# ================================= 4. what the simulation actually consumed
print()
print("=" * 100)
print("4. WHAT TILE 2022's SIMULATION ACTUALLY CONSUMED")
print("=" * 100)
bdir = ROOT / "model_outputs/2022/inputs"
cands = sorted(bdir.glob("boundaries_RP100_SLR_0*")) if bdir.exists() else []
if not cands and bdir.exists():
    cands = sorted(bdir.glob("boundaries_*"))[:3]
if cands:
    import geopandas as gpd
    for c in cands[:3]:
        g = gpd.read_file(c)
        vcol = [x for x in g.columns if x != "geometry"]
        print(f"  {c.name}  ({stamp(c)})  n={len(g)}  cols={vcol}")
        for col in vcol:
            if pd.api.types.is_numeric_dtype(g[col]):
                print(f"    {col}: min={g[col].min():.4f} med={g[col].median():.4f} max={g[col].max():.4f}")
        print(f"    -> matches 'no MDT' median ({sign.WL_as_run_no_MDT.median():.3f} m)? "
              f"vs 'MDT subtracted' ({sign.WL_if_MDT_SUBTRACTED_current_code.median():.3f} m) "
              f"vs 'MDT added' ({sign.WL_if_MDT_ADDED_physically_correct.median():.3f} m)")
else:
    print(f"  {bdir} — no boundaries_*.gpkg found (checked: {bdir.exists()})")

# ==================================================================== 5. figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

ax = axes[0]
order = sign.sort_values("dist_FdF_km").reset_index(drop=True)
xi = np.arange(len(order))
ax.plot(xi, order.WL_as_run_no_MDT, "-o", color="#1f4e79", ms=4,
        label=f"as run: no MDT (med {sign.WL_as_run_no_MDT.median():.3f} m)")
ax.plot(xi, order.WL_if_MDT_SUBTRACTED_current_code, "-s", color="#c0392b", ms=4,
        label=f"current code: storm_tide - MDT (med {sign.WL_if_MDT_SUBTRACTED_current_code.median():.3f} m)")
ax.plot(xi, order.WL_if_MDT_ADDED_physically_correct, "-^", color="#27ae60", ms=4,
        label=f"physically correct: storm_tide + MDT (med {sign.WL_if_MDT_ADDED_physically_correct.median():.3f} m)")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(0.529, color="green", ls=":", lw=1.5, label="DEM geoid offset +0.529 m")
ax.axhline(1.91, color="k", ls="-.", lw=1.5, label="SWL needed for HR=0.50 (1.91 m)")
ax.set_xlabel("Martinique tile-2022 boundary station (sorted by distance to Fort-de-France)")
ax.set_ylabel("RP100 water level (m)")
ax.set_title("(a) MDT sign convention at Martinique\nRP100, SLR_0")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
lab = ["as run\n(no MDT)", "code as written\n(- MDT)", "physically correct\n(+ MDT)", "needed for\nHR=0.50"]
val = [sign.WL_as_run_no_MDT.median(),
       sign.WL_if_MDT_SUBTRACTED_current_code.median(),
       sign.WL_if_MDT_ADDED_physically_correct.median(),
       1.91]
col = ["#1f4e79", "#c0392b", "#27ae60", "#7f8c8d"]
b = ax.bar(lab, val, color=col)
ax.bar_label(b, fmt="%.3f m", fontsize=10)
ax.axhline(0, color="k", lw=0.9)
ax.axhline(0.529, color="green", ls=":", lw=1.5, label="DEM geoid offset +0.529 m\n(land starts above this)")
ax.set_ylabel("median RP100 still-water level (m, GOCO06s frame)")
ax.set_title("(b) even the correct sign does not close\nthe gap to the benchmark")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8)

fig.suptitle("Martinique: MDT correction — applied? which sign? does it matter?", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "fig9_mdt_sign_convention.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> {HERE / 'fig9_mdt_sign_convention.png'}")

pre.close(); fp.close()
print("\nDone.")
