"""Step 14: independently verify the MDT sign fix committed to
prepare_boundary_conditions.py (combine_scenarios, 2026-09).

This does NOT run the pipeline (prepare_boundary_conditions.run()) and does
NOT touch/regenerate any file under processed_inputs/WL_scenarios. It is a
standalone, read-only re-derivation of

    total_wl = storm_tide(RP) + MDT + SLR_fingerprint(target_slr)

directly from the two REAL raw inputs the pipeline itself reads (found via
the hydromt data catalog, snakemake_workflow/config/data_catalog_gfm.yml):

    coast_rp        -> P:/11212688-004-global-floodmaps/modelling/
                        inputs/COAST-RP_dataset/COAST-RP.nc
    mdt_cnes_cls22   -> P:/11212688-004-global-floodmaps/modelling/
                        inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc

For RP100/SLR_0 the SLR fingerprint term is analytically zero (fingerprint
= slr_base * target_slr_m / global_mean_slr_m, and target_slr_m = 0 for the
"SLR_0" scenario, per compute_slr_fingerprints in prepare_boundary_
conditions.py) - so total_wl reduces to storm_tide_RP100 + MDT, which is
exactly what the prior investigation (12_mdt_code_trace.py) computed as
"WL_if_MDT_ADDED_physically_correct" in mdt_sign_convention_martinique.csv.

This script:
  1. Re-derives that same quantity for the same 27 Martinique (tile 2022)
     stations, straight from raw COAST-RP.nc + raw AVISO MDT (does NOT read
     any cached processed_inputs/*.nc intermediate - fully independent of
     anything the pipeline has ever written to disk), reusing only the
     lookup/interpolation helpers (_load_mdt, _nearest_valid_grid) from the
     now-fixed prepare_boundary_conditions.py - those helpers are unrelated
     to the sign bug (the bug was only in combine_scenarios' arithmetic).
  2. Compares against the prior investigation's own predicted values
     (mdt_sign_convention_martinique.csv) and against the two "wrong"
     historical values (no MDT at all; MDT subtracted) to make sure the fix
     lands on the predicted ~1.0-1.02 m and nowhere near the wrong values.
  3. Repeats the same check at 3 other real COAST-RP stations picked from
     clearly different regions (Netherlands/North Sea, Japan, Philippines)
     as a light global sanity check (no NaN, no wild sign flip).

Outputs
-------
  mdt_sign_fix_verification_martinique.csv   (27 stations, before/after)
  mdt_sign_fix_verification_other_regions.csv (3 sanity-check stations)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")

# Import ONLY the low-level MDT lookup helpers (unrelated to the sign bug,
# which lived exclusively in combine_scenarios' arithmetic) from the fixed
# module - NOT combine_scenarios/run themselves, so this stays independent
# of "running the pipeline".
sys.path.insert(0, str(REPO / "preparation"))
sys.path.insert(0, str(REPO / "src"))
from prepare_boundary_conditions import _load_mdt, _nearest_valid_grid  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)

RAW_COASTRP = ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc"
RAW_MDT = ROOT / "inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc"
FALLBACK_DEG = 3.0  # boundary_conditions.mdt_correction.fallback_search_deg in config.yml

assert RAW_COASTRP.exists(), f"missing raw COAST-RP file: {RAW_COASTRP}"
assert RAW_MDT.exists(), f"missing raw AVISO MDT file: {RAW_MDT}"

print("=" * 100)
print("STEP 14: Independent re-derivation of the MDT-sign-fixed formula from REAL raw inputs")
print("=" * 100)
print(f"  raw COAST-RP : {RAW_COASTRP}")
print(f"  raw AVISO MDT: {RAW_MDT}")

# ---------------------------------------------------------------------------
# Load raw inputs directly (NOT the cached processed_inputs/*.nc intermediates)
# ---------------------------------------------------------------------------
ds_coastrp = xr.open_dataset(RAW_COASTRP)
lon_all = ds_coastrp.station_x_coordinate.values.astype(np.float64)
lat_all = ds_coastrp.station_y_coordinate.values.astype(np.float64)
storm_tide_rp100_all = ds_coastrp["storm_tide_rp_0100"].values.astype(np.float64)
print(f"  raw COAST-RP stations (incl. Antarctic, unfiltered): {len(lon_all)}")

mdt_da = _load_mdt(RAW_MDT, "mdt")
lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())
print(f"  raw AVISO MDT grid: dims={dict(mdt_da.sizes)} units={mdt_da.attrs.get('units', '?')}")


def formula(storm_tide: float, mdt: float, slr_fp: float = 0.0) -> float:
    """The CORRECTED formula, reproduced independently of the pipeline code:
    total_wl = storm_tide(RP) + MDT + SLR_fingerprint(target_slr)
    (matches combine_scenarios line ~434: `total_wl = (storm_tide + mdt_arr) + slr_arr`)
    """
    return storm_tide + mdt + slr_fp


# ---------------------------------------------------------------------------
# 1. Martinique: the same 27 tile-2022 stations as the prior investigation
# ---------------------------------------------------------------------------
print()
print("-" * 100)
print("1. Martinique (tile 2022), 27 boundary stations — RP100, SLR_0 (fingerprint == 0 by construction)")
print("-" * 100)

det = pd.read_csv(HERE / "boundary_stations_detail.csv")
mart = det[det.tile_id == 2022].reset_index(drop=True)
assert len(mart) == 27, f"expected 27 Martinique stations, got {len(mart)}"

prior = pd.read_csv(HERE / "mdt_sign_convention_martinique.csv")

recs = []
for _, st in mart.iterrows():
    mdt = _nearest_valid_grid(mdt_da, lon_dim, float(st.lon), lat_dim, float(st.lat), FALLBACK_DEG)
    d2 = (lon_all - st.lon) ** 2 + (lat_all - st.lat) ** 2
    i = int(np.argmin(d2))
    stide = float(storm_tide_rp100_all[i])
    wl_no_mdt = stide  # "as run" historical stale-file value (no MDT at all)
    wl_mdt_subtracted = stide - mdt  # old (buggy) sign convention
    wl_mdt_added = formula(stide, mdt, 0.0)  # NEW, corrected formula (SLR_0 -> fp = 0)
    recs.append({
        "lon": st.lon, "lat": st.lat, "dist_FdF_km": st.dist_to_FortDeFrance_km,
        "storm_tide_RP100_m": round(stide, 4),
        "MDT_m": round(mdt, 4),
        "WL_as_run_no_MDT_OLD_STALE": round(wl_no_mdt, 4),
        "WL_MDT_SUBTRACTED_OLD_BUGGY_SIGN": round(wl_mdt_subtracted, 4),
        "WL_MDT_ADDED_NEW_FIXED_FORMULA": round(wl_mdt_added, 4),
    })

verify = pd.DataFrame(recs)
verify.to_csv(HERE / "mdt_sign_fix_verification_martinique.csv", index=False)
print(verify.to_string(index=False))

print("\n  medians (this script's independent re-derivation):")
for c in ["storm_tide_RP100_m", "MDT_m", "WL_as_run_no_MDT_OLD_STALE",
          "WL_MDT_SUBTRACTED_OLD_BUGGY_SIGN", "WL_MDT_ADDED_NEW_FIXED_FORMULA"]:
    print(f"    {c:<40s} {verify[c].median():+.4f} m")

med_new = float(verify["WL_MDT_ADDED_NEW_FIXED_FORMULA"].median())
med_prior_pred = float(prior["WL_if_MDT_ADDED_physically_correct"].median())
print(f"\n  this script's median (fixed formula)              : {med_new:+.4f} m")
print(f"  prior investigation's predicted median (script 12) : {med_prior_pred:+.4f} m")
print(f"  difference                                          : {med_new - med_prior_pred:+.4e} m")

# Per-station comparison against the prior investigation's own numbers
merged = verify.merge(
    prior[["lon", "lat", "MDT_m", "WL_if_MDT_ADDED_physically_correct"]],
    on=["lon", "lat"], suffixes=("_new", "_prior"),
)
merged["mdt_diff"] = merged.MDT_m_new - merged.MDT_m_prior
merged["wl_diff"] = merged.WL_MDT_ADDED_NEW_FIXED_FORMULA - merged.WL_if_MDT_ADDED_physically_correct
print(f"\n  per-station max |MDT diff|  vs prior script 12: {merged.mdt_diff.abs().max():.2e} m")
print(f"  per-station max |WL diff|   vs prior script 12: {merged.wl_diff.abs().max():.2e} m")

n_nan = int(verify["WL_MDT_ADDED_NEW_FIXED_FORMULA"].isna().sum())
lo, hi = verify["WL_MDT_ADDED_NEW_FIXED_FORMULA"].min(), verify["WL_MDT_ADDED_NEW_FIXED_FORMULA"].max()
print(f"\n  NaN count in fixed-formula output: {n_nan}")
print(f"  range: [{lo:.4f}, {hi:.4f}] m")
in_predicted_band = 0.95 <= med_new <= 1.10
print(f"  median in physically-predicted ~1.0-1.02 m band (using 0.95-1.10 m tolerance): {in_predicted_band}")
far_from_old_stale = abs(med_new - float(verify['WL_as_run_no_MDT_OLD_STALE'].median())) > 0.3
far_from_old_buggy = abs(med_new - float(verify['WL_MDT_SUBTRACTED_OLD_BUGGY_SIGN'].median())) > 0.3
print(f"  far from old stale (~0.42 m) value  (>0.3 m away): {far_from_old_stale}")
print(f"  far from old buggy-sign (~-0.20 m) value (>0.3 m away): {far_from_old_buggy}")

# ---------------------------------------------------------------------------
# 2. Other regions sanity check: Netherlands / North Sea, Japan, Philippines
# ---------------------------------------------------------------------------
print()
print("-" * 100)
print("2. Sanity check at 3 other real COAST-RP stations, different regions")
print("-" * 100)

regions = {
    "Netherlands (North Sea, Hook of Holland area)": (4.12, 51.98),
    "Japan (Tokyo Bay area)": (139.78, 35.60),
    "Philippines (Manila Bay area)": (120.95, 14.60),
}

other_recs = []
for label, (lon0, lat0) in regions.items():
    d2 = (lon_all - lon0) ** 2 + (lat_all - lat0) ** 2
    i = int(np.argmin(d2))
    lon_i, lat_i = float(lon_all[i]), float(lat_all[i])
    stide = float(storm_tide_rp100_all[i])
    mdt = _nearest_valid_grid(mdt_da, lon_dim, lon_i, lat_dim, lat_i, FALLBACK_DEG)
    wl_new = formula(stide, mdt, 0.0)
    other_recs.append({
        "region": label, "lon": round(lon_i, 4), "lat": round(lat_i, 4),
        "storm_tide_RP100_m": round(stide, 4),
        "MDT_m": (round(mdt, 4) if not np.isnan(mdt) else np.nan),
        "WL_MDT_ADDED_NEW_FIXED_FORMULA": (round(wl_new, 4) if not np.isnan(wl_new) else np.nan),
    })

other = pd.DataFrame(other_recs)
other.to_csv(HERE / "mdt_sign_fix_verification_other_regions.csv", index=False)
print(other.to_string(index=False))

n_nan_other = int(other["WL_MDT_ADDED_NEW_FIXED_FORMULA"].isna().sum())
print(f"\n  NaN count across sanity-check regions: {n_nan_other}")
for _, r in other.iterrows():
    plausible = np.isfinite(r.WL_MDT_ADDED_NEW_FIXED_FORMULA) and -5.0 < r.WL_MDT_ADDED_NEW_FIXED_FORMULA < 15.0
    print(f"    {r.region:<48s} storm_tide={r.storm_tide_RP100_m:+.3f}  MDT={r.MDT_m:+.3f}  "
          f"total_wl={r.WL_MDT_ADDED_NEW_FIXED_FORMULA:+.3f}  plausible={plausible}")

ds_coastrp.close()
print("\nDone.")
