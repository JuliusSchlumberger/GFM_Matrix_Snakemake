"""Step 5d: does the vertical-datum inconsistency track the observed validation skill?

For each French benchmark region actually validated in
validation/FRA/metrics_FRA_RP100_SLR_0.csv, tabulate:
  - the observed hit rate (0.10 m threshold, buffered domain)
  - the median COAST-RP RP100/SLR_0 water level actually used as forcing
  - the geoid offset ADDED to the DEM there (N_EGM2008 - N_GOCO06s)
  - the AVISO MDT there (0.000 m of which was actually removed from the forcing)

and, for Martinique specifically, recompute the flood-extent ceiling using the
tile's OWN dem.tif (the surface the model actually ran on) under three
water-level assumptions.

Also writes fig7_datum_gap_vs_hitrate.png.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
INK, INK2, MUTED = "#1b1b1f", "#4a4a55", "#8b8b96"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

# region -> a representative coastal point + the COAST-RP station search box
REGIONS = {
    "metropole":  ((-1.62, 49.64), (-2.0, 2.6, 43.2, 51.2)),
    "guadeloupe": ((-61.53, 16.24), (-61.85, -61.00, 15.80, 16.55)),
    "martinique": ((-61.02, 14.60), (-61.35, -60.75, 14.35, 14.95)),
    "guyane":     ((-52.33, 4.93), (-54.6, -51.6, 3.8, 6.0)),
    "mayotte":    ((45.16, -12.78), (44.9, 45.4, -13.1, -12.5)),
    "reunion":    ((55.45, -20.88), (55.1, 55.9, -21.4, -20.7)),
}

met = pd.read_csv(ROOT / "validation/FRA/metrics_FRA_RP100_SLR_0.csv")
met = met[(met.threshold_m == 0.1) & (met.domain == "buffered")]

raw = xr.open_dataset(ROOT / "inputs/COAST-RP_dataset/COAST-RP.nc")
sx, sy = raw.station_x_coordinate.values, raw.station_y_coordinate.values
rp100 = raw.storm_tide_rp_0100.values

ds_mdt = xr.open_dataset(ROOT / "inputs/AVISO/mdt_hybrid_cnes_cls22_cmems2020_global.nc")
mdt = ds_mdt["mdt"].squeeze()

rows = []
with rasterio.open(ROOT / "inputs/ICGM/geoid_offset_egm2008_goco06s.tif") as gs:
    for reg, ((px, py), (a, b, c, d)) in REGIONS.items():
        r = met[met.region == reg]
        m = (sx >= a) & (sx <= b) & (sy >= c) & (sy <= d)
        wl = np.nanmedian(rp100[m]) if m.any() else np.nan
        goff = float(list(gs.sample([(px, py)]))[0][0])
        mv = float(mdt.sel(longitude=px, latitude=py, method="nearest").values)
        if not np.isfinite(mv):  # nearest cell is land - same +/-deg fallback the pipeline uses
            for r_deg in (0.5, 1.0, 2.0, 3.0):
                sub = mdt.sel(longitude=slice(px - r_deg, px + r_deg),
                              latitude=slice(py - r_deg, py + r_deg)).values.astype(float)
                sub = sub[np.isfinite(sub)]
                if sub.size:
                    mv = float(np.median(sub)); break
        rows.append({
            "region": reg,
            "HR": float(r.HR.iloc[0]) if len(r) else np.nan,
            "CSI": float(r.CSI.iloc[0]) if len(r) else np.nan,
            "benchmark_wet_km2": float(r.benchmark_wet_km2.iloc[0]) if len(r) else np.nan,
            "model_wet_km2": float(r.model_wet_km2.iloc[0]) if len(r) else np.nan,
            "n_coastrp_stations": int(m.sum()),
            "median_RP100_forcing_m": round(float(wl), 3),
            "geoid_offset_added_to_DEM_m": round(goff, 3),
            "AVISO_MDT_m": round(mv, 3),
            "datum_gap_m": round(goff + mv, 3),
            "effective_WL_above_DEM_datum_m": round(float(wl) - goff, 3),
        })
df = pd.DataFrame(rows).sort_values("HR", ascending=False)
print("=== observed validation skill vs forcing and vertical datum, RP100/SLR_0, 0.10 m, buffered ===")
print(df.to_string(index=False))
df.to_csv(HERE / "datum_gap_vs_hitrate.csv", index=False)
print("\neffective_WL_above_DEM_datum_m = the median COAST-RP water level MINUS the geoid offset")
print("that was added to the DEM: i.e. the still-water level the model effectively applied,")
print("expressed against the ORIGINAL DeltaDTM/EGM2008 surface.")

# --------------------------------------------------------------------------
# Martinique: flood-extent ceiling using the tile's OWN dem.tif
# --------------------------------------------------------------------------
bench = gpd.read_file(HERE / "benchmark_FRJ_martinique.gpkg")
geoms = list(bench.geometry.buffer(0))
with rasterio.open(ROOT / "model_outputs/2022/inputs/dem.tif") as s:
    bb = bench.total_bounds
    w = from_bounds(bb[0]-0.005, bb[1]-0.005, bb[2]+0.005, bb[3]+0.005,
                    transform=s.transform).round_offsets().round_lengths()
    raw_a = s.read(1, window=w)
    tr = s.window_transform(w); nod = s.nodata
dem = decode_dem_cm(raw_a.astype(np.float64))
inside = ~geometry_mask(geoms, out_shape=dem.shape, transform=tr, invert=False)
valid = inside & (raw_a != nod)
elev = dem[valid]
print(f"\n=== benchmark polygon on the model's OWN DEM (tile 2022 dem.tif, GOCO06s) ===")
print(f"  cells inside={int(inside.sum())} valid={int(valid.sum())} "
      f"({100*valid.sum()/inside.sum():.1f}%)")
print(f"  elevation (m): min={elev.min():.3f} p5={np.percentile(elev,5):.3f} "
      f"median={np.median(elev):.3f} p95={np.percentile(elev,95):.3f} max={elev.max():.3f}")

MDT_MQ = 0.611
SCEN = {
    "as run (forcing 0.40 m, DEM +0.529 m geoid offset)": 0.400,
    "as run, highest nearby station (0.52 m)": 0.520,
    "MDT ADDED to forcing (0.400 + 0.611 = 1.011 m)": 0.400 + MDT_MQ,
    "MDT SUBTRACTED as documented (0.400 - 0.611)": 0.400 - MDT_MQ,
    "water level needed for HR=0.50": float(np.median(elev)),
}
print("\n  fraction of the benchmark polygon below a given still-water level")
print("  (= the maximum achievable hit rate before any hydraulic attenuation):")
scen_rows = []
for name, wl in SCEN.items():
    frac = float((elev <= wl).mean())
    frac_thr = float((elev <= wl - 0.10).mean())   # 0.10 m depth threshold
    print(f"    {name:<52s} WL={wl:6.3f} m -> {frac:.4f} wet, "
          f"{frac_thr:.4f} above the 0.10 m depth threshold")
    scen_rows.append({"scenario": name, "still_water_level_m": round(wl, 3),
                      "max_frac_benchmark_wet": round(frac, 4),
                      "max_frac_above_0.10m_threshold": round(frac_thr, 4)})
pd.DataFrame(scen_rows).to_csv(HERE / "martinique_hit_rate_ceiling_scenarios.csv", index=False)
print(f"\n  observed HR = 0.003")

# --------------------------------------------------------------------------
# fig7
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
COL = {"metropole": "#0b6e4f", "guadeloupe": "#e07a00", "martinique": "#c1121f",
       "guyane": "#8a2d6b", "mayotte": "#2b6cb0", "reunion": "#7a5c00"}
cols = [COL[r] for r in df.region]

ax = axes[0]
ax.scatter(df.median_RP100_forcing_m, df.HR, s=110, c=cols, ec="white", lw=1.2, zorder=3)
for _, r in df.iterrows():
    ax.annotate(r.region, (r.median_RP100_forcing_m, r.HR), textcoords="offset points",
                xytext=(8, 3), fontsize=7, color=INK2)
ax.set_xlabel("median COAST-RP RP100/SLR_0 forcing (m)")
ax.set_ylabel("hit rate (0.10 m, buffered)")
ax.set_title("(a) skill vs forcing magnitude", loc="left")

ax = axes[1]
ax.scatter(df.datum_gap_m, df.HR, s=110, c=cols, ec="white", lw=1.2, zorder=3)
for _, r in df.iterrows():
    ax.annotate(r.region, (r.datum_gap_m, r.HR), textcoords="offset points",
                xytext=(8, 3), fontsize=7, color=INK2)
ax.set_xlabel("vertical-datum gap: geoid offset added to DEM + AVISO MDT never\n"
              "removed from the forcing (m)")
ax.set_ylabel("hit rate (0.10 m, buffered)")
ax.set_title("(b) skill vs vertical-datum gap", loc="left")

ax = axes[2]
ax.scatter(df.effective_WL_above_DEM_datum_m, df.HR, s=110, c=cols, ec="white", lw=1.2, zorder=3)
ax.axvline(0, color=MUTED, lw=1, ls=":")
for _, r in df.iterrows():
    ax.annotate(f"{r.region}\n{r.effective_WL_above_DEM_datum_m:+.2f} m",
                (r.effective_WL_above_DEM_datum_m, r.HR), textcoords="offset points",
                xytext=(8, 0), fontsize=6.8, color=INK2)
ax.set_xlabel("effective still-water level applied, expressed against the\n"
              "original DeltaDTM/EGM2008 surface (m)")
ax.set_ylabel("hit rate (0.10 m, buffered)")
ax.set_title("(c) skill vs effective forcing after the DEM's geoid shift", loc="left")

for ax in axes:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("Six French benchmark regions, RP100 / SLR_0 — Martinique is the worst case on every axis.\n"
             "CAVEAT: n=6, terrain and benchmark definitions differ per region, and forcing and datum gap "
             "are themselves correlated,\nso panels (b)/(c) are suggestive of a systematic tropical bias, "
             "NOT a demonstrated single cause.", fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(HERE / "fig7_datum_gap_vs_hitrate.png")
plt.close(fig)
print("\nwrote fig7_datum_gap_vs_hitrate.png")
