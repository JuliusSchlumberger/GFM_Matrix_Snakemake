"""Step 13: updated Martinique hit-rate ceiling, now that we know

  (a) COAST-RP.nc ALREADY contains the tropical-cyclone signal (step 11) - the
      TC-only file cannot raise the forcing, and
  (b) the MDT correction was never applied to the on-disk forcing, and the code
      that would apply it uses the wrong sign (step 12).

Re-runs 10_datum_vs_skill.py's ceiling calculation over the full grid of
(COAST-RP variant x return period x MDT sign convention), so we can say exactly
how much of Martinique's HR = 0.003 each factor can possibly explain.

The ceiling = fraction of the benchmark polygon whose model-DEM elevation lies
below a given still-water level.  It is an upper bound on hit rate: no hydraulic
attenuation, no connectivity constraint, perfect bathtub fill.

Outputs
-------
  martinique_hitrate_ceiling_grid.csv
  fig10_hitrate_ceiling_grid.png
"""
from __future__ import annotations

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
REPO = HERE.parents[1]
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
sys.path.insert(0, str(REPO / "src"))
from rasters import decode_dem_cm  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)

OBSERVED_HR = 0.003  # datum_gap_vs_hitrate.csv

# ------------------------------- benchmark polygon on the model's own DEM
bench = gpd.read_file(HERE / "benchmark_FRJ_martinique.gpkg")
geoms = list(bench.geometry.buffer(0))
with rasterio.open(ROOT / "model_outputs/2022/inputs/dem.tif") as s:
    bb = bench.total_bounds
    w = from_bounds(bb[0] - 0.005, bb[1] - 0.005, bb[2] + 0.005, bb[3] + 0.005,
                    transform=s.transform).round_offsets().round_lengths()
    raw_a = s.read(1, window=w)
    tr = s.window_transform(w)
    nod = s.nodata
dem = decode_dem_cm(raw_a.astype(np.float64))
inside = ~geometry_mask(geoms, out_shape=dem.shape, transform=tr, invert=False)
valid = inside & (raw_a != nod)
elev = dem[valid]
print("=" * 100)
print("Benchmark polygon on tile 2022's own dem.tif (GOCO06s frame, +0.529 m geoid offset applied)")
print("=" * 100)
print(f"  valid cells={elev.size}  min={elev.min():.3f}  p5={np.percentile(elev,5):.3f}  "
      f"median={np.median(elev):.3f}  p95={np.percentile(elev,95):.3f}  max={elev.max():.3f} m")

# ------------------------------- forcing levels from step 11 / step 12 outputs
ps = pd.read_csv(HERE / "coastrp_tc_vs_combined_martinique.csv")
mdt = pd.read_csv(HERE / "mdt_sign_convention_martinique.csv")
MDT_MED = float(mdt.MDT_m.median())
print(f"\n  median AVISO MDT over the 27 tile-2022 stations: {MDT_MED:+.4f} m")

RPS = [2, 10, 100, 1000]
VARIANTS = ["combined", "TC_only"]
SIGNS = {
    "no MDT (as run on disk)": 0.0,
    "MDT SUBTRACTED (code as written, line 422)": -MDT_MED,
    "MDT ADDED (physically correct)": +MDT_MED,
}

rows = []
for var in VARIANTS:
    for rp in RPS:
        base = float(np.nanmedian(ps[f"{var}_RP{rp:04d}"]))
        for sname, off in SIGNS.items():
            wl = base + off
            frac = float((elev <= wl).mean())
            frac_thr = float((elev <= wl - 0.10).mean())
            rows.append({
                "coastrp_variant": var, "return_period": rp,
                "median_station_storm_tide_m": round(base, 4),
                "mdt_convention": sname, "mdt_offset_m": round(off, 4),
                "still_water_level_m": round(wl, 4),
                "ceiling_hit_rate": round(frac, 4),
                "ceiling_HR_at_0.10m_depth_threshold": round(frac_thr, 4),
                "x_observed_HR": round(frac / OBSERVED_HR, 1),
            })
grid = pd.DataFrame(rows)

# reference rows
med_elev = float(np.median(elev))
for label, wl in [("SWL needed for ceiling HR = 0.50", med_elev),
                  ("SWL needed for ceiling HR = 0.90", float(np.percentile(elev, 90)))]:
    grid.loc[len(grid)] = {
        "coastrp_variant": "-", "return_period": np.nan,
        "median_station_storm_tide_m": np.nan, "mdt_convention": label,
        "mdt_offset_m": np.nan, "still_water_level_m": round(wl, 4),
        "ceiling_hit_rate": round(float((elev <= wl).mean()), 4),
        "ceiling_HR_at_0.10m_depth_threshold": round(float((elev <= wl - 0.10).mean()), 4),
        "x_observed_HR": np.nan,
    }

print()
print("=" * 100)
print("CEILING HIT RATE (upper bound: bathtub fill of the benchmark polygon, no attenuation)")
print(f"OBSERVED HR = {OBSERVED_HR}")
print("=" * 100)
print(grid.to_string(index=False))
grid.to_csv(HERE / "martinique_hitrate_ceiling_grid.csv", index=False)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

ax = axes[0]
mark = {"combined": "o", "TC_only": "s"}
colr = {"no MDT (as run on disk)": "#1f4e79",
        "MDT SUBTRACTED (code as written, line 422)": "#c0392b",
        "MDT ADDED (physically correct)": "#27ae60"}
for var in VARIANTS:
    for sname in SIGNS:
        sub = grid[(grid.coastrp_variant == var) & (grid.mdt_convention == sname)]
        ax.plot(sub.return_period, sub.ceiling_hit_rate,
                marker=mark[var], ls="-" if var == "combined" else "--",
                color=colr[sname], lw=1.9, ms=5,
                label=f"{var}, {sname.split('(')[0].strip()}")
ax.axhline(OBSERVED_HR, color="k", ls=":", lw=1.8, label=f"observed HR = {OBSERVED_HR}")
ax.axhline(0.50, color="grey", ls="-.", lw=1.4, label="benchmark-class HR = 0.50")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("return period (years)")
ax.set_ylabel("ceiling hit rate (log)")
ax.set_title("(a) maximum achievable hit rate\nunder each dataset x datum combination")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
lo, hi = float(elev.min()), 3.0
xs = np.linspace(lo, hi, 400)
ys = [(elev <= v).mean() for v in xs]
ax.plot(xs, ys, color="#34495e", lw=2.2, label="benchmark-polygon elevation ECDF\n(= ceiling HR vs still-water level)")
pts = [
    ("as run: 0.421 m", float(np.nanmedian(ps["combined_RP0100"])), "#1f4e79"),
    ("code sign: -MDT", float(np.nanmedian(ps["combined_RP0100"])) - MDT_MED, "#c0392b"),
    ("correct sign: +MDT", float(np.nanmedian(ps["combined_RP0100"])) + MDT_MED, "#27ae60"),
    ("RP1000 +MDT", float(np.nanmedian(ps["combined_RP1000"])) + MDT_MED, "#8e44ad"),
]
# staggered label anchors so the four annotations do not overlap
anchor_y = [0.62, 0.72, 0.50, 0.38]
for (lab, v, c), ay in zip(pts, anchor_y):
    f = float((elev <= v).mean())
    ax.axvline(v, color=c, ls="--", lw=1.5)
    ax.plot([v], [f], "o", color=c, ms=8)
    ax.annotate(f"{lab}\nWL={v:.2f} m\nHR$\\leq${f:.3f}", xy=(v, f), xytext=(v - 3.6, ay),
                fontsize=8.5, color=c, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=c, alpha=0.9, lw=1.0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"))
ax.axhline(0.50, color="grey", ls="-.", lw=1.3)
ax.set_xlabel("still-water level (m, GOCO06s / model-DEM frame)")
ax.set_ylabel("fraction of benchmark polygon below that level")
ax.set_title("(b) how far the forcing sits from the\nlevel the benchmark actually implies")
ax.set_xlim(-4.6, hi)
ax.set_ylim(-0.02, 0.88)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Martinique: updated hit-rate ceiling after the TC-dataset and MDT findings", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "fig10_hitrate_ceiling_grid.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> {HERE / 'fig10_hitrate_ceiling_grid.png'}")
print("\nDone.")
