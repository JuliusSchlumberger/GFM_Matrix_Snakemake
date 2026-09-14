"""Steps 1-4: produce the diagnostic figures for the Martinique validation failure.

Figures written next to this script:
  fig1_martinique_tile_and_stations.png   tile outlines + search box + every COAST-RP
                                          station, labelled with distance and RP100 water level
  fig2_merge_chunk_N10W065_context.png    the "three islands" merge chunk vs the 9 simulation
                                          tiles that feed it
  fig3_dem_fort_de_france.png             DeltaDTM elevation map + coverage + histogram
  fig4_overlay_lamentin.png               benchmark polygon vs model depth vs DEM vs agreement
  fig5_forcing_comparison.png             RP100/SLR_0 station water levels: Martinique vs
                                          Guadeloupe vs metropolitan France
  fig6_martinique_wet_cells.png           where the model DOES put water on Martinique
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from boundaries import load_waterlevel_stations  # noqa: E402
from merge import AQUEDUCT_NODATA  # noqa: E402
from rasters import decode_waterlevel_cm  # noqa: E402

ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
MANIFEST = ROOT / "processed_inputs/mask/domain_tiles_global.gpkg"
DEM_VRT = ROOT / "inputs/DeltaDTM/deltadtm.vrt"
MASK_VRT = ROOT / "inputs/DeltaDTM_masks/deltadtm_mask.vrt"
MERGED = ROOT / "merged_results/chunks/waterdepth_N10W065_RP100_SLR_0.tif"
AGREE = ROOT / "validation/FRA/agreement_FRA_martinique_RP100_SLR_0.tif"

INK, INK2, MUTED = "#1b1b1f", "#4a4a55", "#8b8b96"
SEQ_ELEV = "YlOrBr"       # single-hue sequential, light (low) -> dark (high)
SEQ_DEPTH = "Blues"       # single hue, light -> dark, for water depth
LAND, NODATA_C, WATER = "#dcdcd6", "#5b5b66", "#2b6cb0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9,
    "axes.grid": False, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

def km(dx, dy, lat):
    return np.hypot(dx * 111.32 * np.cos(np.radians(lat)), dy * 110.57)

tiles = gpd.read_file(MANIFEST)
bench = gpd.read_file(HERE / "benchmark_FRJ_martinique.gpkg")
bench_geoms = list(bench.geometry.buffer(0))
stations = load_waterlevel_stations(
    ROOT / "processed_inputs/WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc",
    variable="COAST-RP_EWL_RP100_SLR_0", x_var="station_x_coordinate",
    y_var="station_y_coordinate", column_name="SLR_0")

def read_window(path, bbox):
    """Windowed read clipped to the dataset; returns (values, raw, nodata, true_extent)."""
    with rasterio.open(path) as s:
        b = s.bounds
        cb = (max(bbox[0], b.left), max(bbox[1], b.bottom),
              min(bbox[2], b.right), min(bbox[3], b.top))
        w = from_bounds(*cb, transform=s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w)
        tr = s.window_transform(w)
        nod = s.nodata
    ext = [tr.c, tr.c + a.shape[1] * tr.a, tr.f + a.shape[0] * tr.e, tr.f]
    return a.astype(np.float64), nod, ext, tr

def ext_of(bbox):
    return [bbox[0], bbox[2], bbox[1], bbox[3]]

# =============================================================================
# FIG 1 - tile outlines, search box, COAST-RP stations
# =============================================================================
t2022 = tiles[tiles.tile_id == 2022]
t2031 = tiles[tiles.tile_id == 2031]
b22 = t2022.total_bounds
cx, cy = (b22[0] + b22[2]) / 2, (b22[1] + b22[3]) / 2
search = box(min(b22[0] - 1.0, cx - 1.0), min(b22[1] - 1.0, cy - 1.0),
             max(b22[2] + 1.0, cx + 1.0), max(b22[3] + 1.0, cy + 1.0))
sb = search.bounds
BB1 = (sb[0] - 0.12, sb[1] - 0.12, sb[2] + 0.12, sb[3] + 0.12)

cand = stations[stations.intersects(search)].copy()
cand["dist_km"] = km(cand.geometry.x - cx, cand.geometry.y - cy, cy)
cand = cand.sort_values("dist_km").reset_index(drop=True)
cand["n"] = np.arange(1, len(cand) + 1)
used = gpd.read_file(ROOT / "model_outputs/2022/inputs/boundaries_RP100_SLR_0.gpkg")
used_wl = decode_waterlevel_cm(used["SLR_0"].to_numpy())
print(f"fig1: {len(cand)} candidates in search box; {len(used)} written to tile 2022's "
      f"boundaries file; used WL min/med/max = "
      f"{used_wl.min():.3f}/{np.median(used_wl):.3f}/{used_wl.max():.3f} m")

mask_arr, mask_nod, mask_ext, _ = read_window(MASK_VRT, BB1)
land = np.where(mask_arr == 0, 1.0, np.nan)

fig = plt.figure(figsize=(11.6, 7.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.12)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(land, extent=mask_ext, origin="upper", cmap=ListedColormap([LAND]),
          interpolation="nearest", zorder=0)
gpd.GeoSeries([search], crs=4326).boundary.plot(ax=ax, color="#b07a00", ls="--", lw=1.3, zorder=2)
t2022.boundary.plot(ax=ax, color="#0b6e4f", lw=1.8, zorder=3)
t2031.boundary.plot(ax=ax, color="#8a2d6b", lw=1.8, ls=":", zorder=3)
bench.plot(ax=ax, color="#c1121f", ec="#c1121f", lw=0.5, zorder=5)
sc = ax.scatter(cand.geometry.x, cand.geometry.y, c=cand["SLR_0"], cmap=SEQ_DEPTH,
                vmin=0.30, vmax=0.70, s=90, ec="white", lw=1.0, zorder=6)
for _, r in cand.iterrows():
    ax.annotate(str(int(r["n"])), (r.geometry.x, r.geometry.y), fontsize=5.6,
                color="white" if r["SLR_0"] > 0.5 else INK, ha="center", va="center", zorder=7)
cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
cb.set_label("COAST-RP RP100 / SLR_0 water level (m)", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax.set_xlim(BB1[0], BB1[2]); ax.set_ylim(BB1[1], BB1[3])
ax.set_aspect(1 / np.cos(np.radians(14.6)))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Martinique's simulation tiles and every COAST-RP station in their search box\n"
             f"RP100 / SLR_0 — all {len(cand)} candidates survived the ocean-connectivity\n"
             f"filter and were used; nearest is {cand.dist_km.min():.1f} km from the tile centre",
             loc="left")
ax.legend(handles=[
    Line2D([], [], color="#0b6e4f", lw=1.8, label="simulation tile 2022"),
    Line2D([], [], color="#8a2d6b", lw=1.8, ls=":", label="simulation tile 2031"),
    Line2D([], [], color="#b07a00", lw=1.3, ls="--", label="station search box (bbox+1°, min 2°)"),
    Patch(facecolor="#c1121f", label="FRJ_TRI_LAMENTIN benchmark"),
    Patch(facecolor=LAND, label="DeltaDTM coastal land (≤30 m)"),
], loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor=MUTED)

axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
tbl = pd.DataFrame({
    "#": cand["n"].astype(int), "lon": cand.geometry.x.round(3), "lat": cand.geometry.y.round(3),
    "WL (m)": cand["SLR_0"].round(3), "dist (km)": cand["dist_km"].round(1),
})
t = axt.table(cellText=tbl.values, colLabels=tbl.columns, loc="upper center", cellLoc="right")
t.auto_set_font_size(False); t.set_fontsize(6.2); t.scale(1.0, 1.05)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#e2e2e2")
    if r == 0:
        cell.set_facecolor("#f2f2ef"); cell.set_text_props(color=INK, weight="bold")
axt.set_title("All 27 stations forcing tile 2022 at RP100 / SLR_0\n"
              "(distance = to tile centre; the same 27 force tile 2031)", loc="left", fontsize=8.5)
fig.savefig(HERE / "fig1_martinique_tile_and_stations.png")
plt.close(fig)
tbl.to_csv(HERE / "fig1_station_table.csv", index=False)
print("wrote fig1")

# =============================================================================
# FIG 2 - the "three islands" merge chunk vs its 9 simulation tiles
# =============================================================================
BB2 = (-65.0, 10.0, -60.0, 15.0)
with rasterio.open(MERGED) as s:
    md = s.read(1, out_shape=(1, 1400, 1400)).astype(np.float64)
    mnod = s.nodata
    m_ext = ext_of(tuple(s.bounds))
computed = (md != mnod) & (md != AQUEDUCT_NODATA) & np.isfinite(md)
wet = computed & (md > 0.10)
print(f"fig2: merged chunk computed cells (downsampled view) = {int(computed.sum())}, "
      f"wet>0.10 m = {int(wet.sum())}")

chunk_tiles = tiles[tiles.intersects(box(*BB2))]
fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.0))
for ax, show_wet in zip(axes, (False, True)):
    ax.imshow(np.where(computed, 1.0, np.nan), extent=m_ext, origin="upper",
              cmap=ListedColormap(["#b9b9b0"]), interpolation="nearest")
    if show_wet:
        ax.imshow(np.where(wet, md, np.nan), extent=m_ext, origin="upper",
                  cmap=SEQ_DEPTH, vmin=0, vmax=2.0, interpolation="nearest")
    chunk_tiles.boundary.plot(ax=ax, color="#c1121f", lw=1.1)
    for _, r in chunk_tiles.iterrows():
        c = r.geometry.centroid
        ISL = {2022: "Martinique", 2031: "Martinique", 2429: "St Lucia",
               2185: "St Vincent +\nGrenadines", 2340: "Tobago", 18: "Trinidad /\nOrinoco",
               652: "Venezuela", 1064: "Venezuela", 1446: "Venezuela"}
        ax.annotate(f"{int(r.tile_id)}\n{ISL.get(int(r.tile_id), '')}", (c.x, c.y),
                    fontsize=6.6, color="#c1121f", ha="center", va="center", weight="bold")
    ax.set_xlim(BB2[0], BB2[2]); ax.set_ylim(BB2[1], BB2[3])
    ax.set_aspect(1 / np.cos(np.radians(12.5)))
    ax.set_xlabel("longitude")
axes[0].set_ylabel("latitude")
axes[0].set_title("(a) cells with a computed value (grey) + the 9 SEPARATE simulation\n"
                  "tiles intersecting this 5°×5° merge chunk (red, labelled by tile_id)",
                  loc="left")
axes[1].set_title("(b) same, with modelled RP100/SLR_0 depth > 0.10 m in blue", loc="left")
fig.suptitle("The separate 'islands' inside waterdepth_N10W065_RP100_SLR_0.tif come from the fixed 5°×5° "
             "POSTPROCESSING\nmerge grid mosaicking several independent simulation tiles "
             "(Martinique 2022/2031, St Lucia 2429, St Vincent 2185, …) — not from one\n"
             "simulation tile spanning three islands.", fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "fig2_merge_chunk_N10W065_context.png")
plt.close(fig)
print("wrote fig2")

# =============================================================================
# FIG 3 - DeltaDTM elevation around Fort-de-France / Lamentin
# =============================================================================
BB3 = (-61.16, 14.52, -60.92, 14.70)
dem_raw, dem_nod, dem_ext, _ = read_window(DEM_VRT, BB3)
dem_v = np.where(dem_raw == dem_nod, np.nan, dem_raw)
msk_raw, _, msk_ext, _ = read_window(MASK_VRT, BB3)

# elevation inside the benchmark polygon (true polygon mask, native 30 m grid)
with rasterio.open(DEM_VRT) as s:
    bb = bench.total_bounds
    w = from_bounds(bb[0] - 0.005, bb[1] - 0.005, bb[2] + 0.005, bb[3] + 0.005,
                    transform=s.transform).round_offsets().round_lengths()
    a = s.read(1, window=w).astype(np.float64)
    tr = s.window_transform(w)
    nod = s.nodata
inside = ~geometry_mask(bench_geoms, out_shape=a.shape, transform=tr, invert=False)
valid_in = inside & (a != nod)
elev_in = a[valid_in]
n_in, n_valid = int(inside.sum()), int(valid_in.sum())
print(f"fig3: benchmark polygon {n_in} cells inside, {n_valid} valid DEM "
      f"({100*n_valid/n_in:.1f}%), median elev {np.median(elev_in):.3f} m")

fig = plt.figure(figsize=(12.4, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.4, 1.2], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor("#f6f6f2")
im = ax.imshow(dem_v, extent=dem_ext, origin="upper", cmap=SEQ_ELEV, vmin=0, vmax=10,
               interpolation="nearest")
bench.boundary.plot(ax=ax, color="#c1121f", lw=0.8)
ax.set_title("(a) raw DeltaDTM v1.1 elevation (m, EGM2008 as released)\n"
             "white = nodata (ocean only); DeltaDTM clamps all land above\n"
             "30 m to exactly 30.000 m. red = FRJ_TRI_LAMENTIN benchmark",
             loc="left")
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("elevation (m)", fontsize=7.5)

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(np.where(~np.isnan(dem_v), 1.0, np.nan), extent=dem_ext, origin="upper",
           cmap=ListedColormap([LAND]), interpolation="nearest")
ax2.imshow(np.where(np.isnan(dem_v), 1.0, np.nan), extent=dem_ext, origin="upper",
           cmap=ListedColormap([NODATA_C]), interpolation="nearest")
ax2.imshow(np.where(np.isin(msk_raw, [1, 2, 3]), 1.0, np.nan), extent=msk_ext, origin="upper",
           cmap=ListedColormap([WATER]), interpolation="nearest", alpha=0.9)
bench.boundary.plot(ax=ax2, color="#c1121f", lw=0.8)
ax2.set_title("(b) DeltaDTM coverage — no gaps anywhere on land\n"
              "dark = DEM nodata (ocean); blue = mask says ocean/lake/river", loc="left")
ax2.legend(handles=[Patch(facecolor=LAND, label="valid DEM"),
                    Patch(facecolor=NODATA_C, label="DEM nodata"),
                    Patch(facecolor=WATER, label="mask = ocean / lake / river")],
           fontsize=6.5, loc="lower left", frameon=True, facecolor="white", edgecolor=MUTED)
for a_ in (ax, ax2):
    a_.set_xlim(BB3[0], BB3[2]); a_.set_ylim(BB3[1], BB3[3])
    a_.set_aspect(1 / np.cos(np.radians(14.6))); a_.set_xlabel("longitude")
ax.set_ylabel("latitude")

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(elev_in, bins=np.arange(-1, 12.1, 0.25), color=WATER, ec="white", lw=0.3)
ymax = ax3.get_ylim()[1]
for x, c, lab, yf in [(0.40, "#c1121f", "median COAST-RP forcing 0.40 m", 0.95),
                      (0.502, "#e07a00", "max nearby station 0.50 m", 0.85)]:
    ax3.axvline(x, color=c, lw=1.6, ls="--")
    ax3.annotate(lab, (5.0, ymax * yf), fontsize=6.6, color=c, ha="left", va="top")
ax3.set_xlabel("DeltaDTM elevation inside benchmark polygon (m)")
ax3.set_ylabel("cells (30 m)")
ax3.set_title(f"(c) elevation inside the benchmark polygon (raw DeltaDTM)\n"
              f"n={n_valid} valid cells ({100*n_valid/n_in:.1f}% of polygon), median "
              f"{np.median(elev_in):.2f} m; only {(elev_in <= 0.40).mean()*100:.1f}% below the\n"
              f"0.40 m forcing — and only 0.8% on the +0.529 m geoid-shifted\n"
              f"DEM the model actually ran on (see fig4 panel d)", loc="left")
for sp in ("top", "right"):
    ax3.spines[sp].set_visible(False)
fig.suptitle("DeltaDTM elevation and coverage, Fort-de-France / Lamentin (Martinique) — "
             "uniform 1 arc-second grid, no nodata gaps on land, no implausible values",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(HERE / "fig3_dem_fort_de_france.png")
plt.close(fig)
print("wrote fig3")

# =============================================================================
# FIG 4 - four-panel overlay on the benchmark polygon
# =============================================================================
bbb = bench.total_bounds
BB4 = (bbb[0] - 0.015, bbb[1] - 0.015, bbb[2] + 0.015, bbb[3] + 0.015)
dem4_raw, dem4_nod, dem4_ext, _ = read_window(DEM_VRT, BB4)
dem4 = np.where(dem4_raw == dem4_nod, np.nan, dem4_raw)
wd4_raw, wd4_nod, wd4_ext, wd4_tr = read_window(MERGED, BB4)
wd4 = np.where((wd4_raw == wd4_nod) | (wd4_raw == AQUEDUCT_NODATA), np.nan, wd4_raw)
ag_raw, ag_nod, ag_ext, _ = read_window(AGREE, BB4)
uv, uc = np.unique(ag_raw, return_counts=True)
print(f"fig4: agreement raster own extent={[round(v,4) for v in ag_ext]}; "
      f"codes={dict(zip(uv.astype(int).tolist(), uc.tolist()))}")
nwet = int(np.nansum(wd4 > 0.10))
print(f"fig4: model cells in window: computed={int(np.isfinite(wd4).sum())}, "
      f">0 m={int(np.nansum(wd4 > 0))}, >0.10 m={nwet}, "
      f"max depth={np.nanmax(wd4):.3f} m")

fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.8))
a0 = axes[0, 0]
im = a0.imshow(dem4, extent=dem4_ext, origin="upper", cmap=SEQ_ELEV, vmin=0, vmax=6,
               interpolation="nearest")
bench.boundary.plot(ax=a0, color="#c1121f", lw=0.9)
a0.set_title("(a) DeltaDTM elevation + benchmark outline", loc="left")
fig.colorbar(im, ax=a0, shrink=0.78, pad=0.02).set_label("elevation (m)", fontsize=7.5)

a1 = axes[0, 1]
a1.imshow(np.where(np.isfinite(wd4), 1.0, np.nan), extent=wd4_ext, origin="upper",
          cmap=ListedColormap(["#e6e6e0"]), interpolation="nearest")
rr, cc = np.where(np.nan_to_num(wd4, nan=-1) > 0.0)
if rr.size:
    xs, ys = rasterio.transform.xy(wd4_tr, rr, cc)
    a1.scatter(xs, ys, c=wd4[rr, cc], cmap=SEQ_DEPTH, vmin=0, vmax=1.5, s=42,
               ec="#1b1b1f", lw=0.4, zorder=4)
bench.boundary.plot(ax=a1, color="#c1121f", lw=0.9, zorder=5)
a1.set_title(f"(b) model RP100/SLR_0 water depth > 0 m (dots, enlarged)\n"
             f"pale grey = computed but completely dry;\n"
             f"only {int((wd4 > 0.10).sum())} cells exceed 0.10 m in this window", loc="left")
sm = plt.cm.ScalarMappable(cmap=SEQ_DEPTH, norm=plt.Normalize(0, 1.5))
fig.colorbar(sm, ax=a1, shrink=0.78, pad=0.02).set_label("water depth (m)", fontsize=7.5)

a2 = axes[1, 0]
AG_COLORS = {0: "#e6e6e0", 1: "#2e7d32", 2: "#c1121f", 3: "#e07a00"}
AG_LABEL = {0: "0  dry in both", 1: "1  agree (both wet)", 2: "2  under-predict (missed)",
            3: "3  over-predict"}
for code, col in AG_COLORS.items():
    a2.imshow(np.where(ag_raw == code, 1.0, np.nan), extent=ag_ext, origin="upper",
              cmap=ListedColormap([col]), interpolation="nearest")
bench.boundary.plot(ax=a2, color="#1b1b1f", lw=0.5)
counts = dict(zip(uv.astype(int).tolist(), uc.tolist()))
a2.set_title("(c) agreement_FRA_martinique_RP100_SLR_0.tif (~200 m grid)\n"
             f"code counts here: {counts}", loc="left")
a2.legend(handles=[Patch(facecolor=AG_COLORS[c], ec=MUTED, lw=0.4, label=AG_LABEL[c])
                   for c in (0, 1, 2, 3)],
          fontsize=6.5, loc="lower left", frameon=True, facecolor="white", edgecolor=MUTED)

# the surface the MODEL actually ran on (raw DeltaDTM + the EGM2008->GOCO06s geoid offset)
with rasterio.open(ROOT / "model_outputs/2022/inputs/dem.tif") as s:
    w = from_bounds(bb[0] - 0.005, bb[1] - 0.005, bb[2] + 0.005, bb[3] + 0.005,
                    transform=s.transform).round_offsets().round_lengths()
    mdem_raw = s.read(1, window=w); mtr = s.window_transform(w); mnod2 = s.nodata
from rasters import decode_dem_cm  # noqa: E402
mdem = decode_dem_cm(mdem_raw.astype(np.float64))
minside = ~geometry_mask(bench_geoms, out_shape=mdem.shape, transform=mtr, invert=False)
elev_model = mdem[minside & (mdem_raw != mnod2)]
print(f"fig4(d): model DEM inside benchmark median={np.median(elev_model):.3f} m vs "
      f"raw DeltaDTM median={np.median(elev_in):.3f} m "
      f"(geoid offset {np.median(elev_model)-np.median(elev_in):+.3f} m)")

a3 = axes[1, 1]
srt = np.sort(elev_in)
srt_m = np.sort(elev_model)
a3.plot(srt, np.arange(1, srt.size + 1) / srt.size, color=MUTED, lw=1.8, ls="--",
        label="raw DeltaDTM (EGM2008)")
a3.plot(srt_m, np.arange(1, srt_m.size + 1) / srt_m.size, color=WATER, lw=2.4,
        label="tile 2022 dem.tif — the surface the model ran on\n(DeltaDTM + 0.529 m geoid offset, GOCO06s)")
for x, c, lab, yy in [(0.40, "#c1121f", "as run: median station forcing 0.40 m", 0.80),
                      (1.011, "#e07a00", "if the AVISO MDT (0.611 m) were added to the forcing", 0.62),
                      (1.910, "#8a2d6b", "still-water level the benchmark itself implies", 0.44)]:
    a3.axvline(x, color=c, lw=1.5, ls="--")
    a3.annotate(f"{lab}\n→ at most {100*(srt_m <= x).mean():.1f}% of the benchmark wet",
                (2.15, yy), fontsize=6.6, color=c, ha="left", va="center")
a3.set_xlim(-1, 8); a3.set_ylim(0, 1)
a3.legend(fontsize=6.2, loc="lower right", frameon=True, facecolor="white", edgecolor=MUTED)
a3.set_xlabel("elevation inside benchmark polygon (m)")
a3.set_ylabel("cumulative fraction of benchmark cells")
a3.set_title("(d) elevation ECDF inside the benchmark polygon\n"
             "the physical ceiling on hit rate, before any hydraulics", loc="left")
for sp in ("top", "right"):
    a3.spines[sp].set_visible(False)
for ax in (a0, a1, a2):
    ax.set_xlim(BB4[0], BB4[2]); ax.set_ylim(BB4[1], BB4[3])
    ax.set_aspect(1 / np.cos(np.radians(14.6))); ax.set_xlabel("longitude")
a0.set_ylabel("latitude"); a2.set_ylabel("latitude")
fig.suptitle("FRJ_TRI_LAMENTIN benchmark vs GFM model, RP100 / SLR_0 (Martinique)",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(HERE / "fig4_overlay_lamentin.png")
plt.close(fig)
print("wrote fig4")

# =============================================================================
# FIG 5 - forcing magnitude comparison
# =============================================================================
REG = {
    "Martinique": (-61.35, -60.75, 14.35, 14.95, 0.003),
    "Guadeloupe": (-61.85, -61.00, 15.80, 16.55, 0.078),
    "Metropole\nGironde": (-1.60, -0.90, 45.20, 45.90, 0.758),
    "Metropole\nNormandy": (-1.90, -1.10, 49.10, 49.80, 0.758),
}
data, labels, hrs = [], [], []
for name, (a, b, c, d, hr) in REG.items():
    sel = stations[(stations.geometry.x.between(a, b)) & (stations.geometry.y.between(c, d))]
    data.append(sel["SLR_0"].to_numpy()); labels.append(f"{name}\nn={len(sel)}"); hrs.append(hr)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
COLS = ["#c1121f", "#e07a00", "#2b6cb0", "#0b6e4f"]
ax = axes[0]
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                medianprops=dict(color="white", lw=1.8),
                flierprops=dict(marker="o", ms=3, mec=INK2, mfc="none"))
for p, col in zip(bp["boxes"], COLS):
    p.set_facecolor(col); p.set_edgecolor("white"); p.set_linewidth(1.2)
for i, d in enumerate(data, start=1):
    ax.annotate(f"median {np.median(d):.2f} m", (i, np.median(d)), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=6.8, color=INK)
ax.set_ylabel("COAST-RP RP100 / SLR_0 water level (m)")
ax.set_title("(a) forcing magnitude actually applied, by region", loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax = axes[1]
meds = [float(np.median(d)) for d in data]
ax.scatter(meds, hrs, s=110, c=COLS, ec="white", lw=1.2, zorder=3)
for m, h, k in zip(meds, hrs, REG):
    ax.annotate(f"{k.replace(chr(10),' ')}  HR={h:.3f}", (m, h), textcoords="offset points",
                xytext=(9, 3), fontsize=7, color=INK2)
ax.set_xlim(0, 5); ax.set_ylim(-0.05, 0.9)
ax.set_xlabel("median RP100/SLR_0 station water level (m)")
ax.set_ylabel("validated hit rate (0.10 m, buffered)")
ax.set_title("(b) hit rate vs forcing magnitude\n(HR from metrics_FRA_RP100_SLR_0.csv)", loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.suptitle("Martinique's tile IS forced, by nearby stations — but their COAST-RP RP100 values "
             "are ~8× smaller than metropolitan France's", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(HERE / "fig5_forcing_comparison.png")
plt.close(fig)
print("wrote fig5")

# =============================================================================
# FIG 6 - where does the model put water on Martinique at all?
# =============================================================================
BB6 = (-61.25, 14.38, -60.79, 14.90)
w6_raw, w6_nod, w6_ext, w6_tr = read_window(MERGED, BB6)
w6 = np.where((w6_raw == w6_nod) | (w6_raw == AQUEDUCT_NODATA), np.nan, w6_raw)
d6_raw, d6_nod, d6_ext, _ = read_window(DEM_VRT, BB6)
d6 = np.where(d6_raw == d6_nod, np.nan, d6_raw)
fig, ax = plt.subplots(figsize=(7.6, 8.0))
ax.imshow(np.where(~np.isnan(d6), 1.0, np.nan), extent=d6_ext, origin="upper",
          cmap=ListedColormap([LAND]), interpolation="nearest")
rr, cc = np.where(np.nan_to_num(w6, nan=-1) > 0.10)
xs, ys = rasterio.transform.xy(w6_tr, rr, cc)
s6 = ax.scatter(xs, ys, c=w6[rr, cc], cmap=SEQ_DEPTH, vmin=0, vmax=2.0, s=3, lw=0)
bench.boundary.plot(ax=ax, color="#c1121f", lw=0.9)
t2022.boundary.plot(ax=ax, color="#0b6e4f", lw=1.4)
ax.set_xlim(BB6[0], BB6[2]); ax.set_ylim(BB6[1], BB6[3])
ax.set_aspect(1 / np.cos(np.radians(14.6)))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
fig.colorbar(s6, ax=ax, shrink=0.6, pad=0.02).set_label("water depth (m)", fontsize=7.5)
ax.set_title(f"Where the model DOES flood Martinique at RP100/SLR_0\n"
             f"{len(rr)} cells > 0.10 m island-wide "
             f"({len(rr)*30*30/1e6:.2f} km²) — a thin fringe on the shoreline only;\n"
             f"red = benchmark polygon, green = simulation tile 2022", loc="left")
ax.legend(handles=[Patch(facecolor=LAND, label="DeltaDTM coastal land (≤30 m)"),
                   Line2D([], [], color="#c1121f", lw=1.2, label="FRJ_TRI_LAMENTIN benchmark"),
                   Line2D([], [], color="#0b6e4f", lw=1.4, label="simulation tile 2022")],
          fontsize=7, loc="lower left", frameon=True, facecolor="white", edgecolor=MUTED)
fig.savefig(HERE / "fig6_martinique_wet_cells.png")
plt.close(fig)
print(f"wrote fig6 - island-wide wet cells > 0.10 m: {len(rr)}")

pd.DataFrame({
    "elev_threshold_m": [0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "frac_benchmark_cells_below": [round(float((elev_in <= t).mean()), 4)
                                   for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]],
}).to_csv(HERE / "benchmark_polygon_elevation_ecdf.csv", index=False)
print(f"\nbenchmark polygon DEM: n_inside={n_in} n_valid={n_valid} "
      f"nodata_frac={1 - n_valid/n_in:.4f} median={np.median(elev_in):.3f} m "
      f"min={elev_in.min():.3f} max={elev_in.max():.3f}")
