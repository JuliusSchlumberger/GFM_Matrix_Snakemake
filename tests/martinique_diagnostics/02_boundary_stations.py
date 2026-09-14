"""Step 2: inspect the REAL COAST-RP boundary stations that forced Martinique's
simulation tiles (2022, 2031) at RP100/SLR_0, and compare against baselines
(Guadeloupe 1515/1958, St Lucia 2429, a metropolitan-France tile).

Reads the actual on-disk per-tile boundary GeoPackages written by
scripts/extract_boundaries.py, plus the global station cache, and re-runs
select_stations_for_tile so the candidate pool (before the ocean-connectivity
filter) can be compared against what was actually kept.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from boundaries import load_waterlevel_stations, select_stations_for_tile  # noqa: E402
from rasters import decode_waterlevel_cm  # noqa: E402

ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
MANIFEST = ROOT / "processed_inputs/mask/domain_tiles_global.gpkg"
# The per-scenario station cache is mostly absent on this machine (only
# stations_RP1000_SLR_500.gpkg exists), so read the same source the cache rule
# reads: the scenario NetCDF itself.
CACHE = ROOT / "processed_inputs/WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc"
MODEL_OUT = ROOT / "model_outputs"

BUFFER_DEG = 1.0
MIN_SEARCH_DEG = 2.0
WL_NAME = "SLR_0"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

tiles = gpd.read_file(MANIFEST)
stations = load_waterlevel_stations(
    CACHE, variable="COAST-RP_EWL_RP100_SLR_0",
    x_var="station_x_coordinate", y_var="station_y_coordinate", column_name="SLR_0",
)
print(f"global station cache: {CACHE}")
print(f"  {len(stations)} stations, columns={list(stations.columns)}, crs={stations.crs}")
wl_col = [c for c in stations.columns if c != "geometry"][0]
print(f"  water-level column: {wl_col}; min={stations[wl_col].min():.3f} "
      f"max={stations[wl_col].max():.3f} m")

# reference points
FDF = (-61.07, 14.60)            # Fort-de-France / Lamentin (benchmark centre)
MART_CENTRE = (-61.02, 14.64)

def km(dx_deg, dy_deg, lat):
    return np.hypot(dx_deg * 111.32 * np.cos(np.radians(lat)), dy_deg * 110.57)

rows = []
per_tile_stations = {}

TILES_OF_INTEREST = {
    2022: "Martinique (tile A)",
    2031: "Martinique (tile B)",
    2429: "St Lucia",
    2185: "St Vincent / Grenadines",
    1515: "Guadeloupe A",
    1958: "Guadeloupe B",
}

for tid, label in TILES_OF_INTEREST.items():
    tile = tiles[tiles.tile_id == tid]
    if tile.empty:
        print(f"tile {tid} not in manifest"); continue
    b = tile.total_bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    cand = select_stations_for_tile(stations, tile, BUFFER_DEG, MIN_SEARCH_DEG)

    bpath = MODEL_OUT / str(tid) / "inputs" / f"boundaries_RP100_{WL_NAME}.gpkg"
    if bpath.exists():
        used = gpd.read_file(bpath)
        ucol = [c for c in used.columns if c != "geometry"][0]
        used_wl = decode_waterlevel_cm(used[ucol].to_numpy()) if len(used) else np.array([])
    else:
        used = gpd.GeoDataFrame(geometry=[], crs=4326); used_wl = np.array([])

    d_tile = km(cand.geometry.x - cx, cand.geometry.y - cy, cy) if len(cand) else np.array([])
    d_used = km(used.geometry.x - cx, used.geometry.y - cy, cy) if len(used) else np.array([])
    d_used_fdf = km(used.geometry.x - FDF[0], used.geometry.y - FDF[1], FDF[1]) if len(used) else np.array([])

    print(f"\n===== tile {tid} — {label} =====")
    print(f"  bbox = {tuple(round(v,4) for v in b)}  centre=({cx:.4f},{cy:.4f})")
    print(f"  candidates in search box (pre-connectivity-filter): {len(cand)}")
    print(f"  stations actually written to {bpath.name}: {len(used)}")
    if len(used):
        print(f"  distance of used stations to tile centre (km): "
              f"min={d_used.min():.1f} med={np.median(d_used):.1f} max={d_used.max():.1f}")
        print(f"  water level RP100/SLR_0 of used stations (m): "
              f"min={used_wl.min():.3f} med={np.median(used_wl):.3f} max={used_wl.max():.3f}")
        det = pd.DataFrame({
            "lon": used.geometry.x.round(4), "lat": used.geometry.y.round(4),
            "waterlevel_m": np.round(used_wl, 3),
            "dist_to_tile_centre_km": np.round(d_used, 1),
            "dist_to_FortDeFrance_km": np.round(d_used_fdf, 1),
        }).sort_values("dist_to_tile_centre_km")
        print(det.to_string(index=False))
        det.insert(0, "tile_id", tid); det.insert(1, "label", label)
        per_tile_stations[tid] = det
    rows.append({
        "tile_id": tid, "label": label,
        "bbox": tuple(round(v, 4) for v in b),
        "n_candidates": len(cand), "n_used": len(used),
        "min_dist_km": round(float(d_used.min()), 1) if len(used) else None,
        "med_dist_km": round(float(np.median(d_used)), 1) if len(used) else None,
        "max_dist_km": round(float(d_used.max()), 1) if len(used) else None,
        "wl_min_m": round(float(used_wl.min()), 3) if len(used) else None,
        "wl_med_m": round(float(np.median(used_wl)), 3) if len(used) else None,
        "wl_max_m": round(float(used_wl.max()), 3) if len(used) else None,
    })

# --- metropolitan France baseline: a tile near the Gironde/Atlantic coast -----
from shapely.geometry import box  # noqa: E402
for name, bx in {
    "metropole_Gironde": box(-1.4, 45.3, -0.9, 45.8),
    "metropole_Channel": box(-1.8, 49.2, -1.2, 49.7),
}.items():
    sel = tiles[tiles.intersects(bx)]
    for _, t in sel.iterrows():
        tid = int(t.tile_id)
        tile = tiles[tiles.tile_id == tid]
        b = tile.total_bounds; cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        bpath = MODEL_OUT / str(tid) / "inputs" / f"boundaries_RP100_{WL_NAME}.gpkg"
        if not bpath.exists():
            continue
        used = gpd.read_file(bpath)
        if used.empty:
            rows.append({"tile_id": tid, "label": name, "bbox": tuple(round(v,4) for v in b),
                         "n_candidates": None, "n_used": 0}); continue
        ucol = [c for c in used.columns if c != "geometry"][0]
        uwl = decode_waterlevel_cm(used[ucol].to_numpy())
        d = km(used.geometry.x - cx, used.geometry.y - cy, cy)
        rows.append({
            "tile_id": tid, "label": name, "bbox": tuple(round(v, 4) for v in b),
            "n_candidates": None, "n_used": len(used),
            "min_dist_km": round(float(d.min()),1), "med_dist_km": round(float(np.median(d)),1),
            "max_dist_km": round(float(d.max()),1),
            "wl_min_m": round(float(uwl.min()),3), "wl_med_m": round(float(np.median(uwl)),3),
            "wl_max_m": round(float(uwl.max()),3),
        })

summary = pd.DataFrame(rows)
print("\n\n================ SUMMARY ================")
print(summary.to_string(index=False))
summary.to_csv(HERE / "boundary_station_summary.csv", index=False)
if per_tile_stations:
    pd.concat(per_tile_stations.values()).to_csv(HERE / "boundary_stations_detail.csv", index=False)

# --- nearest stations to Fort-de-France in the WHOLE global cache -------------
dall = km(stations.geometry.x - FDF[0], stations.geometry.y - FDF[1], FDF[1])
near = stations.assign(dist_km=dall).nsmallest(20, "dist_km")
near_out = pd.DataFrame({
    "lon": near.geometry.x.round(4), "lat": near.geometry.y.round(4),
    "waterlevel_m": near[wl_col].round(3), "dist_to_FDF_km": near["dist_km"].round(1),
})
print("\n=== 20 nearest COAST-RP stations (global cache) to Fort-de-France ===")
print(near_out.to_string(index=False))
near_out.to_csv(HERE / "nearest_stations_to_fort_de_france.csv", index=False)
