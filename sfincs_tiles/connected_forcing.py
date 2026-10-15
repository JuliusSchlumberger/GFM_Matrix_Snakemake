"""Interior forcing points: COAST-RP stations inside the tile that are NOT a node of the boundary line
(dropped while building it - closed bay, dangling line, shortcut, irrelevant, not open ocean, unlinked)
but lie by water that stays active in SFINCS. Without them such water (a lagoon, a bay behind a
shortcut, an archipelago the line gave up on) is forced only from the line's nodes further out,
not by its own stations.

Rule (along every boundary, like the eikonal's own seeding): each such station becomes a forcing
point at the nearest open-water pixel of the tile's native mask (within STATION_OCEAN_TOL_M), with
  - level: the station's own COAST-RP RP100 level (MDT-corrected, as for the line's nodes);
  - hydrograph shape: the nearest COAST-HG station that is hydrologically connected to it - in the
    same 8-connected ocean water body of the DeltaDTM mask over the production station search box
    (build_station_boundary_lines._search_box / _read_mask_window); a COAST-HG station at the
    station's own position (they share most positions) is taken directly.
Stations closer than LINE_CLEARANCE_M to the boundary line are left out (forced by the line already),
as are stations in the line's inactive (sea-side) faces. build_sfincs_tile.py makes a small inactive
core at each point with a ring of waterlevel-boundary cells around it (same construction as the line:
inactive on one side, boundary cells along it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_station_boundary_lines as bsl  # noqa: E402

STATION_OCEAN_TOL_M = 2000.0   # station -> nearest open-water pixel of the tile's native mask
LINE_CLEARANCE_M = 1000.0      # stations this close to the boundary line are already forced by it
SAME_POSITION_DEG = 0.01       # a COAST-HG station this close counts as at the station itself
SEARCH_BUFFER_DEG, SEARCH_MIN_SIZE_DEG = 1.0, 2.0  # production station search box (config boundary_conditions)


def _snap_to_ocean(lon: float, lat: float, mask: np.ndarray, transform) -> tuple[float, float] | None:
    """Centre of the nearest ocean pixel of the native mask within STATION_OCEAN_TOL_M, or None."""
    px_m = abs(transform.e) * 110570.0
    w = int(np.ceil(STATION_OCEAN_TOL_M / px_m))
    r, c = rasterio.transform.rowcol(transform, lon, lat)
    r0, c0 = max(r - w, 0), max(c - w, 0)
    win = mask[r0:r + w + 1, c0:c + w + 1] == bsl.OCEAN_CODE
    rr, cc = np.nonzero(win)
    if not len(rr):
        return None
    d = np.hypot((rr + r0 - r) * px_m, (cc + c0 - c) * px_m * np.cos(np.radians(lat)))
    k = int(np.argmin(d))
    if d[k] > STATION_OCEAN_TOL_M:
        return None
    x, y = rasterio.transform.xy(transform, rr[k] + r0, cc[k] + c0)
    return float(x), float(y)


def _components(root: Path, tile: gpd.GeoDataFrame):
    """(labels, transform) of the 8-connected ocean water bodies over the station search box."""
    bounds = bsl._search_box(tile, SEARCH_BUFFER_DEG, SEARCH_MIN_SIZE_DEG)
    mask, tr = bsl._read_mask_window(root / bsl.MASK_VRT, bounds)
    labels, _ = ndimage.label(mask == bsl.OCEAN_CODE, structure=np.ones((3, 3)))
    return labels, tr


def _label_at(labels, tr, lon, lat, r_px: int) -> int:
    """Label of the nearest ocean pixel within r_px of (lon, lat), 0 if none."""
    r, c = rasterio.transform.rowcol(tr, lon, lat)
    h, w = labels.shape
    if not (0 <= r < h and 0 <= c < w):
        return 0
    r0, c0 = max(r - r_px, 0), max(c - r_px, 0)
    win = labels[r0:r + r_px + 1, c0:c + r_px + 1]
    rr, cc = np.nonzero(win)
    if not len(rr):
        return 0
    k = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
    return int(win[rr[k], cc[k]])


def interior_points(lines_gpkg: Path, tile_dir: Path, root: Path, hg_xy: np.ndarray) -> gpd.GeoDataFrame:
    """Interior forcing points (EPSG:4326) with station, rp100_level_m, coast_hg_station,
    coast_hg_dist_deg, kind="interior". `tile_dir` is the tile's base dir (with inputs/),
    `hg_xy` the COAST-HG station lon/lat array."""
    cols = ["station", "station_index", "rp100_level_m", "coast_hg_station", "coast_hg_dist_deg", "kind"]
    empty = gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs="EPSG:4326")
    st = gpd.read_file(lines_gpkg, layer="stations")
    tile = gpd.read_file(tile_dir / "inputs" / "tile_geometry.gpkg").to_crs("EPSG:4326")
    tile_geom = tile.geometry.iloc[0]
    cand = st[(st["status"] != "used") & st.within(tile_geom)]
    if cand.empty:
        return empty
    layers = pyogrio.list_layers(lines_gpkg)[:, 0]
    inactive = (gpd.read_file(lines_gpkg, layer="inactive").union_all() if "inactive" in layers else None)
    utm = tile.estimate_utm_crs()
    line_utm = gpd.read_file(lines_gpkg, layer="boundary_line").to_crs(utm).union_all()
    with rasterio.open(tile_dir / "inputs" / "mask.tif") as src:
        mask, mtr = src.read(1), src.transform

    tree = cKDTree(hg_xy)
    labels = ltr = None
    rows = []
    for _, r in cand.iterrows():
        snap = _snap_to_ocean(r.geometry.x, r.geometry.y, mask, mtr)
        if snap is None:
            continue
        pt = shapely.Point(snap)
        if inactive is not None and inactive.contains(pt):
            continue
        if gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(utm).iloc[0].distance(line_utm) < LINE_CLEARANCE_M:
            continue
        d, j = tree.query([r.geometry.x, r.geometry.y])
        if d > SAME_POSITION_DEG:  # nearest COAST-HG station in the same water body (search box)
            if labels is None:
                labels, ltr = _components(root, tile)
            r_px = int(np.ceil(STATION_OCEAN_TOL_M / (abs(ltr.e) * 110570.0)))
            own = _label_at(labels, ltr, *snap, r_px)
            b = bsl._search_box(tile, SEARCH_BUFFER_DEG, SEARCH_MIN_SIZE_DEG)
            inbox = np.nonzero((hg_xy[:, 0] >= b[0]) & (hg_xy[:, 0] <= b[2]) & (hg_xy[:, 1] >= b[1]) & (hg_xy[:, 1] <= b[3]))[0]
            same = [k for k in inbox if own and _label_at(labels, ltr, hg_xy[k, 0], hg_xy[k, 1], r_px) == own]
            if not same:
                print(f"  station {r['station']}: no hydrologically connected COAST-HG station in the search box - skipped")
                continue
            dd = np.hypot(hg_xy[same, 0] - r.geometry.x, hg_xy[same, 1] - r.geometry.y)
            j, d = int(same[int(np.argmin(dd))]), float(dd.min())
        rows.append({"station": int(r["station"]), "station_index": int(r["station_index"]),
                     "rp100_level_m": float(r["rp100_level_m"]), "coast_hg_station": int(j),
                     "coast_hg_dist_deg": round(float(d), 4), "kind": "interior", "geometry": pt})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else empty
