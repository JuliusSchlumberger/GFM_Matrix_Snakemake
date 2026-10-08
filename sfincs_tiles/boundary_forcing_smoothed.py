"""Rewrites a built SFINCS tile's waterlevel boundary forcing (sfincs.bnd /
sfincs.bzs / gis/bnd.geojson) as a smoothed, nearest-boundary projection of
the tile's COAST-RP/COAST-HG stations - replacing hydromt's own
`water_level.create(buffer=...)` selection in build_sfincs_tile.py.

Why (tile 1974 investigation, 2026-10-08): build_sfincs_tile.py passes a
buffer of `2 x (farthest station's distance to the grid bbox) + 25 km`. For
stations inside the bbox that distance is 0, so hydromt keeps only stations
within 25 km of the boundary cells - on a tile whose open boundary is a long
offshore line, a couple of stations at one end then force the whole
boundary (tile 1974: 2 of 20 stations, peaks 2.46/1.31 m vs a 0.54 m station
median, flooding 219 km2 vs the eikonal's 22 km2). Projecting every station
onto its nearest boundary cell fixes that, but placing very different
stations a few km apart directly on the boundary caused numerical
oscillations (tile 1974: 2.6 m water-level swings next to the boundary).
Smoothing along the boundary removes those (0.6 m) at the same flood result
(CSI 0.28 vs 0.29 for the unsmoothed projection).

Method:
  1. Boundary cells (msk == 2) are read from sfincs.msk/sfincs.ind and placed
     with sfincs.inp's own (possibly rotated) grid geometry, thinned to one
     representative cell per THIN_M x THIN_M bin, and linked into a network:
     cells within STRETCH_LINK_M of each other are connected, edge weight =
     distance. Each connected part is one boundary stretch; distances "along
     the boundary" are shortest paths through this network - robust to
     straight, branched, two-cell-wide or gapped boundaries alike (hydromt's
     boundary along an offshore edge is runs of cells with ~1 km gaps).
  2. Stations (see `station_selection`) are projected onto their nearest
     boundary cell.
  3. Forcing points are placed per stretch by farthest-point sampling until
     every boundary cell is within spacing_km / 2 (network distance) of a
     point - neighbouring points <= spacing_km apart, never duplicated. Each
     point gets a Gaussian distance-weighted average (length scale
     `smooth_km`) of station hydrographs: on a stretch with stations
     projected onto it, only those stations, by network distance; on a
     stretch without any, all stations by straight-line distance from their
     real positions. Every forcing value is therefore a convex combination of
     station values - never above the highest or below the lowest station at
     that time.
Hydrographs are corrected_hydrographs.csv truncated to the same window
build_sfincs_tile.py uses; the time axis is checked against the original
sfincs.bzs so the run window is unchanged.

`station_selection`: "all" (default) uses every matched station
(matched_boundary_points.gpkg - the k nearest to the tile, the same set the
original hydromt forcing drew from); "bbox" uses only those inside the tile's
bbox (tile_geometry.gpkg), falling back to all if none is inside. "all" was
chosen after a dry run over all 1052 calibration tiles (2026-10-08): with
"bbox", 141 tiles had an ignored station much closer to some boundary stretch
than any used one (tile 374: nearest used station 263 km away), and tiles
with a single station inside let that one station force everything even when
it was the outlier (tile 529: 4.19 m vs 1.9-3.1 m for the other stations).

Usage:
    python boundary_forcing_smoothed.py --tile-dir <base>/<tile_id> [--stations bbox|all]
        [--smooth-km 25] [--spacing-km 25] [--reference-bzs <original sfincs.bzs>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402

TRUNCATE_WINDOW_HR = (40.0, 110.0)  # build_sfincs_tile.TRUNCATE_WINDOW_HR_DEFAULT
SMOOTH_KM_DEFAULT = 25.0
SPACING_KM_DEFAULT = 25.0
STATION_SELECTION_DEFAULT = "all"
# Boundary cells within this distance of each other are linked into one stretch.
# hydromt's all_touched boundary along an offshore domain edge comes out as runs
# of cells separated by 0.4-2.2 km gaps (calibration tiles 1974/2369/2744/3014),
# while genuinely separate boundaries were 12+ km apart.
STRETCH_LINK_M = 5000.0
# Boundary cells are thinned to one per THIN_M bin before building the network -
# keeps the shortest-path computations cheap on 1000+ km boundaries; well below
# the 25 km smoothing/spacing scales.
THIN_M = 500.0
BOUNDARY = 2


def _parse_inp(path: Path) -> dict:
    cfg = {}
    for line in retry_transient_io(path.read_text).splitlines():
        if "=" in line and not line.strip().startswith("!"):
            key, _, val = line.partition("=")
            cfg[key.strip().lower()] = val.strip()
    return cfg


def read_boundary_cells(sfincs_dir: Path) -> tuple[np.ndarray, int, float]:
    """Cell-centre x/y (model CRS) of every waterlevel-boundary cell, the
    model EPSG and the cell size (max of dx/dy, metres). sfincs.ind: uint32,
    first value N, then N 1-based column-major (mmax x nmax) indices;
    sfincs.msk: N uint8 values. Rotation is about (x0, y0), row 0 at y0."""
    inp = _parse_inp(sfincs_dir / "sfincs.inp")
    mmax, nmax = int(inp["mmax"]), int(inp["nmax"])
    dx, dy = float(inp["dx"]), float(inp["dy"])
    x0, y0 = float(inp["x0"]), float(inp["y0"])
    theta = np.deg2rad(float(inp.get("rotation", 0.0)))
    ind_raw = np.fromfile(sfincs_dir / "sfincs.ind", dtype="<u4")
    ind = ind_raw[1:1 + int(ind_raw[0])].astype(np.int64) - 1
    msk = np.fromfile(sfincs_dir / "sfincs.msk", dtype="u1")
    grid = np.zeros(mmax * nmax, dtype=np.uint8)
    grid[ind] = msk
    mask2d = grid.reshape(mmax, nmax).T  # (nrow=n, ncol=m), row 0 at y0
    rows, cols = np.nonzero(mask2d == BOUNDARY)
    u, v = (cols + 0.5) * dx, (rows + 0.5) * dy
    xy = np.column_stack([x0 + u * np.cos(theta) - v * np.sin(theta),
                          y0 + u * np.sin(theta) + v * np.cos(theta)])
    return xy, int(inp["epsg"]), max(dx, dy)


def thin_cells(xy: np.ndarray, bin_m: float) -> np.ndarray:
    """Indices of one representative cell per bin_m x bin_m bin (the cell
    closest to its bin centre) - always real boundary cells."""
    key = np.floor(xy / bin_m).astype(np.int64)
    d = np.hypot(*(xy - (key + 0.5) * bin_m).T)
    order = np.lexsort((d, key[:, 1], key[:, 0]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(np.diff(key[order], axis=0) != 0, axis=1)
    return np.sort(order[first])


def boundary_network(xy: np.ndarray, link_m: float):
    """Sparse distance-weighted graph linking cells within link_m, and its
    connected components (= boundary stretches)."""
    pairs = cKDTree(xy).query_pairs(link_m, output_type="ndarray")
    w = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) if len(pairs) else np.zeros(0)
    graph = coo_matrix((w, (pairs[:, 0], pairs[:, 1])) if len(pairs) else (w, (np.zeros(0, int), np.zeros(0, int))),
                       shape=(len(xy), len(xy))).tocsr()
    n, lab = connected_components(graph, directed=False)
    return graph, n, lab


def place_points(graph, lab: np.ndarray, spacing_m: float) -> tuple[list[int], np.ndarray]:
    """Farthest-point sampling per stretch until every cell is within
    spacing_m / 2 (network distance) of a point. Returns (point cell indices,
    per-cell network distance to its nearest point)."""
    points, cover = [], np.full(len(lab), np.inf)
    for c in np.unique(lab):
        members = np.nonzero(lab == c)[0]
        if len(members) == 1:
            points.append(int(members[0]))
            cover[members] = 0.0
            continue
        d0 = dijkstra(graph, directed=False, indices=int(members[0]))
        cur = int(members[np.argmax(d0[members])])  # one end of the stretch
        dmin = dijkstra(graph, directed=False, indices=cur)
        points.append(cur)
        while dmin[members].max() > spacing_m / 2.0:
            cur = int(members[np.argmax(dmin[members])])
            points.append(cur)
            dmin = np.minimum(dmin, dijkstra(graph, directed=False, indices=cur))
        cover[members] = dmin[members]
    return points, cover


def build_smoothed_forcing(
    tile_dir: Path, smooth_km: float = SMOOTH_KM_DEFAULT, spacing_km: float = SPACING_KM_DEFAULT,
    reference_bzs: Path | None = None, station_selection: str = STATION_SELECTION_DEFAULT,
) -> dict:
    """Returns {"bnd_xy", "bzs", "epsg", "info", ...diagnostics} for the tile -
    see module docstring."""
    if station_selection not in ("bbox", "all"):
        raise ValueError(f"station_selection must be 'bbox' or 'all', got {station_selection!r}")
    sfincs_dir = tile_dir / "sfincs_model"
    all_xy, epsg, cell_m = read_boundary_cells(sfincs_dir)
    if len(all_xy) == 0:
        raise RuntimeError(f"{tile_dir.name}: no waterlevel boundary cells in sfincs.msk")
    xy = all_xy[thin_cells(all_xy, max(THIN_M, cell_m))]
    graph, n_stretches, lab = boundary_network(xy, max(STRETCH_LINK_M, 3.0 * cell_m))

    pts = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    hydro = retry_transient_io(pd.read_csv, sfincs_dir / "corrected_hydrographs.csv")
    hydro = hydro.loc[hydro.elapsed_hr.between(*TRUNCATE_WINDOW_HR)].reset_index(drop=True)
    t_s = (hydro.elapsed_hr - hydro.elapsed_hr.iloc[0]).to_numpy() * 3600.0
    if reference_bzs is not None:
        ref_t = np.loadtxt(reference_bzs, ndmin=2)[:, 0]
        if len(ref_t) != len(t_s) or not np.allclose(ref_t, t_s):
            raise RuntimeError(f"{tile_dir.name}: truncated hydrograph time axis does not match {reference_bzs}")

    used_all = True
    station_ids = np.arange(len(pts))
    if station_selection == "bbox":
        minx, miny, maxx, maxy = retry_transient_io(gpd.read_file, tile_dir / "inputs" / "tile_geometry.gpkg").to_crs(4326).total_bounds
        p4326 = pts.to_crs(4326)
        inside = np.nonzero((p4326.geometry.x.between(minx, maxx) & p4326.geometry.y.between(miny, maxy)).to_numpy())[0]
        if len(inside):
            station_ids, used_all = inside, False
    pm = pts.to_crs(epsg)
    st_real_xy = np.column_stack([pm.geometry.x, pm.geometry.y])[station_ids]
    moved_m, st_cell = cKDTree(xy).query(st_real_xy)
    st_stretch = lab[st_cell]
    H = hydro[[str(k) for k in station_ids]].to_numpy()  # (time, station)

    points, cover = place_points(graph, lab, spacing_km * 1000.0)
    points = np.array(points)
    pt_stretch = lab[points]
    # network distance from every station's boundary cell to every forcing point (inf across stretches)
    d_st = dijkstra(graph, directed=False, indices=st_cell)[:, points].T / 1000.0  # (point, station)
    own = (pt_stretch[:, None] == st_stretch[None, :]).any(axis=1)
    d_km = np.where(pt_stretch[:, None] == st_stretch[None, :], d_st, np.inf)
    d_km[~own] = np.hypot(*(xy[points][~own, None, :] - st_real_xy[None, :, :]).transpose(2, 0, 1)) / 1000.0
    d2 = d_km ** 2
    W = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / smooth_km ** 2)  # shifted: no all-zero rows; exp(-inf) = 0
    W /= W.sum(axis=1, keepdims=True)
    series = H @ W.T  # (time, point)

    d_pp = dijkstra(graph, directed=False, indices=points)[:, points] / 1000.0  # (point, point), inf across stretches
    np.fill_diagonal(d_pp, np.inf)
    peaks = series.max(axis=0)
    info = {
        "n_boundary_cells": int(len(all_xy)), "n_network_cells": int(len(xy)), "n_stretches": int(n_stretches),
        "n_stations_used": int(len(station_ids)), "n_matched_stations": int(len(pts)),
        "station_selection": station_selection, "used_all_matched_stations": bool(used_all),
        "n_points": int(len(points)), "peak_min_m": float(peaks.min()), "peak_max_m": float(peaks.max()),
        "station_peak_median_m": float(np.median(H.max(axis=0))),
        "max_coverage_km": float(cover.max() / 1000.0),
    }
    return {"bnd_xy": xy[points], "bzs": np.column_stack([t_s, series]), "epsg": epsg, "info": info,
            "point_stretch": pt_stretch,
            # diagnostics (not written)
            "point_dist_km": d_pp, "station_ids": station_ids, "station_stretch": st_stretch,
            "station_moved_km": moved_m / 1000.0, "station_hydrographs": H, "weights": W,
            "network_xy": xy, "network_stretch": lab, "network_coverage_km": cover / 1000.0}


def write_forcing(out_dir: Path, forcing: dict) -> None:
    n = forcing["bnd_xy"].shape[0]
    np.savetxt(out_dir / "sfincs.bnd", forcing["bnd_xy"], fmt="%11.1f")
    np.savetxt(out_dir / "sfincs.bzs", forcing["bzs"], fmt=["%9.1f"] + ["%7.3f"] * n)
    (out_dir / "gis").mkdir(exist_ok=True)
    gpd.GeoDataFrame(
        {"index": range(n), "stretch": forcing["point_stretch"],
         "peak_m": np.round(forcing["bzs"][:, 1:].max(axis=0), 3)},
        geometry=gpd.points_from_xy(forcing["bnd_xy"][:, 0], forcing["bnd_xy"][:, 1]), crs=forcing["epsg"],
    ).to_file(out_dir / "gis" / "bnd.geojson", driver="GeoJSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--stations", choices=["bbox", "all"], default=STATION_SELECTION_DEFAULT)
    parser.add_argument("--smooth-km", type=float, default=SMOOTH_KM_DEFAULT)
    parser.add_argument("--spacing-km", type=float, default=SPACING_KM_DEFAULT)
    parser.add_argument("--reference-bzs", default=None, help="original sfincs.bzs whose time axis must be reproduced")
    parser.add_argument("--out-dir", default=None, help="default: <tile-dir>/sfincs_model")
    args = parser.parse_args()

    tile_dir = Path(args.tile_dir)
    forcing = build_smoothed_forcing(tile_dir, args.smooth_km, args.spacing_km,
                                     Path(args.reference_bzs) if args.reference_bzs else None, args.stations)
    out_dir = Path(args.out_dir) if args.out_dir else tile_dir / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_forcing(out_dir, forcing)
    i = forcing["info"]
    print(f"tile {tile_dir.name}: {i['n_stations_used']}/{i['n_matched_stations']} station(s) "
          f"[{i['station_selection']}{', none inside bbox - all used' if i['used_all_matched_stations'] and i['station_selection'] == 'bbox' else ''}] "
          f"-> {i['n_points']} forcing point(s) on {i['n_stretches']} boundary stretch(es); "
          f"peaks {i['peak_min_m']:.2f}-{i['peak_max_m']:.2f} m (station median {i['station_peak_median_m']:.2f} m)")


if __name__ == "__main__":
    main()
