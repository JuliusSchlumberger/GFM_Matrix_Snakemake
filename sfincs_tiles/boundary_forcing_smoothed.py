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
     with sfincs.inp's own (possibly rotated) grid geometry, split into
     stretches (cells within STRETCH_LINK_M of each other) and ordered along
     each stretch, so every cell has an along-boundary position (km).
  2. Every station inside the tile's bbox (tile_geometry.gpkg; all matched
     stations if none is inside) is projected onto its nearest boundary cell.
  3. Forcing points are placed every `spacing_km` along each stretch. Each
     gets a Gaussian distance-weighted average (length scale `smooth_km`) of
     every station's hydrograph - distance measured along the boundary for
     stations projected onto the same stretch, straight-line otherwise.
Hydrographs are corrected_hydrographs.csv truncated to the same window
build_sfincs_tile.py uses; the time axis is checked against the original
sfincs.bzs so the run window is unchanged.

Usage:
    python boundary_forcing_smoothed.py --tile-dir <base>/<tile_id> [--smooth-km 25] [--spacing-km 25]
        [--reference-bzs <original sfincs.bzs>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402

TRUNCATE_WINDOW_HR = (40.0, 110.0)  # build_sfincs_tile.TRUNCATE_WINDOW_HR_DEFAULT
SMOOTH_KM_DEFAULT = 25.0
SPACING_KM_DEFAULT = 25.0
# Boundary cells within this distance of each other form one stretch. hydromt's
# all_touched boundary along an offshore domain edge comes out as runs of cells
# separated by 0.4-2.2 km gaps (found on calibration tiles 1974/2369/2744/3014),
# while genuinely separate boundaries were 12+ km apart.
STRETCH_LINK_M = 5000.0
BOUNDARY = 2


def _parse_inp(path: Path) -> dict:
    cfg = {}
    for line in retry_transient_io(path.read_text).splitlines():
        if "=" in line and not line.strip().startswith("!"):
            key, _, val = line.partition("=")
            cfg[key.strip().lower()] = val.strip()
    return cfg


def read_boundary_cells(sfincs_dir: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Boundary cells as (n_cells, 2) (row, col) indices - row = SFINCS n
    (row 0 at y0), col = m - plus their cell-centre x/y in the model CRS
    and the model EPSG. sfincs.ind: uint32, first value N, then N 1-based
    column-major (mmax x nmax) indices; sfincs.msk: N uint8 values. Also
    returns the cell size (max of dx/dy, metres)."""
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
    return np.column_stack([rows, cols]), xy, int(inp["epsg"]), max(dx, dy)


def order_stretches(xy: np.ndarray, link_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits boundary cells into stretches and orders each along its length.
    Cells within `link_m` of each other belong to the same stretch - real
    distance, not grid adjacency: hydromt's boundary along an offshore edge
    is runs of cells separated by ~1 km gaps (tile 1974's single offshore
    line would otherwise split into 18 pieces - see STRETCH_LINK_M). Each stretch
    is ordered by a nearest-unvisited-neighbour walk from one end. Returns
    (order, stretch_id, arc_km) - `order` indexes xy, stretch_id/arc_km are
    per ordered cell."""
    pairs = cKDTree(xy).query_pairs(link_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    n, lab = connected_components(graph, directed=False)

    order, stretch, arc = [], [], []
    for s in range(n):
        members = np.nonzero(lab == s)[0]
        pts = xy[members]
        tree = cKDTree(pts)
        cur = int(np.argmax(np.hypot(*(pts - pts.mean(axis=0)).T)))  # an end of the stretch
        seen = np.zeros(len(members), dtype=bool)
        seq = [cur]
        seen[cur] = True
        for _ in range(len(members) - 1):
            _, cand = tree.query(pts[cur], k=min(len(members), 32))
            nxt = next((int(j) for j in np.atleast_1d(cand) if not seen[j]), None)
            if nxt is None:  # all close neighbours visited: nearest unvisited anywhere in the stretch
                rest = np.nonzero(~seen)[0]
                nxt = int(rest[np.argmin(np.hypot(*(pts[rest] - pts[cur]).T))])
            seq.append(nxt)
            seen[nxt] = True
            cur = nxt
        steps = np.hypot(*np.diff(pts[seq], axis=0).T) / 1000.0
        order += list(members[seq])
        stretch += [s] * len(seq)
        arc += list(np.concatenate([[0.0], np.cumsum(steps)]))
    return np.array(order), np.array(stretch), np.array(arc)


def build_smoothed_forcing(
    tile_dir: Path, smooth_km: float = SMOOTH_KM_DEFAULT, spacing_km: float = SPACING_KM_DEFAULT,
    reference_bzs: Path | None = None,
) -> dict:
    """Returns {"bnd_xy", "bzs", "epsg", "info"} for the tile - see module docstring."""
    sfincs_dir = tile_dir / "sfincs_model"
    _, xy, epsg, cell_m = read_boundary_cells(sfincs_dir)
    if len(xy) == 0:
        raise RuntimeError(f"{tile_dir.name}: no waterlevel boundary cells in sfincs.msk")
    order, stretch, arc = order_stretches(xy, link_m=max(STRETCH_LINK_M, 3.0 * cell_m))
    cells = xy[order]

    pts = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    hydro = retry_transient_io(pd.read_csv, sfincs_dir / "corrected_hydrographs.csv")
    hydro = hydro.loc[hydro.elapsed_hr.between(*TRUNCATE_WINDOW_HR)].reset_index(drop=True)
    t_s = (hydro.elapsed_hr - hydro.elapsed_hr.iloc[0]).to_numpy() * 3600.0
    if reference_bzs is not None:
        ref_t = np.loadtxt(reference_bzs, ndmin=2)[:, 0]
        if len(ref_t) != len(t_s) or not np.allclose(ref_t, t_s):
            raise RuntimeError(f"{tile_dir.name}: truncated hydrograph time axis does not match {reference_bzs}")

    minx, miny, maxx, maxy = retry_transient_io(gpd.read_file, tile_dir / "inputs" / "tile_geometry.gpkg").to_crs(4326).total_bounds
    p4326 = pts.to_crs(4326)
    inside = np.nonzero((p4326.geometry.x.between(minx, maxx) & p4326.geometry.y.between(miny, maxy)).to_numpy())[0]
    used_all = len(inside) == 0
    if used_all:
        inside = np.arange(len(pts))
    pm = pts.to_crs(epsg)
    _, nearest = cKDTree(cells).query(np.column_stack([pm.geometry.x, pm.geometry.y])[inside])
    st_stretch, st_arc, st_xy = stretch[nearest], arc[nearest], cells[nearest]
    H = hydro[[str(k) for k in inside]].to_numpy()  # (time, station)

    pt_xy, pt_stretch, pt_arc = [], [], []
    for s in np.unique(stretch):
        sel = np.nonzero(stretch == s)[0]
        length = arc[sel].max()
        n_pts = int(np.floor(length / spacing_km)) + 1 if length > 0 else 1
        targets = np.linspace(0.0, length, max(n_pts, 2)) if length > 0 else np.array([0.0])
        for a in targets:
            k = sel[min(np.searchsorted(arc[sel], a), len(sel) - 1)]
            pt_xy.append(cells[k]); pt_stretch.append(s); pt_arc.append(arc[k])
    pt_xy, pt_stretch, pt_arc = np.array(pt_xy), np.array(pt_stretch), np.array(pt_arc)

    same = pt_stretch[:, None] == st_stretch[None, :]
    d_km = np.where(same, np.abs(pt_arc[:, None] - st_arc[None, :]),
                    np.hypot(*(pt_xy[:, None, :] - st_xy[None, :, :]).transpose(2, 0, 1)) / 1000.0)
    d2 = d_km ** 2
    W = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / smooth_km ** 2)  # shifted: no all-zero rows
    W /= W.sum(axis=1, keepdims=True)
    series = H @ W.T  # (time, point)

    peaks = series.max(axis=0)
    info = {
        "n_boundary_cells": int(len(cells)), "n_stretches": int(len(np.unique(stretch))),
        "boundary_length_km": float(sum(arc[stretch == s].max() for s in np.unique(stretch))),
        "n_stations_used": int(len(inside)), "used_all_matched_stations": bool(used_all),
        "n_points": int(len(pt_xy)), "peak_min_m": float(peaks.min()), "peak_max_m": float(peaks.max()),
        "station_peak_median_m": float(np.median(H.max(axis=0))),
    }
    return {"bnd_xy": pt_xy, "bzs": np.column_stack([t_s, series]), "epsg": epsg, "info": info,
            "point_arc_km": pt_arc, "point_stretch": pt_stretch}


def write_forcing(out_dir: Path, forcing: dict) -> None:
    n = forcing["bnd_xy"].shape[0]
    np.savetxt(out_dir / "sfincs.bnd", forcing["bnd_xy"], fmt="%11.1f")
    np.savetxt(out_dir / "sfincs.bzs", forcing["bzs"], fmt=["%9.1f"] + ["%7.3f"] * n)
    (out_dir / "gis").mkdir(exist_ok=True)
    gpd.GeoDataFrame(
        {"index": range(n), "stretch": forcing["point_stretch"], "arc_km": np.round(forcing["point_arc_km"], 1),
         "peak_m": np.round(forcing["bzs"][:, 1:].max(axis=0), 3)},
        geometry=gpd.points_from_xy(forcing["bnd_xy"][:, 0], forcing["bnd_xy"][:, 1]), crs=forcing["epsg"],
    ).to_file(out_dir / "gis" / "bnd.geojson", driver="GeoJSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--smooth-km", type=float, default=SMOOTH_KM_DEFAULT)
    parser.add_argument("--spacing-km", type=float, default=SPACING_KM_DEFAULT)
    parser.add_argument("--reference-bzs", default=None, help="original sfincs.bzs whose time axis must be reproduced")
    parser.add_argument("--out-dir", default=None, help="default: <tile-dir>/sfincs_model")
    args = parser.parse_args()

    tile_dir = Path(args.tile_dir)
    forcing = build_smoothed_forcing(tile_dir, args.smooth_km, args.spacing_km,
                                     Path(args.reference_bzs) if args.reference_bzs else None)
    out_dir = Path(args.out_dir) if args.out_dir else tile_dir / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_forcing(out_dir, forcing)
    i = forcing["info"]
    print(f"tile {tile_dir.name}: {i['n_stations_used']} station(s){' (all matched - none inside bbox)' if i['used_all_matched_stations'] else ''} "
          f"-> {i['n_points']} forcing point(s) on {i['n_stretches']} boundary stretch(es), {i['boundary_length_km']:.0f} km; "
          f"peaks {i['peak_min_m']:.2f}-{i['peak_max_m']:.2f} m (station median {i['station_peak_median_m']:.2f} m)")


if __name__ == "__main__":
    main()
