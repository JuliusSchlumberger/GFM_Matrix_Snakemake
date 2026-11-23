"""SFINCS forced at the eikonal's coastline cells (variant C1 of findings_coast_boundary.md).

Boundary cells (msk 2) are the coastline cells of the native mask coarsened to the SFINCS grid:
ocean next to land or river in sea reaching the domain edge (subgrid_coast.subgrid_coastline_mask),
or, with an offset of k cells, the open-sea cells k cells (8-connected) seaward of them, the sea in
between active. The rest of that sea is inactive, everything else active. Each boundary cell's series peaks at
production's seed level (build_line_forcing.production_seed_levels, nearest seed) with the
blended hydrograph shape (build_line_forcing.blended_shapes). The forcing points are thinned, then
densified until SFINCS's own two-point interpolation reproduces every boundary cell's series within
INTERP_TOL_M. Used by build_sfincs_tile.py --boundary-mode coast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_line_forcing import (  # noqa: E402
    blended_shapes, level_at_nearest_seed, peak_relative_shapes, production_seed_levels, read_hydrographs,
)
from subgrid_coast import NODATA_CODE, OCEAN_CODE, subgrid_coastline_mask  # noqa: E402

OCEAN_FRACTION_MIN = 0.5
MSK_INACTIVE, MSK_ACTIVE, MSK_BOUNDARY = 0, 1, 2
THIN_BLOCK_CELLS = 4  # first guess: one forcing point per 4 x 4 block of SFINCS cells
INTERP_TOL_M = 0.05   # max |SFINCS-interpolated - own series| over the window, per boundary cell
MAX_DENSIFY_ROUNDS = 50
BLEND_CHUNK = 2000    # boundary cells per blending chunk (memory: chunk x knn x time steps)
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def coarsen_mask(mask: np.ndarray, factor: int, ocean_code: int = OCEAN_CODE) -> np.ndarray:
    """Coarsen a class mask by `factor`: ocean where >= OCEAN_FRACTION_MIN of the block is ocean.

    Other blocks take their most frequent non-ocean class. Trailing rows/columns that do not fill
    a block are dropped.
    """
    h, w = (mask.shape[0] // factor) * factor, (mask.shape[1] // factor) * factor
    blocks = mask[:h, :w].reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(h // factor, w // factor, factor * factor).astype(np.int64)
    ocean = (blocks == ocean_code).mean(axis=2) >= OCEAN_FRACTION_MIN
    non_ocean = [c for c in np.unique(blocks) if c != ocean_code]
    if not non_ocean:
        return np.full(ocean.shape, ocean_code, dtype=mask.dtype)
    counts = np.stack([(blocks == c).sum(axis=2) for c in non_ocean], axis=2)
    out = np.asarray(non_ocean)[counts.argmax(axis=2)]
    return np.where(ocean, ocean_code, out).astype(mask.dtype)


def open_sea(mask: np.ndarray, ocean_code: int = OCEAN_CODE) -> np.ndarray:
    """Return ocean cells in an ocean body reaching the domain edge or the no-data corners."""
    ocean = mask == ocean_code
    labels, _ = ndimage.label(ocean, structure=_STRUCTURE_8)
    edge_labels = np.unique(labels[_domain_edge(mask) & ocean])
    return np.isin(labels, edge_labels[edge_labels != 0])


def _domain_edge(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge | ndimage.binary_dilation(mask == NODATA_CODE, structure=_STRUCTURE_8)


def coast_msk(mask: np.ndarray, river_code: int, offset_cells: int = 0,
              ocean_code: int = OCEAN_CODE) -> np.ndarray:
    """Return SFINCS msk: boundary 2, open sea beyond it and no-data 0, else 1.

    The boundary is the coastline (`offset_cells` 0) or the open-sea cells at chessboard distance
    `offset_cells` from it; open sea closer than that is active, and boundary too where it meets
    the domain edge or no-data (else a wall). Ocean bodies that do not reach the domain edge
    (unresolved lagoons) stay active and unforced, as in the eikonal.
    """
    coast = subgrid_coastline_mask(mask, ocean_code, river_code)
    sea = open_sea(mask, ocean_code)
    msk = np.full(mask.shape, MSK_ACTIVE, dtype=np.int8)
    msk[sea | (mask == NODATA_CODE)] = MSK_INACTIVE
    if offset_cells == 0:
        msk[coast] = MSK_BOUNDARY
        return msk
    dist = ndimage.distance_transform_cdt(~coast, metric="chessboard")
    band = sea & (dist < offset_cells)
    msk[band] = MSK_ACTIVE
    msk[(sea & (dist == offset_cells)) | (band & _domain_edge(mask))] = MSK_BOUNDARY
    return msk


def sfincs_interpolation(cell_xy: np.ndarray, point_xy: np.ndarray, point_series: np.ndarray) -> np.ndarray:
    """Return each cell's series as SFINCS builds it from the forcing points.

    SFINCS v2.4.0 (sfincs_boundaries.f90, find_boundary_indices): the two nearest points by
    straight-line distance in model coordinates, point 1 weighted dst2 / (dst1 + dst2).
    """
    if len(point_xy) == 1:
        return np.repeat(point_series, len(cell_xy), axis=0)
    d, j = cKDTree(point_xy).query(cell_xy, k=2)
    total = d.sum(axis=1)
    w1 = np.where(total > 0, d[:, 1] / np.where(total > 0, total, 1.0), 1.0)
    return w1[:, None] * point_series[j[:, 0]] + (1.0 - w1)[:, None] * point_series[j[:, 1]]


def choose_points(cell_xy: np.ndarray, rows: np.ndarray, cols: np.ndarray, series: np.ndarray,
                  block: int = THIN_BLOCK_CELLS, tol_m: float = INTERP_TOL_M) -> tuple[np.ndarray, int, float]:
    """Return (cell indices used as forcing points, densify rounds, max interpolation error m).

    Start with one point per `block` x `block` cells (the cell nearest the block's mean position),
    then add every cell whose SFINCS-interpolated series (`sfincs_interpolation`) differs from its
    own by more than `tol_m` at any time, until none does.
    """
    blocks = pd.DataFrame({"b": (rows // block) * (cols.max() + 1) + cols // block,
                           "x": cell_xy[:, 0], "y": cell_xy[:, 1]})
    centre = blocks.groupby("b")[["x", "y"]].transform("mean").to_numpy()
    blocks["d"] = np.hypot(*(cell_xy - centre).T)
    chosen = np.zeros(len(cell_xy), dtype=bool)
    chosen[blocks.groupby("b")["d"].idxmin().to_numpy()] = True
    for rounds in range(1, MAX_DENSIFY_ROUNDS + 1):
        idx = np.flatnonzero(chosen)
        err = np.abs(sfincs_interpolation(cell_xy, cell_xy[idx], series[idx]) - series).max(axis=1)
        bad = err > tol_m
        if not bad.any():
            return idx, rounds, float(err.max())
        chosen |= bad
    raise RuntimeError(f"forcing points did not converge in {MAX_DENSIFY_ROUNDS} rounds")


def _cell_shapes(lonlat: np.ndarray, hg_lonlat: np.ndarray, shapes: np.ndarray, knn: int,
                 idw_power: float) -> dict[str, np.ndarray]:
    parts = [blended_shapes(lonlat[s:s + BLEND_CHUNK], hg_lonlat, shapes, knn, idw_power)
             for s in range(0, len(lonlat), BLEND_CHUNK)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    far = out["n_used"] == 0  # no station within MAX_MATCH_KM: the nearest one's shape all the same
    out["shape"][far] = shapes[out["nearest"][far]]
    return out


def coast_forcing(msk: np.ndarray, transform, crs, tile_dir: Path, seed_rule: dict, hydrograph_nc: Path,
                  window_hr: tuple[float, float], start_cells: np.ndarray | None = None,
                  ) -> tuple[gpd.GeoDataFrame, pd.DataFrame, np.ndarray, dict]:
    """Build the forcing of the msk-2 cells of `msk` (SFINCS grid `transform`, `crs`).

    Returns (forcing points EPSG:4326, hydrographs: elapsed_hr + one column per point on the full
    time axis, the window-start level of the boundary cells and `start_cells` (for zsini; NaN
    elsewhere), stats).
    """
    is_bnd = msk == MSK_BOUNDARY
    rows, cols = np.nonzero(is_bnd if start_cells is None else is_bnd | start_cells)
    bnd = is_bnd[rows, cols]
    c, r = cols + 0.5, rows + 0.5
    cell_xy = np.column_stack([transform.c + transform.a * c + transform.b * r,
                               transform.f + transform.d * c + transform.e * r])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cell_xy[:, 0], cell_xy[:, 1])
    lonlat = np.column_stack([lon, lat])
    seed_lonlat, seed_levels = production_seed_levels(tile_dir, seed_rule)
    level = level_at_nearest_seed(lonlat, seed_lonlat, seed_levels).astype(np.float64)

    hg_lon, hg_lat, hg, times = read_hydrographs(hydrograph_nc)
    hg_lonlat = np.column_stack([hg_lon, hg_lat])
    elapsed = (times - times[0]).total_seconds().to_numpy() / 3600.0
    window = (elapsed >= window_hr[0]) & (elapsed <= window_hr[1])
    shapes = peak_relative_shapes(hg)
    knn, power = int(seed_rule["knn"]), float(seed_rule["idw_power"])
    cell = _cell_shapes(lonlat, hg_lonlat, shapes[:, window], knn, power)
    in_window = cell["shape"] + level[:, None]
    start = np.full(msk.shape, np.nan, dtype=np.float32)
    start[rows, cols] = in_window[:, 0]
    n_start_only = int((~bnd).sum())
    rows, cols, cell_xy, lonlat, level, in_window = (a[bnd] for a in (rows, cols, cell_xy, lonlat, level, in_window))
    cell = {k: v[bnd] for k, v in cell.items()}
    idx, rounds, max_err = choose_points(cell_xy, rows, cols, in_window)

    point_shapes = _cell_shapes(lonlat[idx], hg_lonlat, shapes, knn, power)["shape"]
    hydro = pd.DataFrame(point_shapes.T + level[idx], index=times)
    hydro.columns = range(len(idx))
    hydro.insert(0, "elapsed_hr", elapsed)
    points = gpd.GeoDataFrame(
        {"kind": "coast", "row": rows[idx], "col": cols[idx], "rp100_level_m": level[idx],
         "hydrograph_station": cell["nearest"][idx], "hydrograph_dist_km": cell["nearest_km"][idx],
         "n_hydrographs": cell["n_used"][idx]},
        geometry=gpd.points_from_xy(lonlat[idx, 0], lonlat[idx, 1]), crs="EPSG:4326")
    stats = {"n_boundary_cells": len(rows), "n_start_only_cells": n_start_only, "n_forcing_points": len(idx), "densify_rounds": rounds,
             "max_interp_error_m": round(max_err, 4), "seed_m_min": float(level.min()), "seed_m_max": float(level.max()),
             "n_cells_no_station_within_cap": int((cell["n_used"] == 0).sum())}
    return points, hydro, start, stats
