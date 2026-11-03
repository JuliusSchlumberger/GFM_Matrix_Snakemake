"""Shortest-path distance through connected water, and IDW over the k nearest stations by it.

Used to carry station values up estuaries along the water instead of across land, on the
coarse ocean-connectivity raster (`tiles.mosaic_water_fraction_downsampled`). Main entry:
`along_water_idw`.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def water_graph(water: np.ndarray, cell_m: float) -> coo_matrix:
    """8-connected graph over water cells, edge weight = centre distance (m)."""
    rows, cols = water.shape
    index = np.full(water.shape, -1, dtype=np.int64)
    index[water] = np.arange(int(water.sum()))
    src, dst, w = [], [], []
    for dr, dc in _OFFSETS:
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        a = index[r0:r1, c0:c1]
        b = index[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = (a >= 0) & (b >= 0)
        src.append(a[ok])
        dst.append(b[ok])
        w.append(np.full(int(ok.sum()), cell_m * np.hypot(dr, dc)))
    n = int(water.sum())
    return coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n)).tocsr()


def _snap_to_water(water: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Index (into water cells) of the nearest water cell to each (row, col)."""
    wr, wc = np.nonzero(water)
    return cKDTree(np.column_stack([wr, wc])).query(np.column_stack([rows, cols]))[1]


def along_water_distances(water: np.ndarray, cell_m: float, station_rc: np.ndarray,
                          target_rc: np.ndarray, limit_m: float = np.inf) -> np.ndarray:
    """Return (n_stations, n_targets) shortest water-path distances (m); inf if unconnected.

    Stations and targets are snapped to their nearest water cell first, so a point on land
    next to the water (a seed cell, a coastal station) still gets a distance.
    """
    if not water.any():
        return np.full((len(station_rc), len(target_rc)), np.inf)
    graph = water_graph(water, cell_m)
    s = _snap_to_water(water, station_rc[:, 0], station_rc[:, 1])
    t = _snap_to_water(water, target_rc[:, 0], target_rc[:, 1])
    unique_s, inverse = np.unique(s, return_inverse=True)
    dist = dijkstra(graph, directed=False, indices=unique_s, limit=limit_m)
    return dist[inverse][:, t]


def along_water_idw(distances: np.ndarray, values: np.ndarray, k: int,
                    fallback_distances: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """IDW (power 2) of station values over the k nearest stations by water-path distance.

    `distances`: (n_stations, n_targets). Targets with no finite water distance use
    `fallback_distances` (e.g. haversine) when given, else NaN. Returns (values, used_fallback).
    """
    d = distances.astype(np.float64).copy()
    used_fallback = ~np.isfinite(d).any(axis=0)
    if fallback_distances is not None:
        d[:, used_fallback] = fallback_distances[:, used_fallback]
    k = min(k, d.shape[0])
    order = np.argsort(d, axis=0)[:k]
    dk = np.take_along_axis(d, order, axis=0)
    vk = values[order]
    exact = dk < 1e-6
    w = np.where(np.isfinite(dk), 1.0 / np.maximum(dk, 1e-6) ** 2, 0.0)
    w = np.where(exact.any(axis=0), exact.astype(float), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (w * vk).sum(axis=0) / w.sum(axis=0)
    return out, used_fallback
