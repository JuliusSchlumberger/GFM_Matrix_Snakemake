"""Event-volume corrections to the eikonal flood depth (candidates A and B).

A, `basin_volume_cap`: the volume entering each enclosed basin over its sill (broad-crested
weir, storm-tide hydrograph) caps the basin's fill, as RFSM (Gouldby et al. 2008).
B, `duration_friction_multiplier`: friction is raised where the water stands above the
controlling elevation (`spill_level`) only briefly during the hydrograph peak.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit
from scipy import ndimage

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

# Broad-crested weir discharge per metre of crest: q = C * h^1.5 (m^0.5/s, SI).
WEIR_COEFFICIENT = 1.7

# Cells whose spill level exceeds the ground by more than this lie in an enclosed basin (m).
BASIN_MIN_DEPTH_M = 0.05

# Crest cells: ring cells within this height of the sill level (m).
CREST_TOLERANCE_M = 0.10


@njit(cache=True)
def _priority_flood(dem: np.ndarray, sources: np.ndarray) -> np.ndarray:
    n_rows, n_cols = dem.shape
    fill = np.full(dem.shape, np.inf)
    heap = [(0.0, 0)]
    heap.pop()
    for r in range(n_rows):
        for c in range(n_cols):
            if sources[r, c]:
                fill[r, c] = dem[r, c]
                heapq.heappush(heap, (float(dem[r, c]), r * n_cols + c))
    while heap:
        level, idx = heapq.heappop(heap)
        r, c = idx // n_cols, idx % n_cols
        if level > fill[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or cc < 0 or rr >= n_rows or cc >= n_cols:
                    continue
                new = max(level, float(dem[rr, cc]))
                if new < fill[rr, cc]:
                    fill[rr, cc] = new
                    heapq.heappush(heap, (new, rr * n_cols + cc))
    return fill


def spill_level(dem: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return the lowest water level (m) at which each cell connects to `sources` (8-connected).

    Priority flood (Barnes et al. 2014): the minimax of ground elevation over all paths
    from a source. Equals the ground on terrain rising from the coast and the sill
    crest inside an enclosed basin. Unreachable cells are +inf.
    """
    return _priority_flood(np.asarray(dem, dtype=np.float64), np.asarray(sources, dtype=np.bool_))


def time_within_peak(hydrograph: np.ndarray, dt_s: float, head_m: np.ndarray) -> np.ndarray:
    """Return how long (s) the hydrograph stays within `head_m` of its peak, per head value."""
    sorted_h = np.sort(np.asarray(hydrograph, dtype=np.float64))
    threshold = sorted_h[-1] - np.asarray(head_m, dtype=np.float64)
    n_above = len(sorted_h) - np.searchsorted(sorted_h, threshold, side="right")
    return n_above * dt_s


def weir_inflow_volume(sill_hydrograph: np.ndarray, sill_level: float, crest_width_m: float, dt_s: float,
                       coefficient: float = WEIR_COEFFICIENT) -> float:
    """Return the volume (m³) crossing a sill: sum of C * W * (h - z_sill)^1.5 * dt over h > z_sill."""
    head = np.clip(np.asarray(sill_hydrograph, dtype=np.float64) - sill_level, 0.0, None)
    return float(coefficient * crest_width_m * np.sum(head**1.5) * dt_s)


def level_pool_level(dem_cells: np.ndarray, cell_area_m2: float, volume_m3: float) -> float:
    """Return the level L (m) with sum(max(0, L - dem) * area) = volume over the given cells."""
    z = np.sort(np.asarray(dem_cells, dtype=np.float64))
    if volume_m3 <= 0 or z.size == 0:
        return float(z[0]) if z.size else float("nan")
    # Storage below each sorted ground level, then linear in L within each interval.
    storage = (np.arange(1, z.size + 1) * z - np.cumsum(z)) * cell_area_m2
    i = int(np.searchsorted(storage, volume_m3, side="right"))
    base = z[i - 1] if i > 0 else z[0]
    stored = storage[i - 1] if i > 0 else 0.0
    return float(base + (volume_m3 - stored) / (max(i, 1) * cell_area_m2))


def label_basins(dem: np.ndarray, fill: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected wet cells lying more than BASIN_MIN_DEPTH_M below their spill level."""
    return ndimage.label(wet & np.isfinite(fill) & (fill - dem > BASIN_MIN_DEPTH_M), structure=_STRUCTURE_8)


def basin_volume_cap(
    depth: np.ndarray,
    dem: np.ndarray,
    fill: np.ndarray,
    hydrograph_for_cell,
    dt_s: float,
    cell_area_m2: float,
    cell_size_m: float,
    coefficient: float = WEIR_COEFFICIENT,
) -> tuple[np.ndarray, list[dict]]:
    """Cap each enclosed basin's eikonal fill at the volume that crosses its sill.

    `hydrograph_for_cell(row, col)` returns the coastal storm-tide series (m) forcing that
    location; at the sill it is shifted down by the eikonal attenuation loss, i.e. so its
    peak equals the eikonal water level on the basin rim. One level pool per basin
    (nested basins are filled as one). Returns the corrected depth and per-basin records.
    """
    out = depth.copy()
    wet = depth > 0
    labels, n = label_basins(dem, fill, wet)
    records = []
    if n == 0:
        return out, records
    level = np.where(wet, dem + depth, -np.inf)
    objects = ndimage.find_objects(labels)
    for lab, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        pad = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in slc)
        comp = labels[pad] == lab
        ring = ndimage.binary_dilation(comp, structure=_STRUCTURE_8) & ~comp
        z_sill = float(np.median(fill[pad][comp]))
        ring_level = level[pad][ring]
        rim_level = float(ring_level.max()) if np.isfinite(ring_level).any() else float(level[pad][comp].max())
        crest = ring & (dem[pad] <= z_sill + CREST_TOLERANCE_M)
        width_m = max(1, int(crest.sum())) * cell_size_m
        rows, cols = np.nonzero(comp)
        centre = (int(rows.mean()) + pad[0].start, int(cols.mean()) + pad[1].start)
        h = np.asarray(hydrograph_for_cell(*centre), dtype=np.float64)
        h_sill = h - (h.max() - rim_level)
        v_in = weir_inflow_volume(h_sill, z_sill, width_m, dt_s, coefficient)
        v_eik = float(depth[pad][comp].sum() * cell_area_m2)
        capped = v_in < v_eik
        if capped:
            pool = level_pool_level(dem[pad][comp], cell_area_m2, v_in)
            new = np.clip(pool - dem[pad], 0.0, None)
            out[pad][comp] = new[comp]
        touches_edge = (slc[0].start == 0 or slc[1].start == 0
                        or slc[0].stop == depth.shape[0] or slc[1].stop == depth.shape[1])
        records.append({"basin": lab, "n_cells": int(comp.sum()), "sill_m": z_sill, "rim_level_m": rim_level,
                        "crest_width_m": width_m, "inflow_m3": v_in, "eikonal_m3": v_eik,
                        "capped": bool(capped), "touches_edge": bool(touches_edge)})
    return out, records


def duration_friction_multiplier(
    depth: np.ndarray,
    dem: np.ndarray,
    fill: np.ndarray,
    hydrograph_for_cells,
    dt_s: float,
    t_ref_s: float,
    alpha: float,
    max_multiplier: float = 10.0,
) -> np.ndarray:
    """Return a per-cell friction multiplier (T_ref / T)^alpha, clipped to [1, max_multiplier].

    T is how long the water level stands above the controlling elevation (the spill
    level: local ground on rising terrain, the sill crest behind one), from a first
    eikonal solve: the time the forcing hydrograph is within (level - spill) of its peak.
    `hydrograph_for_cells(rows, cols)` returns (series_by_group, group_index_per_cell).
    Dry cells get 1.
    """
    mult = np.ones(depth.shape, dtype=np.float32)
    wet = depth > 0
    rows, cols = np.nonzero(wet)
    if rows.size == 0:
        return mult
    head = (dem[rows, cols] + depth[rows, cols]) - np.minimum(fill[rows, cols], dem[rows, cols] + depth[rows, cols])
    series, group = hydrograph_for_cells(rows, cols)
    duration = np.empty(rows.size)
    for g, h in enumerate(series):
        sel = group == g
        if sel.any():
            duration[sel] = time_within_peak(h, dt_s, head[sel])
    duration = np.maximum(duration, dt_s)
    mult[rows, cols] = np.clip((t_ref_s / duration) ** alpha, 1.0, max_multiplier)
    return mult
