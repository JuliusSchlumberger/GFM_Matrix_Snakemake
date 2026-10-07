"""Connectivity-first domain construction for the global flood-modeling pipeline.

Production implementation (2026-10) of the method documented in full in
docs/methods_01_tile_processing_and_waterlevels.md section 3 - read that for the
conceptual design, the "why this is correct" argument, and real validation
numbers (a full-world run: 4,504 final domains, median 24.8M cells, hop-distance
chains up to depth 10, 114 domains correctly dropped as unreachable). This
module's docstrings below cover implementation detail only.

Supersedes the old 13-stage greedy-covering pipeline (build_chunks/reduce_overlap/
filter_and_shave_chunks/.../compute_run_order - fully retired and removed, 2026-10)
whose per-tile independent shave decisions could silently erode the overlap meant
to preserve flood connectivity across a chunk boundary. This method instead
derives every domain boundary from the real
floodable-land connectivity structure of the terrain itself: a boundary is placed
only where no floodable connectivity crosses it at all, or where a bounded,
explicit overlap compensates for cutting through content that IS connected - never
from an independent, local, per-tile judgment call.

Phases (orchestrated by preparation/build_tile_manifest.py, not in this module):
  0. Raw 1x1deg DeltaDTM tile index (`load_raw_tile_index`, thin wrapper over
     `build_tile_index` below - filename-parsed only, no raster read).
  1. Tile-adjacency graph, edged by real floodable-LAND connectivity only, never
     ocean (`build_connectivity_graph`) - see the module-level rationale below.
  2. Connected components over that graph (`connected_components`, plain
     union-find).
  3. Budget-aware recursive splitting of each component into rectangular domains
     (`split_component_to_budget`/`_split_window`): no split if the component's
     trimmed bbox already fits the budget; prefer a genuine internal gap (free -
     no real content crosses it); only fall back to a forced geometric cut (the
     one place a bounded, explicit overlap is introduced) when no gap exists.
  4. Post-hoc size-based merging of undersized domains within one component
     (`merge_small_domains_tiered`) - a connectivity-preserving union of bboxes,
     never crosses a component boundary.
  5. Hop-distance BFS (`compute_hop_distances`), scoped within one component_id
     only - a domain with real ocean access is hop=0 (self-forced, exactly
     production's own hop_distance==0 rule); every other domain's hop_distance is
     its BFS distance to the nearest ocean-touching domain in its own component.
     A domain with no path to ocean within its own component is unreachable and
     must be dropped before simulation.
  6. Coastal buffer pad (`pad_hop0_domains_bbox`), hop=0 domains only - see that
     function's own docstring for why this is scoped to hop=0 and run strictly
     after Phase 5, not before.

Phase 1's "ocean is never a connector" design choice: a tile's exposure to open
ocean is already fully represented through its own direct coastal boundary forcing
(Phase 5); two separate stretches of coastline being geographically adjacent is not
evidence that flooding in one could propagate overland into the other. Treating
ocean adjacency as a connector would inflate connected-component size with no
corresponding physical justification.

Built from, and replaces, the standalone prototype at
tests/connectivity_tiling_prototype/connectivity_tiling.py - that prototype also
tried two rejected alternative splitting strategies (balance-aware gap selection,
morphological pre-smoothing) before the tiered-merge approach shipped here; see
docs/methods_01_tile_processing_and_waterlevels.md for why they were rejected. Not
carried into this production module.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

from config_utils import retry_transient_io
from tiles import _clamp_window, _coord_str, _degree_tiles_for_bbox, _open_mask_tile

_DEG_TO_NATIVE_PX = 3600.0  # DeltaDTM's own 1 arcsec native resolution - independent of latitude
# (confirmed: native cell count = lon_extent_deg*3600 * lat_extent_deg*3600)
_KM_PER_DEG = 111.32
_M_PER_DEG = 111_320.0  # equatorial approximation, matches tiles.py's own constant - used by
# _mosaic_nearest_coarse below (metres, not km, unlike _KM_PER_DEG above)


@dataclass
class ConnectivityConfig:
    elev_threshold_m: float
    ocean_code: int
    coarse_resolution_m: float = 500.0          # Phase 3's own gap-finding/trim resolution
    strip_resolution_m: float = 500.0           # Phase 1's edge-connectivity check resolution
    strip_width_deg: float = 0.02               # ~2km - cheap, bounded-cost edge check
    budget_cells: float = 25_000_000.0          # split target, native (~30m) cells per domain
    min_split_gap_coarse_cells: int = 10        # min width of a natural gap to split on (coarse cells)
    overlap_target_km: float = 20.0             # bounded overlap width for a FORCED split (no natural gap)
    merge_trigger_cells: float = 20_000_000.0   # below this, a domain is a merge candidate
    preferred_ceiling_cells: float = 30_000_000.0  # tier-1 merge target
    hard_ceiling_cells: float = 100_000_000.0   # tier-2 merge / absolute cap
    coastal_buffer_km: float = 20.0             # Phase 6 pad width for hop=0 domains only
    water_codes: tuple[int, ...] = field(default_factory=lambda: ())  # populated from ocean_code in __post_init__

    def __post_init__(self):
        if not self.water_codes:
            self.water_codes = (self.ocean_code,)


# ---------------------------------------------------------------------------
# Phase 0 - tile index (filenames only, no raster data read). Moved in from
# the now-deleted src/tile_chunking.py (2026-10) - that module used to hold
# the retired 13-stage greedy-covering pipeline too (see this file's own
# module docstring); once that pipeline was gone, all that was left in it
# were these few shared primitives this module already imported, so there
# was no reason to keep them in a separate file.
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"([NS])(\d{2})([EW])(\d{3})")


def _parse_coord_from_filename(name: str) -> tuple[int, int] | None:
    m = _COORD_RE.search(name)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    return int(lat) * (1 if ns == "N" else -1), int(lon) * (1 if ew == "E" else -1)


def build_tile_index(mask_dir: Path) -> gpd.GeoDataFrame:
    """One 1x1deg polygon per real DeltaDTM mask tile file found in
    `mask_dir`, built purely from filenames (the {NS}{lat:02d}{EW}{lon:03d}
    SW-corner coordinate token, same convention as tiles.py's
    _scan_mask_dir/_coord_str) - no raster data read at all.
    """
    rows = []
    for path in mask_dir.glob("*.tif"):
        coord = _parse_coord_from_filename(path.name)
        if coord is None:
            continue
        lat, lon = coord
        rows.append({
            "coord": _coord_str(lat, lon),
            "lat": lat,
            "lon": lon,
            "geometry": box(lon, lat, lon + 1, lat + 1),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Shared coarse mosaic reader - every phase of this module goes through this.
# ---------------------------------------------------------------------------

def _mosaic_nearest_coarse(
    bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict | None, resolution_m: float,
    allowed_coords: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, Affine] | None:
    """Direct-to-coarse mosaic of mask+dem, BOTH via nearest-neighbour
    resampling - same per-source-tile read loop as tiles.mosaic_mask_dem_
    coarse, but nearest (not MIN) for elevation too. mosaic_mask_dem_coarse's
    MIN-for-elevation is a deliberate, biased-low choice for a different
    question ("does any low pocket exist in this coarse cell") - it would
    systematically UNDER-count how much of a cell is above the threshold.
    This gives unbiased point SAMPLES of the true native distribution
    instead, suitable for estimating a fraction. `dem_index=None` skips
    the elevation read entirely (only the mask is needed by some callers).

    `allowed_coords`, if given, restricts which degree-tiles within `bbox`
    are actually read - any coord not in the set is skipped exactly like a
    missing mask file (left at the default nodata fill), regardless of
    whether a real tile exists there. Default (None) is unrestricted,
    unchanged behaviour for every existing caller. Added for Phase 3
    (split_component_to_budget): a connected component's bounding-box
    rectangle can enclose far more tiles than its real members (e.g. a
    continent-scale component's bbox - confirmed on real worldwide data,
    2026-10 - can geometrically contain thousands of degree-cells against a
    few hundred real members), and reading every one of them is both slow
    (far more file opens than the component actually needs) and a latent
    correctness risk: real floodable land belonging to a DIFFERENT,
    Phase-1-unconnected component could otherwise be swept into this
    component's `keep` mask purely because its bbox happens to enclose it
    geographically, contradicting the whole method's core guarantee that
    nothing crosses a component boundary without reason.

    Opens every source tile with `OVERVIEW_LEVEL="NONE"` (forcing the base
    full-resolution layer) - confirmed necessary, not cosmetic, 2026-10:
    DeltaDTM mask tiles carry embedded GDAL overview pyramids, and at least
    one real tile's overviews contain mask value 1 (ocean_code) at pixels
    where the true base-resolution data has NO ocean_code cell anywhere
    (verified directly against DeltaDTM_v1_1_N15E100.tif - base-resolution
    full read: {0, 2, 3} only; its own level-0/level-1 overviews: also
    contain 1). rasterio's windowed `read(..., out_shape=..., resampling=
    Resampling.nearest)` silently lets GDAL substitute a matching overview
    as the read source for a large decimation factor (an internal GDAL
    performance optimization, not controlled by the `resampling=` argument,
    which only governs how pixels are picked FROM whichever layer GDAL
    chose) - so without this override, "nearest" picks a real pixel, but
    from an already-corrupted overview, not the native grid. The overviews
    were evidently built with an averaging-type resampler over categorical
    codes (0=land/2=lake averages to exactly 1=ocean_code) - invalid for
    this data regardless of which GDAL step produced them. This was traced
    from a real, data-grounded false positive during the connectivity-
    first migration - a domain over the Nakhon Sawan, Thailand river
    confluence was marked hop_distance=0 (self-forced, i.e. "touches
    ocean") purely from this artifact; the true base-resolution mask has
    no ocean_code there at all.
    """
    minx, miny, maxx, maxy = bbox
    px_deg = resolution_m / _M_PER_DEG
    out_width = max(1, round((maxx - minx) / px_deg))
    out_height = max(1, round((maxy - miny) / px_deg))
    transform = Affine.translation(minx, maxy) * Affine.scale(px_deg, -px_deg)

    mask_out = np.full((out_height, out_width), 255, dtype=np.uint8)
    dem_out = np.full((out_height, out_width), -9999.0, dtype=np.float32)
    any_coverage = False

    for lat, lon in _degree_tiles_for_bbox(bbox):
        coord = _coord_str(lat, lon)
        if allowed_coords is not None and coord not in allowed_coords:
            continue
        mask_path = mask_index.get(coord)
        if mask_path is None:
            continue
        ix0, iy0 = max(minx, lon), max(miny, lat)
        ix1, iy1 = min(maxx, lon + 1), min(maxy, lat + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        dst_window = from_bounds(ix0, iy0, ix1, iy1, transform).round_lengths().round_offsets()
        r0, c0 = int(dst_window.row_off), int(dst_window.col_off)
        h, w = int(dst_window.height), int(dst_window.width)
        h, w = _clamp_window(r0, c0, h, w, out_height, out_width)
        if h <= 0 or w <= 0:
            continue
        any_coverage = True

        with _open_mask_tile(mask_path, OVERVIEW_LEVEL="NONE") as src:
            src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            mask_out[r0:r0 + h, c0:c0 + w] = retry_transient_io(
                src.read, 1, window=src_window, boundless=True, fill_value=255,
                out_shape=(h, w), resampling=Resampling.nearest,
            )

        dem_path = dem_index.get(coord) if dem_index is not None else None
        if dem_path is not None:
            with _open_mask_tile(dem_path, OVERVIEW_LEVEL="NONE") as src:
                src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
                dem_out[r0:r0 + h, c0:c0 + w] = retry_transient_io(
                    src.read, 1, window=src_window, boundless=True, fill_value=-9999.0,
                    out_shape=(h, w), resampling=Resampling.nearest,
                )

    if not any_coverage:
        return None
    return mask_out, dem_out, transform


# ---------------------------------------------------------------------------
# Phase 3's natural-gap search (_split_window, below)
# ---------------------------------------------------------------------------

def _contiguous_true_runs(arr_1d: np.ndarray, max_extent: int | None = None) -> list[tuple[int, int]]:
    """Maximal contiguous (start, end) index runs of True, inclusive - each
    further split into sub-runs of at most `max_extent` if given.
    """
    runs = []
    start = None
    for i, v in enumerate(arr_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(arr_1d) - 1))

    if not max_extent:
        return runs
    capped = []
    for start, end in runs:
        pos = start
        while pos <= end:
            capped.append((pos, min(pos + max_extent - 1, end)))
            pos += max_extent
    return capped


def _first_interior_gap(mask_1d: np.ndarray, min_len: int) -> tuple[int, int] | None:
    """First (start, end) inclusive run of True in `mask_1d` at least
    `min_len` long, EXCLUDING any run touching either edge of `mask_1d`
    (that's ordinary trimmable margin, already handled by the tight-bbox
    step - not a genuine interior gap). Reuses `_contiguous_true_runs`
    rather than a new scan.
    """
    n = len(mask_1d)
    for start, end in _contiguous_true_runs(mask_1d):
        if start > 0 and end < n - 1 and (end - start + 1) >= min_len:
            return start, end
    return None


def load_raw_tile_index(mask_dir: Path) -> gpd.GeoDataFrame:
    """Phase 0 - thin wrapper over build_tile_index (one 1x1deg
    polygon per real DeltaDTM mask tile file, filename-parsed only)."""
    return build_tile_index(mask_dir).reset_index(drop=True)


def build_connectivity_graph(
    sub: gpd.GeoDataFrame, mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
) -> list[tuple[int, int]]:
    """Phase 1 - edges where real floodable land touches across a shared
    border (checked via a thin strip on each side - cheap, bounded cost).
    Ocean is deliberately never a connector (see module docstring).
    """
    lat_lon_to_idx = {(row.lat, row.lon): i for i, row in sub.iterrows()}

    def has_floodable(bbox):
        result = _mosaic_nearest_coarse(bbox, mask_index, dem_index, cfg.strip_resolution_m)
        if result is None:
            return False
        mask_samples, dem_samples, _ = result
        is_land = mask_samples == 0
        dem_valid = dem_samples != -9999.0
        floodable = is_land & dem_valid & (dem_samples < cfg.elev_threshold_m)
        return bool(floodable.any())

    edges = []
    for i, row in sub.iterrows():
        lat, lon = row.lat, row.lon
        for dlat, dlon, side in [(0, 1, "east"), (1, 0, "north")]:
            j = lat_lon_to_idx.get((lat + dlat, lon + dlon))
            if j is None:
                continue
            minx, miny, maxx, maxy = row.geometry.bounds
            w = cfg.strip_width_deg
            if side == "east":
                strip_a = (maxx - w, miny, maxx, maxy)
                strip_b = (maxx, miny, maxx + w, maxy)
            else:
                strip_a = (minx, maxy - w, maxx, maxy)
                strip_b = (minx, maxy, maxx, maxy + w)
            if has_floodable(strip_a) and has_floodable(strip_b):
                edges.append((i, j))
    return edges


def connected_components(n: int, edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Phase 2 - plain union-find. Returns {root_idx: [member indices]}."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return clusters


@dataclass
class Domain:
    bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy), degrees
    split_reason: str                        # "no_split_needed" | "natural_gap" | "forced" | "merged"
    approx_native_cells: float
    component_id: int = -1                   # which Phase-2 connected component this came from -
    # -1 means "not yet assigned" (set by the caller right after
    # split_component_to_budget returns). Two different components are NEVER
    # hydraulically connected by construction, so hop-ordering and merging must
    # both stay within one component_id - see compute_hop_distances and
    # merge_small_domains_tiered.


def _native_cells(bbox: tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bbox
    return (maxx - minx) * _DEG_TO_NATIVE_PX * (maxy - miny) * _DEG_TO_NATIVE_PX


def bbox_to_polygon(bbox: tuple[float, float, float, float]):
    return box(*bbox)


def split_component_to_budget(
    cluster_bbox: tuple[float, float, float, float],
    mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
    depth: int = 0, max_depth: int = 25, member_coords: set[str] | None = None,
) -> list[Domain]:
    """Phase 3. Builds a coarse floodability grid over `cluster_bbox`, trims to
    real content (a coarse cell is dropped only if ALL its samples are nodata or
    `cfg.water_codes` - never because of elevation/floodability alone), and
    recursively splits only if still over `cfg.budget_cells`, preferring a
    genuine internal gap over a forced geometric cut.

    `member_coords` (this component's real Phase-2 member tile coords) restricts
    the mosaic read to just those tiles rather than every tile inside
    `cluster_bbox`'s bounding rectangle - see `_mosaic_nearest_coarse`'s own
    `allowed_coords` docstring for why this matters beyond speed: without it,
    land belonging to a different, Phase-1-unconnected component could be swept
    into this component's `keep` mask purely because it falls inside the same
    rectangle. Always pass it in production; optional only for ad hoc callers.
    """
    result = _mosaic_nearest_coarse(
        cluster_bbox, mask_index, dem_index, cfg.coarse_resolution_m, allowed_coords=member_coords,
    )
    if result is None:
        return []
    mask_samples, dem_samples, transform = result
    dem_valid = dem_samples != -9999.0
    is_water = np.isin(mask_samples, cfg.water_codes)
    is_nodata_or_water = is_water | ~dem_valid
    keep = ~is_nodata_or_water  # keep unless nodata/water

    return _split_window(keep, transform, 0, keep.shape[0] - 1, 0, keep.shape[1] - 1, cfg, depth, max_depth)


def _split_window(
    keep: np.ndarray, transform, row0: int, row1: int, col0: int, col1: int,
    cfg: ConnectivityConfig, depth: int, max_depth: int,
) -> list[Domain]:
    sub = keep[row0:row1 + 1, col0:col1 + 1]
    if not sub.any():
        return []

    rows = np.where(sub.any(axis=1))[0]
    cols = np.where(sub.any(axis=0))[0]
    r0, r1 = row0 + int(rows.min()), row0 + int(rows.max())
    c0, c1 = col0 + int(cols.min()), col0 + int(cols.max())

    bbox = window_bounds(Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), transform)
    native_cells = _native_cells(bbox)

    if native_cells <= cfg.budget_cells or depth >= max_depth:
        leaf_reason = "no_split_needed" if depth == 0 else "n/a_leaf"  # overwritten by the parent call below
        return [Domain(bbox=bbox, split_reason=leaf_reason, approx_native_cells=native_cells)]

    not_kept = ~keep[r0:r1 + 1, c0:c1 + 1]
    row_gap = _first_interior_gap(not_kept.all(axis=1), cfg.min_split_gap_coarse_cells)
    if row_gap is not None:
        mid = r0 + (row_gap[0] + row_gap[1]) // 2
        top = _split_window(keep, transform, r0, mid, c0, c1, cfg, depth + 1, max_depth)
        bottom = _split_window(keep, transform, mid + 1, r1, c0, c1, cfg, depth + 1, max_depth)
        for d in top + bottom:
            if d.split_reason == "n/a_leaf":
                d.split_reason = "natural_gap"
        return top + bottom

    col_gap = _first_interior_gap(not_kept.all(axis=0), cfg.min_split_gap_coarse_cells)
    if col_gap is not None:
        mid = c0 + (col_gap[0] + col_gap[1]) // 2
        left = _split_window(keep, transform, r0, r1, c0, mid, cfg, depth + 1, max_depth)
        right = _split_window(keep, transform, r0, r1, mid + 1, c1, cfg, depth + 1, max_depth)
        for d in left + right:
            if d.split_reason == "n/a_leaf":
                d.split_reason = "natural_gap"
        return left + right

    # No natural gap - forced geometric split, longer axis, bounded overlap.
    height, width = r1 - r0 + 1, c1 - c0 + 1
    overlap_deg = cfg.overlap_target_km / _KM_PER_DEG
    coarse_deg = cfg.coarse_resolution_m / 111_320.0
    overlap_coarse_cells = max(1, round(overlap_deg / coarse_deg))

    if height >= width:
        mid = r0 + height // 2
        top = _split_window(keep, transform, r0, min(mid + overlap_coarse_cells, r1), c0, c1, cfg, depth + 1, max_depth)
        bottom = _split_window(keep, transform, max(mid - overlap_coarse_cells, r0), r1, c0, c1, cfg, depth + 1, max_depth)
        pieces = top + bottom
    else:
        mid = c0 + width // 2
        left = _split_window(keep, transform, r0, r1, c0, min(mid + overlap_coarse_cells, c1), cfg, depth + 1, max_depth)
        right = _split_window(keep, transform, r0, r1, max(mid - overlap_coarse_cells, c0), c1, cfg, depth + 1, max_depth)
        pieces = left + right
    for d in pieces:
        if d.split_reason == "n/a_leaf":
            d.split_reason = "forced"
    return pieces


# ---------------------------------------------------------------------------
# Phase 4 - post-hoc size-based merging
# ---------------------------------------------------------------------------

def _union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_distance(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _merge_small_domains_tiered_one_component(
    working: list[Domain], min_size_cells: float, preferred_ceiling: float, hard_ceiling: float,
) -> list[Domain]:
    """Core tiered-merge loop, scoped to domains that are ALREADY known to
    share one component_id - see merge_small_domains_tiered for why this
    scoping is split out, and for the tier-1/tier-2 rationale."""
    excluded: set[int] = set()
    while True:
        candidates_idx = [i for i, d in enumerate(working) if d.approx_native_cells < min_size_cells and i not in excluded]
        if not candidates_idx:
            break
        i = min(candidates_idx, key=lambda k: working[k].approx_native_cells)
        others = [j for j in range(len(working)) if j != i]
        if not others:
            excluded.add(i)
            continue

        # Tier 1: nearest neighbour whose merge stays under preferred_ceiling.
        preferred = [j for j in others if _native_cells(_union_bbox(working[i].bbox, working[j].bbox)) <= preferred_ceiling]
        if preferred:
            j = min(preferred, key=lambda k: _bbox_distance(working[i].bbox, working[k].bbox))
        else:
            # Tier 2: best (smallest resulting size) neighbour under hard_ceiling.
            under_hard = [j for j in others if _native_cells(_union_bbox(working[i].bbox, working[j].bbox)) <= hard_ceiling]
            if not under_hard:
                excluded.add(i)
                continue
            j = min(under_hard, key=lambda k: _native_cells(_union_bbox(working[i].bbox, working[k].bbox)))

        merged_bbox = _union_bbox(working[i].bbox, working[j].bbox)
        new_domain = Domain(
            bbox=merged_bbox, split_reason="merged", approx_native_cells=_native_cells(merged_bbox),
            component_id=working[i].component_id,
        )
        keep_idx = [k for k in range(len(working)) if k not in (i, j)]
        working = [working[k] for k in keep_idx] + [new_domain]
        excluded = set()
    return working


def merge_small_domains_tiered(
    domains: list[Domain], min_size_cells: float, preferred_ceiling: float, hard_ceiling: float,
) -> list[Domain]:
    """Two-tier merge: for each too-small domain, prefer the nearest neighbour
    whose merge result stays under `preferred_ceiling` (keeps the typical
    result centred near the ideal band, not just under the hard max); only if
    NO neighbour keeps it under `preferred_ceiling` does it fall back to the
    best neighbour under `hard_ceiling` - a last resort so a genuinely isolated
    small domain with only large neighbours still gets absorbed rather than
    left stranded.

    Partitions by component_id FIRST and merges each component independently,
    rather than one flat O(n^2)-ish loop over every domain regardless of
    component - merging was always restricted to same-component_id pairs (two
    different components are never hydraulically connected by construction), so
    this changes nothing about the result, only the cost: confirmed necessary on
    real data, 2026-10 - at world scale (7895 domains, largest component ~1000
    of them) a flat version rescanning the ENTIRE domain list on every single
    merge made a genuinely slow O(n^2) cost indistinguishable from a hang with
    no progress output. Prints one line per component as its merge pass
    finishes, for the same reason.
    """
    by_component: dict[int, list[Domain]] = {}
    for d in domains:
        by_component.setdefault(d.component_id, []).append(d)

    print(f"  merge: {len(domains)} domain(s) across {len(by_component)} component(s)", flush=True)
    t0 = time.time()
    result: list[Domain] = []
    for progress_i, (component_id, group) in enumerate(sorted(by_component.items(), key=lambda kv: -len(kv[1]))):
        n_before = len(group)
        merged_group = _merge_small_domains_tiered_one_component(group, min_size_cells, preferred_ceiling, hard_ceiling)
        result.extend(merged_group)
        if n_before >= 10 or (progress_i + 1) % 100 == 0 or (progress_i + 1) == len(by_component):
            print(
                f"  [merge {progress_i + 1}/{len(by_component)}] component_id={component_id}: "
                f"{n_before} -> {len(merged_group)} domain(s), t={time.time() - t0:.0f}s total",
                flush=True,
            )
    return result


# ---------------------------------------------------------------------------
# Phase 5 - hop-distance BFS, scoped within one component_id only (see
# module docstring)
# ---------------------------------------------------------------------------

def _domain_touches_ocean(bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict, cfg: ConnectivityConfig) -> bool:
    """Real ocean touch check - same coarse read as everything else here,
    never a native-resolution read. `dem_index` is accepted but unused
    (shared helper signature with _mosaic_nearest_coarse) - only the mask
    band matters for this check."""
    result = _mosaic_nearest_coarse(bbox, mask_index, dem_index, cfg.coarse_resolution_m)
    if result is None:
        return False
    mask_samples, _dem_samples, _transform = result
    return bool((mask_samples == cfg.ocean_code).any())


def _bbox_touches(a: tuple[float, float, float, float], b: tuple[float, float, float, float], tol_deg: float = 1e-6) -> bool:
    """True if two bboxes touch or overlap (shared edge counts, not just
    interior overlap) - `tol_deg` absorbs floating-point edge cases from
    the coarse-grid-cell-boundary arithmetic earlier phases use."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + tol_deg < bx0 or bx1 + tol_deg < ax0 or ay1 + tol_deg < by0 or by1 + tol_deg < ay0)


def compute_hop_distances(
    domains: list[Domain], mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
) -> tuple[list[int | None], list[int]]:
    """Phase 5. Returns (hop_distance per domain - None for unreachable,
    aligned to `domains`; indices of unreachable domains, which should be
    DROPPED before simulation - no path to ocean within their own component).

    Adjacency is checked ONLY between domains sharing the same component_id
    (see Domain's own docstring for why) - this is what makes cross-component
    hop-seeding structurally impossible here, not just rare. `_domain_touches_
    ocean` re-reads each domain's own bbox at the same coarse resolution
    everything else in this module uses - cheap, bounded cost, never native
    resolution.
    """
    n = len(domains)
    print(f"  compute_hop_distances: checking ocean adjacency for {n} domain(s)...", flush=True)
    t0 = time.time()
    touches_ocean = []
    for i, d in enumerate(domains):
        touches_ocean.append(_domain_touches_ocean(d.bbox, mask_index, dem_index, cfg))
        if (i + 1) % 250 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s = (n - (i + 1)) / rate if rate > 0 else float("nan")
            print(
                f"  PROGRESS: {i + 1}/{n} domain(s) checked ({100 * (i + 1) / n:.1f}%) "
                f"elapsed={elapsed:.0f}s ETA={eta_s / 60:.1f} min",
                flush=True,
            )
    print(f"  ocean-adjacency check done, t={time.time() - t0:.0f}s - building domain-adjacency graph...", flush=True)

    t1 = time.time()
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if domains[i].component_id != domains[j].component_id:
                continue
            if _bbox_touches(domains[i].bbox, domains[j].bbox):
                adjacency[i].append(j)
                adjacency[j].append(i)
    print(f"  domain-adjacency graph done, t={time.time() - t1:.0f}s", flush=True)

    hop: list[int | None] = [None] * n
    dq: deque[int] = deque()
    for i in range(n):
        if touches_ocean[i]:
            hop[i] = 0
            dq.append(i)
    while dq:
        i = dq.popleft()
        for j in adjacency[i]:
            if hop[j] is None:
                hop[j] = hop[i] + 1
                dq.append(j)

    unreachable = [i for i in range(n) if hop[i] is None]
    return hop, unreachable


# ---------------------------------------------------------------------------
# Phase 6 - coastal buffer pad, hop=0 domains only
# ---------------------------------------------------------------------------

def pad_hop0_domains_bbox(
    domains: list[Domain], hop: list[int | None], buffer_km: float, hard_ceiling_cells: float,
) -> tuple[list[Domain], int]:
    """Pads every hop_distance==0 domain's bbox outward by `buffer_km` on each
    side. Returns (new domain list, count left unpadded because padding would
    have exceeded `hard_ceiling_cells`).

    Scoped to hop=0 ONLY, and run strictly AFTER Phase 5 (compute_hop_distances)
    rather than before: Phase 3's trim-to-content step gives an IMPLICIT buffer
    around a coastal nose/headland for free (water within the land's own
    row/column bounding extent survives, since domains are rectangles, not
    per-pixel masks) - but not the old pipeline's EXPLICIT minimum-ocean-margin
    guarantee. That guarantee only matters for hop=0 domains, the ones
    self-forced directly from the open coast/COAST-RP, where a headland cutting
    too close to the domain edge would matter - a hop>=1 hinterland domain is
    forced from an already-simulated neighbour's wave, not from direct
    coastline geometry, so it has no analogous concern. Because only already-
    hop=0 domains are touched and their hop_distance cannot change (padding a
    domain that already touches ocean still touches ocean), no re-derivation of
    hop_distance/adjacency is needed afterward.
    """
    buffer_deg = buffer_km / _KM_PER_DEG
    n_capped = 0
    padded: list[Domain] = []
    for d, h in zip(domains, hop):
        if h != 0:
            padded.append(d)
            continue
        minx, miny, maxx, maxy = d.bbox
        candidate_bbox = (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)
        if _native_cells(candidate_bbox) <= hard_ceiling_cells:
            padded.append(Domain(
                bbox=candidate_bbox, split_reason=d.split_reason,
                approx_native_cells=_native_cells(candidate_bbox), component_id=d.component_id,
            ))
        else:
            n_capped += 1
            padded.append(d)
    return padded, n_capped


def assign_tile_id(domains: list[Domain], hop: list[int]) -> gpd.GeoDataFrame:
    """Final step: builds the GeoDataFrame written to `tile_grid.path`, with
    `tile_id` assigned sequentially - ordered by hop_distance ascending (wave-0
    first) then component_id, stable tie-break on original order otherwise.
    Preserves the old schema's "hop_distance correlates with run order" spirit
    without inventing a new tie-break scheme. Keeps split_reason/component_id/
    approx_cells_M as extra diagnostic columns (harmless - confirmed no
    downstream reader is column-position-sensitive; useful for QA).
    """
    gdf = gpd.GeoDataFrame(
        {
            "hop_distance": hop,
            "split_reason": [d.split_reason for d in domains],
            "component_id": [d.component_id for d in domains],
            "approx_cells_M": [round(d.approx_native_cells / 1e6, 2) for d in domains],
        },
        geometry=[bbox_to_polygon(d.bbox) for d in domains],
        crs="EPSG:4326",
    )
    gdf = gdf.sort_values(["hop_distance", "component_id"], kind="stable").reset_index(drop=True)
    gdf["tile_id"] = gdf.index
    return gdf[["tile_id", "hop_distance", "split_reason", "component_id", "approx_cells_M", "geometry"]]
