"""Build connectivity-first flood-simulation domains from the DeltaDTM tile index.

Method: docs/methods_01_tile_processing_and_waterlevels.md section 3; orchestrated by
preparation/build_tile_manifest.py. Domain boundaries fall only where no floodable land
connectivity crosses, or where a forced cut adds a bounded explicit overlap. Phases:
0 tile index, 1 floodable-land adjacency graph (ocean is never a connector: coastal
exposure is handled by each domain's own boundary forcing), 2 connected components,
3 budget-aware splitting, 4 tiered merging, 5 hop-distance BFS, 6 hop=0 coastal pad.
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

_DEG_TO_NATIVE_PX = 3600.0  # DeltaDTM 1 arcsec, independent of latitude
_KM_PER_DEG = 111.32
_M_PER_DEG = 111_320.0  # equatorial approximation, matches tiles.py


@dataclass
class ConnectivityConfig:
    """Tunable parameters shared by all phases (from `tile_generation.*` config).

    Cell counts are native ~30 m (1 arcsec) cells. `water_codes` defaults to
    `(ocean_code,)`.
    """

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


# Phase 0 - tile index (filenames only, no raster data read)

_COORD_RE = re.compile(r"([NS])(\d{2})([EW])(\d{3})")


def _parse_coord_from_filename(name: str) -> tuple[int, int] | None:
    m = _COORD_RE.search(name)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    return int(lat) * (1 if ns == "N" else -1), int(lon) * (1 if ew == "E" else -1)


def build_tile_index(mask_dir: Path) -> gpd.GeoDataFrame:
    """Build one 1x1deg EPSG:4326 box per DeltaDTM mask tile file in `mask_dir`.

    Parsed from the {NS}{lat:02d}{EW}{lon:03d} SW-corner filename token only.
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


def _mosaic_nearest_coarse(
    bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict | None, resolution_m: float,
    allowed_coords: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, Affine] | None:
    # Nearest (not MIN, unlike tiles.mosaic_mask_dem_coarse) for elevation too, giving
    # unbiased samples for fraction estimates. `allowed_coords` restricts reads to a
    # component's member tiles so land from another, unconnected component inside the
    # same bbox is never swept in. OVERVIEW_LEVEL="NONE" is required: DeltaDTM mask
    # overviews were built with an averaging resampler over categorical codes
    # (land 0 / lake 2 -> ocean 1), and GDAL silently reads them for large decimation.
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


def _contiguous_true_runs(arr_1d: np.ndarray, max_extent: int | None = None) -> list[tuple[int, int]]:
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
    # Runs touching either edge are trimmable margin, not interior gaps.
    n = len(mask_1d)
    for start, end in _contiguous_true_runs(mask_1d):
        if start > 0 and end < n - 1 and (end - start + 1) >= min_len:
            return start, end
    return None


def load_raw_tile_index(mask_dir: Path) -> gpd.GeoDataFrame:
    """Load the Phase 0 tile index with a 0..n-1 index."""
    return build_tile_index(mask_dir).reset_index(drop=True)


def build_connectivity_graph(
    sub: gpd.GeoDataFrame, mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
) -> list[tuple[int, int]]:
    """Return `(i, j)` row pairs of adjacent tiles with floodable land across the border.

    Floodable = mask land with valid elevation < `cfg.elev_threshold_m`, required in a
    `cfg.strip_width_deg` strip on BOTH sides. Ocean is never a connector.
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
    """Group tile indices into connected components as ``{root: [members]}``."""
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
    """One rectangular flood-simulation domain.

    `component_id` -1 means unassigned. Components are never hydraulically connected,
    so hop ordering and merging stay within one component_id.
    """

    bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy), degrees
    split_reason: str                        # "no_split_needed" | "natural_gap" | "forced" | "merged"
    approx_native_cells: float
    component_id: int = -1


def _native_cells(bbox: tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bbox
    return (maxx - minx) * _DEG_TO_NATIVE_PX * (maxy - miny) * _DEG_TO_NATIVE_PX


def bbox_to_polygon(bbox: tuple[float, float, float, float]):
    """Convert a ``(minx, miny, maxx, maxy)`` bbox to a shapely box."""
    return box(*bbox)


def split_component_to_budget(
    cluster_bbox: tuple[float, float, float, float],
    mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
    depth: int = 0, max_depth: int = 25, member_coords: set[str] | None = None,
) -> list[Domain]:
    """Recursively split one component's bbox into domains under `cfg.budget_cells`.

    Trims to content (a coarse cell is dropped only if all samples are nodata or
    `cfg.water_codes`, never for elevation alone), then prefers an interior gap of
    >= `cfg.min_split_gap_coarse_cells` over a forced cut with `cfg.overlap_target_km`
    overlap. Always pass `member_coords` in production (see `_mosaic_nearest_coarse`).
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
    keep = ~is_nodata_or_water

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


# Phase 4 - post-hoc size-based merging

def _union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_distance(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _merge_small_domains_tiered_one_component(
    working: list[Domain], min_size_cells: float, preferred_ceiling: float, hard_ceiling: float,
) -> list[Domain]:
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

        preferred = [j for j in others if _native_cells(_union_bbox(working[i].bbox, working[j].bbox)) <= preferred_ceiling]
        if preferred:
            j = min(preferred, key=lambda k: _bbox_distance(working[i].bbox, working[k].bbox))
        else:
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
    """Merge domains below `min_size_cells` within each component, two-tier by size.

    Tier 1: nearest neighbour whose union stays <= `preferred_ceiling`; tier 2: the
    neighbour giving the smallest union <= `hard_ceiling`. Merging per component
    avoids an O(n^2) loop over all domains worldwide.
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


# Phase 5 - hop-distance BFS within one component_id

def _domain_touches_ocean(bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict, cfg: ConnectivityConfig) -> bool:
    result = _mosaic_nearest_coarse(bbox, mask_index, dem_index, cfg.coarse_resolution_m)
    if result is None:
        return False
    mask_samples, _dem_samples, _transform = result
    return bool((mask_samples == cfg.ocean_code).any())


def _bbox_touches(a: tuple[float, float, float, float], b: tuple[float, float, float, float], tol_deg: float = 1e-6) -> bool:
    # tol_deg absorbs float error from coarse-cell boundary arithmetic.
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + tol_deg < bx0 or bx1 + tol_deg < ax0 or ay1 + tol_deg < by0 or by1 + tol_deg < ay0)


def compute_hop_distances(
    domains: list[Domain], mask_index: dict, dem_index: dict, cfg: ConnectivityConfig,
) -> tuple[list[int | None], list[int]]:
    """Compute each domain's BFS hop distance to the nearest ocean-touching domain.

    Ocean-touching domains are hop=0; adjacency (touching bboxes) only links domains
    of the same component_id. Returns ``(hop, unreachable)``: `hop` is None for
    domains with no path to ocean, which must be dropped before simulation.
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


# Phase 6 - coastal buffer pad, hop=0 domains only

def pad_hop0_domains_bbox(
    domains: list[Domain], hop: list[int | None], buffer_km: float, hard_ceiling_cells: float,
) -> tuple[list[Domain], int]:
    """Pad hop=0 domains by `buffer_km` per side unless that exceeds `hard_ceiling_cells`.

    Run after `compute_hop_distances`: only domains forced from the open coast need
    an ocean margin, and padding cannot change hop distance. Returns
    ``(domains, n_capped)``, `n_capped` = hop=0 domains left unpadded.
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
    """Number domains by (hop_distance, component_id) and build the tile-grid GeoDataFrame."""
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
