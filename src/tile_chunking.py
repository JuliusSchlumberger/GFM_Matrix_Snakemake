"""Fixed-DeltaDTM-tile-based chunk generation (2026-08).

Replaces the adaptive parent/child `tile_generation.py` (retired - see
memory.md's TILE GENERATION REDESIGN / FIXED-TILE CHUNKING REDESIGN
entries). Originated as an experimental prototype in
`tests/deltadtm_coverage/` (build_tile_index.py, filter_floodable.py,
chunk_tiles.py, clean_chunks.py - kept there as the validated one-off
record of the design) before being promoted here as the production
implementation, with every previously-hardcoded constant converted to an
explicit parameter sourced from `config.yml`'s `tile_generation:` section
(see `preparation/build_tile_manifest.py`, this module's only caller).

Unlike the retired adaptive pipeline (parent context windows + refined
child domains), this one has a single grid level: DeltaDTM's own real 1x1
degree tile grid is the base unit ("tile"), and simulation domains
("chunks") are built directly from it by greedily covering the floodable
tile footprint with axis-aligned, overlapping rectangles, then cleaning
up redundant overlap and splitting anything too large. No parent/child
distinction, no separate forcing-context grid - each chunk gets its own
boundary-condition stations directly (extract_boundaries.py), same as
every other tile in the Snakemake DAG.

Pipeline stages, in order (see build_tile_manifest.py for orchestration):
  1. build_tile_index       - one 1x1deg polygon per real DeltaDTM mask
                               tile file, from filenames only.
  2. filter_floodable_tiles - keep tiles with >=1 floodable (land, valid
                               DEM, elevation < elev_threshold_m) cell at
                               coarse resolution; also marks each kept
                               tile's ocean/river mask-value fraction and
                               derives `is_river_mouth` (an ocean-river
                               MIXING signature, not just ocean-adjacent
                               or just riverine) for step 3's seeding.
  3. build_chunks            - greedy maximal-rectangle covering of the
                               floodable tile footprint, capped at
                               max_extent_tiles per side. Seeds from every
                               river-mouth tile first (coast-hugging
                               growth via _seed_river_mouth_chunk), falling
                               back to plain row-major seeding + the exact
                               "largest rectangle through a point" solve
                               (_maximal_rectangle_through) elsewhere.
  4. reduce_overlap           - shrink every chunk to its unique core plus
                               a 1-tile buffer against already-settled
                               (larger) chunks.
  5. add_minimum_overlap      - repair any chunk pair left merely touching
                               (zero shared tiles) after step 4, by
                               growing the smaller one by 1 tile.
  6. add_connector_chunks     - bridge any pair step 5 still couldn't fix
                               with a tiny 2-tile connector chunk.
  7. filter_and_shave_chunks  - Stage A: drop chunks with zero WorldPop
                               exposure; shave the rest down to their
                               floodable-plus-1-coarse-cell-ocean-buffer
                               footprint (fully coarse resolution - this
                               is about trimming big irrelevant margins
                               cheaply, not pixel-perfect edges).
  8. [RETIRED, 2026-08] merge_dry_chunks - used to merge any chunk with no
                               wet (ocean-mask) edge into a connected wet
                               neighbour (or drop it if unreachable),
                               deliberately ignoring max_extent to do so -
                               the actual source of this pipeline's
                               >1e9-cell oversized chunks. Superseded by
                               step 13: a dry chunk no longer needs a wet
                               edge of its own, since flood_depth_dense can
                               take boundary forcing from an
                               already-simulated neighbour chunk instead -
                               so dry chunks now simply pass through steps
                               9-12 as their own individually-shaved,
                               max_extent-respecting chunks.
  9. drop_redundant_chunks    - a chunk whose area is covered by the UNION
                               of its other overlapping chunks (not
                               necessarily a single one) by at least
                               dedup_min_coverage_fraction (2026-08 -
                               loosened from exact single-chunk
                               containment, see the function's own
                               docstring) adds nothing; always safe to
                               drop (run before AND after step 10).
  10. split_oversized_chunks  - break up anything still exceeding
                               max_extent_tiles into evenly-sized,
                               1-tile-overlapping pieces, choosing the
                               split axis as whichever keeps every piece
                               wet AND doesn't cut through a river mouth's
                               wet corner - if no axis satisfies both, the
                               chunk is left oversized rather than break
                               either guarantee.
  11. cap_overlap_density      - wherever more than max_overlap_per_cell
                               chunks cover the same underlying tile cell,
                               drop chunks that are safely redundant there
                               (every cell they touch is still covered by
                               >=2 OTHER chunks afterwards).
  13. compute_run_order       - hop-distance-from-ocean BFS over the full
                               chunk-adjacency graph (wave-0 = has a wet
                               edge), provenance-set grouping via
                               union-find (independent hinterland clusters
                               that can run concurrently), and a final
                               tie-break ordering (river-mouth priority,
                               then static adjacency degree, then - for
                               wave-0 only - ocean fraction). The resulting
                               order IS the `tile_id` assignment
                               (bboxes_to_geodataframe enumerates in list
                               order) - simulation just processes tiles in
                               tile_id order, a chunk only ever sourcing
                               boundary forcing from a strictly-lower-
                               hop_distance neighbour. Also drops any chunk
                               with no path to ocean via any chain of
                               neighbours - the sole case the retired step
                               8 used to catch, generalized here to
                               arbitrary chain length.

Output: a GeoDataFrame with one row per chunk, columns `tile_id` (int) and
`geometry` - written to `tile_grid.path` by build_tile_manifest.py, the
same schema the adaptive pipeline produced (see tiles.get_tile_geometry).
`tile_id`'s numeric VALUE (not just its existence) is meaningful since
step 13 - it IS the hop-distance-aware simulation run order, not an
arbitrary index.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from config_utils import retry_transient_io
from tiles import (
    _clamp_window,
    _coord_str,
    _degree_tiles_for_bbox,
    _mosaic_mask_for_trim,
    _open_mask_tile,
    mosaic_mask_dem_coarse,
)

_M_PER_DEG = 111_320.0  # equatorial approximation, matches tiles.py/tile_generation.py's own constant


# ---------------------------------------------------------------------------
# Stage 1 - tile index (filenames only, no raster data read)
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
# Stage 2 - floodability filter + river-mouth marking
# ---------------------------------------------------------------------------

def river_mouth_fractions(mask_band: np.ndarray, ocean_code: int, river_code: int) -> tuple[float, float]:
    """(ocean_frac, river_frac) among VALID (non-nodata, !=255) cells of a
    coarse mask array - shared by the per-tile marking below and
    split_oversized_chunks' split-cut river-mouth check.
    """
    valid = mask_band != 255
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 0.0
    ocean_frac = float((mask_band == ocean_code).sum()) / n_valid
    river_frac = float((mask_band == river_code).sum()) / n_valid
    return ocean_frac, river_frac


def is_river_mouth_signature(
    mask_band: np.ndarray, ocean_code: int, river_code: int, ocean_frac_min: float, river_frac_min: float,
) -> bool:
    """True if `mask_band` (a coarse mask array) shows an ocean-river
    MIXING signature - both `ocean_frac_min` and `river_frac_min` met
    simultaneously, not just "some of each somewhere". River channels
    occupy a much smaller share of a coarse check area than open ocean
    does even at a wide delta, so `river_frac_min` is set much lower than
    `ocean_frac_min`.
    """
    ocean_frac, river_frac = river_mouth_fractions(mask_band, ocean_code, river_code)
    return ocean_frac >= ocean_frac_min and river_frac >= river_frac_min


def _confirm_river_mouth_component(
    lat: int, lon: int, mask_index: dict, ocean_code: int,
    coarse_resolution_m: float, speckle_buffer_deg: float, min_coastal_component_cells: float,
) -> tuple[bool, int]:
    """Re-check a naively-flagged river-mouth tile against ocean
    connectivity over a WIDER buffered window, not just the isolated 1x1
    tile - DeltaDTM regularly miscodes inland water (Arctic thermokarst
    ponds, Chinese aquaculture ponds/paddies, inland lakes) as `ocean_code`,
    which can spuriously clear `ocean_frac_min` within a single tile even
    hundreds of km from real open sea (found via tests/
    river_mouth_tile_validation/validate_river_mouth_tiles.py: 5/79
    initially-flagged tiles - N30E114, N31E118, N31E120, N32E119, N68W135 -
    turned out to be exactly this, none within tens of km of the coast).
    Real coastline, even where naturally fragmented by mangroves/tidal
    channels/braided river mouths (e.g. the Ganges-Brahmaputra or Yangtze
    mouths), forms enormous connected components once viewed over a wider
    window - every confirmed real delta checked cleared >100k connected
    coarse cells within a `speckle_buffer_deg` window, vs <=1500 for the 5
    confirmed false positives, so one generous size threshold cleanly
    separates the two without needing to special-case naturally-fragmented
    coastal geometry the way a per-isolated-tile component filter would
    (that alternative was tried first and wrongly dropped Ganges/Yangtze).

    Returns (confirmed, largest connected ocean-component cell count
    touching this tile's own footprint).
    """
    bbox = (
        lon - speckle_buffer_deg, lat - speckle_buffer_deg,
        lon + 1 + speckle_buffer_deg, lat + 1 + speckle_buffer_deg,
    )
    result = _mosaic_nearest_coarse(bbox, mask_index, None, coarse_resolution_m)
    if result is None:
        return False, 0
    mask_band, _dem, _transform = result
    ocean_mask = mask_band == ocean_code
    if not ocean_mask.any():
        return False, 0
    labels, _n = ndimage.label(ocean_mask, structure=np.ones((3, 3)))
    h, w = mask_band.shape
    px_deg = (bbox[2] - bbox[0]) / w
    c0 = int(round((lon - bbox[0]) / px_deg))
    c1 = int(round((lon + 1 - bbox[0]) / px_deg))
    r1 = int(round((lat + 1 - bbox[1]) / px_deg))
    r0 = int(round((lat - bbox[1]) / px_deg))
    row0, row1 = h - r1, h - r0
    tile_labels = labels[row0:row1, c0:c1]
    touching = np.unique(tile_labels[tile_labels > 0])
    if touching.size == 0:
        return False, 0
    sizes = ndimage.sum(ocean_mask, labels, index=touching)
    return int(sizes.max()) >= min_coastal_component_cells, int(sizes.max())


_worker_mask_index: dict | None = None
_worker_dem_index: dict | None = None
_worker_elev_threshold_m: float | None = None
_worker_coarse_resolution_m: float | None = None
_worker_ocean_code: int | None = None
_worker_river_code: int | None = None
_worker_ocean_frac_min: float | None = None
_worker_river_frac_min: float | None = None


def _init_classify_tile_worker(
    mask_index: dict, dem_index: dict, elev_threshold_m: float, coarse_resolution_m: float,
    ocean_code: int, river_code: int, ocean_frac_min: float, river_frac_min: float,
) -> None:
    global _worker_mask_index, _worker_dem_index, _worker_elev_threshold_m, _worker_coarse_resolution_m
    global _worker_ocean_code, _worker_river_code, _worker_ocean_frac_min, _worker_river_frac_min
    _worker_mask_index = mask_index
    _worker_dem_index = dem_index
    _worker_elev_threshold_m = elev_threshold_m
    _worker_coarse_resolution_m = coarse_resolution_m
    _worker_ocean_code = ocean_code
    _worker_river_code = river_code
    _worker_ocean_frac_min = ocean_frac_min
    _worker_river_frac_min = river_frac_min


def _classify_tile_one(bounds: tuple[float, float, float, float]) -> dict:
    """Single coarse mosaic read, driving BOTH the floodability keep/drop
    decision and the river-mouth marking - avoids reading the same coarse
    data twice for two different questions.
    """
    result = mosaic_mask_dem_coarse(bounds, _worker_mask_index, _worker_dem_index, _worker_coarse_resolution_m)
    if result is None:
        return {"is_floodable": False, "ocean_frac": 0.0, "river_frac": 0.0, "is_river_mouth": False}
    mask_band, dem_band, _transform = result
    is_land = mask_band == 0
    dem_valid = dem_band != -9999.0
    is_floodable = bool((is_land & dem_valid & (dem_band < _worker_elev_threshold_m)).any())

    ocean_frac, river_frac = river_mouth_fractions(mask_band, _worker_ocean_code, _worker_river_code)
    is_river_mouth = ocean_frac >= _worker_ocean_frac_min and river_frac >= _worker_river_frac_min
    return {
        "is_floodable": is_floodable, "ocean_frac": ocean_frac,
        "river_frac": river_frac, "is_river_mouth": is_river_mouth,
    }


def filter_floodable_tiles(
    tiles: gpd.GeoDataFrame, mask_index: dict, dem_index: dict,
    elev_threshold_m: float, coarse_resolution_m: float, ocean_code: int, river_code: int,
    ocean_frac_min: float, river_frac_min: float,
    speckle_buffer_deg: float, min_coastal_component_cells: float,
    max_workers: int | None = None, progress_every: int = 500,
) -> gpd.GeoDataFrame:
    """Keep only tiles with at least one floodable cell (land, valid DEM,
    elevation < elev_threshold_m) - drops tiles that are fully ocean, fully
    nodata, or fully above the threshold. Runs the per-tile checks over a
    process pool - each is an independent, I/O-bound read with no shared
    state, so this parallelizes cleanly.

    Kept tiles carry 3 new columns from the same coarse read: `ocean_frac`,
    `river_frac`, `is_river_mouth` - consumed by build_chunks (below) to
    seed chunk growth at delta mouths first. Every `is_river_mouth`
    candidate from the per-tile pass is then re-checked against a wider
    window via `_confirm_river_mouth_component` (see its docstring) - the
    per-tile-only ocean_frac is spuriously cleared by inland water bodies
    DeltaDTM miscodes as `ocean_code` often enough (5/79 tiles in the real
    global run) that this second pass is not optional.
    """
    t0 = time.perf_counter()
    rows = list(tiles.itertuples(index=False))
    bounds_list = [row.geometry.bounds for row in rows]
    max_workers = max_workers or min(16, os.cpu_count() or 4)

    kept_count = 0
    mouth_count = 0
    results: list[dict] = [None] * len(rows)
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_classify_tile_worker,
        initargs=(mask_index, dem_index, elev_threshold_m, coarse_resolution_m,
                  ocean_code, river_code, ocean_frac_min, river_frac_min),
    ) as ex:
        for i, result in enumerate(ex.map(_classify_tile_one, bounds_list, chunksize=4)):
            results[i] = result
            kept_count += result["is_floodable"]
            mouth_count += result["is_river_mouth"]
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  filter_floodable_tiles: {i + 1}/{len(rows)} checked, {kept_count} floodable, "
                      f"{mouth_count} river-mouth so far ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    new_cols = ["ocean_frac", "river_frac", "is_river_mouth"]
    keep_rows = []
    for row, result in zip(rows, results):
        if not result["is_floodable"]:
            continue
        d = row._asdict()
        d.update({k: result[k] for k in new_cols})
        keep_rows.append(d)
    if not keep_rows:
        return gpd.GeoDataFrame(columns=[*tiles.columns, *new_cols], geometry="geometry", crs=tiles.crs)

    candidates = [d for d in keep_rows if d["is_river_mouth"]]
    if candidates:
        n_dropped = 0
        for d in candidates:
            confirmed, _largest = _confirm_river_mouth_component(
                d["lat"], d["lon"], mask_index, ocean_code,
                coarse_resolution_m, speckle_buffer_deg, min_coastal_component_cells,
            )
            if not confirmed:
                d["is_river_mouth"] = False
                n_dropped += 1
        print(f"  river-mouth speckle re-check: {n_dropped}/{len(candidates)} candidates dropped "
              f"(isolated inland water miscoded as ocean, not real coast)", flush=True)

    return gpd.GeoDataFrame(keep_rows, geometry="geometry", crs=tiles.crs).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 3 - presence/river-mouth grids + maximal-rectangle chunk building
# ---------------------------------------------------------------------------

def build_presence_grid(tiles: gpd.GeoDataFrame) -> tuple[np.ndarray, int, int]:
    """Boolean grid[row, col] = tile present at (lat=row+lat_min,
    lon=col+lon_min). Returns (grid, lat_min, lon_min).

    Row index increases NORTHWARD (row = lat - lat_min, and lat_min is the
    southernmost row) - the geographic convention this whole module uses,
    opposite of the usual image-array "row increases downward/south"
    convention. Column index increases EASTWARD (col = lon - lon_min).
    """
    lats = tiles["lat"].astype(int).to_numpy()
    lons = tiles["lon"].astype(int).to_numpy()
    lat_min, lat_max = int(lats.min()), int(lats.max())
    lon_min, lon_max = int(lons.min()), int(lons.max())
    grid = np.zeros((lat_max - lat_min + 1, lon_max - lon_min + 1), dtype=bool)
    grid[lats - lat_min, lons - lon_min] = True
    return grid, lat_min, lon_min


def build_river_mouth_grids(
    tiles: gpd.GeoDataFrame, lat_min: int, lon_min: int, shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """(river_mouth, ocean_present) boolean grids, same (row, col)
    convention as build_presence_grid, from tiles' `is_river_mouth`/
    `ocean_frac` columns (see filter_floodable_tiles above).
    """
    river_mouth = np.zeros(shape, dtype=bool)
    ocean_present = np.zeros(shape, dtype=bool)
    if "is_river_mouth" not in tiles.columns:
        return river_mouth, ocean_present
    lats = tiles["lat"].astype(int).to_numpy()
    lons = tiles["lon"].astype(int).to_numpy()
    rows = lats - lat_min
    cols = lons - lon_min
    river_mouth[rows, cols] = tiles["is_river_mouth"].to_numpy()
    ocean_present[rows, cols] = tiles["ocean_frac"].to_numpy() > 0
    return river_mouth, ocean_present


def _cap_range(a: int, b: int, anchor: int, max_extent: int) -> tuple[int, int]:
    """Shrink [a, b] (inclusive, containing `anchor`) to at most
    `max_extent` long, keeping as much of the natural range as fits on
    whichever side has room, sliding the window to stay within [a, b].
    """
    if b - a + 1 <= max_extent:
        return a, b
    lo = max(a, anchor - (max_extent - 1))
    hi = lo + max_extent - 1
    if hi > b:
        hi = b
        lo = hi - max_extent + 1
    return lo, hi


def _square_weighted_score(height: int, width: int) -> float:
    """Area, discounted by how far the shape is from square - so a bigger
    but very elongated rectangle doesn't automatically beat a smaller,
    squarer one; a genuine area advantage still wins, since the discount
    grows only with sqrt of the aspect ratio, not linearly with it.
    """
    aspect_ratio = max(height, width) / min(height, width)
    return (height * width) / (aspect_ratio ** 0.5)


def _maximal_rectangle_through(
    grid: np.ndarray, pr: int, pc: int, max_extent: int | None = None,
) -> tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle in `grid` containing (pr,
    pc), at most `max_extent` tiles in either direction if given. Exact
    (not greedy): considers every vertical span through pr that stays
    within the vertical run of True cells at column pc, and for each
    tracks the tightest common horizontal run across that span - correct
    because every row in the candidate vertical run is, by construction,
    part of a horizontal run that itself contains pc.

    Returns (row0, row1, col0, col1), inclusive.
    """
    n_rows, n_cols = grid.shape

    if max_extent:
        # Small search space (at most max_extent^2 candidate column
        # windows containing pc) - for each, extend the row range via the
        # same primitive edge-propagation uses, then cap it. Reuses
        # _extend_perpendicular/_cap_range instead of a separate bounded
        # variant of the unbounded sweep below.
        best_score = -1.0
        best = (pr, pr, pc, pc)
        for width in range(1, max_extent + 1):
            for lo in range(max(0, pc - width + 1), pc + 1):
                hi = lo + width - 1
                if hi >= n_cols or hi < pc:
                    continue
                if not grid[pr, lo:hi + 1].all():
                    continue
                r0, r1 = _extend_perpendicular(grid, "row", pr, (lo, hi))
                r0, r1 = _cap_range(r0, r1, pr, max_extent)
                score = _square_weighted_score(r1 - r0 + 1, width)
                if score > best_score:
                    best_score = score
                    best = (r0, r1, lo, hi)
        return best

    r0 = pr
    while r0 - 1 >= 0 and grid[r0 - 1, pc]:
        r0 -= 1
    r1 = pr
    while r1 + 1 < n_rows and grid[r1 + 1, pc]:
        r1 += 1

    rows = list(range(r0, r1 + 1))
    left = {}
    right = {}
    for r in rows:
        c0 = pc
        while c0 - 1 >= 0 and grid[r, c0 - 1]:
            c0 -= 1
        c1 = pc
        while c1 + 1 < n_cols and grid[r, c1 + 1]:
            c1 += 1
        left[r], right[r] = c0, c1

    best_area = -1
    best = (pr, pr, pc, pc)
    for a_idx, a in enumerate(rows):
        if a > pr:
            break
        max_l, min_r = -1, n_cols
        for b in rows[a_idx:]:
            max_l = max(max_l, left[b])
            min_r = min(min_r, right[b])
            if b < pr:
                continue
            height, width = b - a + 1, min_r - max_l + 1
            score = _square_weighted_score(height, width)
            if score > best_area:
                best_area = score
                best = (a, b, max_l, min_r)
    return best


def _contiguous_true_runs(arr_1d: np.ndarray, max_extent: int | None = None) -> list[tuple[int, int]]:
    """Maximal contiguous (start, end) index runs of True, inclusive - each
    further split into sub-runs of at most `max_extent` if given, so a
    single wide/tall outward run doesn't grow one oversized chunk (the
    leftover beyond a sub-run just becomes its own new chunk on the next
    pass, same as any other uncovered territory).
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


def _extend_perpendicular(
    grid: np.ndarray, fixed_axis: str, fixed_index: int, span: tuple[int, int],
    max_extent: int | None = None,
) -> tuple[int, int]:
    """Extend `fixed_index` (a row if fixed_axis == "row", a column if
    "col") in both directions along the OTHER axis, as far as the
    rectangle spanning `span` (the fixed cross-axis range) on that axis
    stays fully True - capped at `max_extent` tiles total if given.
    Returns the resulting (start, end) inclusive range along the extended
    axis.
    """
    lo, hi = span
    if fixed_axis == "row":
        n = grid.shape[0]
        a, b = fixed_index, fixed_index
        while a - 1 >= 0 and grid[a - 1, lo:hi + 1].all():
            a -= 1
        while b + 1 < n and grid[b + 1, lo:hi + 1].all():
            b += 1
    else:
        n = grid.shape[1]
        a, b = fixed_index, fixed_index
        while a - 1 >= 0 and grid[lo:hi + 1, a - 1].all():
            a -= 1
        while b + 1 < n and grid[lo:hi + 1, b + 1].all():
            b += 1
    if max_extent:
        a, b = _cap_range(a, b, fixed_index, max_extent)
    return a, b


def _seed_river_mouth_chunk(
    grid: np.ndarray, ocean_present: np.ndarray, pr: int, pc: int, max_extent: int,
) -> tuple[int, int, int, int]:
    """Seed a river-mouth cell's chunk so it hugs the coastline right at
    the delta mouth, instead of growing the plain symmetric maximal
    rectangle (which could just as easily grow away from the coast).

    Finds an orthogonal neighbour that independently touches ocean itself
    (per `ocean_present`, from filter_floodable_tiles' coarse ocean_frac) -
    a "wet-edge neighbour" - and anchors the seed to the union of (pr, pc)
    with that neighbour. The coast-facing edge of that union then stays
    FIXED (never grows further seaward); only the opposite (hinterland)
    edge extends further along that same axis, capped at max_extent total.
    The cross axis then extends in both directions as usual (no
    directional preference there), via _extend_perpendicular.

    Falls back to the plain _maximal_rectangle_through if no present,
    ocean-touching neighbour exists (e.g. an isolated river-mouth cell at
    a grid edge, or one whose only ocean-touching neighbour didn't survive
    filter_floodable_tiles' own floodability filter).
    """
    n_rows, n_cols = grid.shape
    neighbors = [
        ("N", pr + 1, pc), ("S", pr - 1, pc), ("E", pr, pc + 1), ("W", pr, pc - 1),
    ]
    anchor = None
    for direction, nr, nc in neighbors:
        if 0 <= nr < n_rows and 0 <= nc < n_cols and grid[nr, nc] and ocean_present[nr, nc]:
            anchor = (direction, nr, nc)
            break
    if anchor is None:
        return _maximal_rectangle_through(grid, pr, pc, max_extent)

    direction, nr, nc = anchor
    if direction in ("N", "S"):
        coast_row = nr  # fixed - never extends further toward open sea
        step = -1 if direction == "N" else 1  # grow away from the coast row
        free = pr
        while True:
            nxt = free + step
            if abs(nxt - coast_row) + 1 > max_extent:
                break
            if nxt < 0 or nxt >= n_rows or not grid[nxt, pc]:
                break
            free = nxt
        row0, row1 = (coast_row, free) if coast_row <= free else (free, coast_row)
        col0, col1 = _extend_perpendicular(grid, "col", pc, (row0, row1), max_extent)
    else:
        coast_col = nc
        step = -1 if direction == "E" else 1
        free = pc
        while True:
            nxt = free + step
            if abs(nxt - coast_col) + 1 > max_extent:
                break
            if nxt < 0 or nxt >= n_cols or not grid[pr, nxt]:
                break
            free = nxt
        col0, col1 = (coast_col, free) if coast_col <= free else (free, coast_col)
        row0, row1 = _extend_perpendicular(grid, "row", pr, (col0, col1), max_extent)
    return row0, row1, col0, col1


def build_chunks(
    grid: np.ndarray, max_extent: int | None = None,
    river_mouth: np.ndarray | None = None, ocean_present: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """Cover every True cell in `grid` with a set of (possibly-overlapping)
    axis-aligned all-True rectangles.

    Repeatedly: pick the first still-uncovered river-mouth cell if any
    remain (else the first still-uncovered cell in row-major order), grow
    a chunk from it (_seed_river_mouth_chunk or _maximal_rectangle_through
    respectively), then propagate outward from each of that chunk's 4
    edges - split the outward strip into contiguous present-runs, and for
    each run grow a new chunk extending freely in the perpendicular
    direction (_extend_perpendicular) - breadth-first, so this radiates
    out and covers an entire connected "pocket" before a new seed is
    picked for the next uncovered pocket. Overlap between chunks reached
    from different directions is expected, not suppressed here (cleanup
    passes handle that - see reduce_overlap/add_minimum_overlap/
    add_connector_chunks below); only an EXACT duplicate rectangle is
    skipped, to avoid ping-ponging between two chunks forever.

    No chunk exceeds `max_extent` tiles in either direction if given -
    whatever a cap excludes just gets covered by a later chunk, the same
    as any other still-uncovered territory. Returns a list of (row0, row1,
    col0, col1), inclusive.
    """
    covered = np.zeros_like(grid, dtype=bool)
    chunks: list[tuple[int, int, int, int]] = []
    seen = set()
    has_river_mouth = river_mouth is not None and river_mouth.any()

    def _add(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        if rect in seen:
            return None
        seen.add(rect)
        chunks.append(rect)
        r0, r1, c0, c1 = rect
        covered[r0:r1 + 1, c0:c1 + 1] = True
        return rect

    remaining = grid & ~covered
    while remaining.any():
        remaining_mouths = (remaining & river_mouth) if has_river_mouth else None
        if remaining_mouths is not None and remaining_mouths.any():
            pr, pc = (int(x) for x in np.argwhere(remaining_mouths)[0])
            rect = _add(_seed_river_mouth_chunk(grid, ocean_present, pr, pc, max_extent))
        else:
            pr, pc = (int(x) for x in np.argwhere(remaining)[0])
            rect = _add(_maximal_rectangle_through(grid, pr, pc, max_extent))
        queue = [rect] if rect else []

        while queue:
            r0, r1, c0, c1 = queue.pop(0)
            pushes = [
                ("up", r0 - 1 >= 0, grid[r0 - 1, c0:c1 + 1] if r0 - 1 >= 0 else None),
                ("down", r1 + 1 < grid.shape[0], grid[r1 + 1, c0:c1 + 1] if r1 + 1 < grid.shape[0] else None),
                ("left", c0 - 1 >= 0, grid[r0:r1 + 1, c0 - 1] if c0 - 1 >= 0 else None),
                ("right", c1 + 1 < grid.shape[1], grid[r0:r1 + 1, c1 + 1] if c1 + 1 < grid.shape[1] else None),
            ]
            for direction, in_bounds, outward_span in pushes:
                if not in_bounds:
                    continue
                for run_start, run_end in _contiguous_true_runs(outward_span, max_extent):
                    if direction == "up":
                        sub_c0, sub_c1 = c0 + run_start, c0 + run_end
                        new_r0, new_r1 = _extend_perpendicular(grid, "row", r0 - 1, (sub_c0, sub_c1), max_extent)
                        new_rect = (new_r0, new_r1, sub_c0, sub_c1)
                    elif direction == "down":
                        sub_c0, sub_c1 = c0 + run_start, c0 + run_end
                        new_r0, new_r1 = _extend_perpendicular(grid, "row", r1 + 1, (sub_c0, sub_c1), max_extent)
                        new_rect = (new_r0, new_r1, sub_c0, sub_c1)
                    elif direction == "left":
                        sub_r0, sub_r1 = r0 + run_start, r0 + run_end
                        new_c0, new_c1 = _extend_perpendicular(grid, "col", c0 - 1, (sub_r0, sub_r1), max_extent)
                        new_rect = (sub_r0, sub_r1, new_c0, new_c1)
                    else:  # right
                        sub_r0, sub_r1 = r0 + run_start, r0 + run_end
                        new_c0, new_c1 = _extend_perpendicular(grid, "col", c1 + 1, (sub_r0, sub_r1), max_extent)
                        new_rect = (sub_r0, sub_r1, new_c0, new_c1)

                    added = _add(new_rect)
                    if added:
                        queue.append(added)

        remaining = grid & ~covered

    return chunks


# ---------------------------------------------------------------------------
# Stage 4 - overlap reduction
# ---------------------------------------------------------------------------

def _peel_to_core(
    rect: tuple[int, int, int, int], settled: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Shrink `rect` by repeatedly removing any of its 4 edge rows/columns
    that's ENTIRELY already `settled` (covered by some other chunk) - a
    row/col with even one not-yet-settled cell is never removed, since
    that cell is uniquely this chunk's to provide. Iterates because
    removing one edge can newly make an adjacent edge's row/col fully
    settled too (its bounds just shrank). None if the whole rect collapses
    (every cell was redundant - this chunk contributes nothing unique).
    """
    r0, r1, c0, c1 = rect
    while r0 <= r1 and c0 <= c1:
        changed = False
        if settled[r0, c0:c1 + 1].all():
            r0 += 1
            changed = True
        if r0 <= r1 and settled[r1, c0:c1 + 1].all():
            r1 -= 1
            changed = True
        if r0 <= r1 and c0 <= c1 and settled[r0:r1 + 1, c0].all():
            c0 += 1
            changed = True
        if c0 <= c1 and settled[r0:r1 + 1, c1].all():
            c1 -= 1
            changed = True
        if not changed:
            break
    if r0 > r1 or c0 > c1:
        return None
    return r0, r1, c0, c1


def _restore_buffer(
    core: tuple[int, int, int, int], original: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Grow `core` back out by exactly one cell on each side that was
    peeled away in `original` - restoring a minimal connecting overlap
    with whatever chunk justified that side's peel, rather than leaving
    zero overlap (chunks should overlap by at least one tile).
    """
    r0, r1, c0, c1 = core
    or0, or1, oc0, oc1 = original
    if r0 > or0:
        r0 -= 1
    if r1 < or1:
        r1 += 1
    if c0 > oc0:
        c0 -= 1
    if c1 < oc1:
        c1 += 1
    return r0, r1, c0, c1


def _has_fat_overlap(rect: tuple[int, int, int, int], settled: np.ndarray) -> bool:
    """True if `rect`'s overlap with `settled` is more than a 1-wide/1-tall
    strip - i.e. the tight bounding box of settled cells within rect has
    BOTH dimensions > 1. Two axis-aligned rectangles always intersect in a
    rectangle, so this is exact for a single settled neighbour; against
    the accumulated union of several it's a safe, slightly coarse proxy
    (good enough to trigger the split-and-retry below).
    """
    r0, r1, c0, c1 = rect
    sub = settled[r0:r1 + 1, c0:c1 + 1]
    if not sub.any():
        return False
    rows = np.where(sub.any(axis=1))[0]
    cols = np.where(sub.any(axis=0))[0]
    return (rows.max() - rows.min() + 1) > 1 and (cols.max() - cols.min() + 1) > 1


def _peel_with_splits(
    rect: tuple[int, int, int, int], settled: np.ndarray, depth: int = 0, max_depth: int = 6,
) -> list[tuple[int, int, int, int]]:
    """_peel_to_core + _restore_buffer, with a fallback for when peeling
    gets stuck with a still-"fat" overlap remaining: this happens when a
    chunk's own unique content wraps around a settled region's corner (an
    "L" shape - e.g. a unique row on one side AND a unique column on
    another), so every edge has SOME unique cell blocking a full-edge
    peel, even though most of the chunk is redundant.

    Tries every row-split and column-split of the stuck rect, peels each
    resulting half independently against the SAME `settled` reference
    (never against each other - the two halves are disjoint, so this
    can't reproduce a mutual-erasure bug), and keeps whichever single
    split eliminates the most fat overlap with the least total area lost;
    recurses (bounded by max_depth) on any half that's still fat.
    """
    core = _peel_to_core(rect, settled)
    if core is None:
        return []
    trimmed = _restore_buffer(core, rect)
    if depth >= max_depth or not _has_fat_overlap(trimmed, settled):
        return [trimmed]

    r0, r1, c0, c1 = trimmed
    best_score = None
    best_pieces = None
    for axis, pos in [("row", p) for p in range(r0, r1)] + [("col", p) for p in range(c0, c1)]:
        if axis == "row":
            half_a, half_b = (r0, pos, c0, c1), (pos + 1, r1, c0, c1)
        else:
            half_a, half_b = (r0, r1, c0, pos), (r0, r1, pos + 1, c1)
        pieces = []
        for half in (half_a, half_b):
            half_core = _peel_to_core(half, settled)
            if half_core is not None:
                pieces.append(_restore_buffer(half_core, half))
        fat_count = sum(_has_fat_overlap(p, settled) for p in pieces)
        total_area = sum((p[1] - p[0] + 1) * (p[3] - p[2] + 1) for p in pieces)
        score = (fat_count, total_area, len(pieces))
        if best_score is None or score < best_score:
            best_score, best_pieces = score, pieces

    if not best_pieces:
        return [trimmed]  # every split dropped everything - keep the unsplit trim instead
    result = []
    for p in best_pieces:
        if _has_fat_overlap(p, settled):
            result.extend(_peel_with_splits(p, settled, depth + 1, max_depth))
        else:
            result.append(p)
    return result


def reduce_overlap(
    chunks: list[tuple[int, int, int, int]], grid_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Shrink EVERY chunk (not just slivers) down to the cells only it
    uniquely covers, plus exactly one cell of overlap on each side
    bordering another chunk - dropping it entirely if it turns out fully
    redundant. Falls back to _peel_with_splits (splitting, not just
    peeling) for chunks plain peeling can't fully reduce.

    Processed in decreasing AREA order, using only chunks ALREADY
    processed (settled) as the redundancy reference for each new one -
    never a chunk still waiting to be trimmed itself. Largest-first (not
    smallest-first) is what makes small, mostly-redundant chunks the ones
    that get peeled down: processing smallest-first would instead let
    every small chunk settle in at its full untrimmed footprint (nothing
    settled yet to peel against), forcing LARGER chunks to shrink around
    them - the opposite of the intended "small chunks yield to big ones"
    outcome.
    """
    def _area(rect: tuple[int, int, int, int]) -> int:
        r0, r1, c0, c1 = rect
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    settled = np.zeros(grid_shape, dtype=bool)
    result = []
    for rect in sorted(chunks, key=_area, reverse=True):
        for piece in _peel_with_splits(rect, settled):
            r0, r1, c0, c1 = piece
            settled[r0:r1 + 1, c0:c1 + 1] = True
            result.append(piece)
    return result


# ---------------------------------------------------------------------------
# Stage 5 - minimum-overlap repair
# ---------------------------------------------------------------------------

def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    r0, r1 = max(ar0, br0), min(ar1, br1)
    c0, c1 = max(ac0, bc0), min(ac1, bc1)
    return (r0, r1, c0, c1) if r0 <= r1 and c0 <= c1 else None


def _is_fat(bool_2d: np.ndarray) -> bool:
    if not bool_2d.any():
        return False
    rows = np.where(bool_2d.any(axis=1))[0]
    cols = np.where(bool_2d.any(axis=0))[0]
    return (rows.max() - rows.min() + 1) > 1 and (cols.max() - cols.min() + 1) > 1


def _grow_together(
    near: tuple[int, int, int, int], far: tuple[int, int, int, int], axis: str,
    grid: np.ndarray, coverage: np.ndarray, max_extent: int | None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
    """`near` sits immediately before `far` along `axis` ("row" or "col") -
    near's high edge on that axis is exactly far's low edge minus one.
    Grows whichever of the two has less area by 1 tile toward the other,
    to create exactly 1 tile of overlap - falling back to the other side
    if the preferred one is already at `max_extent`, if growing it would
    reach outside the underlying grid's actual present cells, OR if
    growing it would push its overlap with some OTHER, unrelated chunk
    (per `coverage`, a reference-counted grid of every chunk's current
    footprint) from a legitimate 1-wide strip into a "fat" (>1 x >1) one.
    `near`'s and `far`'s OWN prior contribution to `coverage` is excluded
    from that check. Returns (new_near, new_far, unresolved) - unresolved
    is True if neither side could be safely grown.
    """
    def _area(rect: tuple[int, int, int, int]) -> int:
        r0, r1, c0, c1 = rect
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    def _creates_fat_overlap(candidate: tuple[int, int, int, int]) -> bool:
        r0, r1, c0, c1 = candidate
        others = coverage[r0:r1 + 1, c0:c1 + 1].copy()
        for excl in (near, far):
            ov = _rect_overlap(candidate, excl)
            if ov is not None:
                ov_r0, ov_r1, ov_c0, ov_c1 = ov
                others[ov_r0 - r0:ov_r1 - r0 + 1, ov_c0 - c0:ov_c1 - c0 + 1] -= 1
        return _is_fat(others > 0)

    def _grow_near(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        r0, r1, c0, c1 = rect
        if axis == "col":
            if max_extent and (c1 + 1 - c0 + 1) > max_extent:
                return None
            if not grid[r0:r1 + 1, c1 + 1].all():
                return None
            candidate = (r0, r1, c0, c1 + 1)
        else:
            if max_extent and (r1 + 1 - r0 + 1) > max_extent:
                return None
            if not grid[r1 + 1, c0:c1 + 1].all():
                return None
            candidate = (r0, r1 + 1, c0, c1)
        return None if _creates_fat_overlap(candidate) else candidate

    def _grow_far(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        r0, r1, c0, c1 = rect
        if axis == "col":
            if max_extent and (c1 - (c0 - 1) + 1) > max_extent:
                return None
            if not grid[r0:r1 + 1, c0 - 1].all():
                return None
            candidate = (r0, r1, c0 - 1, c1)
        else:
            if max_extent and (r1 - (r0 - 1) + 1) > max_extent:
                return None
            if not grid[r0 - 1, c0:c1 + 1].all():
                return None
            candidate = (r0 - 1, r1, c0, c1)
        return None if _creates_fat_overlap(candidate) else candidate

    prefer_near = _area(near) <= _area(far)
    attempts = [(_grow_near, "near"), (_grow_far, "far")] if prefer_near else [(_grow_far, "far"), (_grow_near, "near")]
    for grow_fn, which in attempts:
        grown = grow_fn(near if which == "near" else far)
        if grown is not None:
            return (grown, far, False) if which == "near" else (near, grown, False)
    return near, far, True


def count_unresolved_pairs(chunks: list[tuple[int, int, int, int]]) -> int:
    """Number of chunk pairs that share a border line but zero cells."""
    n = len(chunks)
    count = 0
    for i in range(n):
        ar0, ar1, ac0, ac1 = chunks[i]
        for j in range(i + 1, n):
            br0, br1, bc0, bc1 = chunks[j]
            row_overlap = min(ar1, br1) - max(ar0, br0) + 1
            col_overlap = min(ac1, bc1) - max(ac0, bc0) + 1
            touching = (
                (ac1 + 1 == bc0 and row_overlap > 0) or (bc1 + 1 == ac0 and row_overlap > 0)
                or (ar1 + 1 == br0 and col_overlap > 0) or (br1 + 1 == ar0 and col_overlap > 0)
            )
            if touching:
                count += 1
    return count


def add_minimum_overlap(
    chunks: list[tuple[int, int, int, int]], grid: np.ndarray, max_extent: int | None = None, max_passes: int = 5,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Repair pass following reduce_overlap: that pass only ever REMOVES
    redundant overlap, it never guarantees adjacency in the first place -
    build_chunks' propagation only guarantees a chunk overlaps its direct
    parent, not every other chunk it happens to end up geometrically
    touching after the full pocket is covered.

    For every pair of chunks that share a border line but zero cells,
    grows the smaller one by exactly 1 tile toward the other (see
    _grow_together). Runs multiple passes (bounded by `max_passes`) since
    growing one chunk can surface or resolve adjacency with others -
    confirmed genuinely necessary on real data (needs 2-3 passes to reach
    a fixed point, not just 1). Returns (repaired_chunks, unresolved_count)
    - unresolved_count is pairs where neither side could be safely grown
    (already at max_extent on both sides, or the adjoining strip isn't
    fully present in `grid`) and were left as-is.
    """
    chunks = list(chunks)
    coverage = np.zeros(grid.shape, dtype=np.int16)
    for r0, r1, c0, c1 in chunks:
        coverage[r0:r1 + 1, c0:c1 + 1] += 1

    def _add_new_cells(old: tuple[int, int, int, int], new: tuple[int, int, int, int]) -> None:
        # growth only ever extends a chunk by 1 row/col - add just that
        # strip to coverage, since `old`'s own cells are already counted.
        or0, or1, oc0, oc1 = old
        nr0, nr1, nc0, nc1 = new
        if nc1 > oc1:
            coverage[nr0:nr1 + 1, nc1] += 1
        elif nc0 < oc0:
            coverage[nr0:nr1 + 1, nc0] += 1
        elif nr1 > or1:
            coverage[nr1, nc0:nc1 + 1] += 1
        elif nr0 < or0:
            coverage[nr0, nc0:nc1 + 1] += 1

    for _ in range(max_passes):
        changed = False
        n = len(chunks)
        for i in range(n):
            for j in range(i + 1, n):
                ar0, ar1, ac0, ac1 = chunks[i]
                br0, br1, bc0, bc1 = chunks[j]
                row_overlap = min(ar1, br1) - max(ar0, br0) + 1
                col_overlap = min(ac1, bc1) - max(ac0, bc0) + 1
                if ac1 + 1 == bc0 and row_overlap > 0:
                    new_a, new_b, unfixed = _grow_together(chunks[i], chunks[j], "col", grid, coverage, max_extent)
                elif bc1 + 1 == ac0 and row_overlap > 0:
                    new_b, new_a, unfixed = _grow_together(chunks[j], chunks[i], "col", grid, coverage, max_extent)
                elif ar1 + 1 == br0 and col_overlap > 0:
                    new_a, new_b, unfixed = _grow_together(chunks[i], chunks[j], "row", grid, coverage, max_extent)
                elif br1 + 1 == ar0 and col_overlap > 0:
                    new_b, new_a, unfixed = _grow_together(chunks[j], chunks[i], "row", grid, coverage, max_extent)
                else:
                    continue
                if unfixed:
                    continue
                if new_a != chunks[i]:
                    _add_new_cells(chunks[i], new_a)
                if new_b != chunks[j]:
                    _add_new_cells(chunks[j], new_b)
                if new_a != chunks[i] or new_b != chunks[j]:
                    chunks[i], chunks[j] = new_a, new_b
                    changed = True
        if not changed:
            break

    # independent growth steps can converge two different chunks onto the
    # identical final rectangle - dedupe before the final tally (and the
    # caller's output), preserving first-seen order.
    chunks = list(dict.fromkeys(chunks))
    return chunks, count_unresolved_pairs(chunks)


# ---------------------------------------------------------------------------
# Stage 6 - connector chunks (final adjacency fallback)
# ---------------------------------------------------------------------------

def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return min(ar1, br1) >= max(ar0, br0) and min(ac1, bc1) >= max(ac0, bc0)


def add_connector_chunks(
    chunks: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Final fallback for any pair of chunks that still touch but share
    zero cells after add_minimum_overlap (growth couldn't resolve them -
    either both sides were already at max_extent, or the shared strip
    needed to grow into wasn't entirely present in the underlying data).
    Rather than growing either existing chunk (exactly what already
    failed), adds a tiny new 2-cell "connector" chunk - a 1x2 or 2x1
    sliver straddling the border, built from exactly one row/column out of
    the pair's own known overlap range - guaranteed to overlap both, by
    construction.

    A pair bridged this way no longer directly overlaps each other - the
    connector sits between them instead - which is intentional and still
    serves the same purpose (a shared buffer of >=1 tile at the border);
    count_unresolved_pairs run again on the result would incorrectly
    still flag these pairs, since it only checks direct pairwise overlap.

    Returns (chunks_with_connectors, pairs_bridged).
    """
    connectors: list[tuple[int, int, int, int]] = []
    sources: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    n = len(chunks)
    for i in range(n):
        ar0, ar1, ac0, ac1 = chunks[i]
        for j in range(i + 1, n):
            br0, br1, bc0, bc1 = chunks[j]
            row_lo, row_hi = max(ar0, br0), min(ar1, br1)
            col_lo, col_hi = max(ac0, bc0), min(ac1, bc1)
            if ac1 + 1 == bc0 and row_hi >= row_lo:
                r = (row_lo + row_hi) // 2
                connector = (r, r, ac1, bc0)
            elif bc1 + 1 == ac0 and row_hi >= row_lo:
                r = (row_lo + row_hi) // 2
                connector = (r, r, bc1, ac0)
            elif ar1 + 1 == br0 and col_hi >= col_lo:
                c = (col_lo + col_hi) // 2
                connector = (ar1, br0, c, c)
            elif br1 + 1 == ar0 and col_hi >= col_lo:
                c = (col_lo + col_hi) // 2
                connector = (br1, ar0, c, c)
            else:
                continue
            connectors.append(connector)
            sources.append((chunks[i], chunks[j]))

    for connector, (a, b) in zip(connectors, sources):
        assert _overlaps(connector, a) and _overlaps(connector, b), f"connector {connector} doesn't bridge {a}, {b}"

    pairs_bridged = len(connectors)
    existing = set(chunks)
    new_connectors = list(dict.fromkeys(c for c in connectors if c not in existing))
    return chunks + new_connectors, pairs_bridged


def grid_rect_to_bbox(
    rect: tuple[int, int, int, int], lat_min: int, lon_min: int,
) -> tuple[float, float, float, float]:
    r0, r1, c0, c1 = rect
    return c0 + lon_min, r0 + lat_min, c1 + lon_min + 1, r1 + lat_min + 1


def bbox_to_grid_rect(
    bbox: tuple[float, float, float, float], lat_min: int, lon_min: int,
) -> tuple[int, int, int, int]:
    """Exact inverse of grid_rect_to_bbox - needed to resume the pipeline
    into one of the grid-rect-space stages (reduce_overlap/add_minimum_
    overlap/add_connector_chunks) from a debug GeoPackage written in
    geographic bbox space. `round()` guards against float noise from a
    round-tripped-through-disk geometry (GeoPackage storage is float64, but
    a value that started as an exact integer tile boundary can pick up
    negligible drift through I/O).
    """
    minx, miny, maxx, maxy = bbox
    r0 = round(miny - lat_min)
    r1 = round(maxy - lat_min) - 1
    c0 = round(minx - lon_min)
    c1 = round(maxx - lon_min) - 1
    return r0, r1, c0, c1


# ---------------------------------------------------------------------------
# Stage 7 - exposure filter + shave (parallel: each chunk's real-data
# checks - WorldPop exposure, DeltaDTM mask/dem classification - are fully
# independent I/O, same pattern as filter_floodable_tiles above).
# ---------------------------------------------------------------------------

def _mosaic_nearest_coarse(
    bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict | None, resolution_m: float,
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

        with _open_mask_tile(mask_path) as src:
            src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            mask_out[r0:r0 + h, c0:c0 + w] = retry_transient_io(
                src.read, 1, window=src_window, boundless=True, fill_value=255,
                out_shape=(h, w), resampling=Resampling.nearest,
            )

        dem_path = dem_index.get(coord) if dem_index is not None else None
        if dem_path is not None:
            with _open_mask_tile(dem_path) as src:
                src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
                dem_out[r0:r0 + h, c0:c0 + w] = retry_transient_io(
                    src.read, 1, window=src_window, boundless=True, fill_value=-9999.0,
                    out_shape=(h, w), resampling=Resampling.nearest,
                )

    if not any_coverage:
        return None
    return mask_out, dem_out, transform


def _first_interior_gap(mask_1d: np.ndarray, min_len: int) -> tuple[int, int] | None:
    """First (start, end) inclusive run of True in `mask_1d` at least
    `min_len` long, EXCLUDING any run touching either edge of `mask_1d`
    (that's ordinary trimmable margin, already handled by the tight-bbox
    step - not a genuine interior gap). Reuses `_contiguous_true_runs`
    (already used elsewhere for the analogous "widest all-True run"
    problem in build_chunks) rather than a new scan.
    """
    n = len(mask_1d)
    for start, end in _contiguous_true_runs(mask_1d):
        if start > 0 and end < n - 1 and (end - start + 1) >= min_len:
            return start, end
    return None


def _shave_and_split_window(
    keep: np.ndarray, ocean_any: np.ndarray,
    row0: int, row1: int, col0: int, col1: int,
    min_split_gap_cells: int, allow_split: bool,
    depth: int = 0, max_depth: int = 20,
) -> list[tuple[int, int, int, int]]:
    """Recursively tighten the coarse-cell sub-window [row0:row1+1,
    col0:col1+1] of `keep`/`ocean_any` (both full-array-relative indices)
    to its floodable content, first splitting off any wide (>=
    min_split_gap_cells) fully-unfloodable interior row/column band - e.g.
    two islands sharing one bbox with open ocean between them - before
    applying the tight-bbox-plus-ocean-buffer trim. Ported from the
    retired pre-2026-08 pipeline's `split_disconnected_tiles` (see
    memory.md's FIXED-TILE CHUNKING REDESIGN entry), simplified per
    explicit user direction: takes the FIRST qualifying gap band found
    (rows checked before columns), not the most balanced one, and doesn't
    need to explicitly preserve the gap as shared overlap in both
    children - each recursive call's own tightening step (below) already
    shaves off whatever gap remnant is left attached to it, however wide
    the real gap turns out to be (the threshold is only a MINIMUM to
    trigger a split at all, not a cap on how much gets shaved).

    Returns a list of (r0, r1, c0, c1) windows, each already tightened -
    empty if this window has no floodable content at all. When
    `allow_split` is False, always returns at most one window (the plain
    tight-bbox-plus-buffer trim, matching the pre-2026-08-this-session
    behaviour exactly) - for a caller that needs a shave/trim without ever
    letting the result fragment into multiple pieces.
    """
    sub_keep = keep[row0:row1 + 1, col0:col1 + 1]
    if not sub_keep.any():
        return []

    # Tighten to floodable content + 1-coarse-cell ocean buffer, bounded
    # by THIS window (not the full array) - a split piece must never grow
    # back past its own boundary into territory now owned by its sibling.
    rows = np.where(sub_keep.any(axis=1))[0]
    cols = np.where(sub_keep.any(axis=0))[0]
    r0, r1 = row0 + int(rows.min()), row0 + int(rows.max())
    c0, c1 = col0 + int(cols.min()), col0 + int(cols.max())
    if r0 > row0 and ocean_any[r0 - 1, c0:c1 + 1].any():
        r0 -= 1
    if r1 < row1 and ocean_any[r1 + 1, c0:c1 + 1].any():
        r1 += 1
    if c0 > col0 and ocean_any[r0:r1 + 1, c0 - 1].any():
        c0 -= 1
    if c1 < col1 and ocean_any[r0:r1 + 1, c1 + 1].any():
        c1 += 1

    if not allow_split or depth >= max_depth:
        return [(r0, r1, c0, c1)]

    unfloodable = ~keep[r0:r1 + 1, c0:c1 + 1]

    row_gap = _first_interior_gap(unfloodable.all(axis=1), min_split_gap_cells)
    if row_gap is not None:
        mid = r0 + (row_gap[0] + row_gap[1]) // 2
        top = _shave_and_split_window(
            keep, ocean_any, r0, mid, c0, c1, min_split_gap_cells, allow_split, depth + 1, max_depth,
        )
        bottom = _shave_and_split_window(
            keep, ocean_any, mid + 1, r1, c0, c1, min_split_gap_cells, allow_split, depth + 1, max_depth,
        )
        return top + bottom

    col_gap = _first_interior_gap(unfloodable.all(axis=0), min_split_gap_cells)
    if col_gap is not None:
        mid = c0 + (col_gap[0] + col_gap[1]) // 2
        left = _shave_and_split_window(
            keep, ocean_any, r0, r1, c0, mid, min_split_gap_cells, allow_split, depth + 1, max_depth,
        )
        right = _shave_and_split_window(
            keep, ocean_any, r0, r1, mid + 1, c1, min_split_gap_cells, allow_split, depth + 1, max_depth,
        )
        return left + right

    return [(r0, r1, c0, c1)]


def _shave_chunk(
    bbox: tuple[float, float, float, float], mask_index: dict, dem_index: dict,
    elev_threshold_m: float, ocean_code: int,
    resolution_m: float, sample_resolution_m: float, unfloodable_fraction: float,
    min_split_gap_cells: int, allow_split: bool,
) -> list[tuple[float, float, float, float]]:
    """Fully coarse-resolution shave - shaving is about removing BIG
    irrelevant areas cheaply, not pixel-perfect edges; never reads true
    native ~30m resolution at all. Crops `bbox` to the tight bounding
    box(es) of `resolution_m`-sized coarse cells that are NOT "mostly
    unfloodable" - a coarse cell is dropped if >= `unfloodable_fraction`
    of its estimated native pixels are ocean/lake/river, nodata, or dem >=
    threshold - keeping exactly one COARSE cell of buffer (not one native
    pixel) at edges bordering real ocean content.

    The fraction is estimated from an intermediate `sample_resolution_m`
    grid (still a direct, cheap decimated read via _mosaic_nearest_coarse,
    never native resolution) - e.g. 5x finer than `resolution_m` gives 25
    point samples per coarse cell, a statistical proxy for the true
    native-pixel fraction, not an exact count. Reading directly at
    `resolution_m` would give exactly 1 sample per cell - no fraction is
    possible from a single point, hence the two-stage sampling.

    When `allow_split` is True (2026-08), also splits off disconnected
    features (e.g. two islands sharing one bbox with open ocean between
    them) rather than only ever trimming the OUTER edges - see
    `_shave_and_split_window`'s docstring for the algorithm. Returns a
    LIST of bboxes (was a single `bbox | None`) - empty if nothing
    floodable survives, one entry in the common case, more than one only
    when a genuine internal gap gets split off.
    """
    factor = max(1, round(resolution_m / sample_resolution_m))
    sample_result = _mosaic_nearest_coarse(bbox, mask_index, dem_index, sample_resolution_m)
    if sample_result is None:
        return []
    mask_samples, dem_samples, sample_transform = sample_result

    h, w = mask_samples.shape
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        mask_samples = np.pad(mask_samples, ((0, pad_h), (0, pad_w)), constant_values=255)
        dem_samples = np.pad(dem_samples, ((0, pad_h), (0, pad_w)), constant_values=-9999.0)

    is_ocean = mask_samples == ocean_code
    is_land = mask_samples == 0
    dem_valid = dem_samples != -9999.0
    is_unfloodable = ~(is_land & dem_valid & (dem_samples < elev_threshold_m))

    ph, pw = mask_samples.shape
    coarse_h, coarse_w = ph // factor, pw // factor
    unfloodable_frac = is_unfloodable.reshape(coarse_h, factor, coarse_w, factor).mean(axis=(1, 3))
    ocean_any = is_ocean.reshape(coarse_h, factor, coarse_w, factor).any(axis=(1, 3))
    coarse_transform = sample_transform * Affine.scale(factor, factor)

    keep = unfloodable_frac < unfloodable_fraction
    if not keep.any():
        return []

    windows = _shave_and_split_window(
        keep, ocean_any, 0, coarse_h - 1, 0, coarse_w - 1, min_split_gap_cells, allow_split,
    )
    return [
        window_bounds(Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), coarse_transform)
        for r0, r1, c0, c1 in windows
    ]


def _has_exposure(bbox: tuple[float, float, float, float], data_catalog, population_source: str) -> bool:
    """True if the WorldPop population raster has any positive value
    within `bbox` - present-day product, so this filter stays in the
    frozen geometry pipeline (not scenario-dependent).

    Found in production (2026-08, second tile_generation run): for a
    handful of bboxes, `get_rasterdataset(..., bbox=...)`'s clip silently
    falls back to the FULL global population raster (43200x18720, ~3GB
    float32) instead of a small windowed read - root cause not fully
    diagnosed, but the effect is reproducible enough to guard against
    directly rather than chase further, since a global 3GB materialize per
    affected bbox both risks an ArrayMemoryError crash (which it did,
    partway through a ~50min run) and wastes time/memory even when it
    doesn't crash. Guarded two ways: (1) compare the returned raster's own
    bounds to the requested bbox BEFORE reading values - if it's far
    larger, the clip evidently failed, so skip the materialize entirely;
    (2) the actual `.values` read is still wrapped in case of a normal
    out-of-memory or I/O failure. On EITHER guard tripping, the chunk is
    KEPT (returns True), never silently dropped - a false "no exposure"
    would remove a real simulation domain from the final manifest, which
    this codebase never does silently elsewhere (see compute_run_order's
    "dropped and reported, not kept silently" for unreachable chunks -
    same principle, opposite direction: here, uncertain means "keep", not
    "drop").
    """
    try:
        da = data_catalog.get_rasterdataset(population_source, bbox=list(bbox))
    except Exception as exc:
        print(f"  WARNING: _has_exposure read failed for bbox={bbox} "
              f"({type(exc).__name__}: {exc}) - keeping chunk rather than silently dropping it", flush=True)
        return True

    req_w, req_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    try:
        b = da.raster.bounds
        got_w, got_h = b[2] - b[0], b[3] - b[1]
    except Exception:
        got_w, got_h = req_w, req_h  # can't tell - fall through to the normal read

    if req_w > 0 and req_h > 0 and (got_w > 20 * req_w or got_h > 20 * req_h):
        print(f"  WARNING: _has_exposure got a much larger raster than requested for bbox={bbox} "
              f"(requested {req_w:.2f}x{req_h:.2f}deg, got {got_w:.2f}x{got_h:.2f}deg) - bbox clip "
              f"likely failed; keeping chunk without materializing the oversized read", flush=True)
        return True

    try:
        vals = da.values
        nodata = da.raster.nodata
        valid = vals != nodata if nodata is not None else np.isfinite(vals)
        return bool(np.nansum(vals[valid]) > 0) if valid.any() else False
    except Exception as exc:
        print(f"  WARNING: _has_exposure materialize failed for bbox={bbox} "
              f"({type(exc).__name__}: {exc}) - keeping chunk rather than silently dropping it", flush=True)
        return True


_worker_data_catalog = None
_worker_population_source: str | None = None
_worker_resolution_m: float | None = None
_worker_sample_resolution_m: float | None = None
_worker_unfloodable_fraction: float | None = None
_worker_min_split_gap_cells: int | None = None


def _init_stage_a_worker(
    mask_index: dict, dem_index: dict, elev_threshold_m: float, ocean_code: int,
    catalog_path: Path, catalog_root: str, population_source: str,
    resolution_m: float, sample_resolution_m: float, unfloodable_fraction: float,
    min_split_gap_cells: int,
) -> None:
    global _worker_mask_index, _worker_dem_index, _worker_elev_threshold_m, _worker_ocean_code
    global _worker_data_catalog, _worker_population_source
    global _worker_resolution_m, _worker_sample_resolution_m, _worker_unfloodable_fraction
    global _worker_min_split_gap_cells
    from config_utils import get_data_catalog
    _worker_mask_index = mask_index
    _worker_dem_index = dem_index
    _worker_elev_threshold_m = elev_threshold_m
    _worker_ocean_code = ocean_code
    _worker_data_catalog = get_data_catalog(catalog_path, root=catalog_root)
    _worker_population_source = population_source
    _worker_resolution_m = resolution_m
    _worker_sample_resolution_m = sample_resolution_m
    _worker_unfloodable_fraction = unfloodable_fraction
    _worker_min_split_gap_cells = min_split_gap_cells


def _stage_a_one(
    bbox: tuple[float, float, float, float],
) -> tuple[list[tuple[float, float, float, float]], list[tuple[tuple[float, float, float, float], str]]]:
    """Returns (kept, dropped) - `kept` is 0+ shaved/split piece(s) (more
    than one only when a genuine internal island/gap split fired - see
    _shave_chunk); `dropped` is a list of (piece, reason) pairs.

    Order (2026-08): shave/split FIRST, then check WorldPop exposure on
    EACH resulting piece independently - reordered from the original
    "check exposure on the whole pre-split input, then shave" per explicit
    user direction, since checking once on the ORIGINAL bbox could wrongly
    keep an uninhabited split-off piece (e.g. an empty islet) just because
    SOME other part of the original bbox had population, once one input
    can produce more than one independent output piece. This does give up
    the previous early-exit optimization (skip the expensive shave/split
    read entirely for a chunk with zero population anywhere) - accepted,
    since correctness of the per-piece exposure filter matters more than
    that optimization, and most candidate chunks reaching this stage
    already passed the floodability filter, so a zero-exposure chunk is
    the less common case anyway.

    _shave_chunk's raster reads are wrapped in a try/except: found necessary
    in production (2026-08) when a single transient read failure (a
    momentary file-lock/unavailability that outlasted retry_transient_io's
    3 retries over 15s - the file existed and was readable moments later)
    crashed an entire ~78-minute global run. On failure the chunk is KEPT
    UNSHAVED (its original, un-cropped bbox, as a single piece - never
    split, since the read that would have driven splitting is exactly what
    failed) UNCONDITIONALLY, skipping the exposure check entirely - same
    "uncertain means keep, not drop" principle as _has_exposure itself;
    having already decided not to trust this read enough to shave/split,
    layering an exposure decision on top of it would undermine that same
    caution.
    """
    try:
        shaved = _shave_chunk(
            bbox, _worker_mask_index, _worker_dem_index, _worker_elev_threshold_m, _worker_ocean_code,
            _worker_resolution_m, _worker_sample_resolution_m, _worker_unfloodable_fraction,
            _worker_min_split_gap_cells, True,
        )
    except Exception as exc:
        print(f"  WARNING: _shave_chunk failed for bbox={bbox} ({type(exc).__name__}: {exc}) - "
              f"keeping chunk UNSHAVED rather than dropping it or crashing the run", flush=True)
        return [bbox], []

    if not shaved:
        return [], [(bbox, "shave_found_no_floodable_area")]

    kept: list[tuple[float, float, float, float]] = []
    dropped: list[tuple[tuple[float, float, float, float], str]] = []
    for piece in shaved:
        if _has_exposure(piece, _worker_data_catalog, _worker_population_source):
            kept.append(piece)
        else:
            dropped.append((piece, "no_population_exposure"))
    return kept, dropped


def filter_and_shave_chunks(
    bboxes: list[tuple[float, float, float, float]],
    mask_index: dict, dem_index: dict, elev_threshold_m: float, ocean_code: int,
    catalog_path: Path, catalog_root: str, population_source: str,
    coarse_resolution_m: float, shave_sample_resolution_m: float, shave_unfloodable_fraction: float,
    min_split_gap_cells: int,
    max_workers: int | None = None, progress_every: int = 200,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[tuple[float, float, float, float], str]]]:
    """Stage A: shave each chunk (possibly splitting off a disconnected
    interior feature in the process - see _shave_chunk), THEN drop any
    resulting piece with zero WorldPop exposure (see _stage_a_one for why
    exposure is checked per-piece, after shaving/splitting, not once on
    the original pre-split input). Each chunk's checks are fully
    independent, so this runs over a process pool.

    Returns (kept, dropped) - `kept` is a FLAT list of every surviving
    output piece across all inputs (one input can produce zero, one, or
    several pieces - see _shave_chunk's internal-split behaviour);
    `dropped` is a list of (piece_or_input_bbox, reason) pairs (see
    _stage_a_one's drop reasons) - keyed by the SHAVED PIECE for a
    per-piece drop (e.g. "no_population_exposure"), or by the original
    input bbox for a whole-input drop (e.g. "shave_found_no_floodable_
    area", where no piece ever existed to key by). Consumed by
    build_tile_manifest.py to write the tile-generation debug GeoPackage
    output alongside the production manifest.
    """
    t0 = time.perf_counter()
    max_workers = max_workers or min(16, os.cpu_count() or 4)
    kept: list[tuple[float, float, float, float]] = []
    dropped: list[tuple[tuple[float, float, float, float], str]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_stage_a_worker,
        initargs=(mask_index, dem_index, elev_threshold_m, ocean_code, catalog_path, catalog_root,
                  population_source, coarse_resolution_m, shave_sample_resolution_m, shave_unfloodable_fraction,
                  min_split_gap_cells),
    ) as ex:
        for i, (piece_kept, piece_dropped) in enumerate(ex.map(_stage_a_one, bboxes, chunksize=4)):
            kept.extend(piece_kept)
            dropped.extend(piece_dropped)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  filter_and_shave_chunks: {i + 1}/{len(bboxes)} checked, {len(kept)} kept so far "
                      f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    return kept, dropped


# ---------------------------------------------------------------------------
# Stage 8 - wet-edge check + two-phase dry-chunk merge
# ---------------------------------------------------------------------------

def _native_pixel_size(
    bbox: tuple[float, float, float, float], mask_index: dict,
) -> tuple[float, float] | None:
    """(px_size_x, px_size_y) in degrees - the finest resolution among mask
    tiles overlapping `bbox`, read from file headers only (no pixel data).
    """
    resolutions = []
    for lat, lon in _degree_tiles_for_bbox(bbox):
        path = mask_index.get(_coord_str(lat, lon))
        if path is None:
            continue
        with _open_mask_tile(path) as src:
            resolutions.append((abs(src.transform.a), abs(src.transform.e)))
    if not resolutions:
        return None
    return min(r[0] for r in resolutions), min(r[1] for r in resolutions)


def _has_wet_edge(bbox: tuple[float, float, float, float], mask_index: dict, ocean_code: int) -> bool:
    """True if the single outermost native-resolution pixel row/column on
    any of the 4 sides of `bbox` contains an ocean-mask cell (mask ==
    `ocean_code` specifically - never lake/river).

    A chunk can span many tiles, so mosaicking the WHOLE bbox at native
    resolution just to inspect its boundary would be wasted work - can be
    hundreds of millions of pixels read and allocated to look at a handful
    of edge rows/columns. Instead, reads only 4 thin (1-native-pixel-wide)
    strips directly.
    """
    px = _native_pixel_size(bbox, mask_index)
    if px is None:
        return False
    px_size_x, px_size_y = px
    minx, miny, maxx, maxy = bbox
    strips = [
        (minx, maxy - px_size_y, maxx, maxy),  # top row
        (minx, miny, maxx, miny + px_size_y),  # bottom row
        (minx, miny, minx + px_size_x, maxy),  # left column
        (maxx - px_size_x, miny, maxx, maxy),  # right column
    ]
    for strip_bbox in strips:
        result = _mosaic_mask_for_trim(strip_bbox, mask_index)
        if result is None:
            continue
        mask_band, _transform = result
        if (mask_band == ocean_code).any():
            return True
    return False


def _bboxes_touch_or_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _init_wet_edge_worker(mask_index: dict, ocean_code: int) -> None:
    global _worker_mask_index, _worker_ocean_code
    _worker_mask_index = mask_index
    _worker_ocean_code = ocean_code


def _wet_edge_one(bbox: tuple[float, float, float, float]) -> bool:
    return _has_wet_edge(bbox, _worker_mask_index, _worker_ocean_code)


def _classify_wet(
    grid_map: dict[int, tuple[float, float, float, float]], mask_index: dict, ocean_code: int,
    max_workers: int | None = None, progress_every: int = 500,
) -> dict[int, bool]:
    """Wet/dry classification for every id currently in `grid_map`, over a
    process pool - used by compute_run_order (Stage 13) to identify
    wave-0 (ocean-touching) chunks, the BFS sources for hop-distance.
    """
    ids = list(grid_map.keys())
    bboxes = [grid_map[i] for i in ids]
    max_workers = max_workers or min(16, os.cpu_count() or 4)
    t0 = time.perf_counter()
    wet_list = []
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_wet_edge_worker, initargs=(mask_index, ocean_code),
    ) as ex:
        for i, is_wet in enumerate(ex.map(_wet_edge_one, bboxes, chunksize=4)):
            wet_list.append(is_wet)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    classify_wet: {i + 1}/{len(bboxes)} checked "
                      f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    return dict(zip(ids, wet_list))


# ---------------------------------------------------------------------------
# Stage 9/10/11 - subset dedup, split oversized (+ river-mouth skip),
# cap overlap density
# ---------------------------------------------------------------------------

def drop_redundant_chunks(
    bboxes: list[tuple[float, float, float, float]], coverage_threshold: float = 0.998,
) -> list[tuple[float, float, float, float]]:
    """Iteratively drop any chunk whose area is covered - by the UNION of
    every other still-alive OVERLAPPING chunk, not just a single one - by
    at least `coverage_threshold`.

    Loosened 2026-08 from exact (100%) single-chunk containment: real
    chunks coming out of the shave step can differ from a neighbour by a
    sliver of a coarse cell's width (a few hundred metres), which exact
    containment - and even a single-chunk-only containment check - both
    missed even though the chunk was still genuinely redundant once every
    overlapping neighbour's territory is combined. Confirmed against real
    data (tile_id 536/850, 2026-08): 536 was only ~99.47% covered by its
    single largest neighbour; 850 was covered 100% only by the UNION of 5
    neighbours, no single one covering more than ~60% of it alone.

    Processes candidates smallest-area-first (same convention as
    cap_overlap_density), dropping one at a time and recomputing coverage
    for its neighbours before continuing - removing a chunk can only ever
    REDUCE another chunk's coverage fraction (never increase it, since
    coverage is a union of a shrinking set), so a chunk that never
    qualified can never later qualify, but a chunk that currently
    qualifies can stop qualifying once one of ITS OWN covering neighbours
    gets dropped first (a real mutual-dependency case, not just a
    theoretical one - see tests/tile_dedup_validation/). This means only
    the dropped chunk's own neighbours ever need re-checking, never the
    whole set.
    """
    n = len(bboxes)
    geoms = [box(*b) for b in bboxes]
    areas = [g.area for g in geoms]
    alive = [True] * n

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _bboxes_touch_or_overlap(bboxes[i], bboxes[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    def _coverage_frac(i: int) -> float:
        others = [geoms[j] for j in adjacency[i] if alive[j]]
        if not others:
            return 0.0
        return geoms[i].intersection(unary_union(others)).area / areas[i]

    candidates = {i for i in range(n) if _coverage_frac(i) >= coverage_threshold}
    n_dropped = 0
    while candidates:
        i = min(candidates, key=lambda k: (areas[k], k))
        candidates.discard(i)
        if _coverage_frac(i) < coverage_threshold:
            continue  # a neighbour dropped since this was queued pushed it below threshold
        alive[i] = False
        n_dropped += 1
        for j in adjacency[i]:
            if alive[j] and j in candidates and _coverage_frac(j) < coverage_threshold:
                candidates.discard(j)

    if n_dropped:
        print(f"  drop_redundant_chunks: dropped {n_dropped} chunk(s) "
              f"(>= {coverage_threshold * 100:.1f}% covered by other overlapping chunks)", flush=True)
    return [b for b, a in zip(bboxes, alive) if a]


def _split_piece_count(n_tiles: int, max_extent: int) -> int:
    """Smallest k such that n_tiles can be covered by k pieces, each
    overlapping its neighbour(s) by exactly 1 tile, with no piece
    exceeding max_extent."""
    if n_tiles <= max_extent:
        return 1
    k = 2
    while True:
        total = n_tiles + (k - 1)
        if -(-total // k) <= max_extent:  # ceil division
            return k
        k += 1


def _even_overlap_sizes(n_tiles: int, max_extent: int) -> list[int]:
    """Piece sizes (tiles), each overlapping the next by 1, distinct total
    = n_tiles, distributed as evenly as possible (varying by at most 1
    tile) rather than lopsided - e.g. n_tiles=8, max_extent=4 -> [4,3,3],
    not [4,4,2].
    """
    k = _split_piece_count(n_tiles, max_extent)
    if k == 1:
        return [n_tiles]
    total = n_tiles + (k - 1)
    base, rem = divmod(total, k)
    return [base + 1 if i < rem else base for i in range(k)]


def _piece_offsets(sizes: list[int]) -> list[int]:
    """Cumulative start offset (tile units) for each piece - each starts 1
    tile before the previous piece's end, creating the 1-tile overlap."""
    offsets = [0]
    for i in range(1, len(sizes)):
        offsets.append(offsets[i - 1] + sizes[i - 1] - 1)
    return offsets


def _split_axis_pieces(bbox: tuple, axis: str, max_extent: int) -> list[tuple]:
    minx, miny, maxx, maxy = bbox
    if axis == "col":
        n_tiles = max(1, round(maxx - minx))
        sizes = _even_overlap_sizes(n_tiles, max_extent)
        offsets = _piece_offsets(sizes)
        pieces = []
        for i, (off, sz) in enumerate(zip(offsets, sizes)):
            px0 = minx if i == 0 else minx + off
            px1 = maxx if i == len(sizes) - 1 else minx + off + sz
            pieces.append((px0, miny, px1, maxy))
        return pieces
    n_tiles = max(1, round(maxy - miny))
    sizes = _even_overlap_sizes(n_tiles, max_extent)
    offsets = _piece_offsets(sizes)
    pieces = []
    for i, (off, sz) in enumerate(zip(offsets, sizes)):
        py0 = miny if i == 0 else miny + off
        py1 = maxy if i == len(sizes) - 1 else miny + off + sz
        pieces.append((minx, py0, maxx, py1))
    return pieces


def _cut_check_bbox(
    bbox: tuple, axis: str, pieces: list[tuple], i: int, band_cells: float, resolution_m: float,
) -> tuple:
    """Narrow band straddling the boundary between pieces[i] and
    pieces[i + 1] (their 1-tile overlap zone), expanded by `band_cells`
    coarse cells on each side along the split axis - full cross-axis
    extent, since a delta mouth could sit anywhere along that cross
    section, not just at its midpoint.
    """
    minx, miny, maxx, maxy = bbox
    band_deg = band_cells * resolution_m / _M_PER_DEG
    if axis == "col":
        cut_lo, cut_hi = pieces[i + 1][0], pieces[i][2]
        return (cut_lo - band_deg, miny, cut_hi + band_deg, maxy)
    cut_lo, cut_hi = pieces[i + 1][1], pieces[i][3]
    return (minx, cut_lo - band_deg, maxx, cut_hi + band_deg)


def _axis_hits_river_mouth(
    bbox: tuple, axis: str, pieces: list[tuple], mask_index: dict,
    ocean_code: int, river_code: int, ocean_frac_min: float, river_frac_min: float,
    band_cells: float, resolution_m: float,
) -> bool:
    """True if ANY adjacent-piece cut along `axis` shows a river-mouth
    signature in its narrow surrounding band - splitting there risks
    severing a delta mouth's wet corner from the rest of its floodplain.
    """
    for i in range(len(pieces) - 1):
        check_bbox = _cut_check_bbox(bbox, axis, pieces, i, band_cells, resolution_m)
        result = _mosaic_nearest_coarse(check_bbox, mask_index, None, resolution_m)
        if result is None:
            continue
        mask_band, _dem_band, _transform = result
        if is_river_mouth_signature(mask_band, ocean_code, river_code, ocean_frac_min, river_frac_min):
            return True
    return False


def split_oversized_chunks(
    bboxes: list[tuple], mask_index: dict, ocean_code: int, river_code: int,
    ocean_frac_min: float, river_frac_min: float, river_mouth_band_cells: float, coarse_resolution_m: float,
    max_extent: int,
) -> list[tuple]:
    """Break up any chunk exceeding max_extent in either dimension into
    evenly-sized, 1-tile-overlapping pieces (see _even_overlap_sizes),
    choosing the split axis as whichever leaves every resulting piece with
    a wet edge AND doesn't cut through a river mouth's wet corner (see
    _axis_hits_river_mouth). If NO candidate axis satisfies BOTH - every
    way of splitting this chunk would either strand a dry piece or sever a
    delta mouth - the chunk is left oversized and unsplit instead: neither
    is worth trading away just to stay under max_extent. Queue-based: each
    (successfully) split piece is re-checked and split again if it's still
    oversized in the axis not just split.
    """
    queue = list(bboxes)
    n_input = len(queue)
    done = []
    t0 = time.perf_counter()
    n_processed = 0
    while queue:
        bbox = queue.pop(0)
        n_processed += 1
        if n_processed % 500 == 0:
            print(f"    split_oversized_chunks: {n_processed} processed so far "
                  f"({n_input} in original input, {len(queue)} still queued, "
                  f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)
        minx, miny, maxx, maxy = bbox
        width_tiles = round(maxx - minx)
        height_tiles = round(maxy - miny)
        if width_tiles <= max_extent and height_tiles <= max_extent:
            done.append(bbox)
            continue

        candidates = []
        if width_tiles > max_extent:
            candidates.append("col")
        if height_tiles > max_extent:
            candidates.append("row")

        all_wet_options = []
        for axis in candidates:
            pieces = _split_axis_pieces(bbox, axis, max_extent)
            if not all(_has_wet_edge(p, mask_index, ocean_code) for p in pieces):
                continue
            if _axis_hits_river_mouth(
                bbox, axis, pieces, mask_index, ocean_code, river_code,
                ocean_frac_min, river_frac_min, river_mouth_band_cells, coarse_resolution_m,
            ):
                continue
            all_wet_options.append((axis, pieces))

        if not all_wet_options:
            # every candidate axis would either strand a dry piece or cut
            # through a river mouth - keep this chunk intact (oversized)
            # rather than accept either.
            done.append(bbox)
            continue

        _, best_pieces = all_wet_options[0]
        queue.extend(best_pieces)

    return done


def _chunk_cells(bbox: tuple, lat_min: int, lon_min: int) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bbox
    r0 = int(np.floor(miny)) - lat_min
    r1 = int(np.ceil(maxy)) - lat_min
    c0 = int(np.floor(minx)) - lon_min
    c1 = int(np.ceil(maxx)) - lon_min
    return r0, r1, c0, c1


def cap_overlap_density(bboxes: list[tuple], max_overlap: int) -> list[tuple]:
    """Wherever more than `max_overlap` chunks cover the same underlying
    1deg DeltaDTM tile cell, drop chunks that are safely redundant there
    (smallest first) until the cap holds. "Safely redundant" means every
    cell the chunk touches is ALSO covered by at least 2 OTHER chunks
    (count >= 3 including itself) - not just >=1 other. Requiring only
    one other covering chunk isn't enough: if a cell is covered by exactly
    this chunk and one other, removing this one leaves that other chunk
    with ZERO remaining overlap there - breaking the buffer/connectivity
    invariant even though bare coverage (at least one chunk present)
    survives.
    """
    bboxes = list(bboxes)
    all_minx = min(b[0] for b in bboxes)
    all_miny = min(b[1] for b in bboxes)
    all_maxx = max(b[2] for b in bboxes)
    all_maxy = max(b[3] for b in bboxes)
    lat_min, lon_min = int(np.floor(all_miny)), int(np.floor(all_minx))
    n_rows = int(np.ceil(all_maxy)) - lat_min
    n_cols = int(np.ceil(all_maxx)) - lon_min

    count = np.zeros((n_rows, n_cols), dtype=np.int32)
    cells = [None] * len(bboxes)
    for i, bbox in enumerate(bboxes):
        r0, r1, c0, c1 = _chunk_cells(bbox, lat_min, lon_min)
        cells[i] = (r0, r1, c0, c1)
        count[r0:r1, c0:c1] += 1

    alive = [True] * len(bboxes)

    def _safely_redundant(i: int) -> bool:
        r0, r1, c0, c1 = cells[i]
        return bool((count[r0:r1, c0:c1] >= 3).all())

    def _remove(i: int) -> None:
        r0, r1, c0, c1 = cells[i]
        count[r0:r1, c0:c1] -= 1
        alive[i] = False

    n_dropped = 0
    while True:
        over = np.argwhere(count > max_overlap)
        if len(over) == 0:
            break
        r, c = (int(x) for x in over[0])
        involved = [
            i for i in range(len(bboxes))
            if alive[i] and cells[i][0] <= r < cells[i][1] and cells[i][2] <= c < cells[i][3]
        ]
        candidates = sorted(
            (i for i in involved if _safely_redundant(i)),
            key=lambda i: (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]),
        )
        if not candidates:
            # nothing here is safely removable - leave this cell over the
            # cap rather than risk coverage/connectivity.
            print(f"  cap_overlap_density: cell (row {r}, col {c}) still over cap "
                  f"({int(count[r, c])} chunks) - no safely-removable chunk found, leaving as-is", flush=True)
            count[r, c] = max_overlap  # stop re-flagging this exact cell
            continue
        _remove(candidates[0])
        n_dropped += 1

    print(f"  cap_overlap_density: dropped {n_dropped} chunk(s)", flush=True)
    return [b for b, a in zip(bboxes, alive) if a]


# ---------------------------------------------------------------------------
# Stage 13 - run-order: hop-distance BFS, provenance-set grouping, tie-break
# ordering (2026-08 - see module docstring's retired step 8 for why dry
# chunks no longer get merged away instead of ordered).
# ---------------------------------------------------------------------------

def _bbox_ocean_fraction(
    bbox: tuple[float, float, float, float], mask_index: dict, ocean_code: int, coarse_resolution_m: float,
) -> float:
    """Fraction of `bbox`'s valid (non-nodata) coarse cells that are
    ocean-mask - used only as a wave-0 chunk's ordering tie-break (most
    reliable ocean signal first), NOT for wet/dry classification itself
    (`_classify_wet`/`_has_wet_edge` is a stricter edge-only test used for
    that). Reuses `mosaic_mask_dem_coarse` - the same coarse-resolution
    reader steps 7/10 already use at chunk scale.
    """
    result = mosaic_mask_dem_coarse(bbox, mask_index, None, coarse_resolution_m)
    if result is None:
        return 0.0
    mask_band, _dem_band, _transform = result
    valid = mask_band != 255
    if not valid.any():
        return 0.0
    return float((mask_band[valid] == ocean_code).sum()) / float(valid.sum())


def _build_adjacency_graph(bboxes: list[tuple[float, float, float, float]]) -> list[list[int]]:
    """Adjacency list over `bboxes` under `_bboxes_touch_or_overlap` - O(n^2)
    pairwise, shared by compute_run_order and compute_neighbor_sample_points
    (each builds its own copy; cheap enough at this chunk count - a few
    thousand chunks, low tens of millions of cheap 4-tuple comparisons -
    not to be worth threading a shared graph through both signatures).
    """
    n = len(bboxes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        bi = bboxes[i]
        for j in range(i + 1, n):
            if _bboxes_touch_or_overlap(bi, bboxes[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def compute_run_order(
    bboxes: list[tuple[float, float, float, float]],
    mask_index: dict,
    ocean_code: int,
    river_mouth_seeds: gpd.GeoDataFrame,
    coarse_resolution_m: float,
    max_workers: int | None = None,
) -> tuple[list[int], list[int], dict[str, list]]:
    """Stage 13: hop-distance-from-ocean BFS + provenance-set grouping +
    tie-break ordering, over the FULL final chunk set (no chunk has been
    merged away for lacking a wet edge - see module docstring's retired
    step 8).

    A chunk with any ocean-mask edge (`_classify_wet`/`_has_wet_edge`) is
    "wave 0" (hop_distance=0) - these need no ordering relative to each
    other and sort purely by descending ocean fraction. Every other
    chunk's hop_distance is its BFS distance, over the chunk-adjacency
    graph (`_bboxes_touch_or_overlap`), to the nearest wave-0 chunk. A
    chunk may only source boundary forcing from a neighbour at STRICTLY
    lower hop_distance (never same-or-higher, even if touching) - so
    grouping and ordering both only ever look "backwards" along
    hop_distance.

    Provenance-set grouping (union-find): a hinterland chunk is unioned
    with every strictly-lower-hop_distance neighbour it touches. Since
    provenance flows along exactly those edges, this directly computes
    "shares >= 1 common wave-0 ancestor" groups without ever materializing
    the actual ancestor sets - two chunks end up in the same group iff
    some chain of such edges connects them, which is exactly the
    "inherits both ocean tile ids -> merges the groups" behaviour the
    design calls for. Different groups are provably independent (no chain
    of strictly-decreasing-hop_distance edges connects them) and can run
    concurrently.

    Returns `(order, unreachable_indices, diagnostics)`:
      - `order`: permutation of the REACHABLE indices (0..len(bboxes)-1,
        excluding `unreachable_indices`), in final processing order. The
        caller reindexes `bboxes` by `order` before assigning `tile_id`
        (`bboxes_to_geodataframe` just enumerates list order), so `order`
        directly becomes the `tile_id` sequence.
      - `unreachable_indices`: chunks with NO path to any wave-0 chunk
        through any chain of neighbours - the sole remaining case the
        retired `merge_dry_chunks` used to catch (as "no reachable wet
        neighbour"), just generalized here to arbitrary chain length.
      - `diagnostics`: dict of per-original-index lists (`hop_distance`,
        `group_id`, `is_river_mouth_seed`, `degree`, `ocean_fraction`),
        aligned to `bboxes`' original order - for the Stage 13 debug
        GeoPackage, never written to the production manifest itself.
    """
    n = len(bboxes)
    grid_map = dict(enumerate(bboxes))

    print(f"  compute_run_order: classifying wet/dry ({n} chunks)", flush=True)
    wet = _classify_wet(grid_map, mask_index, ocean_code, max_workers)

    print("  compute_run_order: building adjacency graph", flush=True)
    adjacency = _build_adjacency_graph(bboxes)

    print("  compute_run_order: BFS hop-distance from wave-0 chunks", flush=True)
    hop_distance = [-1] * n
    dq: deque[int] = deque()
    for i in range(n):
        if wet[i]:
            hop_distance[i] = 0
            dq.append(i)
    while dq:
        cur = dq.popleft()
        for nxt in adjacency[cur]:
            if hop_distance[nxt] == -1:
                hop_distance[nxt] = hop_distance[cur] + 1
                dq.append(nxt)
    unreachable_indices = [i for i in range(n) if hop_distance[i] == -1]
    if unreachable_indices:
        print(f"  compute_run_order: {len(unreachable_indices)} chunk(s) unreachable "
              f"from any wave-0 chunk via any chain", flush=True)

    print("  compute_run_order: provenance-set grouping (union-find)", flush=True)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if hop_distance[i] <= 0:
            continue
        for nb in adjacency[i]:
            if hop_distance[nb] != -1 and hop_distance[nb] < hop_distance[i]:
                _union(i, nb)
    # -1 sentinel for wave-0 rows: group concept is irrelevant there since
    # hop_distance=0 already sorts them into their own contiguous block.
    group_id = [_find(i) if hop_distance[i] > 0 else -1 for i in range(n)]

    print("  compute_run_order: river-mouth flag + ocean fraction", flush=True)
    is_river_mouth = [False] * n
    if len(river_mouth_seeds):
        seeds_prepared = prep(unary_union(list(river_mouth_seeds.geometry)))
        for i in range(n):
            if seeds_prepared.intersects(box(*bboxes[i])):
                is_river_mouth[i] = True

    ocean_fraction = [0.0] * n
    for i in range(n):
        if wet[i]:
            ocean_fraction[i] = _bbox_ocean_fraction(bboxes[i], mask_index, ocean_code, coarse_resolution_m)

    degree = [len(adjacency[i]) for i in range(n)]

    reached = [i for i in range(n) if hop_distance[i] != -1]
    order = sorted(
        reached,
        key=lambda i: (hop_distance[i], group_id[i], not is_river_mouth[i], -degree[i], -ocean_fraction[i]),
    )

    diagnostics = {
        "hop_distance": hop_distance,
        "group_id": group_id,
        "is_river_mouth_seed": is_river_mouth,
        "degree": degree,
        "ocean_fraction": ocean_fraction,
    }
    n_wave0 = sum(1 for h in hop_distance if h == 0)
    print(f"  compute_run_order: {n_wave0}/{n} wave-0 chunks, "
          f"{len(set(group_id)) - (1 if -1 in group_id else 0)} hinterland group(s)", flush=True)
    return order, unreachable_indices, diagnostics


# NOT IMPLEMENTED (2026-08) - inter-chunk boundary propagation for hop>=1
# hinterland chunks (see compute_run_order above and the retired
# merge_dry_chunks entry in this module's own docstring) still needs a step
# that extracts real neighbour-water-level boundary points for each hop>=1
# chunk. A tile-generation-time precompute (candidate point LOCATIONS from
# frozen geometry alone, values filled in later) was tried and rejected:
# gridding a hop>=1 chunk's full overlap with its earlier neighbour(s) at
# any reasonable fixed resolution produces a huge, mostly-useless point
# count (a real test run: 8.65 MILLION candidate points across only 119
# chunks, avg ~72,700/chunk, since real overlaps here span up to hundreds
# of km, not thin border strips) - almost all of which would land on dry
# land anyway. The right place to pick points is AFTER a wave-0 chunk has
# actually been simulated: read its real waterdepth output and keep only
# the (very few, genuinely wet) non-zero/non-NaN cells within the overlap
# - a step that has to run BETWEEN wave-0 and wave>=1 simulations, which
# doesn't exist yet (see the inter-chunk boundary propagation plan's
# "Deferred consideration" section for the related DAG/HPC-orchestration
# gap). `boundaries.sample_waterlevel_at_points` (the raster point-lookup
# primitive, dtype-agnostic, dry/nodata-aware) already exists and is ready
# to be used by that step once it's built - only the "which points, and
# when" piece is still open.


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def bboxes_to_geodataframe(bboxes: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    """One row per chunk: `tile_id` (int) + geometry - the schema every
    downstream consumer (tiles.get_tile_geometry, Snakefile's TILE_IDS)
    expects, regardless of which tile-generation approach produced it.
    """
    rows = [{"tile_id": i, "geometry": box(*bbox)} for i, bbox in enumerate(bboxes)]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
