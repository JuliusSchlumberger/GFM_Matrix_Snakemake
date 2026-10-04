"""Shared low-level tile/raster primitives for tile generation.

RETIRED (2026-10): this module used to hold the full 13-stage greedy-
covering/shave chunk-generation pipeline (build_tile_index -> filter_
floodable_tiles -> build_chunks -> reduce_overlap/add_minimum_overlap/
add_connector_chunks -> filter_and_shave_chunks -> drop_redundant_chunks ->
split_oversized_chunks -> cap_overlap_density -> compute_run_order). That
whole pipeline is gone - its per-tile independent shave decisions could
silently erode the overlap meant to preserve flood connectivity across a
chunk boundary, among other issues found during the connectivity-first
migration. Superseded by src/connectivity_tiling.py's connectivity-first
domain construction, orchestrated by preparation/build_tile_manifest.py -
see docs/methods_01_tile_processing_and_waterlevels.md section 3 for the
full design, the "why this is correct" argument, and real validation
numbers (a full-world run: 4,504 final domains, median 24.8M cells,
hop-distance chains up to depth 10, 114 domains correctly dropped as
unreachable).

What's left here are the few primitives src/connectivity_tiling.py still
imports directly, kept in place rather than moved, since nothing about
them was specific to the retired pipeline:
  - build_tile_index / _parse_coord_from_filename - Phase 0's raw tile
    index, built purely from mask tile filenames, no raster read.
  - _mosaic_nearest_coarse - the shared coarse mask+DEM reader every phase
    of the new method uses (Phase 1's edge-connectivity check, Phase 3's
    trim/split grid, Phase 5's ocean-touch check).
  - _first_interior_gap / _contiguous_true_runs - Phase 3's natural-gap
    search.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

from config_utils import retry_transient_io
from tiles import _clamp_window, _coord_str, _degree_tiles_for_bbox, _open_mask_tile

_M_PER_DEG = 111_320.0  # equatorial approximation, matches tiles.py's own constant


# ---------------------------------------------------------------------------
# Phase 0 - tile index (filenames only, no raster data read)
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
# Shared coarse mosaic reader - every phase of the new connectivity-first
# method (src/connectivity_tiling.py) goes through this.
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
    unchanged behaviour for every existing caller. Added for
    src/connectivity_tiling.py's Phase 3 (split_component_to_budget): a
    connected component's bounding-box rectangle can enclose far more
    tiles than its real members (e.g. a continent-scale component's bbox
    - confirmed on real worldwide data, 2026-10 - can geometrically
    contain thousands of degree-cells against a few hundred real
    members), and reading every one of them is both slow (far more file
    opens than the component actually needs) and a latent correctness
    risk: real floodable land belonging to a DIFFERENT, Phase-1-
    unconnected component could otherwise be swept into this component's
    `keep` mask purely because its bbox happens to enclose it
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
# Phase 3's natural-gap search (src/connectivity_tiling.py::_split_window)
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
