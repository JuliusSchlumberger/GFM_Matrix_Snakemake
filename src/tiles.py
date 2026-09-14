"""Functions for loading the tile grid and mosaicing DeltaDTM mask/DEM
coverage onto an arbitrary bbox window.
"""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from config_utils import retry_transient_io

_RASTER_OPEN_RETRIES = 4
_RASTER_OPEN_RETRY_DELAY_S = 5.0


def _open_mask_tile(path: Path):
    """`rasterio.open(path)`, retrying briefly on I/O errors before giving up.

    merge_tiles.py's per-tile loops (compute_tile_stats_batch,
    compute_trimmed_geometries) can run for hours over thousands of mask
    tiles on the P:\\ network share - a single transient SMB hiccup surfaces
    as `rasterio.errors.RasterioIOError` ("No such file or directory") on a
    file that both exists and is reachable moments later, and previously
    took the whole multi-hour run down with it. A genuinely missing or
    corrupted file fails the same way every attempt and still raises after
    the retries are exhausted - this only adds a few seconds of delay to
    that case, in exchange for shrugging off a momentary drop in the far
    more common case.
    """
    last_err = None
    for attempt in range(1, _RASTER_OPEN_RETRIES + 1):
        try:
            return rasterio.open(path)
        except rasterio.errors.RasterioIOError as e:
            last_err = e
            if attempt < _RASTER_OPEN_RETRIES:
                print(f"    [retry {attempt}/{_RASTER_OPEN_RETRIES - 1}] "
                      f"failed to open {path}: {e} - retrying in {_RASTER_OPEN_RETRY_DELAY_S:.0f}s",
                      flush=True)
                time.sleep(_RASTER_OPEN_RETRY_DELAY_S)
    raise last_err


def load_tile_grid(tile_grid_path: str | Path) -> gpd.GeoDataFrame:
    """Load the overlapping tile grid produced by `tile_mask_creation.py`."""
    return retry_transient_io(gpd.read_file, tile_grid_path)


def get_tile_geometry(tile_grid: gpd.GeoDataFrame, tile_id: int) -> gpd.GeoDataFrame:
    """Select a single tile from the tile grid by its tile_id.

    Args:
        tile_grid: The full tile grid, as returned by `load_tile_grid`.
        tile_id: Unique identifier of the tile to select.

    Returns:
        A single-row GeoDataFrame containing the selected tile.

    Raises:
        ValueError: If `tile_id` is not present in `tile_grid`.
    """
    tile = tile_grid[tile_grid["tile_id"] == tile_id]
    if tile.empty:
        raise ValueError(f"tile_id {tile_id} not found in tile grid")
    return tile.reset_index(drop=True)


def save_tile_geometry(tile: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save a single tile geometry to a GeoPackage file."""
    retry_transient_io(tile.to_file, output_path, driver="GPKG")


def _coord_str(lat: int, lon: int) -> str:
    """Format a 1°×1° SW-corner (lat, lon) as the {NS}{lat:02d}{EW}{lon:03d} token."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _scan_mask_dir(mask_dir: Path) -> dict[str, Path]:
    """Scan *mask_dir* once and return a {coord_str: path} dict for every .tif found.

    Files are indexed by their embedded {NS}{lat:02d}{EW}{lon:03d} coordinate
    token so any naming prefix or suffix is tolerated.
    """
    import re
    coord_re = re.compile(r"([NS])(\d{2})([EW])(\d{3})")
    index: dict[str, Path] = {}
    for p in mask_dir.glob("*.tif"):
        m = coord_re.search(p.name)
        if m:
            index[m.group(0)] = p
    return index


def _degree_tiles_for_bbox(bbox: tuple[float, float, float, float]):
    """Yield (lat_sw, lon_sw) integer pairs for all 1°×1° cells that overlap *bbox*."""
    import math
    minx, miny, maxx, maxy = bbox
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            yield lat, lon


def _clamp_window(r0: int, c0: int, h: int, w: int, out_height: int, out_width: int) -> tuple[int, int]:
    """Clamp a per-source-tile destination window to fit inside an
    (out_height, out_width) mosaic array.

    `from_bounds(...).round_lengths().round_offsets()` rounds each source
    tile's window independently of the overall output array's own rounding
    (`round((maxx - minx) / px_size)`), so on a bbox whose edges aren't
    aligned to a whole number of pixels (e.g. an arbitrary union of two
    tiles at Stage 3e's merge step), individual pieces can together overshoot
    the whole array by a pixel - confirmed the hard way (ValueError:
    could not broadcast input array from shape (3525,233) into shape
    (3525,232), first hit once Stage 3e started allowing oversized merges
    through to `classify_mosaic` instead of rejecting them outright).
    """
    return min(h, out_height - r0), min(w, out_width - c0)


def _mosaic_mask_for_trim(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
) -> tuple[np.ndarray, Affine] | None:
    """Mosaic all 1°×1° DeltaDTM mask tiles overlapping ``bbox`` into one array.

    Used only by `compute_trimmed_bbox` — kept separate from
    `_compute_fractions_from_tiles` because pre-rounding each source window
    to place it into a shared array shifts pixel content by fractions of a
    pixel relative to the unrounded fractional-window reads
    `_compute_fractions_from_tiles` relies on (differs by ~1e-4 in the
    resulting fractions - small, but not worth introducing into a
    computation that drives real merge/drop decisions). The shave algorithm
    below only needs "is this row/column pure ocean" booleans, so the ~1
    pixel-scale imprecision from rounding is inconsequential, especially
    against the generous buffer_arcsec margin kept beyond the land edge.

    Safe at single-tile scale (a handful of source files, ~180M pixels worst
    case for the largest nominal tile) — mosaicking DeltaDTM's full global
    extent this way would need hundreds of GB and is not attempted here.

    Returns ``(band, transform)`` for the single mask band, or ``None`` if no
    mask file overlaps ``bbox`` at all. Areas inside ``bbox`` not covered by
    any mask file are filled with 255 (not a valid mask value).

    DeltaDTM mask tiles do NOT all share one native resolution: y-resolution
    is a constant 1 arcsec, but x-resolution coarsens at higher latitudes to
    compensate for longitude convergence near the poles (e.g. 3 arcsec at
    76-77N vs 5 arcsec at 80-81N). A bbox spanning such a boundary would
    produce mismatched array shapes if source windows were read at each
    file's own resolution. To handle this, the combined array uses the
    FINEST resolution found among the overlapping files, and every source
    window is read with `out_shape`/nearest-neighbour resampling to exactly
    match its destination slot - safe for this categorical mask (values
    0/1/2/3/255), unlike averaging resamplers.
    """
    minx, miny, maxx, maxy = bbox
    windows = []  # (path, ix0, iy0, ix1, iy1)
    resolutions = []  # (px_size_x, px_size_y) per overlapping file
    for lat, lon in _degree_tiles_for_bbox(bbox):
        coord = _coord_str(lat, lon)
        path = mask_index.get(coord)
        if path is None:
            continue
        ix0 = max(minx, lon)
        iy0 = max(miny, lat)
        ix1 = min(maxx, lon + 1)
        iy1 = min(maxy, lat + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        windows.append((path, ix0, iy0, ix1, iy1))
        with _open_mask_tile(path) as src:
            resolutions.append((abs(src.transform.a), abs(src.transform.e)))

    if not windows:
        return None

    # Finest (smallest) resolution among all overlapping files - avoids
    # downsampling/losing precision from any single source.
    px_size_x = min(r[0] for r in resolutions)
    px_size_y = min(r[1] for r in resolutions)

    transform = Affine.translation(minx, maxy) * Affine.scale(px_size_x, -px_size_y)
    out_width = max(1, round((maxx - minx) / px_size_x))
    out_height = max(1, round((maxy - miny) / px_size_y))
    out = np.full((out_height, out_width), 255, dtype=np.uint8)

    for path, ix0, iy0, ix1, iy1 in windows:
        dst_window = from_bounds(ix0, iy0, ix1, iy1, transform).round_lengths().round_offsets()
        r0, c0 = int(dst_window.row_off), int(dst_window.col_off)
        h, w = int(dst_window.height), int(dst_window.width)
        h, w = _clamp_window(r0, c0, h, w, out_height, out_width)
        if h <= 0 or w <= 0:
            continue
        with _open_mask_tile(path) as src:
            src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            data = retry_transient_io(
                src.read, 1, window=src_window, boundless=True, fill_value=255,
                out_shape=(h, w), resampling=Resampling.nearest,
            )
        out[r0:r0 + h, c0:c0 + w] = data

    return out, transform


def mosaic_water_fraction_downsampled(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
    target_resolution_m: float,
    ocean_code: int = 1,
) -> tuple[np.ndarray, Affine] | None:
    """Mosaic DeltaDTM mask coverage for ``bbox`` and downsample to a coarse
    ocean-fraction grid, for long-range connectivity checks (e.g. "is this
    COAST-RP station ocean-connected to this tile's coastline at all",
    src.boundaries.filter_stations_by_ocean_connectivity) where native ~30m
    resolution is unnecessary and, for a multi-degree bbox, intractable.

    Reuses `_mosaic_mask_for_trim` to assemble the native-resolution mosaic
    (uncovered areas filled with 255, same convention as everywhere else in
    this module), then block-reduces it to ``target_resolution_m``-ish square
    cells, each holding the fraction of its underlying NATIVE pixels that are
    ocean (``mask == ocean_code``) - nodata/land/lake/river pixels all count
    against the fraction, not just land. This is a deliberately conservative
    choice: a coarse cell that's mostly outside DeltaDTM coverage should not
    read as confidently "water" just because the covered remainder happens to
    be ocean.

    NOT block-wise/streaming - materializes the full native-resolution mosaic
    for ``bbox`` before reducing, same tractability envelope as
    `_mosaic_mask_for_trim` itself (documented there as safe up to a single
    large tile's worth of overlapping source files). Fine for the
    tile-plus-nearby-candidate-station windows this is built for; would need
    a genuinely streaming rewrite before pointing it at a much larger area.

    Args:
        bbox: Bounding box to mosaic, as ``(minx, miny, maxx, maxy)``.
        mask_index: ``{coord_str: path}`` from `_scan_mask_dir`.
        target_resolution_m: Approximate coarse cell size in metres (native
            pixel size is converted via a fixed equatorial 111,320 m/degree
            approximation - fine for an aggregation factor, not a precision
            measurement).
        ocean_code: Mask value meaning "ocean" (default 1).

    Returns:
        ``(water_fraction, transform)`` - ``water_fraction`` is a float32
        array in ``[0, 1]``, one cell per coarse pixel - or ``None`` if no
        mask file overlaps ``bbox`` at all.
    """
    mosaic = _mosaic_mask_for_trim(bbox, mask_index)
    if mosaic is None:
        return None
    band, transform = mosaic

    px_deg = abs(transform.a)
    px_m = px_deg * 111_320.0
    factor = max(1, round(target_resolution_m / px_m))

    is_ocean = band == ocean_code
    h, w = is_ocean.shape
    # Pad to a whole multiple of `factor` (bottom/right only) so the reshape
    # below is exact; padded cells count as non-ocean, diluting the coarse
    # cells along the mosaic's own far edge rather than silently dropping them.
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    if pad_h or pad_w:
        is_ocean = np.pad(is_ocean, ((0, pad_h), (0, pad_w)), constant_values=False)

    out_h, out_w = is_ocean.shape[0] // factor, is_ocean.shape[1] // factor
    water_fraction = is_ocean.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3), dtype=np.float32)
    coarse_transform = transform * Affine.scale(factor, factor)
    return water_fraction, coarse_transform


def _mosaic_dem_onto(
    bbox: tuple[float, float, float, float],
    dem_index: dict[str, Path],
    transform: Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Mosaic DeltaDTM ELEVATION tiles overlapping ``bbox`` directly onto an
    already-chosen ``(transform, shape)`` grid - typically a mask mosaic's
    own grid from `_mosaic_mask_for_trim`, so mask and elevation values come
    out pixel-aligned by construction rather than depending on both
    directories happening to offer identical per-file resolutions.

    Same per-1-degree file layout and ``{NS}{lat:02d}{EW}{lon:03d}`` filename
    convention as the mask directory (confirmed empirically: DeltaDTM ships
    elevation and mask tiles under identical filenames in sibling
    directories), so ``dem_index`` is built the same way as ``mask_index``
    (`_scan_mask_dir` on the ``deltadtm`` catalog entry's directory instead
    of ``deltadtm_mask``'s).

    Cells with no overlapping elevation file are filled with -9999.0 -
    DeltaDTM's own native elevation nodata value (see the ``deltadtm``
    catalog entry's ``nodata_value``), distinct from the mask layer's 255
    sentinel and the post-extraction, gap-filled per-tile DEM's 99m
    ``land_fill_value_m``.
    """
    out_height, out_width = shape
    minx, miny, maxx, maxy = bbox
    out = np.full((out_height, out_width), -9999.0, dtype=np.float32)

    for lat, lon in _degree_tiles_for_bbox(bbox):
        coord = _coord_str(lat, lon)
        path = dem_index.get(coord)
        if path is None:
            continue
        ix0 = max(minx, lon)
        iy0 = max(miny, lat)
        ix1 = min(maxx, lon + 1)
        iy1 = min(maxy, lat + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        dst_window = from_bounds(ix0, iy0, ix1, iy1, transform).round_lengths().round_offsets()
        r0, c0 = int(dst_window.row_off), int(dst_window.col_off)
        h, w = int(dst_window.height), int(dst_window.width)
        h, w = _clamp_window(r0, c0, h, w, out_height, out_width)
        if h <= 0 or w <= 0:
            continue
        with _open_mask_tile(path) as src:
            src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            data = retry_transient_io(
                src.read, 1, window=src_window, boundless=True, fill_value=-9999.0,
                out_shape=(h, w), resampling=Resampling.nearest,
            )
        out[r0:r0 + h, c0:c0 + w] = data

    return out


def _min_pool(native: np.ndarray, out_h: int, out_w: int, nodata: float) -> np.ndarray:
    """Reduce `native` (any shape) to `(out_h, out_w)` by taking the MIN of
    each output cell's mapped native pixels, ignoring `nodata` (a `nodata`-
    only output cell stays `nodata`, never spuriously wins the min). Uses a
    direct pixel->output-cell index mapping (`np.minimum.at`) rather than a
    reshape, so it works for any native/output shape ratio, not just exact
    integer factors - used for elevation, where `Resampling.min` (the
    natural choice) is warp-only and unusable on a plain decimated read.
    """
    native_h, native_w = native.shape
    dest_r = np.clip(((np.arange(native_h) + 0.5) / native_h * out_h).astype(np.intp), 0, out_h - 1)
    dest_c = np.clip(((np.arange(native_w) + 0.5) / native_w * out_w).astype(np.intp), 0, out_w - 1)
    DR, DC = np.meshgrid(dest_r, dest_c, indexing="ij")

    valid = native != nodata
    out = np.full((out_h, out_w), np.inf, dtype=np.float64)
    if valid.any():
        np.minimum.at(out, (DR[valid], DC[valid]), native[valid].astype(np.float64))
    out[~np.isfinite(out)] = nodata
    return out.astype(native.dtype)


def mosaic_mask_dem_coarse(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
    dem_index: dict[str, Path] | None,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, Affine] | None:
    """Mosaic mask+elevation for `bbox` DIRECTLY at a coarse target
    resolution - each source tile is read with `out_shape` resampling
    straight onto the coarse output grid, NEVER materializing a native-
    resolution intermediate array.

    Required for PARENT-scale (9deg+, up to ~45deg at extreme latitude
    bands) bboxes, where a native-~1-arcsec mosaic would be gigabytes to
    hundreds of GB - confirmed the hard way (a real 13.5x9deg parent window
    hung/thrashed memory) - unlike `_mosaic_mask_for_trim`/
    `mosaic_water_fraction_downsampled`/`_mosaic_dem_onto`, which mosaic
    natively THEN downsample and are fine at single-child-tile (or
    tile-plus-nearby-station) scale but NOT at parent scale.

    Mask uses nearest-neighbour resampling (categorical - averaging would be
    meaningless). Elevation uses MIN resampling - the conservative choice
    for a "does ANY sub-pixel qualify as low enough to be floodable"
    question: a coarse cell's minimum elevation is the right proxy for "is
    there possibly a floodable pocket in here", where mean/nearest could
    miss one. Both fill uncovered areas the same as their native-resolution
    counterparts (255 for mask, -9999.0 for elevation).

    Returns ``(mask_band, dem_band, transform)`` at the coarse resolution,
    or ``None`` if no mask file overlaps `bbox` at all.
    """
    minx, miny, maxx, maxy = bbox
    px_deg = resolution_m / 111_320.0
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
            # Resampling.min is a warp-only algorithm - not usable on a plain
            # decimated .read() (confirmed: raises ResamplingAlgorithmError).
            # Read this ONE source tile's native window instead (bounded,
            # <=3600x3600 - small) and min-pool it down to (h, w) manually,
            # masking nodata out of the reduction first so an offshore/
            # uncovered native pixel can never masquerade as "very low
            # elevation" and dominate the min.
            with _open_mask_tile(dem_path) as src:
                src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
                native = retry_transient_io(src.read, 1, window=src_window, boundless=True, fill_value=-9999.0)
            dem_out[r0:r0 + h, c0:c0 + w] = _min_pool(native, h, w, nodata=-9999.0)

    if not any_coverage:
        return None
    return mask_out, dem_out, transform


