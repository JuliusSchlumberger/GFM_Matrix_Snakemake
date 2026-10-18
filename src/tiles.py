"""Load the tile grid and mosaic DeltaDTM mask/DEM coverage onto a bbox window."""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box as shapely_box

from config_utils import retry_transient_io

_RASTER_OPEN_RETRIES = 4
_RASTER_OPEN_RETRY_DELAY_S = 5.0


def _open_mask_tile(path: Path, **open_kwargs):
    # Retries transient P:\ SMB blips (RasterioIOError on files that exist).
    last_err = None
    for attempt in range(1, _RASTER_OPEN_RETRIES + 1):
        try:
            return rasterio.open(path, **open_kwargs)
        except rasterio.errors.RasterioIOError as e:
            last_err = e
            if attempt < _RASTER_OPEN_RETRIES:
                print(f"    [retry {attempt}/{_RASTER_OPEN_RETRIES - 1}] "
                      f"failed to open {path}: {e} - retrying in {_RASTER_OPEN_RETRY_DELAY_S:.0f}s",
                      flush=True)
                time.sleep(_RASTER_OPEN_RETRY_DELAY_S)
    raise last_err


def load_tile_grid(tile_grid_path: str | Path) -> gpd.GeoDataFrame:
    """Load the tile grid (``tile_grid.path``, built by preparation/build_tile_manifest.py)."""
    return retry_transient_io(gpd.read_file, tile_grid_path)


def get_tile_geometry(tile_grid: gpd.GeoDataFrame, tile_id: int) -> gpd.GeoDataFrame:
    """Select a single tile by tile_id; raise ValueError if absent."""
    tile = tile_grid[tile_grid["tile_id"] == tile_id]
    if tile.empty:
        raise ValueError(f"tile_id {tile_id} not found in tile grid")
    return tile.reset_index(drop=True)


def save_tile_geometry(tile: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save a single tile geometry to a GeoPackage file."""
    retry_transient_io(tile.to_file, output_path, driver="GPKG")


def chunk_id_from_corner(lon_deg: int, lat_deg: int) -> str:
    """Format a chunk ID from its SW corner in whole degrees, e.g. "S10E030"."""
    lat = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{-lat_deg:02d}"
    lon = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{-lon_deg:03d}"
    return f"{lat}{lon}"


def build_chunk_grid(tile_gdf: gpd.GeoDataFrame, chunk_size_deg: float) -> gpd.GeoDataFrame:
    """Build the postprocessing chunk grid covering the tile grid.

    Single definition shared by the Snakefile, HPC job generator and progress
    monitor. Cells are degree squares aligned to multiples of `chunk_size_deg`,
    kept only if they intersect a tile; ordered by x, then y.
    """
    minx, miny, maxx, maxy = tile_gdf.total_bounds
    sz = chunk_size_deg
    xs = np.arange(np.floor(minx / sz) * sz, np.ceil(maxx / sz) * sz, sz)
    ys = np.arange(np.floor(miny / sz) * sz, np.ceil(maxy / sz) * sz, sz)
    rows = []
    for x in xs:
        for y in ys:
            cell = shapely_box(x, y, x + sz, y + sz)
            if not tile_gdf.geometry.intersects(cell).any():
                continue
            rows.append({
                "chunk_id": chunk_id_from_corner(int(round(x)), int(round(y))),
                "bounds": [x, y, x + sz, y + sz],
                "geometry": cell,
            })
    return gpd.GeoDataFrame(rows, crs=tile_gdf.crs)


def _coord_str(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _scan_mask_dir(mask_dir: Path) -> dict[str, Path]:
    import re
    coord_re = re.compile(r"([NS])(\d{2})([EW])(\d{3})")
    index: dict[str, Path] = {}
    for p in mask_dir.glob("*.tif"):
        m = coord_re.search(p.name)
        if m:
            index[m.group(0)] = p
    return index


def _degree_tiles_for_bbox(bbox: tuple[float, float, float, float]):
    import math
    minx, miny, maxx, maxy = bbox
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            yield lat, lon


def _clamp_window(r0: int, c0: int, h: int, w: int, out_height: int, out_width: int) -> tuple[int, int]:
    # Per-window rounding can overshoot the array's own rounding by a pixel on
    # bboxes not aligned to whole pixels.
    return min(h, out_height - r0), min(w, out_width - c0)


def _mosaic_mask_for_trim(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
) -> tuple[np.ndarray, Affine] | None:
    # Native-resolution mosaic; safe at single-tile scale (~180M px worst case),
    # not globally. Uncovered = 255. Source windows are pre-rounded, so content
    # can shift by sub-pixel amounts. DeltaDTM y-res is 1 arcsec but x-res
    # coarsens with latitude (3 arcsec at 76-77N, 5 at 80-81N), so the finest
    # resolution is used, nearest-resampled (mask is categorical 0/1/2/3/255).
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
    """Mosaic DeltaDTM mask for ``bbox`` and downsample to a coarse ocean fraction.

    For long-range station connectivity checks. Each coarse cell holds the
    fraction [0, 1] of native pixels equal to ``ocean_code``; nodata, land,
    lake and river all count against it (conservative). Metres->degrees uses
    111,320 m/deg. Materializes the native mosaic first, so only tractable at
    single-tile scale. Returns None if no mask file overlaps.
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
    # Padded cells count as non-ocean, diluting far-edge cells instead of dropping them.
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
    # Onto a given (mask-mosaic) grid so mask and DEM are pixel-aligned.
    # DeltaDTM DEM tiles share the mask's filenames. Uncovered = -9999.0,
    # DeltaDTM's native nodata (not the per-tile DEM's 99 m land fill).
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
    # Min-pool ignoring nodata, for any shape ratio (Resampling.min is warp-only).
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
    """Mosaic mask + elevation for `bbox` directly at a coarse resolution.

    Never materializes a native array, so usable at parent scale (9-45 deg),
    where native ~1 arcsec mosaics need GB-to-hundreds-of-GB. Mask: nearest
    (categorical). Elevation: min (conservative proxy for a floodable pocket).
    Uncovered: 255 / -9999.0. Metres->degrees uses 111,320 m/deg.
    ``dem_index=None`` leaves the DEM all-nodata. Returns None if no mask overlaps.
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
            # Resampling.min raises on a decimated read; min-pool one native
            # window (<=3600x3600) instead, excluding nodata from the min.
            with _open_mask_tile(dem_path) as src:
                src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
                native = retry_transient_io(src.read, 1, window=src_window, boundless=True, fill_value=-9999.0)
            dem_out[r0:r0 + h, c0:c0 + w] = _min_pool(native, h, w, nodata=-9999.0)

    if not any_coverage:
        return None
    return mask_out, dem_out, transform


