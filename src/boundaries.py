"""Functions for extracting water-level boundary points for a tile and scenario."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import xarray as xr
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import box

from config_utils import retry_transient_io
from flood_model import effective_dem
from merge import EIKONAL_NODATA, decode_waterdepth_array
from rasters import decode_dem_cm, encode_waterlevel_cm
from tiles import _scan_mask_dir, mosaic_water_fraction_downsampled

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def load_waterlevel_stations(nc_path: str | Path, variable: str, x_var: str, y_var: str, column_name: str) -> gpd.GeoDataFrame:
    """Load COAST-RP_EWL water level stations from NetCDF as a GeoDataFrame.

    `column_name` must match the Aqueduct TOML `waterlevels.name`. Water levels are
    float64 metres, geometries EPSG:4326. NaN stations are dropped (the eikonal
    model cannot handle missing boundary values).
    """
    with retry_transient_io(xr.open_dataset, nc_path) as ds:
        waterlevel = ds[variable].values.astype(np.float64)
        lon = ds[x_var].values
        lat = ds[y_var].values

    stations = gpd.GeoDataFrame(
        {column_name: waterlevel},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    return stations[stations[column_name].notna()].reset_index(drop=True)


def select_stations_for_tile(
    stations: gpd.GeoDataFrame,
    tile: gpd.GeoDataFrame,
    buffer_deg: float = 0.0,
    min_search_size_deg: float = 0.0,
) -> gpd.GeoDataFrame:
    """Select stations within the tile bbox plus `buffer_deg` (or a minimum search box).

    The search area is the envelope of "bbox + buffer_deg" (plain lon/lat, not
    latitude-corrected) and a `min_search_size_deg` square centred on the tile. The
    floor exists because domain bboxes are not uniformly padded, so small chunks
    could otherwise starve the solver's k-nearest-neighbour IDW candidate pool.
    """
    minx, miny, maxx, maxy = tile.total_bounds
    buf_minx, buf_miny, buf_maxx, buf_maxy = minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg

    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = min_search_size_deg / 2.0
    min_minx, min_miny, min_maxx, min_maxy = cx - half, cy - half, cx + half, cy + half

    search_area = box(
        min(buf_minx, min_minx), min(buf_miny, min_miny),
        max(buf_maxx, min_maxx), max(buf_maxy, min_maxy),
    )
    return stations[stations.intersects(search_area)].reset_index(drop=True)


def filter_stations_by_ocean_connectivity(
    stations: gpd.GeoDataFrame,
    tile: gpd.GeoDataFrame,
    mask_dir: str | Path,
    target_resolution_m: float,
    water_fraction_threshold: float,
) -> gpd.GeoDataFrame:
    """Drop candidate stations that aren't ocean-connected to this tile.

    The bbox buffer is distance-only, so stations across a land barrier pass it.
    Labels a downsampled ocean mask (8-connectivity, as `flood_model.coastline_mask`)
    over tile + stations and keeps stations sharing a component with the tile.
    Conservative pre-filter: inconclusive cases (no confident water in the tile
    bbox, or a station cell below `water_fraction_threshold`, the minimum ocean
    fraction of a coarse cell) are kept.
    """
    if stations.empty:
        return stations

    tile_minx, tile_miny, tile_maxx, tile_maxy = tile.total_bounds
    station_xs = stations.geometry.x.to_numpy()
    station_ys = stations.geometry.y.to_numpy()
    envelope = (
        min(tile_minx, float(station_xs.min())),
        min(tile_miny, float(station_ys.min())),
        max(tile_maxx, float(station_xs.max())),
        max(tile_maxy, float(station_ys.max())),
    )

    mask_index = _scan_mask_dir(Path(mask_dir))
    result = mosaic_water_fraction_downsampled(envelope, mask_index, target_resolution_m)
    if result is None:
        return stations  # no DeltaDTM coverage anywhere in the envelope - can't determine connectivity
    water_fraction, transform = result
    is_water = water_fraction >= water_fraction_threshold
    labels, _n = ndimage.label(is_water, structure=_STRUCTURE_8)

    inv_transform = ~transform

    def _rowcol(x: float, y: float) -> tuple[int, int]:
        col, row = inv_transform * (x, y)
        row = min(max(int(row), 0), labels.shape[0] - 1)
        col = min(max(int(col), 0), labels.shape[1] - 1)
        return row, col

    tr0, tc0 = _rowcol(tile_minx, tile_maxy)
    tr1, tc1 = _rowcol(tile_maxx, tile_miny)
    r0, r1 = sorted((tr0, tr1))
    c0, c1 = sorted((tc0, tc1))
    tile_window = labels[r0:r1 + 1, c0:c1 + 1]
    tile_water_window = is_water[r0:r1 + 1, c0:c1 + 1]
    tile_components = set(np.unique(tile_window[tile_water_window]).tolist()) - {0}

    if not tile_components:
        return stations  # no confident water found within the tile's own bbox at this resolution

    keep = np.ones(len(stations), dtype=bool)
    for i in range(len(stations)):
        row, col = _rowcol(float(station_xs[i]), float(station_ys[i]))
        if not is_water[row, col]:
            continue  # station's own coarse cell isn't confidently water - inconclusive, keep
        if labels[row, col] not in tile_components:
            keep[i] = False

    n_dropped = int((~keep).sum())
    if n_dropped:
        print(
            f"  ocean-connectivity filter: dropped {n_dropped}/{len(stations)} candidate "
            f"station(s) not ocean-connected to this tile at ~{target_resolution_m:.0f}m resolution",
            flush=True,
        )
    return stations[keep].reset_index(drop=True)


def save_boundary_points(stations: gpd.GeoDataFrame, output_path: str | Path, column_name: str) -> None:
    """Save stations to a GeoPackage with water levels encoded as int16 centimetres.

    Input `column_name` is float64 metres; readers must decode with
    `rasters.decode_waterlevel_cm`.
    """
    stations = stations.copy()
    stations[column_name] = encode_waterlevel_cm(stations[column_name].to_numpy())
    retry_transient_io(stations.to_file, output_path, driver="GPKG")


def collect_neighbor_wave_seeds(
    target_dem_path: str | Path,
    source_paths: list[tuple[str | Path, str | Path, str | Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect direct eikonal seeds for a hop>=1 tile from lower-hop neighbour outputs.

    `source_paths` holds `(dem, mask, waterdepth)` per already-simulated neighbour
    (same scenario). Wet cells (depth > 0, not nodata) give absolute water level
    `effective_dem(dem, mask) + depth` in metres; the raw DEM would double-count
    terrain on non-land cells, since depth is `waterlevel - effective_dem`. Cell
    centres snap to the target grid with floor; collisions keep the maximum. Seeds
    are used directly, not IDW-interpolated. Returns `(rows, cols, values)`, empty if
    nothing contributed. Raises ValueError if a source's mask/waterdepth grid differs
    from its DEM.
    """
    with retry_transient_io(rasterio.open, target_dem_path) as target_src:
        target_transform = target_src.transform
        target_shape = (target_src.height, target_src.width)
        target_bounds = target_src.bounds

    seed_grid = np.full(target_shape, -np.inf, dtype=np.float32)

    for source_dem_path, source_mask_path, source_waterdepth_path in source_paths:
        with retry_transient_io(rasterio.open, source_dem_path) as dem_src:
            source_transform = dem_src.transform
            raw_window = from_bounds(*target_bounds, transform=source_transform).round_offsets().round_lengths()
            col_off = max(0, raw_window.col_off)
            row_off = max(0, raw_window.row_off)
            col_end = min(dem_src.width, raw_window.col_off + raw_window.width)
            row_end = min(dem_src.height, raw_window.row_off + raw_window.height)
            if col_end <= col_off or row_end <= row_off:
                continue
            window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
            source_dem = decode_dem_cm(dem_src.read(1, window=window))
            source_win_transform = dem_src.window_transform(window)
            source_shape = (dem_src.height, dem_src.width)

        with retry_transient_io(rasterio.open, source_mask_path) as mask_src:
            if (mask_src.height, mask_src.width) != source_shape:
                raise ValueError(
                    f"mask raster grid does not match its DEM's ({source_mask_path} vs {source_dem_path})"
                )
            source_mask = mask_src.read(1, window=window)
        source_effective_dem = effective_dem(source_dem, source_mask)

        with retry_transient_io(rasterio.open, source_waterdepth_path) as wd_src:
            if wd_src.transform != source_transform or (wd_src.height, wd_src.width) != source_shape:
                raise ValueError(
                    f"waterdepth raster grid does not match its DEM's ({source_waterdepth_path} vs "
                    f"{source_dem_path}) - rasters.save_waterdepth_raster's own contract requires them identical"
                )
            source_depth = decode_waterdepth_array(wd_src.read(1, window=window))

        wet_rows, wet_cols = np.nonzero((source_depth != EIKONAL_NODATA) & (source_depth > 0))
        if len(wet_rows) == 0:
            continue

        xs, ys = rasterio.transform.xy(source_win_transform, wet_rows, wet_cols)
        water_levels = (source_effective_dem[wet_rows, wet_cols] + source_depth[wet_rows, wet_cols]).astype(np.float32)

        t_rows, t_cols = rasterio.transform.rowcol(target_transform, xs, ys, op=math.floor)
        t_rows, t_cols = np.asarray(t_rows), np.asarray(t_cols)
        in_bounds = (t_rows >= 0) & (t_rows < target_shape[0]) & (t_cols >= 0) & (t_cols < target_shape[1])
        if not in_bounds.any():
            continue
        np.maximum.at(seed_grid, (t_rows[in_bounds], t_cols[in_bounds]), water_levels[in_bounds])

    seed_rows, seed_cols = np.nonzero(seed_grid > -np.inf)
    seed_values = seed_grid[seed_rows, seed_cols].astype(np.float64)
    return seed_rows, seed_cols, seed_values
