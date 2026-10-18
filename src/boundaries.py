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
from flood_extent import effective_dem
from merge import AQUEDUCT_NODATA, decode_waterdepth_array
from rasters import decode_dem_cm, encode_waterlevel_cm
from tiles import _scan_mask_dir, mosaic_water_fraction_downsampled

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def load_waterlevel_stations(nc_path: str | Path, variable: str, x_var: str, y_var: str, column_name: str) -> gpd.GeoDataFrame:
    """Load water level stations from a COAST-RP_EWL NetCDF file as a GeoDataFrame.

    Parameters
    ----------
    nc_path : str or Path
        Path to the NetCDF file containing one water-level scenario.
    variable : str
        Name of the data variable holding the water level values.
    x_var : str
        Name of the coordinate variable holding station longitudes.
    y_var : str
        Name of the coordinate variable holding station latitudes.
    column_name : str
        Name to give the water level column in the output GeoDataFrame. Must match
        the `waterlevels.name` value written into the tile/scenario's Aqueduct TOML
        configuration.

    Returns
    -------
    geopandas.GeoDataFrame
        Stations with a `column_name` column (plain float64 metres - see
        `save_boundary_points` for the int16-centimetre encoding used on disk,
        matching DEM/friction/waterdepth's convention) and point geometries in
        EPSG:4326. Stations with a NaN water level are dropped, since the Aqueduct
        flood model cannot handle missing boundary values.
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
    """Select water level stations within a tile's bounding box, plus a buffer.

    Also enforces a minimum search-area floor centred on the tile, since a domain's
    own bbox (src/connectivity_tiling.py) is NOT uniformly padded and small chunks
    (post drop_redundant_chunks/split_oversized_chunks) need an explicit floor so a
    shrinking buffer_deg can never silently starve the candidate pool Aqueduct's
    k-nearest-neighbour IDW interpolation draws from. The area actually used is the
    ENVELOPE of "bbox + buffer_deg" and "min_search_size_deg centred on the tile" -
    whichever is larger per dimension.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        Water level stations, as returned by `load_waterlevel_stations`.
    tile : geopandas.GeoDataFrame
        Single-row GeoDataFrame of the tile geometry, as returned by
        `tiles.get_tile_geometry`.
    buffer_deg : float, optional
        Degrees added to each side of the tile's bbox before selecting candidates
        (plain lon/lat expansion in EPSG:4326, not latitude-corrected). Should be
        `boundary_conditions.station_search_buffer_deg`. Default 0.0.
    min_search_size_deg : float, optional
        Minimum width/height of the search area in degrees, centred on the tile's
        own centre. Should be `boundary_conditions.station_search_min_size_deg`.
        Default 0.0.

    Returns
    -------
    geopandas.GeoDataFrame
        The water level stations within the tile's search area.
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

    `select_stations_for_tile`'s bbox buffer is purely distance-based, so a station
    across a land barrier (a thin isthmus, a strait's far shore) still passes it,
    and `flood_model._idw_seed_values`'s own connectivity fix can't catch that
    either since it only ever sees the tile's own (tightly trimmed) mask array.
    This runs a coarser, longer-range version of the same idea: build a
    downsampled ocean mask spanning the tile and every candidate station, label
    its connected components (8-connectivity, same convention as
    `flood_model.coastline_mask`), and keep only stations sharing a component with
    the tile's own coastline. Deliberately conservative: this is a pre-filter, not
    the final say, so ambiguous cases (no confident water in the tile's own bbox,
    or a station's cell below `water_fraction_threshold`) are kept rather than
    dropped.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        Candidate stations, as returned by `select_stations_for_tile`.
    tile : geopandas.GeoDataFrame
        Single-row GeoDataFrame of the tile geometry.
    mask_dir : str or Path
        Directory containing the 1x1 degree DeltaDTM mask GeoTIFF tiles (see
        `tiles.mosaic_water_fraction_downsampled`).
    target_resolution_m : float
        Coarse cell size in metres for the connectivity check
        (`boundary_conditions.connectivity_resolution_m`).
    water_fraction_threshold : float
        Minimum fraction of a coarse cell's underlying native pixels that must be
        ocean for the cell to count as water
        (`boundary_conditions.connectivity_water_fraction_threshold`).

    Returns
    -------
    geopandas.GeoDataFrame
        The subset of `stations` that are ocean-connected to `tile` (or all of
        `stations`, unfiltered, in the inconclusive cases described above).
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
    """Save selected water level stations to a GeoPackage.

    Encodes `column_name` (the water level column - see `load_waterlevel_stations`)
    to int16 centimetres before writing (`rasters.encode_waterlevel_cm`), matching
    the precision convention DEM/friction/waterdepth already use on disk. A caller
    reading this file back for computation (`aqueduct_runner.run_aqueduct_python`)
    must decode it with `rasters.decode_waterlevel_cm` before use.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        Water level stations to save, with `column_name` holding plain float64
        metres.
    output_path : str or Path
        Destination GeoPackage path.
    column_name : str
        Name of the water level column to encode and write.
    """
    stations = stations.copy()
    stations[column_name] = encode_waterlevel_cm(stations[column_name].to_numpy())
    retry_transient_io(stations.to_file, output_path, driver="GPKG")


def collect_neighbor_wave_seeds(
    target_dem_path: str | Path,
    source_paths: list[tuple[str | Path, str | Path, str | Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect direct eikonal seed cells for a hop>=1 tile from already-simulated,
    lower-hop_distance neighbour tile(s).

    For each `(dem_path, mask_path, waterdepth_path)` neighbour (its own DEM + mask
    + this SAME scenario's waterdepth raster), reads a window restricted to the
    target tile's own bounds, keeps only cells the neighbour actually flooded
    (depth not nodata and > 0), computes absolute water level as
    `effective_dem(dem, mask) + depth`, and snaps each cell's centre onto the
    nearest target-grid cell (`rasterio.transform.rowcol`, floor convention).
    Uses `effective_dem` (not the raw DEM) because `flood_depth_dense` already
    computes `depth` as `waterlevel - effective_dem`, so adding the raw DEM would
    double-count terrain elevation on non-land cells (found 2026-08: a river cell
    with dem=29.89m produced a seed of 35.63m from a true ~5.74m water level).
    Unlike real COAST-RP stations, these values are used directly as eikonal
    seeds, not IDW interpolated. Multiple source cells snapping to the same target
    cell are combined with `np.maximum.at` - the maximum wins.

    Parameters
    ----------
    target_dem_path : str or Path
        Path to the hop>=1 tile's own DEM raster (read for its transform/shape/
        bounds only - never for elevation values here).
    source_paths : list of tuple of (str or Path, str or Path, str or Path)
        `(dem_path, mask_path, waterdepth_path)` for each candidate earlier-wave
        neighbour whose output already exists on disk; a not-yet-computed
        neighbour should simply be omitted by the caller.

    Returns
    -------
    tuple of numpy.ndarray
        `(seed_rows, seed_cols, seed_values)` - row/col indices into the target
        tile's own grid and the absolute water level (metres) at each, ready for
        `flood_depth_dense`'s explicit-seed path. Empty arrays if no source
        contributed any usable cell (including empty `source_paths`).

    Raises
    ------
    ValueError
        If a source's mask or waterdepth raster grid does not match its own DEM's.
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
                continue  # no real overlap with the target's own extent
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

        wet_rows, wet_cols = np.nonzero((source_depth != AQUEDUCT_NODATA) & (source_depth > 0))
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
