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

    Args:
        nc_path: Path to the NetCDF file containing one water-level scenario.
        variable: Name of the data variable holding the water level values.
        x_var: Name of the coordinate variable holding station longitudes.
        y_var: Name of the coordinate variable holding station latitudes.
        column_name: Name to give the water level column in the output GeoDataFrame.
            This must match the `waterlevels.name` value written into the
            tile/scenario's Aqueduct TOML configuration.

    Returns:
        A GeoDataFrame with a `column_name` column (plain float64 metres -
        see `save_boundary_points` for where this gets encoded to int16
        centimetres, matching DEM/friction/waterdepth's convention) and
        point geometries in EPSG:4326. Stations with a NaN water level
        value are dropped, since the Aqueduct flood model cannot handle
        missing values in its boundary points.
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
    """Select water level stations within a tile's bounding box (+ buffer),
    with a minimum search-area floor centred on the tile.

    Args:
        stations: GeoDataFrame of water level stations, as returned by
            `load_waterlevel_stations`.
        tile: Single-row GeoDataFrame of the tile geometry, as returned by
            `tiles.get_tile_geometry`.
        buffer_deg: Degrees added to each side of the tile's bbox before
            selecting candidate stations (plain lon/lat expansion in
            EPSG:4326, not latitude-corrected). Should be
            `boundary_conditions.station_search_buffer_deg`. Needed because a
            chunk's own bbox (src/tile_chunking.py's fixed-tile-chunking
            pipeline) is shaved down to its floodable-plus-coastal-buffer
            footprint, which would otherwise shrink the candidate pool
            available to Aqueduct's k-nearest-neighbour IDW interpolation
            (simulation.flooding.knn).
        min_search_size_deg: Minimum width/height (degrees) of the search
            area, applied per-dimension and centred on the TILE's own
            centre (not the buffered box's centre - the same point unless
            the tile itself is off-centre within its own bbox, which it
            never is). Should be `boundary_conditions.station_search_min_
            size_deg` (2026-08). `bbox + buffer_deg` already gives a small
            tile a search box of at least `2 * buffer_deg` per side, but
            that's an accident of the buffer value, not a guarantee - with
            many very small chunks now (post drop_redundant_chunks/
            split_oversized_chunks), the search box needs an explicit,
            independent floor so a shrinking buffer_deg can never
            silently starve a small chunk of candidate stations. The
            actual search box used is the ENVELOPE of "bbox + buffer_deg"
            and "min_search_size_deg centred on the tile" - i.e. whichever
            of the two is larger in each dimension, never smaller than
            either.

    Returns:
        A GeoDataFrame of the water level stations within the tile's
        search area.
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

    `select_stations_for_tile`'s bbox buffer is purely distance-based - a
    station within the buffer but on the far side of a land barrier (a thin
    isthmus, a strait's far shore) still passes it, and the coastal-cell IDW
    seeding's own ocean-connectivity fix (`flood_model._idw_seed_values`)
    can't catch that either, since it only ever sees the tile's own (now
    tightly trimmed) mask array - a barrier lying outside that array is
    invisible to it. This runs a coarser, longer-range version of the same
    idea: build a downsampled (`target_resolution_m`-ish cells, NOT the
    native ~30m grid - intractable at the multi-degree distances a legitimate
    station search can span) ocean mask spanning the tile and every candidate
    station's own location, label its connected components (8-connectivity,
    same convention as `flood_model.coastline_mask`/`prune_to_coast_connected`),
    and keep only stations whose coarse cell shares a component with the
    tile's own coastline.

    Deliberately conservative in the ambiguous cases, since this is a coarse
    long-range pre-filter, not the final say - the native-resolution
    connectivity fix inside `_idw_seed_values` remains the actual authority
    on which specific coastal cells within the tile receive a station's
    influence and by how much; this function only ever removes candidates
    from the pool it draws from, it never assigns anything itself. So: if the
    tile's own bbox shows no confident water at all at this coarse
    resolution, or a station's own coarse cell doesn't reach
    `water_fraction_threshold`, that candidate is kept rather than dropped -
    the ambiguity is resolved in favour of not silently losing a legitimate
    station.

    Args:
        stations: Candidate stations, as returned by `select_stations_for_tile`.
        tile: Single-row GeoDataFrame of the tile geometry.
        mask_dir: Directory containing the 1°×1° DeltaDTM mask GeoTIFF tiles
            (see `tiles.mosaic_water_fraction_downsampled`).
        target_resolution_m: Coarse cell size for the connectivity check
            (`boundary_conditions.connectivity_resolution_m`).
        water_fraction_threshold: Minimum fraction of a coarse cell's
            underlying native pixels that must be ocean for the cell to
            count as water (`boundary_conditions.connectivity_water_fraction_threshold`).

    Returns:
        The subset of `stations` that are ocean-connected to `tile` (or all
        of `stations`, unfiltered, in the inconclusive cases described above).
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

    `column_name` (the water level column - see `load_waterlevel_stations`)
    is encoded to int16 centimetres before writing (2026-08 -
    `rasters.encode_waterlevel_cm`), matching the precision convention
    DEM/friction/waterdepth already use on disk. A caller reading this file
    back for actual computation (aqueduct_runner.run_aqueduct_python) must
    decode it (`rasters.decode_waterlevel_cm`) before use - this function
    only ever writes plain metres in, int16 centimetres out.
    """
    stations = stations.copy()
    stations[column_name] = encode_waterlevel_cm(stations[column_name].to_numpy())
    retry_transient_io(stations.to_file, output_path, driver="GPKG")


def collect_neighbor_wave_seeds(
    target_dem_path: str | Path,
    source_paths: list[tuple[str | Path, str | Path, str | Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect direct eikonal seed cells for a hop>=1 hinterland tile from
    already-simulated, lower-hop_distance neighbour tile(s) (2026-08 - see
    flood_model.flood_depth_dense's seed_rows/seed_cols/seed_values, and
    scripts/run_aqueduct.py, which builds `source_paths` from `domain_
    tiles_global.gpkg`'s hop_distance column).

    For every `(dem_path, mask_path, waterdepth_path)` in `source_paths` (a
    neighbour's own DEM + mask + this SAME scenario's waterdepth raster - a
    neighbour's boundary values only mean anything if they're from the
    matching forcing scenario the target tile is about to run), reads a
    WINDOWED region restricted to the target tile's own bounds (never the
    neighbour's full raster - a real neighbour tile can be up to ~200M
    cells), keeps only cells the neighbour actually flooded (depth not
    nodata and > 0 - a dry/uncomputed cell isn't a meaningful water-level
    observation, same convention `load_waterlevel_stations` uses for NaN),
    computes absolute water level (`effective_dem(dem, mask) + depth`) there,
    and snaps each cell's real-world centre onto the nearest cell of the
    TARGET tile's own grid (`rasterio.transform.rowcol` with `op=math.floor`
    - the standard "which cell contains this point" convention). Unlike real
    COAST-RP stations, these are used directly as eikonal seeds, not IDW
    interpolated - the exact water level is already known at the source.

    Uses `effective_dem` (zeroes DEM on ocean/lake/river cells), NOT the raw
    DEM - `flood_depth_dense` itself computes the saved `depth` as
    `waterlevel - effective_dem`, so for a non-land cell `depth` already IS
    the full water level (effective_dem is 0 there); adding the RAW dem on
    top double-counts that cell's real terrain elevation, inflating the
    reconstructed water level by however high the raw DEM reads at that
    pixel (found 2026-08 - a real river cell with dem=29.89m produced a
    seed value of 35.63m from a true ~5.74m water level, propagating a
    spurious near-total flood into the seeded hop>=1 tile).

    Multiple source tiles (or multiple source cells snapping to the same
    target cell) are combined via `np.maximum.at` in one pass - if they
    disagree, the MAXIMUM wins (matches the shared underlying physical
    reality: at least that much water reached this location per some real
    upstream computation).

    Args:
        target_dem_path: Path to the hop>=1 tile's own DEM raster (read
            for its transform/shape/bounds only - never for elevation
            values here).
        source_paths: `[(dem_path, mask_path, waterdepth_path), ...]` for
            each candidate earlier-wave neighbour whose output already
            exists on disk - a not-yet-computed neighbour should simply be
            omitted by the caller (graceful degradation, same pattern as
            the rest of this session's deferred-DAG-work areas), not
            passed in and expected to fail gracefully here.

    Returns:
        `(seed_rows, seed_cols, seed_values)` - row/col indices into the
        TARGET tile's own grid and the absolute water level (metres) at
        each, ready for `flood_depth_dense`'s explicit-seed path. Empty
        arrays if no source contributed any usable cell (including empty
        `source_paths`).
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
