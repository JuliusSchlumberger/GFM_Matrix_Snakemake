"""Functions for extracting water-level boundary points for a tile and scenario."""

from pathlib import Path

import geopandas as gpd
import xarray as xr
from shapely.geometry import box


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
        A GeoDataFrame with a `column_name` column and point geometries in EPSG:4326.
        Stations with a NaN water level value are dropped, since the Aqueduct
        flood model cannot handle missing values in its boundary points.
    """
    with xr.open_dataset(nc_path) as ds:
        waterlevel = ds[variable].values
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
) -> gpd.GeoDataFrame:
    """Select water level stations within a tile's bounding box (+ buffer).

    Args:
        stations: GeoDataFrame of water level stations, as returned by
            `load_waterlevel_stations`.
        tile: Single-row GeoDataFrame of the tile geometry, as returned by
            `tiles.get_tile_geometry`.
        buffer_deg: Degrees added to each side of the tile's bbox before
            selecting candidate stations (plain lon/lat expansion in
            EPSG:4326, not latitude-corrected - same convention already used
            by tile_mask_creation.py's `filter_grid_by_coastrp` buffer_deg).
            Should be `boundary_conditions.station_search_buffer_deg`. Needed
            because tile_grid.trim_buffer_arcsec tightens tile bboxes to
            their coastal content, which would otherwise shrink the
            candidate pool available to Aqueduct's k-nearest-neighbour IDW
            interpolation (simulation.flooding.knn).

    Returns:
        A GeoDataFrame of the water level stations within the tile's
        buffered bounds.
    """
    minx, miny, maxx, maxy = tile.total_bounds
    search_area = box(minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)
    return stations[stations.intersects(search_area)].reset_index(drop=True)


def save_boundary_points(stations: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save selected water level stations to a GeoPackage."""
    stations.to_file(output_path, driver="GPKG")
