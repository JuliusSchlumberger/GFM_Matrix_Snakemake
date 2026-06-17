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


def select_stations_for_tile(stations: gpd.GeoDataFrame, tile: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select water level stations within a tile's bounding box.

    Args:
        stations: GeoDataFrame of water level stations, as returned by
            `load_waterlevel_stations`.
        tile: Single-row GeoDataFrame of the tile geometry, as returned by
            `tiles.get_tile_geometry`.

    Returns:
        A GeoDataFrame of the water level stations within the tile's bounds.
    """
    search_area = box(*tile.total_bounds)
    return stations[stations.intersects(search_area)].reset_index(drop=True)


def save_boundary_points(stations: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save selected water level stations to a GeoPackage."""
    stations.to_file(output_path, driver="GPKG")
