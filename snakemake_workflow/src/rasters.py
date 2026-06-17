"""Raster extraction and processing functions for a single tile.

Tiles in the overlapping tile grid (see `python/tile_mask_creation.py`) are
exact rectangles, so a tile's bounding box is identical to its geometry.
This means raster datasets can simply be clipped with `bbox=...` - no
additional `geometry_mask` step is required.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import hydromt
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from shapely.geometry import box as shapely_box

from merge import AQUEDUCT_NODATA

_DEM_LAND_FILL = 9999.0  # elevation written for land cells without DEM coverage


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as an xarray DataArray with hydromt raster accessor."""
    import rioxarray  # noqa: F401 - registers the rasterio xarray backend and spatial metadata
    return xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)


def _land_raster(
    data_catalog: hydromt.DataCatalog,
    land_polygons_source: str,
    bbox: list[float],
    dem: xr.DataArray,
) -> np.ndarray:
    """Return a (height, width) bool array: True where land polygons cover a cell.

    Uses gpd.read_file with a bbox filter and the 'land_polygons' layer (the
    GeoPackage also contains a 'marine_buffer' layer which is the default and
    does not cover inland areas).
    """
    path = data_catalog.get_source(land_polygons_source).path
    land_gdf = gpd.read_file(path, layer="land_polygons", bbox=tuple(bbox))
    height, width = dem.raster.height, dem.raster.width
    if land_gdf.empty:
        return np.zeros((height, width), dtype=bool)
    # invert=True → True inside land polygons
    return geometry_mask(
        land_gdf.geometry,
        out_shape=(height, width),
        transform=dem.raster.transform,
        invert=True,
    )


def get_tile_bbox(tile: gpd.GeoDataFrame) -> list[float]:
    """Return the [minx, miny, maxx, maxy] bounding box of a single-row tile GeoDataFrame."""
    return tile.total_bounds.tolist()


def compute_model_bbox(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    tile_bbox: list[float],
    buffer_arcsec: float = 1.0,
) -> list[float]:
    """Compute the model domain bbox as the DEM's valid-cell extent plus a buffer.

    The full tile bbox is much larger than needed: most of the tile is inland
    land without DeltaDTM coverage.  This function shrinks the domain to the
    bounding box of valid DeltaDTM cells expanded by ``buffer_arcsec`` on all
    four sides, then clips the result to ``tile_bbox``.

    DeltaDTM covers right to the shoreline, so a buffer of even 1 arc-second
    (~30 m) is enough to include a thin strip of ocean cells on the seaward
    edge, giving the Aqueduct model the coastal entry points it needs for flood
    propagation.  Increase ``buffer_arcsec`` (via ``simulation.model_bbox_buffer_arcsec``
    in config) if coastal tiles show incomplete flood entry.

    If the DEM has no valid cells within ``tile_bbox``, the full ``tile_bbox``
    is returned unchanged.

    Args:
        data_catalog: HydroMT data catalog.
        dem_source: Catalog key for the DEM (DeltaDTM) RasterDataset.
        tile_bbox: Full tile bounding box as ``[minx, miny, maxx, maxy]``.
        buffer_arcsec: Buffer added to each side of the DEM extent, in
            arc-seconds.

    Returns:
        Reduced bounding box ``[minx, miny, maxx, maxy]``, clipped to ``tile_bbox``.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=tile_bbox)
    valid = (da.values != da.raster.nodata).squeeze()

    if not valid.any():
        return list(tile_bbox)

    y_coords = da.y.values  # descending (north first)
    x_coords = da.x.values  # ascending
    dy = float(abs(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 0.0
    dx = float(abs(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 0.0
    rows, cols = np.where(valid)

    buf = buffer_arcsec / 3600.0
    minx = float(x_coords[cols.min()]) - dx / 2 - buf
    miny = float(y_coords[rows.max()]) - dy / 2 - buf   # rows.max() = southernmost
    maxx = float(x_coords[cols.max()]) + dx / 2 + buf
    maxy = float(y_coords[rows.min()]) + dy / 2 + buf   # rows.min() = northernmost

    return [
        max(minx, tile_bbox[0]),
        max(miny, tile_bbox[1]),
        min(maxx, tile_bbox[2]),
        min(maxy, tile_bbox[3]),
    ]


def extract_dem(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    bbox: list[float],
    land_polygons_source: str,
) -> xr.DataArray:
    """Clip the DEM to the model domain bbox and fill all missing cells.

    Every cell in the output has a real value — no nodata cells remain:
    - Cells with valid DeltaDTM elevation: kept as-is.
    - Missing cells inside the land polygon: set to `_DEM_LAND_FILL` (9999 m)
      so Julia reads them as real, very high elevation and never floods them.
    - Missing cells outside the land polygon (ocean): set to 0.0 (Julia
      overwrites these anyway via ``dem[.!landmask] .= 0.0``).

    Args:
        data_catalog: HydroMT data catalog containing `dem_source`.
        dem_source: Name of the DEM RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        land_polygons_source: Name of the land-polygon GeoDataFrame in
            `data_catalog` used to distinguish missing-but-land cells from
            missing-ocean cells.

    Returns:
        The DEM clipped to `bbox` with all missing cells filled.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=bbox)
    is_land = _land_raster(data_catalog, land_polygons_source, bbox, da)
    fill = np.where(is_land, _DEM_LAND_FILL, 0.0).reshape(da.shape).astype(np.float32)
    return da.where(da != da.raster.nodata, da.copy(data=fill))


def extract_dem_mask(
    data_catalog: hydromt.DataCatalog,
    mask_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    nodata_sentinel: int = 255,
    land_polygons_source: str | None = None,
) -> xr.DataArray:
    """Clip the DEM-validity mask and reproject it onto the DEM's grid.

    The mask source raster has a different native grid/resolution than the
    DEM, so it is reprojected (nearest-neighbour) onto the DEM's grid to
    ensure matching dimensions, as required by the flood model.

    Args:
        data_catalog: HydroMT data catalog containing `mask_source`.
        mask_source: Name of the DEM-mask RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        dem: The tile's DEM, as returned by `extract_dem`, used as the
            reprojection target grid.
        nodata_sentinel: Value used by `mask_source` to indicate no data.
        land_polygons_source: Name of the land-polygon GeoDataFrame in
            `data_catalog`. Cells where `mask_source` has no data
            (value == `nodata_sentinel`) are filled using land polygon coverage:
            inside the polygon → ``0`` (land), outside → ``1`` (ocean).
            Valid DeltaDTM values (0 = land, 1 = ocean, 2 = lake, 3 = river)
            are kept unchanged, preserving river and lake channels that drive
            inland flood propagation.

    Returns:
        The DEM-validity mask reprojected onto `dem`'s grid with no nodata cells.
    """
    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(nodata_sentinel)
    da_mask_repr = da_mask.raster.reproject_like(dem, method="nearest")
    if land_polygons_source is not None:
        is_land = _land_raster(data_catalog, land_polygons_source, bbox, dem)
        arr = da_mask_repr.values.squeeze().copy()
        is_land_2d = is_land.reshape(arr.shape)
        # Cells with generic land/ocean/nodata classification: use land polygon as
        # authoritative source. Lake (2) and river (3) cells are kept unchanged since
        # they represent water channels that propagate flooding inland.
        generic = (arr == 0) | (arr == 1) | (arr == nodata_sentinel)
        arr[generic & is_land_2d] = 0
        arr[generic & ~is_land_2d] = 1
        return da_mask_repr.copy(data=arr.reshape(da_mask_repr.shape))
    return da_mask_repr.where(da_mask_repr != nodata_sentinel, 0)


def compute_friction(
    data_catalog: hydromt.DataCatalog,
    lulc_source: str,
    lookup_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    default_friction: float,
) -> xr.DataArray:
    """Clip land use, reproject it onto the DEM's grid, and convert it to friction.

    Land use classes are reclassified to Manning's n roughness coefficients
    using the `lookup_source` lookup table, then converted to a friction
    value via `roughness / 100` (empirical formula). Cells with no valid
    land use classification are filled with `default_friction`.

    Args:
        data_catalog: HydroMT data catalog containing `lulc_source` and `lookup_source`.
        lulc_source: Name of the land use RasterDataset in `data_catalog`.
        lookup_source: Name of the land-use-to-roughness lookup DataFrame in
            `data_catalog`. Must have an index named `copernicus_worldcover`
            and a `manning_n` column.
        bbox: Tile bounding box as `[minx, miny, maxx, maxy]`.
        dem: The tile's DEM, as returned by `extract_dem`, used as the
            reprojection target grid.
        default_friction: Friction value written for cells with no valid land
            use classification (e.g. ocean, outside LULC coverage). Should
            match the Aqueduct core default (flooding.default_friction in config).

    Returns:
        A friction raster on `dem`'s grid with no nodata cells.
    """
    _MISSING = np.float32(-9999.0)  # internal sentinel for unclassified cells

    da_lulc = data_catalog.get_rasterdataset(lulc_source, bbox=bbox)
    da_lulc_repr = da_lulc.raster.reproject_like(dem, method="mode")
    da_lulc_repr = da_lulc_repr.where(da_lulc_repr != da_lulc_repr.attrs["_FillValue"], _MISSING)

    df_map = data_catalog.get_dataframe(lookup_source)
    df_map = df_map.set_index("copernicus_worldcover")
    df_map = df_map.rename(columns={"manning_n": "N"})
    df_map["N"] = df_map["N"].replace(-999.000, _MISSING)

    max_code = int(df_map.index.max())
    lookup = np.full(max_code + 1, _MISSING, dtype="float32")
    lookup[df_map.index.to_numpy(dtype=int)] = df_map["N"].to_numpy(dtype="float32")

    lulc = da_lulc_repr.values
    valid = np.isfinite(lulc) & (lulc >= 0) & (lulc <= max_code)
    roughness = np.full(lulc.shape, _MISSING, dtype="float32")
    roughness[valid] = lookup[lulc[valid].astype(int)]

    friction = np.where(roughness != _MISSING, roughness / 100, default_friction).astype(np.float32)
    return da_lulc_repr.copy(data=friction)


def save_raster(
    da: xr.DataArray,
    output_path: str | Path,
    raster_config: dict[str, Any],
    dtype: str = "float32",
) -> None:
    """Save a DataArray as a raster file using the workflow's raster output settings.

    Args:
        da: The data array to save. Must have `da.raster.crs`, `da.raster.transform`,
            `da.raster.width` and `da.raster.height` accessors (provided by rioxarray
            via HydroMT's raster accessor).
        output_path: Destination file path.
        raster_config: The workflow's `raster` configuration section, with keys
            `driver`, `compression`, `predictor` and `nodata`.
        dtype: Output data type.
    """
    with rasterio.open(
        output_path,
        "w",
        driver=raster_config["driver"],
        crs=da.raster.crs,
        transform=da.raster.transform,
        dtype=dtype,
        count=1,
        nodata=raster_config["nodata"],
        compress=raster_config["compression"],
        predictor=raster_config["predictor"],
        width=da.raster.width,
        height=da.raster.height,
    ) as dst:
        dst.write(da.values, indexes=1)


def save_nodata_raster(reference_path: str | Path, output_path: str | Path, raster_config: dict[str, Any]) -> None:
    """Write an all-nodata waterdepth raster on the same grid as `reference_path`.

    Used as a placeholder `waterdepth` output for tiles that are skipped
    instead of being run through Aqueduct (see
    `aqueduct_runner.log_skipped_tile`). Filled with `merge.AQUEDUCT_NODATA`,
    the same sentinel the Aqueduct model itself writes for cells outside the
    area it computed, so `merge.merge_tile_rasters` ignores this tile
    entirely when merging results.

    Args:
        reference_path: Path to a raster (e.g. the tile's DEM) whose grid
            (transform, CRS, width, height) the output should match.
        output_path: Destination file path.
        raster_config: The workflow's `raster` configuration section, with
            keys `driver`, `compression` and `predictor`.
    """
    with rasterio.open(reference_path) as ref:
        profile = ref.profile

    profile.update(
        driver=raster_config["driver"],
        dtype="float32",
        count=1,
        compress=raster_config["compression"],
        predictor=raster_config["predictor"],
        nodata=AQUEDUCT_NODATA,
    )
    data = np.full((profile["height"], profile["width"]), AQUEDUCT_NODATA, dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, indexes=1)
