"""Raster extraction and processing functions for a single tile.

Tiles in the overlapping tile grid (see `tile_mask_creation.py`) are
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
from rasterio.fill import fillnodata
from scipy import ndimage
from shapely.geometry import box as shapely_box

from config_utils import retry_transient_io

from merge import AQUEDUCT_NODATA


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as an xarray DataArray with hydromt raster accessor."""
    import rioxarray  # noqa: F401 - registers the rasterio xarray backend and spatial metadata
    return retry_transient_io(xr.open_dataarray, path, engine="rasterio").squeeze("band", drop=True)


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

    # Final clamp to the valid global coordinate range, independent of
    # tile_bbox: a tile whose own bbox already overshoots -180/180 (e.g. an
    # antimeridian-adjacent tile, however slightly, from upstream trim/
    # buffer arithmetic in tiles.py) would otherwise pass that overshoot
    # straight through - `max(minx, tile_bbox[0])` doesn't help when
    # tile_bbox[0] is ITSELF already invalid. A bbox even a fraction of a
    # degree past +/-180 can make downstream hydromt/rioxarray reprojection
    # windows wrap around almost the entire globe instead of clipping
    # cleanly, blowing up into a hundreds-of-GB allocation for what should
    # be a single small tile (see extract_dem's da_mask.raster.reproject_like).
    return [
        max(max(minx, tile_bbox[0]), -180.0),
        max(max(miny, tile_bbox[1]), -90.0),
        min(min(maxx, tile_bbox[2]), 180.0),
        min(min(maxy, tile_bbox[3]), 90.0),
    ]


def extract_dem(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    bbox: list[float],
    mask_source: str,
    geoid_offset_raster: str | Path | None = None,
    min_hard_fill_component_size: int = 10,
    interp_max_search_distance: float = 100.0,
    interp_smoothing_iterations: int = 0,
    land_fill_value_m: float = 9999.0,
) -> xr.DataArray:
    """Clip the DEM to the model domain bbox and fill all missing cells.

    Every cell in the output has a real value — no nodata cells remain. The
    fill value is decided entirely from `mask_source` (the DeltaDTM validity
    mask), reprojected onto the DEM's own grid — not from any separately
    sourced land polygon dataset, which can misalign with DeltaDTM's own
    coastline:
    - Cells with valid DeltaDTM elevation: kept as-is (optionally geoid-
      corrected first, see `geoid_offset_raster` below).
    - Missing cells where the mask says land (0), OR where the mask itself
      has no coverage at all (nodata): split by the size of the connected
      (4-connectivity) group of missing-land cells they belong to.
        - SMALL group (< `min_hard_fill_component_size` cells, e.g. an
          isolated pixel or a small cluster): treated as a plausible
          measurement/sensor gap and INTERPOLATED from the surrounding
          valid DeltaDTM elevation (`rasterio.fill.fillnodata`, inverse-
          distance weighted), rather than hard-filled - a lone nodata pixel
          surrounded by real elevation is very unlikely to be exactly
          9999 m in reality.
        - LARGE group (>= `min_hard_fill_component_size` cells): set to
          `land_fill_value_m` (9999 m) so Julia reads it as real, very high
          elevation and never floods it - genuinely unmeasured terrain (the
          common case for land nodata), not a small gap worth interpolating
          over.
      Areas outside the mask's own coverage are treated the same as known
      land throughout this split - irrelevant, definitely-dry terrain if
      the gap is large, an interpolation candidate if it's small.
    - Missing cells where the mask says ocean (1), lake (2), or river (3):
      set to 0.0, regardless of gap size. Julia re-zeros ocean cells anyway
      (``dem[.!landmask] .= 0.0``); lake/river cells are permanent inland
      water bodies where DeltaDTM has no terrain elevation, and filling with
      0 lets the flood model propagate through them rather than blocking it
      with an artificially high land elevation.

    Args:
        data_catalog: HydroMT data catalog containing `dem_source` and `mask_source`.
        dem_source: Name of the DEM RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        mask_source: Name of the DeltaDTM validity mask RasterDataset in
            `data_catalog`, used to decide the fill value for missing DEM cells.
        geoid_offset_raster: Path to the cached global EGM2008 -> GOCO06s
            geoid-offset raster (see vertical_datum.write_geoid_offset_raster
            / rule compute_geoid_offset_raster), or None to skip the
            correction entirely (vertical_datum_correction.enabled=false,
            the default). When given, the offset is resampled onto this
            tile's own DEM grid and ADDED to every cell with valid DeltaDTM
            elevation - never to the fill cells above, which are not real
            elevations. See src/vertical_datum.py for the full reasoning.
        min_hard_fill_component_size: Connected-component size (in cells,
            4-connectivity) at/above which a missing-land gap is hard-filled
            to `land_fill_value_m` rather than interpolated. Default 10 (i.e. a
            cell connected to at least 9 other missing-land cells).
        interp_max_search_distance: `rasterio.fill.fillnodata`'s search
            radius (pixels) for small-gap interpolation.
        interp_smoothing_iterations: `rasterio.fill.fillnodata`'s post-fill
            smoothing pass count for small-gap interpolation.
        land_fill_value_m: Elevation (m) written for large missing-land gaps
            (simulation.dem_gap_fill.land_fill_value_m).

    Returns:
        The DEM clipped to `bbox` with all missing cells filled.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=bbox)

    if geoid_offset_raster is not None:
        from vertical_datum import sample_geoid_offset

        valid_dem = (da.values != da.raster.nodata)
        offset = sample_geoid_offset(
            geoid_offset_raster, da.raster.transform, da.raster.crs,
            da.raster.height, da.raster.width,
        ).reshape(da.shape)
        corrected = np.where(valid_dem, da.values + offset, da.values).astype(da.dtype)
        da = da.copy(data=corrected)

    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(255)  # uint8 nodata; mirrors extract_dem_mask
    mask_vals = da_mask.raster.reproject_like(da, method="nearest").values.squeeze()

    # Land (0) or no mask coverage at all (255) -> irrelevant dry land.
    # Ocean (1), lake (2), river (3) -> flood-passable, fill with 0.
    is_land = (mask_vals == 0) | (mask_vals == 255)

    dem_vals = da.values.reshape(mask_vals.shape)
    missing = dem_vals == da.raster.nodata
    missing_land = missing & is_land

    # Connected components of missing-land cells only (4-connectivity) -
    # ocean/lake/river nodata cells never participate here and always get
    # 0.0 regardless of gap size.
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labels, _ = ndimage.label(missing_land, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    cell_component_size = component_sizes[labels]
    small_gap = missing_land & (cell_component_size < min_hard_fill_component_size)

    if small_gap.any():
        # Interpolates a candidate value for every originally-missing cell
        # (land and water alike) from real valid DeltaDTM cells only - only
        # the small_gap subset of these candidates is actually used below;
        # large land gaps and water cells get their own fixed fill value
        # regardless of what this computes for them.
        interpolated = fillnodata(
            dem_vals.astype(np.float32).copy(),
            mask=(~missing).astype(np.uint8),
            max_search_distance=interp_max_search_distance,
            smoothing_iterations=interp_smoothing_iterations,
        )
    else:
        interpolated = dem_vals  # never read below (small_gap is all-False)

    fill = np.where(
        is_land,
        np.where(small_gap, interpolated, land_fill_value_m),
        0.0,
    ).reshape(da.shape).astype(np.float32)

    result = da.where(da != da.raster.nodata, da.copy(data=fill))
    # Rounding to cm precision (well within DeltaDTM's own vertical accuracy)
    # roughly halves the compressed file size - elevation varies continuously
    # almost everywhere, so most of a float32 mantissa is incompressible noise.
    return result.round(2)


def extract_dem_mask(
    data_catalog: hydromt.DataCatalog,
    mask_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    nodata_sentinel: int = 255,
) -> xr.DataArray:
    """Clip the DEM-validity mask and reproject it onto the DEM's grid.

    The mask source raster has a different native grid/resolution than the
    DEM, so it is reprojected (nearest-neighbour) onto the DEM's grid to
    ensure matching dimensions, as required by the flood model. Cells with no
    mask coverage at all (value == `nodata_sentinel`) are set to land (0) —
    consistent with `extract_dem`'s own fill rule: areas outside DeltaDTM's
    own coverage are irrelevant, definitely-dry terrain, not an unknown for a
    separately sourced land polygon dataset to arbitrate. Valid DeltaTM
    values (0 = land, 1 = ocean, 2 = lake, 3 = river) are kept unchanged.

    Args:
        data_catalog: HydroMT data catalog containing `mask_source`.
        mask_source: Name of the DEM-mask RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        dem: The tile's DEM, as returned by `extract_dem`, used as the
            reprojection target grid.
        nodata_sentinel: Value used by `mask_source` to indicate no data.

    Returns:
        The DEM-validity mask reprojected onto `dem`'s grid with no nodata cells.
    """
    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(nodata_sentinel)
    da_mask_repr = da_mask.raster.reproject_like(dem, method="nearest")
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

    # Copernicus land-use coverage doesn't reach the poles (cuts off around
    # 80N) - a tile whose bbox extends beyond that has NO real land-use data
    # at all, and the clipped da_lulc ends up with fewer than 2 cells in one
    # spatial dim. reproject_like needs >=2 cells in each dim to compute a
    # valid affine transform (HydroMT raises "Invalid raster: less than 2
    # cells in y_dim ..." otherwise, crashing the whole preprocessing job) -
    # treat this the same as "no valid land use classification anywhere in
    # this tile" (the same fallback every individual unclassified cell
    # already gets below), using `dem` itself as the coordinate/CRS
    # template since it's already on the exact target grid.
    if da_lulc.raster.height < 2 or da_lulc.raster.width < 2:
        friction = np.full((dem.raster.height, dem.raster.width), default_friction, dtype=np.float32)
        return dem.copy(data=friction)

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

    Tiled explicitly (512x512 blocks) regardless of `raster_config["driver"]`, since
    plain GTiff (unlike COG) defaults to striped layout - tiling matters here because
    downstream code (merge.py's per-block reads, tile_split.py's windowed land-pixel
    count) reads small rectangular windows, not full scanlines.

    Args:
        da: The data array to save. Must have `da.raster.crs`, `da.raster.transform`,
            `da.raster.width` and `da.raster.height` accessors (provided by rioxarray
            via HydroMT's raster accessor).
        output_path: Destination file path.
        raster_config: The workflow's `raster_format` configuration section,
            with keys `driver`, `compression`, `predictor` and `nodata`.
        dtype: Output data type.
    """
    with retry_transient_io(
        rasterio.open,
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
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(da.values, indexes=1)


def save_nodata_raster(reference_path: str | Path, output_path: str | Path, raster_config: dict[str, Any]) -> None:
    """Write an all-nodata waterdepth raster on the same grid as `reference_path`.

    Used as a placeholder `waterdepth` output for tiles that are skipped
    instead of being run through Aqueduct (see
    `aqueduct_runner.log_skipped_tile`). Filled with `merge.AQUEDUCT_NODATA`,
    the same sentinel the Aqueduct model itself writes for cells outside the
    area it computed, so `merge.merge_tile_rasters_chunk` ignores this tile
    entirely when merging results.

    Args:
        reference_path: Path to a raster (e.g. the tile's DEM) whose grid
            (transform, CRS, width, height) the output should match.
        output_path: Destination file path.
        raster_config: The workflow's `raster_format` configuration section,
            with keys `driver`, `compression` and `predictor`.
    """
    with retry_transient_io(rasterio.open, reference_path) as ref:
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
    with retry_transient_io(rasterio.open, output_path, "w", **profile) as dst:
        dst.write(data, indexes=1)


def save_waterdepth_raster(
    reference_path: str | Path, waterdepth: np.ndarray, output_path: str | Path,
) -> None:
    """Write a genuinely-computed `waterdepth` raster from `flood_model.flood_depth_dense`.

    Matches real `aqueduct.exe` output's OWN on-disk convention exactly
    (confirmed by inspecting a real production file: `driver=GTiff,
    dtype=float32, nodata=0.0, compress=zstd, predictor=None, tiled=True,
    blockxsize=blockysize=512`) - deliberately NOT the pipeline's generic
    `raster_format` config (`predictor=3, nodata=-9999`), which Julia's own
    writer never receives and does not use. `merge.merge_tile_rasters_chunk`
    doesn't actually read this file's `nodata` tag (it compares pixel values
    directly against `merge.AQUEDUCT_NODATA`), but matching Julia's real
    convention keeps every waterdepth file - regardless of which engine
    produced it - self-consistent for any other tool (QGIS, gdalinfo, ...)
    that does read the tag.

    Args:
        reference_path: Path to a raster (e.g. the tile's DEM) whose grid
            (transform, CRS, width, height) the output should match.
        waterdepth: The computed water depth array, same shape as the
            reference raster, `0.0` where not flooded.
        output_path: Destination file path.
    """
    with retry_transient_io(rasterio.open, reference_path) as ref:
        transform, crs = ref.transform, ref.crs
        height, width = ref.height, ref.width

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": 0.0,
        "compress": "zstd",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    with retry_transient_io(rasterio.open, output_path, "w", **profile) as dst:
        dst.write(waterdepth.astype("float32"), indexes=1)
