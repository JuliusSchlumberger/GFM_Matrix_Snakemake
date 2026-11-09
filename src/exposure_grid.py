"""Cache population and geogunit-ID rasters (native ~1 km) once per spatial chunk; population per pixel."""

from pathlib import Path

import hydromt
import numpy as np
import rasterio
from hydromt.exceptions import NoDataException
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

from config_utils import retry_transient_io
from protection import GEOGUNIT_INVALID, load_geogunit_ids
from rasters import _atomic_raster_write


def prepare_exposure_grid_chunk(
    reference_path: str | Path,
    data_catalog: hydromt.DataCatalog,
    population_source: str,
    geogunit_source: str,
    population_output_path: str | Path,
    geogunit_output_path: str | Path,
) -> None:
    """Write the chunk's population raster and geogunit-ID raster on the population grid.

    `reference_path` supplies only the chunk bounds. Geogunit IDs are int32 with
    nodata `GEOGUNIT_INVALID`. Without population coverage both outputs are
    written as zero-byte placeholders.
    """
    # Plain rasterio avoids ~4 s of rioxarray plugin-discovery overhead per process.
    with retry_transient_io(rasterio.open, reference_path) as ref:
        bbox = list(ref.bounds)

    # Only "no coverage" yields the placeholder; any other read error fails the job,
    # since a placeholder counts as done and is never rerun.
    try:
        pop_da = retry_transient_io(
            data_catalog.get_rasterdataset, population_source, bbox=bbox
        ).squeeze(drop=True)
    except NoDataException:
        pop_da = None

    if pop_da is None or pop_da.size == 0:
        retry_transient_io(Path(population_output_path).touch)
        retry_transient_io(Path(geogunit_output_path).touch)
        return

    # Atomic write: concurrent compute_flood_fraction_chunk readers can otherwise
    # open a partial file.
    pop_profile = {
        "driver": "GTiff", "crs": pop_da.raster.crs, "transform": pop_da.raster.transform,
        "width": pop_da.raster.width, "height": pop_da.raster.height,
        "count": 1, "dtype": pop_da.dtype, "nodata": pop_da.raster.nodata,
    }
    _atomic_raster_write(population_output_path, pop_profile, lambda dst: dst.write(pop_da.values, indexes=1))

    geo_ids = load_geogunit_ids(data_catalog, geogunit_source, pop_da)
    geo_profile = {
        "driver": "GTiff", "crs": pop_da.raster.crs,
        "transform": pop_da.raster.transform,
        "width": pop_da.raster.width, "height": pop_da.raster.height,
        "count": 1, "dtype": "int32", "nodata": GEOGUNIT_INVALID, "compress": "lzw",
    }
    _atomic_raster_write(geogunit_output_path, geo_profile, lambda dst: dst.write(geo_ids.astype("int32"), indexes=1))


EARTH_RADIUS_KM = 6371.0088


def cell_area_km2(lat_top: float, lat_bottom: float, dlon_deg: float) -> float:
    """Area (km2) of a lon/lat cell on the sphere."""
    return (EARTH_RADIUS_KM ** 2 * np.radians(dlon_deg)
            * abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))))


def population_per_pixel(pop_path: str | Path, shape: tuple, transform, crs) -> tuple[np.ndarray, np.ndarray]:
    """Return (people per pixel, people per km2 of the pixel's population cell) on a finer grid.

    Each pixel takes count x pixel area / cell area of the population cell its centre lies in, so
    a fully covered cell sums to its count: production's equal share (rasters.average_pool_to_grid
    divides by all fine pixels of a cell). Pixel area from `transform` (metres or degrees at the
    pixel's latitude). NaN-free; zero outside settled cells.
    """
    h, w = shape
    bounds = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(h, w, transform), densify_pts=21)
    with retry_transient_io(rasterio.open, pop_path) as src:
        pad = 2 * abs(src.transform.a)
        win = from_bounds(bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad, src.transform)
        win = win.round_offsets().round_lengths()
        count = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        nodata = src.nodata
        wtr = src.window_transform(win)
    count[(count == nodata) | ~np.isfinite(count) | (count < 0)] = 0.0
    lat_top = wtr.f + wtr.e * np.arange(count.shape[0])
    area = np.array([cell_area_km2(t, t + wtr.e, abs(wtr.a)) for t in lat_top])[:, None] * np.ones(count.shape[1])
    density = np.divide(count, area, out=np.zeros_like(count), where=area > 0)
    out_density = np.zeros(shape, dtype=np.float64)
    reproject(density, out_density, src_transform=wtr, src_crs="EPSG:4326", dst_transform=transform, dst_crs=crs,
              resampling=Resampling.nearest)
    if rasterio.crs.CRS.from_user_input(crs).is_geographic:
        lat = transform.f + transform.e * (np.arange(h) + 0.5)
        pixel_km2 = np.array([cell_area_km2(t - transform.e / 2, t + transform.e / 2, abs(transform.a))
                              for t in lat])[:, None]
    else:
        pixel_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    return out_density * pixel_km2, out_density
