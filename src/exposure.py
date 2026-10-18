"""Exposure-grid preparation for the GFM postprocessing pipeline.

prepare_exposure_grid_chunk caches the population raster and geogunit-ID raster
(at population's native ~1 km resolution) once per spatial chunk.  All exposure
analysis is performed downstream at this coarse resolution by the standalone
scripts in analysis/.
"""

from pathlib import Path

import hydromt
import rasterio

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
    """Fetch and write the population raster and geogunit-ID raster for one spatial chunk.

    Both outputs depend only on the chunk's bounds, not on return_period,
    waterlevel_name or adaptation measure, so resolving them once here avoids
    repeating the HydroMT catalog fetch and nearest-neighbour geogunit
    reprojection in every one of the ~100+ jobs per chunk. Writes empty
    (zero-byte) placeholder files at both output paths when the chunk has no
    population coverage (pure ocean/buffer, outside the raster's extent).

    Parameters
    ----------
    reference_path : str or Path
        Path to a raster defining the chunk's bounds (only its bounds are read).
    data_catalog : hydromt.DataCatalog
        Catalog used to fetch the population and geogunit source rasters.
    population_source : str
        Data-catalog key for the population raster.
    geogunit_source : str
        Data-catalog key for the WRI geogunit-ID raster.
    population_output_path : str or Path
        Destination for the cached population raster, on its native grid.
    geogunit_output_path : str or Path
        Destination for the geogunit-ID raster, reprojected onto the population grid
        (int32, nodata `GEOGUNIT_INVALID`).

    Returns
    -------
    None
    """
    # Only the bounds are needed here - a plain rasterio open avoids the
    # xarray/rioxarray backend-plugin-discovery overhead `load_raster`
    # carries (profiling on real data showed ~4s of pure import/reflection
    # cost per process, unrelated to this raster's size), which otherwise
    # gets paid once per chunk for no benefit here.
    with retry_transient_io(rasterio.open, reference_path) as ref:
        bbox = list(ref.bounds)

    try:
        pop_da = data_catalog.get_rasterdataset(population_source, bbox=bbox).squeeze(drop=True)
    except Exception:
        pop_da = None

    if pop_da is None or pop_da.size == 0:
        retry_transient_io(Path(population_output_path).touch)
        retry_transient_io(Path(geogunit_output_path).touch)
        return

    # _atomic_raster_write, not pop_da.raster.to_raster directly - confirmed
    # live (2026-10-15) that a concurrent reader (another dispatch's
    # compute_flood_fraction_chunk, racing against this chunk's own
    # (re)computation) can open this path mid-write and get a
    # "not recognized as a supported file format" RasterioIOError - the
    # exact corrupted-partial-read failure mode _atomic_raster_write exists
    # to prevent (see that function's own docstring), just never applied to
    # this particular writer before.
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
