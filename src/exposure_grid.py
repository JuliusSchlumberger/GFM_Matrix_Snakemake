"""Cache population and geogunit-ID rasters (native ~1 km) once per spatial chunk."""

from pathlib import Path

import hydromt
import rasterio
from hydromt.exceptions import NoDataException

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
