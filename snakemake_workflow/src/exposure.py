"""Exposure-grid preparation for the GFM Aqueduct postprocessing pipeline.

prepare_exposure_grid_chunk caches the population raster and geogunit-ID raster
(at population's native ~1 km resolution) once per spatial chunk.  All exposure
analysis is performed downstream at this coarse resolution by the standalone
scripts in snakemake_workflow/analysis/.
"""

from pathlib import Path

import hydromt
import numpy as np
import rasterio
import xarray as xr

from config_utils import retry_transient_io
from protection import GEOGUNIT_INVALID, load_geogunit_ids
from rasters import load_raster


def prepare_exposure_grid_chunk(
    reference_path: str | Path,
    data_catalog: hydromt.DataCatalog,
    population_source: str,
    geogunit_source: str,
    population_output_path: str | Path,
    geogunit_output_path: str | Path,
) -> None:
    """Cache the population raster and geogunit-ID raster (on population's grid) once per chunk.

    Both depend only on the chunk's bounds, not on return_period,
    waterlevel_name or adaptation measure.  Resolving them once here avoids
    repeating the HydroMT catalog fetch and nearest-neighbour geogunit
    reprojection in every one of the ~100+ jobs per chunk.

    Writes empty (zero-byte) placeholder files when the chunk has no
    population coverage (outside the raster's extent).
    """
    ref = load_raster(reference_path)
    bbox = list(ref.raster.bounds)

    try:
        pop_da = data_catalog.get_rasterdataset(population_source, bbox=bbox).squeeze(drop=True)
    except Exception:
        pop_da = None

    if pop_da is None or pop_da.size == 0:
        retry_transient_io(Path(population_output_path).touch)
        retry_transient_io(Path(geogunit_output_path).touch)
        return

    retry_transient_io(pop_da.raster.to_raster, population_output_path, driver="GTiff")

    geo_ids = load_geogunit_ids(data_catalog, geogunit_source, pop_da)
    profile = {
        "driver": "GTiff", "crs": pop_da.raster.crs,
        "transform": pop_da.raster.transform,
        "width": pop_da.raster.width, "height": pop_da.raster.height,
        "count": 1, "dtype": "int32", "nodata": GEOGUNIT_INVALID, "compress": "lzw",
    }
    with retry_transient_io(rasterio.open, geogunit_output_path, "w", **profile) as dst:
        dst.write(geo_ids.astype("int32"), 1)
