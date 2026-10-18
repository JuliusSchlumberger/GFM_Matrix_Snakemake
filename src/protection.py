"""WRI geogunit-ID resolution for the GFM exposure pipeline."""

import hydromt
import numpy as np
import xarray as xr

# Geogunit raster nodata sentinel (in addition to NaN from CF decoding).
_GEOGUNIT_NODATA_VALUES = (-9999.0,)

# Sentinel for cells with no resolvable WRI geogunit; always < 0, so callers
# can test validity with `geo_ids >= 0`.
GEOGUNIT_INVALID = -1


def load_geogunit_ids(
    data_catalog: hydromt.DataCatalog,
    geogunit_source: str,
    reference: xr.DataArray,
) -> np.ndarray:
    """Reproject WRI geogunit IDs onto `reference`'s grid.

    Uses nearest-neighbour reprojection, since geogunit IDs are categorical
    region codes rather than a continuous field that could be interpolated.

    Parameters
    ----------
    data_catalog : hydromt.DataCatalog
        Catalog used to fetch the geogunit source raster.
    geogunit_source : str
        Name of the geogunit dataset in `data_catalog` (must expose a
        "Geogunits" variable).
    reference : xr.DataArray
        Grid (CRS, bounds, resolution) to reproject the geogunit IDs onto.

    Returns
    -------
    np.ndarray
        (height, width) int32 array of geogunit IDs on `reference`'s grid.
        Cells with no resolvable geogunit (nodata, NaN, or outside source
        coverage) are set to `GEOGUNIT_INVALID` (-1).
    """
    bbox = list(reference.raster.bounds)
    da_geo = data_catalog.get_rasterdataset(geogunit_source, bbox=bbox, variables=["Geogunits"])
    da_geo_repr = da_geo.raster.reproject_like(reference, method="nearest")
    geo_ids = da_geo_repr.values.squeeze().astype("float64")

    valid = np.isfinite(geo_ids) & ~np.isin(geo_ids, _GEOGUNIT_NODATA_VALUES) & (geo_ids >= 0)
    out = np.full(geo_ids.shape, GEOGUNIT_INVALID, dtype="int32")
    out[valid] = geo_ids[valid].astype("int32")
    return out
