"""Effective-elevation preprocessing shared by the flood model and (removed)
flood-extent cropping.

`effective_dem` mirrors core.jl's own preprocessing (`dem[.!landmask] .= 0.0`)
rather than reading dem.tif's raw values directly: `rasters.extract_dem`
only fills cells that are nodata in the DEM SOURCE, so a river/lake/ocean
cell that happens to have a real (non-nodata) DeltaDTM elevation keeps that
real value in dem.tif - core.jl still zeroes it unconditionally at run time
because it is not land.
"""

import numpy as np

LAND_CODE = 0


def effective_dem(dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Elevation Aqueduct will actually use at run time.

    Mirrors core.jl's `dem[.!landmask] .= 0.0`: every non-land cell (ocean,
    lake, river) is treated as elevation 0 regardless of what value happens
    to be stored in dem.tif for it.
    """
    return np.where(mask == LAND_CODE, dem, dem.dtype.type(0.0))
