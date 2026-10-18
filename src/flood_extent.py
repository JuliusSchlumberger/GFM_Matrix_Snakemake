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
    """Zero out elevation on non-land cells, mirroring core.jl's run-time preprocessing.

    Every non-land cell (ocean, lake, river) is treated as elevation 0.0 regardless of
    what value is stored in dem.tif for it, matching core.jl's `dem[.!landmask] .= 0.0`
    (needed because `rasters.extract_dem` only fills DEM-source nodata, so a water-body
    cell can still carry a real DeltaDTM elevation that core.jl zeroes anyway).

    Parameters
    ----------
    dem : np.ndarray
        Elevation raster (metres), same shape as `mask`.
    mask : np.ndarray
        Land/water code raster (0=land, 1=ocean, 2=lake, 3=river); any non-land code is
        zeroed.

    Returns
    -------
    np.ndarray
        Elevation array with the same dtype as `dem`, non-land cells set to 0.0.
    """
    return np.where(mask == LAND_CODE, dem, dem.dtype.type(0.0))
