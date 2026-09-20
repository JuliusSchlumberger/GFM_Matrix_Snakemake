"""Mean Dynamic Topography (MDT) lookup, for re-referencing local-MSL data
onto the GOCO06s geoid this pipeline's DEM/boundary forcing already uses.

`_load_mdt`/`_nearest_valid_grid` are copied (not imported) from
`preparation/prepare_boundary_conditions.py` - same functions, same
behaviour, duplicated here rather than cross-imported across the
preparation/ vs sfincs_tiles/ package boundary for two small, private
(underscore-prefixed) helpers never meant to be a public library API.

Sign convention (matches prepare_boundary_conditions.py's own module
docstring, fixed 2026-09 after a real, confirmed sign-error investigation):
MDT is ADDED to re-reference local-MSL data onto GOCO06s -
`H_GOCO06s = H_MSL + MDT`. GEBCO (bathymetry) and COAST-HG (storm-tide
hydrographs) are both local-MSL-referenced at the source, exactly like
COAST-RP was before this same correction - so the same ADD convention
applies to both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _load_mdt(mdt_path: Path, mdt_variable: str = "mdt") -> xr.DataArray:
    """Load the AVISO MDT as a 2-D lat/lon DataArray, ascending coordinates."""
    with xr.open_dataset(mdt_path) as ds:
        da = ds[mdt_variable].load()

    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())
    for extra in [d for d in da.dims if d not in (lat_dim, lon_dim)]:
        da = da.isel({extra: 0})

    if float(da[lon_dim].max()) > 180:
        da = da.assign_coords(
            {lon_dim: xr.where(da[lon_dim] > 180, da[lon_dim] - 360, da[lon_dim])}
        )
    return da.sortby([lat_dim, lon_dim])


def _nearest_valid_grid(
    da: xr.DataArray, lon_dim: str, lon: float, lat_dim: str, lat: float, fallback_deg: float,
) -> float:
    """Value of a 2-D lat/lon grid nearest (lon, lat), NaN-safe.

    Falls back to the nearest non-NaN cell within +/-fallback_deg if the
    nearest cell itself is NaN. `da` must have ascending lat/lon
    coordinates (see `_load_mdt`).
    """
    val = float(da.sel({lon_dim: lon, lat_dim: lat}, method="nearest").values)
    if not np.isnan(val):
        return val

    window = da.sel({
        lon_dim: slice(lon - fallback_deg, lon + fallback_deg),
        lat_dim: slice(lat - fallback_deg, lat + fallback_deg),
    })
    if window.size == 0:
        return np.nan

    values = window.values
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan

    lons2d, lats2d = np.meshgrid(window[lon_dim].values, window[lat_dim].values)
    dist2 = (lons2d - lon) ** 2 + (lats2d - lat) ** 2
    dist2 = np.where(valid, dist2, np.inf)
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(values[idx])


def mdt_lookup_fn(mdt_path: Path, mdt_variable: str = "mdt", fallback_deg: float = 3.0):
    """Return a `f(lon, lat) -> MDT value (m)` closure, loading the MDT grid once.

    `fallback_deg` matches `boundary_conditions.mdt_correction.fallback_search_deg`
    in config.yml (default 3.0) - same tolerance the production COAST-RP MDT
    lookup uses for stations with no valid MDT cell exactly at their own
    location (e.g. a station right at a coastline pixel the MDT grid itself
    treats as land/nodata).
    """
    da = _load_mdt(mdt_path, mdt_variable)
    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())

    def _lookup(lon: float, lat: float) -> float:
        return _nearest_valid_grid(da, lon_dim, lon, lat_dim, lat, fallback_deg)

    return _lookup
