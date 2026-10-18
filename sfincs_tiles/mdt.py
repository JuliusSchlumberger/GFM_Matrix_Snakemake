"""Look up Mean Dynamic Topography (MDT) to re-reference local-MSL data onto GOCO06s.

Sign convention: `H_GOCO06s = H_MSL + MDT`. Applies to GEBCO and COAST-HG, both
local-MSL-referenced at the source. Helpers copied from
preparation/prepare_boundary_conditions.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402


def _load_mdt(mdt_path: Path, mdt_variable: str = "mdt") -> xr.DataArray:
    with retry_transient_io(xr.open_dataset, mdt_path) as ds:
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
    # `da` must have ascending lat/lon coordinates (slice() below).
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
    """Return a `(lon, lat) -> MDT (m)` lookup over the AVISO MDT NetCDF, loaded once.

    If the nearest cell is NaN, uses the nearest valid cell within +/-`fallback_deg`
    (default matches config `boundary_conditions.mdt_correction.fallback_search_deg`), else NaN.
    """
    da = _load_mdt(mdt_path, mdt_variable)
    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())

    def _lookup(lon: float, lat: float) -> float:
        return _nearest_valid_grid(da, lon_dim, lon, lat_dim, lat, fallback_deg)

    return _lookup
