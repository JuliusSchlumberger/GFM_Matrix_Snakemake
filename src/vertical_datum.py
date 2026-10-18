"""EGM2008 -> GOCO06s vertical datum correction for the DeltaDTM DEM.

DeltaDTM (Pronk et al. 2024) is referenced to EGM2008; the AVISO MDT
(`mdt_cnes_cls22`) used for water levels is relative to GOCO06s. A pure
geoid-to-geoid shift avoids extrapolating MDT across land. EGM2008 is truncated
to GOCO06s's max spherical-harmonic degree; the offset N_EGM2008 - N_GOCO06s (m)
is ADDED to valid DEM pixels. No MDT term is applied here. Method ported from
GCFM_UU 05a_get_elevation.py. The global offset is computed once and cached;
per-tile resampling is valid because the field is smooth at arc-second scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config_utils import retry_transient_io


def compute_geoid_offset_grid(
    goco06s_gfc: str | Path, egm2008_gfc: str | Path,
) -> tuple[np.ndarray, Affine, str]:
    """Compute the global geoid offset N_EGM2008 - N_GOCO06s in metres.

    Inputs are ICGEM .gfc files. Returns `(offset, transform, crs)`: a north-up
    float32 grid, lon -180..180, at GOCO06s's native resolution. Raises
    ImportError without `pyshtools` / `boule`.
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError(
            "pyshtools is required for geoid height computation.\n"
            "Install with: pixi add pyshtools  (or: conda install -c conda-forge pyshtools)"
        )
    try:
        import boule as _boule
    except ImportError:
        raise ImportError(
            "boule is required by pyshtools for ellipsoid definitions.\n"
            "Install with: pixi add boule  (or: conda install -c conda-forge boule)"
        )
    from rasterio.transform import from_origin as _from_origin

    goco = retry_transient_io(pysh.SHGravCoeffs.from_file, str(goco06s_gfc), format="icgem")
    egm = retry_transient_io(pysh.SHGravCoeffs.from_file, str(egm2008_gfc), format="icgem")
    lmax = goco.lmax
    egm_trunc = egm.pad(lmax)

    # pyshtools 4.12 has no `ellipsoid=` kwarg; pass the WGS84 normal-gravity
    # parameters individually.
    wgs84 = _boule.WGS84
    geoid_kwargs = dict(
        potref=wgs84.reference_normal_gravity_potential,
        a=wgs84.semimajor_axis,
        f=wgs84.flattening,
        omega=wgs84.angular_velocity,
        lmax=lmax,
    )
    grid_goco = goco.geoid(**geoid_kwargs)
    grid_egm = egm_trunc.geoid(**geoid_kwargs)

    da_goco = grid_goco.to_xarray()
    da_egm = grid_egm.to_xarray()
    offset = (da_egm.values - da_goco.values).astype(np.float32)

    lat_dim = next(d for d in da_goco.dims if "lat" in d.lower())
    lon_dim = next(d for d in da_goco.dims if "lon" in d.lower())
    lats = da_goco[lat_dim].values
    lons = da_goco[lon_dim].values

    # pyshtools grids run 0..360 deg longitude; DEM tiles use -180..180.
    half = len(lons) // 2
    offset = np.roll(offset, -half, axis=1)
    lons = np.concatenate([lons[half:] - 360.0, lons[:half]])

    dlat = float(np.abs(lats[0] - lats[1]))
    dlon = float(lons[1] - lons[0])
    transform = _from_origin(
        float(lons[0]) - dlon / 2, float(lats[0]) + dlat / 2, dlon, dlat,
    )
    return offset, transform, "EPSG:4326"


def write_geoid_offset_raster(
    goco06s_gfc: str | Path, egm2008_gfc: str | Path, out_path: str | Path,
) -> None:
    """Compute the global geoid offset once and cache it as a GeoTIFF."""
    offset_arr, transform, crs = compute_geoid_offset_grid(goco06s_gfc, egm2008_gfc)

    retry_transient_io(Path(out_path).parent.mkdir, parents=True, exist_ok=True)
    with retry_transient_io(
        rasterio.open,
        out_path, "w",
        driver="GTiff", dtype="float32",
        width=offset_arr.shape[1], height=offset_arr.shape[0],
        count=1, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(offset_arr, 1)


def sample_geoid_offset(
    offset_raster_path: str | Path,
    dst_transform: Affine,
    dst_crs,
    dst_height: int,
    dst_width: int,
) -> np.ndarray:
    """Bilinearly resample the cached geoid offset (metres) onto a destination grid."""
    with retry_transient_io(rasterio.open, offset_raster_path) as src:
        offset_on_dst = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=offset_on_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return offset_on_dst
