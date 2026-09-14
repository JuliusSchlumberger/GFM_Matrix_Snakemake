"""EGM2008 -> GOCO06s vertical datum correction for the DeltaDTM DEM.

DeltaDTM's original release (4TU.ResearchData, Pronk et al. 2024) is
referenced to the EGM2008 geoid. GOCO06s is the geoid the AVISO MDT
(`mdt_cnes_cls22`) - and therefore the water-level MDT correction in
preparation/prepare_boundary_conditions.py - are expressed relative to.
Converting the DEM from EGM2008 to GOCO06s directly (a pure geoid-to-geoid
shift, smooth at spherical-harmonic degree ~300 and computed once globally)
avoids extrapolating MDT across land, which is what the alternative
pre-corrected release (`*_GOCO06s_MDT`-suffixed tiles, hosted on the WUR
YODA vault) does.

Applied PER MODEL TILE, folded into rasters.extract_dem's own clip step
(rule extract_dem in preprocessing.smk) - not as a separate whole-tile
correction pass over the raw DeltaDTM release. The expensive part (the
spherical-harmonic geoid synthesis, ~12s) is therefore computed exactly
once, by the one-time rule compute_geoid_offset_raster, and cached to a
small GeoTIFF (`write_geoid_offset_raster`); every per-tile extract_dem job
then just reprojects a tiny window of that cached raster onto its own DEM
grid (`sample_geoid_offset`) - cheap, and correct because the offset field
itself is smooth at DEM (arc-second) scale, so resampling it is lossless
for this purpose.

Method ported from GCFM_UU/workflow/scripts/05a_get_elevation.py and
src/raster.compute_geoid_offset_arr: EGM2008 is truncated to GOCO06s's own
maximum spherical-harmonic degree before synthesis so both grids share the
same spectral bandwidth; the per-pixel offset N_EGM2008 - N_GOCO06s is
ADDED to every valid DEM pixel. Unlike 05a_get_elevation.py's separate GEBCO
correction, no MDT term is ever applied here - this module only performs
the geoid conversion.
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
    """Compute the global per-pixel geoid offset N_EGM2008 - N_GOCO06s.

    Requires the `pyshtools` and `boule` packages (conda-forge). Returns a
    global, north-up, float32 grid (lon in -180..180) at GOCO06s's own
    native spherical-harmonic resolution - reprojected onto each DEM tile's
    much finer grid by `sample_geoid_offset` below, since the offset field
    itself is smooth at DEM (arc-second) scale.

    Args:
        goco06s_gfc: Path to GOCO06s.gfc (ICGEM format).
        egm2008_gfc: Path to EGM2008.gfc (ICGEM format).

    Returns:
        (offset_arr, transform, crs) - offset_arr in metres.
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

    # pyshtools>=4.14's SHGravCoeffs.geoid() accepts a convenience
    # `ellipsoid=` kwarg that unpacks a boule.Ellipsoid itself; the
    # installed 4.12.2 predates that and still takes the four normal-
    # gravity-field parameters individually (potref/a/f/omega), so they're
    # pulled off `wgs84` explicitly here instead - same values either way.
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

    # pyshtools grids run 0..360 deg longitude - roll to -180..180 so the
    # offset lines up with the DEM tiles' own -180..180 convention.
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
    """Compute the global geoid offset once and cache it as a small GeoTIFF.

    Called by the one-time Snakemake rule `compute_geoid_offset_raster`
    (preprocessing.smk) - every per-tile `extract_dem` job then reads this
    cached raster via `sample_geoid_offset` instead of resynthesizing the
    spherical harmonics itself.
    """
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
    """Reproject the cached global geoid-offset raster onto an arbitrary grid.

    Used by rasters.extract_dem to sample the offset onto a single tile's
    own DEM clip window (cheap - just a small bilinear resample of an
    already-computed, already-cached global field).

    Returns a float32 array of shape (dst_height, dst_width), in metres.
    """
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
