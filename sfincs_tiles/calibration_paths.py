"""Output names and the waterdepth decoder of the final SFINCS calibration
(calibration_roughness_SFINCS), shared by the gfm-env and hydromt-sfincs-dev-env
scripts - no imports beyond numpy/rasterio (loaded lazily), so either env can use it."""

from __future__ import annotations

from pathlib import Path

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
FRICTION_SCALE_FACTORS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0)


def eikonal_path(out_dir: Path, fsf: float) -> Path:
    """Build the per-tile eikonal waterdepth raster path for a friction scale factor.

    Parameters
    ----------
    out_dir : Path
        Tile's ``outputs`` directory.
    fsf : float
        Friction scale factor used for the eikonal run.

    Returns
    -------
    Path
        Path to ``eikonal_on_subgrid_waterdepth_<RP>_<SLR>_fsf<fsf>.tif`` under `out_dir`.
    """
    return out_dir / f"eikonal_on_subgrid_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}_fsf{fsf:g}.tif"


def bathtub_path(out_dir: Path) -> Path:
    """Build the per-tile bathtub waterdepth raster path.

    Parameters
    ----------
    out_dir : Path
        Tile's ``outputs`` directory.

    Returns
    -------
    Path
        Path to ``bathtub_waterdepth_<RP>_<SLR>.tif`` under `out_dir`.
    """
    return out_dir / f"bathtub_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"


WATERDEPTH_SCALE = 100.0  # waterdepth rasters are int16 cm
WATERDEPTH_NODATA_INT16 = 32767


def decode_waterdepth_cm(path: Path):
    """Decode an int16-cm waterdepth raster into metres with NaN for dry/nodata.

    Parameters
    ----------
    path : Path
        Path to the int16, centimetre-scaled waterdepth GeoTIFF.

    Returns
    -------
    tuple
        ``(depth_m, transform, crs, shape)`` - depth in metres as float32,
        NaN where the raster is dry or nodata.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
        transform, crs, shape = src.transform, src.crs, src.shape
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m, transform, crs, shape
