"""Output names and waterdepth decoder of the SFINCS calibration (both envs).

Imports only numpy/rasterio (lazily) so gfm and hydromt-sfincs-dev can share it.
"""

from __future__ import annotations

from pathlib import Path

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
FRICTION_SCALE_FACTORS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0)


def eikonal_path(out_dir: Path, fsf: float) -> Path:
    """Return the tile's eikonal waterdepth raster path for friction scale factor `fsf`."""
    return out_dir / f"eikonal_on_subgrid_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}_fsf{fsf:g}.tif"


def bathtub_path(out_dir: Path) -> Path:
    """Return the tile's bathtub waterdepth raster path."""
    return out_dir / f"bathtub_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"


WATERDEPTH_SCALE = 100.0  # waterdepth rasters are int16 cm
WATERDEPTH_NODATA_INT16 = 32767


# Twin of src/rasters.py's decode_waterdepth_cm: hydromt-sfincs-dev can't import src/
# (src/config_utils needs hydromt 0.9). Pinned by tests/metric_parity_validation/validate_sfincs_src_parity.py.
def decode_waterdepth_cm(path: Path):
    """Decode an int16-cm waterdepth raster to float32 metres, NaN at nodata.

    Returns ``(depth_m, transform, crs, shape)``.
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
