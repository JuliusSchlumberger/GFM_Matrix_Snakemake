"""Output names and waterdepth decoder of the SFINCS calibration (both envs).

Imports only numpy/rasterio (lazily) so gfm and hydromt-sfincs-dev can share it.
"""

from __future__ import annotations

from pathlib import Path

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
# Sweep grids of the calibration group: land-cover friction_scale_factor (per metre, x the
# subgrid cell size) and uniform attenuation (m/km); duration exponents at the best of each.
FRICTION_SCALE_FACTORS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)
UNIFORM_RATES_M_PER_KM = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.45)
DURATION_ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5)
RUNS_NAME = "model_runs.json"  # per tile: run key -> variant, parameter, alpha, solver diagnostics
SELECTION_NAME = "variant_selection.json"  # per base dir: the calibrated run of each variant
_RUN_SUFFIX = f"_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"


def run_path(out_dir: Path, key: str) -> Path:
    """Return the waterdepth raster path of run `key` (flood_variants.run_key)."""
    return out_dir / f"{key}{_RUN_SUFFIX}"


WATERDEPTH_SCALE = 100.0  # waterdepth rasters are int16 cm
WATERDEPTH_NODATA_INT16 = 32767


# Twin of src/rasters.py's decode_waterdepth_cm: hydromt-sfincs-dev can't import src/
# (src/config_utils needs hydromt 0.9). Pinned by tests/metric_parity_validation/validate_sfincs_src_parity.py.
def decode_waterdepth_cm(path: Path):
    """Decode an int16-cm waterdepth raster to float32 metres, NaN at nodata.

    Only nodata cells become NaN; a stored 0 cm (dry, or below 1 cm) stays 0.0. So NaN
    does not mean "dry": callers test wetness as ``np.isfinite(d) & (d > threshold)``.
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
