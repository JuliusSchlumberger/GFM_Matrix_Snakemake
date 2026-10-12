"""Output names of the final SFINCS calibration (calibration_roughness_SFINCS),
shared by run_calibration_models.py (gfm env) and compute_calibration_result.py
(hydromt-sfincs-dev env) - dependency-free so either env can import it."""

from __future__ import annotations

from pathlib import Path

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
FRICTION_SCALE_FACTORS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0)


def eikonal_path(out_dir: Path, fsf: float) -> Path:
    return out_dir / f"eikonal_on_subgrid_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}_fsf{fsf:g}.tif"


def bathtub_path(out_dir: Path) -> Path:
    return out_dir / f"bathtub_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"
