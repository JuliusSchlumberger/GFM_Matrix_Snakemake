"""EAI along a pathway through a scenario-neutral growth matrix.

A growth matrix (visualization.build_growth_matrix_from_grid) gives EAI for every combination of
global SLR (columns, mm) and population change (rows, fraction relative to 2020). Any scenario -
an SSP-RCP, or an assumed continuation beyond 2100 - is a pathway through that space: one
(SLR, population change) pair per year. EAI along it is read by bilinear interpolation, with
linear extrapolation where the pathway leaves the matrix (exact for no adaptation in the
population direction, where EAI is linear in population; approximate otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def matrix_interpolator(matrix: np.ndarray, slr_mm, growth_fracs) -> RegularGridInterpolator:
    """Interpolator over (growth fraction, SLR mm) for a (len(growth_fracs), len(slr_mm)) matrix."""
    return RegularGridInterpolator((np.asarray(growth_fracs, float), np.asarray(slr_mm, float)), matrix,
                                   bounds_error=False, fill_value=None)


def eai_along_pathway(matrix: np.ndarray, slr_mm, growth_fracs, slr_path_mm, growth_path) -> np.ndarray:
    """EAI at each (slr_path_mm[i], growth_path[i]) point; negative extrapolations clipped to 0."""
    f = matrix_interpolator(matrix, slr_mm, growth_fracs)
    pts = np.column_stack([np.asarray(growth_path, float), np.asarray(slr_path_mm, float)])
    return np.clip(f(pts), 0.0, None)


def slr_pathway(slr_traj: pd.DataFrame, column: str, years) -> np.ndarray:
    """Global SLR (mm) of one trajectory column (e.g. "SSP2_p50") at `years`, linearly
    interpolated between the trajectory's (decadal) points."""
    return np.interp(np.asarray(years, float), slr_traj.index.to_numpy(float), slr_traj[column].to_numpy(float))
