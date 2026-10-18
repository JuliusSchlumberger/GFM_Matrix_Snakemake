"""EAI along a (SLR, population change) pathway through a scenario-neutral growth matrix.

Matrix rows: population change (fraction relative to 2020); columns: global SLR (mm).
Bilinear interpolation, linear extrapolation outside the matrix (exact in the
population direction without adaptation, where EAI is linear in population).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def matrix_interpolator(matrix: np.ndarray, slr_mm, growth_fracs) -> RegularGridInterpolator:
    """Build an interpolator over (growth fraction, SLR mm) for the matrix."""
    return RegularGridInterpolator((np.asarray(growth_fracs, float), np.asarray(slr_mm, float)), matrix,
                                   bounds_error=False, fill_value=None)


def eai_along_pathway(matrix: np.ndarray, slr_mm, growth_fracs, slr_path_mm, growth_path) -> np.ndarray:
    """Return EAI at each pathway point; negative extrapolations are clipped to 0."""
    f = matrix_interpolator(matrix, slr_mm, growth_fracs)
    pts = np.column_stack([np.asarray(growth_path, float), np.asarray(slr_path_mm, float)])
    return np.clip(f(pts), 0.0, None)


def slr_pathway(slr_traj: pd.DataFrame, column: str, years) -> np.ndarray:
    """Linearly interpolate global SLR (mm) of one trajectory column at `years`."""
    return np.interp(np.asarray(years, float), slr_traj.index.to_numpy(float), slr_traj[column].to_numpy(float))
