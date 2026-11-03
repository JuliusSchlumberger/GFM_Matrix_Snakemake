"""High-water statistics of a tide series (MHWS, MHW, HW99, HAT); main entry `high_water_statistics`."""

import numpy as np
from scipy.signal import find_peaks

HALF_LUNATION_S = 14.765294 * 86400.0  # spring-neap cycle
SPRING_WINDOW_S = 36 * 3600.0
# Separates successive high waters of semi-diurnal and mixed tides but not the double high
# waters of shallow-water tides, which the prominence filter merges.
MIN_HIGH_WATER_SEPARATION_S = 6 * 3600.0
MIN_HIGH_WATER_PROMINENCE_M = 0.01


def high_waters(tide: np.ndarray, dt_s: float) -> np.ndarray:
    """Return the indices of the high waters of a tide series."""
    idx, _ = find_peaks(np.asarray(tide, dtype=np.float64),
                        distance=max(1, int(round(MIN_HIGH_WATER_SEPARATION_S / dt_s))),
                        prominence=MIN_HIGH_WATER_PROMINENCE_M)
    return idx


def spring_high_waters(tide: np.ndarray, dt_s: float, hw_idx: np.ndarray | None = None) -> np.ndarray:
    """Return the indices of the high waters within 36 h of each spring maximum.

    The spring maximum is the highest high water of each half-lunation (14.77 d) block,
    so there are two per lunar month.
    """
    if hw_idx is None:
        hw_idx = high_waters(tide, dt_s)
    if hw_idx.size == 0:
        return hw_idx
    block = np.floor(hw_idx * dt_s / HALF_LUNATION_S).astype(int)
    window = SPRING_WINDOW_S / dt_s
    keep = np.zeros(hw_idx.size, dtype=bool)
    for b in np.unique(block):
        in_block = np.flatnonzero(block == b)
        # Skip incomplete blocks at the series ends: their maximum need not be a spring.
        if (b + 1) * HALF_LUNATION_S > len(tide) * dt_s or in_block.size < 20:
            continue
        spring = hw_idx[in_block[np.argmax(tide[hw_idx[in_block]])]]
        keep |= np.abs(hw_idx - spring) <= window
    return hw_idx[keep]


def high_water_statistics(tide: np.ndarray, dt_s: float) -> dict:
    """Return MHWS, MHW, HW99 and HAT (m, relative to the series' own datum).

    MHWS is the mean of the high waters within 36 h of each half-lunation spring maximum.
    HAT is the series maximum, so the series should span a nodal cycle (18.61 y) for it to
    be the astronomical HAT. `high_waters_m` holds every high water, for joint probability.
    """
    tide = np.asarray(tide, dtype=np.float64)
    hw = high_waters(tide, dt_s)
    spring = spring_high_waters(tide, dt_s, hw)
    hw_values = tide[hw]
    return {
        "mhws_m": float(tide[spring].mean()) if spring.size else np.nan,
        "mhw_m": float(hw_values.mean()) if hw.size else np.nan,
        "hw99_m": float(np.quantile(hw_values, 0.99)) if hw.size else np.nan,
        "hat_m": float(tide.max()),
        "n_high_waters": int(hw.size),
        "n_spring_high_waters": int(spring.size),
        "high_waters_m": hw_values,
    }
