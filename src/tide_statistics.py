"""High-water statistics of a tide series (MHWS, MHW, HW99, HAT); main entry `high_water_statistics`."""

import numpy as np
from scipy.signal import find_peaks

HALF_LUNATION_S = 14.765294 * 86400.0  # spring-neap cycle
SPRING_WINDOW_S = 36 * 3600.0
# Separates successive high waters of semi-diurnal and mixed tides but not the double high
# waters of shallow-water tides, which the prominence filter merges.
MIN_HIGH_WATER_SEPARATION_S = 6 * 3600.0
MIN_HIGH_WATER_PROMINENCE_M = 0.01
# A complete half-lunation holds ~28.5 semi-diurnal or ~14.8 diurnal high waters; fewer means a gap.
MIN_HIGH_WATERS_PER_BLOCK = 10


def high_waters(tide: np.ndarray, dt_s: float) -> np.ndarray:
    """Return the indices of the high waters of a tide series."""
    idx, _ = find_peaks(np.asarray(tide, dtype=np.float64),
                        distance=max(1, int(round(MIN_HIGH_WATER_SEPARATION_S / dt_s))),
                        prominence=MIN_HIGH_WATER_PROMINENCE_M)
    return idx


def spring_high_waters(tide: np.ndarray, dt_s: float, hw_idx: np.ndarray | None = None) -> np.ndarray:
    """Return the indices of the high waters within 36 h of each spring maximum.

    The spring maximum is the highest high water of each half-lunation (14.77 d) block,
    so there are two per lunar month. For diurnal tides this maximum is the tropic (declination)
    maximum, whose 13.66 d cycle the 14.77 d blocks sample once each.
    """
    if hw_idx is None:
        hw_idx = high_waters(tide, dt_s)
    if hw_idx.size == 0:
        return hw_idx
    crests = peak_values(np.asarray(tide, dtype=np.float64), hw_idx)
    block = np.floor(hw_idx * dt_s / HALF_LUNATION_S).astype(int)
    window = SPRING_WINDOW_S / dt_s
    keep = np.zeros(hw_idx.size, dtype=bool)
    for b in np.unique(block):
        in_block = np.flatnonzero(block == b)
        # Skip incomplete blocks at the series ends: their maximum need not be a spring.
        if (b + 1) * HALF_LUNATION_S > len(tide) * dt_s or in_block.size < MIN_HIGH_WATERS_PER_BLOCK:
            continue
        spring = hw_idx[in_block[np.argmax(crests[in_block])]]
        keep |= np.abs(hw_idx - spring) <= window
    return hw_idx[keep]


def peak_values(tide: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return the vertex of the parabola through each peak and its two neighbours.

    Corrects the sampling offset of the crest (~0.1 % of the M2 amplitude at 10 min). Hourly
    series cannot be corrected this way where shallow-water constituents sharpen the crest:
    errors of 2-8 cm remained at a 4.3 m MHWS GTSM station, so tide series are used at 10 min.
    """
    inner = (idx > 0) & (idx < len(tide) - 1)
    values = tide[idx].astype(np.float64)
    i = idx[inner]
    a, b, c = tide[i - 1], tide[i], tide[i + 1]
    curvature = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        refined = b - (a - c) ** 2 / (8 * curvature)
    values[inner] = np.where(curvature < 0, refined, b)
    return values


def high_water_statistics(tide: np.ndarray, dt_s: float) -> dict:
    """Return MHWS, MHW, HW99 and HAT (m, relative to the series' own datum).

    MHWS is the mean of the high waters within 36 h of each half-lunation spring maximum.
    High-water values are parabola-refined crests (`peak_values`), so hourly and 10-min
    series agree. HAT is the highest crest, so the series should span a nodal cycle
    (18.61 y) for it to be the astronomical HAT. `high_waters_m` holds every high water.
    """
    tide = np.asarray(tide, dtype=np.float64)
    hw = high_waters(tide, dt_s)
    spring = spring_high_waters(tide, dt_s, hw)
    hw_values = peak_values(tide, hw)
    spring_values = hw_values[np.isin(hw, spring)]
    return {
        "mhws_m": float(spring_values.mean()) if spring.size else np.nan,
        "mhw_m": float(hw_values.mean()) if hw.size else np.nan,
        "hw99_m": float(np.quantile(hw_values, 0.99)) if hw.size else np.nan,
        "hat_m": float(max(tide.max(), hw_values.max(initial=-np.inf))),
        "n_high_waters": int(hw.size),
        "n_spring_high_waters": int(spring.size),
        "high_waters_m": hw_values,
    }
