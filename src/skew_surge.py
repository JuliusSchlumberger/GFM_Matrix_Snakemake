"""Skew surges per tidal cycle, their GPD tail, and the convolution with the tide to RP levels.

Skew surge: highest level in a tidal cycle minus the predicted high water, cycles bounded
halfway between predicted high waters (Williams et al., 2016). A GPD on the declustered
exceedances above THRESHOLD_Q (clusters separated by more than RUN_CYCLES cycles; extremal
index = clusters / exceedances) with the empirical distribution below. Assuming skew surge
independent of the tide, the annual rate of a level z is
    lambda(z) = cycles per year x theta x mean over predicted high waters t_k of P(S > z - t_k).
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d
from scipy.signal import find_peaks
from scipy.stats import genpareto

THRESHOLD_Q = 0.995
RUN_CYCLES = 3  # ~1.5 days: one storm raises several consecutive high waters
HW_PROMINENCE = 0.1  # of the 25-h tidal range; drops the low-water bumps of shallow-water tides
LOW_WATER_BAND = 0.25  # of the 25-h range above its minimum: where a low-water bump can sit
SPIKE_M = 2.0  # GTSM residual jumps no storm produces within the SPIKE_HALF_HR median window
SPIKE_HALF_HR = 2


def annual_rate(rp) -> np.ndarray:
    """Annual exceedance rate -ln(1 - 1/RP) of return periods in years."""
    return -np.log1p(-1.0 / np.asarray(rp, dtype=float))


def high_waters(tide: np.ndarray, min_prominence: float | None = None,
                steps_per_hr: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the predicted high waters and their parabola-refined heights.

    With `min_prominence`, a local maximum in the lowest LOW_WATER_BAND of the 25-h tidal range
    is dropped when its prominence is under that fraction of the range: the low-water bumps of
    shallow-water tides (double low waters), otherwise taken for high waters whose cycles hold a
    spurious skew surge of the rising tide. Weak maxima higher up (the lesser or neap high
    waters of mixed tides) stay high waters, as skew surge defines.
    """
    if min_prominence is None:
        i = np.flatnonzero((tide[1:-1] > tide[:-2]) & (tide[1:-1] >= tide[2:])) + 1
    else:
        span = 25 * steps_per_hr + 1
        local_min = minimum_filter1d(tide, span)
        local_range = maximum_filter1d(tide, span) - local_min
        i, props = find_peaks(tide, prominence=0)
        bump = ((props["prominences"] < min_prominence * local_range[i])
                & (tide[i] < local_min[i] + LOW_WATER_BAND * local_range[i]))
        i = i[~bump]
    a, b, c = tide[i - 1], tide[i], tide[i + 1]
    curv = a - 2 * b + c
    return i, b - np.where(curv < 0, (c - a) ** 2 / (8 * curv), 0.0)


def skew_surges(total: np.ndarray, hw: np.ndarray, hw_height: np.ndarray) -> np.ndarray:
    """Skew surge per cycle around each high water; NaN where the cycle is not fully observed."""
    bounds = np.r_[0, (hw[:-1] + hw[1:]) // 2]
    peak = np.maximum.reduceat(np.where(np.isfinite(total), total, -np.inf), bounds)
    gaps = np.add.reduceat(~np.isfinite(total), bounds)
    return np.where(gaps == 0, peak - hw_height, np.nan)


def surge_model(s: np.ndarray, n_years: float, max_shape: float | None = None) -> dict:
    """Empirical body + GPD tail of per-cycle skew surges, with the extremal index.

    With `max_shape`, a fitted shape above it is fixed at max_shape and the scale refitted.
    """
    u = float(np.quantile(s, THRESHOLD_Q))
    exc = np.flatnonzero(s > u)
    starts = np.r_[0, np.flatnonzero(np.diff(exc) > RUN_CYCLES) + 1]
    cmax = np.maximum.reduceat(s[exc], starts)
    c, _, scale = genpareto.fit(cmax - u, floc=0)
    shape_mle = float(c)
    if max_shape is not None and c > max_shape:
        c, _, scale = genpareto.fit(cmax - u, f0=max_shape, floc=0)
    return {"sorted": np.sort(s), "u": u, "p_u": len(exc) / len(s), "shape": float(c), "scale": float(scale),
            "shape_mle": shape_mle, "theta": len(starts) / len(exc), "cycles_per_yr": len(s) / n_years,
            "n_years": n_years}


def scaled(model: dict, r: float) -> dict:
    """The model of r x S for S > 0 (the negative skew surges stay as they are)."""
    srt = model["sorted"]
    return {**model, "sorted": np.sort(np.where(srt > 0, r * srt, srt)), "u": model["u"] * r if model["u"] > 0
            else model["u"], "scale": model["scale"] * r}


def exceedance(model: dict, s: np.ndarray) -> np.ndarray:
    srt = model["sorted"]
    body = 1.0 - np.searchsorted(srt, s, side="right") / len(srt)
    tail = model["p_u"] * genpareto.sf(s - model["u"], model["shape"], 0, model["scale"])
    return np.where(s > model["u"], tail, body)


def skew_level(model: dict, rp: float) -> float:
    """Skew surge exceeded at a rate of -ln(1 - 1/rp) clusters per year."""
    rate_u = model["cycles_per_yr"] * model["theta"] * model["p_u"]
    p = -np.log1p(-1.0 / rp) / rate_u
    return float(model["u"] + genpareto.isf(p, model["shape"], 0, model["scale"])) if p < 1 else np.nan


def tide_peaks(tide: np.ndarray, step_hr: float,
               min_prominence: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """High-water heights binned to 1 cm (values, weights) and high waters per year."""
    _, h = high_waters(tide, min_prominence, int(round(1.0 / step_hr)))
    v, n = np.unique(np.round(h, 2), return_counts=True)
    return v, n / n.sum(), len(h) / (len(tide) * step_hr / 8766.0)


def despike(g: np.ndarray) -> np.ndarray:
    """Drop hours more than SPIKE_M from their (2 SPIKE_HALF_HR + 1)-h median (NaN)."""
    med = median_filter(np.nan_to_num(g), 2 * SPIKE_HALF_HR + 1, mode="nearest")
    return np.where(np.abs(g - med) > SPIKE_M, np.nan, g)
