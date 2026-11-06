"""Synthetic storm-tide hydrograph per boundary cell: local spring tide + RP-specific surge shape.

HGRAPHER (Dullaart et al. 2023, NHESS 23:1847, https://doi.org/10.5194/nhess-23-1847-2023)
as ported from GCFM_UUdev workflow/src/storm_tide_hydrograph.py, with three changes: the
tide is a spring-tide window of the local tide (FES2022 or GTSM tidal_elevation) instead of
the average tide cycle; the surge target per RP is level - MHWS; and the shape is either
selected per RP from events (HGRAPHER-RP), pooled over all events with a fitted time scale
tau(peak), or a Gaussian for tropical-cyclone stations. Main entry: `cell_hydrograph`.

The per-RP generator of INPUT_ANALYSIS §8.3 (`rp_hydrograph`) instead takes the surge shape
from the station's annual maxima of matching empirical return period and lays it on a high
water at the station's own tide level during extremes. Levels are relative to local MSL, as
COAST-RP; MDT and SLR are added by the caller.
"""

import numpy as np
import pandas as pd
import scipy.signal as ss

STEPS_PER_HR = 6  # 10-min, as HGRAPHER
LEVELS = np.round(np.arange(0, 1.0001, 0.005), 3)
EVENT_SEPARATION_HR = 72
MSL_WINDOW_HR = 8766  # 1 year
SHAPE_MODES = ("per_rp", "pooled", "gaussian")
HIGH_WATER_SEPARATION_HR = 6
RP_WINDOW_FACTOR = 2.0
RECORD_RP_YR = 50.0  # beyond the ~46-y GTSM record no event of that RP class exists
LARGEST_EVENT_RP_YR = 20.0


def fill_short_gaps(series: pd.Series, max_gap_hr: int = 3) -> pd.Series | None:
    """Linearly fill gaps up to max_gap_hr; return None if longer gaps remain."""
    filled = series.interpolate(limit=max_gap_hr, limit_area="inside")
    return None if filled.isna().any() else filled


def remove_running_mean(series: pd.Series) -> pd.Series:
    """Subtract the 1-yr centred running mean (annual MSL variation in the surge residual)."""
    return series - series.rolling(MSL_WINDOW_HR, center=True, min_periods=MSL_WINDOW_HR // 2).mean()


def to_10min(series: pd.Series) -> np.ndarray:
    """Resample an hourly series to 10 min by cubic interpolation."""
    return series.resample("10min").interpolate("cubic").values


def storm_events(surge10: np.ndarray, half_window_hr: float, min_peak_m: float = 0.0) -> dict:
    """Find surge peaks >= 72 h apart and the time spent above each fraction of the peak.

    Peaks below min_peak_m are skipped (they cannot be selected for a larger target).

    Returns arrays over events: surge_peak_m, index (10-min position), dur_before /
    dur_after (n_events, len(LEVELS)) in 10-min steps above each level per limb, and
    dur50_hr (hours above half the peak).
    """
    half = int(round(half_window_hr * STEPS_PER_HR))
    peaks, _ = ss.find_peaks(surge10, distance=EVENT_SEPARATION_HR * STEPS_PER_HR)
    peaks = peaks[surge10[peaks] >= min_peak_m]
    out = {k: [] for k in ("surge_peak_m", "index", "dur_before", "dur_after")}
    for s in peaks:
        peak = surge10[s]
        if peak <= 0 or s - half < 0 or s + half >= len(surge10):
            continue
        before = surge10[s - half: s + 1] / peak
        after = surge10[s: s + half + 1] / peak
        neg_b, neg_a = np.where(before < 0)[0], np.where(after < 0)[0]
        before = before[neg_b[-1]:] if neg_b.size else before[1:]
        after = after[: neg_a[0]] if neg_a.size else after[:half]
        out["surge_peak_m"].append(peak)
        out["index"].append(s)
        out["dur_before"].append((before[:, None] > LEVELS).sum(axis=0))
        out["dur_after"].append((after[:, None] > LEVELS).sum(axis=0))
    events = {k: np.asarray(v) for k, v in out.items()}
    i50 = int(np.flatnonzero(LEVELS == 0.5)[0])
    events["dur50_hr"] = ((events["dur_before"][:, i50] + events["dur_after"][:, i50]) / STEPS_PER_HR
                          if len(events["surge_peak_m"]) else np.zeros(0))
    return events


def select_events_for_level(peaks: np.ndarray, target: float, min_events: int,
                            window_m: float) -> tuple[np.ndarray, float, str]:
    """Select events whose peak is within +-window_m of `target` (HGRAPHER-RP).

    Fewer than min_events inside: the min_events nearest ("widened"); a target above every
    recorded peak: the min_events largest ("above_record"). Returns (indices, window, mode).
    """
    n = min(min_events, len(peaks))
    if target > peaks.max():
        idx = np.argsort(peaks)[-n:]
        return idx, float(target - peaks[idx].min()), "above_record"
    dist = np.abs(peaks - target)
    idx = np.flatnonzero(dist <= window_m)
    if len(idx) >= n:
        return idx, float(window_m), "window"
    idx = np.argsort(dist)[:n]
    return idx, float(dist[idx].max()), "widened"


def surge_shape(dur_before: np.ndarray, dur_after: np.ndarray, half_window_hr: float) -> np.ndarray:
    """Build the normalised surge (peak 1 at t = 0) from mean per-level limb durations (10-min steps)."""
    half = int(round(half_window_hr * STEPS_PER_HR))
    steps = np.arange(-half, half + 1)
    rise = np.interp(np.abs(steps), dur_before[::-1], LEVELS[::-1], right=0.0)
    fall = np.interp(np.abs(steps), dur_after[::-1], LEVELS[::-1], right=0.0)
    return np.where(steps <= 0, rise, fall)


def fit_duration_scale(surge_peak_m: np.ndarray, dur50_hr: np.ndarray) -> tuple[float, float]:
    """Fit log(dur50) = a + b * surge peak over all events; returns (a, b)."""
    ok = (dur50_hr > 0) & np.isfinite(surge_peak_m)
    b, a = np.polyfit(surge_peak_m[ok], np.log(dur50_hr[ok]), 1)
    return float(a), float(b)


def pooled_shape(events: dict, target_m: float, half_window_hr: float,
                 fit: tuple[float, float] | None = None) -> tuple[np.ndarray, float]:
    """Mean shape of all events, stretched so its dur50 equals tau(target) from the fit.

    Each event's limb durations are divided by its own dur50 before averaging, so the
    pooled shape carries the events' form and the fit alone sets the time scale. Returns
    (shape, tau_hr).
    """
    if fit is None:
        fit = fit_duration_scale(events["surge_peak_m"], events["dur50_hr"])
    tau_hr = float(np.exp(fit[0] + fit[1] * target_m))
    ok = events["dur50_hr"] > 0
    scale = (tau_hr * STEPS_PER_HR) / (events["dur50_hr"][ok] * STEPS_PER_HR)
    before = (events["dur_before"][ok] * scale[:, None]).mean(axis=0)
    after = (events["dur_after"][ok] * scale[:, None]).mean(axis=0)
    return surge_shape(before, after, half_window_hr), tau_hr


def gaussian_shape(half_width_hr: float, half_window_hr: float) -> np.ndarray:
    """Gaussian surge with half width at half maximum `half_width_hr` (dur50 = 2 x half width)."""
    half = int(round(half_window_hr * STEPS_PER_HR))
    t = np.arange(-half, half + 1) / STEPS_PER_HR
    return np.exp(-np.log(2.0) * (t / half_width_hr) ** 2)


def spring_tide_window(tide10: np.ndarray, spring_idx: np.ndarray, mhws_m: float,
                       half_window_hr: float) -> np.ndarray:
    """Return the tide on +-half_window_hr around the spring high water closest to MHWS.

    `spring_idx`: 10-min positions of the spring high waters (tide_statistics). Choosing
    the one nearest MHWS makes the window's high water consistent with the MHWS used in
    the surge target.
    """
    return high_water_window(tide10, spring_idx, mhws_m, half_window_hr)


def high_water_window(tide10: np.ndarray, hw_idx: np.ndarray, target_m: float,
                      half_window_hr: float) -> np.ndarray:
    """Return the tide on +-half_window_hr around the high water in `hw_idx` closest to target_m."""
    half = int(round(half_window_hr * STEPS_PER_HR))
    valid = hw_idx[(hw_idx - half >= 0) & (hw_idx + half < len(tide10))]
    if valid.size == 0:
        raise ValueError("tide series shorter than the hydrograph window around every high water")
    centre = int(valid[np.argmin(np.abs(tide10[valid] - target_m))])
    return tide10[centre - half: centre + half + 1]


def high_water_indices(tide10: np.ndarray) -> np.ndarray:
    """10-min positions of all tidal high waters (local maxima at least 6 h apart)."""
    return ss.find_peaks(tide10, distance=HIGH_WATER_SEPARATION_HR * STEPS_PER_HR)[0]


def annual_maximum_events(level: np.ndarray, years: np.ndarray, half_window_hr: int,
                          min_year_coverage: float = 0.8, min_window_coverage: float = 0.9) -> pd.DataFrame:
    """Annual maxima of an hourly water-level series, with their empirical return period.

    `years`: calendar year of every hour. A year counts when >= min_year_coverage of it is
    present; its maximum is kept when its +-half_window_hr window is >= min_window_coverage
    complete. Empirical RP = (n + 1) / rank among the n kept maxima. Returns centre (hour
    index), peak_m and rp_yr.
    """
    rows = []
    for year in np.unique(years):
        sel = np.flatnonzero(years == year)
        if np.isfinite(level[sel]).mean() < min_year_coverage:
            continue
        centre = int(sel[np.nanargmax(level[sel])])
        lo, hi = centre - half_window_hr, centre + half_window_hr + 1
        if lo < 0 or hi > len(level) or np.isfinite(level[lo:hi]).mean() < min_window_coverage:
            continue
        rows.append({"centre": centre, "peak_m": float(level[centre])})
    events = pd.DataFrame(rows, columns=["centre", "peak_m"])
    rank = events.peak_m.rank(ascending=False, method="first")
    return events.assign(rp_yr=(len(events) + 1) / rank)


def rp_event_selection(rp_yr: np.ndarray, target_rp_yr: float) -> np.ndarray:
    """Indices of events whose empirical RP is within RP_WINDOW_FACTOR of the target RP.

    Targets at or beyond RECORD_RP_YR take the events with RP >= LARGEST_EVENT_RP_YR: the
    record holds nothing rarer, so the largest observed events give the shape.
    """
    if target_rp_yr >= RECORD_RP_YR:
        return np.flatnonzero(rp_yr >= LARGEST_EVENT_RP_YR)
    return np.flatnonzero((rp_yr > target_rp_yr / RP_WINDOW_FACTOR) & (rp_yr <= target_rp_yr * RP_WINDOW_FACTOR))


def median_surge_shape(windows: np.ndarray, centre_on_max: bool = True, search_hr: int = 12) -> np.ndarray:
    """Median normalised surge (1 at the centre) of hourly event windows, resampled to 10 min.

    `windows`: (n_events, 2 * half + 1) surge in m with the water-level peak at the centre.
    With centre_on_max each event is first shifted so that its largest surge within
    +-search_hr of the water-level peak sits at the centre (surge peak placed on the high
    water); otherwise the observed phase between surge and water-level peak is kept.
    """
    n_hr = windows.shape[1]
    c = n_hr // 2
    shaped = []
    for w in windows:
        seg = w[c - search_hr: c + search_hr + 1]
        if not np.isfinite(seg).any():
            continue
        k = c - search_hr + int(np.nanargmax(seg)) if centre_on_max else c
        if not w[k] > 0:
            continue
        shift = c - k
        out = np.full(n_hr, np.nan)
        if shift >= 0:
            out[shift:] = w[:n_hr - shift]
        else:
            out[:shift] = w[-shift:]
        shaped.append(out / w[k])
    if not shaped:
        raise ValueError("no event with a positive surge at its peak")
    median = np.nanmedian(np.vstack(shaped), axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    hours = np.arange(n_hr) - c
    steps = np.arange(-c * STEPS_PER_HR, c * STEPS_PER_HR + 1) / STEPS_PER_HR
    shape = np.interp(steps, hours, median)
    return shape / shape[len(shape) // 2]


def duration_mean_shape(windows: np.ndarray, search_hr: int = 12) -> np.ndarray:
    """Surge shape from the mean time each event stays above each fraction of its peak (HGRAPHER).

    `windows`: (n_events, 2 * half + 1) hourly surge in m with the water-level peak at the
    centre. Each event is taken around its largest surge within +-search_hr of that peak;
    per limb, the hours above each level of LEVELS (up to the first negative surge) are
    averaged over the events and turned into a 10-min shape by `surge_shape`. Averaging
    durations keeps the events' width; a median of aligned profiles narrows the top.
    """
    n_hr = windows.shape[1]
    c = n_hr // 2
    before, after = [], []
    for w in windows:
        seg = w[c - search_hr: c + search_hr + 1]
        if not np.isfinite(seg).any():
            continue
        k = c - search_hr + int(np.nanargmax(seg))
        peak = w[k]
        if not peak > 0:
            continue
        rise = np.nan_to_num(w[:k + 1][::-1] / peak, nan=-1.0)
        fall = np.nan_to_num(w[k:] / peak, nan=-1.0)
        for limb, store in ((rise, before), (fall, after)):
            neg = np.flatnonzero(limb < 0)
            limb = limb[:neg[0]] if neg.size else limb
            store.append((limb[:, None] > LEVELS).sum(axis=0) * STEPS_PER_HR)
    if not before:
        raise ValueError("no event with a positive surge at its peak")
    return surge_shape(np.mean(before, axis=0), np.mean(after, axis=0), c)


def rp_hydrograph(tide_window: np.ndarray, shape: np.ndarray, target_level_m: float) -> dict:
    """Storm-tide hydrograph of the per-RP generator: tide window + surge scaled to the target.

    `tide_window` and `shape` share the 10-min axis with the selected high water / surge peak
    at the centre. Returns time_hr, level and amplitude_m (surge scale).
    """
    level, amplitude = scale_to_peak(tide_window, shape, target_level_m)
    half = len(level) // 2
    return {"time_hr": np.arange(-half, half + 1) / STEPS_PER_HR, "level": level, "amplitude_m": amplitude}


def scale_to_peak(tide: np.ndarray, shape: np.ndarray, target_m: float) -> tuple[np.ndarray, float]:
    """Scale the surge so that max(tide + a * shape) = target_m; returns (hydrograph, a).

    The combined maximum need not sit at t = 0 (tide and surge peaks of different width), so
    `a` is found by bisection; a = 0 when the tide alone reaches the target.
    """
    if tide.max() >= target_m:
        return tide.copy(), 0.0
    lo, hi = 0.0, max(1.0, 2.0 * (target_m - tide.min()))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (tide + mid * shape).max() < target_m:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    return tide + a * shape, a


def time_above(hydrograph: np.ndarray, z_m: np.ndarray, dt_hr: float = 1.0 / STEPS_PER_HR) -> np.ndarray:
    """Return hours the hydrograph stays above each level in z_m (T(z) lookup)."""
    h = np.sort(np.asarray(hydrograph, dtype=np.float64))
    n_above = len(h) - np.searchsorted(h, np.asarray(z_m, dtype=np.float64), side="right")
    return n_above * dt_hr


def cell_hydrograph(tide_window: np.ndarray, events: dict, target_level_m: float, mhws_m: float,
                    half_window_hr: float, shape_mode: str, min_events: int = 10,
                    match_window_m: float = 0.05, tc_half_width_hr: float = 8.0,
                    fit: tuple[float, float] | None = None) -> dict:
    """Build one cell's storm-tide hydrograph for one return level (m, local MSL).

    Surge target = target_level_m - MHWS; the surge peak is placed on the window's tidal
    high water (t = 0), as COAST-HG. Returns time_hr, level, shape, amplitude_m, and the
    shape provenance (mode, n_events, tau_hr).
    """
    if shape_mode not in SHAPE_MODES:
        raise ValueError(f"shape_mode must be one of {SHAPE_MODES}, got {shape_mode!r}")
    surge_target = target_level_m - mhws_m
    info = {"shape_mode": shape_mode, "selection": "", "n_events": 0, "tau_hr": np.nan}
    if shape_mode == "gaussian":
        shape = gaussian_shape(tc_half_width_hr, half_window_hr)
        info["tau_hr"] = 2.0 * tc_half_width_hr
    elif shape_mode == "pooled":
        shape, info["tau_hr"] = pooled_shape(events, surge_target, half_window_hr, fit)
        info["n_events"] = int((events["dur50_hr"] > 0).sum())
    else:
        idx, _, info["selection"] = select_events_for_level(events["surge_peak_m"], surge_target,
                                                            min_events, match_window_m)
        shape = surge_shape(events["dur_before"][idx].mean(axis=0), events["dur_after"][idx].mean(axis=0),
                            half_window_hr)
        info["n_events"] = len(idx)
        info["tau_hr"] = float((shape > 0.5).sum()) / STEPS_PER_HR
    level, amplitude = scale_to_peak(tide_window, shape, target_level_m)
    half = int(round(half_window_hr * STEPS_PER_HR))
    return {"time_hr": np.arange(-half, half + 1) / STEPS_PER_HR, "level": level, "shape": shape,
            "amplitude_m": amplitude, **info}
