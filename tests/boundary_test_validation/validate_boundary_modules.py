"""Synthetic checks of tide_statistics, storm_tide_hydrograph and along_water."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from along_water import along_water_distances, along_water_idw  # noqa: E402
from storm_tide_hydrograph import (  # noqa: E402
    STEPS_PER_HR,
    annual_maximum_events,
    cell_hydrograph,
    fit_duration_scale,
    gaussian_shape,
    high_water_indices,
    high_water_window,
    median_surge_shape,
    rp_event_selection,
    rp_hydrograph,
    scale_to_peak,
    spring_tide_window,
    storm_events,
    time_above,
)
from tide_statistics import high_water_statistics, spring_high_waters  # noqa: E402

DT_S = 600.0
M2_HR, S2_HR = 12.4206, 12.0


def _tide(days: float = 120.0, m2: float = 1.0, s2: float = 0.3) -> np.ndarray:
    t_hr = np.arange(0, days * 24, DT_S / 3600)
    return m2 * np.cos(2 * np.pi * t_hr / M2_HR) + s2 * np.cos(2 * np.pi * t_hr / S2_HR)


def test_mhws_is_m2_plus_s2_and_mhw_is_m2() -> None:
    stats = high_water_statistics(_tide(), DT_S)
    assert abs(stats["mhws_m"] - 1.3) < 0.03
    assert abs(stats["mhw_m"] - 1.0) < 0.05
    assert stats["mhws_m"] < stats["hat_m"] <= 1.3 + 1e-6


def test_spring_window_is_centred_on_a_spring_high_water() -> None:
    tide = _tide()
    stats = high_water_statistics(tide, DT_S)
    window = spring_tide_window(tide, spring_high_waters(tide, DT_S), stats["mhws_m"], 72.0)
    half = 72 * STEPS_PER_HR
    assert window.size == 2 * half + 1
    assert abs(window[half] - stats["mhws_m"]) < 0.05


def test_scale_to_peak_hits_target_and_time_above_counts_hours() -> None:
    tide = _tide(days=7)[: 2 * 72 * STEPS_PER_HR + 1]
    shape = gaussian_shape(8.0, 72.0)
    level, a = scale_to_peak(tide, shape, 3.0)
    assert abs(level.max() - 3.0) < 1e-6 and a > 0
    assert time_above(np.ones(60), np.array([0.5, 1.5])).tolist() == [10.0, 0.0]


def test_gaussian_dur50_is_twice_half_width() -> None:
    shape = gaussian_shape(8.0, 72.0)
    assert abs((shape > 0.5).sum() / STEPS_PER_HR - 16.0) < 0.5


def _surge_with_events(peaks_m, widths_hr) -> np.ndarray:
    n = int(len(peaks_m) * 10 * 24 * STEPS_PER_HR)
    t = np.arange(n) / STEPS_PER_HR
    surge = np.zeros(n)
    for i, (p, w) in enumerate(zip(peaks_m, widths_hr)):
        surge += p * np.exp(-np.log(2) * ((t - (i * 240 + 120)) / w) ** 2)
    return surge


def test_pooled_fit_recovers_longer_events_for_larger_peaks() -> None:
    peaks = np.linspace(0.3, 1.5, 12)
    surge = _surge_with_events(peaks, 4.0 * np.exp(0.8 * peaks))
    events = storm_events(surge, 60.0)
    a, b = fit_duration_scale(events["surge_peak_m"], events["dur50_hr"])
    assert b > 0.5
    tide = np.zeros(2 * 60 * STEPS_PER_HR + 1)
    big = cell_hydrograph(tide, events, 2.0, 0.0, 60.0, "pooled")
    small = cell_hydrograph(tide, events, 0.5, 0.0, 60.0, "pooled")
    assert big["tau_hr"] > small["tau_hr"]
    assert abs(big["level"].max() - 2.0) < 1e-6


def test_along_water_distance_follows_the_channel_round_a_peninsula() -> None:
    water = np.zeros((20, 20), dtype=bool)
    water[:, :3] = True      # open sea on the west
    water[17:, :] = True     # channel along the south
    water[:, 17:] = True     # inlet up the east side
    station = np.array([[0, 0]])
    target = np.array([[0, 19]])  # head of the inlet, 19 cells east of the station
    d = along_water_distances(water, 100.0, station, target)[0, 0]
    assert d > 3000.0  # down 17, across 17, up 17 cells, not 1900 m straight
    values, fallback = along_water_idw(np.array([[1000.0], [4000.0]]), np.array([1.0, 2.0]), k=2)
    assert abs(values[0] - (1 / 1e6 + 2 / 16e6) / (1 / 1e6 + 1 / 16e6)) < 1e-9 and not fallback[0]


def test_annual_maxima_rank_gives_empirical_return_periods() -> None:
    years = np.repeat(np.arange(2000, 2010), 8760)
    level = np.zeros(len(years))
    for k in range(10):
        level[k * 8760 + 4000] = 1.0 + k
    events = annual_maximum_events(level, years, 72)
    assert len(events) == 10
    assert abs(events.rp_yr.max() - 11.0) < 1e-9 and abs(events.rp_yr.min() - 1.1) < 1e-9


def test_rp_selection_window_and_largest_events_beyond_the_record() -> None:
    rp = np.array([1.2, 1.8, 3.0, 6.0, 12.0, 25.0, 45.0])
    assert set(rp_event_selection(rp, 2.0)) == {0, 1, 2}
    assert set(rp_event_selection(rp, 10.0)) == {3, 4}
    assert set(rp_event_selection(rp, 100.0)) == {5, 6}


def test_median_surge_shape_recentres_on_the_surge_maximum() -> None:
    t = np.arange(-72, 73)
    windows = np.vstack([np.exp(-np.log(2) * ((t - lag) / 9.0) ** 2) * amp for lag, amp in ((-3, 0.5), (4, 1.2))])
    shape = median_surge_shape(windows)
    centre = len(shape) // 2
    assert int(np.argmax(shape)) == centre and abs(shape[centre] - 1.0) < 1e-9
    half = (shape >= 0.5).sum() / STEPS_PER_HR
    assert abs(half - 18.0) < 1.0  # half width at half maximum 9 h


def test_rp_hydrograph_reaches_the_target_on_a_high_water_at_the_chosen_level() -> None:
    tide = _tide()
    hw = high_water_indices(tide)
    window = high_water_window(tide, hw, 1.25, 72.0)
    assert abs(window[len(window) // 2] - 1.25) < 0.05
    shape = gaussian_shape(8.0, 72.0)
    out = rp_hydrograph(window, shape, 2.5)
    assert abs(out["level"].max() - 2.5) < 1e-6 and out["amplitude_m"] > 0


def test_high_water_window_ceiling_keeps_neighbouring_high_waters_below_the_target() -> None:
    tide = _tide()
    hw = high_water_indices(tide)
    assert high_water_window(tide, hw, 1.25, 72.0).max() >= 1.28
    window = high_water_window(tide, hw, 1.25, 72.0, ceiling_m=1.28)
    assert window.max() < 1.28
    out = rp_hydrograph(window, gaussian_shape(8.0, 72.0), 1.28)
    assert abs(out["level"].max() - 1.28) < 1e-6 and out["amplitude_m"] > 0
