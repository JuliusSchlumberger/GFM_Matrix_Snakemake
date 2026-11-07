"""Synthetic checks of the storm-tide product's rules and the storm-tide source switch."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from boundaries import storm_tide_hydrograph_path, storm_tide_levels_path  # noqa: E402
from flood_model import _idw_nearest_k  # noqa: E402
from skew_surge import HW_PROMINENCE, high_waters  # noqa: E402
from storm_tide_product import R_CAP, station_ratios  # noqa: E402
from tide_prediction import consistent_version_flag  # noqa: E402

HOURS = np.arange(24 * 30, dtype=float)


def test_high_waters_drop_a_low_water_bump_and_keep_a_lesser_high_water() -> None:
    m2 = 2 * np.pi / 12.42
    # Shallow-water tide: an M4 overtide above a quarter of M2 puts a bump into every low water.
    shallow = 4.0 * np.cos(m2 * HOURS) + 1.5 * np.cos(2 * m2 * HOURS)
    every, _ = high_waters(shallow)
    prominent, _ = high_waters(shallow, HW_PROMINENCE)
    assert len(every) > len(prominent)
    assert len(prominent) == pytest.approx(len(HOURS) / 12.42, abs=2)
    # Mixed tide: the lesser high water is weak but sits well above the low waters.
    k1 = 2 * np.pi / 23.93
    mixed = 0.5 * np.cos(m2 * HOURS) + 0.45 * np.cos(k1 * HOURS)
    assert len(high_waters(mixed, HW_PROMINENCE)[0]) == len(high_waters(mixed)[0])


def test_version_flag_grows_to_a_neighbour_on_the_same_artefact_unless_a_gauge_vetoes() -> None:
    lonlat = np.array([[0.0, 0.0], [0.03, 0.0], [5.0, 0.0]])
    mh22 = np.array([2.54, 2.37, 2.37])
    mh14 = np.array([1.20, 1.20, 1.20])
    assert consistent_version_flag(mh22, mh14, lonlat).tolist() == [True, True, False]
    veto = np.array([[0.05, 0.0]])
    assert consistent_version_flag(mh22, mh14, lonlat, veto).tolist() == [True, False, False]


def test_idw_power_weights_the_nearest_station_more() -> None:
    cell = np.radians([[0.0, 0.0]])
    stations = np.radians([[0.0, 0.01], [0.0, 0.03]])
    values = np.array([1.0, 0.0])
    p2, p3 = (_idw_nearest_k(cell, stations, values, 2, p)[0] for p in (2.0, 3.0))
    assert p2 == pytest.approx(0.9)
    assert p3 == pytest.approx(27 / 28)


def test_station_ratio_takes_the_gauge_median_without_neighbours_and_is_capped() -> None:
    gauges = pd.DataFrame({"lat": [0.0, 0.1, 0.2, 40.0], "lon": [0.0, 0.0, 0.0, 0.0],
                           "ratio_fes": [3.0, 3.0, 3.0, 1.1], "calibrate": True})
    stations = pd.DataFrame({"lat": [0.05, -40.0], "lon": [0.0, 0.0]})
    r = station_ratios(stations, gauges)
    assert r.r.iloc[0] == R_CAP
    assert r.r_neighbours.iloc[1] == 0
    assert r.r_uncapped.iloc[1] == pytest.approx(3.0)


def test_storm_tide_source_picks_the_files() -> None:
    cfg = {"boundary_conditions": {"storm_tide_source": "gfm_product", "storm_tide_product_dir": "/p/product"},
           "simulation": {"flooding": {"duration_correction": {"coast_hg_path": "/p/COAST-HG_RP100.nc"}}}}
    assert storm_tide_levels_path(cfg, "/p/COAST-RP.nc") == Path("/p/product/GFM_storm_tide_levels.nc")
    assert storm_tide_hydrograph_path(cfg) == Path("/p/product/GFM_storm_tide_hydrographs.nc")
    cfg["boundary_conditions"]["storm_tide_source"] = "coast_rp"
    assert storm_tide_levels_path(cfg, "/p/COAST-RP.nc") == Path("/p/COAST-RP.nc")
    assert storm_tide_hydrograph_path(cfg) == Path("/p/COAST-HG_RP100.nc")
    cfg["boundary_conditions"]["storm_tide_source"] = "coastrp"
    with pytest.raises(ValueError):
        storm_tide_hydrograph_path(cfg)
