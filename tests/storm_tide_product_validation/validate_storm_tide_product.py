"""Synthetic checks of the storm-tide product's rules and the storm-tide source switch."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preparation"))
from boundaries import storm_tide_hydrograph_path, storm_tide_levels_path  # noqa: E402
from flood_model import _idw_nearest_k  # noqa: E402
from skew_surge import HW_PROMINENCE, high_waters  # noqa: E402
from prepare_boundary_conditions import nearest_slr_m  # noqa: E402
from storm_tide_product import (  # noqa: E402
    BLEND_QUANTILES, NODE_LEVEL_SOURCE, RPS, blended_model, export_levels, model_vector, ratio_covariate, station_ratio,
    station_ratios)
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


def test_station_ratio_needs_three_gauges() -> None:
    gauges = pd.DataFrame({"lat": [0.0, 0.1, 0.2, 40.0], "lon": [0.0, 0.0, 0.0, 0.0],
                           "ratio_fes": [3.0, 3.0, 3.0, 1.1], "calibrate": True})
    stations = pd.DataFrame({"lat": [0.05, -40.0], "lon": [0.0, 0.0]})
    r = station_ratios(stations, gauges)
    assert r.r_idw.iloc[0] == pytest.approx(3.0)
    assert np.isnan(r.r_idw.iloc[1]) and r.r_neighbours.iloc[1] == 0


def test_ratio_covariate_recovers_a_power_law_and_falls_back_to_it() -> None:
    s10 = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    gauges = pd.DataFrame({"skew_gtsmfes_m": s10, "ratio_fes": 1.03 * s10 ** -0.32, "calibrate": True})
    slope, intercept = ratio_covariate(gauges)
    assert slope == pytest.approx(-0.32) and np.exp(intercept) == pytest.approx(1.03)
    model = {"sorted": np.linspace(0.0, 1.0, 100)}
    st = {"r_idw": np.nan, "r_covariate": (slope, intercept)}
    with patch("storm_tide_product.skew_level", return_value=0.1):
        assert station_ratio(st, model) == pytest.approx(1.03 * 0.1 ** -0.32)
        assert station_ratio({**st, "r_idw": 1.2}, model) == 1.2


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


def test_blended_model_averages_quantiles_and_gpd_parameters() -> None:
    a = {"sorted": np.linspace(0.0, 1.0, 101), "u": 0.9, "p_u": 0.1, "shape": 0.1, "scale": 0.2, "theta": 0.5}
    b = {"sorted": np.linspace(1.0, 3.0, 51), "u": 2.8, "p_u": 0.1, "shape": -0.1, "scale": 0.4, "theta": 0.7}
    m = blended_model(np.vstack([model_vector(a), model_vector(b)]), np.array([0.75, 0.25]))
    assert len(m["sorted"]) == len(BLEND_QUANTILES)
    assert m["sorted"][0] == pytest.approx(0.25) and m["sorted"][-1] == pytest.approx(1.5)
    assert m["shape"] == pytest.approx(0.05) and m["theta"] == pytest.approx(0.55)


def test_export_levels_appends_the_coastline_nodes(tmp_path: Path) -> None:
    coast = xr.Dataset({f"storm_tide_rp_{rp:04d}": ("stations", [1.0, 2.0]) for rp in RPS},
                       coords={"station_id": ("stations", ["a", "b"]), "station_x_coordinate": ("stations", [0.0, 1.0]),
                               "station_y_coordinate": ("stations", [50.0, 50.0])})
    coast.to_netcdf(tmp_path / "coast_rp.nc")
    lev = pd.DataFrame({"station_id": ["a", "b"], "level_source": ["product", None], "r": [1.1, np.nan],
                        "atlas": ["fes2022", "fes2022"], "gpd_shape": [0.1, np.nan],
                        **{f"rp{rp}_m": [3.0, np.nan] for rp in RPS}})
    nodes = pd.DataFrame({"lon": [0.5], "lat": [50.1], "atlas": ["fes2014"], "gpd_shape": [0.05],
                          **{f"rp{rp}_m": [4.0] for rp in RPS}}, index=pd.Index([7], name="node"))
    export_levels(lev, tmp_path / "coast_rp.nc", tmp_path / "levels.nc", nodes)
    with xr.open_dataset(tmp_path / "levels.nc") as ds:
        assert ds.station_id.values.tolist() == ["a", "b", "node_7"]
        assert ds.storm_tide_rp_0100.values.tolist() == [3.0, 2.0, 4.0]
        assert ds.level_source.values.tolist() == ["product", "coast_rp", NODE_LEVEL_SOURCE]
        assert ds.station_x_coordinate.values[2] == 0.5


def test_slr_lookup_takes_the_nearest_value_inside_the_open_one_degree_box() -> None:
    slr_lonlat = np.array([[0.9, 0.9], [1.0, 0.0], [0.5, 0.5], [0.2, 0.0]])
    slr_mm = np.array([100.0, 200.0, 300.0, np.nan])
    station = np.array([[0.0, 0.0], [10.0, 10.0]])
    # [1.0, 0.0] lies on the box edge, [0.2, 0.0] has no value
    assert nearest_slr_m(slr_lonlat, slr_mm, station)[0] == pytest.approx(0.3)
    assert np.isnan(nearest_slr_m(slr_lonlat, slr_mm, station)[1])
