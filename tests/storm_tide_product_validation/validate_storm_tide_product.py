"""Synthetic checks of the storm-tide product's rules and the storm-tide source switch."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preparation"))
from boundaries import storm_tide_hydrograph_path, storm_tide_levels_path  # noqa: E402
from flood_model import _idw_nearest_k  # noqa: E402
from skew_surge import HW_PROMINENCE, high_waters, profile_loglik  # noqa: E402
from prepare_boundary_conditions import nearest_slr_m  # noqa: E402
from storm_tide_product import (  # noqa: E402
    BLEND_QUANTILES, FLAG_COLUMNS, NODE_LEVEL_SOURCE, RPS, SHAPE_GRID, TC_SHIFT_ALL_M, TC_SHIFT_M, blended_model,
    export_levels, model_vector, pooled_shapes, ratio_covariate, station_basins, station_flags, station_ratio,
    station_ratios, station_records, station_surge_model, tc_hours, tc_input_levels, tc_shift_m)
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
                        **{f"rp{rp}_m": [3.0, np.nan] for rp in RPS}, **{c: [True, False] for c in FLAG_COLUMNS}})
    nodes = pd.DataFrame({"lon": [0.5], "lat": [50.1], "atlas": ["fes2014"], "gpd_shape": [0.05],
                          **{f"rp{rp}_m": [4.0] for rp in RPS}, **{c: [True] for c in FLAG_COLUMNS}},
                         index=pd.Index([7], name="node"))
    export_levels(lev, tmp_path / "coast_rp.nc", tmp_path / "levels.nc", nodes)
    with xr.open_dataset(tmp_path / "levels.nc") as ds:
        assert ds.station_id.values.tolist() == ["a", "b", "node_7"]
        assert ds.storm_tide_rp_0100.values.tolist() == [3.0, 2.0, 4.0]
        assert ds.level_source.values.tolist() == ["product", "coast_rp", NODE_LEVEL_SOURCE]
        assert ds.station_x_coordinate.values[2] == 0.5
        assert ds.flag_fallback_r.values.tolist() == [1, 0, 1]


def test_slr_lookup_takes_the_nearest_value_inside_the_open_one_degree_box() -> None:
    slr_lonlat = np.array([[0.9, 0.9], [1.0, 0.0], [0.5, 0.5], [0.2, 0.0]])
    slr_mm = np.array([100.0, 200.0, 300.0, np.nan])
    station = np.array([[0.0, 0.0], [10.0, 10.0]])
    # [1.0, 0.0] lies on the box edge, [0.2, 0.0] has no value
    assert nearest_slr_m(slr_lonlat, slr_mm, station)[0] == pytest.approx(0.3)
    assert np.isnan(nearest_slr_m(slr_lonlat, slr_mm, station)[1])


def test_profile_likelihood_peaks_at_the_sample_shape() -> None:
    x = genpareto.rvs(0.2, scale=0.3, size=20000, random_state=np.random.default_rng(1))
    prof = profile_loglik(x, SHAPE_GRID)
    assert SHAPE_GRID[np.argmax(prof)] == pytest.approx(0.2, abs=0.03)
    assert np.isfinite(prof).all()


def test_pooled_shapes_sum_the_profiles_within_the_radius() -> None:
    def profile(c, weight=1.0):
        return -weight * (SHAPE_GRID - c) ** 2
    # two stations 50 km apart (the second with ten times the information), one 2,000 km away
    lonlat = np.array([[0.0, 0.0], [0.45, 0.0], [18.0, 0.0]])
    profiles = np.vstack([profile(0.4), profile(0.1, 10.0), profile(-0.1)])
    query = np.vstack([lonlat, [[90.0, 0.0]]])
    shapes = pooled_shapes(lonlat, profiles, query, radius_km=300.0)
    assert shapes[0] == shapes[1] == pytest.approx(round((0.4 + 1.0) / 11.0, 2))
    assert shapes[2] == pytest.approx(-0.1)
    assert np.isnan(shapes[3])


def _synthetic_station(n_years: int = 20, seed: int = 0):
    hours = np.arange(n_years * 8766)
    th = np.cos(2 * np.pi * hours / 12.42)
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, 0.02, len(hours))
    # a heavy-tailed surge at every high water, small enough to pass the despike
    hw = high_waters(th)[0]
    g[hw] += genpareto.rvs(0.5, scale=0.005, size=len(hw), random_state=rng)
    years = 1950 + hours // 8766
    return {"station": 0, "atlas": "fes2022", "mhws_m": 1.0, "r": 1.0}, g, th, years


def test_station_fit_has_no_shape_cap_and_takes_the_pooled_shape() -> None:
    st, g, th, years = _synthetic_station()
    row, _, _, _ = station_surge_model(st, g, th, years)
    assert row["gpd_shape"] == row["gpd_shape_mle"] > 0.35
    row, model, _, _ = station_surge_model({**st, "pooled_shape": 0.12}, g, th, years)
    assert model["shape"] == pytest.approx(0.12) and row["gpd_shape_mle"] > 0.35
    row, _, _, _ = station_surge_model({**st, "pooled_shape": np.nan}, g, th, years)
    assert row["gpd_shape"] == row["gpd_shape_mle"]


def test_cyclone_mask_drops_the_masked_cycles_from_the_fit() -> None:
    st, g, th, years = _synthetic_station()
    storm = np.arange(5000, 5049)
    g[storm] += 3.0  # a step, not a spike: despike keeps it
    open_row, open_model, _, _ = station_surge_model(st, g, th, years)
    row, model, _, _ = station_surge_model({**st, "tc_hours": storm}, g, th, years)
    assert open_row["tc_masked_share"] == 0.0 and 0.0 < row["tc_masked_share"] < 0.001
    assert len(model["sorted"]) < len(open_model["sorted"])
    assert open_model["sorted"].max() > 2.5 > model["sorted"].max()


def test_tc_shift_follows_the_basin_and_reaches_the_tc_levels() -> None:
    assert tc_shift_m("WP") == TC_SHIFT_M["WP"] and tc_shift_m("EP") == tc_shift_m("none") == TC_SHIFT_ALL_M
    assert all(0.18 <= v <= 0.26 for v in [*TC_SHIFT_M.values(), TC_SHIFT_ALL_M])
    st = {"tc": np.array([1.0, 2.0]), "tc_shift_m": tc_shift_m("NA")}
    assert tc_input_levels(st) == pytest.approx([1.0 + TC_SHIFT_M["NA"], 2.0 + TC_SHIFT_M["NA"]])
    assert tc_input_levels({"tc": None}) is None


def _ibtracs(path: Path, rows: list[tuple]) -> Path:
    header = "SID,ISO_TIME,BASIN,LAT,LON,WMO_WIND,USA_WIND\n ,Year-Month-Day,,degrees_north,degrees_east,kts,kts\n"
    path.write_text(header + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return path


def test_station_basins_keep_the_north_atlantic_code(tmp_path: Path) -> None:
    path = _ibtracs(tmp_path / "ibtracs.csv", [("a", "1990-09-01 00:00:00", "NA", 25.0, -80.0, 50, " "),
                                               ("a", "1990-09-01 06:00:00", "NA", 25.5, -80.5, 60, " "),
                                               ("b", "1990-09-02 00:00:00", "EP", 24.0, -82.0, 40, " ")])
    assert station_basins(np.array([[-80.2, 25.2], [0.0, 0.0]]), path).tolist() == ["NA", "none"]


def test_tc_hours_take_storms_with_wind_or_without_any_wind_on_record(tmp_path: Path) -> None:
    path = _ibtracs(tmp_path / "ibtracs.csv", [("weak", "1950-01-02 00:00:00", "SI", -20.0, 50.0, 20, " "),
                                               ("strong", "1950-01-03 00:00:00", "SI", -20.0, 50.0, 40, " "),
                                               ("old", "1950-01-05 00:00:00", "SI", -20.0, 50.0, " ", " ")])
    stations = pd.DataFrame({"lat": [-20.5, 40.0], "lon": [50.0, 0.0]}, index=[3, 4])
    hours = tc_hours(stations, 24 * 30, path, np.datetime64("1950-01-01T00:00"), half_hr=1)
    assert list(hours) == [3]
    assert hours[3].tolist() == [47, 48, 49, 95, 96, 97]


def test_station_flags() -> None:
    flags = station_flags([0.25, 0.1, np.nan], [True, False, False], [False, True, False])
    assert list(flags.columns) == list(FLAG_COLUMNS)
    assert flags.to_numpy().tolist() == [[True, True, False], [False, False, True], [False, False, False]]


def test_station_records_join_the_early_and_main_surge_stores(tmp_path: Path) -> None:
    start = np.datetime64("1950-01-01T00:00")
    n_hours = 48
    for name, first, values in (("early", start, [1.0] * 12 + [2.0] * 12),
                                ("main", start + np.timedelta64(24, "h"), [3.0] * 24)):
        store = tmp_path / name
        store.mkdir()
        pd.DataFrame({"gtsm_index": [7]}).to_parquet(store / "stations.parquet")
        np.save(store / "time.npy", (first + np.arange(24).astype("timedelta64[h]")).astype("datetime64[ns]"))
        np.save(store / "series.npy", (np.array(values) * 1000).astype(np.int16)[None, :])
    bases = []
    for k, n in enumerate((n_hours, n_hours, 6, 6)):
        bases.append(tmp_path / f"b{k}.npy")
        np.save(bases[-1], np.full((n, 1), 1.0 if k in (0, 2) else 0.0))
    st = {"station": 0, "atlas": "fes2022", "gtsm_index": 7, "hc": np.array([1.0 + 0j])}
    (_, g, th, t10, years), = station_records([st], [tmp_path / "early", tmp_path / "main"], {"fes2022": bases}, start)
    assert g.tolist() == [1.0] * 12 + [2.0] * 12 + [3.0] * 24
    assert th.tolist() == [1.0] * n_hours and t10.tolist() == [1.0] * 6
    assert years[0] == 1950 and len(years) == n_hours
