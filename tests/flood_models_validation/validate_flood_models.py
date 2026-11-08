"""Synthetic checks of the bathtub, straight-line and variant models of flood_depth_dense."""

import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flood_model import flood_depth_dense, seed_level_idw, straight_line_level  # noqa: E402
from flood_variants import VARIANTS, parameter_key, run_key, variant_flooding  # noqa: E402

TRANSFORM = Affine(30.0, 0, 0, 0, -30.0, 0)


def _strip_tile(n_rows: int = 20, n_cols: int = 30):
    # Ocean in the first column, land rising 0.1 m per cell inland, a closed pit at (10, 20).
    mask = np.zeros((n_rows, n_cols), dtype=np.int8)
    mask[:, 0] = 1
    dem = np.tile(np.arange(n_cols, dtype=np.float32) * 0.1, (n_rows, 1))
    dem[8:13, 18:23] = 5.0
    dem[10, 20] = -1.0
    rows = np.arange(n_rows)
    cols = np.zeros(n_rows, dtype=int)
    return dem, mask, rows, cols


def test_straight_line_level_matches_brute_force_within_a_bin() -> None:
    rng = np.random.default_rng(1)
    shape = (25, 35)
    rows, cols = rng.integers(0, shape[0], 12), rng.integers(0, shape[1], 12)
    values = rng.uniform(1.0, 2.0, 12)
    level = straight_line_level(shape, rows, cols, values, 2e-3, 30.0, bin_m=0.01)
    rr, cc = np.indices(shape)
    d = np.hypot(rr[..., None] - rows, cc[..., None] - cols) * 30.0
    exact = (values - 2e-3 * d).max(axis=-1)
    assert np.all(level <= exact + 1e-9)
    assert np.all(exact - level < 0.01 + 1e-9)


def test_seed_level_idw_is_the_seed_value_on_a_flat_coast() -> None:
    level = seed_level_idw((40, 40), np.arange(40), np.zeros(40, dtype=int), np.full(40, 1.5), k=5, idw_power=2)
    assert np.allclose(level, 1.5)


def test_bathtub_floods_the_closed_pit_and_the_connected_one_does_not() -> None:
    dem, mask, rows, cols = _strip_tile()
    common = dict(seed_rows=rows, seed_cols=cols, seed_values=np.full(len(rows), 1.0), ocean_code=1)
    bath, _ = flood_depth_dense(dem, mask, np.ones_like(dem), TRANSFORM, model="bathtub", **common)
    conn, _ = flood_depth_dense(dem, mask, np.ones_like(dem), TRANSFORM, model="bathtub_connected", **common)
    assert bath[10, 20] == pytest.approx(2.0)
    assert conn[10, 20] == 0.0
    # Land below 1.0 m along the open strip is wet in both, at level - ground.
    assert conn[0, 5] == pytest.approx(0.5) and bath[0, 5] == pytest.approx(0.5)
    assert conn[0, 10] == 0.0


def test_straight_line_attenuates_with_distance() -> None:
    dem, mask, rows, cols = _strip_tile()
    depth, diag = flood_depth_dense(dem, mask, np.ones_like(dem), TRANSFORM, seed_rows=rows, seed_cols=cols,
                                    seed_values=np.full(len(rows), 1.0), ocean_code=1, resolution=30.0,
                                    model="straight_line", attenuation_m_per_m=1e-3)
    # Level 1.0 - 0.03 m per cell: wet while 1.0 - 0.03 c > 0.1 c, i.e. c < 7.7.
    assert depth[0, 7] > 0 and depth[0, 8] == 0.0
    assert depth[0, 5] == pytest.approx(1.0 - 0.15 - 0.5, abs=0.03)
    assert diag["model"] == "straight_line"


def test_level_models_refuse_the_duration_correction() -> None:
    dem, mask, rows, cols = _strip_tile()
    with pytest.raises(ValueError):
        flood_depth_dense(dem, mask, np.ones_like(dem), TRANSFORM, seed_rows=rows, seed_cols=cols,
                          seed_values=np.ones(len(rows)), model="bathtub", duration_correction=object())
    with pytest.raises(ValueError):
        flood_depth_dense(dem, mask, np.ones_like(dem), TRANSFORM, seed_rows=rows, seed_cols=cols,
                          seed_values=np.ones(len(rows)), model="straight_line")


def test_variant_flooding_sets_model_parameter_and_duration() -> None:
    flooding = {"model": "eikonal", "friction_source": "landcover", "friction_scale_factor": 0.25,
                "uniform_attenuation_m_per_km": 0.2, "duration_correction": {"enabled": True, "alpha": 1.0}}
    v = variant_flooding(flooding, "eikonal_uniform_duration", 0.15, 0.5)
    assert (v["model"], v["friction_source"], v["uniform_attenuation_m_per_km"]) == ("eikonal", "uniform", 0.15)
    assert v["duration_correction"] == {"enabled": True, "alpha": 0.5}
    assert flooding["duration_correction"]["alpha"] == 1.0
    b = variant_flooding(flooding, "bathtub", 0.4)
    assert b["model"] == "bathtub" and not b["duration_correction"]["enabled"] and b["friction_scale_factor"] == 0.25
    assert parameter_key("eikonal_landcover") == "friction_scale_factor" and parameter_key("bathtub") is None
    assert run_key("eikonal_landcover_duration", 0.25, 1.0) == "eikonal_landcover_duration_fsf0.25_a1"
    assert run_key("straight_line", 0.2) == "straight_line_rate0.2"
    assert len(VARIANTS) == 7


def test_variant_config_moves_outputs_and_keeps_inputs() -> None:
    from flood_variants import variant_config

    config = {"simulation": {"model_outputs": "{root}/x/model_outputs", "preprocessing_inputs_dir": None,
                             "flooding": {"model": "eikonal", "friction_source": "landcover",
                                          "friction_scale_factor": 0.25, "uniform_attenuation_m_per_km": 0.2,
                                          "duration_correction": {"enabled": True, "alpha": 1.0}}},
              "postprocessing": {"merged_outputs": "{root}/x/merged"},
              "visualization": {"output_dir": "{root}/x/figures"},
              "validation": {"output_dir": "{root}/x/validation", "flood_totals_dir": "{root}/global/flood_totals"},
              "hpc": {"jobs_dir": "{root}/x/model_outputs/hpc_jobs"}}
    v = variant_config(config, "straight_line", 0.15)
    assert v["simulation"]["model_outputs"] == "{root}/x/model_outputs__straight_line"
    assert v["simulation"]["preprocessing_inputs_dir"] == "{root}/x/model_outputs"
    assert v["hpc"]["jobs_dir"] == "{root}/x/model_outputs__straight_line/hpc_jobs"
    assert v["postprocessing"]["merged_outputs"] == "{root}/x/merged__straight_line"
    assert v["validation"]["output_dir"] == "{root}/x/validation__straight_line"
    assert v["validation"]["flood_totals_dir"] == "{root}/global/flood_totals"
    assert v["simulation"]["flooding"]["model"] == "straight_line"
    assert v["simulation"]["flooding"]["uniform_attenuation_m_per_km"] == 0.15
    assert config["simulation"]["model_outputs"] == "{root}/x/model_outputs"
