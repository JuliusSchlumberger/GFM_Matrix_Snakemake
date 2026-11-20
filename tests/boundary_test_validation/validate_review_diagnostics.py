"""Synthetic checks of the TC raise helpers of boundary_test/review_diagnostics."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "boundary_test"))

from review_diagnostics import _basin_values, _raised  # noqa: E402


def test_surge_scale_keeps_the_tide_floor_and_drops_the_production_shift() -> None:
    tc = np.array([1.0, 1.0, 1.2, 1.5, 2.0])
    out = _raised([{"tc": tc, "tc_shift_m": 0.2}, {"tc": None}], "tc_surge_scale", 1.5)
    assert out[0]["tc_shift_m"] == 0.0 and out[1]["tc"] is None
    assert np.allclose(out[0]["tc"], [1.0, 1.0, 1.3, 1.75, 2.5])


def test_shift_replaces_the_production_shift() -> None:
    out = _raised([{"tc": np.ones(3), "tc_shift_m": 0.2}], "tc_shift_m", 0.3)
    assert out[0]["tc_shift_m"] == 0.3 and np.allclose(out[0]["tc"], 1.0)


def test_basin_fits_zero_their_gauges_median_error() -> None:
    trials = np.array([0.0, 0.1, 0.2, 0.3])
    # basin A needs 0.1, basin B needs 0.3: error = trial - need
    basin = np.array(["A", "A", "B", "B"])
    need = np.array([0.1, 0.1, 0.3, 0.3])
    err = trials[:, None] - need[None, :]
    loo, own, all_value = _basin_values(trials, err, basin)
    assert np.isclose(own["A"], 0.1) and np.isclose(own["B"], 0.3)
    assert np.isclose(loo["A"], 0.3) and np.isclose(loo["B"], 0.1)
    assert 0.1 <= all_value <= 0.3
