"""Test eikonal.py's kernel: metric distance on rectangular cells, precision, sentinel, blocks."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eikonal  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep, _update, solve_eikonal_dense, sweep_constants  # noqa: E402


def _point_source_cost(cell_dx_m: float, cell_dy_m: float, m: int = 81, n: int = 81) -> np.ndarray:
    friction = np.full((m, n), 0.001, dtype=np.float64)
    t = solve_eikonal_dense(friction, np.array([m // 2]), np.array([n // 2]), np.array([0.0]), epsilon=0.0,
                            cell_dx_m=cell_dx_m, cell_dy_m=cell_dy_m, max_rounds=200)
    return t[: m + 1, : n + 1]


def test_uniform_cost_follows_metric_distance_on_narrow_cells() -> None:
    dx, dy = 15.0, 30.0  # cells at 60 deg latitude: half as wide as tall
    t = _point_source_cost(dx, dy)
    c = 40
    assert np.isclose(t[c, c + 20], 0.001 * 20 * dx, rtol=1e-9)
    assert np.isclose(t[c + 20, c], 0.001 * 20 * dy, rtol=1e-9)
    # off-axis: the first-order scheme overestimates the Euclidean distance by a few %
    for di, dj in ((20, 20), (10, 30), (30, 10)):
        exact = 0.001 * np.hypot(di * dy, dj * dx)
        assert exact <= t[c + di, c + dj] <= 1.08 * exact, (di, dj, t[c + di, c + dj], exact)


def test_square_cells_give_the_classic_two_sided_update() -> None:
    r2, one_plus_r2, inv_r = sweep_constants(30.0, 30.0, np.dtype(np.float64))
    t_a, t_b, v = 1.0, 1.2, 0.5
    classic = (t_a + t_b + np.sqrt(2 * v * v - (t_a - t_b) ** 2)) / 2
    assert np.isclose(_update(t_a, t_b, v, r2, one_plus_r2, inv_r), classic, rtol=1e-12)
    assert _update(1.0, 5.0, 0.5, r2, one_plus_r2, inv_r) == 1.5  # one-sided when the other is far


def test_update_is_accurate_in_float32_near_cancellation() -> None:
    args32 = (np.float32(-3.5), np.float32(-3.5), np.float32(0.001), *sweep_constants(15.0, 30.0, np.dtype(np.float32)))
    args64 = (-3.5, -3.5, 0.001, *sweep_constants(15.0, 30.0, np.dtype(np.float64)))
    r32, r64 = float(_update(*args32)), float(_update(*args64))
    assert abs(r32 - r64) / abs(r64) < 1e-6


def test_unseeded_cells_default_to_never_flooded_sentinel_not_sea_level() -> None:
    """Unreached cells keep t=+99 (water level -99 m); t=0 would flood any cell below sea level."""
    m, n = 5, 12
    friction = np.full((m, n), 0.5, dtype=np.float32)
    friction[:, 6] = 9999.0
    t = solve_eikonal_dense(friction, np.array([0]), np.array([0]), np.array([0.0], dtype=np.float32), epsilon=0.03,
                            cell_dx_m=1.0, cell_dy_m=1.0, max_rounds=50)
    assert np.all(-t[1:, 1:][:, 8:] <= -90.0)


def _winding_test_grid(seed: int, m: int = 61, n: int = 83):
    rng = np.random.default_rng(seed)
    friction = rng.uniform(0.002, 0.06, (m, n)).astype(np.float32)
    friction[rng.random((m, n)) < 0.3] = np.float32(9999.0)
    seed_rows = rng.integers(0, m, 5)
    seed_cols = rng.integers(0, n, 5)
    seed_values = -rng.uniform(1.0, 4.0, 5).astype(np.float32)
    return friction, seed_rows, seed_cols, seed_values


def _reference_dense_solve(friction, seed_rows, seed_cols, seed_values, epsilon, max_rounds, dx, dy):
    m, n = friction.shape
    t = np.full((m + 1, n + 1), 99.0, dtype=friction.dtype)
    t[seed_rows, seed_cols] = seed_values
    step = (friction * friction.dtype.type(dy)).astype(friction.dtype)
    consts = sweep_constants(dx, dy, friction.dtype)
    max_change, n_rounds = 0.0, 0
    for n_rounds in range(1, max_rounds + 1):
        max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            max_change = max(max_change, _dense_sweep(t, step, orthant, *consts))
        if max_change <= epsilon:
            break
    return t, n_rounds, max_change


def test_block_skipping_solve_is_bit_identical_to_dense_sweeps() -> None:
    """Block skipping must match dense sweeps exactly (t, rounds, max_change), any block size."""
    production_block_size = eikonal.SWEEP_BLOCK_SIZE
    try:
        for grid_seed, (dx, dy) in ((2, (30.0, 30.0)), (3, (14.0, 30.9))):
            friction, sr, sc, sv = _winding_test_grid(grid_seed)
            for epsilon in (0.0, 0.05):
                t_ref, rounds_ref, mc_ref = _reference_dense_solve(friction, sr, sc, sv, epsilon, 500, dx, dy)
                for block in (1, 3, 8, 16, production_block_size, 200):
                    eikonal.SWEEP_BLOCK_SIZE = block
                    t_blk, diag = solve_eikonal_dense(friction, sr, sc, sv, epsilon, cell_dx_m=dx, cell_dy_m=dy,
                                                      max_rounds=500, return_diagnostics=True)
                    assert np.array_equal(t_ref, t_blk), (grid_seed, epsilon, block)
                    assert diag["n_rounds_used"] == rounds_ref and diag["max_change"] == mc_ref
    finally:
        eikonal.SWEEP_BLOCK_SIZE = production_block_size


def main() -> None:
    test_uniform_cost_follows_metric_distance_on_narrow_cells()
    test_square_cells_give_the_classic_two_sided_update()
    test_update_is_accurate_in_float32_near_cancellation()
    test_unseeded_cells_default_to_never_flooded_sentinel_not_sea_level()
    test_block_skipping_solve_is_bit_identical_to_dense_sweeps()
    print("All eikonal kernel validation checks passed.")


if __name__ == "__main__":
    main()
