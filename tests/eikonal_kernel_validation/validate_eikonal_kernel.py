"""Test eikonal.py's kernel: sweep order, dtype, unseeded default, block skipping."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eikonal  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep, _update, solve_eikonal_dense  # noqa: E402


def test_orthant_order_is_julias_real_gray_code_order() -> None:
    """Julia's sweep order is (1, 4, 3, 2); (1, 2, 3, 4) matched only 14-29% of cells."""
    print("=== _ORTHANT_ORDER: matches Julia's real Gray-code sweep order ===")
    assert _ORTHANT_ORDER == (1, 4, 3, 2), _ORTHANT_ORDER
    print(f"PASS: _ORTHANT_ORDER == {_ORTHANT_ORDER}")
    print()


def test_sweep_order_actually_changes_the_result() -> None:
    """Sweep order must change an under-converged result, or the order test is vacuous."""
    print("=== sweep order: naive (1,2,3,4) vs real (1,4,3,2) diverge before convergence ===")
    rng = np.random.default_rng(0)
    m, n = 9, 13  # non-square: symmetry could hide order dependence
    friction = rng.uniform(0.5, 3.0, (m, n)).astype(np.float32)
    seed_rows = np.array([0], dtype=np.int64)
    seed_cols = np.array([0], dtype=np.int64)
    seed_values = np.array([0.0], dtype=np.float32)

    def run(order, n_sweeps):
        t = np.full((m + 1, n + 1), 99.0, dtype=np.float32)
        t[seed_rows, seed_cols] = seed_values
        neg_two, eight, four = np.float32(-2.0), np.float32(8.0), np.float32(4.0)
        for i in range(n_sweeps):
            _dense_sweep(t, friction, order[i % 4], neg_two, eight, four)
        return t

    n_sweeps = 2  # full convergence masks order effects
    t_real = run(_ORTHANT_ORDER, n_sweeps)
    t_naive = run((1, 2, 3, 4), n_sweeps)

    diff = np.abs(t_real - t_naive)
    assert diff.max() > 1e-6, (
        "sweep order made no difference at all on this grid after 2 sweeps - either the test "
        "grid is degenerate, or _dense_sweep stopped depending on sweep order (worth investigating)."
    )
    print(f"PASS: real vs naive sweep order diverge (max diff {diff.max():.4f}) before full convergence, "
          f"confirming order genuinely matters")
    print()


def test_update_uses_the_callers_own_dtype_not_promoted() -> None:
    """_update must compute in the caller's dtype, not promote to float64.

    Julia uses float32; at real friction scales the discriminant is a ~1e-5
    catastrophic-cancellation result, so float64 promotion biases depths.
    """
    print("=== _update: computes natively in the caller's own dtype (float32-sensitive) ===")
    # Near-cancellation case (cand~-3.51, as in the Julia trace) where both dtypes
    # take the quadratic branch, so only precision differs.
    t_a_val, t_b_val, v_val = -3.5, -3.5, 0.001

    t_a32, t_b32, v32 = np.float32(t_a_val), np.float32(t_b_val), np.float32(v_val)
    neg_two32, eight32, four32 = np.float32(-2.0), np.float32(8.0), np.float32(4.0)
    result_f32 = _update(t_a32, t_b32, v32, neg_two32, eight32, four32)

    t_a64, t_b64, v64 = np.float64(t_a_val), np.float64(t_b_val), np.float64(v_val)
    neg_two64, eight64, four64 = np.float64(-2.0), np.float64(8.0), np.float64(4.0)
    result_f64 = _update(t_a64, t_b64, v64, neg_two64, eight64, four64)

    rel_diff = abs(float(result_f32) - float(result_f64)) / max(abs(float(result_f64)), 1e-12)
    print(f"  float32 call -> {result_f32!r}")
    print(f"  float64 call -> {result_f64!r}")
    print(f"  relative difference: {rel_diff:.6g}")

    assert rel_diff > 1e-5, (
        f"float32 and float64 calls agree to {rel_diff:.2e} relative precision on a deliberately "
        f"near-cancellation input - _update may have gained an internal float64 promotion "
        f"(e.g. an .astype() cast), which would silently reintroduce the systematic depth bias "
        f"this fix resolved. A genuinely dtype-native implementation should show a real, "
        f"visible difference here, not agreement to full float64 precision."
    )
    print("PASS: float32 and float64 calls give measurably different results on a near-cancellation "
          "input - confirms _update computes natively in the caller's own precision, not promoted")
    print()


def test_unseeded_cells_default_to_never_flooded_sentinel_not_sea_level() -> None:
    """Unreached cells must default to t=+99 (waterlevel -99), not t=0 (sea level).

    t=0 would flood any unreached below-sea-level cell (0 > -5).
    """
    print("=== solve_eikonal_dense: unreached cells never read as 'flooded at sea level' ===")
    m, n = 5, 12
    friction = np.full((m, n), 0.5, dtype=np.float32)
    # Impassable wall at OBSTACLE_BLOCK_FRICTION scale.
    friction[:, 6] = 9999.0

    seed_rows = np.array([0], dtype=np.int64)
    seed_cols = np.array([0], dtype=np.int64)
    seed_values = np.array([0.0], dtype=np.float32)

    t = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.03, max_rounds=50)
    waterlevel = -t[1:, 1:]

    far_side_waterlevel = waterlevel[:, 8:]
    unreached_dem_m = -5.0
    would_flood_if_unreached_is_zero = far_side_waterlevel > unreached_dem_m

    print(f"  far-side waterlevel range: [{far_side_waterlevel.min():.2f}, {far_side_waterlevel.max():.2f}] m")
    assert np.all(far_side_waterlevel <= -90.0), (
        f"unreached cells did not stay near the +99 sentinel (waterlevel<=-90 expected), "
        f"got range [{far_side_waterlevel.min():.2f}, {far_side_waterlevel.max():.2f}] - "
        f"the wall may not be impassable enough for this test's own epsilon/max_rounds, or the "
        f"sentinel default has changed."
    )
    assert not np.any(would_flood_if_unreached_is_zero), (
        "unreached cells would incorrectly read as flooded against a real negative-elevation DEM cell - "
        "unreached cells must keep the +99 sentinel, not t=0."
    )
    print(f"PASS: unreached region stays near the +99 sentinel (never reads as flooded against "
          f"a {unreached_dem_m}m DEM cell); a t=0 default would flood it")
    print()


def _reference_sweep_julia_loop_order(t, friction, orthant, neg_two, eight, four) -> float:
    """Reference kernel in Julia loop order (column-outer, no early-out), Aqueduct-exact."""
    m, n = friction.shape
    if orthant == 1:
        cells = [(i, j, i - 1, j, i, j - 1, i - 1, j - 1) for j in range(1, n + 1) for i in range(1, m + 1)]
    elif orthant == 2:
        cells = [(i, j, i + 1, j, i, j - 1, i, j - 1) for j in range(1, n + 1) for i in range(m - 1, -1, -1)]
    elif orthant == 3:
        cells = [(i, j, i + 1, j, i, j + 1, i, j) for j in range(n - 1, -1, -1) for i in range(m - 1, -1, -1)]
    else:
        cells = [(i, j, i - 1, j, i, j + 1, i - 1, j) for j in range(n - 1, -1, -1) for i in range(1, m + 1)]
    max_change = 0.0
    for i, j, ai, aj, bi, bj, fi, fj in cells:
        cand = _update(t[ai, aj], t[bi, bj], friction[fi, fj], neg_two, eight, four)
        if cand < t[i, j]:
            max_change = max(max_change, float(t[i, j] - cand))
            t[i, j] = cand
    return max_change


def test_dense_sweep_matches_julia_loop_order_bit_for_bit() -> None:
    """Row-major looping and no-improvement skipping must be exact, sweep by sweep."""
    print("=== _dense_sweep: bit-identical to the original Julia-loop-order kernel ===")
    rng = np.random.default_rng(1)
    m, n = 23, 31
    friction = rng.uniform(0.002, 0.06, (m, n)).astype(np.float32)
    friction[rng.random((m, n)) < 0.25] = np.float32(9999.0)
    neg_two, eight, four = np.float32(-2.0), np.float32(8.0), np.float32(4.0)

    t_fast = np.full((m + 1, n + 1), 99.0, dtype=np.float32)
    t_fast[rng.integers(0, m + 1, 6), rng.integers(0, n + 1, 6)] = -rng.uniform(1.0, 4.0, 6).astype(np.float32)
    t_ref = t_fast.copy()
    n_sweeps = 24
    for s in range(n_sweeps):
        orthant = _ORTHANT_ORDER[s % 4]
        mc_fast = _dense_sweep(t_fast, friction, orthant, neg_two, eight, four)
        mc_ref = _reference_sweep_julia_loop_order(t_ref, friction, orthant, neg_two, eight, four)
        assert np.array_equal(t_fast, t_ref), (
            f"sweep {s + 1} (orthant {orthant}): t differs from the reference kernel, "
            f"max diff {np.abs(t_fast - t_ref).max()}"
        )
        assert np.float32(mc_fast) == np.float32(mc_ref), (
            f"sweep {s + 1}: max_change {mc_fast} != reference {mc_ref}"
        )
    print(f"PASS: t and max_change identical to the reference kernel after each of {n_sweeps} sweeps")
    print()


def _winding_test_grid(seed: int, m: int = 61, n: int = 83):
    rng = np.random.default_rng(seed)
    friction = rng.uniform(0.002, 0.06, (m, n)).astype(np.float32)
    friction[rng.random((m, n)) < 0.3] = np.float32(9999.0)
    seed_rows = rng.integers(0, m, 5)
    seed_cols = rng.integers(0, n, 5)
    seed_values = -rng.uniform(1.0, 4.0, 5).astype(np.float32)
    return friction, seed_rows, seed_cols, seed_values


def _reference_dense_solve(friction, seed_rows, seed_cols, seed_values, epsilon, max_rounds):
    m, n = friction.shape
    t = np.full((m + 1, n + 1), 99.0, dtype=friction.dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two, eight, four = (friction.dtype.type(x) for x in (-2.0, 8.0, 4.0))
    max_change, n_rounds = 0.0, 0
    for n_rounds in range(1, max_rounds + 1):
        max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            max_change = max(max_change, _dense_sweep(t, friction, orthant, neg_two, eight, four))
        if max_change <= epsilon:
            break
    return t, n_rounds, max_change


def test_block_skipping_solve_is_bit_identical_to_dense_sweeps() -> None:
    """Block skipping must match dense sweeps exactly (t, rounds, max_change), any block size."""
    print("=== solve_eikonal_dense: block skipping bit-identical to plain dense sweeps ===")
    production_block_size = eikonal.SWEEP_BLOCK_SIZE
    try:
        for grid_seed in (2, 3):
            friction, sr, sc, sv = _winding_test_grid(grid_seed)
            for epsilon in (0.0, 0.05):
                t_ref, rounds_ref, mc_ref = _reference_dense_solve(friction, sr, sc, sv, epsilon, 500)
                for block in (1, 3, 8, 16, production_block_size, 200):
                    eikonal.SWEEP_BLOCK_SIZE = block
                    t_blk, diag = solve_eikonal_dense(friction, sr, sc, sv, epsilon, max_rounds=500,
                                                      return_diagnostics=True)
                    assert np.array_equal(t_ref, t_blk), (
                        f"grid {grid_seed}, eps={epsilon}, block={block}: t differs, "
                        f"max diff {np.abs(t_ref - t_blk).max()}"
                    )
                    assert diag["n_rounds_used"] == rounds_ref and diag["max_change"] == mc_ref, (
                        f"grid {grid_seed}, eps={epsilon}, block={block}: rounds/max_change "
                        f"{diag['n_rounds_used']}/{diag['max_change']} != {rounds_ref}/{mc_ref}"
                    )
                    if grid_seed == 2 and epsilon == 0.0:
                        print(f"  block={block:3d}: identical ({rounds_ref} rounds, "
                              f"{100 * diag['frac_blocks_swept']:.0f}% of block visits swept)")
    finally:
        eikonal.SWEEP_BLOCK_SIZE = production_block_size
    print("PASS")
    print()


def main() -> None:
    test_orthant_order_is_julias_real_gray_code_order()
    test_sweep_order_actually_changes_the_result()
    test_dense_sweep_matches_julia_loop_order_bit_for_bit()
    test_block_skipping_solve_is_bit_identical_to_dense_sweeps()
    test_update_uses_the_callers_own_dtype_not_promoted()
    test_unseeded_cells_default_to_never_flooded_sentinel_not_sea_level()
    print("All eikonal kernel validation checks passed.")


if __name__ == "__main__":
    main()
