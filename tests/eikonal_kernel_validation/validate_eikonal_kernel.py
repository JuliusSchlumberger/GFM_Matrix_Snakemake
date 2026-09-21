"""Direct unit tests for eikonal.py's own numerical kernel
(`_ORTHANT_ORDER`, `_update`, `solve_eikonal_dense`'s unseeded-cell
default) - previously only exercised indirectly through
`flood_depth_dense`'s tiny synthetic seed-path test and two non-asserting
calibration scripts. Each test here locks in one specific, real, already-
fixed bug documented in eikonal.py's own comments.

Usage:
    python validate_eikonal_kernel.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eikonal import _ORTHANT_ORDER, _dense_sweep, _update, solve_eikonal_dense  # noqa: E402


def test_orthant_order_is_julias_real_gray_code_order() -> None:
    """Locks in (1, 4, 3, 2), not the numerically-obvious (1, 2, 3, 4) -
    confirmed bit-for-bit against a live Julia run (module docstring): the
    naive order only matched 14-29% of cells by sweep 2/3 on a real tile."""
    print("=== _ORTHANT_ORDER: matches Julia's real Gray-code sweep order ===")
    assert _ORTHANT_ORDER == (1, 4, 3, 2), _ORTHANT_ORDER
    print(f"PASS: _ORTHANT_ORDER == {_ORTHANT_ORDER}")
    print()


def test_sweep_order_actually_changes_the_result() -> None:
    """If sweep order didn't matter, locking in a specific tuple above would
    be theatre - this proves a real, non-square friction grid genuinely
    produces a DIFFERENT result under the naive (1,2,3,4) order than under
    the real (1,4,3,2) order after a fixed, small sweep budget (before full
    convergence, where order-dependence is real and visible)."""
    print("=== sweep order: naive (1,2,3,4) vs real (1,4,3,2) diverge before convergence ===")
    rng = np.random.default_rng(0)
    m, n = 9, 13  # non-square, asymmetric - avoids accidental order-invariance from symmetry
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

    n_sweeps = 2  # deliberately under-converged - full convergence can mask order effects
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
    """A real, confirmed bug: this port originally computed the quadratic
    discriminant in effective float64 precision (Julia/numpy value-based
    promotion), which does NOT match Julia's actual float32 rounding at
    real friction scales, where the discriminant is a ~1e-5-magnitude
    catastrophic-cancellation result dominated by float32 rounding noise.

    This constructs a genuine near-cancellation case (t_a, t_b close
    together relative to their own magnitude, small v - the same regime
    that produced the documented Δ~1.5e-5 in Julia's own live trace) and
    confirms _update's result MEASURABLY DIFFERS depending on whether it's
    called with float32 or float64 arguments/constants - i.e. it genuinely
    computes in whatever dtype it's given, rather than silently promoting
    everything to float64 internally (which would make the two calls
    return the same answer)."""
    print("=== _update: computes natively in the caller's own dtype (float32-sensitive) ===")
    # Numerically searched near-cancellation regime (magnitudes matching the
    # docstring's own documented Julia trace, cand~-3.51): both float32 and
    # float64 calls take the quadratic (non-fallback) branch here, so the
    # comparison below isolates precision itself, not a branch difference.
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
    """Real bug fixed 2026-08: unseeded/unreached cells used to default to
    t=0 (waterlevel=0, i.e. "flooded at exactly sea level"), which silently
    marked any below-sea-level cell no real seed's influence ever reached
    as flooded. Fixed to t=+99 (waterlevel=-99, comfortably below any real
    DEM value), so an unreached cell correctly reads as "never flooded"
    even where its own elevation is negative.

    Constructs a friction grid with an impassable high-friction wall
    splitting it into two regions, seeds only one side, and confirms the
    OTHER side's own eikonal result never triggers `waterlevel > dem` for a
    realistic negative-elevation cell there - which the old t=0 default
    would have (0 > -5)."""
    print("=== solve_eikonal_dense: unreached cells never read as 'flooded at sea level' ===")
    m, n = 5, 12
    friction = np.full((m, n), 0.5, dtype=np.float32)
    # An effectively-impassable wall down the middle column - OBSTACLE_BLOCK_FRICTION-scale,
    # mirrors flood_model.py's own real use of a very high friction value to block a region.
    friction[:, 6] = 9999.0

    seed_rows = np.array([0], dtype=np.int64)
    seed_cols = np.array([0], dtype=np.int64)
    seed_values = np.array([0.0], dtype=np.float32)  # sea level on the seeded (left) side

    t = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.03, max_rounds=50)
    waterlevel = -t[1:, 1:]

    far_side_waterlevel = waterlevel[:, 8:]  # well past the wall, unreachable within any real budget
    unreached_dem_m = -5.0  # a real, physically plausible below-sea-level elevation
    would_flood_under_old_bug = far_side_waterlevel > unreached_dem_m

    print(f"  far-side waterlevel range: [{far_side_waterlevel.min():.2f}, {far_side_waterlevel.max():.2f}] m")
    assert np.all(far_side_waterlevel <= -90.0), (
        f"unreached cells did not stay near the +99 sentinel (waterlevel<=-90 expected), "
        f"got range [{far_side_waterlevel.min():.2f}, {far_side_waterlevel.max():.2f}] - "
        f"the wall may not be impassable enough for this test's own epsilon/max_rounds, or the "
        f"sentinel default has changed."
    )
    assert not np.any(would_flood_under_old_bug), (
        "unreached cells would incorrectly read as flooded against a real negative-elevation DEM cell - "
        "this is exactly the old t=0 default bug."
    )
    print(f"PASS: unreached region stays near the +99 sentinel (never reads as flooded against "
          f"a {unreached_dem_m}m DEM cell) - old t=0 default would have wrongly flooded it")
    print()


def main() -> None:
    test_orthant_order_is_julias_real_gray_code_order()
    test_sweep_order_actually_changes_the_result()
    test_update_uses_the_callers_own_dtype_not_promoted()
    test_unseeded_cells_default_to_never_flooded_sentinel_not_sea_level()
    print("All eikonal kernel validation checks passed.")


if __name__ == "__main__":
    main()
