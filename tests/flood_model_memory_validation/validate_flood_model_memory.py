"""Two real, confirmed memory/convergence regressions in flood_model.py,
found during 2026-08 obstacle-coupling calibration on large real tiles:
  - `mask` was cast to int64 by default (8x more memory than the handful of
    small integer codes it ever holds needed) - fixed via an explicit
    `.astype(np.int8)` cast. Not observable from `flood_depth_dense`'s own
    return value (mask is a local variable), so this is a source-text
    check, not a runtime one - see the test's own docstring for why that's
    still a real, effective regression guard for this specific case.
  - the round-based eikonal solve (the default since 2026-08, replacing a
    fixed 3-sweep budget) must actually CONVERGE within max_rounds for a
    real, moderately complex domain - previously only checked by two
    calibration scripts (test_obstacle_coupling_calibration.py,
    test_sweep_budget_calibration.py) that need real tile data unavailable
    on this machine (D:/GFM) and, more importantly, only ever printed
    diagnostics rather than asserting anything.

Usage:
    python validate_flood_model_memory.py
"""

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import flood_model as fm  # noqa: E402
from eikonal import solve_eikonal_dense  # noqa: E402


def test_mask_is_cast_to_int8_not_left_at_default_dtype() -> None:
    print("=== flood_depth_dense: mask is explicitly cast to int8 (not left at 8x-larger default) ===")
    source = inspect.getsource(fm.flood_depth_dense)
    assert "mask.astype(np.int8)" in source, (
        "flood_depth_dense no longer casts mask to int8 - this is exactly the regression that "
        "caused real OOM failures on large tiles during 2026-08 obstacle-coupling calibration "
        "(mask defaulting to int64 is 8x larger than the handful of small codes it ever holds needs)."
    )
    print("PASS: 'mask.astype(np.int8)' present in flood_depth_dense's own source")
    print()


def test_round_based_solve_converges_within_max_rounds_on_a_real_shaped_domain() -> None:
    """A moderately complex synthetic domain (varied friction, seeded along
    one whole edge, non-square/non-trivial shape) - the round-based solve
    (the production default since 2026-08, replacing a fixed 3-sweep
    budget) must both actually CONVERGE (max_change <= epsilon) within
    max_rounds, and take noticeably more than 1 round (a test that
    converges trivially in round 1 wouldn't exercise the round-based logic
    at all - previously only checked informally by two calibration scripts
    that print diagnostics but assert nothing, against real tile data
    unavailable on this machine)."""
    print("=== solve_eikonal_dense: round-based mode converges within max_rounds on a non-trivial domain ===")
    rng = np.random.default_rng(1)
    m, n = 30, 40
    friction = rng.uniform(0.5, 3.0, (m, n)).astype(np.float32)

    seed_rows = np.arange(m, dtype=np.int64)  # whole left edge seeded (vertex row 0..m-1, col 0)
    seed_cols = np.zeros(m, dtype=np.int64)
    seed_values = rng.uniform(1.0, 2.0, m).astype(np.float32)

    t, diagnostics = solve_eikonal_dense(
        friction, seed_rows, seed_cols, -seed_values, epsilon=0.03,
        max_rounds=200, return_diagnostics=True,
    )

    assert diagnostics["converged"] is True, (
        f"round-based solve did not converge within 200 rounds on a {m}x{n} synthetic domain "
        f"(final max_change={diagnostics['max_change']:.6g}) - either this domain got harder to "
        f"solve, or something regressed in the round-based convergence logic."
    )
    print(f"  n_rounds_used={diagnostics['n_rounds_used']}, max_change={diagnostics['max_change']:.6g}")
    assert diagnostics["n_rounds_used"] > 1, (
        "converged in a single round - domain too trivial to actually exercise round-based logic, "
        "adjust the synthetic grid/seeding to be harder."
    )
    print(f"PASS: converged in {diagnostics['n_rounds_used']} rounds (>1, not trivial), "
          f"final max_change={diagnostics['max_change']:.6g} <= epsilon")
    print()


def main() -> None:
    test_mask_is_cast_to_int8_not_left_at_default_dtype()
    test_round_based_solve_converges_within_max_rounds_on_a_real_shaped_domain()
    print("All flood_model.py memory/convergence validation checks passed.")


if __name__ == "__main__":
    main()
