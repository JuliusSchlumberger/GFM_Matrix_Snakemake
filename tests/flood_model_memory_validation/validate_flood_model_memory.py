"""Guard flood_model memory (int8 mask) and round-based solve convergence.

The int8 cast is a source-text check: mask is local to flood_depth_dense, so
its dtype is not observable from the return value. int64 caused OOM on large tiles.
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
        "flood_depth_dense does not cast mask to int8: a default int64 mask is 8x larger than "
        "the handful of small codes it holds needs and runs large tiles out of memory."
    )
    print("PASS: 'mask.astype(np.int8)' present in flood_depth_dense's own source")
    print()


def test_round_based_solve_converges_in_more_than_one_round() -> None:
    """Converge (max_change <= epsilon) within max_rounds, using more than 1 round."""
    print("=== solve_eikonal_dense: round-based mode converges within max_rounds on a non-trivial domain ===")
    rng = np.random.default_rng(1)
    m, n = 30, 40
    friction = rng.uniform(0.5, 3.0, (m, n)).astype(np.float32)

    seed_rows = np.arange(m, dtype=np.int64)
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
    test_round_based_solve_converges_in_more_than_one_round()
    print("All flood_model.py memory/convergence validation checks passed.")


if __name__ == "__main__":
    main()
