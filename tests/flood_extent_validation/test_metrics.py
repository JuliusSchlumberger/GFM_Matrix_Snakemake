"""Unit tests for validation.metrics_from_counts / validation.confusion_counts
against hand-computed contingency tables, including every degenerate case.

Plain assert-based script (matches this repo's existing tests/ convention,
e.g. tests/river_mouth_tile_validation/validate_river_mouth_tiles.py) - no
pytest dependency anywhere in this repo.

Usage:
    python tests/flood_extent_validation/test_metrics.py
"""

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from validation import confusion_counts, confusion_counts_tolerant, metrics_from_counts  # noqa: E402

_FAILURES: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def test_hand_computed_contingency_table() -> None:
    """The plan doc's own §3 probe numbers, as a regression fixture:
    TP=3954, FP=9382, FN=38 -> HR=0.990, FAR=0.704, CSI=0.296, EB=0.996 (FP/FN=247).
    """
    print("test_hand_computed_contingency_table")
    m = metrics_from_counts(tp=3954, fp=9382, fn=38, tn=0)
    _check("HR ~= 0.990", math.isclose(m["HR"], 0.990, abs_tol=5e-4), f"got {m['HR']}")
    _check("FAR ~= 0.704", math.isclose(m["FAR"], 0.704, abs_tol=5e-4), f"got {m['FAR']}")
    _check("CSI ~= 0.296", math.isclose(m["CSI"], 0.296, abs_tol=5e-4), f"got {m['CSI']}")
    _check("EB ~= 0.996", math.isclose(m["EB"], 0.996, abs_tol=5e-4), f"got {m['EB']}")
    _check("EB_ratio ~= 247", math.isclose(m["EB_ratio"], 247.0, abs_tol=1.0), f"got {m['EB_ratio']}")


def test_perfect_match() -> None:
    print("test_perfect_match")
    m = metrics_from_counts(tp=100, fp=0, fn=0, tn=50)
    _check("HR == 1.0", m["HR"] == 1.0)
    _check("FAR == 0.0", m["FAR"] == 0.0)
    _check("CSI == 1.0", m["CSI"] == 1.0)
    _check("EB is NaN (0/0)", _is_nan(m["EB"]), f"got {m['EB']}")
    _check("bias == 1.0", m["bias"] == 1.0)


def test_eb_exactly_half_when_fp_equals_fn() -> None:
    print("test_eb_exactly_half_when_fp_equals_fn")
    for fp_fn in (1, 10, 1234.5, 0.001):
        m = metrics_from_counts(tp=5, fp=fp_fn, fn=fp_fn, tn=5)
        _check(f"EB == 0.5 exactly for FP=FN={fp_fn}", m["EB"] == 0.5, f"got {m['EB']}")
        _check(f"EB_ratio == 1.0 exactly for FP=FN={fp_fn}", m["EB_ratio"] == 1.0, f"got {m['EB_ratio']}")


def test_no_benchmark_wet() -> None:
    """No benchmark-wet area at all in the domain: HR/CSI's TP+FN denominator is 0."""
    print("test_no_benchmark_wet")
    m = metrics_from_counts(tp=0, fp=25, fn=0, tn=100)
    _check("HR is NaN, not 0 (0/0)", _is_nan(m["HR"]), f"got {m['HR']}")
    _check("FAR == 1.0 (all model-wet is false alarm)", m["FAR"] == 1.0)
    _check("CSI == 0.0 (TP=0, denom=25>0)", m["CSI"] == 0.0)
    _check("EB == 1.0 (FP=25, FN=0)", m["EB"] == 1.0)
    _check("bias is NaN, not 0 (TP+FN=0)", _is_nan(m["bias"]), f"got {m['bias']}")


def test_no_model_wet() -> None:
    """Model predicts nothing wet at all: FAR's TP+FP denominator is 0."""
    print("test_no_model_wet")
    m = metrics_from_counts(tp=0, fp=0, fn=40, tn=100)
    _check("HR == 0.0", m["HR"] == 0.0)
    _check("FAR is NaN, not 0 (0/0)", _is_nan(m["FAR"]), f"got {m['FAR']}")
    _check("CSI == 0.0 (TP=0, denom=40>0)", m["CSI"] == 0.0)
    _check("EB == 0.0 (FP=0, FN=40)", m["EB"] == 0.0)
    _check("EB_ratio == 0.0 (FP=0, FN=40>0)", m["EB_ratio"] == 0.0)
    _check("bias == 0.0 (TP+FP=0, TP+FN=40>0)", m["bias"] == 0.0)


def test_completely_empty_domain() -> None:
    """Every count is zero - every ratio must be NaN, none may silently read as 0."""
    print("test_completely_empty_domain")
    m = metrics_from_counts(tp=0, fp=0, fn=0, tn=0)
    for key, val in m.items():
        _check(f"{key} is NaN for an all-zero contingency table", _is_nan(val), f"got {val}")


def test_confusion_counts_area_weighted() -> None:
    """A small synthetic grid with a known, hand-countable overlap - confirms
    confusion_counts' area-weighting (weight != 1 everywhere) is a real
    elementwise multiply-then-sum, not accidentally cell-counting."""
    print("test_confusion_counts_area_weighted")
    #        col: 0    1    2    3
    model = np.array([True, True, False, False])
    bench = np.array([True, False, True, False])
    domain = np.array([True, True, True, True])
    weight = np.array([2.0, 3.0, 5.0, 7.0])  # e.g. km2 per cell
    tp, fp, fn, tn = confusion_counts(model, bench, domain, weight)
    _check("tp == 2.0 (col 0: model&bench)", tp == 2.0, f"got {tp}")
    _check("fp == 3.0 (col 1: model&~bench)", fp == 3.0, f"got {fp}")
    _check("fn == 5.0 (col 2: ~model&bench)", fn == 5.0, f"got {fn}")
    _check("tn == 7.0 (col 3: ~model&~bench)", tn == 7.0, f"got {tn}")

    # Restricting the domain excludes a cell from every count, weighted or not.
    domain_partial = np.array([True, True, True, False])
    tp2, fp2, fn2, tn2 = confusion_counts(model, bench, domain_partial, weight)
    _check("tn == 0.0 when the tn cell is outside the domain", tn2 == 0.0, f"got {tn2}")
    _check("tp/fp/fn unaffected by excluding the tn cell",
           (tp2, fp2, fn2) == (2.0, 3.0, 5.0), f"got {(tp2, fp2, fn2)}")


def test_confusion_counts_tolerant_forgives_only_adjacent_disagreement() -> None:
    """One-row synthetic grid, hand-traced, for v.confusion_counts_tolerant:

        idx:        0    1    2    3    4    5    6    7    8
        model_wet:  T    T    F    F    F    F    T    F    F
        bench_wet:  F    T    T    F    F    F    F    F    T

    Strict (tolerance_cells=0): tp={1}, fp={0,6}, fn={2,8}, tn={3,4,5,7}.
    idx0's fp sits right next to idx1/2's real benchmark-wet cells - a
    textbook "same boundary, one pixel off" disagreement - while idx6's fp
    and idx8's fn each sit alone, with no opposite-type cell within one
    pixel. A correct tolerant check forgives idx0 and idx2 only, leaving
    idx6/idx8 as real (unforgiven) disagreement - confirming the mechanism
    discriminates "boundary noise" from "isolated, unexplained mismatch"
    rather than forgiving everything within reach.
    """
    print("test_confusion_counts_tolerant_forgives_only_adjacent_disagreement")
    model_wet = np.array([[True, True, False, False, False, False, True, False, False]])
    bench_wet = np.array([[False, True, True, False, False, False, False, False, True]])
    fraction = bench_wet.astype(float)
    domain = np.ones_like(model_wet, dtype=bool)
    weight = np.ones_like(model_wet, dtype=float)

    strict = confusion_counts_tolerant(model_wet, fraction, domain, weight, tolerance_cells=0)
    _check("tolerance=0: tp == 1.0 (no forgiveness yet)", strict["tp"] == 1.0, f"got {strict['tp']}")
    _check("tolerance=0: fp == 2.0 (idx0, idx6 - identical to the hard set)", strict["fp"] == 2.0, f"got {strict['fp']}")
    _check("tolerance=0: fn == 2.0 (idx2, idx8 - identical to the hard set)", strict["fn"] == 2.0, f"got {strict['fn']}")
    _check("tolerance=0: nothing forgiven", strict["fp_forgiven"] == 0.0 and strict["fn_forgiven"] == 0.0,
           f"got fp_forgiven={strict['fp_forgiven']} fn_forgiven={strict['fn_forgiven']}")

    tol = confusion_counts_tolerant(model_wet, fraction, domain, weight, tolerance_cells=1)
    _check("tolerance=1: idx0's fp is forgiven (adjacent to real bench-wet idx1)",
           tol["fp_forgiven"] == 1.0, f"got {tol['fp_forgiven']}")
    _check("tolerance=1: idx6's fp is NOT forgiven (isolated)", tol["fp"] == 1.0, f"got {tol['fp']}")
    _check("tolerance=1: idx2's fn is forgiven (adjacent to real model-wet idx1)",
           tol["fn_forgiven"] == 1.0, f"got {tol['fn_forgiven']}")
    _check("tolerance=1: idx8's fn is NOT forgiven (isolated)", tol["fn"] == 1.0, f"got {tol['fn']}")
    _check("tolerance=1: tp/tn untouched by forgiveness",
           tol["tp"] == strict["tp"] and tol["tn"] == strict["tn"],
           f"got tp={tol['tp']} tn={tol['tn']}")

    csi_strict = strict["tp"] / (strict["tp"] + strict["fp"] + strict["fn"])
    csi_tol = tol["tp"] / (tol["tp"] + tol["fp"] + tol["fn"])
    _check("CSI_tol > strict CSI (forgiveness can only help, never hurt)", csi_tol > csi_strict,
           f"got csi_strict={csi_strict}, csi_tol={csi_tol}")

    hard_disagreement = tol["fp"] + tol["fn"] + tol["fp_forgiven"] + tol["fn_forgiven"]
    _check("hard disagreement is conserved (forgiven + unforgiven == original fp+fn)",
           hard_disagreement == strict["fp"] + strict["fn"], f"got {hard_disagreement}")


def main() -> None:
    test_hand_computed_contingency_table()
    test_perfect_match()
    test_eb_exactly_half_when_fp_equals_fn()
    test_no_benchmark_wet()
    test_no_model_wet()
    test_completely_empty_domain()
    test_confusion_counts_area_weighted()
    test_confusion_counts_tolerant_forgives_only_adjacent_disagreement()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
