"""Test validation.py rasterisation and (soft) scoring against closed-form values.

Geometries align with pixel/subpixel edges, so expected fractions are exact.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box as shapely_box

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from validation import (  # noqa: E402
    benchmark_fraction_from_vector,
    confusion_counts,
    confusion_counts_soft,
    metrics_from_counts,
    model_domain_mask,
    wet_mask_from_fraction,
)

def _check(name: str, condition: bool, detail: str = "") -> None:
    """Print a PASS/FAIL line and assert, so pytest fails on a wrong value."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    assert condition, name + (f" - {detail}" if detail else "")


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_fraction_full_cell_aligned() -> None:
    print("test_fraction_full_cell_aligned")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)
    poly = shapely_box(0.0, 0.0, 1.0, 2.0)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs=4326)

    for supersample in (1, 4, 7):
        fraction = benchmark_fraction_from_vector(gdf, transform, 4326, (2, 2), supersample=supersample)
        _check(f"supersample={supersample}: left column fraction == 1.0",
               _close(fraction[0, 0], 1.0) and _close(fraction[1, 0], 1.0),
               f"got {fraction[:, 0]}")
        _check(f"supersample={supersample}: right column fraction == 0.0",
               _close(fraction[0, 1], 0.0) and _close(fraction[1, 1], 0.0),
               f"got {fraction[:, 1]}")


def test_fraction_partial_cell_supersample_aligned() -> None:
    print("test_fraction_partial_cell_supersample_aligned")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)
    poly = shapely_box(0.0, 0.0, 0.5, 1.0)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs=4326)

    for supersample in (2, 4, 10):  # 0.5 is a subpixel edge for each
        fraction = benchmark_fraction_from_vector(gdf, transform, 4326, (1, 1), supersample=supersample)
        _check(f"supersample={supersample}: half-cell fraction == 0.5 exactly",
               _close(fraction[0, 0], 0.5), f"got {fraction[0, 0]}")


def test_confusion_and_metrics_closed_form() -> None:
    print("test_confusion_and_metrics_closed_form")
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    benchmark_wet = rows < 2
    domain = np.ones((4, 4), dtype=bool)
    weight = np.ones((4, 4), dtype=float)

    tp, fp, fn, tn = confusion_counts(model_wet, benchmark_wet, domain, weight)
    _check("tp == 4 (top-left quadrant)", tp == 4.0, f"got {tp}")
    _check("fp == 4 (bottom-left quadrant)", fp == 4.0, f"got {fp}")
    _check("fn == 4 (top-right quadrant)", fn == 4.0, f"got {fn}")
    _check("tn == 4 (bottom-right quadrant)", tn == 4.0, f"got {tn}")

    m = metrics_from_counts(tp, fp, fn, tn)
    _check("HR == 0.5 (4 / (4+4))", _close(m["HR"], 0.5))
    _check("FAR == 0.5 (4 / (4+4))", _close(m["FAR"], 0.5))
    _check("CSI == 1/3 (4 / (4+4+4))", _close(m["CSI"], 1.0 / 3.0))
    _check("EB == 0.5 (FP == FN)", m["EB"] == 0.5)
    _check("bias == 1.0 ((4+4)/(4+4))", _close(m["bias"], 1.0))


def test_confusion_soft_reduces_to_hard_on_binary_fraction() -> None:
    """Soft scoring must equal hard scoring exactly on a pure 0/1 fraction."""
    print("test_confusion_soft_reduces_to_hard_on_binary_fraction")
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    benchmark_wet = rows < 2
    fraction = benchmark_wet.astype("float64")
    domain = np.ones((4, 4), dtype=bool)
    weight = np.ones((4, 4), dtype=float)

    tp_hard, fp_hard, fn_hard, tn_hard = confusion_counts(model_wet, benchmark_wet, domain, weight)
    tp_soft, fp_soft, fn_soft, tn_soft = confusion_counts_soft(model_wet, fraction, domain, weight)
    _check("tp: soft == hard on binary fraction", _close(tp_soft, tp_hard), f"{tp_soft} vs {tp_hard}")
    _check("fp: soft == hard on binary fraction", _close(fp_soft, fp_hard), f"{fp_soft} vs {fp_hard}")
    _check("fn: soft == hard on binary fraction", _close(fn_soft, fn_hard), f"{fn_soft} vs {fn_hard}")
    _check("tn: soft == hard on binary fraction", _close(tn_soft, tn_hard), f"{tn_soft} vs {tn_hard}")


def test_confusion_soft_partial_coverage_closed_form() -> None:
    """Wet cells split weight f tp + (1-f) fp; dry cells f fn + (1-f) tn."""
    print("test_confusion_soft_partial_coverage_closed_form")
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    fraction = np.where(model_wet, 0.25, 0.75)
    domain = np.ones((4, 4), dtype=bool)
    weight = np.ones((4, 4), dtype=float)

    tp, fp, fn, tn = confusion_counts_soft(model_wet, fraction, domain, weight)
    _check("tp == 2.0 (8 model-wet cells x 0.25)", _close(tp, 2.0), f"got {tp}")
    _check("fp == 6.0 (8 model-wet cells x 0.75)", _close(fp, 6.0), f"got {fp}")
    _check("fn == 6.0 (8 model-dry cells x 0.75)", _close(fn, 6.0), f"got {fn}")
    _check("tn == 2.0 (8 model-dry cells x 0.25)", _close(tn, 2.0), f"got {tn}")
    _check("tp + fp == 8 (every model-wet cell's weight is fully accounted for)",
           _close(tp + fp, 8.0), f"got {tp + fp}")
    _check("fn + tn == 8 (every model-dry cell's weight is fully accounted for)",
           _close(fn + tn, 8.0), f"got {fn + tn}")

    m = metrics_from_counts(tp, fp, fn, tn)
    _check("HR == 0.25 (2 / (2+6))", _close(m["HR"], 0.25), f"got {m['HR']}")
    _check("FAR == 0.75 (6 / (2+6))", _close(m["FAR"], 0.75), f"got {m['FAR']}")
    _check("CSI == 1/7 (2 / (2+6+6))", _close(m["CSI"], 1.0 / 7.0), f"got {m['CSI']}")


def test_area_weighted_matches_pixel_area_grid() -> None:
    """Weight by km2 pixel area near the equator (cos(lat) ~ 1, quadrants ~equal)."""
    print("test_area_weighted_matches_pixel_area_grid")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)  # lat [-2, 2]
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    benchmark_wet = rows < 2
    domain = np.ones((4, 4), dtype=bool)
    area_km2 = pixel_area_km2_grid(transform, 4, 4)

    tp, fp, fn, tn = confusion_counts(model_wet, benchmark_wet, domain, area_km2)
    expected_quadrant_km2 = float(area_km2[:2, :2].sum())
    _check("tp matches the top-left quadrant's own summed area exactly",
           _close(tp, expected_quadrant_km2), f"got {tp} vs {expected_quadrant_km2}")
    quadrants = [area_km2[:2, :2].sum(), area_km2[:2, 2:].sum(), area_km2[2:, :2].sum(), area_km2[2:, 2:].sum()]
    _check("all four near-equator quadrants have nearly equal area (cos(lat) ~ 1)",
           max(quadrants) / min(quadrants) < 1.001, f"got ratios {quadrants}")


def test_wet_mask_threshold() -> None:
    print("test_wet_mask_threshold")
    fraction = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    wet = wet_mask_from_fraction(fraction, wet_fraction=0.5)
    _check("0.5 rule: >= 0.5 is wet, < 0.5 is dry",
           list(wet) == [False, False, True, True, True], f"got {wet}")


def test_model_domain_mask() -> None:
    print("test_model_domain_mask")
    depth = np.array([-9999.0, 0.0, 0.3, -9999.0])
    domain = model_domain_mask(depth, nodata=-9999.0)
    _check("nodata cells excluded, everything else (incl. dry=0.0) included",
           list(domain) == [False, True, True, False], f"got {domain}")


_TESTS = (
    test_fraction_full_cell_aligned,
    test_fraction_partial_cell_supersample_aligned,
    test_confusion_and_metrics_closed_form,
    test_confusion_soft_reduces_to_hard_on_binary_fraction,
    test_confusion_soft_partial_coverage_closed_form,
    test_area_weighted_matches_pixel_area_grid,
    test_wet_mask_threshold,
    test_model_domain_mask,
)


def main() -> None:
    """Run every test in script mode; a failed check ends only its own test."""
    failures: list[str] = []
    for test in _TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s): {'; '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
