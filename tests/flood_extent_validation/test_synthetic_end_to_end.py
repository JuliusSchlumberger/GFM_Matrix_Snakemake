"""Synthetic end-to-end test for validation.py against analytically known,
closed-form expected values - proves the supersampled rasterisation and the
area-weighted population disaggregation are exact, not just "close" (plan
doc §6.2). No real production data needed - every geometry below is chosen
to align exactly with pixel/subpixel boundaries so the expected values are
exact fractions, not approximations.

Usage:
    python tests/flood_extent_validation/test_synthetic_end_to_end.py
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box as shapely_box

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from plotting import pixel_area_km2_grid  # noqa: E402
from validation import (  # noqa: E402
    benchmark_fraction_from_vector,
    confusion_counts,
    metrics_from_counts,
    model_domain_mask,
    population_by_class,
    wet_mask_from_fraction,
)

_FAILURES: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_fraction_full_cell_aligned() -> None:
    """A polygon covering exactly the left column of a 2x2 grid, boundary-
    aligned to whole cells - every fraction must be exactly 0.0 or 1.0,
    regardless of supersample factor (no partial coverage anywhere)."""
    print("test_fraction_full_cell_aligned")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)  # 2x2 cells, lon[0,2] x lat[0,2]
    poly = shapely_box(0.0, 0.0, 1.0, 2.0)  # left column, both rows, full height
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
    """A polygon covering exactly the left HALF of one cell, with the
    polygon boundary landing exactly on a subpixel boundary at the given
    supersample factor - fraction must be exactly 0.5, not an approximation."""
    print("test_fraction_partial_cell_supersample_aligned")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)  # single 1x1 cell, lon[0,1] x lat[0,1]
    poly = shapely_box(0.0, 0.0, 0.5, 1.0)  # left half of the cell
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs=4326)

    for supersample in (2, 4, 10):  # 0.5 lands exactly on a subpixel edge for all of these
        fraction = benchmark_fraction_from_vector(gdf, transform, 4326, (1, 1), supersample=supersample)
        _check(f"supersample={supersample}: half-cell fraction == 0.5 exactly",
               _close(fraction[0, 0], 0.5), f"got {fraction[0, 0]}")


def test_confusion_and_metrics_closed_form() -> None:
    """4x4 grid: model wet = left two columns, benchmark wet = top two rows.
    TP/FP/FN/TN each cover exactly a 2x2 quadrant -> closed-form metrics."""
    print("test_confusion_and_metrics_closed_form")
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2       # left half
    benchmark_wet = rows < 2   # top half
    domain = np.ones((4, 4), dtype=bool)
    weight = np.ones((4, 4), dtype=float)  # plain cell counts

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


def test_area_weighted_matches_pixel_area_grid() -> None:
    """Same 4x4 scenario, but weighted by real km2 area near the equator
    (cos(lat) ~ 1, so area-weighted and cell-count-weighted metrics must
    come out numerically close - not a coincidence, a consequence of the
    near-equator latitude choice) - proves confusion_counts really applies
    the given weight array elementwise, not e.g. per-row only."""
    print("test_area_weighted_matches_pixel_area_grid")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)  # lat [-2, 2] - straddles the equator
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    benchmark_wet = rows < 2
    domain = np.ones((4, 4), dtype=bool)
    area_km2 = pixel_area_km2_grid(transform, 4, 4)

    tp, fp, fn, tn = confusion_counts(model_wet, benchmark_wet, domain, area_km2)
    expected_quadrant_km2 = float(area_km2[:2, :2].sum())  # top-left quadrant's own real area
    _check("tp matches the top-left quadrant's own summed area exactly",
           _close(tp, expected_quadrant_km2), f"got {tp} vs {expected_quadrant_km2}")
    # Near the equator every quadrant's area should be within ~0.1% of the others
    quadrants = [area_km2[:2, :2].sum(), area_km2[:2, 2:].sum(), area_km2[2:, :2].sum(), area_km2[2:, 2:].sum()]
    _check("all four near-equator quadrants have nearly equal area (cos(lat) ~ 1)",
           max(quadrants) / min(quadrants) < 1.001, f"got ratios {quadrants}")


def test_population_disaggregation_closed_form() -> None:
    """One coarse population cell (value=1000) exactly covering a 4x4 fine
    grid; class_mask = left half (cols < 2) of the fine grid, fully inside
    the domain. The disaggregated population in that class must be exactly
    half of 1000 - the plan §4.5 identity in its simplest possible case."""
    print("test_population_disaggregation_closed_form")
    fine_transform = Affine(0.25, 0, 0.0, 0, -0.25, 1.0)  # 4x4 fine cells over lon/lat [0,1]
    class_mask = np.zeros((4, 4), dtype=bool)
    class_mask[:, :2] = True  # left half
    domain_mask = np.ones((4, 4), dtype=bool)

    pop_transform = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)  # single 1x1 coarse cell, same extent
    population = np.array([[1000.0]])

    result = population_by_class(
        class_mask, domain_mask, fine_transform, "EPSG:4326",
        population, pop_transform, "EPSG:4326",
    )
    _check("disaggregated population == 500.0 (exactly half of 1000)",
           _close(float(result[0, 0]), 500.0, tol=1e-6), f"got {result[0, 0]}")

    # Sanity: the complementary class (right half) must account for the rest.
    complement_mask = ~class_mask
    result_complement = population_by_class(
        complement_mask, domain_mask, fine_transform, "EPSG:4326",
        population, pop_transform, "EPSG:4326",
    )
    total = float(result[0, 0]) + float(result_complement[0, 0])
    _check("class + complement == full population (500 + 500 == 1000)",
           _close(total, 1000.0, tol=1e-6), f"got {total}")


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


def main() -> None:
    test_fraction_full_cell_aligned()
    test_fraction_partial_cell_supersample_aligned()
    test_confusion_and_metrics_closed_form()
    test_area_weighted_matches_pixel_area_grid()
    test_population_disaggregation_closed_form()
    test_wet_mask_threshold()
    test_model_domain_mask()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
