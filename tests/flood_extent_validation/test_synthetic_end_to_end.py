"""Synthetic end-to-end test for validation.py against analytically known,
closed-form expected values - proves the supersampled rasterisation is exact,
not just "close". No real production data needed - every geometry below is
chosen to align exactly with pixel/subpixel boundaries so the expected
values are exact fractions, not approximations.

Also covers confusion_counts_soft (the production scoring path since
2026-09-30, replacing confusion_counts/wet_mask_from_fraction for scoring -
see that function's own docstring in src/validation.py) - both a
reduces-to-the-hard-function-exactly check on binary fraction input, and a
genuinely partial (non-0/1) closed-form check of the soft credit math itself.

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
    BenchmarkSpec,
    benchmark_fraction_from_vector,
    confusion_counts,
    confusion_counts_soft,
    metrics_from_counts,
    model_domain_mask,
    read_benchmark_raster_fraction,
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


def test_confusion_soft_reduces_to_hard_on_binary_fraction() -> None:
    """confusion_counts_soft, given a `fraction` array that is already pure
    0/1 (never actually partial), must reproduce confusion_counts exactly -
    the soft formulation is a strict generalization of the hard one, not a
    different metric that happens to agree in the common case. Same 4x4
    scenario as test_confusion_and_metrics_closed_form (production's own
    before/after pair for the 2026-09-30 scoring change)."""
    print("test_confusion_soft_reduces_to_hard_on_binary_fraction")
    rows, cols = np.indices((4, 4))
    model_wet = cols < 2
    benchmark_wet = rows < 2
    fraction = benchmark_wet.astype("float64")  # pure 0/1 - no partial coverage
    domain = np.ones((4, 4), dtype=bool)
    weight = np.ones((4, 4), dtype=float)

    tp_hard, fp_hard, fn_hard, tn_hard = confusion_counts(model_wet, benchmark_wet, domain, weight)
    tp_soft, fp_soft, fn_soft, tn_soft = confusion_counts_soft(model_wet, fraction, domain, weight)
    _check("tp: soft == hard on binary fraction", _close(tp_soft, tp_hard), f"{tp_soft} vs {tp_hard}")
    _check("fp: soft == hard on binary fraction", _close(fp_soft, fp_hard), f"{fp_soft} vs {fp_hard}")
    _check("fn: soft == hard on binary fraction", _close(fn_soft, fn_hard), f"{fn_soft} vs {fn_hard}")
    _check("tn: soft == hard on binary fraction", _close(tn_soft, tn_hard), f"{tn_soft} vs {tn_hard}")


def test_confusion_soft_partial_coverage_closed_form() -> None:
    """4x4 grid, model wet = left two columns (8 cells). Benchmark coverage
    fraction is genuinely partial: 0.25 under the model-wet columns, 0.75
    under the model-dry columns - deliberately NOT 0/1, to prove the soft
    credit math itself (not just the binary-reduction case above). Every
    model-wet cell splits its own unit weight into 0.25 tp + 0.75 fp; every
    model-dry cell splits into 0.75 fn + 0.25 tn - closed-form, not an
    approximation."""
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


def test_read_benchmark_raster_fraction_gives_true_coverage() -> None:
    """read_benchmark_raster_fraction (Denmark's RasterDataset path) must
    return a real continuous 0-1 coverage fraction (Resampling.average on
    the native-resolution wet/dry mask) - NOT a degenerate 0/1 "any wet
    sub-pixel -> whole cell wet" call (the old Resampling.max behaviour,
    which silently kept Denmark on hard-threshold scoring even after
    confusion_counts_soft became the one production scoring path for every
    other benchmark - see that function's own docstring).

    4x4 native-resolution source, downsampled to 1x2 destination cells
    (left half / right half, each a cell-aligned 2x4 block of 8 native
    pixels - no partial-overlap ambiguity, so the expected fractions are
    exact): left block has exactly 2/8 wet, right block exactly 6/8 wet.
    """
    print("test_read_benchmark_raster_fraction_gives_true_coverage")
    import shutil
    import tempfile

    import hydromt
    import rasterio

    wet, dry = 0.5, 0.0  # above/below depth_threshold_m=0.1 below
    depth = np.array([
        [wet, dry, wet, wet],
        [wet, dry, wet, wet],
        [dry, dry, wet, dry],
        [dry, dry, wet, dry],
    ], dtype="float32")
    src_transform = Affine(1.0, 0, 0.0, 0, -1.0, 4.0)  # 4x4 cells, lon[0,4] x lat[0,4]

    # Manual mkdtemp + best-effort rmtree (not tempfile.TemporaryDirectory's own
    # __exit__ cleanup): rioxarray/rasterio's CachingFileManager can keep the
    # source file's OS handle open on Windows even after an eager .load(),
    # which makes TemporaryDirectory's own strict cleanup raise - a real file
    # handle, not a test bug, so just tolerate it rather than chase it.
    tmpdir = tempfile.mkdtemp()
    try:
        src_path = Path(tmpdir) / "denmark_depth.tif"
        with rasterio.open(
            src_path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="float32", crs="EPSG:4326", transform=src_transform, nodata=-9999.0,
        ) as dst:
            dst.write(depth, 1)
        da = hydromt.io.open_raster(src_path).load()

        class _StubCatalog:
            def get_rasterdataset(self, key, bbox=None):
                return da

        spec = BenchmarkSpec(
            key="denmark_test", data_type="RasterDataset", country_iso="DNK",
            hazard_type="coastal", depth_threshold_m=0.1,
        )
        dst_transform = Affine(2.0, 0, 0.0, 0, -4.0, 4.0)  # 1 row x 2 cols: left half / right half
        fraction = read_benchmark_raster_fraction(_StubCatalog(), spec, [0.0, 0.0, 4.0, 4.0], dst_transform, (1, 2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    _check("left block (2/8 wet) == 0.25, not rounded up to 1.0", _close(fraction[0, 0], 0.25),
           f"got {fraction[0, 0]}")
    _check("right block (6/8 wet) == 0.75, not rounded up to 1.0", _close(fraction[0, 1], 0.75),
           f"got {fraction[0, 1]}")
    _check("fraction is genuinely continuous (not degenerate 0/1 for every cell)",
           0.0 < fraction[0, 0] < 1.0 and 0.0 < fraction[0, 1] < 1.0, f"got {fraction}")


def main() -> None:
    test_fraction_full_cell_aligned()
    test_fraction_partial_cell_supersample_aligned()
    test_confusion_and_metrics_closed_form()
    test_confusion_soft_reduces_to_hard_on_binary_fraction()
    test_confusion_soft_partial_coverage_closed_form()
    test_area_weighted_matches_pixel_area_grid()
    test_wet_mask_threshold()
    test_model_domain_mask()
    test_read_benchmark_raster_fraction_gives_true_coverage()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
