"""Direct unit tests for rasters.py's real, documented historical bugs:
  - encode_dem_cm's deliberate clip (not raise) for below-int16-range
    elevations (real tiles: 1464 Dead Sea -369.29m, 303 Danakil -379.08m).
  - decode_friction_int16's float32-preserving divide (numpy's legacy
    value-based casting silently promoted `float32_array / python_int` to
    float64 when the int needs >16 bits - doubled peak memory of every
    flood solve until caught during OOM debugging).
  - _compute_dem_gap_fill's stray-nodata routing (an all-nodata-window tile
    used to leak a raw -9999 sentinel into encode_dem_cm and crash).
  - save_waterdepth_raster creating its own output directory (broke live
    on 2026-08-10 HPC run: run_aqueduct_cli.py, unlike Snakemake, never
    got its results/ dir auto-created).
  - average_pool_to_grid's closed-form correctness (the two-separate-
    reproject-calls structure this locks in - packing numerator/domain as
    two bands of one multi-band reproject() call was tried and rejected:
    confirmed live to silently corrupt ~360,000 real coarse cells, since
    reproject() shares one nodata mask across bands under Resampling.average).

Usage:
    python validate_rasters_encoding.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rasters import (  # noqa: E402
    _compute_dem_gap_fill,
    average_pool_to_grid,
    decode_friction_int16,
    encode_dem_cm,
    save_waterdepth_raster,
)


def test_encode_dem_cm_clips_below_int16_range_instead_of_wrapping() -> None:
    print("=== encode_dem_cm: clips out-of-range elevations (real Dead Sea/Danakil tiles) instead of wrapping ===")
    dem_m = np.array([[-369.29, -379.08], [10.0, 327.67]], dtype=np.float64)  # real values from tiles 1464/303
    encoded = encode_dem_cm(dem_m)

    assert encoded.dtype == np.int16, encoded.dtype
    lo_int = np.iinfo(np.int16).min  # -32768 -> -327.68m
    # Both deep values must clip to the int16 floor, NOT wrap around to a
    # small/positive value the way a naive `.astype(np.int16)` would.
    assert encoded[0, 0] == lo_int, f"expected clip to {lo_int} (-327.68m), got {encoded[0, 0]} - possible silent int16 wraparound"
    assert encoded[0, 1] == lo_int, f"expected clip to {lo_int} (-327.68m), got {encoded[0, 1]} - possible silent int16 wraparound"
    assert encoded[1, 0] == 1000, encoded[1, 0]  # 10.0m -> 1000 cm, well within range, untouched
    assert encoded[1, 1] == np.iinfo(np.int16).max, encoded[1, 1]  # exactly at the ceiling
    print(f"PASS: -369.29m/-379.08m both clip to {lo_int} (-327.68m), not wrapped; in-range values untouched")
    print()


def test_decode_friction_int16_preserves_float32_no_silent_promotion() -> None:
    print("=== decode_friction_int16: output stays float32, no silent float64 promotion ===")
    friction_int16 = np.array([[100, 500], [1000, 5000]], dtype=np.int16)
    decoded = decode_friction_int16(friction_int16)

    assert decoded.dtype == np.float32, (
        f"decode_friction_int16 returned {decoded.dtype}, expected float32 - this is exactly the "
        f"regression that doubled peak memory of every flood solve (solve_eikonal_dense inherits "
        f"this dtype for its dominant array allocation, `t`)."
    )
    expected = friction_int16.astype(np.float64) / 1_000_000.0
    assert np.allclose(decoded, expected, atol=1e-9), decoded
    print(f"PASS: dtype={decoded.dtype}, values correct: {decoded.ravel()}")
    print()


def test_dem_gap_fill_small_gap_interpolates_from_real_neighbors() -> None:
    print("=== _compute_dem_gap_fill: a small isolated land gap interpolates from real neighbours ===")
    nodata = -9999.0
    dem_vals = np.full((5, 5), 10.0, dtype=np.float32)
    dem_vals[2, 2] = nodata  # one isolated missing land cell, surrounded by real elevation
    is_land = np.ones((5, 5), dtype=bool)

    fill = _compute_dem_gap_fill(
        dem_vals, nodata, is_land,
        min_hard_fill_component_size=10, interp_max_search_distance=100.0,
        interp_smoothing_iterations=0, land_fill_value_m=99.0,
    )
    assert np.isclose(fill[2, 2], 10.0, atol=0.5), (
        f"single isolated land gap should interpolate close to its real neighbours (~10.0m), got {fill[2, 2]}"
    )
    print(f"PASS: isolated gap interpolated to {fill[2, 2]:.2f}m (real neighbours were 10.0m)")
    print()


def test_dem_gap_fill_large_gap_hard_fills() -> None:
    print("=== _compute_dem_gap_fill: a large land gap hard-fills to land_fill_value_m ===")
    nodata = -9999.0
    dem_vals = np.full((6, 6), 10.0, dtype=np.float32)
    dem_vals[1:5, 1:5] = nodata  # a 4x4=16-cell gap, >= min_hard_fill_component_size (10)
    is_land = np.ones((6, 6), dtype=bool)

    fill = _compute_dem_gap_fill(
        dem_vals, nodata, is_land,
        min_hard_fill_component_size=10, interp_max_search_distance=100.0,
        interp_smoothing_iterations=0, land_fill_value_m=99.0,
    )
    assert np.all(fill[1:5, 1:5] == 99.0), fill[1:5, 1:5]
    print("PASS: large (16-cell) land gap hard-filled to 99.0m, not interpolated")
    print()


def test_dem_gap_fill_all_nodata_window_never_leaks_raw_nodata() -> None:
    """Real bug fixed 2026-08: a ~3K-pixel tile with ZERO valid DEM coverage
    in its read window crashed encode_dem_cm with a stray raw -9999 -
    fillnodata has nothing to interpolate FROM when every cell is missing,
    so those cells must fall through to the hard land_fill_value_m instead
    of being left as the raw nodata sentinel."""
    print("=== _compute_dem_gap_fill: all-nodata window never leaks the raw nodata sentinel ===")
    nodata = -9999.0
    dem_vals = np.full((4, 4), nodata, dtype=np.float32)  # zero valid coverage at all
    is_land = np.ones((4, 4), dtype=bool)

    fill = _compute_dem_gap_fill(
        dem_vals, nodata, is_land,
        min_hard_fill_component_size=10, interp_max_search_distance=100.0,
        interp_smoothing_iterations=0, land_fill_value_m=99.0,
    )
    assert not np.any(fill == nodata), f"raw nodata sentinel leaked into the fill result: {fill}"
    assert np.all(fill == 99.0), (
        f"all-nodata window (nothing to interpolate from) should hard-fill to land_fill_value_m, got {fill}"
    )
    print("PASS: all-nodata window hard-fills to 99.0m, no raw -9999 leak")
    print()


def test_save_waterdepth_raster_creates_missing_output_directory() -> None:
    """Real, live HPC failure (2026-08-10): run_aqueduct_cli.py (a plain
    standalone CLI, not a Snakemake rule with automatic output-dir
    creation) crashed with RasterioIOError writing waterdepth_*.tif because
    results/ had never been created for that tile."""
    print("=== save_waterdepth_raster: creates its own output directory if missing ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="waterdepth_mkdir_test_"))
    try:
        ref_path = tmpdir / "dem.tif"
        transform = Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0)
        profile = {
            "driver": "GTiff", "dtype": "int16", "count": 1,
            "height": 4, "width": 4, "transform": transform, "crs": "EPSG:4326", "nodata": -9999,
        }
        with rasterio.open(ref_path, "w", **profile) as dst:
            dst.write(np.zeros((4, 4), dtype=np.int16), 1)

        waterdepth = np.zeros((4, 4), dtype=np.float32)
        # Deliberately nested, not-yet-existing output dir - mirrors the real
        # failure ({tile_dir}/results/ never created before this write).
        out_path = tmpdir / "results" / "nested" / "waterdepth_RP100_SLR_0.tif"
        assert not out_path.parent.exists()

        save_waterdepth_raster(ref_path, waterdepth, out_path)
        assert out_path.exists(), "save_waterdepth_raster did not create its own output file/directory"
        print(f"PASS: wrote {out_path} despite its parent directory not existing beforehand")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_average_pool_to_grid_closed_form_correctness() -> None:
    """Locks in the correct A x B closed-form result (sum(class)/n_total,
    NOT sum(class)/n_in_domain) - a hand-computable, pixel-aligned 4x4 -> 2x2
    downsample. This is exactly the computation that must stay as TWO
    separate reproject() calls (see average_pool_to_grid's own docstring) -
    if a future edit merged them into one multi-band call, this closed-form
    check would silently start producing a DIFFERENT (wrong) value for any
    block containing partial domain coverage, like the top-left block below."""
    print("=== average_pool_to_grid: closed-form correctness on a hand-computable case ===")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 4.0)  # 4x4 grid, 1-unit cells
    # Top-left 2x2 block: domain covers 3 of 4 cells, 2 of those 3 are "class" ->
    # expected fraction = 2/4 = 0.5 (NOT 2/3).
    domain = np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.float32)
    numerator = np.array([
        [1, 1, np.nan, np.nan],
        [0, np.nan, np.nan, np.nan],  # in-domain but NOT class - 0.0, not nan (see contract note above)
        [np.nan, np.nan, 1, 1],
        [np.nan, np.nan, 1, 1],
    ], dtype=np.float32)  # nan marks "outside domain" per the function's own contract
    # bottom-right 2x2 block: full domain coverage, all 4 cells are class -> fraction = 1.0

    dst_transform = Affine(2.0, 0, 0.0, 0, -2.0, 4.0)  # 2x2 output, 2-unit cells

    result = average_pool_to_grid(
        numerator, domain, transform, "EPSG:32633", dst_transform, "EPSG:32633",
        dst_shape=(2, 2), numerator_nodata=np.nan,
    )

    assert np.isclose(result[0, 0], 0.5, atol=1e-4), (
        f"top-left block: expected 2/4=0.5 (sum(class)/n_TOTAL, not /n_in_domain), got {result[0, 0]}"
    )
    assert np.isclose(result[1, 1], 1.0, atol=1e-4), f"bottom-right block: expected 1.0, got {result[1, 1]}"
    print(f"PASS: top-left block={result[0, 0]:.4f} (expected 0.5), bottom-right block={result[1, 1]:.4f} (expected 1.0)")
    print()


def main() -> None:
    test_encode_dem_cm_clips_below_int16_range_instead_of_wrapping()
    test_decode_friction_int16_preserves_float32_no_silent_promotion()
    test_dem_gap_fill_small_gap_interpolates_from_real_neighbors()
    test_dem_gap_fill_large_gap_hard_fills()
    test_dem_gap_fill_all_nodata_window_never_leaks_raw_nodata()
    test_save_waterdepth_raster_creates_missing_output_directory()
    test_average_pool_to_grid_closed_form_correctness()
    print("All rasters.py encoding/gap-fill validation checks passed.")


if __name__ == "__main__":
    main()
