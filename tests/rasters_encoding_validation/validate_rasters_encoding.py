"""Test rasters.py encoding, DEM gap fill, mask fixes and average pooling.

Below-int16 DEMs must clip (tile 1464 Dead Sea -369.29 m, 303 Danakil -379.08 m).
average_pool_to_grid needs two reproject() calls: one multi-band call shares one
nodata mask across bands under Resampling.average and corrupted ~360,000 cells.
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
    relabel_small_isolated_ocean_patches,
    resolve_offshore_mask_gaps_via_gebco,
    save_waterdepth_raster,
)


def test_encode_dem_cm_clips_below_int16_range_instead_of_wrapping() -> None:
    print("=== encode_dem_cm: clips out-of-range elevations (real Dead Sea/Danakil tiles) instead of wrapping ===")
    dem_m = np.array([[-369.29, -379.08], [10.0, 327.67]], dtype=np.float64)
    encoded = encode_dem_cm(dem_m)

    assert encoded.dtype == np.int16, encoded.dtype
    lo_int = np.iinfo(np.int16).min  # -327.68 m
    assert encoded[0, 0] == lo_int, f"expected clip to {lo_int} (-327.68m), got {encoded[0, 0]} - possible silent int16 wraparound"
    assert encoded[0, 1] == lo_int, f"expected clip to {lo_int} (-327.68m), got {encoded[0, 1]} - possible silent int16 wraparound"
    assert encoded[1, 0] == 1000, encoded[1, 0]
    assert encoded[1, 1] == np.iinfo(np.int16).max, encoded[1, 1]
    print(f"PASS: -369.29m/-379.08m both clip to {lo_int} (-327.68m), not wrapped; in-range values untouched")
    print()


def test_decode_friction_int16_preserves_float32_no_silent_promotion() -> None:
    print("=== decode_friction_int16: output stays float32, no silent float64 promotion ===")
    friction_int16 = np.array([[100, 500], [1000, 5000]], dtype=np.int16)
    decoded = decode_friction_int16(friction_int16)

    assert decoded.dtype == np.float32, (
        f"decode_friction_int16 returned {decoded.dtype}, expected float32 - float64 here "
        f"doubles peak memory of every flood solve (solve_eikonal_dense inherits "
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
    dem_vals[2, 2] = nodata
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
    dem_vals[1:5, 1:5] = nodata  # 16 cells >= min_hard_fill_component_size
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
    """With no valid cell, fillnodata has no source: hard-fill instead of leaking -9999."""
    print("=== _compute_dem_gap_fill: all-nodata window never leaks the raw nodata sentinel ===")
    nodata = -9999.0
    dem_vals = np.full((4, 4), nodata, dtype=np.float32)
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


def test_resolve_offshore_mask_gaps_via_gebco_connected_negative_becomes_ocean() -> None:
    print("=== resolve_offshore_mask_gaps_via_gebco: nodata cell 4-connected to real ocean + negative GEBCO -> ocean ===")
    mask_vals = np.array([[1, 255, 255, 0]], dtype=np.uint8)
    gebco_vals = np.array([[-50.0, -40.0, -30.0, 5.0]], dtype=np.float32)

    corrected, stats = resolve_offshore_mask_gaps_via_gebco(mask_vals, 255, gebco_vals, -32767.0)

    assert not np.any(corrected == 255), corrected
    assert corrected[0, 1] == 1 and corrected[0, 2] == 1, (
        f"nodata cells chained to real ocean with negative GEBCO should resolve to ocean, got {corrected}"
    )
    assert stats["n_resolved_ocean"] == 2, stats
    print(f"PASS: corrected={corrected.ravel()}, stats={stats}")
    print()


def test_resolve_offshore_mask_gaps_via_gebco_isolated_basin_stays_land() -> None:
    print("=== resolve_offshore_mask_gaps_via_gebco: isolated below-sea-level basin (Dead Sea/Danakil analog) stays land ===")
    # Land at col 1 cuts the nodata depression off from the ocean.
    mask_vals = np.array([[1, 0, 255, 0]], dtype=np.uint8)
    gebco_vals = np.array([[-50.0, 5.0, -400.0, 10.0]], dtype=np.float32)

    corrected, stats = resolve_offshore_mask_gaps_via_gebco(mask_vals, 255, gebco_vals, -32767.0)

    assert corrected[0, 2] == 0, (
        f"isolated inland nodata cell (no path to real ocean) must stay land despite negative GEBCO, got {corrected}"
    )
    assert stats["n_resolved_ocean"] == 0, stats
    assert stats["n_still_land"] == 1, stats
    print(f"PASS: corrected={corrected.ravel()}, stats={stats}")
    print()


def test_resolve_offshore_mask_gaps_via_gebco_positive_gebco_stays_land() -> None:
    print("=== resolve_offshore_mask_gaps_via_gebco: nodata cell connected to ocean but non-negative/nodata GEBCO -> stays land ===")
    mask_vals = np.array([[1, 255, 255]], dtype=np.uint8)
    gebco_vals = np.array([[-50.0, 5.0, -32767.0]], dtype=np.float32)  # positive, then GEBCO nodata

    corrected, stats = resolve_offshore_mask_gaps_via_gebco(mask_vals, 255, gebco_vals, -32767.0)

    assert corrected[0, 1] == 0 and corrected[0, 2] == 0, (
        f"nodata cells with non-negative or missing GEBCO must stay land regardless of ocean connectivity, got {corrected}"
    )
    assert stats["n_resolved_ocean"] == 0, stats
    print(f"PASS: corrected={corrected.ravel()}, stats={stats}")
    print()


def test_relabel_small_isolated_ocean_patches_relabels_small_component() -> None:
    print("=== relabel_small_isolated_ocean_patches: small isolated ocean blob -> lake ===")
    mask_vals = np.zeros((5, 5), dtype=np.uint8)
    mask_vals[1:3, 1:3] = 1

    corrected, stats = relabel_small_isolated_ocean_patches(mask_vals, min_cells=30)

    assert np.all(corrected[1:3, 1:3] == 2), f"small ocean-coded blob should become lake, got {corrected}"
    assert not np.any(corrected == 1), corrected
    assert stats == {"n_relabeled": 4, "n_components_relabeled": 1}, stats
    print(f"PASS: corrected=\n{corrected}, stats={stats}")
    print()


def test_relabel_small_isolated_ocean_patches_keeps_large_component() -> None:
    print("=== relabel_small_isolated_ocean_patches: large ocean body stays ocean, small one still relabeled ===")
    mask_vals = np.zeros((6, 10), dtype=np.uint8)
    mask_vals[:, 0:6] = 1
    mask_vals[4:6, 8:10] = 1

    corrected, stats = relabel_small_isolated_ocean_patches(mask_vals, min_cells=30)

    assert np.all(corrected[:, 0:6] == 1), f"large ocean body must stay ocean, got {corrected}"
    assert np.all(corrected[4:6, 8:10] == 2), f"small isolated blob should still become lake, got {corrected}"
    assert stats == {"n_relabeled": 4, "n_components_relabeled": 1}, stats
    print(f"PASS: corrected=\n{corrected}, stats={stats}")
    print()


def test_save_waterdepth_raster_creates_missing_output_directory() -> None:
    """run_eikonal_cli.py, unlike Snakemake, does not create results/ beforehand."""
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
        out_path = tmpdir / "results" / "nested" / "waterdepth_RP100_SLR_0.tif"
        assert not out_path.parent.exists()

        save_waterdepth_raster(ref_path, waterdepth, out_path)
        assert out_path.exists(), "save_waterdepth_raster did not create its own output file/directory"
        print(f"PASS: wrote {out_path} despite its parent directory not existing beforehand")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_average_pool_to_grid_closed_form_correctness() -> None:
    """Pool to sum(class)/n_total, not /n_in_domain; a merged multi-band reproject breaks it."""
    print("=== average_pool_to_grid: closed-form correctness on a hand-computable case ===")
    transform = Affine(1.0, 0, 0.0, 0, -1.0, 4.0)
    # Top-left block: 3 of 4 cells in domain, 2 are class -> 2/4, not 2/3.
    domain = np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.float32)
    numerator = np.array([
        [1, 1, np.nan, np.nan],
        [0, np.nan, np.nan, np.nan],  # in-domain, not class: 0.0
        [np.nan, np.nan, 1, 1],
        [np.nan, np.nan, 1, 1],
    ], dtype=np.float32)  # nan = outside domain

    dst_transform = Affine(2.0, 0, 0.0, 0, -2.0, 4.0)

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
    test_resolve_offshore_mask_gaps_via_gebco_connected_negative_becomes_ocean()
    test_resolve_offshore_mask_gaps_via_gebco_isolated_basin_stays_land()
    test_resolve_offshore_mask_gaps_via_gebco_positive_gebco_stays_land()
    test_relabel_small_isolated_ocean_patches_relabels_small_component()
    test_relabel_small_isolated_ocean_patches_keeps_large_component()
    test_save_waterdepth_raster_creates_missing_output_directory()
    test_average_pool_to_grid_closed_form_correctness()
    print("All rasters.py encoding/gap-fill validation checks passed.")


if __name__ == "__main__":
    main()
