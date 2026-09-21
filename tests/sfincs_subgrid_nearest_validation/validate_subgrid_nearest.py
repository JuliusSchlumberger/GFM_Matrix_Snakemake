"""Validate build_sfincs_tile.py's nearest-neighbour subgrid pre-reprojection
workaround (`_reproject_nearest_to_grid`) and its fail-fast subgrid
parameter validation (`_validate_subgrid_params`).

Both guard real, confirmed 2026-09 bugs:
  - hydromt_sfincs silently ignores `reproj_method` and always forces
    bilinear internally (confirmed by reading merge.py's own
    merge_multi_dataarrays) - the workaround is to pre-reproject onto the
    EXACT destination grid ourselves so hydromt's own bilinear pass becomes
    a no-op copy. Bilinear at the fine subgrid resolution fabricated real,
    confirmed spurious flooding (tile 1907, 7-9m of "flooding" that
    vanished once nearest-neighbour sourcing was used instead - see
    conversation, 2026-09).
  - hydromt_sfincs hard-enforces `nr_subgrid_pixels` to be a multiple of 2
    (components/grid/subgrid.py ~line 690) - without our own guard, an odd
    value only surfaces as a hydromt-internal error AFTER the (non-trivial)
    grid/mask/pre-reprojection steps have already run.

Usage:
    python validate_subgrid_nearest.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_sfincs_tile import _reproject_nearest_to_grid, _validate_subgrid_params  # noqa: E402


def _write_synthetic_raster(path: Path, arr: np.ndarray, transform: Affine, crs: str = "EPSG:32633") -> None:
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": arr.shape[0], "width": arr.shape[1],
        "transform": transform, "crs": crs, "nodata": np.nan, "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_reproject_nearest_is_upsample_only_no_blending() -> None:
    """Every fine-grid output pixel must be one of the coarse source's own
    real values - bilinear would instead produce new, interpolated values
    strictly between neighbouring source cells (the exact fabricated-
    intermediate-value failure mode confirmed live on tile 1907)."""
    print("=== _reproject_nearest_to_grid: no interpolated values, exact upsample ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="subgrid_nearest_test_"))
    try:
        # 3x3 coarse source, distinct real values (no two adjacent cells equal,
        # so a bilinear blend would be easy to detect as a genuinely new value).
        src_transform = Affine(120.0, 0, 500000.0, 0, -120.0, 6000000.0)
        src_arr = np.array([
            [1.0, 5.0, 9.0],
            [13.0, 17.0, 21.0],
            [25.0, 29.0, 33.0],
        ], dtype=np.float32)
        src_path = tmpdir / "src.tif"
        _write_synthetic_raster(src_path, src_arr, src_transform)

        subgrid_nr_pixels = 4  # matches production default (120m / 4 = 30m)
        fine_transform = src_transform * src_transform.scale(1.0 / subgrid_nr_pixels)
        fine_shape = (src_arr.shape[0] * subgrid_nr_pixels, src_arr.shape[1] * subgrid_nr_pixels)

        fine_arr = _reproject_nearest_to_grid(src_path, fine_transform, "EPSG:32633", fine_shape)

        assert fine_arr.shape == fine_shape, fine_arr.shape
        src_values = set(src_arr.ravel().tolist())
        fine_values = set(np.unique(fine_arr[np.isfinite(fine_arr)]).tolist())
        assert fine_values.issubset(src_values), (
            f"fine-grid output contains values not present in the source (blended, not nearest): "
            f"{fine_values - src_values}"
        )
        # Every coarse cell's own footprint must be filled entirely by that
        # cell's own value (not just SOME nearest pixel, EVERY pixel in it).
        for i in range(src_arr.shape[0]):
            for j in range(src_arr.shape[1]):
                block = fine_arr[i * subgrid_nr_pixels:(i + 1) * subgrid_nr_pixels,
                                  j * subgrid_nr_pixels:(j + 1) * subgrid_nr_pixels]
                assert np.all(block == src_arr[i, j]), f"cell ({i},{j}): block not uniformly {src_arr[i, j]}: {block}"
        print(f"PASS: {subgrid_nr_pixels}x upsample of a {src_arr.shape} grid produced only real source values, "
              f"each coarse cell's footprint uniformly filled (no blending)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_reproject_nearest_transform_matches_scaled_source() -> None:
    """The fine grid this function is asked to fill must be EXACTLY the
    source transform scaled by 1/nr_subgrid_pixels - build_sfincs_tile()'s
    own step 3 computes it this way; this just documents/locks that
    contract from the reprojection function's own side."""
    print("=== _reproject_nearest_to_grid: exact-grid contract (no partial-pixel misalignment) ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="subgrid_nearest_grid_test_"))
    try:
        src_transform = Affine(30.0, 0, 100000.0, 0, -30.0, 5000000.0)
        src_arr = np.full((2, 2), 7.0, dtype=np.float32)
        src_path = tmpdir / "src.tif"
        _write_synthetic_raster(src_path, src_arr, src_transform)

        # A transform that does NOT exactly align (quarter-pixel offset, but
        # still fully within the source's own extent) - nearest reprojection
        # must still run (no crash) but the aligned-block-purity guarantee
        # from the test above is what actually depends on exact alignment;
        # this just proves the function doesn't require it to work.
        misaligned = Affine(15.0, 0, 100000.0 + 3.0, 0, -15.0, 5000000.0 - 3.0)
        out = _reproject_nearest_to_grid(src_path, misaligned, "EPSG:32633", (4, 4))
        assert out.shape == (4, 4)
        finite = out[np.isfinite(out)]
        assert finite.size > 0 and np.all(finite == 7.0)  # uniform source -> uniform output regardless of alignment
        print("PASS: reprojection runs correctly for both aligned and misaligned target grids")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_validate_subgrid_params_rejects_odd_nr_pixels() -> None:
    print("=== _validate_subgrid_params: rejects odd nr_subgrid_pixels ===")
    with pytest.raises(ValueError, match="multiple of 2"):
        _validate_subgrid_params(120.0, 3)
    print("PASS: nr_subgrid_pixels=3 raises ValueError before any expensive work")
    print()


def test_validate_subgrid_params_rejects_non_positive() -> None:
    print("=== _validate_subgrid_params: rejects non-positive values ===")
    with pytest.raises(ValueError):
        _validate_subgrid_params(120.0, 0)
    with pytest.raises(ValueError):
        _validate_subgrid_params(120.0, -4)
    with pytest.raises(ValueError):
        _validate_subgrid_params(0.0, 4)
    with pytest.raises(ValueError):
        _validate_subgrid_params(-120.0, 4)
    print("PASS: nr_subgrid_pixels<=0 and resolution_m<=0 both rejected")
    print()


def test_validate_subgrid_params_accepts_production_defaults() -> None:
    print("=== _validate_subgrid_params: accepts the real production defaults ===")
    _validate_subgrid_params(120.0, 4)  # current production default (120m main / 30m subgrid)
    _validate_subgrid_params(90.0, 6)  # the earlier (superseded) default - still structurally valid
    print("PASS: 120/4 and 90/6 both accepted (even-pixel, positive resolution)")
    print()


def main() -> None:
    test_reproject_nearest_is_upsample_only_no_blending()
    test_reproject_nearest_transform_matches_scaled_source()
    test_validate_subgrid_params_rejects_odd_nr_pixels()
    test_validate_subgrid_params_rejects_non_positive()
    test_validate_subgrid_params_accepts_production_defaults()
    print("All subgrid nearest-neighbour validation checks passed.")


if __name__ == "__main__":
    main()
