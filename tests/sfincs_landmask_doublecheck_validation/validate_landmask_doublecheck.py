"""Validate run_sfincs_tile.py's second, independent land-mask check
(`_apply_land_mask_doublecheck`) - the fix for a real, confirmed leak
(2026-09, tile 1907's first subgrid run): `downscale_floodmap()`'s own
`gdf_mask` (a vectorized land polygon) let a small number of genuinely-
ocean cells slip through at the coastline, due to vector/raster
rasterization edge effects right at the boundary. The fix reprojects the
tile's own native mask.tif directly onto the output grid (nearest-
neighbour, no vectorization step at all) and requires BOTH checks to agree
a cell is land.

This test constructs a synthetic case that reproduces the exact failure
mode: an `hmax` array with a "flooded" value sitting on a subgrid cell that
sits right on a native-mask ocean/land boundary - the kind of cell a
polygon rasterization is most likely to misclassify - and confirms the
raster-vs-raster doublecheck masks it out.

Usage:
    python validate_landmask_doublecheck.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from run_sfincs_tile import _apply_land_mask_doublecheck  # noqa: E402


def _write_mask_raster(path: Path, mask: np.ndarray, transform: Affine) -> None:
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": mask.shape[0], "width": mask.shape[1],
        "transform": transform, "crs": "EPSG:4326", "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.float32), 1)


def test_doublecheck_masks_out_ocean_cell_gdf_mask_missed() -> None:
    """gdf_mask's own leak lets a real ocean cell keep a positive hmax
    value - the doublecheck must NaN it out once the native mask says
    ocean (code 1) there, regardless of what gdf_mask already allowed."""
    print("=== _apply_land_mask_doublecheck: masks a gdf_mask-leaked ocean cell ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="landmask_doublecheck_test_"))
    try:
        # 4x4 output grid (hmax); left column ocean(1), right 3 columns land(0)
        # in the NATIVE mask - one deliberately "leaked" flooded value sits at
        # (1, 0), squarely in the ocean column, as if gdf_mask's own boundary
        # rasterization had wrongly let it through.
        transform = Affine(0.0001, 0, 10.0, 0, -0.0001, 50.0)
        hmax = np.full((4, 4), np.nan, dtype=np.float32)
        hmax[1, 0] = 9.77  # the leaked cell (mirrors the real tile-1907 magnitude)
        hmax[2, 2] = 0.62  # a genuine, real land-side flood value

        native_mask = np.zeros((4, 4), dtype=np.float32)  # default land (0)
        native_mask[:, 0] = 1.0  # left column: real ocean

        mask_path = tmpdir / "mask.tif"
        _write_mask_raster(mask_path, native_mask, transform)

        result = _apply_land_mask_doublecheck(hmax, transform, "EPSG:4326", mask_path)

        assert np.isnan(result[1, 0]), f"leaked ocean cell not masked out: {result[1, 0]}"
        assert result[2, 2] == np.float32(0.62), f"genuine land flood cell wrongly masked: {result[2, 2]}"
        print(f"PASS: ocean-side leaked cell (was {hmax[1, 0]}) -> NaN; "
              f"genuine land-side flood cell (was {hmax[2, 2]}) -> preserved")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_doublecheck_preserves_all_dry_input() -> None:
    """An all-NaN hmax (nothing flooded anywhere) must stay all-NaN - the
    doublecheck should never CREATE a flood value, only ever remove one."""
    print("=== _apply_land_mask_doublecheck: never introduces flooding ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="landmask_doublecheck_dry_test_"))
    try:
        transform = Affine(0.0001, 0, 10.0, 0, -0.0001, 50.0)
        hmax = np.full((3, 3), np.nan, dtype=np.float32)
        native_mask = np.zeros((3, 3), dtype=np.float32)  # all land
        mask_path = tmpdir / "mask.tif"
        _write_mask_raster(mask_path, native_mask, transform)

        result = _apply_land_mask_doublecheck(hmax, transform, "EPSG:4326", mask_path)
        assert np.all(np.isnan(result)), result
        print("PASS: all-dry input stays all-dry after the doublecheck")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_doublecheck_masks_out_ocean_cell_gdf_mask_missed()
    test_doublecheck_preserves_all_dry_input()
    print("All land-mask doublecheck validation checks passed.")


if __name__ == "__main__":
    main()
