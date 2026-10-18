"""Test run_sfincs_tile._apply_land_mask_doublecheck (native mask.tif, nearest).

downscale_floodmap's vector gdf_mask leaks ocean cells at the coastline through
rasterization edge effects; both checks must agree a cell is land.
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
    print("=== _apply_land_mask_doublecheck: masks a gdf_mask-leaked ocean cell ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="landmask_doublecheck_test_"))
    try:
        transform = Affine(0.0001, 0, 10.0, 0, -0.0001, 50.0)
        hmax = np.full((4, 4), np.nan, dtype=np.float32)
        hmax[1, 0] = 9.77  # leaked ocean cell
        hmax[2, 2] = 0.62

        native_mask = np.zeros((4, 4), dtype=np.float32)
        native_mask[:, 0] = 1.0  # ocean

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
    print("=== _apply_land_mask_doublecheck: never introduces flooding ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="landmask_doublecheck_dry_test_"))
    try:
        transform = Affine(0.0001, 0, 10.0, 0, -0.0001, 50.0)
        hmax = np.full((3, 3), np.nan, dtype=np.float32)
        native_mask = np.zeros((3, 3), dtype=np.float32)
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
