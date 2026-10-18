"""Test that build_manning_n raises on an implausible Manning's n range.

friction.tif stores Manning's n / 100 (decode /1_000_000); build_manning_n
multiplies by 100. Losing or doubling that factor must raise, not write.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_roughness import FRICTION_SCALE, build_manning_n  # noqa: E402


def _write_friction_raster(path: Path, manning_n_x100_int16: np.ndarray) -> None:
    profile = {
        "driver": "GTiff", "dtype": "int16", "count": 1,
        "height": manning_n_x100_int16.shape[0], "width": manning_n_x100_int16.shape[1],
        "transform": rasterio.Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0),
        "crs": "EPSG:4326", "nodata": -9999,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(manning_n_x100_int16, 1)


def _encode_real_manning_n(manning_n: np.ndarray) -> np.ndarray:
    return np.round((manning_n / 100.0) * FRICTION_SCALE).astype(np.int16)


def test_realistic_land_cover_range_passes() -> None:
    print("=== build_manning_n: realistic land-cover range (0.02-0.12) passes ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="roughness_range_test_"))
    try:
        manning_n = np.array([[0.02, 0.05], [0.08, 0.12]], dtype=np.float64)
        friction_path = tmpdir / "friction.tif"
        _write_friction_raster(friction_path, _encode_real_manning_n(manning_n))
        out_path = tmpdir / "manning_n.tif"

        lo, hi = build_manning_n(friction_path, out_path)
        assert np.isclose(lo, 0.02, atol=1e-3), lo
        assert np.isclose(hi, 0.12, atol=1e-3), hi
        assert out_path.exists()
        print(f"PASS: realistic range [{lo:.5f}, {hi:.5f}] accepted, output written")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_missing_scale_factor_is_rejected() -> None:
    print("=== build_manning_n: rejects a *100-missing regression (values ~100x too small) ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="roughness_range_missing_scale_"))
    try:
        too_small = np.array([[0.00005, 0.0001], [0.00008, 0.00012]], dtype=np.float64)
        friction_path = tmpdir / "friction.tif"
        _write_friction_raster(friction_path, _encode_real_manning_n(too_small))
        out_path = tmpdir / "manning_n.tif"

        with pytest.raises(ValueError, match="unit"):
            build_manning_n(friction_path, out_path)
        print("PASS: implausibly small range raises ValueError instead of silently writing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_double_applied_scale_factor_is_rejected() -> None:
    print("=== build_manning_n: rejects a double-*100 regression (values ~100x too large) ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="roughness_range_double_scale_"))
    try:
        too_large = np.array([[2.0, 5.0], [8.0, 12.0]], dtype=np.float64)
        friction_path = tmpdir / "friction.tif"
        _write_friction_raster(friction_path, _encode_real_manning_n(too_large))
        out_path = tmpdir / "manning_n.tif"

        with pytest.raises(ValueError, match="unit"):
            build_manning_n(friction_path, out_path)
        print("PASS: implausibly large range raises ValueError instead of silently writing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_realistic_land_cover_range_passes()
    test_missing_scale_factor_is_rejected()
    test_double_applied_scale_factor_is_rejected()
    print("All Manning's n range validation checks passed.")


if __name__ == "__main__":
    main()
