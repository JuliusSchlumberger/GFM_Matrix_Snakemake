"""Validate build_roughness.py's Manning's n range guard (`build_manning_n`
raising on an implausible decoded range).

friction.tif stores Manning's_n / 100 (an eikonal-solver-specific
"slowness" convention, decode /1_000_000) - NOT a real Manning's n.
build_manning_n() multiplies the decoded value back up by 100 to recover a
real Manning's n for SFINCS. This is a one-line fix, but the kind that's
easy to silently break (remove the *100, or apply it twice) without any
visible symptom other than a physically-wrong (100x too smooth or too
rough) SFINCS model - the print statement alone ("expected ~0.01-0.15")
never used to stop a build from proceeding. This test locks in that the
function now actually RAISES on the two realistic regressions (forgetting
the *100, or double-applying it), not just prints a hint.

Usage:
    python validate_roughness_range.py
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
    """Write a synthetic friction.tif in its real on-disk convention: int16,
    encoding Manning's_n/100 scaled by FRICTION_SCALE (see build_roughness.py's
    own module docstring)."""
    profile = {
        "driver": "GTiff", "dtype": "int16", "count": 1,
        "height": manning_n_x100_int16.shape[0], "width": manning_n_x100_int16.shape[1],
        "transform": rasterio.Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0),
        "crs": "EPSG:4326", "nodata": -9999,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(manning_n_x100_int16, 1)


def _encode_real_manning_n(manning_n: np.ndarray) -> np.ndarray:
    """The real, correct on-disk encoding: (manning_n / 100) * FRICTION_SCALE."""
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
    """Forgetting the *100 fix: values ~100x too small (order 1e-4) -
    must raise, not silently write a physically-wrong roughness raster."""
    print("=== build_manning_n: rejects a *100-missing regression (values ~100x too small) ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="roughness_range_missing_scale_"))
    try:
        # What build_manning_n's own decode (slowness * 100) would produce if
        # the *100 step were somehow missing/undone upstream: values ~100x
        # smaller than a realistic 0.02-0.12 Manning's n range.
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
    """Double-applying the *100 fix: values ~100x too large (order 1-15) -
    must also raise."""
    print("=== build_manning_n: rejects a double-*100 regression (values ~100x too large) ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="roughness_range_double_scale_"))
    try:
        too_large = np.array([[2.0, 5.0], [8.0, 12.0]], dtype=np.float64)  # 100x a realistic 0.02-0.12
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
