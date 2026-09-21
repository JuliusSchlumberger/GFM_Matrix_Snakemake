"""Validate run_sfincs_tile.py's `reproject_to_4326` latitude-corrected
target resolution - the fix for a real, confirmed bug (2026-09, comparing
tile 1573 against eikonal): letting `calculate_default_transform()` pick
its own degree resolution from a projected (UTM, isotropic-METRE) source
produces a grid that's SQUARE IN DEGREES, which is NOT square in real
ground distance except at the equator. At tile 1573 (~55 deg N), a true
30x30m UTM subgrid reprojected out to ~29 x ~50m in real ground distance
under the old (uncorrected) behaviour - inflating flooded-AREA comparisons
against the eikonal model's own (correctly latitude-corrected) grid by
~1.6x despite an almost identical flooded CELL COUNT. Not a physical model
disagreement, a reprojection resolution bug.

This test reprojects a small synthetic UTM array at a known, clearly
non-equatorial latitude and asserts the resulting EPSG:4326 pixel is
anisotropic in degrees (dx_deg != dy_deg, matching the expected
1/cos(lat) ratio) - not square, which is what the old, uncorrected
behaviour produced.

Usage:
    python validate_latlon_reprojection.py
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

from run_sfincs_tile import reproject_to_4326  # noqa: E402


def test_reprojected_pixel_is_latitude_corrected_not_square() -> None:
    print("=== reproject_to_4326: output pixel is anisotropic in degrees at non-equatorial latitude ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="latlon_reprojection_test_"))
    try:
        # A small array in UTM zone 33N (EPSG:32633), which at its own
        # tile-1573-like ~55 deg N centre latitude, should have real column
        # spacing (longitude direction) SHRUNK relative to row spacing
        # (latitude direction) once reprojected to degrees - a 30x30m true
        # UTM pixel is NOT the same number of degrees in each direction there.
        res_m = 30.0
        # UTM 33N easting/northing roughly corresponding to ~55 deg N (chosen
        # to land near tile 1573's own real location for a realistic check).
        transform = Affine(res_m, 0, 500000.0, 0, -res_m, 6096000.0)  # ~55.0 deg N at this northing
        arr = np.ones((20, 20), dtype=np.float32)
        out_path = tmpdir / "reprojected.tif"

        reproject_to_4326(arr, transform, "EPSG:32633", out_path)

        with rasterio.open(out_path) as src:
            out_transform = src.transform
            out_crs = src.crs

        assert str(out_crs).upper() in ("EPSG:4326",), out_crs
        dx_deg = abs(out_transform.a)
        dy_deg = abs(out_transform.e)
        ratio = dx_deg / dy_deg
        expected_ratio = 1.0 / np.cos(np.radians(55.0))  # ~1.74 at 55 deg N

        print(f"  dx_deg={dx_deg:.8f}, dy_deg={dy_deg:.8f}, ratio(dx/dy)={ratio:.4f}, "
              f"expected~{expected_ratio:.4f} (1/cos(55 deg))")

        assert not np.isclose(dx_deg, dy_deg, rtol=0.02), (
            f"output pixel is square in degrees (dx_deg={dx_deg}, dy_deg={dy_deg}) - "
            "this is exactly the un-fixed bug: calculate_default_transform()'s own default "
            "resolution guess was used instead of the latitude-corrected one."
        )
        assert np.isclose(ratio, expected_ratio, rtol=0.05), (
            f"anisotropy ratio {ratio:.4f} doesn't match the expected 1/cos(lat) ({expected_ratio:.4f}) "
            "within 5% - latitude correction may be using the wrong latitude or formula."
        )
        print(f"PASS: output pixel is correctly anisotropic (ratio {ratio:.4f} matches "
              f"1/cos(55 deg)={expected_ratio:.4f} within 5%), not square-in-degrees")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_reprojected_pixel_near_equator_is_nearly_square() -> None:
    """Sanity check on the other end: near the equator, cos(lat)~1, so the
    anisotropy correction should be small (ratio~1) - confirms the fix
    doesn't OVER-correct at low latitude, only where it's real."""
    print("=== reproject_to_4326: near-equator output pixel stays nearly square ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="latlon_reprojection_equator_test_"))
    try:
        res_m = 30.0
        # UTM 32N near the equator (~7 deg N, tile-1907-like).
        transform = Affine(res_m, 0, 500000.0, 0, -res_m, 774000.0)
        arr = np.ones((20, 20), dtype=np.float32)
        out_path = tmpdir / "reprojected_equator.tif"

        reproject_to_4326(arr, transform, "EPSG:32632", out_path)

        with rasterio.open(out_path) as src:
            out_transform = src.transform

        dx_deg = abs(out_transform.a)
        dy_deg = abs(out_transform.e)
        ratio = dx_deg / dy_deg
        print(f"  dx_deg={dx_deg:.8f}, dy_deg={dy_deg:.8f}, ratio(dx/dy)={ratio:.4f}")
        assert np.isclose(ratio, 1.0, atol=0.02), f"near-equator ratio {ratio:.4f} should be close to 1.0"
        print(f"PASS: near-equator ratio {ratio:.4f} is close to 1.0, as expected (1/cos(~7 deg)~1.0075)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_reprojected_pixel_is_latitude_corrected_not_square()
    test_reprojected_pixel_near_equator_is_nearly_square()
    print("All latitude-corrected reprojection validation checks passed.")


if __name__ == "__main__":
    main()
