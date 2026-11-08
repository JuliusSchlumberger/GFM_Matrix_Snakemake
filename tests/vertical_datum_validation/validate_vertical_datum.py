"""Cover vertical_datum's datum-bridge registry and nodata/sign handling on synthetic grids.

No pyshtools / <data_root> dependency: compute_single_geoid_grid and the real
OSGM15/GOCO06s sign convention are instead confirmed empirically against PROJ's
own registered ETRS89->ODN transform (see TODO.md, Section 3 verification notes).
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from vertical_datum import (  # noqa: E402
    compute_vertical_datum_offset,
    compute_vertical_datum_offset_at_points,
    sample_geoid_offset,
    sample_geoid_offset_at_points,
)


def _write_raster(path: Path, values: np.ndarray, transform: Affine, nodata: float | None) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", count=1,
        height=values.shape[0], width=values.shape[1],
        crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def test_unknown_native_datum_raises_not_implemented() -> None:
    print("=== compute_vertical_datum_offset: unknown native_datum raises, never falls through ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        egm_path = tmpdir / "egm_goco_diff.tif"
        _write_raster(egm_path, np.full((100, 100), 2.5), transform, None)
        with pytest.raises(NotImplementedError):
            compute_vertical_datum_offset(
                "bogus_datum", transform, "EPSG:4326", 100, 100, geoid_offset_raster=egm_path,
            )
        print("PASS: a bogus datum raises NotImplementedError")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_odn_osgm15_combines_hybrid_and_goco_grids_correctly() -> None:
    print("=== compute_vertical_datum_offset: odn_osgm15 = N_hybrid - N_goco, fully covered grid ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        hybrid_path = tmpdir / "hybrid.tif"
        goco_path = tmpdir / "goco_abs.tif"
        egm_path = tmpdir / "egm_goco_diff.tif"  # unused by this branch, must still be accepted
        _write_raster(hybrid_path, np.full((50, 50), 45.5), transform, None)
        _write_raster(goco_path, np.full((50, 50), 46.6), transform, None)
        _write_raster(egm_path, np.full((50, 50), 0.0), transform, None)

        offset = compute_vertical_datum_offset(
            "odn_osgm15", transform, "EPSG:4326", 50, 50,
            geoid_offset_raster=egm_path, hybrid_grid_raster=hybrid_path, goco06s_absolute_raster=goco_path,
        )
        np.testing.assert_allclose(offset, -1.1, atol=1e-4)
        print(f"PASS: offset = {offset[0, 0]:.4f} (expected 45.5 - 46.6 = -1.1)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_navd88_geoid18_combines_hybrid_and_goco_grids_correctly() -> None:
    print("=== compute_vertical_datum_offset: navd88_geoid18 = N_hybrid - N_goco, same formula as odn_osgm15 ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        hybrid_path = tmpdir / "hybrid.tif"  # stands in for the cached NAD83(2011)->ITRF2014 shift raster
        goco_path = tmpdir / "goco_abs.tif"
        egm_path = tmpdir / "egm_goco_diff.tif"  # unused by this branch, must still be accepted
        _write_raster(hybrid_path, np.full((50, 50), -32.9), transform, None)
        _write_raster(goco_path, np.full((50, 50), -33.5), transform, None)
        _write_raster(egm_path, np.full((50, 50), 0.0), transform, None)

        offset = compute_vertical_datum_offset(
            "navd88_geoid18", transform, "EPSG:4326", 50, 50,
            geoid_offset_raster=egm_path, hybrid_grid_raster=hybrid_path, goco06s_absolute_raster=goco_path,
        )
        np.testing.assert_allclose(offset, 0.6, atol=1e-4)
        print(f"PASS: offset = {offset[0, 0]:.4f} (expected -32.9 - (-33.5) = 0.6)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_odn_osgm15_nodata_near_boundary_is_filled_not_blended() -> None:
    print("=== compute_vertical_datum_offset: a small nodata gap is nearest-filled, not blended ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        hybrid = np.full((50, 50), 45.5, dtype=np.float32)
        hybrid[:, 40:] = -32768.0  # "offshore", no OSGM15 coverage
        hybrid_path = tmpdir / "hybrid.tif"
        goco_path = tmpdir / "goco_abs.tif"
        egm_path = tmpdir / "egm_goco_diff.tif"
        _write_raster(hybrid_path, hybrid, transform, -32768.0)
        _write_raster(goco_path, np.full((50, 50), 46.6), transform, None)
        _write_raster(egm_path, np.full((50, 50), 0.0), transform, None)

        offset = compute_vertical_datum_offset(
            "odn_osgm15", transform, "EPSG:4326", 50, 50,
            geoid_offset_raster=egm_path, hybrid_grid_raster=hybrid_path, goco06s_absolute_raster=goco_path,
        )
        assert not np.isnan(offset).any(), "nodata leaked into the result instead of being filled"
        # Filled cells take the nearest valid value (45.5 - 46.6 = -1.1), not a blend
        # with the raw -32768 sentinel (which would be a wildly different number).
        np.testing.assert_allclose(offset, -1.1, atol=1e-3)
        print("PASS: the nodata region is nearest-filled to -1.1, not contaminated by the raw sentinel")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_odn_osgm15_gap_too_large_to_fill_raises() -> None:
    print("=== compute_vertical_datum_offset: a gap beyond fillnodata's reach raises, not NaN ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        hybrid = np.full((400, 400), -32768.0, dtype=np.float32)
        hybrid[0, 0] = 45.5  # one valid pixel, everything else is "offshore"
        hybrid_path = tmpdir / "hybrid.tif"
        goco_path = tmpdir / "goco_abs.tif"
        egm_path = tmpdir / "egm_goco_diff.tif"
        _write_raster(hybrid_path, hybrid, transform, -32768.0)
        _write_raster(goco_path, np.full((400, 400), 46.6), transform, None)
        _write_raster(egm_path, np.full((400, 400), 0.0), transform, None)

        with pytest.raises(ValueError, match="no valid coverage"):
            compute_vertical_datum_offset(
                "odn_osgm15", transform, "EPSG:4326", 400, 400,
                geoid_offset_raster=egm_path, hybrid_grid_raster=hybrid_path, goco06s_absolute_raster=goco_path,
            )
        print("PASS: a gap too large for fillnodata raises ValueError instead of injecting NaN")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_point_sampling_matches_raster_sampling() -> None:
    print("=== sample_geoid_offset_at_points agrees with sample_geoid_offset at the same points ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, -1.0, 0, -0.01, 1.0)
        values = np.linspace(40.0, 50.0, 100 * 100).reshape(100, 100).astype(np.float32)
        raster_path = tmpdir / "grid.tif"
        _write_raster(raster_path, values, transform, None)

        lons = np.array([-0.5, 0.0, 0.3])
        lats = np.array([0.5, 0.0, -0.2])
        point_result = sample_geoid_offset_at_points(raster_path, lons, lats)

        # Independent cross-check: reproject onto a 1-pixel destination grid centred
        # exactly on each point - GDAL's own bilinear, not sample_geoid_offset_at_points'
        # map_coordinates-based one - and compare.
        for lon, lat, expected_point_value in zip(lons, lats, point_result):
            tiny_transform = Affine(1e-7, 0, lon - 5e-8, 0, -1e-7, lat + 5e-8)
            reprojected = sample_geoid_offset(raster_path, tiny_transform, "EPSG:4326", 1, 1)
            np.testing.assert_allclose(expected_point_value, reprojected[0, 0], atol=1e-3)
        print("PASS: point sampling matches an independent GDAL-reproject point sample")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_points_outside_hybrid_coverage_raises() -> None:
    print("=== compute_vertical_datum_offset_at_points: an out-of-coverage point raises ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="vertical_datum_test_"))
    try:
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 1.0)
        hybrid_path = tmpdir / "hybrid.tif"
        goco_path = tmpdir / "goco_abs.tif"
        egm_path = tmpdir / "egm_goco_diff.tif"
        _write_raster(hybrid_path, np.full((50, 50), 45.5), transform, None)
        _write_raster(goco_path, np.full((50, 50), 46.6), transform, None)
        _write_raster(egm_path, np.full((50, 50), 0.0), transform, None)

        with pytest.raises(ValueError, match="no coverage"):
            compute_vertical_datum_offset_at_points(
                "odn_osgm15", np.array([100.0]), np.array([100.0]),
                geoid_offset_raster=egm_path, hybrid_grid_raster=hybrid_path, goco06s_absolute_raster=goco_path,
            )
        print("PASS: a station outside the hybrid grid's coverage raises instead of silently continuing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_unknown_native_datum_raises_not_implemented()
    test_odn_osgm15_combines_hybrid_and_goco_grids_correctly()
    test_navd88_geoid18_combines_hybrid_and_goco_grids_correctly()
    test_odn_osgm15_nodata_near_boundary_is_filled_not_blended()
    test_odn_osgm15_gap_too_large_to_fill_raises()
    test_point_sampling_matches_raster_sampling()
    test_points_outside_hybrid_coverage_raises()
    print("All vertical_datum.py validation checks passed.")


if __name__ == "__main__":
    main()
