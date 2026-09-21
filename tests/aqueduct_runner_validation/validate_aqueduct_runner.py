"""End-to-end smoke test for `aqueduct_runner.run_aqueduct_python` - the
shared function BOTH real production entry points (scripts/run_aqueduct.py,
the Snakemake rule, and scripts/run_aqueduct_cli.py, the standalone HPC
CLI) call, and which had zero tests of its own before this (per this
session's own eikonal-pipeline test-coverage stocktake).

Fully synthetic: builds tiny dem.tif/mask.tif/friction.tif in the exact
real on-disk int16 conventions (via rasters.py's own encoders, not a
hand-rolled approximation of them) plus a boundaries.gpkg, runs the real
function against them, and checks it produces a real, decodable
waterdepth_*.tif and sane diagnostics - the "did I break the actual
production entry point" check equivalent to the SFINCS pipeline's own
`sfincs_synthetic_end_to_end` test.

Usage:
    python validate_aqueduct_runner.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aqueduct_runner import run_aqueduct_python  # noqa: E402
from rasters import decode_waterdepth_cm, encode_dem_cm, encode_friction_int16, encode_waterlevel_cm  # noqa: E402


def _write_int16_raster(path: Path, encoded: np.ndarray, transform: Affine, nodata=None) -> None:
    profile = {
        "driver": "GTiff", "dtype": "int16", "count": 1,
        "height": encoded.shape[0], "width": encoded.shape[1],
        "transform": transform, "crs": "EPSG:4326", "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(encoded.astype(np.int16), 1)


def test_run_aqueduct_python_end_to_end_on_synthetic_tile() -> None:
    print("=== run_aqueduct_python: full synthetic tile, real on-disk encoding conventions ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="aqueduct_runner_test_"))
    try:
        m, n = 10, 12
        transform = Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0)

        dem_m = np.full((m, n), 3.0, dtype=np.float64)
        mask = np.zeros((m, n), dtype=np.uint8)
        mask[:, :3] = 1  # ocean strip on the left
        dem_m = np.where(mask == 1, 0.0, dem_m)  # ocean cells at 0m, matching real extract_dem convention
        friction = np.full((m, n), 0.0005, dtype=np.float64)  # Manning's n/100, realistic magnitude

        dem_path = tmpdir / "dem.tif"
        mask_path = tmpdir / "mask.tif"
        friction_path = tmpdir / "friction.tif"
        _write_int16_raster(dem_path, encode_dem_cm(dem_m), transform)
        _write_int16_raster(mask_path, mask.astype(np.int16), transform)
        _write_int16_raster(friction_path, encode_friction_int16(friction), transform)

        # One boundary station just off the ocean strip, a real storm-tide-like water level.
        boundaries_gdf = gpd.GeoDataFrame(
            {"SLR_0": encode_waterlevel_cm(np.array([1.2]))},
            geometry=[Point(10.0002, 49.9985)],
            crs="EPSG:4326",
        )
        boundaries_path = tmpdir / "boundaries_RP100_SLR_0.gpkg"
        boundaries_gdf.to_file(boundaries_path, driver="GPKG")

        output_path = tmpdir / "waterdepth_RP100_SLR_0.tif"

        diagnostics = run_aqueduct_python(
            dem_path, mask_path, friction_path, output_path,
            resolution=30.0, k=5, variable="SLR_0",
            boundaries_path=boundaries_path,
            ocean_code=1, river_code=None,
        )

        assert output_path.exists(), "run_aqueduct_python did not write an output raster"
        assert diagnostics["obstacle_coupling"] is False

        with rasterio.open(output_path) as src:
            encoded = src.read(1)
        waterdepth = decode_waterdepth_cm(encoded)
        finite = waterdepth[np.isfinite(waterdepth)]
        assert finite.size > 0, "output raster is entirely nodata"
        assert finite.min() >= 0.0, f"negative water depth: {finite.min()}"
        assert finite.max() < 50.0, f"implausible max water depth {finite.max()}m - likely a units/decode bug"

        print(f"  wrote {output_path}: {int((waterdepth > 0).sum())} flooded cell(s) of {waterdepth.size}, "
              f"max depth {finite.max():.3f} m")
        print("PASS: run_aqueduct_python completed end to end with a decodable, bounded, plausible result")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_run_aqueduct_python_friction_scale_factor_is_applied() -> None:
    """`friction_scale_factor` (calibration-sweep knob, default 1.0 no-op) -
    a value > 1.0 makes propagation more expensive, so a station that
    reaches inland at factor=1.0 should flood fewer (or equal) cells at a
    much higher factor - a real, directional sanity check that the
    multiplier is actually wired into the solve, not silently dropped."""
    print("=== run_aqueduct_python: friction_scale_factor actually changes the result ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="aqueduct_runner_scale_test_"))
    try:
        # A long, gently-rising-inland domain (not a small flat one) - numerically
        # confirmed (see conversation) to give friction a real, measurable effect on
        # how far inland the flood reaches, rather than everything flooding regardless.
        m, n = 15, 60
        transform = Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0)
        dem_m = np.tile(np.linspace(0.5, 3.0, n), (m, 1)).astype(np.float64)
        mask = np.zeros((m, n), dtype=np.uint8)
        mask[:, :3] = 1
        dem_m = np.where(mask == 1, 0.0, dem_m)
        friction = np.full((m, n), 0.02, dtype=np.float64)

        dem_path = tmpdir / "dem.tif"
        mask_path = tmpdir / "mask.tif"
        friction_path = tmpdir / "friction.tif"
        _write_int16_raster(dem_path, encode_dem_cm(dem_m), transform)
        _write_int16_raster(mask_path, mask.astype(np.int16), transform)
        _write_int16_raster(friction_path, encode_friction_int16(friction), transform)

        boundaries_gdf = gpd.GeoDataFrame(
            {"SLR_0": encode_waterlevel_cm(np.array([2.0]))},
            geometry=[Point(10.0002, 49.9985)],
            crs="EPSG:4326",
        )
        boundaries_path = tmpdir / "boundaries_RP100_SLR_0.gpkg"
        boundaries_gdf.to_file(boundaries_path, driver="GPKG")

        def run(scale_factor: float) -> np.ndarray:
            out_path = tmpdir / f"waterdepth_scale_{scale_factor}.tif"
            run_aqueduct_python(
                dem_path, mask_path, friction_path, out_path,
                resolution=30.0, k=5, variable="SLR_0",
                boundaries_path=boundaries_path, ocean_code=1,
                friction_scale_factor=scale_factor,
            )
            with rasterio.open(out_path) as src:
                return decode_waterdepth_cm(src.read(1))

        wd_normal = run(1.0)
        wd_expensive = run(200.0)  # friction 200x higher -> propagation should reach much less far

        n_flooded_normal = int((wd_normal > 0).sum())
        n_flooded_expensive = int((wd_expensive > 0).sum())
        print(f"  scale=1.0: {n_flooded_normal} flooded cell(s); scale=200.0: {n_flooded_expensive} flooded cell(s)")
        assert n_flooded_expensive < n_flooded_normal, (
            f"higher friction_scale_factor did not flood STRICTLY fewer cells "
            f"({n_flooded_expensive} vs {n_flooded_normal}) - the multiplier may not be reaching the "
            f"actual solve (monotonicity: raising friction anywhere can only lower or hold eikonal "
            f"solution values, never raise them, so a real 200x increase should visibly shrink the extent)."
        )
        print("PASS: a 200x higher friction_scale_factor floods strictly fewer cells than the baseline "
              "(confirms the multiplier genuinely reaches the actual solve)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_run_aqueduct_python_end_to_end_on_synthetic_tile()
    test_run_aqueduct_python_friction_scale_factor_is_applied()
    print("All aqueduct_runner.py validation checks passed.")


if __name__ == "__main__":
    main()
