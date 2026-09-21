"""Validate the MDT (Mean Dynamic Topography) sign convention on both sides
of the SFINCS pipeline that apply it:
  - build_elevation.py::build_combined_elevation - ADDS MDT to GEBCO
    bathymetry (H_GOCO06s = H_MSL + MDT, per mdt.py's own module docstring,
    "fixed 2026-09 after a real, confirmed sign-error investigation").
  - build_boundary_forcing.py::match_boundary_points_to_coast_hg - the
    empirical `mdt_offset = boundary_value_m - hydrograph_max_m`, which
    recovers the same real-world MDT correction indirectly (the tile's own
    boundaries_RP100_SLR_0.gpkg value already carries it; COAST-HG's raw
    hydrograph doesn't).

Both are exactly the kind of geodetic sign convention that's easy to flip
by accident and hard to notice visually (a flipped sign DOUBLES the error
instead of cancelling it, but a coastline elevation/forcing plot can still
"look plausible" either way) - this test locks in the correct sign with a
known, deliberately non-zero synthetic MDT value.

Usage:
    python validate_mdt_sign.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_boundary_forcing import match_boundary_points_to_coast_hg  # noqa: E402
from build_elevation import build_combined_elevation  # noqa: E402


def _write_raster(path: Path, arr: np.ndarray, dtype: str, nodata, transform, crs="EPSG:4326") -> None:
    profile = {
        "driver": "GTiff", "dtype": dtype, "count": 1,
        "height": arr.shape[0], "width": arr.shape[1],
        "transform": transform, "crs": crs, "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def test_elevation_mdt_is_added_not_subtracted() -> None:
    """A KNOWN, deliberately non-zero MDT value must be ADDED to GEBCO's raw
    bathymetry - a flipped sign would produce `gebco - mdt_m` instead."""
    print("=== build_combined_elevation: MDT is ADDED to GEBCO, not subtracted ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="mdt_sign_elevation_test_"))
    try:
        transform = rasterio.Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0)
        shape = (4, 4)

        dem_cm = np.full(shape, 500, dtype=np.int16)  # 5.00 m land everywhere by default
        dem_path = tmpdir / "dem.tif"
        _write_raster(dem_path, dem_cm, "int16", -9999, transform)

        mask = np.zeros(shape, dtype=np.float32)
        mask[:, 0] = 1.0  # left column: ocean
        mask_path = tmpdir / "mask.tif"
        _write_raster(mask_path, mask, "float32", None, transform)

        gebco_bathy_m = -20.0  # known raw (local-MSL) GEBCO value
        gebco_arr = np.full(shape, gebco_bathy_m, dtype=np.float32)
        gebco_path = tmpdir / "gebco.tif"
        _write_raster(gebco_path, gebco_arr, "float32", np.nan, transform)

        known_mdt_m = 0.4336  # deliberately non-zero, matches the real magnitude seen live on tile 691
        mdt_arr = np.full((10, 10), known_mdt_m, dtype=np.float64)
        mdt_transform = rasterio.Affine(1.0, 0, 0.0, 0, -1.0, 90.0)
        mdt_path = tmpdir / "mdt.nc"
        mdt_da = xr.DataArray(
            mdt_arr, dims=("lat", "lon"),
            coords={"lat": 90.0 - 0.5 - np.arange(10), "lon": np.arange(10) + 0.5},
        )
        mdt_da.name = "mdt"
        mdt_da.to_dataset().to_netcdf(mdt_path)

        combined, info = build_combined_elevation(
            dem_path, mask_path, gebco_path, mdt_path,
            min_bathymetry_m=-50.0,  # generous floor - not what this test is checking
        )

        assert np.isclose(info["mdt_m"], known_mdt_m, atol=1e-6), info["mdt_m"]
        ocean_value = combined[0, 0]  # left column is ocean
        expected_added = gebco_bathy_m + known_mdt_m
        expected_subtracted = gebco_bathy_m - known_mdt_m
        assert np.isclose(ocean_value, expected_added, atol=1e-3), (
            f"ocean elevation {ocean_value:.4f} doesn't match ADD convention "
            f"(expected {expected_added:.4f}) - got closer to SUBTRACT ({expected_subtracted:.4f})? "
            f"sign may be flipped."
        )
        assert not np.isclose(ocean_value, expected_subtracted, atol=1e-3), (
            "ocean elevation matches the SUBTRACTED value - MDT sign is flipped!"
        )
        print(f"PASS: GEBCO {gebco_bathy_m} m + MDT {known_mdt_m:+.4f} m = {ocean_value:.4f} m (ADD confirmed)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_hydrograph_mdt_offset_recovers_known_boundary_value() -> None:
    """The empirical offset (`boundary_val - hg_max`) must make the
    corrected hydrograph's own peak land EXACTLY on the known boundary
    value - by construction, so this is really checking the offset is
    ADDED (not subtracted) when building the corrected series."""
    print("=== match_boundary_points_to_coast_hg: mdt_offset correctly recovers the boundary value ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="mdt_sign_hydrograph_test_"))
    try:
        station_lon, station_lat = 10.0, 50.0
        n_time = 5
        hg_raw = np.array([0.1, 0.3, 0.9, 0.4, 0.1])  # raw hydrograph, local-MSL, no MDT, peak 0.9

        ds = xr.Dataset(
            {
                "hydrograph_average_tide_signal": (("station", "time"), hg_raw[None, :]),
            },
            coords={
                "station_x_coordinate": ("station", [station_lon]),
                "station_y_coordinate": ("station", [station_lat]),
                "time": pd.date_range("2026-01-01", periods=n_time, freq="h"),
            },
        )
        coast_hg_path = tmpdir / "coast_hg.nc"
        ds.to_netcdf(coast_hg_path)

        known_boundary_val_m = 1.35  # deliberately != hg_raw.max() (0.9), by a known, non-zero amount
        boundaries_gdf = gpd.GeoDataFrame(
            {"SLR_0": [int(round(known_boundary_val_m * 100))]},  # int64 cm, matches real on-disk convention
            geometry=[Point(station_lon, station_lat)],
            crs="EPSG:4326",
        )

        df, times, dropped, offsets_m = match_boundary_points_to_coast_hg(boundaries_gdf, "SLR_0", coast_hg_path)

        assert not dropped, dropped
        expected_offset = known_boundary_val_m - hg_raw.max()
        assert np.isclose(offsets_m[0], expected_offset, atol=1e-6), (
            f"offset {offsets_m[0]:.4f} != expected {expected_offset:.4f} (boundary_val - hg_max)"
        )
        corrected_peak = df[0].max()
        assert np.isclose(corrected_peak, known_boundary_val_m, atol=1e-6), (
            f"corrected hydrograph peak {corrected_peak:.4f} != known boundary value {known_boundary_val_m} - "
            "offset must be ADDED to the whole series, not subtracted."
        )
        print(f"PASS: offset={offsets_m[0]:+.4f} m correctly recovers boundary value "
              f"{known_boundary_val_m} m as the corrected hydrograph's own peak")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_elevation_mdt_is_added_not_subtracted()
    test_hydrograph_mdt_offset_recovers_known_boundary_value()
    print("All MDT sign-convention validation checks passed.")


if __name__ == "__main__":
    main()
