"""Pin the MDT (Mean Dynamic Topography) sign in the SFINCS elevation and forcing.

Elevation: H_GOCO06s = H_MSL (GEBCO) + MDT. Forcing: mdt_offset =
boundary_value_m - hydrograph_max_m (COAST-RP levels carry MDT, raw COAST-HG
does not). A flipped sign doubles the error yet still looks plausible.
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

from build_line_forcing import build_line_forcing  # noqa: E402
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
    print("=== build_combined_elevation: MDT is ADDED to GEBCO, not subtracted ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="mdt_sign_elevation_test_"))
    try:
        transform = rasterio.Affine(0.0003, 0, 10.0, 0, -0.0003, 50.0)
        # Column 0 must lie beyond the GEBCO_COAST_CLIP_M (200 m) band, where GEBCO is
        # replaced by interpolation from land: ~21.5 m cells at 50N -> ~430 m from land.
        shape = (4, 24)

        dem_cm = np.full(shape, 500, dtype=np.int16)
        dem_path = tmpdir / "dem.tif"
        _write_raster(dem_path, dem_cm, "int16", -9999, transform)

        mask = np.zeros(shape, dtype=np.float32)
        mask[:, :20] = 1.0
        mask_path = tmpdir / "mask.tif"
        _write_raster(mask_path, mask, "float32", None, transform)

        gebco_bathy_m = -20.0  # local MSL
        gebco_arr = np.full(shape, gebco_bathy_m, dtype=np.float32)
        gebco_path = tmpdir / "gebco.tif"
        _write_raster(gebco_path, gebco_arr, "float32", np.nan, transform)

        known_mdt_m = 0.4336  # magnitude seen on tile 691
        mdt_arr = np.full((10, 10), known_mdt_m, dtype=np.float64)
        mdt_path = tmpdir / "mdt.nc"
        mdt_da = xr.DataArray(
            mdt_arr, dims=("lat", "lon"),
            coords={"lat": 90.0 - 0.5 - np.arange(10), "lon": np.arange(10) + 0.5},
        )
        mdt_da.name = "mdt"
        mdt_da.to_dataset().to_netcdf(mdt_path)

        combined, info = build_combined_elevation(
            dem_path, mask_path, gebco_path, mdt_path,
            min_bathymetry_m=-50.0,
        )

        assert np.isclose(info["mdt_m"], known_mdt_m, atol=1e-6), info["mdt_m"]
        ocean_value = combined[0, 0]
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
    """The offset must be added to the series, so its peak equals the boundary value."""
    print("=== build_line_forcing: mdt_offset correctly recovers the boundary value ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="mdt_sign_hydrograph_test_"))
    try:
        station_lon, station_lat = 10.0, 50.0
        n_time = 5
        hg_raw = np.array([0.1, 0.3, 0.9, 0.4, 0.1])  # local MSL, no MDT

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

        known_boundary_val_m = 1.35
        stations = gpd.GeoDataFrame(
            {"station": [0], "station_index": [0], "status": ["used"], "rp100_level_m": [known_boundary_val_m],
             "node_lon": [station_lon + 0.01], "node_lat": [station_lat], "node_shift_km": [0.7]},
            geometry=[Point(station_lon, station_lat)],
            crs="EPSG:4326",
        )
        lines_gpkg = tmpdir / "line.gpkg"
        stations.to_file(lines_gpkg, layer="stations", driver="GPKG")

        points, hydro = build_line_forcing(lines_gpkg, coast_hg_path)

        assert len(points) == 1, points
        offset = float(points["mdt_offset_m"].iloc[0])
        expected_offset = known_boundary_val_m - hg_raw.max()
        assert np.isclose(offset, expected_offset, atol=1e-4), (
            f"offset {offset:.4f} != expected {expected_offset:.4f} (boundary_val - hg_max)"
        )
        corrected_peak = hydro[0].max()
        assert np.isclose(corrected_peak, known_boundary_val_m, atol=1e-6), (
            f"corrected hydrograph peak {corrected_peak:.4f} != known boundary value {known_boundary_val_m} - "
            "offset must be ADDED to the whole series, not subtracted."
        )
        print(f"PASS: offset={offset:+.4f} m correctly recovers boundary value "
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
