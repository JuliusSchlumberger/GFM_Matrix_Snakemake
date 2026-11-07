"""SFINCS forcing from either storm-tide source: the product is put on COAST-HG's time axis."""

import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_line_forcing import COAST_HG_PEAK_HR, COAST_HG_STEPS, build_line_forcing, read_hydrographs  # noqa: E402
from gfm_config import storm_tide_hydrograph_path  # noqa: E402

LON, LAT = 10.0, 50.0
STEP = np.timedelta64(600, "s")
HALF = 720  # +-120 h


def _product(path: Path, half_window_steps: int = 432) -> np.ndarray:
    t = np.arange(-HALF, HALF + 1)
    series = np.full((1, 2, len(t)), np.nan, dtype=np.float32)
    inside = np.abs(t) <= half_window_steps
    for j, peak in enumerate((0.8, 1.2)):
        series[0, j, inside] = peak * np.cos(np.pi * t[inside] / (2 * half_window_steps + 2)) ** 2
    xr.Dataset({"storm_tide_hydrograph": (("stations", "return_period", "time"), series)},
               coords={"station_x_coordinate": ("stations", [LON]), "station_y_coordinate": ("stations", [LAT]),
                       "return_period": [10, 100], "time": t * STEP}).to_netcdf(path)
    return series[0, 1]


def _line(path: Path, level_m: float) -> None:
    gpd.GeoDataFrame(
        {"station": [0], "station_index": [0], "status": ["used"], "rp100_level_m": [level_m],
         "node_lon": [LON + 0.01], "node_lat": [LAT], "node_shift_km": [0.7]},
        geometry=[Point(LON, LAT)], crs="EPSG:4326",
    ).to_file(path, layer="stations", driver="GPKG")


def test_product_rp100_lands_on_the_coast_hg_axis_with_its_peak_at_74_5_h() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "product.nc"
        _product(nc)
        _, _, series, times = read_hydrographs(nc)
        assert series.shape == (1, COAST_HG_STEPS)
        assert times[0] == pd.Timestamp("2000-01-01")
        peak_hr = (times[int(np.argmax(series[0]))] - times[0]).total_seconds() / 3600
        assert peak_hr == COAST_HG_PEAK_HR
        assert np.isclose(series.max(), 1.2)  # RP100, not RP10
        assert np.isfinite(series).all()  # NaN outside the window holds the window's end level


def test_product_forcing_peak_equals_the_boundary_level() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nc, lines = Path(tmp) / "product.nc", Path(tmp) / "line.gpkg"
        _product(nc)
        _line(lines, 2.35)
        points, hydro = build_line_forcing(lines, nc)
        assert np.isclose(hydro[0].max(), 2.35)
        assert np.isclose(float(points.mdt_offset_m.iloc[0]), 2.35 - 1.2, atol=1e-4)
        assert hydro.elapsed_hr.iloc[int(np.argmax(hydro[0].to_numpy()))] == COAST_HG_PEAK_HR


def test_config_picks_the_hydrograph_file_of_the_storm_tide_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"paths": {"root": "/data", "processed_inputs_dir": "{root}/processed_inputs"},
               "boundary_conditions": {"storm_tide_source": "gfm_product",
                                       "storm_tide_product_dir": "{processed_inputs_dir}/storm_tide_product"},
               "simulation": {"flooding": {"duration_correction": {
                   "coast_hg_path": "{root}/inputs/COAST_HG/COAST-HG_RP100.nc"}}}}
        path = Path(tmp) / "config.yml"
        path.write_text(yaml.safe_dump(cfg))
        assert storm_tide_hydrograph_path(path) == Path(
            "/data/processed_inputs/storm_tide_product/GFM_storm_tide_hydrographs.nc")
        cfg["boundary_conditions"]["storm_tide_source"] = "coast_rp"
        path.write_text(yaml.safe_dump(cfg))
        assert storm_tide_hydrograph_path(path) == Path("/data/inputs/COAST_HG/COAST-HG_RP100.nc")
