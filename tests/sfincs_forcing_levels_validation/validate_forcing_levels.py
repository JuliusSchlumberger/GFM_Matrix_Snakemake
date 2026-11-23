"""SFINCS forcing points take production's seed levels and sample the whole boundary line."""

import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely.geometry import LineString, box

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_line_forcing import (  # noqa: E402
    FORCING_SPACING_M, attach_hydrographs, level_at_nearest_seed, line_samples, production_seed_levels,
)
from calibration_paths import RETURN_PERIOD, WATERLEVEL_NAME  # noqa: E402

RULE = {"knn": 2, "idw_power": 3.0, "river_code": 3}


def _tile(root: Path) -> Path:
    # 1 x 1 deg at 0.01 deg: ocean west of lon 10.5, land east of it.
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 1
    with rasterio.open(inputs / "mask.tif", "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=rasterio.Affine(0.01, 0, 10.0, 0, -0.01, 51.0)) as dst:
        dst.write(mask, 1)
    gpd.GeoDataFrame({WATERLEVEL_NAME: np.array([300, 100], dtype=np.int16)},
                     geometry=gpd.points_from_xy([10.2, 10.2], [50.9, 50.1]), crs="EPSG:4326",
                     ).to_file(inputs / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    gpd.GeoDataFrame(geometry=[box(10.0, 50.0, 11.0, 51.0)], crs="EPSG:4326").to_file(inputs / "tile_geometry.gpkg")
    return root


def test_forcing_point_takes_the_seed_level_of_the_coast_behind_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cells, levels = production_seed_levels(_tile(Path(tmp)), RULE)
        assert np.allclose(cells[:, 0], 10.495)  # the ocean column next to land
        offshore = np.array([[10.3, 50.9], [10.3, 50.5], [10.3, 50.1]])
        north, middle, south = level_at_nearest_seed(offshore, cells, levels)
        assert north > 2.8 and south < 1.2
        assert south < middle < north


def test_blended_hydrograph_peaks_at_the_level_and_lies_between_its_stations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "hg.nc"
        t = np.arange(9)
        narrow = np.array([0.0, 0.0, 0.1, 0.5, 1.0, 0.5, 0.1, 0.0, 0.0])
        broad = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2])  # peaks at the same step
        xr.Dataset({"hydrograph_average_tide_signal": (("station", "time"), np.vstack([narrow, broad]))},
                   coords={"station_x_coordinate": ("station", [10.0, 10.2]),
                           "station_y_coordinate": ("station", [50.0, 50.0]),
                           "time": pd.date_range("2026-01-01", periods=len(t), freq="h")}).to_netcdf(nc)
        points = gpd.GeoDataFrame({"rp100_level_m": [3.0, 3.0, 3.0]},
                                  geometry=gpd.points_from_xy([10.0, 10.1, 10.2], [50.0, 50.0, 50.0]), crs="EPSG:4326")
        pts, hydro = attach_hydrographs(points, nc, knn=2, idw_power=3.0)
        curves = hydro.drop(columns="elapsed_hr").to_numpy().T
        assert np.allclose(curves.max(axis=1), 3.0)
        assert np.allclose(curves[0], narrow + 2.0, atol=1e-6) and np.allclose(curves[2], broad + 2.0, atol=1e-6)
        assert np.allclose(curves[1], 0.5 * (narrow + broad) + 2.0, atol=1e-3)
        assert pts["n_hydrographs"].tolist() == [2, 2, 2]


def test_line_samples_are_at_most_the_spacing_apart_and_shared_nodes_count_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tile = _tile(Path(tmp))
        lines = Path(tmp) / "line.gpkg"
        gpd.GeoDataFrame({"kind": ["coastline", "tile_edge", "coastline"]},
                         geometry=[LineString([(10.45, 50.1), (10.45, 50.5)]),
                                   LineString([(10.45, 50.5), (10.45, 50.9)]),
                                   LineString([(12.5, 50.1), (12.5, 50.9)])],  # > TILE_MARGIN_M from the tile
                         crs="EPSG:4326").to_file(lines, layer="segments", driver="GPKG")
        pts = line_samples(lines, tile / "inputs" / "tile_geometry.gpkg")
        assert set(pts["segment_kind"]) == {"coastline", "tile_edge"}
        gaps = np.diff(np.sort(pts.to_crs(pts.estimate_utm_crs()).geometry.y.to_numpy()))
        assert gaps.min() > 1.0
        assert gaps.max() <= FORCING_SPACING_M + 1.0
