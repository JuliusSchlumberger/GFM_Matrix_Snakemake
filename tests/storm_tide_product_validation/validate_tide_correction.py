"""Synthetic checks of the through-water gauge selection of src/tide_correction."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tide_correction as tc  # noqa: E402


def _two_fjords(tmp_path: Path) -> Path:
    # land everywhere except two north-south fjords at lon -0.05 and +0.05 joined by open sea south of
    # lat -0.3; within each fjord the gauge and point are 5.6 km apart, across fjords 11 km
    land = gpd.GeoDataFrame(geometry=[box(-1, -0.3, -0.06, 1), box(-0.04, -0.3, 0.04, 1), box(0.06, -0.3, 1, 1)],
                            crs="EPSG:4326")
    path = tmp_path / "land.gpkg"
    land.to_file(path)
    return path


def test_gauge_in_the_next_fjord_does_not_correct_the_point(tmp_path: Path) -> None:
    gauges = np.array([[-0.05, 0.05], [0.05, 0.05]])
    points = np.array([[-0.05, 0.0], [0.05, 0.0]])
    links = tc.water_links(points, gauges, _two_fjords(tmp_path), workers=1)
    assert links.toarray().tolist() == [[True, False], [False, True]]
    factor, count = tc.correction_factors(points, gauges, np.log([2.0, 0.5]), links=links)
    assert np.allclose(factor, [2.0, 0.5]) and count.tolist() == [1, 1]
    unlinked, n = tc.correction_factors(points, gauges, np.log([2.0, 0.5]))
    assert np.allclose(unlinked, 1.0) and n.tolist() == [2, 2]
