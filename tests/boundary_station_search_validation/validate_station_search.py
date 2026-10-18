"""Test the minimum search-area floor of boundaries.select_stations_for_tile.

The floor is a centred square of side min_search_size_deg (radius = half).
"""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import boundaries as bnd  # noqa: E402


def _tile(bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs="EPSG:4326")


def _stations(points: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"waterlevel": [1.0] * len(points)},
        geometry=[Point(x, y) for x, y in points],
        crs="EPSG:4326",
    )


def test_min_size_floor_kicks_in_for_tiny_tile() -> None:
    print("=== tiny tile: search area floored to min_search_size_deg, centred on the tile ===")
    # 0.01 deg tile + 0.1 deg buffer = 0.21 deg box; the 2 deg floor must dominate.
    tile = _tile((9.995, 49.995, 10.005, 50.005))
    near_station = _stations([(10.9, 50.0)])
    result = bnd.select_stations_for_tile(near_station, tile, buffer_deg=0.1, min_search_size_deg=2.0)
    assert len(result) == 1, result
    print("PASS: station 0.9deg away found - outside the tiny bbox+buffer, but inside the 2deg floor")

    far_station = _stations([(11.1, 50.0)])
    result_far = bnd.select_stations_for_tile(far_station, tile, buffer_deg=0.1, min_search_size_deg=2.0)
    assert len(result_far) == 0, result_far
    print("PASS: station 1.1deg away NOT found - just outside the 2deg floor")
    print()


def test_large_tile_unaffected_by_floor() -> None:
    print("=== large tile: bbox+buffer already exceeds the floor, floor has no effect ===")
    # 12x12 deg bbox+buffer exceeds the 2 deg floor, which must not change it.
    tile = _tile((0.0, 0.0, 10.0, 10.0))
    station = _stations([(-0.9, 5.0)])
    result = bnd.select_stations_for_tile(station, tile, buffer_deg=1.0, min_search_size_deg=2.0)
    assert len(result) == 1, result

    station_far = _stations([(-1.5, 5.0)])
    result_far = bnd.select_stations_for_tile(station_far, tile, buffer_deg=1.0, min_search_size_deg=2.0)
    assert len(result_far) == 0, result_far
    print("PASS: large tile's search area matches plain bbox+buffer behaviour, floor irrelevant")
    print()


def test_zero_min_size_is_plain_bbox_plus_buffer() -> None:
    print("=== min_search_size_deg=0.0: plain bbox+buffer selection, no minimum search size ===")
    tile = _tile((0.0, 0.0, 1.0, 1.0))
    station_in = _stations([(1.4, 0.5)])
    station_out = _stations([(1.6, 0.5)])
    result_in = bnd.select_stations_for_tile(station_in, tile, buffer_deg=0.5, min_search_size_deg=0.0)
    result_out = bnd.select_stations_for_tile(station_out, tile, buffer_deg=0.5, min_search_size_deg=0.0)
    assert len(result_in) == 1, result_in
    assert len(result_out) == 0, result_out
    print("PASS: min_search_size_deg=0.0 behaves exactly like plain bbox+buffer (no floor)")
    print()


def main() -> None:
    test_min_size_floor_kicks_in_for_tiny_tile()
    test_large_tile_unaffected_by_floor()
    test_zero_min_size_is_plain_bbox_plus_buffer()
    print("All station-search validation checks passed.")


if __name__ == "__main__":
    main()
