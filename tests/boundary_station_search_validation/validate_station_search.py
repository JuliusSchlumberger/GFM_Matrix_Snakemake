"""Validate `select_stations_for_tile`'s minimum search-area floor
(src/boundaries.py, 2026-08) - added because many chunks are now very
small (post drop_redundant_chunks/split_oversized_chunks), and relying on
`bbox + station_search_buffer_deg` alone to guarantee a reasonable search
area was an accident of the buffer's specific value, not a real guarantee.

Pure geometry-logic unit tests - no raster/NetCDF I/O.

Usage:
    python validate_station_search.py
"""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

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
    # tiny 0.01deg tile centred at (10, 50); buffer_deg=0.1 alone would only
    # give a 0.21deg search box - min_search_size_deg=2.0 must dominate.
    tile = _tile((9.995, 49.995, 10.005, 50.005))
    # station just outside the tiny buffered box but well within the 2deg floor
    near_station = _stations([(10.9, 50.0)])  # 0.9deg away - inside 2deg floor (radius 1.0), outside 0.1deg buffer
    result = bnd.select_stations_for_tile(near_station, tile, buffer_deg=0.1, min_search_size_deg=2.0)
    assert len(result) == 1, result
    print("PASS: station 0.9deg away found - outside the tiny bbox+buffer, but inside the 2deg floor")

    far_station = _stations([(11.1, 50.0)])  # 1.1deg away - just outside the 2deg floor (radius 1.0)
    result_far = bnd.select_stations_for_tile(far_station, tile, buffer_deg=0.1, min_search_size_deg=2.0)
    assert len(result_far) == 0, result_far
    print("PASS: station 1.1deg away NOT found - just outside the 2deg floor")
    print()


def test_large_tile_unaffected_by_floor() -> None:
    print("=== large tile: bbox+buffer already exceeds the floor, floor has no effect ===")
    # 10x10deg tile with buffer=1.0deg -> 12x12deg search box, far bigger
    # than a 2deg floor - floor must not shrink or otherwise change this.
    tile = _tile((0.0, 0.0, 10.0, 10.0))
    station = _stations([(-0.9, 5.0)])  # 0.9deg outside the tile's own edge, inside the 1deg buffer
    result = bnd.select_stations_for_tile(station, tile, buffer_deg=1.0, min_search_size_deg=2.0)
    assert len(result) == 1, result

    station_far = _stations([(-1.5, 5.0)])  # 1.5deg outside - beyond both the buffer AND the floor
    result_far = bnd.select_stations_for_tile(station_far, tile, buffer_deg=1.0, min_search_size_deg=2.0)
    assert len(result_far) == 0, result_far
    print("PASS: large tile's search area matches plain bbox+buffer behaviour, floor irrelevant")
    print()


def test_zero_min_size_matches_old_behaviour() -> None:
    print("=== min_search_size_deg=0.0: identical to the pre-2026-08 bbox+buffer-only behaviour ===")
    tile = _tile((0.0, 0.0, 1.0, 1.0))
    station_in = _stations([(1.4, 0.5)])   # 0.4deg beyond the tile's edge - inside a 0.5deg buffer
    station_out = _stations([(1.6, 0.5)])  # 0.6deg beyond - outside a 0.5deg buffer
    result_in = bnd.select_stations_for_tile(station_in, tile, buffer_deg=0.5, min_search_size_deg=0.0)
    result_out = bnd.select_stations_for_tile(station_out, tile, buffer_deg=0.5, min_search_size_deg=0.0)
    assert len(result_in) == 1, result_in
    assert len(result_out) == 0, result_out
    print("PASS: min_search_size_deg=0.0 behaves exactly like plain bbox+buffer (no floor)")
    print()


def main() -> None:
    test_min_size_floor_kicks_in_for_tiny_tile()
    test_large_tile_unaffected_by_floor()
    test_zero_min_size_matches_old_behaviour()
    print("All station-search validation checks passed.")


if __name__ == "__main__":
    main()
