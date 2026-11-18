"""Production tiles under a local-input footprint, closed over hop seeding."""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preparation"))
from select_country_tiles import _hop_distance_closure, core_tile_ids_under_footprint  # noqa: E402


def _grid() -> gpd.GeoDataFrame:
    # Three 1-degree tiles in a row; tile 3 (hop 1) is seeded by tile 2 (hop 0).
    return gpd.GeoDataFrame({"tile_id": [1, 2, 3], "hop_distance": [0, 0, 1]},
                            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)], crs="EPSG:4326")


def test_footprint_core_excludes_tiles_that_only_touch() -> None:
    fine = gpd.GeoDataFrame(geometry=[box(2.2, 0.2, 2.4, 0.4), box(2.4, 0.2, 2.6, 0.4)], crs="EPSG:4326")
    assert core_tile_ids_under_footprint(_grid(), fine) == {3}
    edge = gpd.GeoDataFrame(geometry=[box(0.5, 0, 1, 1)], crs="EPSG:4326")
    assert core_tile_ids_under_footprint(_grid(), edge) == {1}


def test_footprint_in_another_crs_selects_the_same_tiles() -> None:
    fine = gpd.GeoDataFrame(geometry=[box(2.2, 0.2, 2.4, 0.4)], crs="EPSG:4326").to_crs("EPSG:3857")
    assert core_tile_ids_under_footprint(_grid(), fine) == {3}


def test_closure_adds_the_seeding_tile() -> None:
    assert _hop_distance_closure(_grid(), {3}) == {2, 3}
