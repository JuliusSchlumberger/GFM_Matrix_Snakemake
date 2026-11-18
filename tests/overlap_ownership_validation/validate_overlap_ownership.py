"""Ownership of overlaps between selected SFINCS tiles (sfincs_grid.core_mask)."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
from sfincs_grid import core_mask, load_footprints  # noqa: E402

CRS = "EPSG:32631"
SHAPE = (300, 400)
TRANSFORM = from_origin(450_000, 5_560_000, 500, 500)


def _footprints(tmp_path):
    tiles = gpd.GeoDataFrame(
        {"tile_id": [1, 2, 3, 4], "category": ["selected", "selected", "selected", "eligible"]},
        geometry=[box(2.4, 49.0, 3.4, 50.0), box(3.2, 49.2, 3.8, 49.8), box(3.7, 49.3, 4.0, 49.6),
                  box(2.4, 49.0, 4.0, 50.0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "tile_selection.gpkg"
    tiles.to_file(path, layer="tiles", driver="GPKG")
    return tiles, load_footprints(path)


def test_every_covered_cell_has_exactly_one_owner(tmp_path) -> None:
    tiles, footprints = _footprints(tmp_path)
    cores = [core_mask(SHAPE, TRANSFORM, CRS, tid, footprints) for tid in footprints.index]
    selected = tiles[tiles.category == "selected"].geometry.segmentize(0.01).to_crs(CRS)
    covered = geometry_mask(list(selected), out_shape=SHAPE, transform=TRANSFORM, invert=True)
    owners = np.sum(cores, axis=0)
    assert owners[covered].min() == 1 and owners.max() == 1
    assert not owners[~covered].any()


def test_overlap_goes_to_the_larger_tile(tmp_path) -> None:
    _, footprints = _footprints(tmp_path)
    assert list(footprints.index) == ["1", "2", "3"]  # by area; the eligible tile takes no part
    big, mid = (core_mask(SHAPE, TRANSFORM, CRS, t, footprints) for t in ("1", "2"))
    both = gpd.GeoSeries([box(3.2, 49.2, 3.4, 49.8)], crs="EPSG:4326").segmentize(0.01).to_crs(CRS)
    overlap = geometry_mask(list(both), out_shape=SHAPE, transform=TRANSFORM, invert=True)
    assert big[overlap].mean() > 0.99 and not mid[overlap].any()
