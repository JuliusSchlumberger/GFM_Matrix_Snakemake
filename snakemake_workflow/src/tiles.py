"""Functions for loading tiles from the overlapping tile grid and selecting
tiles for which the DEM has data coverage.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.ops import unary_union


def load_tile_grid(tile_grid_path: str | Path) -> gpd.GeoDataFrame:
    """Load the overlapping tile grid produced by `python/tile_mask_creation.py`."""
    return gpd.read_file(tile_grid_path)


def get_tile_geometry(tile_grid: gpd.GeoDataFrame, tile_id: int) -> gpd.GeoDataFrame:
    """Select a single tile from the tile grid by its tile_id.

    Args:
        tile_grid: The full tile grid, as returned by `load_tile_grid`.
        tile_id: Unique identifier of the tile to select.

    Returns:
        A single-row GeoDataFrame containing the selected tile.

    Raises:
        ValueError: If `tile_id` is not present in `tile_grid`.
    """
    tile = tile_grid[tile_grid["tile_id"] == tile_id]
    if tile.empty:
        raise ValueError(f"tile_id {tile_id} not found in tile grid")
    return tile.reset_index(drop=True)


def save_tile_geometry(tile: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save a single tile geometry to a GeoPackage file."""
    tile.to_file(output_path, driver="GPKG")


def has_dem_coverage(dem_path: str | Path, bbox: tuple[float, float, float, float], sample_size: int = 128) -> bool:
    """Check whether a DEM raster has any valid (non-nodata) data within a bounding box.

    The bounding box is read at a coarse resolution (at most `sample_size`
    pixels per side), so that large DEM mosaics (e.g. global VRTs) can be
    checked cheaply for many tiles.

    Args:
        dem_path: Path to the DEM raster (e.g. a VRT mosaic).
        bbox: Bounding box as `(minx, miny, maxx, maxy)`.
        sample_size: Maximum number of pixels per side to read.

    Returns:
        True if `dem_path` has at least one valid pixel within `bbox`,
        False if `bbox` does not overlap the raster at all or only
        overlaps nodata pixels.
    """
    with rasterio.open(dem_path) as src:
        full_window = Window(0, 0, src.width, src.height)
        try:
            window = from_bounds(*bbox, transform=src.transform).intersection(full_window)
        except WindowError:
            return False

        out_shape = (
            max(1, min(sample_size, round(window.height))),
            max(1, min(sample_size, round(window.width))),
        )
        data = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.nearest)

        if src.nodata is None:
            return True
        return bool(np.any(data != src.nodata))


def filter_tiles_with_dem_coverage(
    tile_grid: gpd.GeoDataFrame,
    dem_path: str | Path,
    sample_size: int = 128,
) -> gpd.GeoDataFrame:
    """Filter a tile grid to tiles for which the DEM has data coverage.

    Tiles without DEM coverage (see `has_dem_coverage`) are logged and
    excluded from the result.

    Args:
        tile_grid: Tile grid, as returned by `load_tile_grid`.
        dem_path: Path to the DEM raster (e.g. a VRT mosaic).
        sample_size: Passed to `has_dem_coverage`.

    Returns:
        The subset of `tile_grid` for which the DEM has data coverage.
    """
    keep = []
    for _, row in tile_grid.iterrows():
        covered = has_dem_coverage(dem_path, row.geometry.bounds, sample_size)
        if not covered:
            print(f"Tile {int(row['tile_id'])}: no DEM coverage, skipping")
        keep.append(covered)
    return tile_grid[keep].reset_index(drop=True)


def _fraction_for_window(src: rasterio.DatasetReader, bounds: tuple[float, float, float, float], ocean_code: int, nodata: int) -> tuple[float, float]:
    """Compute the ocean and land fraction of a bounding box from an open land use raster.

    Args:
        src: Open land use raster.
        bounds: Bounding box as `(minx, miny, maxx, maxy)`, in `src`'s CRS.
        ocean_code: Land use class code representing open water/ocean.
        nodata: Land use class code representing no data, excluded from
            both fractions.

    Returns:
        A tuple `(ocean_fraction, land_fraction)`, each in [0, 1] and
        summing to 1, except when `bounds` has no valid pixels at all, in
        which case both are 0.
    """
    window = from_bounds(*bounds, transform=src.transform)
    data = src.read(1, window=window, boundless=True, fill_value=nodata)

    valid = data != nodata
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 0.0

    ocean_fraction = float(np.sum(valid & (data == ocean_code))) / n_valid
    return ocean_fraction, 1.0 - ocean_fraction


def compute_land_ocean_fractions(
    tile_grid: gpd.GeoDataFrame,
    land_use_path: str | Path,
    ocean_code: int,
    nodata: int = 255,
) -> gpd.GeoDataFrame:
    """Compute the ocean and land area fractions of each tile from a land use raster.

    For each tile, the full-resolution land use data within the tile's
    bounding box is read and classified into ocean (`ocean_code`), land
    (anything else) and nodata pixels. Tiles outside the raster's extent
    read as all-nodata and get a fraction of 0 for both.

    Args:
        tile_grid: Tile grid, as returned by `load_tile_grid`.
        land_use_path: Path to a global land use raster (e.g. Copernicus
            LandUse) in the same CRS as `tile_grid`.
        ocean_code: Land use class code representing open water/ocean.
        nodata: Land use class code representing no data, excluded from
            both fractions.

    Returns:
        `tile_grid` with added `ocean_fraction` and `land_fraction` columns,
        each the fraction (0-1) of valid (non-nodata) pixels in the tile's
        bounding box. The two fractions sum to 1, except for tiles with no
        valid pixels at all, where both are 0.
    """
    ocean_fractions = []
    land_fractions = []
    with rasterio.open(land_use_path) as src:
        for _, row in tile_grid.iterrows():
            ocean_fraction, land_fraction = _fraction_for_window(src, row.geometry.bounds, ocean_code, nodata)
            ocean_fractions.append(ocean_fraction)
            land_fractions.append(land_fraction)

    tile_grid = tile_grid.copy()
    tile_grid["ocean_fraction"] = ocean_fractions
    tile_grid["land_fraction"] = land_fractions
    return tile_grid


def _is_cardinal_neighbor(geom_a, geom_b, threshold: float = 0.5) -> bool:
    """Check whether two axis-aligned tile geometries are immediate top/bottom/left/right neighbors.

    Two tiles are cardinal neighbors if their bounding boxes overlap and
    that overlap spans at least `threshold` of the smaller tile's width (a
    north/south relationship) or height (an east/west relationship). Tiles
    that only overlap in a small corner square (diagonal neighbors) fail
    both checks and are excluded.

    Args:
        geom_a: First tile geometry.
        geom_b: Second tile geometry.
        threshold: Minimum fraction of the smaller tile's width or height
            that the overlap must span to count as a cardinal neighbor.

    Returns:
        True if `geom_a` and `geom_b` are cardinal (N/S/E/W) neighbors.
    """
    ax0, ay0, ax1, ay1 = geom_a.bounds
    bx0, by0, bx1, by1 = geom_b.bounds

    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_y = min(ay1, by1) - max(ay0, by0)
    if overlap_x <= 0 or overlap_y <= 0:
        return False

    width = min(ax1 - ax0, bx1 - bx0)
    height = min(ay1 - ay0, by1 - by0)
    return overlap_x >= threshold * width or overlap_y >= threshold * height


def _pick_merge_partner(candidates: list[tuple], own_size: float):
    """Pick a merge partner from candidates, preferring equal size, then smallest, then max overlap.

    Args:
        candidates: List of `(other_id, other_size, overlap_area, ...)` tuples.
        own_size: Area of the tile looking for a merge partner.

    Returns:
        The candidate tuple with the highest overlap area, among those with
        the same size as `own_size` if any exist, otherwise among those
        with the smallest size.
    """
    same_size = [c for c in candidates if np.isclose(c[1], own_size)]
    pool = same_size if same_size else candidates
    if not same_size:
        smallest_size = min(c[1] for c in pool)
        pool = [c for c in pool if np.isclose(c[1], smallest_size)]
    return max(pool, key=lambda c: c[2])


def merge_undersized_tiles(
    tile_grid: gpd.GeoDataFrame,
    land_use_path: str | Path,
    ocean_code: int,
    min_fraction: float,
    nodata: int = 255,
) -> gpd.GeoDataFrame:
    """Merge tiles with too little ocean or land coverage into a neighboring tile.

    Tiles where `ocean_fraction` or `land_fraction` (see
    `compute_land_ocean_fractions`) is below `min_fraction` are "bad" and are
    resolved one at a time:

    0. Dropping tiles with no land at all: if a bad tile has `land_fraction`
       of exactly 0 (100% ocean), it is dropped from the grid outright. There
       is nothing to flood-model there, and merging it would only dilute a
       neighbor's ocean/land balance.
    1. Unionizing with a "bad" neighbor: if a cardinal (immediate
       top/bottom/left/right) neighbor is also bad, and the bounding box of
       their union meets `min_fraction` for both ocean and land, the two
       tiles are merged into one "good" tile. This is preferred, since it
       resolves two bad tiles at once without growing an existing tile.
    2. Absorbing into a "good" neighbor: otherwise, the bad tile is merged
       into a cardinal "good" neighbor (one that already meets both
       thresholds). Neighbors with the same area as the bad tile are
       preferred; if none exist, the smallest "good" neighbor is used. Among
       the resulting candidates, the one with the largest overlap area with
       the bad tile is chosen. A "good" tile may absorb more than one bad
       tile this way.

    For steps 1 and 2, the surviving tile's geometry becomes the bounding box
    of the union of both tiles' geometries, and the other tile is dropped from
    the grid.

    Bad tiles with none of the above applicable (e.g. small islands
    surrounded by ocean) are kept unchanged.

    Args:
        tile_grid: Tile grid with `ocean_fraction` and `land_fraction`
            columns, as returned by `compute_land_ocean_fractions`.
        land_use_path: Path to the land use raster used by
            `compute_land_ocean_fractions`, used here to evaluate candidate
            bad-bad unions.
        ocean_code: Land use class code representing open water/ocean.
        min_fraction: Minimum required fraction of ocean and of land for a
            tile (or a candidate union) to be considered "good".
        nodata: Land use class code representing no data, excluded from
            both fractions.

    Returns:
        `tile_grid` with bad tiles either merged into a neighbor or, if no
        suitable neighbor exists, left unchanged. The `ocean_fraction` and
        `land_fraction` columns are dropped, since they no longer apply to
        tiles whose geometry changed.
    """
    grid = tile_grid.set_index("tile_id", drop=False)
    good = (grid["ocean_fraction"] >= min_fraction) & (grid["land_fraction"] >= min_fraction)
    bad_tile_ids = grid.index[~good].tolist()

    with rasterio.open(land_use_path) as src:
        for tile_id in bad_tile_ids:
            if tile_id not in grid.index or good[tile_id]:
                continue  # already resolved by a previous tile's merge

            # 0. Drop tiles with no land at all (100% ocean) outright.
            if grid.loc[tile_id, "land_fraction"] == 0:
                grid = grid.drop(index=tile_id)
                continue

            bad_geom = grid.loc[tile_id, "geometry"]
            bad_size = bad_geom.area

            # 1. Unionize with a bad cardinal neighbor, if the union becomes good.
            unionize_candidates = []
            for other_id in bad_tile_ids:
                if other_id == tile_id or other_id not in grid.index or good[other_id]:
                    continue
                other_geom = grid.loc[other_id, "geometry"]
                if not _is_cardinal_neighbor(bad_geom, other_geom):
                    continue
                union_geom = box(*unary_union([bad_geom, other_geom]).bounds)
                ocean_fraction, land_fraction = _fraction_for_window(src, union_geom.bounds, ocean_code, nodata)
                if ocean_fraction >= min_fraction and land_fraction >= min_fraction:
                    overlap_area = bad_geom.intersection(other_geom).area
                    unionize_candidates.append((other_id, other_geom.area, overlap_area, union_geom))

            if unionize_candidates:
                chosen_id, _, _, union_geom = _pick_merge_partner(unionize_candidates, bad_size)
                grid.loc[chosen_id, "geometry"] = union_geom
                good[chosen_id] = True
                grid = grid.drop(index=tile_id)
                continue

            # 2. Absorb into a good cardinal neighbor.
            candidates = []
            for other_id in grid.index:
                if other_id == tile_id or not good[other_id]:
                    continue
                other_geom = grid.loc[other_id, "geometry"]
                if not _is_cardinal_neighbor(bad_geom, other_geom):
                    continue
                overlap_area = bad_geom.intersection(other_geom).area
                if overlap_area > 0:
                    candidates.append((other_id, other_geom.area, overlap_area))

            if not candidates:
                continue  # no suitable neighbor, e.g. an island - keep as-is

            chosen_id = _pick_merge_partner(candidates, bad_size)[0]
            merged_geom = box(*unary_union([bad_geom, grid.loc[chosen_id, "geometry"]]).bounds)
            grid.loc[chosen_id, "geometry"] = merged_geom
            grid = grid.drop(index=tile_id)

    return grid.drop(columns=["ocean_fraction", "land_fraction"]).reset_index(drop=True)
