"""Split an out-of-memory tile into two overlapping sub-tiles.

Solve memory scales with pixel count, so an OOM tile is split rather than retried.
Child ids are `max(tile_id) + 1` onward: collision-free and sorting after all
existing tiles (run_eikonal_dag.py --batch relies on this). Depth is tracked in
the `split_depth` column (0 = original tile).
"""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, box

from config_utils import retry_transient_io
from tiles import load_tile_grid


def split_depth(tile_grid: gpd.GeoDataFrame, tile_id: int) -> int:
    """Return how often `tile_id` has been split (0 if no `split_depth` column)."""
    if "split_depth" not in tile_grid.columns:
        return 0
    row = tile_grid.loc[tile_grid["tile_id"] == tile_id]
    if row.empty:
        raise ValueError(f"tile_id {tile_id} not found in tile grid")
    val = row.iloc[0]["split_depth"]
    return 0 if pd.isna(val) else int(val)


def build_split_candidates(
    geom: Polygon, fraction: float,
) -> dict[str, tuple[Polygon, Polygon]]:
    """Return {"lat": (south, north), "lon": (west, east)} overlapping halves of `geom`.

    `fraction` (e.g. 2/3) is each half's share of the axis; overlap is
    `2*fraction - 1` of the extent.
    """
    minx, miny, maxx, maxy = geom.bounds
    H = maxy - miny
    W = maxx - minx
    south = box(minx, miny, maxx, miny + fraction * H)
    north = box(minx, maxy - fraction * H, maxx, maxy)
    west = box(minx, miny, minx + fraction * W, maxy)
    east = box(maxx - fraction * W, miny, maxx, maxy)
    return {"lat": (south, north), "lon": (west, east)}


def count_land_pixels(mask_path: str | Path, bounds: tuple[float, float, float, float]) -> int:
    """Count land/lake/river pixels (mask in {0, 2, 3}) within `bounds`.

    Out-of-raster reads count as ocean.
    """
    minx, miny, maxx, maxy = bounds
    with retry_transient_io(rasterio.open, mask_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=1)
    return int(np.isin(data, [0, 2, 3]).sum())


def choose_split(
    geom: Polygon, mask_path: str | Path, fraction: float,
) -> tuple[str, Polygon, Polygon]:
    """Pick the split axis that best balances land between halves.

    Returns (axis_name, half_a, half_b). Raises ValueError only in the
    pathological case where every candidate axis produces a zero-land half
    on both sides (caller should fall back to giving up on this tile).
    """
    candidates = build_split_candidates(geom, fraction)
    scored: dict[str, tuple[int, int]] = {
        axis: (count_land_pixels(mask_path, a.bounds), count_land_pixels(mask_path, b.bounds))
        for axis, (a, b) in candidates.items()
    }

    valid = {axis: counts for axis, counts in scored.items() if min(counts) > 0}
    if valid:
        best_axis = min(valid, key=lambda axis: abs(valid[axis][0] - valid[axis][1]))
    else:
        best_axis = max(scored, key=lambda axis: min(scored[axis]))
        print(
            f"  WARNING: no split axis keeps land on both halves (land counts: {scored}); "
            f"falling back to '{best_axis}' (least unbalanced)."
        )

    return best_axis, *candidates[best_axis]


def split_tile(
    tile_id: int,
    tile_grid_path: str | Path,
    model_outputs_dir: str | Path,
    fraction: float,
) -> tuple[int, int]:
    """Split `tile_id` into two children; rewrites the tile grid and deletes its outputs."""
    tile_grid_path = Path(tile_grid_path)
    model_outputs_dir = Path(model_outputs_dir)

    tile_grid = load_tile_grid(tile_grid_path)
    if "split_depth" not in tile_grid.columns:
        tile_grid["split_depth"] = 0
    row = tile_grid.loc[tile_grid["tile_id"] == tile_id]
    if row.empty:
        raise ValueError(f"tile_id {tile_id} not found in {tile_grid_path}")
    row = row.iloc[0]
    parent_depth = 0 if pd.isna(row["split_depth"]) else int(row["split_depth"])

    mask_path = model_outputs_dir / str(tile_id) / "inputs" / "mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Expected {mask_path} (DEM/mask extraction should have completed "
            f"before the eikonal solve could OOM on tile {tile_id})"
        )

    _axis, half_a, half_b = choose_split(row.geometry, mask_path, fraction)
    next_id = int(tile_grid["tile_id"].max()) + 1
    child_a_id, child_b_id = next_id, next_id + 1

    # Children inherit hop_distance (a sub-region keeps the parent's forcing
    # classification); the runners int() it, so it must not be NaN.
    parent_hop_distance = row["hop_distance"]
    new_rows = gpd.GeoDataFrame(
        [
            {"tile_id": child_a_id, "split_depth": parent_depth + 1,
             "hop_distance": parent_hop_distance, "geometry": half_a},
            {"tile_id": child_b_id, "split_depth": parent_depth + 1,
             "hop_distance": parent_hop_distance, "geometry": half_b},
        ],
        crs=tile_grid.crs,
    )
    updated = pd.concat(
        [tile_grid[tile_grid["tile_id"] != tile_id], new_rows], ignore_index=True,
    )
    retry_transient_io(gpd.GeoDataFrame(updated, crs=tile_grid.crs).to_file, tile_grid_path, driver="GPKG")

    shutil.rmtree(model_outputs_dir / str(tile_id), ignore_errors=True)
    (model_outputs_dir / "oom_tiles" / f"{tile_id}.txt").unlink(missing_ok=True)
    for f in (model_outputs_dir / "skipped_tiles").glob(f"{tile_id}_*.txt"):
        f.unlink()

    return child_a_id, child_b_id
