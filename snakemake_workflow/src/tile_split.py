"""Split an OOM'd Aqueduct tile into two smaller, overlapping sub-tiles.

Aqueduct's memory cost is dominated by tile pixel count (see
aqueduct_runner.OOM_SIGNATURE docstring), so a tile that runs out of memory
is unlikely to ever succeed at its current size. Splitting it into two
overlapping halves and re-running the normal pipeline on those instead is
cheaper than trying to make Aqueduct itself more memory-efficient.

Tile ID scheme: existing tile_ids are `parent_5deg_id * 10 + quadrant_id`
(tile_mask_creation.py), where quadrant_id is always 0-3 - no existing
tile_id can end in a digit 4-9. A split child's id is `parent_tile_id * 10 +
{8, 9}`, which is therefore collision-free by construction, keeps tile_id a
plain int everywhere in the pipeline (no "a"/"b" string suffix needed), and
encodes split depth in the digits themselves (see split_depth) - no separate
tracking file required.
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

# Split children's tile_ids always end in one of these digits (never used by
# an original quadrant tile_id, which only ever ends in 0-3).
_SPLIT_DIGITS = (8, 9)


def split_depth(tile_id: int) -> int:
    """Number of times `tile_id` has already been split.

    0 for an original tile (parent_5deg_id*10+quadrant_id, last digit 0-3).
    Each split appends one more trailing digit from {8, 9}, so depth is just
    the count of trailing digits in {8, 9} when peeling them off the right.
    """
    depth = 0
    n = tile_id
    while n % 10 in _SPLIT_DIGITS:
        depth += 1
        n //= 10
    return depth


def build_split_candidates(
    geom: Polygon, fraction: float,
) -> dict[str, tuple[Polygon, Polygon]]:
    """Build the two candidate ways to split `geom` into overlapping halves.

    `fraction` (e.g. 2/3) is the share of the axis each half spans, so the
    overlap in the middle is `2*fraction - 1` of that axis's extent.

    Returns {"lat": (south, north), "lon": (west, east)}.
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
    """Count land/lake/river pixels (mask value in {0, 2, 3}) within `bounds`
    of the per-tile mask.tif produced by extract_dem_mask. Out-of-raster
    reads are treated as ocean (non-land), the conservative default."""
    minx, miny, maxx, maxy = bounds
    with retry_transient_io(rasterio.open, mask_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=1)
    return int(np.isin(data, [0, 2, 3]).sum())


def choose_split(
    geom: Polygon, mask_path: str | Path, fraction: float,
) -> tuple[str, Polygon, Polygon]:
    """Pick the split axis that best balances land coverage between the two
    halves, never choosing an axis where either half has zero land.

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
        # Pathological tile shape: every axis leaves one half with no land.
        # Fall back to whichever axis is least bad (largest minimum half-count).
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
    """Split `tile_id` into two overlapping children, updating tile_grid_path
    and removing the failed tile's stale outputs/markers.

    Returns the two new child tile_ids.
    """
    tile_grid_path = Path(tile_grid_path)
    model_outputs_dir = Path(model_outputs_dir)

    tile_grid = load_tile_grid(tile_grid_path)
    row = tile_grid.loc[tile_grid["tile_id"] == tile_id]
    if row.empty:
        raise ValueError(f"tile_id {tile_id} not found in {tile_grid_path}")
    row = row.iloc[0]

    mask_path = model_outputs_dir / str(tile_id) / "inputs" / "mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Expected {mask_path} (DEM/mask extraction should have completed "
            f"before Aqueduct could OOM on tile {tile_id})"
        )

    _axis, half_a, half_b = choose_split(row.geometry, mask_path, fraction)
    child_a_id = tile_id * 10 + _SPLIT_DIGITS[0]
    child_b_id = tile_id * 10 + _SPLIT_DIGITS[1]

    new_rows = gpd.GeoDataFrame(
        [
            {"tile_id": child_a_id, "parent_grid": row["parent_grid"], "geometry": half_a},
            {"tile_id": child_b_id, "parent_grid": row["parent_grid"], "geometry": half_b},
        ],
        crs=tile_grid.crs,
    )
    updated = pd.concat(
        [tile_grid[tile_grid["tile_id"] != tile_id], new_rows], ignore_index=True,
    )
    retry_transient_io(gpd.GeoDataFrame(updated, crs=tile_grid.crs).to_file, tile_grid_path, driver="GPKG")

    # Remove the failed tile's stale outputs so nothing downstream references it.
    shutil.rmtree(model_outputs_dir / str(tile_id), ignore_errors=True)
    (model_outputs_dir / "oom_tiles" / f"{tile_id}.txt").unlink(missing_ok=True)
    for f in (model_outputs_dir / "skipped_tiles").glob(f"{tile_id}_*.txt"):
        f.unlink()

    return child_a_id, child_b_id
