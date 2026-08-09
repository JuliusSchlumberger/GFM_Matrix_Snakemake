"""Builds the postprocessing chunk grid (see rules/postprocessing.smk's
merge_chunk and friends) and the chunk_id -> tile_id lookup it's built from.

Extracted from the Snakefile's own inline `_build_chunk_grid` (previously
defined only there) into a shared module so `list_region_chunks.py` (used by
run_postprocess_regions.sh) can compute the exact same chunk grid the
Snakefile itself uses, without re-deriving or duplicating this logic -
mirrors regions.py/list_regions.py's existing single-source-of-truth
pattern for the analogous simulate-side region grouping.

Each chunk is `postprocessing.chunk_size_deg` degrees square, identified by
the N/S latitude and E/W longitude of its south-west corner (e.g. "S10E030"
for the chunk at lon=30, lat=-10). The grid is built programmatically from
the tile grid's own bounds - no external file needed.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box as _shapely_box


def build_chunk_grid(tile_gdf: gpd.GeoDataFrame, chunk_size_deg: float) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of chunk_id/bounds/geometry, one row per chunk
    that actually intersects at least one tile (empty-ocean chunks outside
    the tile grid's footprint are never created).

    Args:
        tile_gdf: Tile grid, as returned by tiles.load_tile_grid - must have
            polygon geometries; only `.total_bounds` and `.geometry` are used.
        chunk_size_deg: Chunk edge length in degrees (config's
            `postprocessing.chunk_size_deg`).
    """
    minx, miny, maxx, maxy = tile_gdf.total_bounds
    sz = chunk_size_deg
    xs = np.arange(np.floor(minx / sz) * sz, np.ceil(maxx / sz) * sz, sz)
    ys = np.arange(np.floor(miny / sz) * sz, np.ceil(maxy / sz) * sz, sz)
    rows = []
    for x in xs:
        for y in ys:
            cell = _shapely_box(x, y, x + sz, y + sz)
            if not tile_gdf.geometry.intersects(cell).any():
                continue
            xi, yi = int(round(x)), int(round(y))
            lat = f"N{yi:02d}" if yi >= 0 else f"S{-yi:02d}"
            lon = f"E{xi:03d}" if xi >= 0 else f"W{-xi:03d}"
            rows.append({
                "chunk_id": f"{lat}{lon}",
                "bounds": [x, y, x + sz, y + sz],
                "geometry": cell,
            })
    return gpd.GeoDataFrame(rows, crs=tile_gdf.crs)


def chunk_tile_lookup(chunk_grid: gpd.GeoDataFrame, tile_gdf: gpd.GeoDataFrame) -> dict[str, list[int]]:
    """Return {chunk_id: [tile_id, ...]} - every tile_id whose polygon
    intersects that chunk's cell (a tile can appear under more than one
    chunk_id if it straddles a chunk boundary; a chunk can list tiles
    belonging to different simulate_region groupings - see
    list_region_chunks.py, which uses this to tell "fully-contained-in-one-
    region" chunks apart from chunks straddling a region boundary)."""
    return {
        row["chunk_id"]: tile_gdf.loc[
            tile_gdf.geometry.intersects(row["geometry"]), "tile_id"
        ].astype(int).tolist()
        for _, row in chunk_grid.iterrows()
    }


def safe_chunks_for_region(
    chunk_lookup: dict[str, list[int]],
    tile_regions: dict[int, str],
    region: str,
) -> tuple[list[str], list[str]]:
    """Split every chunk_id touching `region` into (safe, partial).

    `region` is a name from regions.assign_regions (e.g. "Europe_West") -
    see rule simulate_region/run_simulate_regions.sh. Postprocessing has no
    awareness of this grouping on its own (merge_chunk reads whichever
    tiles a chunk needs, regardless of which region simulated them), so
    postprocessing a region's chunks before every OTHER region touching
    those same chunks is done would either fail on missing input files (a
    tile not yet simulated) or - if that tile happens to exist from a
    partially-run different region - silently merge in a mix of finished
    and not-yet-representative coverage. This tells the two cases apart:

    safe: every tile in the chunk belongs to `region` - postprocessable as
        soon as `region`'s own simulate output is done, regardless of any
        other region's progress.
    partial: the chunk contains at least one tile from `region` AND at
        least one tile assigned to a different region - NOT
        postprocessable until every region touching this chunk is done.

    Returns:
        (safe_chunk_ids, partial_chunk_ids), both sorted for determinism.
    """
    safe, partial = [], []
    for chunk_id, tile_ids in chunk_lookup.items():
        regions_in_chunk = {tile_regions[tid] for tid in tile_ids}
        if region not in regions_in_chunk:
            continue
        if regions_in_chunk == {region}:
            safe.append(chunk_id)
        else:
            partial.append(chunk_id)
    return sorted(safe), sorted(partial)
