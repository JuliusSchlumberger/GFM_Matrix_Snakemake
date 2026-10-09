"""Functions for merging per-tile flood model results into spatial-chunk rasters.

All input tiles share a common pixel grid (extracted from the same DEM mosaic
at the same resolution), so no resampling is needed: offsets between any tile
and the chunk output grid are always whole-pixel.

The merge strategy is:
  - The study area is partitioned into regular chunks (size set by
    `merge.chunk_size_deg` in config).  Each chunk is merged independently.
  - For each chunk, tile files are kept open but data is read block by block
    inside the write loop — only the current block's data is ever in RAM.

AQUEDUCT_NODATA (`np.finfo(np.float32).max`) is the sentinel written by the
Aqueduct model for cells it did not compute.  `0.0` means "computed, no
flooding".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine

from config_utils import retry_transient_io
from rasterio.windows import Window, from_bounds

AQUEDUCT_NODATA = np.finfo(np.float32).max

# int16 counterpart of AQUEDUCT_NODATA, for per-tile waterdepth rasters (see
# rasters.encode_waterdepth_cm/WATERDEPTH_NODATA_INT16). Every read in this
# module decodes int16 tiles to this module's existing float32+
# AQUEDUCT_NODATA convention immediately, before any of the merge/overlap-
# correction arithmetic below - none of that arithmetic needs to think in
# int16-centimetres.
WATERDEPTH_NODATA_INT16 = np.iinfo(np.int16).max
WATERDEPTH_SCALE = 100  # cm per metre - must match rasters.WATERDEPTH_SCALE


def decode_waterdepth_array(raw: np.ndarray) -> np.ndarray:
    """Decode an already-read int16-centimetre waterdepth array to float32
    metres, with AQUEDUCT_NODATA marking cells the tile didn't cover/compute.

    Any waterdepth-reading code outside this module (e.g. diagnostic
    scripts that reproject/warp a tile directly rather than doing a plain
    windowed `.read()`) should decode through this function too, rather
    than duplicating the scale/nodata conversion.
    """
    patch = raw.astype(np.float32) / WATERDEPTH_SCALE
    patch[raw == WATERDEPTH_NODATA_INT16] = AQUEDUCT_NODATA
    return patch


def _read_waterdepth_patch(src: rasterio.DatasetReader, window: Window) -> np.ndarray:
    """Read one tile's waterdepth patch, decoded to float32 metres with
    AQUEDUCT_NODATA marking cells the tile didn't cover/compute.
    """
    raw = src.read(1, window=window, boundless=True, fill_value=WATERDEPTH_NODATA_INT16)
    return decode_waterdepth_array(raw)


def _bounds_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return True if bounding boxes a and b (minx, miny, maxx, maxy) overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


@dataclass
class _TileMeta:
    """Open file handle and chunk-intersection geometry for one tile.

    Keeping the file open (rather than re-opening per block) lets GDAL reuse
    its internal cache between block reads.  Call .src.close() when done.
    """
    src: rasterio.DatasetReader
    path: str
    row_off: int    # row in output grid where this tile's intersection starts
    col_off: int    # col in output grid where this tile's intersection starts
    n_rows: int     # height of the intersection (output-grid pixels)
    n_cols: int     # width  of the intersection (output-grid pixels)
    src_r0: int     # first row inside the source tile for that intersection
    src_c0: int     # first col inside the source tile for that intersection


def _make_chunk_transform(
    chunk_bounds: tuple[float, float, float, float],
    ref_transform: Affine,
) -> tuple[Affine, int, int]:
    """Snap chunk bounds to the shared input pixel grid.

    Because all tiles share the same origin and pixel size (same DEM mosaic),
    snapping to the grid ensures whole-pixel alignment between every tile and
    the chunk output, so ``round()`` offsets are exact with no sub-pixel error.

    Returns:
        (out_transform, out_width, out_height)
    """
    px = ref_transform.a
    py = -ref_transform.e
    ox, oy = ref_transform.c, ref_transform.f
    minx, miny, maxx, maxy = chunk_bounds
    c0 = round((minx - ox) / px)
    r0 = round((oy - maxy) / py)
    c1 = round((maxx - ox) / px)
    r1 = round((oy - miny) / py)
    out_transform = Affine(px, 0.0, ox + c0 * px, 0.0, -py, oy - r0 * py)
    return out_transform, c1 - c0, r1 - r0


def _open_overlapping_tiles(
    tile_rasters: list[str | Path],
    out_transform: Affine,
    out_w: int,
    out_h: int,
) -> list[_TileMeta]:
    """Open each tile and compute its intersection with the chunk.

    Files that do not intersect the chunk are closed immediately.  For tiles
    that do intersect, the file is left open so GDAL can reuse its read cache
    across multiple block reads.  The caller must close each ``tm.src`` when
    finished.

    Returns:
        List of _TileMeta, one per overlapping tile.
    """
    px = out_transform.a
    py = -out_transform.e
    ox, oy = out_transform.c, out_transform.f
    chunk_minx = ox
    chunk_maxy = oy
    chunk_maxx = ox + out_w * px
    chunk_miny = oy - out_h * py

    metas: list[_TileMeta] = []
    for path in tile_rasters:
        src = retry_transient_io(rasterio.open, path)
        ix0 = max(chunk_minx, src.bounds.left)
        ix1 = min(chunk_maxx, src.bounds.right)
        iy0 = max(chunk_miny, src.bounds.bottom)
        iy1 = min(chunk_maxy, src.bounds.top)
        if ix0 >= ix1 or iy0 >= iy1:
            src.close()
            continue
        row_off = max(0, round((chunk_maxy - iy1) / py))
        col_off = max(0, round((ix0 - chunk_minx) / px))
        n_rows = max(1, round((iy1 - iy0) / py))
        n_cols = max(1, round((ix1 - ix0) / px))
        src_r0 = max(0, round((src.bounds.top - iy1) / py))
        src_c0 = max(0, round((ix0 - src.bounds.left) / px))
        metas.append(_TileMeta(
            src=src,
            path=str(path),
            row_off=row_off,
            col_off=col_off,
            n_rows=n_rows,
            n_cols=n_cols,
            src_r0=src_r0,
            src_c0=src_c0,
        ))
    return metas


PROVENANCE_NODATA = -1  # no tile covers this cell


def _tile_id_from_path(path: str) -> int:
    """Recover a tile_id from a per-tile waterdepth raster path, relying on
    this pipeline's established `model_outputs/{tile_id}/results/...`
    directory convention (see e.g. compute_model_bbox's own
    `model_outputs/{tile_id}/inputs/...` sibling path). Fragile in the
    sense that it depends on that convention rather than an explicit
    tile_id passed alongside each path - acceptable here since every other
    part of this pipeline already depends on the same convention.
    """
    return int(Path(path).parents[1].name)


def merge_tile_rasters_chunk(
    tile_rasters: list[str | Path],
    chunk_bounds: tuple[float, float, float, float],
    waterdepth_output_path: str | Path,
    provenance_output_path: str | Path,
    block_size: int,
    raster_config: dict[str, Any],
) -> None:
    """Merge per-tile water depth rasters within one spatial chunk by
    PER-CELL MAXIMUM (2026-08 - replaces the previous valid-count-weighted
    mean). Rationale (tile-generation spec): if water reached a cell in one
    tile, it reached it, regardless of what another tile says - this flips
    the error asymmetry so under-resolution in one tile is recoverable by
    any other tile that got it right, while over-estimation is the only
    error a max-combine can no longer self-correct. Also writes a parallel
    PROVENANCE raster (int32 tile_id per winning cell, `PROVENANCE_NODATA`
    where no tile covers the cell at all) - under max-combine this is the
    only practical debugging handle when a value looks wrong, since there's
    no "average" to inspect the contributing tiles of.

    Correctness gate this combine strategy leans on: because no tile can
    ever pull a value back down, a station wrongly assigned across a thin
    land barrier poisons the mosaic PERMANENTLY (unlike under the old mean,
    where it was merely diluted). The native-resolution ocean-connectivity
    fix in flood_model.py remains sole authority on per-cell boundary
    assignment for exactly this reason - it should reject on ambiguity
    rather than accept; the coarse long-range connectivity check
    (boundaries.filter_stations_by_ocean_connectivity) only ever filters
    the candidate station pool, never assigns a value itself.

    Tile files are opened once (for GDAL cache reuse) but data is read one
    block at a time — only the current block's worth of tile data is ever held
    in RAM simultaneously.  This keeps peak memory proportional to
    ``block_size² × tiles_per_block`` rather than to the full intersection area.

    Args:
        tile_rasters: Paths to the per-tile waterdepth rasters for the chunk.
        chunk_bounds: (minx, miny, maxx, maxy) of the chunk in the tile CRS.
        waterdepth_output_path: Path for the max-combined water-depth output raster.
        provenance_output_path: Path for the parallel int32 winning-tile_id raster.
        block_size: Side-length in pixels of each write block.
        raster_config: Raster format config dict (driver, compression,
            predictor, nodata).
    """
    if not tile_rasters:
        raise ValueError("tile_rasters must not be empty")

    with retry_transient_io(rasterio.open, tile_rasters[0]) as ref:
        ref_transform = ref.transform
        crs = ref.crs

    out_transform, out_w, out_h = _make_chunk_transform(chunk_bounds, ref_transform)
    tile_metas = _open_overlapping_tiles(tile_rasters, out_transform, out_w, out_h)
    tile_ids = {tm.path: _tile_id_from_path(tm.path) for tm in tile_metas}

    common = {
        "crs": crs,
        "transform": out_transform,
        "width": out_w,
        "height": out_h,
        "count": 1,
        "driver": raster_config["driver"],
        "compress": raster_config["compression"],
        "tiled": True,
        "bigtiff": "YES",
    }
    wd_profile = {
        **common,
        "dtype": "float32",
        "nodata": raster_config["nodata"],
        "predictor": raster_config["predictor"],
    }
    prov_profile = {
        **common,
        "dtype": "int32",
        "nodata": PROVENANCE_NODATA,
        "predictor": 2,  # integer predictor - tile_ids are arbitrary integers, not a smooth field
    }

    try:
        with retry_transient_io(rasterio.open, waterdepth_output_path, "w", **wd_profile) as wd_dst, \
             retry_transient_io(rasterio.open, provenance_output_path, "w", **prov_profile) as prov_dst:
            for row_off in range(0, out_h, block_size):
                block_h = min(block_size, out_h - row_off)
                for col_off in range(0, out_w, block_size):
                    block_w = min(block_size, out_w - col_off)

                    # Per-cell running maximum + which tile_id is currently
                    # winning (PROVENANCE_NODATA/-inf where no tile has
                    # covered the cell yet) - replaces the pre-2026-08
                    # valid_count-weighted mean. See this function's own
                    # docstring for why max, not mean.
                    best_depth = np.full((block_h, block_w), -np.inf, dtype="float64")
                    best_tile_id = np.full((block_h, block_w), PROVENANCE_NODATA, dtype="int32")

                    # Read one patch per tile that overlaps this block.
                    for tm in tile_metas:
                        out_r0 = max(row_off, tm.row_off)
                        out_r1 = min(row_off + block_h, tm.row_off + tm.n_rows)
                        out_c0 = max(col_off, tm.col_off)
                        out_c1 = min(col_off + block_w, tm.col_off + tm.n_cols)
                        if out_r0 >= out_r1 or out_c0 >= out_c1:
                            continue
                        # Position within the source tile.
                        s_r0 = tm.src_r0 + (out_r0 - tm.row_off)
                        s_c0 = tm.src_c0 + (out_c0 - tm.col_off)
                        win_h = out_r1 - out_r0
                        win_w = out_c1 - out_c0
                        win = Window(s_c0, s_r0, win_w, win_h)
                        patch = _read_waterdepth_patch(tm.src, win)

                        br0 = out_r0 - row_off
                        br1 = out_r1 - row_off
                        bc0 = out_c0 - col_off
                        bc1 = out_c1 - col_off

                        valid = patch < AQUEDUCT_NODATA
                        sub_best = best_depth[br0:br1, bc0:bc1]
                        sub_id = best_tile_id[br0:br1, bc0:bc1]
                        depth64 = patch.astype("float64")
                        wins = valid & (depth64 > sub_best)
                        sub_best[wins] = depth64[wins]
                        sub_id[wins] = tile_ids[tm.path]
                        best_depth[br0:br1, bc0:bc1] = sub_best
                        best_tile_id[br0:br1, bc0:bc1] = sub_id

                    covered = best_tile_id != PROVENANCE_NODATA
                    merged = np.where(covered, best_depth, raster_config["nodata"]).astype("float32")
                    # cm-precision rounding shrinks the compressed file some
                    # (nodata sentinel is an integer value, unaffected).
                    merged = np.round(merged, 2)
                    window = Window(col_off, row_off, block_w, block_h)
                    wd_dst.write(merged, 1, window=window)
                    prov_dst.write(best_tile_id, 1, window=window)
    finally:
        for tm in tile_metas:
            tm.src.close()
