"""Functions for merging per-tile flood model results into spatial-chunk rasters.

All input tiles share a common pixel grid (extracted from the same DEM mosaic
at the same resolution), so no resampling is needed: offsets between any tile
and the chunk output grid are always whole-pixel.

The merge strategy is:
  - The study area is partitioned into regular chunks (size set by
    `merge.chunk_size_deg` in config).  Each chunk is merged independently.
  - For each chunk, tile files are kept open but data is read block by block
    inside the write loop — only the current block's data is ever in RAM.
  - Overlap zones (cells where ≥2 tiles both report depth ≥ threshold) are
    collected with reservoir-style sub-sampling during the block loop and
    returned for correlation analysis.

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
from rasterio.windows import Window, from_bounds

AQUEDUCT_NODATA = np.finfo(np.float32).max


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


class _PairSamples:
    """Accumulates (xi, xj) depth pairs with bounded memory.

    Periodically sub-samples to ``max_samples * _OVERFLOW_FACTOR`` once the
    buffer grows beyond that, so peak memory per pair is always bounded.
    """

    _OVERFLOW_FACTOR = 10

    def __init__(self, max_samples: int, rng: np.random.Generator) -> None:
        self.max_samples = max_samples
        self.rng = rng
        self._xi: list[np.ndarray] = []
        self._xj: list[np.ndarray] = []
        self._total = 0

    def add(self, xi: np.ndarray, xj: np.ndarray) -> None:
        self._xi.append(xi.astype(np.float32, copy=False))
        self._xj.append(xj.astype(np.float32, copy=False))
        self._total += len(xi)
        cap = self.max_samples * self._OVERFLOW_FACTOR
        if self._total > cap:
            all_xi = np.concatenate(self._xi)
            all_xj = np.concatenate(self._xj)
            idx = self.rng.choice(len(all_xi), self.max_samples, replace=False)
            self._xi = [all_xi[idx]]
            self._xj = [all_xj[idx]]
            self._total = self.max_samples

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._xi:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        xi = np.concatenate(self._xi)
        xj = np.concatenate(self._xj)
        if len(xi) > self.max_samples:
            idx = self.rng.choice(len(xi), self.max_samples, replace=False)
            xi, xj = xi[idx], xj[idx]
        return xi, xj


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
        src = rasterio.open(path)
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


def merge_tile_rasters_chunk(
    tile_rasters: list[str | Path],
    chunk_bounds: tuple[float, float, float, float],
    count_output_path: str | Path,
    waterdepth_output_path: str | Path,
    block_size: int,
    raster_config: dict[str, Any],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Merge per-tile water depth rasters within one spatial chunk.

    Tile files are opened once (for GDAL cache reuse) but data is read one
    block at a time — only the current block's worth of tile data is ever held
    in RAM simultaneously.  This keeps peak memory proportional to
    ``block_size² × tiles_per_block`` rather than to the full intersection area.

    Cross-tile overlap pairs (cells where ≥2 tiles both report depth ≥
    ``raster_config["flood_area_threshold_m"]``) are collected with bounded
    reservoir-style sampling during the block loop and returned for correlation
    plots.

    Args:
        tile_rasters: Paths to the per-tile waterdepth rasters for the chunk.
        chunk_bounds: (minx, miny, maxx, maxy) of the chunk in the tile CRS.
        count_output_path: Path for the flood-count output raster.
        waterdepth_output_path: Path for the averaged water-depth output raster.
        block_size: Side-length in pixels of each write block.
        raster_config: Merge config dict (driver, compression, predictor,
            nodata, flood_area_threshold_m, overlap_corr_max_samples).

    Returns:
        ``overlap_pairs``: mapping ``(tile_i_name, tile_j_name)`` →
        ``(depths_i, depths_j)`` — float32 arrays, sub-sampled to at most
        ``raster_config.get("overlap_corr_max_samples", 50_000)`` per pair.
    """
    if not tile_rasters:
        raise ValueError("tile_rasters must not be empty")

    with rasterio.open(tile_rasters[0]) as ref:
        ref_transform = ref.transform
        crs = ref.crs

    out_transform, out_w, out_h = _make_chunk_transform(chunk_bounds, ref_transform)
    tile_metas = _open_overlapping_tiles(tile_rasters, out_transform, out_w, out_h)

    corr_threshold = float(raster_config.get("flood_area_threshold_m", 0.05))
    max_samples = int(raster_config.get("overlap_corr_max_samples", 50_000))
    rng = np.random.default_rng(42)

    pair_samplers: dict[tuple[str, str], _PairSamples] = {}

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
    count_profile = {**common, "dtype": "uint8", "nodata": 0, "predictor": 2}
    wd_profile = {
        **common,
        "dtype": "float32",
        "nodata": raster_config["nodata"],
        "predictor": raster_config["predictor"],
    }

    try:
        with (
            rasterio.open(count_output_path, "w", **count_profile) as count_dst,
            rasterio.open(waterdepth_output_path, "w", **wd_profile) as wd_dst,
        ):
            for row_off in range(0, out_h, block_size):
                block_h = min(block_size, out_h - row_off)
                for col_off in range(0, out_w, block_size):
                    block_w = min(block_size, out_w - col_off)

                    count_block = np.zeros((block_h, block_w), dtype="uint8")
                    valid_count = np.zeros((block_h, block_w), dtype="float64")
                    depth_sum = np.zeros((block_h, block_w), dtype="float64")

                    # Read one patch per tile that overlaps this block.
                    block_patches: list[tuple[np.ndarray, int, int, int, int, str]] = []

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
                        patch = tm.src.read(
                            1, window=win, boundless=True,
                            fill_value=AQUEDUCT_NODATA,
                        ).astype(np.float32)

                        br0 = out_r0 - row_off
                        br1 = out_r1 - row_off
                        bc0 = out_c0 - col_off
                        bc1 = out_c1 - col_off

                        valid = patch < AQUEDUCT_NODATA
                        count_block[br0:br1, bc0:bc1] += (valid & (patch > 0)).astype("uint8")
                        valid_count[br0:br1, bc0:bc1] += valid.astype("float64")
                        depth_sum[br0:br1, bc0:bc1] += np.where(
                            valid, patch.astype("float64"), 0.0
                        )
                        block_patches.append((patch, br0, br1, bc0, bc1, tm.path))

                    # Correlation: assign each cell to exactly one pair —
                    # the two lowest-indexed tiles (by tile_metas order) that
                    # both flood it.  This ensures each geographic location
                    # contributes at most one (xi, xj) point to the pooled plot
                    # even when three or more tiles overlap the same area.
                    if len(block_patches) >= 2:
                        N = len(block_patches)
                        # Expand patches into a unified (N, H, W) depth array.
                        tile_depths = np.full(
                            (N, block_h, block_w), np.nan, dtype=np.float32
                        )
                        for ti, (patch, br0, br1, bc0, bc1, _) in enumerate(block_patches):
                            tile_depths[ti, br0:br1, bc0:bc1] = patch

                        floods = (
                            (tile_depths >= corr_threshold)
                            & (tile_depths < AQUEDUCT_NODATA)
                            & ~np.isnan(tile_depths)
                        )

                        # For each cell find the index of its first and second
                        # flooded tile.  The canonical pair is (first, second).
                        first_idx = np.full((block_h, block_w), -1, dtype=np.int32)
                        second_idx = np.full((block_h, block_w), -1, dtype=np.int32)
                        for ti in range(N):
                            f = floods[ti]
                            first_idx[(first_idx == -1) & f] = ti
                            already_has_first = (first_idx != -1) & (first_idx != ti)
                            second_idx[already_has_first & (second_idx == -1) & f] = ti

                        for ti in range(N):
                            for tj in range(ti + 1, N):
                                pair_mask = (first_idx == ti) & (second_idx == tj)
                                if not pair_mask.any():
                                    continue
                                di = tile_depths[ti][pair_mask]
                                dj = tile_depths[tj][pair_mask]
                                key = (
                                    Path(block_patches[ti][-1]).parts[-3],
                                    Path(block_patches[tj][-1]).parts[-3],
                                )
                                if key not in pair_samplers:
                                    pair_samplers[key] = _PairSamples(max_samples, rng)
                                pair_samplers[key].add(di, dj)

                    merged = np.where(
                        valid_count > 0,
                        depth_sum / np.where(valid_count > 0, valid_count, 1.0),
                        raster_config["nodata"],
                    ).astype("float32")
                    window = Window(col_off, row_off, block_w, block_h)
                    count_dst.write(count_block, 1, window=window)
                    wd_dst.write(merged, 1, window=window)
    finally:
        for tm in tile_metas:
            tm.src.close()

    overlap_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, sampler in pair_samplers.items():
        xi, xj = sampler.get()
        if len(xi) > 0:
            overlap_pairs[key] = (xi, xj)
    return overlap_pairs
