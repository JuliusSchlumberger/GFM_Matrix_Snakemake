"""Merged-chunk lookup and mosaic helpers for the validation scripts.

Moved out of validate_country.py (cleanup Phase 7c) so the standalone England scripts
(validate_england_fz2_band.py, rescore_england_fz3.py) can share them without importing the
whole country-validation module.
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.merge import merge as rasterio_merge
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import retry_transient_io  # noqa: E402
from tiles import chunk_id_from_corner  # noqa: E402


def _postprocess_chunk_id(x: float, y: float, chunk_size_deg: float) -> str:
    """Which postprocessing (chunk_size_deg, typically 5deg) chunk a point falls in."""
    xi = int(math.floor(x / chunk_size_deg) * chunk_size_deg)
    yi = int(math.floor(y / chunk_size_deg) * chunk_size_deg)
    return chunk_id_from_corner(xi, yi)


def _chunks_overlapping_bbox(
    bbox: list[float], pp_chunk_deg: float, merged_chunks_dir: Path, rp: str, slr: str,
) -> list[Path]:
    """Every existing merged waterdepth chunk file whose 5deg cell overlaps `bbox`."""
    minx, miny, maxx, maxy = bbox
    eps = 1e-9
    x0 = math.floor(minx / pp_chunk_deg) * pp_chunk_deg
    x1 = math.floor((maxx - eps) / pp_chunk_deg) * pp_chunk_deg
    y0 = math.floor(miny / pp_chunk_deg) * pp_chunk_deg
    y1 = math.floor((maxy - eps) / pp_chunk_deg) * pp_chunk_deg
    paths = []
    x = x0
    while x <= x1 + eps:
        y = y0
        while y <= y1 + eps:
            chunk_id = _postprocess_chunk_id(x, y, pp_chunk_deg)
            p = merged_chunks_dir / f"waterdepth_{chunk_id}_{rp}_{slr}.tif"
            if p.exists():
                paths.append(p)
            y += pp_chunk_deg
        x += pp_chunk_deg
    return paths


def _mosaic_read(chunk_paths: list[Path], bbox: list[float]) -> tuple[np.ndarray | None, Affine | None]:
    """Mosaic (possibly several) merged waterdepth chunk files onto one array covering `bbox`.

    A region's bounding box is not bounded to a single 5deg
    postprocessing chunk the way a 0.5deg block was guaranteed to be, so
    this can genuinely span more than one source file - rasterio.merge
    handles that directly instead of hand-rolled multi-file windowing.
    """
    if not chunk_paths:
        return None, None
    srcs = [retry_transient_io(rasterio.open, p) for p in chunk_paths]
    try:
        mosaic, out_transform = rasterio_merge(srcs, bounds=bbox, nodata=-9999.0)
    finally:
        for s in srcs:
            s.close()
    return mosaic[0], out_transform


def _tile_paths_overlapping_bbox(
    bbox: list[float], tile_grid: gpd.GeoDataFrame, model_outputs: str, rp: str, slr: str,
) -> list[Path]:
    """Every production tile's own raw `results/waterdepth_{rp}_{slr}.tif` whose
    tile intersects `bbox` AND already exists on disk - the direct per-tile
    counterpart to `_chunks_overlapping_bbox`, for scoring a small, ad hoc
    tile subset that was never (and is not meant to be) run through
    `merge_chunk` (2026-10-08 - a country-validation run over a handful of
    tiles would otherwise require simulating every OTHER tile sharing their
    5deg postprocessing chunk too, see merge_chunk's own `waterdepth_tiles_for_chunk`
    input function in the Snakefile). A tile simply not yet simulated is
    silently skipped here, same spirit as `_chunks_overlapping_bbox` skipping
    a chunk that was never merged - not an error, just not part of this run.
    """
    region_geom = box(*bbox)
    hits = tile_grid[tile_grid.geometry.intersects(region_geom)]
    paths = []
    for tid in hits["tile_id"].astype(int):
        p = Path(model_outputs) / str(tid) / "results" / f"waterdepth_{rp}_{slr}.tif"
        if p.exists():
            paths.append(p)
    return paths
