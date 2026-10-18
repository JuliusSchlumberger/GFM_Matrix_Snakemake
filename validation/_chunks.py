"""Merged-chunk lookup and mosaic helpers shared by the validation scripts."""

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
    xi = int(math.floor(x / chunk_size_deg) * chunk_size_deg)
    yi = int(math.floor(y / chunk_size_deg) * chunk_size_deg)
    return chunk_id_from_corner(xi, yi)


def _chunks_overlapping_bbox(
    bbox: list[float], pp_chunk_deg: float, merged_chunks_dir: Path, rp: str, slr: str,
) -> list[Path]:
    """Chunks never merged are silently skipped."""
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
    """Per-tile counterpart of `_chunks_overlapping_bbox` for unmerged tile subsets.

    Tiles not yet simulated are silently skipped.
    """
    region_geom = box(*bbox)
    hits = tile_grid[tile_grid.geometry.intersects(region_geom)]
    paths = []
    for tid in hits["tile_id"].astype(int):
        p = Path(model_outputs) / str(tid) / "results" / f"waterdepth_{rp}_{slr}.tif"
        if p.exists():
            paths.append(p)
    return paths
