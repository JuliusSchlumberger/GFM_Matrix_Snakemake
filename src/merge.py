"""Merge per-tile flood depth rasters into spatial-chunk rasters.

All tiles share one pixel grid (same DEM mosaic), so tile/chunk offsets are
whole-pixel and no resampling is needed. EIKONAL_NODATA marks cells the eikonal
model did not compute; 0.0 means computed, not flooded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine

from config_utils import retry_transient_io
from rasterio.windows import Window

EIKONAL_NODATA = np.finfo(np.float32).max

# int16-cm per-tile rasters (rasters.encode_waterdepth_cm) are decoded to
# float32 m + EIKONAL_NODATA on read, before any merge arithmetic.
WATERDEPTH_NODATA_INT16 = np.iinfo(np.int16).max
WATERDEPTH_SCALE = 100  # cm per metre - must match rasters.WATERDEPTH_SCALE


def decode_waterdepth_array(raw: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre waterdepth array to float32 metres.

    ``WATERDEPTH_NODATA_INT16`` cells become ``EIKONAL_NODATA``.
    """
    patch = raw.astype(np.float32) / WATERDEPTH_SCALE
    patch[raw == WATERDEPTH_NODATA_INT16] = EIKONAL_NODATA
    return patch


def _read_waterdepth_patch(src: rasterio.DatasetReader, window: Window) -> np.ndarray:
    raw = src.read(1, window=window, boundless=True, fill_value=WATERDEPTH_NODATA_INT16)
    return decode_waterdepth_array(raw)


@dataclass
class _TileMeta:
    # src stays open across block reads so GDAL reuses its cache; caller closes it.
    src: rasterio.DatasetReader
    path: str
    row_off: int    # output-grid offsets/size of the tile-chunk intersection
    col_off: int
    n_rows: int
    n_cols: int
    src_r0: int     # same intersection's origin inside the source tile
    src_c0: int


def _make_chunk_transform(
    chunk_bounds: tuple[float, float, float, float],
    ref_transform: Affine,
) -> tuple[Affine, int, int]:
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
    # Relies on the model_outputs/{tile_id}/results/... path convention.
    return int(Path(path).parents[1].name)


def merge_tile_rasters_chunk(
    tile_rasters: list[str | Path],
    chunk_bounds: tuple[float, float, float, float],
    waterdepth_output_path: str | Path,
    provenance_output_path: str | Path,
    block_size: int,
    raster_config: dict[str, Any],
) -> None:
    """Merge per-tile water depths (m) in one chunk by per-cell maximum.

    Max-combine lets one tile recover another's under-resolution but cannot
    correct over-estimation, so a station wrongly assigned across a thin land
    barrier poisons the mosaic; flood_model.py's native-resolution
    ocean-connectivity check is therefore sole authority on boundary assignment.

    Also writes an int32 provenance raster (winning tile_id, PROVENANCE_NODATA
    where uncovered). ``chunk_bounds`` is (minx, miny, maxx, maxy) in tile CRS.
    Peak memory ~ block_size**2 x tiles_per_block. Outputs are written to temp
    files and os.replace-d so a killed job cannot leave a truncated file that
    Snakemake treats as complete.
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

    # Inline temp+rename (not rasters._atomic_raster_write): two files share one block loop.
    wd_tmp = Path(waterdepth_output_path).with_name(f"{Path(waterdepth_output_path).name}.tmp{os.getpid()}")
    prov_tmp = Path(provenance_output_path).with_name(f"{Path(provenance_output_path).name}.tmp{os.getpid()}")
    try:
        with retry_transient_io(rasterio.open, wd_tmp, "w", **wd_profile) as wd_dst, \
             retry_transient_io(rasterio.open, prov_tmp, "w", **prov_profile) as prov_dst:
            for row_off in range(0, out_h, block_size):
                block_h = min(block_size, out_h - row_off)
                for col_off in range(0, out_w, block_size):
                    block_w = min(block_size, out_w - col_off)

                    best_depth = np.full((block_h, block_w), -np.inf, dtype="float64")
                    best_tile_id = np.full((block_h, block_w), PROVENANCE_NODATA, dtype="int32")

                    for tm in tile_metas:
                        out_r0 = max(row_off, tm.row_off)
                        out_r1 = min(row_off + block_h, tm.row_off + tm.n_rows)
                        out_c0 = max(col_off, tm.col_off)
                        out_c1 = min(col_off + block_w, tm.col_off + tm.n_cols)
                        if out_r0 >= out_r1 or out_c0 >= out_c1:
                            continue
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

                        valid = patch < EIKONAL_NODATA
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
                    # cm rounding shrinks the compressed file; integer nodata unaffected.
                    merged = np.round(merged, 2)
                    window = Window(col_off, row_off, block_w, block_h)
                    wd_dst.write(merged, 1, window=window)
                    prov_dst.write(best_tile_id, 1, window=window)
        retry_transient_io(os.replace, wd_tmp, waterdepth_output_path)
        retry_transient_io(os.replace, prov_tmp, provenance_output_path)
    except BaseException:
        wd_tmp.unlink(missing_ok=True)
        prov_tmp.unlink(missing_ok=True)
        raise
    finally:
        for tm in tile_metas:
            tm.src.close()
