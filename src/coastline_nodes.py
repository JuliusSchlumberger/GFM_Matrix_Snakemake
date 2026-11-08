"""Coastline nodes of the storm-tide product: DeltaDTM coastline cells binned to the FES2022 grid.

Entry point: `coastline_nodes`. The coastline is `flood_model.coastline_mask` on each 1-degree
DeltaDTM mask tile, the definition the eikonal seeds use.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from config_utils import retry_transient_io
from flood_model import coastline_mask
from tiles import _scan_mask_dir

# The FES2022 grid step: the tide at a node is FES's, which does not resolve finer coastline detail.
NODE_BIN_DEG = 1.0 / 30.0


def _read(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as src:
        return src.read(1), src.transform


def _tile_nodes(args) -> np.ndarray:
    path, ocean_code, river_code = args
    mask, tr = retry_transient_io(_read, path)
    rows, cols = np.nonzero(coastline_mask(mask, ocean_code, river_code))
    if not len(rows):
        return np.empty((0, 2))
    x, y = tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e
    key = np.floor(x / NODE_BIN_DEG).astype(np.int64) * 100_000 + np.floor(y / NODE_BIN_DEG).astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv)
    return np.column_stack([np.bincount(inv, x) / n, np.bincount(inv, y) / n])


def coastline_nodes(mask_dir: Path, ocean_code: int, river_code: int, workers: int) -> pd.DataFrame:
    """One node (lon, lat) per NODE_BIN_DEG cell holding coastline cells: their mean position.

    Bins align with the 1-degree tile edges, so no node is split between tiles.
    """
    paths = sorted(_scan_mask_dir(mask_dir).values())
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_tile_nodes, [(p, ocean_code, river_code) for p in paths], chunksize=8))
    lonlat = np.vstack(parts)
    return pd.DataFrame({"lon": lonlat[:, 0], "lat": lonlat[:, 1]})
