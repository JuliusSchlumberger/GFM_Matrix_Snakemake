"""The tile's native land/ocean mask on a SFINCS (sub)grid, and the eikonal coastline on it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
LAND_CODE, OCEAN_CODE = 0, 1
NODATA_CODE = 4  # subgrid cells the native lon/lat mask.tif does not cover


def native_mask_on_grid(native_mask_path: Path, transform, crs, shape: tuple[int, int]) -> np.ndarray:
    """Reproject the native lon/lat mask.tif (nearest) onto a UTM grid, int8 codes.

    The UTM rectangle overshoots the lon/lat mask at the corners (meridian convergence); those
    cells and any non-mask code get NODATA_CODE.
    """
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        mask_f = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mask_f,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
    valid_code = np.isin(mask_f, [0, 1, 2, 3])
    return np.where(valid_code, mask_f, np.float32(NODATA_CODE)).astype(np.int8)


def subgrid_coastline_mask(mask: np.ndarray, ocean_code: int, river_code: int | None) -> np.ndarray:
    """Mark coastline cells on the rotated subgrid, as flood_model.coastline_mask does.

    Ocean within 1 px of land (or river) in an ocean body reaching the domain edge:
    the array edge or the NODATA_CODE corners (the tile edge on this grid).
    """
    ocean = mask == ocean_code
    components, _ = ndimage.label(ocean, structure=_STRUCTURE_8)
    edge = np.zeros_like(ocean)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= ndimage.binary_dilation(mask == NODATA_CODE, structure=_STRUCTURE_8)
    edge_labels = np.unique(components[edge & ocean])
    edge_connected_ocean = np.isin(components, edge_labels[edge_labels != 0])
    landlike = mask == LAND_CODE
    if river_code is not None:
        landlike |= mask == river_code
    return ndimage.binary_dilation(landlike, structure=_STRUCTURE_8) & edge_connected_ocean
