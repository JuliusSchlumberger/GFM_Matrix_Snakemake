"""Cheap, provably-safe pre-crop of DEM/mask/friction to the region that
could possibly flood, before handing inputs to Aqueduct.

Safety argument: Aqueduct's flood model (core/src/core.jl, flood_depth)
computes, for every cell x, waterlevel(x) = max over boundary seeds c of
[seed(c) - cost(x, c)], where cost is a friction-weighted path integral that
is never negative (friction is coalesced to a minimum of 0.001). Therefore
waterlevel(x) can never exceed the single highest boundary water level used
anywhere for that tile, regardless of friction, path, or which
(return_period, waterlevel_name) scenario is running. Any cell whose
*effective* elevation (see `effective_dem` below) is >= that maximum can
therefore never flood, in ANY scenario that shares the same tile - so
cropping to the bounding box of cells below that threshold is exact, not an
approximation, and only needs to be computed once per tile.

`effective_dem` mirrors core.jl's own preprocessing
(`dem[.!landmask] .= 0.0`) rather than reading dem.tif's raw values
directly: `rasters.extract_dem` only fills cells that are nodata in the DEM
SOURCE, so a river/lake/ocean cell that happens to have a real (non-nodata)
DeltaDTM elevation keeps that real value in dem.tif - core.jl still zeroes
it unconditionally at run time because it is not land. Thresholding the raw
file value instead of this effective value could wrongly exclude such a
cell from the crop.

Lake and river cells (mask codes 2 and 3) are deliberately always
"candidate" here, since their effective elevation is always 0.0: a river
corridor threading far inland stays part of the candidate region for its
whole length. That is correct, not a bug - this threshold only excludes
area that provably cannot flood; it makes no attempt to guess how far
friction attenuation will actually let a flood signal travel up that river.
That refinement happens for real in Aqueduct's own (unmodified) solve on
whatever region survives this crop.
"""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from config_utils import retry_transient_io

LAND_CODE = 0


def effective_dem(dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Elevation Aqueduct will actually use at run time.

    Mirrors core.jl's `dem[.!landmask] .= 0.0`: every non-land cell (ocean,
    lake, river) is treated as elevation 0 regardless of what value happens
    to be stored in dem.tif for it.
    """
    return np.where(mask == LAND_CODE, dem, dem.dtype.type(0.0))


def flood_candidate_mask(
    dem: np.ndarray,
    mask: np.ndarray,
    max_waterlevel: float,
    ocean_code: int,
) -> np.ndarray:
    """Boolean mask of cells that could possibly flood for any scenario sharing `max_waterlevel`.

    Ocean cells are excluded here (core.jl's `flood[mask .== 1] .= false`
    means they can never be flagged as flooded) even though their effective
    elevation is also 0.0. They are pulled back in as a thin buffer by
    `compute_flood_candidate_window`'s margin, purely so the coastlinemask
    dilation has real ocean cells adjacent to compare against - not because
    they themselves need to be "candidate".
    """
    return (mask != ocean_code) & (effective_dem(dem, mask) < max_waterlevel)


def compute_flood_candidate_window(
    dem: np.ndarray,
    mask: np.ndarray,
    max_waterlevel: float | None,
    margin_px: int,
    ocean_code: int,
) -> Window | None:
    """Return the bounding Window of the flood-candidate region (+ margin), or None if empty.

    `max_waterlevel=None` (no boundary stations anywhere for this tile, in
    any scenario) is treated the same as "no candidate cells": nothing can
    ever flood.

    Args:
        dem: Full (uncropped) DEM array for the tile.
        mask: Full (uncropped) DEM-validity mask array, same grid as `dem`.
        max_waterlevel: Highest boundary water level across every
            (return_period, waterlevel_name) this tile will be run for, or
            None if it has no boundary stations at all.
        margin_px: Pixels added on every side of the candidate bounding box.
            Must exceed the dilation radius used for `coastlinemask` in
            core.jl (1 pixel for a 3x3 structuring element); a generous
            default is cheap.
        ocean_code: Mask value for ocean cells.

    Returns:
        A `rasterio.windows.Window` in the full array's pixel coordinates,
        or None if no cell in the tile can ever flood.
    """
    if max_waterlevel is None:
        return None
    candidate = flood_candidate_mask(dem, mask, max_waterlevel, ocean_code)
    if not candidate.any():
        return None
    rows = np.where(candidate.any(axis=1))[0]
    cols = np.where(candidate.any(axis=0))[0]
    row0 = max(int(rows[0]) - margin_px, 0)
    row1 = min(int(rows[-1]) + margin_px + 1, dem.shape[0])
    col0 = max(int(cols[0]) - margin_px, 0)
    col1 = min(int(cols[-1]) + margin_px + 1, dem.shape[1])
    return Window(col0, row0, col1 - col0, row1 - row0)


def crop_raster_to_window(
    src_path: str | Path,
    window: Window,
    dst_path: str | Path,
    raster_config: dict[str, Any],
) -> None:
    """Write the portion of `src_path` covered by `window` to `dst_path`.

    A pure spatial crop: pixel values are copied unchanged (never edited or
    reclassified), dtype/nodata are kept from the source. Only the raster's
    extent (transform, width, height) shrinks.

    Args:
        src_path: Path to the source raster (e.g. the tile's full dem.tif).
        window: Pixel window to extract, in `src_path`'s own row/col space.
        dst_path: Destination file path.
        raster_config: The workflow's `raster_format` config section
            (`driver`, `compression`, `predictor`), for the output file.
    """
    with retry_transient_io(rasterio.open, src_path) as src:
        data = src.read(1, window=window)
        transform = rasterio.windows.transform(window, src.transform)
        profile = src.profile

    profile.update(
        height=int(window.height),
        width=int(window.width),
        transform=transform,
        driver=raster_config["driver"],
        compress=raster_config["compression"],
        predictor=raster_config["predictor"],
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with retry_transient_io(rasterio.open, dst_path, "w", **profile) as dst:
        dst.write(data, 1)
