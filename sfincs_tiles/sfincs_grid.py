"""SFINCS grid files read directly (no hydromt_sfincs), so the plotting and
analysis scripts run under `gfm` (`gfm_python_preprocessing` on the Windows dev machine):

  read_sfincs_mask(sfincs_dir)                        -> (mask2d, transform, crs) of the 120 m grid
  active_on_subgrid(sfincs_dir, sg_shape, sg_transform) -> SFINCS-active cells on the 30 m subgrid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def parse_sfincs_inp(inp_path: Path) -> dict:
    """`key = value` lines in sfincs.inp - only the handful of grid-geometry
    keys needed here (mmax, nmax, dx, dy, x0, y0, epsg, rotation)."""
    out: dict = {}
    for line in inp_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def read_sfincs_mask(sfincs_dir: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Reconstructs the 2D active/boundary mask directly from SFINCS's own
    `sfincs.ind`/`sfincs.msk` binary files, without hydromt_sfincs.

    Format: sfincs.ind is uint32 little-endian, first value N = active-cell
    count, remaining N values are 1-based flat (column-major, ncol x nrow)
    indices of active cells. sfincs.msk is N raw uint8 values (0/1/2/3),
    same order as sfincs.ind's indices. Reshape to (ncol, nrow) then
    transpose to get a normal (nrow, ncol) row-major array.
    """
    inp = parse_sfincs_inp(sfincs_dir / "sfincs.inp")
    mmax, nmax = int(inp["mmax"]), int(inp["nmax"])  # mmax=ncol, nmax=nrow
    dx, dy = float(inp["dx"]), float(inp["dy"])
    x0, y0 = float(inp["x0"]), float(inp["y0"])
    rotation = float(inp.get("rotation", 0.0))
    epsg = inp.get("epsg")

    ind_raw = np.fromfile(sfincs_dir / "sfincs.ind", dtype="<u4")
    n = int(ind_raw[0])
    ind = ind_raw[1:1 + n].astype(np.int64) - 1
    msk = np.fromfile(sfincs_dir / "sfincs.msk", dtype="u1")
    assert msk.size == n, f"sfincs.msk has {msk.size} values, expected {n} from sfincs.ind"

    grid = np.zeros(mmax * nmax, dtype=np.uint8)
    grid[ind] = msk
    mask2d = grid.reshape(mmax, nmax).T  # (ncol, nrow) -> (nrow, ncol)
    crs = f"EPSG:{epsg}" if epsg else None

    if rotation != 0.0:
        # Rotated grid (counter-clockwise by `rotation` degrees about (x0, y0)):
        # keep SFINCS's own row order (row 0 = n=0, the y0 edge) and use the
        # matching rotated affine - the same transform hydromt writes to
        # gis/mask.tif. Plots draw cell corners from it (see _grid_corners_lonlat).
        theta = np.deg2rad(rotation)
        transform = rasterio.Affine(dx * np.cos(theta), -dy * np.sin(theta), x0,
                                    dx * np.sin(theta), dy * np.cos(theta), y0)
        return mask2d, transform, crs

    # SFINCS's (m, n) indexing has n=0 at y0 (south edge); flip so row 0 is
    # north, matching the north-up Affine transform below.
    mask2d = np.flipud(mask2d)
    transform = rasterio.Affine(dx, 0.0, x0, 0.0, -dy, y0 + dy * nmax)
    return mask2d, transform, crs


def active_on_subgrid(sfincs_dir: Path, sg_shape: tuple, sg_transform) -> np.ndarray:
    """sfincs.msk > 0 expanded onto the subgrid (hmax_subgrid.tif / dep_subgrid.tif
    grid): the subgrid shares the coarse grid's origin and rotation at an
    integer fraction of its cell size, so each coarse cell is an f x f block."""
    msk, tr, _ = read_sfincs_mask(sfincs_dir)
    f = int(round(tr.a / sg_transform.a)) if tr.b == 0 else int(round(np.hypot(tr.a, tr.d) / np.hypot(sg_transform.a, sg_transform.d)))
    if not (np.isclose(tr.c, sg_transform.c) and np.isclose(tr.f, sg_transform.f)
            and (msk.shape[0] * f, msk.shape[1] * f) == tuple(sg_shape)):
        raise ValueError(f"sfincs grid {msk.shape} @ {tr} does not nest into subgrid {sg_shape} @ {sg_transform}")
    return np.kron(msk > 0, np.ones((f, f), dtype=bool))
