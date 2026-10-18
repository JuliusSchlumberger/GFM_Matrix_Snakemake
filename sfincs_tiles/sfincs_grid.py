"""Read SFINCS grid files without hydromt_sfincs, so gfm-env scripts can use them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def parse_sfincs_inp(inp_path: Path) -> dict:
    """Parse the ``key = value`` lines of a sfincs.inp file into raw strings."""
    out: dict = {}
    for line in inp_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def read_sfincs_mask(sfincs_dir: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Return the (nrow, ncol) SFINCS mask (0/1/2/3), its affine and "EPSG:<code>" (or None).

    ``sfincs.ind`` is uint32 little-endian: active-cell count N, then N 1-based
    flat (column-major, ncol x nrow) indices; ``sfincs.msk`` holds N uint8 values
    in the same order. Rotated grids keep SFINCS's row 0 = n=0 (south) edge with
    a rotated affine; unrotated grids are flipped to north-up.
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
        # Counter-clockwise by `rotation` degrees about (x0, y0); the same transform
        # hydromt writes to gis/mask.tif.
        theta = np.deg2rad(rotation)
        transform = rasterio.Affine(dx * np.cos(theta), -dy * np.sin(theta), x0,
                                    dx * np.sin(theta), dy * np.cos(theta), y0)
        return mask2d, transform, crs

    mask2d = np.flipud(mask2d)
    transform = rasterio.Affine(dx, 0.0, x0, 0.0, -dy, y0 + dy * nmax)
    return mask2d, transform, crs


def active_on_subgrid(sfincs_dir: Path, sg_shape: tuple, sg_transform) -> np.ndarray:
    """Expand the SFINCS active-cell mask (sfincs.msk > 0) onto a finer subgrid.

    Assumes the subgrid shares the coarse grid's origin and rotation at an integer
    fraction of its cell size; raises ValueError if it does not nest exactly.
    """
    msk, tr, _ = read_sfincs_mask(sfincs_dir)
    f = int(round(tr.a / sg_transform.a)) if tr.b == 0 else int(round(np.hypot(tr.a, tr.d) / np.hypot(sg_transform.a, sg_transform.d)))
    if not (np.isclose(tr.c, sg_transform.c) and np.isclose(tr.f, sg_transform.f)
            and (msk.shape[0] * f, msk.shape[1] * f) == tuple(sg_shape)):
        raise ValueError(f"sfincs grid {msk.shape} @ {tr} does not nest into subgrid {sg_shape} @ {sg_transform}")
    return np.kron(msk > 0, np.ones((f, f), dtype=bool))
