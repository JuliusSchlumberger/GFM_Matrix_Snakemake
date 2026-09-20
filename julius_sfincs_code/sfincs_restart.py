"""
sfincs_restart.py -- Read/patch/write a SFINCS binary restart (.rst) file.

SFINCS restart files are NOT documented or parsed anywhere else in this
codebase (src.sfincs_run explicitly treats "rstfile" as an opaque forwarded
path) -- this module is the first place that actually opens one. The format
was reverse-engineered and empirically validated against this repo's own
basin 2433835 (confirmed byte-exact round-trip, and record[1] cross-checked
against mod.grid.mask via sfincs.ind's own active-cell ordering: zero
mismatches across all 263,907 grid cells, and zs-zb >= 0 everywhere with
~46% of cells exactly dry -- the correct physical signature for a water-level
restart state) before being trusted here.

File layout: classic Fortran unformatted sequential access -- a sequence of
records, each `[int32 length][length bytes of data][int32 length]` (the two
length markers must match). This basin's restart has 4 records:
    0: 1 int32   (unidentified header/flag -- preserved untouched)
    1: N float32 (zs, water surface elevation -- one value per ACTIVE cell,
                  in the same order as sfincs.ind's own active-cell list)
    2: ~2N float32 (unidentified -- likely flux/velocity state -- preserved untouched)
    3: small float32 array (unidentified -- preserved untouched)
Only record 1 (zs) is ever modified here; every other record is copied
through byte-for-byte, so whatever physics they represent is left exactly
as the spin-up run computed it.

sfincs.ind pairs with this: a header int32 (active-cell count) followed by
that many int32 1-based FORTRAN (column-major) flat indices into the
model's own (nmax, mmax) grid -- i.e. `np.unravel_index(ind - 1,
(nmax, mmax), order="F")` gives each active cell's (row, col), in the SAME
order as zs's own entries. This is what lets a zone polygon (rasterized
onto mod.grid.mask, which is confirmed identical to this same (nmax, mmax)
grid) be mapped onto specific zs entries.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_records(path: Path) -> list[bytes]:
    """Parse a Fortran unformatted sequential file into its raw data records."""
    with open(path, "rb") as f:
        data = f.read()

    records: list[bytes] = []
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError(f"{path}: truncated record header at byte {pos}")
        (rec_len,) = struct.unpack_from("<i", data, pos)
        start = pos + 4
        end = start + rec_len
        if end + 4 > len(data):
            raise ValueError(
                f"{path}: record at byte {pos} claims length {rec_len}, exceeds file size"
            )
        (rec_len2,) = struct.unpack_from("<i", data, end)
        if rec_len2 != rec_len:
            raise ValueError(
                f"{path}: record at byte {pos} has mismatched length markers "
                f"({rec_len} != {rec_len2}) -- not a valid Fortran unformatted file, "
                f"or this format assumption is wrong for this file"
            )
        records.append(data[start:end])
        pos = end + 4
    return records


def _write_records(path: Path, records: list[bytes]) -> None:
    out = bytearray()
    for rec in records:
        n = len(rec)
        out += struct.pack("<i", n) + rec + struct.pack("<i", n)
    with open(path, "wb") as f:
        f.write(bytes(out))


def read_ind(ind_path: Path) -> np.ndarray:
    """0-based (row, col) for each active cell, in sfincs.ind's own order.

    Returns:
        (rows, cols): two (n_active,) int arrays.
    """
    ind_raw = np.fromfile(ind_path, dtype="<i4")
    n_active = int(ind_raw[0])
    if len(ind_raw) - 1 != n_active:
        raise ValueError(
            f"{ind_path}: header count {n_active} does not match "
            f"{len(ind_raw) - 1} index entries actually present"
        )
    return ind_raw[1:] - 1  # 1-based Fortran -> 0-based flat index


def patch_restart_zs(
    src_rst_path: Path,
    dst_rst_path: Path,
    ind_path: Path,
    grid_shape: tuple[int, int],
    active_cell_mask: np.ndarray,
    dz: np.ndarray | float,
) -> int:
    """
    Write a copy of src_rst_path with zs shifted by `dz` at active cells
    where `active_cell_mask` is True, everything else byte-identical.

    Args:
        src_rst_path      : source restart file (e.g. the basin's own
                             spin-up restart -- never modified in place).
        dst_rst_path       : output path for the patched copy.
        ind_path           : this basin's own sfincs.ind (defines active-cell
                             order/positions -- must match src_rst_path's own
                             grid, i.e. the SAME basin/skeleton).
        grid_shape          : (nmax, mmax) of the coarse SFINCS grid --
                             must match mod.grid.mask.shape.
        active_cell_mask    : (nmax, mmax) boolean array (e.g. a rasterized
                             zone polygon on mod.grid.mask's own grid) --
                             True where zs should be shifted.
        dz                  : scalar or (nmax, mmax) array, meters added to
                             zs at the selected cells (e.g. a negative
                             `-lowering` to follow a DEM lowering downward).

    Returns:
        Number of active cells actually patched.
    """
    ind0 = read_ind(ind_path)  # 0-based flat index, Fortran order, per active cell
    nmax, mmax = grid_shape
    if active_cell_mask.shape != grid_shape:
        raise ValueError(
            f"active_cell_mask shape {active_cell_mask.shape} != grid_shape {grid_shape}"
        )
    rows, cols = np.unravel_index(ind0, (nmax, mmax), order="F")

    records = _read_records(src_rst_path)
    if len(records) < 2:
        raise ValueError(f"{src_rst_path}: expected >=2 records, found {len(records)}")

    n_active = ind0.size
    zs = np.frombuffer(records[1], dtype="<f4", count=n_active).copy()
    if zs.size * 4 != len(records[1]):
        raise ValueError(
            f"{src_rst_path}: record[1] size ({len(records[1])} bytes) does not match "
            f"{n_active} active cells (sfincs.ind) at float32 -- refusing to patch a "
            f"file that doesn't match this basin's own grid"
        )

    select = active_cell_mask[rows, cols]  # (n_active,) bool, per zs entry
    n_patched = int(select.sum())
    if n_patched > 0:
        dz_per_cell = dz if np.isscalar(dz) else np.asarray(dz)[rows, cols]
        zs = zs.astype(np.float64)
        zs[select] = zs[select] + (dz_per_cell[select] if not np.isscalar(dz) else dz)
        zs = zs.astype(np.float32)

    records[1] = zs.tobytes()
    dst_rst_path = Path(dst_rst_path)
    dst_rst_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(dst_rst_path, records)
    return n_patched
