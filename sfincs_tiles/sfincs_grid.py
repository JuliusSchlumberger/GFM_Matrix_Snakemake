"""Read SFINCS grid files without hydromt_sfincs, so gfm-env scripts can use them."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


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


_AREA_CRS = "EPSG:6933"  # equal-area, so tile sizes compare fairly at any latitude
_SEGMENT_DEG = 0.01  # densify lon/lat edges before projecting, so they stay true on the model grid


def load_footprints(tile_selection_gpkg: Path) -> gpd.GeoDataFrame:
    """Return the selected tiles' polygons (EPSG:4326) by tile id, with their size rank (0 = largest).

    Ties in area go to the lower tile id. Uses ALL selected tiles, so a tile without a usable
    result still claims its overlaps instead of ceding them to a smaller neighbour.
    """
    gdf = gpd.read_file(tile_selection_gpkg)
    sel = gdf[gdf["category"] == "selected"].to_crs("EPSG:4326")
    sel = sel.assign(tile_id=sel["tile_id"].astype(int), area=sel.to_crs(_AREA_CRS).area)
    sel = sel.sort_values(["area", "tile_id"], ascending=[False, True])
    sel["rank"] = np.arange(len(sel))
    sel["tile_id"] = sel["tile_id"].astype(str)
    return sel.set_index("tile_id")[["rank", "geometry"]]


def core_mask(shape: tuple, transform, crs, this_tile_id: str, footprints: gpd.GeoDataFrame) -> np.ndarray:
    """Mark the cells of `shape` inside tile `this_tile_id` and inside no larger selected tile.

    A place covered by several selected tiles belongs to the largest of them, so pooled sums
    count it once and no covered place is dropped. Cell centres decide (rasterio geometry_mask).
    """
    own = footprints.loc[this_tile_id]
    others = footprints[(footprints["rank"] < own["rank"]) & footprints.intersects(own.geometry)]

    def _inside(geoms: gpd.GeoSeries) -> np.ndarray:
        projected = geoms.segmentize(_SEGMENT_DEG).to_crs(crs)
        return geometry_mask(list(projected), out_shape=shape, transform=transform, invert=True)

    core = _inside(gpd.GeoSeries([own.geometry], crs=footprints.crs))
    if len(others):
        core &= ~_inside(others.geometry)
    return core
