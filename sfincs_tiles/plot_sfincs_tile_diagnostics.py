"""Default per-tile SFINCS validation inspection: three diagnostic figures
for one built (and ideally run) tile, written to
`{base_dir_name}/{tile_id}/figures/`.

Reads `sfincs.msk`/`sfincs.ind` directly as SFINCS's own raw binary format
(see `_read_sfincs_mask`) instead of going through `hydromt_sfincs.
SfincsModel`, so it needs only rasterio/scipy/matplotlib and runs under
`gfm_python_preprocessing` like every other sfincs_tiles/ plotting script.

Figures:
  1. {tile_id}_mask.png - SFINCS's own active/inactive/waterlevel-boundary
     mask (sfincs.msk), boundary cells drawn as an enlarged, high-contrast
     overlay so a 1-cell-wide boundary line stays visible at figure scale.
  2. {tile_id}_boundary_forcing.png - (a) map of the matched boundary
     stations (matched_boundary_points.gpkg) and the boundary cell line(s)
     (mask==2) together, (b) the forcing timeseries for every station
     (corrected_hydrographs.csv).
  3. {tile_id}_biggest_disagreement.png - zoomed to the single largest
     contiguous eikonal-only-or-SFINCS-only patch (four-way classification
     reused from plot_worst_tiles_panel.py's own build_classification()),
     labelled with which model over-predicts there and its area.

Usage:
    python plot_sfincs_tile_diagnostics.py --tile-id 1693 --base-dir-name validation_sfincs_v5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import transform as warp_transform
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402
from plot_worst_tiles_panel import (  # noqa: E402
    CATEGORY_COLORS, CATEGORY_LABELS, DRY, EIKONAL_ONLY, MATCHED, SFINCS_ONLY,
    build_classification,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import WATER_COLOR, WATER_LABEL, draw_caption_box, draw_panel_letter  # noqa: E402

INACTIVE, ACTIVE, BOUNDARY = 0, 1, 2
MASK_COLORS = ["#f5f5f2", "#a8c8e8", "#d62728"]  # inactive, active, boundary
MASK_LABELS = ["inactive", "active", "waterlevel boundary"]


def _parse_sfincs_inp(inp_path: Path) -> dict:
    """`key = value` lines in sfincs.inp - only the handful of grid-geometry
    keys this script needs (mmax, nmax, dx, dy, x0, y0, epsg; rotation
    defaults to 0 - none of this pipeline's own tiles set it)."""
    out: dict = {}
    for line in inp_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def _read_sfincs_mask(sfincs_dir: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Reconstructs the 2D active/boundary mask directly from SFINCS's own
    `sfincs.ind`/`sfincs.msk` binary files, without hydromt_sfincs.

    Format: sfincs.ind is uint32 little-endian, first value N = active-cell
    count, remaining N values are 1-based flat (column-major, ncol x nrow)
    indices of active cells. sfincs.msk is N raw uint8 values (0/1/2/3),
    same order as sfincs.ind's indices. Reshape to (ncol, nrow) then
    transpose to get a normal (nrow, ncol) row-major array.
    """
    inp = _parse_sfincs_inp(sfincs_dir / "sfincs.inp")
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
    # SFINCS's (m, n) indexing has n=0 at y0 (south edge); flip so row 0 is
    # north, matching the north-up Affine transform below.
    mask2d = np.flipud(mask2d)

    if rotation != 0.0:
        raise NotImplementedError(f"sfincs.inp has a nonzero rotation ({rotation}) - "
                                   "this reader only handles axis-aligned grids")
    transform = rasterio.Affine(dx, 0.0, x0, 0.0, -dy, y0 + dy * nmax)
    crs = f"EPSG:{epsg}" if epsg else None
    return mask2d, transform, crs


def _grid_lonlat(transform: rasterio.Affine, crs: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Corner lon/lat for imshow's `extent=` (tiles are axis-aligned UTM, so
    a bounding-box reprojection is exact enough for a diagnostic plot)."""
    nrow, ncol = shape
    x0, y0 = transform.c, transform.f
    x1, y1 = transform.c + transform.a * ncol, transform.f + transform.e * nrow
    lons, lats = warp_transform(crs, "EPSG:4326", [x0, x1], [y0, y1])
    return [min(lons), max(lons)], [min(lats), max(lats)]


def plot_mask(mask2d: np.ndarray, transform: rasterio.Affine, crs: str, out_path: Path,
               stations_path: Path | None = None) -> None:
    lon_range, lat_range = _grid_lonlat(transform, crs, mask2d.shape)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]

    # dilate boundary cells by 1px for display - a 1-cell-wide line is easy to miss at figure scale.
    boundary = mask2d == BOUNDARY
    boundary_display = ndimage.binary_dilation(boundary, iterations=1)
    display = np.where(boundary_display, BOUNDARY, mask2d)

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.imshow(display, extent=extent, origin="upper", cmap=ListedColormap(MASK_COLORS),
               vmin=0, vmax=2, aspect="auto", interpolation="nearest")

    n_stations = 0
    if stations_path is not None and stations_path.exists():
        stations = gpd.read_file(stations_path).to_crs("EPSG:4326")
        n_stations = len(stations)
        ax.scatter(stations.geometry.x, stations.geometry.y, s=90, facecolors="none", edgecolors="black",
                   linewidths=1.6, marker="o", zorder=5, label=f"COAST-RP station(s) (n={n_stations})")

    handles = [Patch(facecolor=c, edgecolor="grey", linewidth=0.5, label=l)
               for c, l in zip(MASK_COLORS, MASK_LABELS)]
    if n_stations:
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                                    markeredgecolor="black", markeredgewidth=1.6, markersize=9,
                                    label=f"COAST-RP station(s) (n={n_stations})"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    n_active, n_bnd = int((mask2d == ACTIVE).sum()), int(boundary.sum())
    draw_caption_box(ax, [
        f"{n_active:,} active cell(s) (excl. boundary)",
        f"{n_bnd:,} waterlevel-boundary cell(s)",
        "(boundary shown 1px dilated for visibility)",
    ])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_boundary_forcing(mask2d: np.ndarray, transform: rasterio.Affine, crs: str,
                           sfincs_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) map: active domain (context) + boundary cell line + boundary station points
    ax = axes[0]
    lon_range, lat_range = _grid_lonlat(transform, crs, mask2d.shape)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    active_display = np.where(mask2d == ACTIVE, 1, 0)
    ax.imshow(active_display, extent=extent, origin="upper", cmap=ListedColormap(["white", "#d8d8d4"]),
               vmin=0, vmax=1, aspect="auto", interpolation="nearest", alpha=0.8)

    b_rows, b_cols = np.nonzero(mask2d == BOUNDARY)
    bxs = transform.c + transform.a * (b_cols + 0.5)
    bys = transform.f + transform.e * (b_rows + 0.5)
    blons, blats = warp_transform(crs, "EPSG:4326", bxs, bys)
    ax.scatter(blons, blats, s=8, color="#d62728", marker="s", label=f"boundary cells (n={len(blons)})", zorder=4)

    points_path = sfincs_dir / "matched_boundary_points.gpkg"
    if points_path.exists():
        points = gpd.read_file(points_path).to_crs("EPSG:4326")
        ax.scatter(points.geometry.x, points.geometry.y, s=80, facecolors="none", edgecolors="#1f4e8c",
                   linewidths=1.5, marker="o", label=f"forcing stations (n={len(points)})", zorder=5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    draw_panel_letter(ax, "a")
    draw_caption_box(ax, "Boundary stations + boundary cell line(s)", loc="lower left")
    ax.legend(fontsize=8, loc="best")

    # (b) forcing timeseries
    ax = axes[1]
    hydro_path = sfincs_dir / "corrected_hydrographs.csv"
    hydro = pd.read_csv(hydro_path)
    station_cols = [c for c in hydro.columns if c != "elapsed_hr"]
    for col in station_cols:
        ax.plot(hydro["elapsed_hr"], hydro[col], color="#1f4e8c", alpha=0.35, linewidth=0.9)
    ax.plot(hydro["elapsed_hr"], hydro[station_cols].median(axis=1), color="#d62728", linewidth=2,
             label=f"median across {len(station_cols)} station(s)")
    ax.set_xlabel("elapsed time (h)")
    ax.set_ylabel("water level (m)")
    draw_panel_letter(ax, "b")
    draw_caption_box(ax, "Forcing hydrographs (MDT-corrected)", loc="lower right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_biggest_disagreement(tile_dir: Path, out_path: Path) -> None:
    cls = build_classification(tile_dir)
    if cls is None:
        print(f"SKIP {out_path}: no usable SFINCS/eikonal subgrid data for this tile")
        return
    cls_filled = cls.filled(DRY)

    only_mask = np.isin(cls_filled, [EIKONAL_ONLY, SFINCS_ONLY])
    if not only_mask.any():
        print(f"SKIP {out_path}: no eikonal-only or SFINCS-only cells at all (perfect agreement)")
        return

    labels, n_components = ndimage.label(only_mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(only_mask, labels, index=np.arange(1, n_components + 1))
    biggest_label = int(np.argmax(sizes)) + 1
    biggest_size = int(sizes[biggest_label - 1])
    biggest_component = labels == biggest_label

    rows, cols = np.nonzero(biggest_component)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    pad_r = max(int((r1 - r0) * 0.3), 10)
    pad_c = max(int((c1 - c0) * 0.3), 10)
    r0, r1 = max(r0 - pad_r, 0), min(r1 + pad_r + 1, cls.shape[0])
    c0, c1 = max(c0 - pad_c, 0), min(c1 + pad_c + 1, cls.shape[1])
    crop = cls[r0:r1, c0:c1]

    n_eik = int((cls_filled[biggest_component] == EIKONAL_ONLY).sum())
    n_sfincs = int((cls_filled[biggest_component] == SFINCS_ONLY).sum())
    dominant = "eikonal" if n_eik >= n_sfincs else "SFINCS"

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor(WATER_COLOR)  # non-land (ocean/river/lake) cells within the crop -
    # same shared colour every agreement map in the repo uses, not a plain white background.
    ax.imshow(crop, cmap=ListedColormap(CATEGORY_COLORS), vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Patch(facecolor=c, edgecolor="grey", linewidth=0.5, label=l)
               for c, l in zip(CATEGORY_COLORS, CATEGORY_LABELS)]
    handles.append(Patch(facecolor=WATER_COLOR, edgecolor="grey", linewidth=0.5, label=WATER_LABEL))
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    draw_caption_box(ax, [
        f"Biggest single disagreement patch: {biggest_size} cells, {dominant}-dominant",
        f"(eikonal-only={n_eik}, SFINCS-only={n_sfincs})",
        f"of {n_components} disagreement patch(es) total",
    ])
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--base-dir-name", default="validation_sfincs_v5")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    root = read_root(Path(args.config))
    tile_dir = root / args.base_dir_name / args.tile_id
    sfincs_dir = tile_dir / "sfincs_model"
    fig_dir = tile_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mask2d, transform, crs = _read_sfincs_mask(sfincs_dir)
    plot_mask(mask2d, transform, crs, fig_dir / f"{args.tile_id}_mask.png",
              stations_path=sfincs_dir / "matched_boundary_points.gpkg")
    plot_boundary_forcing(mask2d, transform, crs, sfincs_dir, fig_dir / f"{args.tile_id}_boundary_forcing.png")
    plot_biggest_disagreement(tile_dir, fig_dir / f"{args.tile_id}_biggest_disagreement.png")


if __name__ == "__main__":
    main()
