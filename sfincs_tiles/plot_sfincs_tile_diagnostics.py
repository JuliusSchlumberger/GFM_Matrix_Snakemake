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
    python plot_sfincs_tile_diagnostics.py --tile-id 1974 --base-dir-name sfincs_calibration \
        --friction-scale-factor 9 --max-outer-iterations 5
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
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import transform as warp_transform
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402
from compute_friction_sweep_metrics import FRICTION_SCALE_FACTOR_DEFAULT, MAX_OUTER_ITERATIONS_DEFAULT  # noqa: E402
from plot_worst_tiles_panel import (  # noqa: E402
    CATEGORY_COLORS, CATEGORY_LABELS, DRY, EIKONAL_ONLY, MATCHED, SFINCS_ONLY,
    build_classification,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import WATER_COLOR, WATER_LABEL, draw_caption_box, draw_panel_letter  # noqa: E402
from rasters import decode_waterlevel_cm  # noqa: E402

OCEAN_CODE = 1

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


def _read_sfincs_bnd_bzs(sfincs_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """The boundary points SFINCS is actually forced with (`sfincs.bnd`, x/y
    in the model CRS) and their water-level series (`sfincs.bzs`: time in
    seconds, then one column per bnd point) - which can be far fewer than
    the matched_boundary_points.gpkg candidates."""
    bnd_path, bzs_path = sfincs_dir / "sfincs.bnd", sfincs_dir / "sfincs.bzs"
    if not (bnd_path.exists() and bzs_path.exists()):
        return None, None
    return np.loadtxt(bnd_path, ndmin=2), np.loadtxt(bzs_path, ndmin=2)


def _grid_corners_lonlat(transform: rasterio.Affine, crs: str,
                         shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(nrow+1, ncol+1) lon/lat of every cell corner, for pcolormesh - exact
    for rotated SFINCS grids too, unlike a rectangular imshow extent."""
    nrow, ncol = shape
    cols, rows = np.meshgrid(np.arange(ncol + 1), np.arange(nrow + 1))
    xs, ys = transform * (cols.ravel(), rows.ravel())
    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
    return np.asarray(lons).reshape(nrow + 1, ncol + 1), np.asarray(lats).reshape(nrow + 1, ncol + 1)


def _cell_centres_lonlat(transform: rasterio.Affine, crs: str, rows: np.ndarray,
                         cols: np.ndarray) -> tuple[list, list]:
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return warp_transform(crs, "EPSG:4326", xs, ys)


def plot_mask(mask2d: np.ndarray, transform: rasterio.Affine, crs: str, out_path: Path,
               stations_path: Path | None = None) -> None:
    corner_lon, corner_lat = _grid_corners_lonlat(transform, crs, mask2d.shape)

    # dilate boundary cells by 1px for display - a 1-cell-wide line is easy to miss at figure scale.
    boundary = mask2d == BOUNDARY
    boundary_display = ndimage.binary_dilation(boundary, iterations=1)
    display = np.where(boundary_display, BOUNDARY, mask2d)

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.pcolormesh(corner_lon, corner_lat, display, cmap=ListedColormap(MASK_COLORS),
                  vmin=0, vmax=2, shading="flat", rasterized=True)

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
    corner_lon, corner_lat = _grid_corners_lonlat(transform, crs, mask2d.shape)
    active_display = np.where(mask2d == ACTIVE, 1, 0)
    ax.pcolormesh(corner_lon, corner_lat, active_display, cmap=ListedColormap(["white", "#d8d8d4"]),
                  vmin=0, vmax=1, shading="flat", alpha=0.8, rasterized=True)

    b_rows, b_cols = np.nonzero(mask2d == BOUNDARY)
    blons, blats = _cell_centres_lonlat(transform, crs, b_rows, b_cols)
    ax.scatter(blons, blats, s=8, color="#d62728", marker="s", label=f"boundary cells (n={len(blons)})", zorder=4)

    points_path = sfincs_dir / "matched_boundary_points.gpkg"
    if points_path.exists():
        points = gpd.read_file(points_path).to_crs("EPSG:4326")
        ax.scatter(points.geometry.x, points.geometry.y, s=80, facecolors="none", edgecolors="#1f4e8c",
                   linewidths=1.5, marker="o", label=f"matched COAST-RP stations (n={len(points)})", zorder=5)
    bnd, bzs = _read_sfincs_bnd_bzs(sfincs_dir)
    if bnd is not None:
        bnd_lons, bnd_lats = warp_transform(crs, "EPSG:4326", bnd[:, 0], bnd[:, 1])
        ax.scatter(bnd_lons, bnd_lats, s=220, color="#f2b705", edgecolors="black", linewidths=1.0, marker="*",
                   label=f"points actually forcing SFINCS (sfincs.bnd, n={len(bnd)})", zorder=6)
        for k, (lo, la) in enumerate(zip(bnd_lons, bnd_lats)):
            ax.annotate(f"bnd {k + 1}: peak {bzs[:, k + 1].max():.2f} m", (lo, la), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, weight="bold", zorder=7)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    draw_panel_letter(ax, "a")
    draw_caption_box(ax, "Boundary stations + boundary cell line(s)", loc="lower left")
    ax.legend(fontsize=8, loc="best")

    # (b) forcing timeseries: every matched station's hydrograph (context) and
    # the bzs series SFINCS actually receives at its bnd points.
    ax = axes[1]
    hydro_path = sfincs_dir / "corrected_hydrographs.csv"
    hydro = pd.read_csv(hydro_path)
    station_cols = [c for c in hydro.columns if c != "elapsed_hr"]
    for col in station_cols:
        ax.plot(hydro["elapsed_hr"], hydro[col], color="#1f4e8c", alpha=0.35, linewidth=0.9)
    ax.plot(hydro["elapsed_hr"], hydro[station_cols].median(axis=1), color="#1f4e8c", linewidth=2,
             label=f"median across {len(station_cols)} matched station(s)")
    if bzs is not None:
        for k in range(bzs.shape[1] - 1):
            ax.plot(bzs[:, 0] / 3600.0, bzs[:, k + 1], color="#f2b705", linewidth=2.2,
                    label="sfincs.bzs (applied forcing)" if k == 0 else None)
        ax.axvline(bzs[-1, 0] / 3600.0, color="grey", linestyle=":", linewidth=1.2, label="end of bzs / SFINCS run")
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


def plot_biggest_disagreement(tile_dir: Path, out_path: Path, friction_scale_factor: float,
                              max_outer_iterations: int) -> None:
    cls = build_classification(tile_dir, friction_scale_factor, max_outer_iterations)
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
    # Draw every cell at its real corner coordinates (subgrid CRS) rather than
    # imshow's "row 0 = north" assumption: SFINCS subgrid rasters are stored
    # south-up (positive y step) and, on rotated grids, rotated - imshow would
    # show them mirrored north-south. Ticks are labelled in lon/lat along the
    # crop's centre lines (UTM grid convergence over a crop this size is far
    # below the label precision).
    with rasterio.open(tile_dir / "sfincs_model" / "hmax_subgrid.tif") as src:
        sg_transform, sg_crs = src.transform, src.crs
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    xs, ys = sg_transform * (cc, rr)
    ax.pcolormesh(xs, ys, crop, cmap=ListedColormap(CATEGORY_COLORS), vmin=0, vmax=3, shading="flat",
                  rasterized=True)
    ax.set_aspect("equal")
    x_lo, x_hi, y_lo, y_hi = xs.min(), xs.max(), ys.min(), ys.max()
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    xc, yc = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    xt, yt = np.linspace(x_lo, x_hi, 6)[1:-1], np.linspace(y_lo, y_hi, 6)[1:-1]
    xt_lon, _ = warp_transform(sg_crs, "EPSG:4326", xt, np.full_like(xt, yc))
    _, yt_lat = warp_transform(sg_crs, "EPSG:4326", np.full_like(yt, xc), yt)
    ax.set_xticks(xt, [f"{v:.3f}°E" for v in xt_lon], fontsize=8)
    ax.set_yticks(yt, [f"{v:.3f}°N" for v in yt_lat], fontsize=8)
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


def _order_along_coast(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour chain through the stations, starting from the
    one farthest from their centroid (an end of the coastline) - an along-coast
    order without needing the coastline geometry itself. Returns (order,
    cumulative distance in km)."""
    kx = 111.32 * np.cos(np.radians(np.mean(lat)))
    xy = np.column_stack([lon * kx, lat * 110.57])
    start = int(np.argmax(np.hypot(*(xy - xy.mean(axis=0)).T)))
    order, left = [start], set(range(len(xy))) - {start}
    while left:
        nxt = min(left, key=lambda k: np.hypot(*(xy[k] - xy[order[-1]])))
        order.append(nxt)
        left.remove(nxt)
    steps = np.hypot(*np.diff(xy[order], axis=0).T)
    return np.array(order), np.concatenate([[0.0], np.cumsum(steps)])


def plot_coastrp_forcing(tile_dir: Path, sfincs_dir: Path, mask2d: np.ndarray, transform: rasterio.Affine,
                         crs: str, out_path: Path, knn: int, river_code: int) -> None:
    """(a) COAST-RP RP100 levels across the tile: every station, the eikonal's
    own IDW seed level along the coastline (production `_idw_seed_values`,
    `knn` nearest stations) and SFINCS's actual forcing points/boundary line.
    (b) Along the coast, per station inside the tile: the station's own level,
    the eikonal seed level and SFINCS's peak water level (zsmax) nearby - i.e.
    what each model is actually driven by at that stretch of coast."""
    from flood_model import _idw_seed_values, coastline_mask  # src/, heavy imports only needed here

    with rasterio.open(tile_dir / "inputs" / "mask.tif") as src:
        mask_native = src.read(1).astype(np.int8)
        native_transform, native_bounds = src.transform, src.bounds
    stations = gpd.read_file(tile_dir / "inputs" / "boundaries_RP100_SLR_0.gpkg").to_crs("EPSG:4326")
    level_m = decode_waterlevel_cm(stations["SLR_0"].to_numpy())
    st_lon, st_lat = stations.geometry.x.to_numpy(), stations.geometry.y.to_numpy()

    coast = coastline_mask(mask_native, ocean_code=OCEAN_CODE, river_code=river_code)
    c_rows, c_cols = np.nonzero(coast)
    seed = _idw_seed_values(c_rows, c_cols, native_transform, np.column_stack([st_lon, st_lat]), level_m,
                            min(knn, len(level_m)), mask_native, OCEAN_CODE)
    c_lon, c_lat = rasterio.transform.xy(native_transform, c_rows, c_cols)
    c_lon, c_lat = np.asarray(c_lon), np.asarray(c_lat)

    zsmax_lon = zsmax_lat = zsmax = None
    map_path = sfincs_dir / "sfincs_map.nc"
    if map_path.exists():
        import xarray as xr
        with xr.open_dataset(map_path) as ds:
            z = ds["zsmax"].values.reshape(ds["zsmax"].shape[-2:])
            cx, cy = ds["corner_x"].values, ds["corner_y"].values
        cx = 0.25 * (cx[:-1, :-1] + cx[1:, :-1] + cx[:-1, 1:] + cx[1:, 1:])
        cy = 0.25 * (cy[:-1, :-1] + cy[1:, :-1] + cy[:-1, 1:] + cy[1:, 1:])
        ok = np.isfinite(z)
        zsmax_lon, zsmax_lat = (np.asarray(v) for v in warp_transform(crs, "EPSG:4326", cx[ok], cy[ok]))
        zsmax = z[ok]

    vmax = float(np.ceil(level_m.max() * 2) / 2)
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) map
    ax = axes[0]
    step = max(1, mask_native.shape[0] // 1500)
    land_sea = np.where(mask_native[::step, ::step] == 0, 1, 0)
    ax.imshow(land_sea, extent=[native_bounds.left, native_bounds.right, native_bounds.bottom, native_bounds.top],
              cmap=ListedColormap([WATER_COLOR, "#e4e4e0"]), vmin=0, vmax=1, interpolation="nearest")
    sc = ax.scatter(c_lon, c_lat, c=seed, s=2, cmap=cmap, vmin=0, vmax=vmax, zorder=3, rasterized=True)
    ax.scatter(st_lon, st_lat, c=level_m, s=70, cmap=cmap, vmin=0, vmax=vmax, edgecolors="black",
               linewidths=1.0, zorder=5)
    b_rows, b_cols = np.nonzero(mask2d == BOUNDARY)
    blons, blats = _cell_centres_lonlat(transform, crs, b_rows, b_cols)
    ax.scatter(blons, blats, s=4, color="#1f4e8c", marker="s", zorder=4)
    bnd, bzs = _read_sfincs_bnd_bzs(sfincs_dir)
    if bnd is not None:
        bnd_lons, bnd_lats = warp_transform(crs, "EPSG:4326", bnd[:, 0], bnd[:, 1])
        ax.scatter(bnd_lons, bnd_lats, s=260, facecolors="none", edgecolors="#1f4e8c", linewidths=2.0,
                   marker="*", zorder=6)
    ax.set_xlim(native_bounds.left, native_bounds.right)
    ax.set_ylim(native_bounds.bottom, native_bounds.top)
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (native_bounds.bottom + native_bounds.top))))
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="COAST-RP RP100 water level (m)")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#f4a261", markeredgecolor="black",
                   markersize=8, label="COAST-RP station (colour = level)"),
        plt.Line2D([0], [0], color=cmap(0.5), linewidth=3, label=f"eikonal coastline seed (IDW, k={knn})"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f4e8c", markersize=5,
                   label="SFINCS waterlevel-boundary cells"),
        plt.Line2D([0], [0], marker="*", color="none", markeredgecolor="#1f4e8c", markersize=14,
                   label="SFINCS forcing points (sfincs.bnd)"),
    ], loc="lower right", fontsize=8)
    draw_panel_letter(ax, "a")

    # (b) along the coast, stations inside the tile only
    ax = axes[1]
    inside = ((st_lon >= native_bounds.left) & (st_lon <= native_bounds.right)
              & (st_lat >= native_bounds.bottom) & (st_lat <= native_bounds.top))
    idx = np.nonzero(inside)[0]
    order, dist_km = _order_along_coast(st_lon[idx], st_lat[idx])
    idx = idx[order]
    kx = 111.32 * np.cos(np.radians(np.mean(st_lat[idx])))

    def near_median(lon: np.ndarray, lat: np.ndarray, values: np.ndarray, radius_km: float = 3.0) -> np.ndarray:
        out = np.full(len(idx), np.nan)
        for i, k in enumerate(idx):
            d = np.hypot((lon - st_lon[k]) * kx, (lat - st_lat[k]) * 110.57)
            if (d <= radius_km).any():
                out[i] = np.median(values[d <= radius_km])
        return out

    ax.plot(dist_km, level_m[idx], "o-", color="black", linewidth=1.2, label="COAST-RP station level")
    ax.plot(dist_km, near_median(c_lon, c_lat, seed), "s-", color=cmap(0.6), linewidth=2,
            label=f"eikonal coastline seed (IDW k={knn}, median within 3 km)")
    if zsmax is not None:
        ax.plot(dist_km, near_median(zsmax_lon, zsmax_lat, zsmax), "^-", color="#1f4e8c", linewidth=2,
                label="SFINCS peak water level zsmax (median within 3 km)")
    if bzs is not None:
        for k in range(bzs.shape[1] - 1):
            ax.axhline(bzs[:, k + 1].max(), color="#1f4e8c", linestyle=":", linewidth=1,
                       label="SFINCS bnd forcing peaks" if k == 0 else None)
    ax.set_xticks(dist_km)
    ax.set_xticklabels([f"{st_lon[k]:.2f}E\n{st_lat[k]:.2f}N" for k in idx], rotation=90, fontsize=7)
    ax.set_xlabel("COAST-RP stations inside the tile, ordered along the coast")
    ax.set_ylabel("water level (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    draw_panel_letter(ax, "b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--base-dir-name", default="validation_sfincs_v5")
    parser.add_argument("--friction-scale-factor", type=float, default=FRICTION_SCALE_FACTOR_DEFAULT,
                        help="which friction-sweep point's eikonal raster to compare against")
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_DEFAULT)
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
    plot_biggest_disagreement(tile_dir, fig_dir / f"{args.tile_id}_biggest_disagreement.png",
                              args.friction_scale_factor, args.max_outer_iterations)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    plot_coastrp_forcing(tile_dir, sfincs_dir, mask2d, transform, crs,
                         fig_dir / f"{args.tile_id}_coastrp_forcing.png",
                         knn=int(cfg["simulation"]["flooding"]["knn"]),
                         river_code=int(cfg["tile_generation"]["river_code"]))


if __name__ == "__main__":
    main()
