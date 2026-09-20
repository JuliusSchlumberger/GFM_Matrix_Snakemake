"""Diagnostic plots for a built (not yet necessarily run) SFINCS tile model,
to sanity-check build_sfincs_tile.py's own output BEFORE spending time
running the simulation: domain/grid, mask (active/boundary cells),
elevation, roughness, zsini, and the boundary-condition hydrograph
timeseries (with each station's original COAST-RP peak value overlaid as a
cross-check - see build_boundary_forcing.py's own MDT-offset construction,
which makes the corrected hydrograph's peak land close to that value by
construction, so a large mismatch here would flag a real bug).

Reads only build_sfincs_tile.py's own already-written output (gis/*.tif,
matched_boundary_points.gpkg, corrected_hydrographs.csv) - does not re-open
the model via hydromt_sfincs, so this also works for the model-build step
before a run and needs nothing beyond rasterio/geopandas/pandas/matplotlib.

Run this under the MAIN gfm_python_preprocessing env, NOT hydromt-sfincs-dev:
matplotlib's Agg canvas crashes with a hard native abort (exit 0xC06D007F)
in hydromt-sfincs-dev specifically - confirmed reproducible for any real
FigureCanvasAgg.draw() call regardless of pyplot vs the OO API, PNG vs SVG,
or whether any text is drawn, i.e. a broken matplotlib install in that env,
unrelated to this script. gfm_python_preprocessing has a working matplotlib
and every other library this script needs, with no hydromt_sfincs
dependency at all:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\gfm_python_preprocessing\\python.exe plot_sfincs_build.py --tile-ids 1573 1907 929
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.plot import show as rio_show

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

ELEVATION_CBAR_RANGE_M = 30.0  # fixed +-30 m display window, see build_elevation.py's MIN_BATHYMETRY_M


def _read_band(path: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        return arr, src.transform, str(src.crs)


def plot_tile_build(tile_id: str, root: Path, out_path: Path | None = None) -> Path:
    sfincs_dir = root / "validation_sfincs" / tile_id / "sfincs_model"
    gis_dir = sfincs_dir / "gis"
    if not gis_dir.exists():
        raise FileNotFoundError(f"{gis_dir} not found - has build_sfincs_tile.py been run for tile {tile_id}?")

    dep, transform, crs = _read_band(gis_dir / "dep.tif")
    manning, _, _ = _read_band(gis_dir / "manning.tif")
    mask, _, _ = _read_band(gis_dir / "mask.tif")
    zsini, _, _ = _read_band(gis_dir / "zs.tif")

    matched_points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg").to_crs(crs)
    hydrographs = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")
    elapsed_hr = hydrographs["elapsed_hr"].to_numpy()
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]

    # boundary/COAST-HG stations are real GTSM points, matched with a buffer
    # of up to 100 km from the tile (see build_sfincs_tile.py's own
    # water_level.create(buffer=100_000.0) comment) - genuinely far outside
    # a ~2-10 km tile's own extent, so a scatter overlay of them auto-scales
    # the axes out until the tile's own raster shrinks to an invisible
    # sliver. Grid extent (raster bounds) is fixed explicitly after any such
    # overlay so the tile itself stays the visible subject of the panel - with
    # a small margin on every side so the domain outline itself isn't drawn
    # flush against the panel border (every raster panel uses this, not just
    # the ones with a point overlay, so all four stay visually consistent).
    left, bottom, right, top = rasterio.transform.array_bounds(*dep.shape, transform)
    _margin = 0.06 * max(right - left, top - bottom)
    xlim = (left - _margin, right + _margin)
    ylim = (bottom - _margin, top + _margin)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SFINCS build check - tile {tile_id}  ({crs}, grid {dep.shape[1]}x{dep.shape[0]})", fontsize=13)

    with rasterio.open(gis_dir / "dep.tif") as dep_src:
        # -- mask: active (1) / waterlevel boundary (2) --
        ax = axes[0, 0]
        mask_cmap = ListedColormap(["#a6cee3", "#e31a1c"])
        mask_norm = BoundaryNorm([0.5, 1.5, 2.5], mask_cmap.N)
        rio_show(np.where(np.isnan(mask), 0, mask), transform=transform, ax=ax, cmap=mask_cmap, norm=mask_norm)

        # waterlevel-boundary cells are typically a real, physically-expected
        # ONE-cell-wide fringe along the seaward domain edge (confirmed here:
        # tile 1573 has them along the bottom row + left/right columns only) -
        # imshow alone renders that as a sub-pixel-wide line that disappears
        # under anti-aliasing at this figure's resolution, so scatter the
        # boundary-cell centers explicitly on top to keep them visible.
        bnd_rows, bnd_cols = np.where(mask == 2)
        bnd_x, bnd_y = rasterio.transform.xy(transform, bnd_rows, bnd_cols)
        ax.scatter(bnd_x, bnd_y, color="#e31a1c", s=4, marker="s", zorder=4, label="waterlevel boundary cell")

        matched_points.plot(ax=ax, color="black", markersize=10, marker="x", zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"mask (active={int(np.nansum(mask == 1))}, boundary={int(np.nansum(mask == 2))})\n"
                      f"x = matched COAST-RP/COAST-HG boundary points (k-nearest, may still be off-panel)")
        # (a wide-view "station context" inset was tried here to show
        # stations relative to the tile at full 100 km scale, but for an
        # oblong tile the equal-aspect data band occupies only a thin strip
        # of the axes' bounding box, so a corner-anchored inset can overlap
        # and hide the real boundary-cell markers along that edge - dropped
        # rather than risk silently hiding real diagnostic content.)

        # -- elevation --
        ax = axes[0, 1]
        # fixed +-30 m, not the tile's own min/max: ocean depth is now
        # floored at build_elevation.py's own -50 m (storm surge never
        # reaches deeper, see MIN_BATHYMETRY_M there), and most land in
        # these tiles sits within a similar band, so a fixed symmetric
        # window keeps every tile's panel visually comparable and keeps
        # near-shore/land detail visible - real values outside +-30 m
        # (rare - e.g. a genuine hill, or the pre-floor deep shelf) are
        # shown clipped/saturated (colorbar arrows) rather than silently
        # rescaling per tile.
        vmax = ELEVATION_CBAR_RANGE_M
        cmap = plt.get_cmap("BrBG_r").copy()
        cmap.set_under("#003c30")
        cmap.set_over("#3d1d02")
        im = rio_show(dep, transform=transform, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(im.get_images()[0], ax=ax, label="elevation (m, GOCO06s-corrected)", fraction=0.046, extend="both")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"elevation (dep): {np.nanmin(dep):.2f} to {np.nanmax(dep):.2f} m")

        # -- roughness --
        ax = axes[0, 2]
        im = rio_show(manning, transform=transform, ax=ax, cmap="viridis")
        fig.colorbar(im.get_images()[0], ax=ax, label="Manning's n", fraction=0.046)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"roughness (manning): {np.nanmin(manning):.4f} to {np.nanmax(manning):.4f}")

        # -- zsini --
        ax = axes[1, 0]
        im = rio_show(zsini, transform=transform, ax=ax, cmap="viridis")
        fig.colorbar(im.get_images()[0], ax=ax, label="zsini (m)", fraction=0.046)
        matched_points.plot(ax=ax, color="red", markersize=10, marker="x", zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"initial water level (zsini): {np.nanmin(zsini):.4f} to {np.nanmax(zsini):.4f} m")

    # -- boundary hydrographs --
    ax = axes[1, 1]
    cmap = plt.get_cmap("tab20")
    for i, col in enumerate(station_cols):
        ax.plot(elapsed_hr, hydrographs[col], color=cmap(i % 20), lw=0.8, alpha=0.7)
    # cross-check: each matched point's ORIGINAL COAST-RP value (cm -> m) plotted
    # as an 'x' at t=0 - build_boundary_forcing.py's own MDT-offset construction
    # makes the corrected hydrograph's own peak land close to this value, so a
    # large mismatch here (not just a small residual) indicates a real bug.
    orig_vals_m = matched_points.set_index("boundary_idx")["SLR_0"].astype(float) / 100.0
    for i, col in enumerate(station_cols):
        boundary_idx = int(col)
        if boundary_idx in orig_vals_m.index:
            ax.scatter([elapsed_hr[0]], [orig_vals_m.loc[boundary_idx]], color=cmap(i % 20), marker="x", s=25, zorder=5)
    ax.set_xlabel("elapsed time (h)")
    ax.set_ylabel("water level (m)")
    ax.set_title(f"MDT-corrected COAST-HG hydrographs (n={len(station_cols)})\n"
                 "x = original COAST-RP boundary value (cross-check)")
    ax.grid(alpha=0.3)

    # -- summary text --
    ax = axes[1, 2]
    ax.axis("off")
    n_active = int(np.nansum(mask == 1))
    n_bnd = int(np.nansum(mask == 2))
    # real applied offset, from build_boundary_forcing.py's own
    # mdt_offset_m column - NOT "boundary_val - max(corrected hydrograph)":
    # the corrected hydrograph already has the offset added in, so its own
    # max is, by construction, always == boundary_val (a self-referential
    # calculation that silently produces ~0 regardless of the real offset -
    # the actual bug behind this panel showing "always 0" before this fix).
    if "mdt_offset_m" in matched_points.columns:
        offsets = matched_points.set_index("boundary_idx")["mdt_offset_m"].astype(float).to_numpy()
    else:
        offsets = np.array([np.nan])  # built before mdt_offset_m was persisted - rerun build_boundary_forcing.py
    summary = (
        f"tile_id: {tile_id}\n"
        f"crs: {crs}\n"
        f"grid: {dep.shape[1]} x {dep.shape[0]} cells\n"
        f"active cells: {n_active}\n"
        f"waterlevel boundary cells: {n_bnd}\n"
        f"\n"
        f"elevation: {np.nanmin(dep):.2f} to {np.nanmax(dep):.2f} m\n"
        f"manning's n: {np.nanmin(manning):.4f} to {np.nanmax(manning):.4f}\n"
        f"zsini: {np.nanmin(zsini):.4f} to {np.nanmax(zsini):.4f} m\n"
        f"\n"
        f"boundary stations: {len(station_cols)}\n"
        f"hydrograph span: {elapsed_hr[-1]:.1f} h\n"
        f"MDT offset (boundary - raw hg_max):\n"
        f"  min={offsets.min():+.4f}  max={offsets.max():+.4f}\n"
        f"  mean={offsets.mean():+.4f}  std={offsets.std():.4f} m\n"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", fontfamily="monospace", fontsize=10)

    fig.tight_layout()
    out_path = out_path or (root / "validation_sfincs" / tile_id / "diagnostics" / "build_check.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-ids", nargs="+", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    root = read_root(Path(args.config))
    for tile_id in args.tile_ids:
        try:
            out_path = plot_tile_build(str(tile_id), root)
            print(f"tile {tile_id}: wrote {out_path}")
        except FileNotFoundError as e:
            print(f"tile {tile_id}: SKIPPED - {e}")


if __name__ == "__main__":
    main()
