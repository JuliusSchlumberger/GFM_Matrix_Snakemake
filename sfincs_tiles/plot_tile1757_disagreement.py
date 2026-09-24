"""Overview + zoomed visual confirmation of tile 1757's eikonal-vs-SFINCS
disagreement (eikonal_km2=0.0, sfincs_km2=2.43 - SFINCS floods land that
eikonal's peak-IDW-seeded geometric fill never overtops onto, per the
2026-09-24 investigation: eikonal's only nonzero output on this tile sits on
river cells, never crossing onto land).

Matches the established tile 1736/865 investigation pattern (regional
overview + zoomed elevation/water-level panels, not just summary numbers) -
6-panel layout: overview categorical agreement map (with zoom-box marker),
zoomed categorical agreement map, zoomed elevation, zoomed mask (to show the
river channel geometry), zoomed SFINCS depth, zoomed eikonal depth.

Run under gfm_python_preprocessing (NOT hydromt-sfincs-dev) - matplotlib
savefig crashes under hydromt-sfincs-dev in this environment.

Usage:
    python plot_tile1757_disagreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import WET_THRESHOLD_M  # noqa: E402
from gfm_config import read_root  # noqa: E402
from plot_worst_tiles_comparison import (  # noqa: E402
    COLOR_AGREE, COLOR_DRY, COLOR_EIKONAL_ONLY, COLOR_OCEAN, COLOR_SFINCS_ONLY, COLOR_WATERBODY,
    LAND_CODE, LAKE_CODE, OCEAN_CODE, RIVER_CODE, _decode_waterdepth_cm, _hex_to_rgb, build_rgb,
)

TILE_ID = "1757"
BASE_DIR_NAME = "validation_sfincs_v3"
MARGIN_PX = 60  # zoom-window margin around the disagreement bbox, in subgrid pixels (~1.8km at 30m)


def main() -> None:
    root = read_root(Path(__file__).resolve().parent.parent / "snakemake_workflow" / "config" / "config.yml")
    tile_dir = root / BASE_DIR_NAME / TILE_ID
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    native_mask_path = tile_dir / "inputs" / "mask.tif"

    hmax_path = sfincs_dir / "hmax_subgrid.tif"
    dep_path = sfincs_dir / "subgrid" / "dep_subgrid.tif"
    eikonal_path = out_dir / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"

    with rasterio.open(hmax_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sfincs_nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = src.shape
    if sfincs_nodata is not None and not np.isnan(sfincs_nodata):
        sfincs_depth = np.where(sfincs_depth == sfincs_nodata, np.nan, sfincs_depth)

    with rasterio.open(dep_path) as src:
        dep = src.read(1).astype(np.float32)
        dep_nodata = src.nodata
    if dep_nodata is not None:
        dep = np.where(dep == dep_nodata, np.nan, dep)

    eikonal_depth = _decode_waterdepth_cm(eikonal_path)

    with rasterio.open(native_mask_path) as src:
        mog = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )

    land = mog == LAND_CODE
    ocean = mog == OCEAN_CODE
    waterbody = (mog == LAKE_CODE) | (mog == RIVER_CODE)

    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)
    eikonal_wet = land & np.isfinite(eikonal_depth) & (eikonal_depth > WET_THRESHOLD_M)
    sfincs_only = land & sfincs_wet & ~eikonal_wet

    n_sfincs_only = int(sfincs_only.sum())
    print(f"SFINCS-only flooded land cells: {n_sfincs_only}")
    if n_sfincs_only == 0:
        raise RuntimeError("no SFINCS-only disagreement cells found - nothing to zoom in on")

    # sfincs_only is scattered across the whole tile (452 separate components) - a naive
    # full bbox around every cell spans nearly the entire tile width, which isn't a
    # "zoom" at all. Zoom instead on the single largest connected cluster (the real,
    # substantial disagreement, not the scattered 1-2 cell noise elsewhere).
    from scipy import ndimage
    labels, n_components = ndimage.label(sfincs_only, structure=np.ones((3, 3), dtype=bool))
    sizes = ndimage.sum(sfincs_only, labels, index=np.arange(1, n_components + 1))
    biggest_label = int(np.argmax(sizes)) + 1
    print(f"{n_components} disagreement clusters total; biggest has {int(sizes.max())} cells "
          f"(of {n_sfincs_only} total) - zooming on that one")
    rows, cols = np.nonzero(labels == biggest_label)
    r0, r1 = rows.min() - MARGIN_PX, rows.max() + MARGIN_PX
    c0, c1 = cols.min() - MARGIN_PX, cols.max() + MARGIN_PX
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0] - 1), min(c1, shape[1] - 1)
    print(f"zoom window: rows [{r0}:{r1}] cols [{c0}:{c1}] (full tile shape {shape})")

    rgb_full = build_rgb(TILE_ID, root, BASE_DIR_NAME)
    rgb_zoom = rgb_full[r0:r1, c0:c1]
    dep_zoom = dep[r0:r1, c0:c1]
    mog_zoom = mog[r0:r1, c0:c1]
    sfincs_depth_zoom = sfincs_depth[r0:r1, c0:c1]
    eikonal_depth_zoom = eikonal_depth[r0:r1, c0:c1]

    # 2026-09-24 addition: the same zsmax-vs-zs(time) numerical-instability check used on
    # tiles 1736/1575 - zsmax (fine-timestep running max) spiking way above zs's own
    # smoothly-varying max-over-30min-snapshots max is the established artifact signature.
    # Computed on the COARSE grid (sfincs_map.nc's own native resolution), then upsampled
    # to the subgrid via pure block-repetition (same coarse/fine grid, same origin/CRS -
    # no reprojection, so no risk of re-introducing a rotation artifact into the check
    # itself, per this session's own established artifact-free methodology).
    with xr.open_dataset(sfincs_dir / "sfincs_map.nc") as map_ds:
        zsmax_coarse = map_ds["zsmax"].isel(timemax=-1).values
        zs_coarse = map_ds["zs"].values
        msk_coarse = map_ds["msk"].values
    zs_max_over_time_coarse = np.nanmax(zs_coarse, axis=0)
    gap_coarse = zsmax_coarse - zs_max_over_time_coarse
    gap_coarse = np.where(msk_coarse > 0, gap_coarse, np.nan)
    refi = sfincs_depth.shape[0] // gap_coarse.shape[0]
    gap_subgrid = np.repeat(np.repeat(gap_coarse, refi, axis=0), refi, axis=1)
    gap_subgrid = gap_subgrid[:sfincs_depth.shape[0], :sfincs_depth.shape[1]]
    gap_zoom = gap_subgrid[r0:r1, c0:c1]
    n_gap_big_zoom = int(np.nansum(gap_zoom > 0.5))
    n_finite_zoom = int(np.isfinite(gap_zoom).sum())
    print(f"zsmax-vs-zs gap in zoom window: {n_gap_big_zoom} of {n_finite_zoom} ever-wet cells > 0.5m "
          f"({100 * n_gap_big_zoom / n_finite_zoom:.1f}%) - numerical-instability check")

    fig, axes = plt.subplots(2, 4, figsize=(23, 11))

    ax = axes[0, 0]
    ax.imshow(rgb_full, origin="upper")
    rect = mpatches.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="black", facecolor="none", linewidth=2)
    ax.add_patch(rect)
    ax.set_title(f"tile {TILE_ID} overview (agreement map)\nblack box = zoom region below", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(rgb_zoom, origin="upper")
    ax.set_title("zoomed agreement map\n(red = SFINCS floods, eikonal doesn't)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    mask_rgb = np.empty(mog_zoom.shape + (3,), dtype=np.float32)
    mask_rgb[...] = _hex_to_rgb(COLOR_DRY)
    mask_rgb[mog_zoom == OCEAN_CODE] = _hex_to_rgb(COLOR_OCEAN)
    mask_rgb[(mog_zoom == LAKE_CODE) | (mog_zoom == RIVER_CODE)] = _hex_to_rgb(COLOR_WATERBODY)
    ax.imshow(mask_rgb, origin="upper")
    ax.set_title("zoomed land/ocean/river mask\n(grey = river/lake channel)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 3]
    im = ax.imshow(gap_zoom, origin="upper", cmap="magma", vmin=0, vmax=np.nanpercentile(gap_zoom, 99) if np.isfinite(gap_zoom).any() else 1)
    ax.set_title(
        f"zsmax-vs-zs(time) gap (m)\nnumerical-instability check: {n_gap_big_zoom}/{n_finite_zoom} "
        f"({100 * n_gap_big_zoom / max(n_finite_zoom, 1):.0f}%) cells > 0.5m",
        fontsize=10,
    )
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(dep_zoom, origin="upper", cmap="terrain", vmin=np.nanpercentile(dep_zoom, 1), vmax=np.nanpercentile(dep_zoom, 99))
    ax.set_title("zoomed elevation (dep_subgrid, m)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    sfincs_plot = np.where(np.isfinite(sfincs_depth_zoom) & (sfincs_depth_zoom > 0), sfincs_depth_zoom, np.nan)
    im = ax.imshow(sfincs_plot, origin="upper", cmap="Blues", vmin=0, vmax=np.nanmax(sfincs_plot) if np.isfinite(sfincs_plot).any() else 1)
    ax.set_title(f"zoomed SFINCS depth (hmax_subgrid, m)\nmax={np.nanmax(sfincs_plot):.2f}m" if np.isfinite(sfincs_plot).any() else "zoomed SFINCS depth", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    eikonal_plot = np.where(np.isfinite(eikonal_depth_zoom) & (eikonal_depth_zoom > 0), eikonal_depth_zoom, np.nan)
    has_eik = np.isfinite(eikonal_plot).any()
    im = ax.imshow(eikonal_plot, origin="upper", cmap="Blues", vmin=0, vmax=(np.nanmax(eikonal_plot) if has_eik else 1))
    title = f"zoomed eikonal depth (m)\nmax={np.nanmax(eikonal_plot):.2f}m" if has_eik else "zoomed eikonal depth\n(no cells > 0)"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 3].set_visible(False)

    fig.suptitle(
        f"tile {TILE_ID}: SFINCS floods {n_sfincs_only} land cells eikonal never wets (eikonal_km2=0.0, sfincs_km2=2.43) - "
        f"but {100 * n_gap_big_zoom / max(n_finite_zoom, 1):.0f}% of this cluster's cells show a zsmax numerical-instability "
        f"spike >0.5m, so SFINCS's own depth here (up to 10.83m) is itself suspect, not a clean ground truth",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / "tile1757_disagreement_confirmation.png"
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
