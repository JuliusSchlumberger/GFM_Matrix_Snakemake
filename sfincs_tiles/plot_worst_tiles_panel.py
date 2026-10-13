"""Two 3x3 panels of the worst-matching tiles between EA-bathtub and SFINCS
(lowest CSI), split by which model is over-predicting - one panel where
SFINCS-only area dominates the disagreement, one where EA-bathtub-only area
dominates. Each panel tile is a local map of the four-way classification
(matched / EA-bathtub-only / SFINCS-only / dry land) at SFINCS's own native
subgrid resolution - pixel-for-pixel what the CSI metric actually counts.

Tiles are ranked by their CSI at --friction-scale-factor from
{base}/friction_sweep_per_tile_metrics.csv (compute_friction_sweep_metrics.py).
Candidate tiles are restricted to MIN_UNION_KM2 km2 of total union area
(matched + EA-bathtub-only + SFINCS-only, CSI's own denominator), so a
technically-low CSI from a handful of noisy pixels doesn't crowd out a
real, substantial disagreement. Like the metrics, the maps count only land
that is active in SFINCS; land the boundary line cut off is drawn as
off-domain.

Needs only rasterio/scipy/matplotlib (sfincs_grid.py reads sfincs.msk directly).

Usage:
    python plot_worst_tiles_panel.py --base-dir-name calibration_roughness_SFINCS --friction-scale-factor 9 \
        [--n-tiles 9] [--min-union-km2 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import decode_waterdepth_cm, eikonal_path  # noqa: E402
from flood_agreement import WET_THRESHOLD_M  # noqa: E402
from gfm_config import read_root  # noqa: E402
from sfincs_grid import active_on_subgrid  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import (  # noqa: E402
    AGREE_COLOR, A_ONLY_COLOR, B_ONLY_COLOR, LAND_COLOR, LAND_LABEL, WATER_COLOR, WATER_LABEL,
    draw_caption_box, orient_north_up,
)

LAND_CODE = 0

# category codes for the 4-way classification. Colors/labels from
# src/map_style.py - the same palette validation/plot_agreement_map.py uses (SFINCS=A_ONLY/blue, EA-bathtub
# (eikonal)=B_ONLY/red, matched=AGREE/green, same convention everywhere
# this pairing appears).
DRY, MATCHED, EIKONAL_ONLY, SFINCS_ONLY = 0, 1, 2, 3
CATEGORY_COLORS = [LAND_COLOR, AGREE_COLOR, B_ONLY_COLOR, A_ONLY_COLOR]  # dry, matched, eikonal-only, sfincs-only
CATEGORY_LABELS = [LAND_LABEL, "Agree (both wet)", "EA-bathtub only", "SFINCS only"]
OFF_DOMAIN_COLOR = WATER_COLOR  # non-land (ocean/river/lake) - previously left blank/white,
# now drawn the same "permanent water" colour every other agreement map in the repo uses.


def build_classification(tile_dir: Path, friction_scale_factor: float) -> np.ndarray | None:
    """4-way classification array at SFINCS's own native subgrid
    resolution over SFINCS-active land (everything else masked), or None if
    SFINCS / the eikonal at `friction_scale_factor` hasn't been run for this tile."""
    hmax_subgrid_path = tile_dir / "sfincs_model" / "hmax_subgrid.tif"
    ek_path = eikonal_path(tile_dir / "outputs", friction_scale_factor)
    native_mask_path = tile_dir / "inputs" / "mask.tif"
    if not (hmax_subgrid_path.exists() and ek_path.exists() and native_mask_path.exists()):
        return None

    with rasterio.open(hmax_subgrid_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sg_nodata, sg_transform, sg_crs, sg_shape = src.nodata, src.transform, src.crs, src.shape
    if sg_nodata is not None and not np.isnan(sg_nodata):
        sfincs_depth = np.where(sfincs_depth == sg_nodata, np.nan, sfincs_depth)

    eikonal_depth, ek_transform, ek_crs, ek_shape = decode_waterdepth_cm(ek_path)
    if ek_shape != sg_shape:
        return None  # not pixel-identical (stale/mismatched raster) - skip rather than misrepresent

    mog = np.empty(sg_shape, dtype=np.float32)
    with rasterio.open(native_mask_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=sg_transform, dst_crs=sg_crs, resampling=Resampling.nearest,
        )
    land = (mog == LAND_CODE) & active_on_subgrid(tile_dir / "sfincs_model", sg_shape, sg_transform)

    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)
    eikonal_wet = land & np.isfinite(eikonal_depth) & (eikonal_depth > WET_THRESHOLD_M)

    cls = np.full(sg_shape, DRY, dtype=np.uint8)
    cls[land & eikonal_wet & sfincs_wet] = MATCHED
    cls[land & eikonal_wet & ~sfincs_wet] = EIKONAL_ONLY
    cls[land & ~eikonal_wet & sfincs_wet] = SFINCS_ONLY
    cls = np.ma.masked_where(~land, cls)  # non-land and SFINCS-inactive land drawn as off-domain
    return cls


def _crop_to_disagreement(cls: np.ma.MaskedArray, pad_frac: float = 0.25) -> np.ma.MaskedArray:
    """Crop to the bounding box of any wet cell (matched or either-only),
    padded - a coastal tile is mostly dry land/ocean, so a full-domain view
    would render the disagreement as a few stray pixels."""
    wet = np.isin(cls.filled(DRY), [MATCHED, EIKONAL_ONLY, SFINCS_ONLY])
    if not wet.any():
        return cls
    rows, cols = np.nonzero(wet)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    pad_r = max(int((r1 - r0) * pad_frac), 5)
    pad_c = max(int((c1 - c0) * pad_frac), 5)
    r0, r1 = max(r0 - pad_r, 0), min(r1 + pad_r + 1, cls.shape[0])
    c0, c1 = max(c0 - pad_c, 0), min(c1 + pad_c + 1, cls.shape[1])
    return cls[r0:r1, c0:c1]


def _make_panel(worst, base_dir: Path, out_path: Path, friction_scale_factor: float) -> None:
    ncols = 3
    nrows = int(np.ceil(len(worst) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    cmap = ListedColormap(CATEGORY_COLORS)

    for ax, row in zip(axes.flat, worst.itertuples()):
        tile_dir = base_dir / str(row.tile_id)
        cls = build_classification(tile_dir, friction_scale_factor)
        ax.set_facecolor(OFF_DOMAIN_COLOR)
        if cls is None:
            ax.text(0.5, 0.5, "no subgrid data", ha="center", va="center", transform=ax.transAxes)
        else:
            # build_classification keeps the subgrid's own (south-up) row order -
            # flipped here for display only.
            with rasterio.open(tile_dir / "sfincs_model" / "hmax_subgrid.tif") as src:
                cls = orient_north_up(cls, src.transform)
            cls = _crop_to_disagreement(cls)
            ax.imshow(cls, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        draw_caption_box(ax, f"tile {row.tile_id}  CSI={row.csi:.3f}  union={row.union_km2:.2f} km2")

    for ax in axes.flat[len(worst):]:
        ax.axis("off")

    legend_handles = [Patch(facecolor=c, edgecolor="grey", linewidth=0.5, label=l)
                       for c, l in zip(CATEGORY_COLORS, CATEGORY_LABELS)]
    legend_handles.append(Patch(facecolor=OFF_DOMAIN_COLOR, edgecolor="grey", linewidth=0.5, label=WATER_LABEL))
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--friction-scale-factor", type=float, required=True, help="the friction point to rank and plot")
    parser.add_argument("--n-tiles", type=int, default=9)
    parser.add_argument("--min-union-km2", type=float, default=1.0,
                         help="minimum matched+EA-bathtub-only+SFINCS-only area (CSI's own denominator) "
                              "for a tile to be eligible")
    args = parser.parse_args()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fsf = args.friction_scale_factor

    per_tile_path = base_dir / "friction_sweep_per_tile_metrics.csv"
    per_tile = pd.read_csv(per_tile_path)
    df = per_tile[per_tile["friction_scale_factor"] == fsf].copy()
    df = df.rename(columns={"CSI": "csi", "sfincs_only_km2": "eikonal_sfincs_only_km2"})
    df["union_km2"] = df["matched_km2"] + df["eikonal_only_km2"] + df["eikonal_sfincs_only_km2"]
    df = df[np.isfinite(df["csi"]) & (df["union_km2"] >= args.min_union_km2)]
    print(f"{len(df)} tile(s) with union area >= {args.min_union_km2} km2 eligible "
          f"(friction_scale_factor={fsf:g}, from {per_tile_path.name})")

    n = args.n_tiles
    sfincs_over = df[df["eikonal_sfincs_only_km2"] > df["eikonal_only_km2"]].sort_values("csi").head(n)
    eikonal_over = df[df["eikonal_only_km2"] > df["eikonal_sfincs_only_km2"]].sort_values("csi").head(n)

    cols = ["tile_id", "csi", "union_km2", "eikonal_only_km2", "eikonal_sfincs_only_km2"]
    print(f"\nWorst {len(sfincs_over)} tile(s), SFINCS over-predicting:")
    print(sfincs_over[cols].to_string(index=False))
    print(f"\nWorst {len(eikonal_over)} tile(s), EA-bathtub over-predicting:")
    print(eikonal_over[cols].to_string(index=False))

    sfx = f"_fsf{fsf:g}"
    selection = pd.concat([sfincs_over[cols].assign(panel="sfincs_overpredicts"),
                           eikonal_over[cols].assign(panel="eikonal_overpredicts")])
    selection["rank"] = selection.groupby("panel").cumcount() + 1
    selection_path = fig_dir / f"worst_tiles_selection{sfx}.csv"
    selection[["panel", "rank", *cols]].to_csv(selection_path, index=False)
    print(f"Wrote {selection_path}")

    _make_panel(sfincs_over, base_dir, fig_dir / f"worst_tiles_panel_sfincs_overpredicts{sfx}.png", fsf)
    _make_panel(eikonal_over, base_dir, fig_dir / f"worst_tiles_panel_eikonal_overpredicts{sfx}.png", fsf)


if __name__ == "__main__":
    main()
