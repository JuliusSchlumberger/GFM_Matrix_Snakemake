"""Spatial agreement maps for the worst N eikonal-vs-SFINCS tiles, ranked by
eikonal_km2/sfincs_km2 ratio ascending (worst underestimation only): where
SFINCS and eikonal agree, where SFINCS floods but eikonal doesn't, and
where eikonal floods but SFINCS doesn't. All on the SFINCS subgrid UTM
grid, using flood_agreement.py's own WET_THRESHOLD_M.

Superseded as a standalone CLI by plot_eikonal_disagreement_extremes.py,
but `build_rgb()` below is still imported directly by other scripts as a
library function.

Run under gfm_python_preprocessing, not hydromt-sfincs-dev - matplotlib
savefig() is broken in that environment.

Usage:
    python plot_worst_tiles_comparison.py
    python plot_worst_tiles_comparison.py --n-tiles 30 --min-sfincs-km2 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import WET_THRESHOLD_M  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import (  # noqa: E402
    AGREE_COLOR as COLOR_AGREE,
    A_ONLY_COLOR as COLOR_SFINCS_ONLY,
    B_ONLY_COLOR as COLOR_EIKONAL_ONLY,
    LAND_COLOR as COLOR_DRY,
    LAND_LABEL,
    WATER_COLOR as COLOR_WATER,
    WATER_LABEL,
    draw_caption_box,
)

LAND_CODE = 0
OCEAN_CODE = 1
LAKE_CODE = 2
RIVER_CODE = 3
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767


def _decode_waterdepth_cm(path: Path) -> np.ndarray:
    """int16-cm -> float32 metres, NaN at nodata."""
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m

# COLOR_DRY/COLOR_WATER/COLOR_AGREE/COLOR_SFINCS_ONLY/COLOR_EIKONAL_ONLY all
# come from src/map_style.py (imported above as COLOR_DRY/COLOR_WATER/etc.) -
# the same land/water/agree/A-only/B-only palette every agreement map in the
# repo now shares (validation/plot_agreement_map.py included). Ocean and
# lake/river are ONE "permanent water" colour here, not two - matching
# validation's own permanent_water_source convention (ocean+lake+river
# excluded as a single category), not a SFINCS-specific distinction.

N_TILES_DEFAULT = 30
MIN_SFINCS_KM2_DEFAULT = 1.0
NCOLS_DEFAULT = 6


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_rgb(tile_id: str, root: Path, base_dir_name: str) -> np.ndarray:
    tile_dir = root / base_dir_name / tile_id
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    native_mask_path = tile_dir / "inputs" / "mask.tif"

    hmax_subgrid_path = sfincs_dir / "hmax_subgrid.tif"
    eikonal_path = out_dir / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"

    with retry_transient_io(rasterio.open, hmax_subgrid_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sfincs_nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = src.shape
    if sfincs_nodata is not None and not np.isnan(sfincs_nodata):
        sfincs_depth = np.where(sfincs_depth == sfincs_nodata, np.nan, sfincs_depth)

    eikonal_depth = _decode_waterdepth_cm(eikonal_path)

    with retry_transient_io(rasterio.open, native_mask_path) as src:
        mog = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )

    water = (mog == OCEAN_CODE) | (mog == LAKE_CODE) | (mog == RIVER_CODE)
    land = mog == LAND_CODE

    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)
    eikonal_wet = land & np.isfinite(eikonal_depth) & (eikonal_depth > WET_THRESHOLD_M)

    rgb = np.empty(shape + (3,), dtype=np.float32)
    rgb[...] = _hex_to_rgb(COLOR_DRY)
    rgb[water] = _hex_to_rgb(COLOR_WATER)
    rgb[land & sfincs_wet & eikonal_wet] = _hex_to_rgb(COLOR_AGREE)
    rgb[land & sfincs_wet & ~eikonal_wet] = _hex_to_rgb(COLOR_SFINCS_ONLY)
    rgb[land & ~sfincs_wet & eikonal_wet] = _hex_to_rgb(COLOR_EIKONAL_ONLY)
    return rgb


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    parser.add_argument("--n-tiles", type=int, default=N_TILES_DEFAULT)
    parser.add_argument("--min-sfincs-km2", type=float, default=MIN_SFINCS_KM2_DEFAULT)
    parser.add_argument("--ncols", type=int, default=NCOLS_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {base_dir_name}/worst_tiles_comparison.png")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    # Ranked by eikonal_km2/sfincs_km2 ratio, ascending = worst underestimation.
    # For ranking both worst-under and worst-over tiles, see
    # plot_eikonal_disagreement_extremes.py instead.
    df = pd.read_csv(base_dir / "all_tiles_summary.csv")
    df = df.dropna(subset=["eikonal_km2", "sfincs_km2"])
    df = df[df["sfincs_km2"] > args.min_sfincs_km2].copy()
    df["ratio"] = df["eikonal_km2"] / df["sfincs_km2"]
    df = df.sort_values("ratio").head(args.n_tiles).reset_index(drop=True)
    print(f"{len(df)} tile(s) selected (worst eikonal_km2/sfincs_km2 ratio, sfincs_km2 > {args.min_sfincs_km2})")

    n = len(df)
    ncols = args.ncols
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, df.iterrows()):
        tile_id = str(int(row["tile_id"]))
        try:
            rgb = build_rgb(tile_id, root, args.base_dir_name)
        except Exception as e:
            ax.text(0.5, 0.5, f"{tile_id}\nERROR: {type(e).__name__}", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.imshow(rgb, origin="upper")
        draw_caption_box(ax, [
            f"tile {tile_id} (set {row.get('set', '?')})",
            f"eikonal={row['eikonal_km2']:.2f} sfincs={row['sfincs_km2']:.2f} km2, ratio={row['ratio']:.2f}",
        ])
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(facecolor=COLOR_AGREE, edgecolor="black", label="Agree (both wet)"),
        mpatches.Patch(facecolor=COLOR_SFINCS_ONLY, edgecolor="black", label="SFINCS only"),
        mpatches.Patch(facecolor=COLOR_EIKONAL_ONLY, edgecolor="black", label="EA-bathtub only"),
        mpatches.Patch(facecolor=COLOR_DRY, edgecolor="black", label=LAND_LABEL),
        mpatches.Patch(facecolor=COLOR_WATER, edgecolor="black", label=WATER_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = Path(args.out) if args.out else base_dir / "worst_tiles_comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
