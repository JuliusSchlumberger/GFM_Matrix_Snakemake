"""Spatial agreement maps for the worst N eikonal-vs-SFINCS tiles (ranked by
eikonal_km2/sfincs_km2 ratio, ascending - i.e. worst underestimation only;
for BOTH worst-under and worst-over tiles in one run, see the dedicated
plot_eikonal_disagreement_extremes.py instead): where SFINCS and eikonal
agree, where SFINCS floods but eikonal doesn't (SFINCS-only/over-predicts
relative to eikonal), and where eikonal floods but SFINCS doesn't
(eikonal-only). All on the SFINCS subgrid UTM grid, reusing
flood_agreement.py's own WET_THRESHOLD_M (same convention as the pooled
HT/FAR/CSI/bias numbers) so the picture matches those numbers exactly.
`build_rgb()` below is the reusable part - imported directly by
plot_eikonal_disagreement_extremes.py.

Run under gfm_python_preprocessing (NOT hydromt-sfincs-dev) -
matplotlib.pyplot.savefig() crashes with exit code 127 under
hydromt-sfincs-dev, a real documented issue this session (broken native
BLAS/font-rendering backend in that env).

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

LAND_CODE = 0
OCEAN_CODE = 1
LAKE_CODE = 2
RIVER_CODE = 3
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767


def _decode_waterdepth_cm(path: Path) -> np.ndarray:
    """int16-cm -> float32 metres, NaN at nodata - same convention as
    postprocess_tile_summary.py's own _decode_waterdepth_cm (kept as a
    separate local copy rather than a shared import, matching that
    module's own precedent: this tiny decode is duplicated per-script on
    purpose, not centralized, since it's the one piece of raster I/O each
    of these otherwise-independent scripts needs)."""
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m

COLOR_OCEAN = "#a3b9cc"        # grey-blue (2026-09-24, user direction - was light grey; eikonal-only
# recolored to orange below as a result, since blue was no longer distinct enough from ocean)
COLOR_WATERBODY = "#8c8c8c"    # medium grey - lake/river (outside the compared land-only domain)
COLOR_DRY = "#ffffff"          # land, dry in both
COLOR_AGREE = "#2ca02c"        # green - both wet
COLOR_SFINCS_ONLY = "#d62728"  # red - SFINCS wet, eikonal dry (SFINCS over-predicts vs eikonal)
COLOR_EIKONAL_ONLY = "#eda100"  # yellow (was blue, then orange - both too close to the new
# grey-blue ocean and/or to COLOR_SFINCS_ONLY's red respectively; user-confirmed orange/red were
# "literally indistinguishable") - eikonal wet, SFINCS dry (eikonal over-predicts vs SFINCS)

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

    ocean = mog == OCEAN_CODE
    waterbody = (mog == LAKE_CODE) | (mog == RIVER_CODE)
    land = mog == LAND_CODE

    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)
    eikonal_wet = land & np.isfinite(eikonal_depth) & (eikonal_depth > WET_THRESHOLD_M)

    rgb = np.empty(shape + (3,), dtype=np.float32)
    rgb[...] = _hex_to_rgb(COLOR_DRY)
    rgb[ocean] = _hex_to_rgb(COLOR_OCEAN)
    rgb[waterbody] = _hex_to_rgb(COLOR_WATERBODY)
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

    # Ranked by eikonal_km2/sfincs_km2 ratio (ascending = worst underestimation),
    # not eikonal_HT/CSI - those per-tile ratio columns were deliberately dropped
    # from calibration_metrics_per_tile.csv (2026-09-24, user direction: HT/FAR/
    # CSI/bias should only ever be computed once, pooled across every tile, never
    # per-tile - see compute_calibration_metrics.py's own module docstring). For
    # ranking BOTH worst-under and worst-over tiles, see the dedicated
    # plot_eikonal_disagreement_extremes.py instead of this script's own CLI.
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
        ax.set_title(
            f"tile {tile_id} (set {row.get('set', '?')})\n"
            f"eikonal={row['eikonal_km2']:.2f} sfincs={row['sfincs_km2']:.2f} km2, ratio={row['ratio']:.2f}",
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(facecolor=COLOR_OCEAN, edgecolor="black", label="ocean"),
        mpatches.Patch(facecolor=COLOR_WATERBODY, edgecolor="black", label="lake/river"),
        mpatches.Patch(facecolor=COLOR_DRY, edgecolor="black", label="dry land"),
        mpatches.Patch(facecolor=COLOR_AGREE, edgecolor="black", label="agree (both wet)"),
        mpatches.Patch(facecolor=COLOR_SFINCS_ONLY, edgecolor="black", label="SFINCS only"),
        mpatches.Patch(facecolor=COLOR_EIKONAL_ONLY, edgecolor="black", label="eikonal only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Worst {n} tiles by eikonal hit-rate (SFINCS vs eikonal, land cells only)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = Path(args.out) if args.out else base_dir / "worst_tiles_comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
