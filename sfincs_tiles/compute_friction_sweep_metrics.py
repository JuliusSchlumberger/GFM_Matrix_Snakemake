"""Scores a friction_scale_factor sweep of run_eikonal_on_sfincs_subgrid.py
(eikonal-vs-SFINCS, on SFINCS's own subgrid) against SFINCS as ground truth,
pooled CSI/bias per sweep point across a fixed tile set
(select_friction_sweep_tiles.py).

Reimplements postprocess_tile_summary.py's summarize_extent_model agreement
logic directly on an arbitrary eikonal raster path (instead of going through
its fixed MODEL_WATERDEPTH_FILENAME + summary_eikonal.json indirection,
which assumes one eikonal result per tile) - same WET_THRESHOLD_M cutoff,
same confusion_counts/metrics_from_counts primitives
(sfincs_tiles/flood_agreement.py), same land-only domain mask. Needs only
rasterio/scipy/numpy - no hydromt_sfincs import, so this runs fine under
gfm_python_preprocessing (unlike postprocess_tile_summary.py, which needs
the separate, currently-matplotlib-broken hydromt-sfincs-dev env only for
SFINCS's own boundary-cell mask - not needed here).

Usage:
    python compute_friction_sweep_metrics.py --base-dir-name validation_sfincs_v5 \\
        --tile-ids-file validation_sfincs_v5/friction_sweep_tile_ids.txt \\
        --friction-scale-factors 3 6 9 12 15 18 21 24 27 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import WET_THRESHOLD_M, confusion_counts, metrics_from_counts  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

LAND_CODE = 0
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767
FRICTION_SCALE_FACTOR_DEFAULT = 30.0  # matches run_eikonal_on_sfincs_subgrid.py


def _decode_waterdepth_cm(path: Path) -> tuple[np.ndarray, object, object, tuple]:
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
        transform, crs, shape = src.transform, src.crs, src.shape
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m, transform, crs, shape


def _eikonal_path(tile_dir: Path, friction_scale_factor: float) -> Path:
    tag = "" if friction_scale_factor == FRICTION_SCALE_FACTOR_DEFAULT else f"_fsf{friction_scale_factor:g}"
    return tile_dir / "outputs" / f"eikonal_on_subgrid_waterdepth_RP100_SLR_0{tag}.tif"


def tile_counts(tile_dir: Path, friction_scale_factor: float) -> tuple[float, float, float] | None:
    """(matched_km2, eikonal_only_km2, sfincs_only_km2) for one tile at one
    friction scale, or None if either raster is missing."""
    eikonal_path = _eikonal_path(tile_dir, friction_scale_factor)
    hmax_path = tile_dir / "sfincs_model" / "hmax_subgrid.tif"
    native_mask_path = tile_dir / "inputs" / "mask.tif"
    if not (eikonal_path.exists() and hmax_path.exists() and native_mask_path.exists()):
        return None

    depth_m, transform, crs, shape = _decode_waterdepth_cm(eikonal_path)

    with retry_transient_io(rasterio.open, hmax_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sg_nodata, sg_shape = src.nodata, src.shape
    if sg_nodata is not None and not np.isnan(sg_nodata):
        sfincs_depth = np.where(sfincs_depth == sg_nodata, np.nan, sfincs_depth)
    if shape != sg_shape:
        return None  # not pixel-identical (stale/mismatched raster) - skip rather than misrepresent

    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    land = mog == LAND_CODE

    eikonal_wet = land & np.isfinite(depth_m) & (depth_m > WET_THRESHOLD_M)
    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)

    cell_km2 = abs(transform.a) * abs(transform.e) / 1e6
    weight = np.full(shape, cell_km2, dtype=np.float64)
    return confusion_counts(eikonal_wet, sfincs_wet, land, weight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v5")
    parser.add_argument("--tile-ids-file", required=True, help="one tile_id per line (select_friction_sweep_tiles.py)")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", required=True)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids = [int(line) for line in Path(args.tile_ids_file).read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) in sweep tile set")

    per_tile_rows = []
    pooled_rows = []
    for fsf in args.friction_scale_factors:
        matched_sum = only_sum = sfincs_only_sum = 0.0
        n_scored = 0
        for tid in tile_ids:
            counts = tile_counts(base_dir / str(tid), fsf)
            if counts is None:
                continue
            matched, only, sfincs_only = counts
            per_tile_rows.append({
                "tile_id": tid, "friction_scale_factor": fsf,
                "matched_km2": matched, "eikonal_only_km2": only, "sfincs_only_km2": sfincs_only,
                **metrics_from_counts(matched, only, sfincs_only),
            })
            matched_sum += matched
            only_sum += only
            sfincs_only_sum += sfincs_only
            n_scored += 1

        pooled = metrics_from_counts(matched_sum, only_sum, sfincs_only_sum)
        pooled_rows.append({
            "friction_scale_factor": fsf, "fraction_of_current": fsf / FRICTION_SCALE_FACTOR_DEFAULT,
            "n_tiles_scored": n_scored, "matched_km2": matched_sum,
            "eikonal_only_km2": only_sum, "sfincs_only_km2": sfincs_only_sum,
            **pooled,
        })
        print(f"friction_scale_factor={fsf:g} ({fsf/FRICTION_SCALE_FACTOR_DEFAULT:.1f}x current): "
              f"n_scored={n_scored}, CSI={pooled['CSI']:.4f}, bias={pooled['bias']:.4f}, "
              f"HT={pooled['HT']:.4f}, FAR={pooled['FAR']:.4f}")

    per_tile_df = pd.DataFrame(per_tile_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    per_tile_out = base_dir / "friction_sweep_per_tile_metrics.csv"
    pooled_out = base_dir / "friction_sweep_pooled_metrics.csv"
    per_tile_df.to_csv(per_tile_out, index=False)
    pooled_df.to_csv(pooled_out, index=False)
    print(f"\nWrote {per_tile_out} ({len(per_tile_df)} row(s))")
    print(f"Wrote {pooled_out} ({len(pooled_df)} row(s))")

    pooled_df = pooled_df.assign(abs_bias_minus_1=np.abs(pooled_df["bias"] - 1.0))
    best_csi = pooled_df.loc[pooled_df["CSI"].idxmax()]
    best_bias = pooled_df.loc[pooled_df["abs_bias_minus_1"].idxmin()]
    print(f"\nBest CSI: friction_scale_factor={best_csi['friction_scale_factor']:g} "
          f"({best_csi['fraction_of_current']:.1f}x current), CSI={best_csi['CSI']:.4f}, bias={best_csi['bias']:.4f}")
    print(f"Best |bias-1|: friction_scale_factor={best_bias['friction_scale_factor']:g} "
          f"({best_bias['fraction_of_current']:.1f}x current), CSI={best_bias['CSI']:.4f}, bias={best_bias['bias']:.4f}")


if __name__ == "__main__":
    main()
