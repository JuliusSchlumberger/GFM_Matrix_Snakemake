"""Per-tile flood-extent agreement counts, comparing SFINCS against bathtub
and SFINCS against eikonal-on-subgrid.

Per tile, only three raw km2 counts are written per comparison: cells where
model and SFINCS agree on wet ("matched"), cells where the model alone
claims flood ("_only"), and cells where SFINCS alone claims flood
("_sfincs_only"). No per-tile ratio (HT/FAR/CSI/bias) is computed or
written - a ratio computed from one tile's own small counts and then
averaged across tiles would let small tiles with little flooding dominate
the average just as much as large, heavily-flooded tiles. HT/FAR/CSI/bias
are only ever computed once, in `pooled_metrics`, from counts SUMMED across
every tile - the statistically correct way to combine contingency-table
metrics across regions of very different size.

SFINCS is treated as the reference ("benchmark") since it's the
hydrodynamic solver; bathtub/eikonal are the "model" being scored against
it. All three are read directly on the SFINCS subgrid UTM grid (hmax_subgrid.tif,
bathtub_waterdepth_*.tif, eikonal_on_subgrid_waterdepth_*.tif) - confirmed
pixel-identical (same shape/transform/crs) for every tile, since all three
ultimately derive from the same sfincs_model/subgrid/dep_subgrid.tif, so no
reprojection is needed between them (only the native mask.tif needs
reprojecting onto that grid, same as postprocess_tile_summary.py's own
land-domain masking).

Wet-cell threshold is a uniform WET_THRESHOLD_M applied to all three depths,
not the >0m convention summary.json's own area stats use - hmax_subgrid.tif
already has hydromt_sfincs's own hmin=0.05 baked in (see run_sfincs_tile.py's
downscale_floodmap call), so bathtub/eikonal are thresholded the same way
here for a fair, consistently-defined comparison. This is a separate,
categorical-comparison-specific choice; summary.json's own depth/area stats
are untouched.

Usage:
    python compute_calibration_metrics.py
    python compute_calibration_metrics.py --base-dir-name validation_sfincs_v2
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
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

LAND_CODE = 0
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767
WET_THRESHOLD_M = 0.05


def _decode_waterdepth_cm(path: Path) -> np.ndarray:
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m


def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted (matched, model_only, sfincs_only) km2 - same tp/fp/fn
    convention as src/validation.py's own confusion_counts (matched = tp =
    model wet AND benchmark wet, model_only = fp = model wet AND NOT
    benchmark wet, sfincs_only = fn = NOT model wet AND benchmark wet),
    renamed here for direct readability in the per-tile CSV. tn (neither
    wet) is never needed - HT/FAR/CSI/bias don't use it."""
    d = domain_mask
    matched = float(weight[d & model_wet & benchmark_wet].sum())
    model_only = float(weight[d & model_wet & ~benchmark_wet].sum())
    sfincs_only = float(weight[d & ~model_wet & benchmark_wet].sum())
    return matched, model_only, sfincs_only


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(matched: float, model_only: float, sfincs_only: float) -> dict[str, float]:
    """HT, FAR, CSI, bias - same formulas as src/validation.py's
    metrics_from_counts (HT here == that module's "HR", same hit-rate
    definition, renamed to match this comparison's own terminology). Only
    ever called once, in `pooled_metrics`, on counts already summed across
    every tile - never per-tile, see module docstring."""
    tp, fp, fn = matched, model_only, sfincs_only
    return {
        "HT": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }


def compute_tile_metrics(tile_id: str, root: Path, base_dir_name: str) -> dict | None:
    tile_dir = root / base_dir_name / tile_id
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    native_mask_path = tile_dir / "inputs" / "mask.tif"

    hmax_subgrid_path = sfincs_dir / "hmax_subgrid.tif"
    bathtub_path = out_dir / "bathtub_waterdepth_RP100_SLR_0.tif"
    eikonal_path = out_dir / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"

    if not (hmax_subgrid_path.exists() and bathtub_path.exists()):
        return None

    with retry_transient_io(rasterio.open, hmax_subgrid_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sfincs_nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = src.shape
    if sfincs_nodata is not None and not np.isnan(sfincs_nodata):
        sfincs_depth = np.where(sfincs_depth == sfincs_nodata, np.nan, sfincs_depth)

    with retry_transient_io(rasterio.open, native_mask_path) as src:
        mog = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    land = mog == LAND_CODE

    cell_km2 = abs(transform.a) * abs(transform.e) / 1e6
    weight = np.full(shape, cell_km2, dtype=np.float64)

    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)

    row: dict = {"tile_id": tile_id, "domain_land_km2": float(weight[land].sum())}

    bathtub_depth = _decode_waterdepth_cm(bathtub_path)
    bathtub_wet = land & np.isfinite(bathtub_depth) & (bathtub_depth > WET_THRESHOLD_M)
    matched, model_only, sfincs_only = confusion_counts(bathtub_wet, sfincs_wet, land, weight)
    row.update({
        "bathtub_matched_km2": matched, "bathtub_only_km2": model_only, "bathtub_sfincs_only_km2": sfincs_only,
    })

    if eikonal_path.exists():
        eikonal_depth = _decode_waterdepth_cm(eikonal_path)
        eikonal_wet = land & np.isfinite(eikonal_depth) & (eikonal_depth > WET_THRESHOLD_M)
        matched, model_only, sfincs_only = confusion_counts(eikonal_wet, sfincs_wet, land, weight)
        row.update({
            "eikonal_matched_km2": matched, "eikonal_only_km2": model_only, "eikonal_sfincs_only_km2": sfincs_only,
        })
    else:
        row.update({"eikonal_matched_km2": None, "eikonal_only_km2": None, "eikonal_sfincs_only_km2": None})

    return row


def pooled_metrics(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    matched = df[f"{prefix}_matched_km2"].sum()
    model_only = df[f"{prefix}_only_km2"].sum()
    sfincs_only = df[f"{prefix}_sfincs_only_km2"].sum()
    return metrics_from_counts(matched, model_only, sfincs_only)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    tile_ids = sorted(
        {p.parent.parent.name for p in base_dir.glob("*/sfincs_model/hmax_subgrid.tif")},
        key=lambda t: int(t),
    )
    print(f"{len(tile_ids)} tile(s) with a SFINCS hmax_subgrid.tif found under {base_dir}")

    rows = []
    skipped = []
    for tile_id in tile_ids:
        row = compute_tile_metrics(tile_id, root, args.base_dir_name)
        if row is None:
            skipped.append(tile_id)
        else:
            rows.append(row)

    if not rows:
        print("No tiles had both hmax_subgrid.tif and bathtub_waterdepth_*.tif - nothing to compute.")
        return

    df = pd.DataFrame(rows)
    out_path = base_dir / "calibration_metrics_per_tile.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} tile(s), {len(skipped)} skipped - missing bathtub output)")

    for prefix in ("bathtub", "eikonal"):
        n_valid = df[f"{prefix}_matched_km2"].notna().sum()
        pooled = pooled_metrics(df.dropna(subset=[f"{prefix}_matched_km2"]), prefix)
        print(f"\nSFINCS vs {prefix} - pooled across {n_valid} tile(s):")
        for k, v in pooled.items():
            print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: NaN")


if __name__ == "__main__":
    main()
