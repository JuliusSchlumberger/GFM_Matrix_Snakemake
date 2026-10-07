"""Per-tile eikonal-sweep-vs-SFINCS (and bathtub-vs-SFINCS) comparison
cache: one small JSON file per tile
(`{base_dir_name}/{tile_id}/outputs/sweep_comparison_cache.json`), built by
opening SFINCS's own hmax_subgrid.tif + native mask ONCE and comparing it
against every friction_scale_factor's eikonal raster AND the bathtub
raster in that same pass - not once per (tile, fsf) pair every time
someone wants updated pooled metrics (compute_friction_sweep_metrics.py
used to re-open SFINCS/mask per fsf per tile per invocation; even after an
earlier tile-outer/fsf-inner optimization it still re-did the comparison
math from raw rasters on every single run).

Meant to be built ONCE PER TILE, right after that tile's full sweep is
done (see run_friction_sweep_batch.py's own end-of-batch call to
build_and_write_cache() for every distinct tile in that batch) - "the
required postprocess which compares the SFINCS map against the sweep(s)
[...] done as part of the main pipeline too, so that the postprocessing
analysis actually is very quick" (2026-10-07). compute_friction_sweep_metrics.py
then just reads these cache files and sums already-computed numbers -
no raster I/O, no recomputation - falling back to building a cache live
(and writing it, so the NEXT read is fast too) only for a tile that
somehow doesn't have one yet (e.g. swept before this cache existed).

Cache file shape:
    {
      "tile_id": "...", "max_outer_iterations": 5, "built": "<timestamp>",
      "bathtub": <comparison dict, see _compare_one()>  | null,
      "points": {"3.0": <comparison dict>, "6.0": <comparison dict>, ...}
    }
Each <comparison dict>: matched_km2, model_only_km2, sfincs_only_km2, HT,
FAR, CSI, bias, model_depth_median_m, depth_joint (n/sum_x/.../hist_fine/
hist_category - same shape flood_agreement.pool_depth_joint's own per-tile
summary_eikonal.json entries use, so plot_validation_results.py's pooling
code can consume either source interchangeably).

Usage:
    python tile_sweep_cache.py --base-dir-name sfincs_calibration --tile-id 2002 --max-outer-iterations 5
    python tile_sweep_cache.py --base-dir-name sfincs_calibration --tile-ids-file sfincs_calibration/tile_ids.txt --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_friction_sweep_metrics import (  # noqa: E402
    MAX_OUTER_ITERATIONS_DEFAULT, _decode_waterdepth_cm, _eikonal_path, load_tile_sfincs,
)
from flood_agreement import (  # noqa: E402
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, WET_THRESHOLD_M, confusion_counts, depth_corr_sufficient_stats,
    depth_joint_hist, metrics_from_counts,
)
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

BATHTUB_FILENAME = "bathtub_waterdepth_RP100_SLR_0.tif"


def cache_path(tile_dir: Path) -> Path:
    return tile_dir / "outputs" / "sweep_comparison_cache.json"


def _compare_one(raster_path: Path, cached: dict) -> dict | None:
    """One raster (eikonal at some fsf, or bathtub) vs SFINCS's own cached
    depth/land/wet/weight - extent agreement + depth-joint stats, same
    primitives postprocess_tile_summary.py itself uses for the default
    point (flood_agreement.py's confusion_counts/depth_joint_hist/
    depth_corr_sufficient_stats)."""
    if not raster_path.exists():
        return None
    depth_m, _, _, shape = _decode_waterdepth_cm(raster_path)
    if shape != cached["shape"]:
        return None  # not pixel-identical (stale/mismatched raster) - skip rather than misrepresent

    land = cached["land"]
    model_wet = land & np.isfinite(depth_m) & (depth_m > WET_THRESHOLD_M)
    sfincs_wet = cached["sfincs_wet"]
    matched, model_only, sfincs_only = confusion_counts(model_wet, sfincs_wet, land, cached["weight"])
    metrics = metrics_from_counts(matched, model_only, sfincs_only)

    both_wet = model_wet & sfincs_wet
    depth_median_m = float(np.median(depth_m[model_wet])) if model_wet.any() else None
    depth_joint = None
    if both_wet.any():
        x = cached["sfincs_depth"][both_wet].astype(np.float64)
        y = depth_m[both_wet].astype(np.float64)
        stats = depth_corr_sufficient_stats(x, y)
        depth_joint = {
            **stats,
            "hist_fine": depth_joint_hist(x, y, DEPTH_CORR_FINE_EDGES).tolist(),
            "hist_category": depth_joint_hist(x, y, DEPTH_CATEGORY_EDGES).tolist(),
        }

    return {
        "matched_km2": matched, "model_only_km2": model_only, "sfincs_only_km2": sfincs_only,
        "model_depth_median_m": depth_median_m, "depth_joint": depth_joint,
        **metrics,
    }


def build_tile_sweep_cache(
    tile_dir: Path, friction_scale_factors: list[float], max_outer_iterations: int,
) -> dict | None:
    """Opens SFINCS hmax_subgrid.tif + mask ONCE (load_tile_sfincs), then
    compares it against the bathtub raster and every fsf's eikonal raster
    in that single pass. None if SFINCS hasn't been built yet for this tile
    (nothing to compare against)."""
    cached = load_tile_sfincs(tile_dir)
    if cached is None:
        return None

    result = {
        "tile_id": tile_dir.name,
        "max_outer_iterations": max_outer_iterations,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bathtub": _compare_one(tile_dir / "outputs" / BATHTUB_FILENAME, cached),
        "points": {},
    }
    for fsf in friction_scale_factors:
        point = _compare_one(_eikonal_path(tile_dir, fsf, max_outer_iterations), cached)
        if point is not None:
            result["points"][f"{fsf:g}"] = point
    return result


def write_cache(tile_dir: Path, cache: dict) -> Path:
    # Transient P:\ drive drops (SMB idle-session timeout/blip) happen on this
    # codebase's shared mount - retry rather than losing a just-finished sweep's
    # own comparison because of a momentary hiccup (see retry_io.py's own docstring).
    p = cache_path(tile_dir)
    retry_transient_io(p.parent.mkdir, parents=True, exist_ok=True)
    retry_transient_io(p.write_text, json.dumps(cache), encoding="utf-8")
    return p


def load_or_build_cache(
    tile_dir: Path, friction_scale_factors: list[float], max_outer_iterations: int, write_if_missing: bool = True,
) -> dict | None:
    """Fast path: read the existing cache, but ONLY if it already covers
    every fsf point being asked for now. Rebuilds (and rewrites) otherwise -
    both for a tile swept before this cache existed at all, AND for a tile
    whose cache was written EARLY, before its own sweep had finished (found
    2026-10-08: run_friction_sweep_batch.py's own end-of-batch cache build
    ran per BATCH, and batches for the same tile's remaining fsf points
    landed later via resume_calibration.py's targeted resubmission - the
    cache from the first batch never got invalidated, so most tiles sat
    with a stale 1-point cache, typically just friction_scale_factor=9
    specifically since that was the point resubmitted first/separately as
    the newly-chosen production default - while every later fsf point's
    real .tif was silently ignored by every subsequent read of this
    "fast path". `points` keys are the ONLY thing checked (not `bathtub` or
    `max_outer_iterations`) - a tile missing just the bathtub raster still
    has every fsf point correctly cached and must not be needlessly rebuilt
    on every call.
    """
    p = cache_path(tile_dir)
    requested = {f"{fsf:g}" for fsf in friction_scale_factors}
    if p.exists():
        try:
            cached = json.loads(retry_transient_io(p.read_text, encoding="utf-8"))
            if requested <= cached.get("points", {}).keys():
                return cached
        except (json.JSONDecodeError, OSError):
            pass  # fall through and rebuild
    cache = build_tile_sweep_cache(tile_dir, friction_scale_factors, max_outer_iterations)
    if cache is not None and write_if_missing:
        write_cache(tile_dir, cache)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--tile-id", default=None, help="build the cache for one tile")
    parser.add_argument("--tile-ids-file", default=None, help="build the cache for every tile listed, one per line")
    parser.add_argument(
        "--friction-scale-factors", type=float, nargs="+",
        default=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
    )
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_DEFAULT)
    args = parser.parse_args()
    if not args.tile_id and not args.tile_ids_file:
        parser.error("pass --tile-id or --tile-ids-file")

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids = [args.tile_id] if args.tile_id else [
        line.strip() for line in Path(args.tile_ids_file).read_text().splitlines() if line.strip()
    ]

    n_built = 0
    for i, tid in enumerate(tile_ids, start=1):
        cache = build_tile_sweep_cache(base_dir / tid, args.friction_scale_factors, args.max_outer_iterations)
        if cache is not None:
            write_cache(base_dir / tid, cache)
            n_built += 1
        if i % 100 == 0 or i == len(tile_ids):
            print(f"  [{i}/{len(tile_ids)}] {n_built} cache(s) built so far", flush=True)

    print(f"Built {n_built}/{len(tile_ids)} tile sweep comparison cache(s)")


if __name__ == "__main__":
    main()
