"""Pools per-tile eikonal-sweep-vs-SFINCS agreement (CSI/bias per
friction_scale_factor) across a fixed tile set (select_friction_sweep_tiles.py)
into one pooled table.

2026-10-07: reads each tile's own sweep_comparison_cache.json
(tile_sweep_cache.py), built ONCE per tile as part of run_friction_sweep_batch.py's
own end-of-batch step (SFINCS opened once, compared against every sweep
point AND bathtub in one pass) - not recomputed from raw rasters here
anymore. A tile missing a cache (swept before this existed) falls back to
building+writing one live via load_or_build_cache(), so this is
self-healing: slow the first time only. `load_tile_sfincs`/
`_decode_waterdepth_cm`/`_eikonal_path` below are the low-level raster
primitives tile_sweep_cache.py's own cache-building imports FROM this
module - kept here, not moved, since this was the original home and
nothing about ownership needed to change, just where the per-tile loop
itself lives.

Needs only rasterio/scipy/numpy - no hydromt_sfincs import, so this runs
fine under gfm_python_preprocessing (unlike postprocess_tile_summary.py,
which needs the separate, currently-matplotlib-broken hydromt-sfincs-dev
env only for SFINCS's own boundary-cell mask - not needed here).

Usage:
    python compute_friction_sweep_metrics.py --base-dir-name validation_sfincs_v5 \\
        --tile-ids-file validation_sfincs_v5/friction_sweep_tile_ids.txt \\
        --friction-scale-factors 3 6 9 12 15 18 21 24 27 30
    # if the sweep itself was run with a non-default --max-outer-iterations (run_eikonal_on_sfincs_subgrid.py),
    # pass the SAME value here too, or every lookup silently misses (see _eikonal_path()):
    python compute_friction_sweep_metrics.py --base-dir-name sfincs_calibration \\
        --tile-ids-file sfincs_calibration/tile_ids.txt \\
        --friction-scale-factors 3 6 9 12 15 18 21 24 27 30 --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import WET_THRESHOLD_M, metrics_from_counts  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

LAND_CODE = 0
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767
FRICTION_SCALE_FACTOR_DEFAULT = 30.0  # matches run_eikonal_on_sfincs_subgrid.py
MAX_OUTER_ITERATIONS_DEFAULT = 4  # matches run_eikonal_on_sfincs_subgrid.py's own default


def _decode_waterdepth_cm(path: Path) -> tuple[np.ndarray, object, object, tuple]:
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
        transform, crs, shape = src.transform, src.crs, src.shape
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m, transform, crs, shape


def _eikonal_path(tile_dir: Path, friction_scale_factor: float, max_outer_iterations: int) -> Path:
    # Same tagging scheme run_eikonal_on_sfincs_subgrid.py itself writes with - both tags
    # independent and composable (a run can differ in friction, outer-iterations, both, or
    # neither), so this must build the path identically or every file lookup here silently
    # misses (tile_counts() below treats a missing file as "skip this tile", not an error).
    fsf_tag = "" if friction_scale_factor == FRICTION_SCALE_FACTOR_DEFAULT else f"_fsf{friction_scale_factor:g}"
    outer_tag = "" if max_outer_iterations == MAX_OUTER_ITERATIONS_DEFAULT else f"_outer{max_outer_iterations}"
    return tile_dir / "outputs" / f"eikonal_on_subgrid_waterdepth_RP100_SLR_0{fsf_tag}{outer_tag}.tif"


def load_tile_sfincs(tile_dir: Path) -> dict | None:
    """Loads this tile's SFINCS hmax_subgrid + native mask ONCE - neither
    depends on friction_scale_factor, so this used to be re-opened and
    re-reprojected once per SWEEP POINT (10x redundant reads of the exact
    same unchanging rasters per tile - confirmed the dominant cost of this
    script's own wall-clock, worse now that the sweep's own HPC batches are
    concurrently writing to the same shared tree, each re-open risking a
    transient-file retry). Call once per tile, reuse the result across every
    fsf point via tile_counts_for_fsf() below."""
    hmax_path = tile_dir / "sfincs_model" / "hmax_subgrid.tif"
    native_mask_path = tile_dir / "inputs" / "mask.tif"
    if not (hmax_path.exists() and native_mask_path.exists()):
        return None

    with retry_transient_io(rasterio.open, hmax_path) as src:
        sfincs_depth = src.read(1).astype(np.float32)
        sg_nodata, transform, crs, shape = src.nodata, src.transform, src.crs, src.shape
    if sg_nodata is not None and not np.isnan(sg_nodata):
        sfincs_depth = np.where(sfincs_depth == sg_nodata, np.nan, sfincs_depth)

    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    land = mog == LAND_CODE
    sfincs_wet = land & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)

    cell_km2 = abs(transform.a) * abs(transform.e) / 1e6
    weight = np.full(shape, cell_km2, dtype=np.float64)
    return {
        "transform": transform, "shape": shape, "land": land, "sfincs_wet": sfincs_wet, "weight": weight,
        "sfincs_depth": sfincs_depth,  # raw depth (not just the wet boolean) - needed for depth-joint pooling
    }


def main() -> None:
    # Deferred, not top-level: tile_sweep_cache.py itself imports load_tile_sfincs/
    # _decode_waterdepth_cm/_eikonal_path/MAX_OUTER_ITERATIONS_DEFAULT FROM this
    # module - a top-level import here would be circular. Safe deferred to inside
    # main(): by the time this runs, this module's own top-level names are already
    # fully defined, so tile_sweep_cache's own import of them succeeds.
    from tile_sweep_cache import cache_path, load_or_build_cache

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v5")
    parser.add_argument("--tile-ids-file", required=True, help="one tile_id per line (select_friction_sweep_tiles.py)")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", required=True)
    parser.add_argument(
        "--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_DEFAULT,
        help=f"must match whatever --max-outer-iterations the eikonal sweep was actually run with "
             f"(default: {MAX_OUTER_ITERATIONS_DEFAULT}, matching run_eikonal_on_sfincs_subgrid.py's "
             f"own default) - determines which tagged output filename this reads, see _eikonal_path().",
    )
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids = [int(line) for line in Path(args.tile_ids_file).read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) in sweep tile set", flush=True)

    per_tile_rows = []
    sums = {fsf: {"matched": 0.0, "only": 0.0, "sfincs_only": 0.0, "n_scored": 0} for fsf in args.friction_scale_factors}

    # Reads each tile's own sweep_comparison_cache.json (tile_sweep_cache.py) -
    # built ONCE per tile as part of run_friction_sweep_batch.py's own
    # end-of-batch step, not recomputed here. A tile missing a cache (swept
    # before that existed) falls back to building+writing one live, so this
    # is self-healing: slow the first time only, fast on every run after
    # (2026-10-07 - this used to re-open SFINCS/mask and redo the full
    # confusion-matrix computation from raw rasters on every single
    # invocation, the dominant cost of the "postprocessing analysis" step).
    t_start = time.time()
    progress_every = 1
    n_tiles_loaded = 0
    n_from_cache = n_built_live = 0
    for i, tid in enumerate(tile_ids, start=1):
        tile_dir = base_dir / str(tid)
        was_cached = cache_path(tile_dir).exists()
        cache = load_or_build_cache(tile_dir, args.friction_scale_factors, args.max_outer_iterations)
        if cache is None:
            if i % progress_every == 0 or i == len(tile_ids):
                elapsed = time.time() - t_start
                print(f"  [{i}/{len(tile_ids)}] tile {tid}: no usable SFINCS hmax/mask, skipped "
                      f"({n_tiles_loaded} usable so far, {elapsed:.0f}s elapsed)", flush=True)
            continue
        n_tiles_loaded += 1
        n_from_cache += was_cached
        n_built_live += not was_cached

        for fsf in args.friction_scale_factors:
            point = cache["points"].get(f"{fsf:g}")
            if point is None:
                continue
            matched, only, sfincs_only = point["matched_km2"], point["model_only_km2"], point["sfincs_only_km2"]
            per_tile_rows.append({
                "tile_id": tid, "friction_scale_factor": fsf,
                "matched_km2": matched, "eikonal_only_km2": only, "sfincs_only_km2": sfincs_only,
                "HT": point["HT"], "FAR": point["FAR"], "CSI": point["CSI"], "bias": point["bias"],
            })
            s = sums[fsf]
            s["matched"] += matched
            s["only"] += only
            s["sfincs_only"] += sfincs_only
            s["n_scored"] += 1

        if i % progress_every == 0 or i == len(tile_ids):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0.0
            eta_s = (len(tile_ids) - i) / rate if rate > 0 else float("nan")
            print(f"  [{i}/{len(tile_ids)}] tile {tid}: scored from {'cache' if was_cached else 'live rebuild'} "
                  f"({n_tiles_loaded} usable so far, {elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining)", flush=True)

    print(f"{n_tiles_loaded}/{len(tile_ids)} tile(s) scored ({n_from_cache} from existing cache, "
          f"{n_built_live} cache(s) built live just now and written for next time)", flush=True)

    pooled_rows = []
    for fsf in args.friction_scale_factors:
        s = sums[fsf]
        pooled = metrics_from_counts(s["matched"], s["only"], s["sfincs_only"])
        pooled_rows.append({
            "friction_scale_factor": fsf, "fraction_of_current": fsf / FRICTION_SCALE_FACTOR_DEFAULT,
            "n_tiles_scored": s["n_scored"], "matched_km2": s["matched"],
            "eikonal_only_km2": s["only"], "sfincs_only_km2": s["sfincs_only"],
            **pooled,
        })
        print(f"friction_scale_factor={fsf:g} ({fsf/FRICTION_SCALE_FACTOR_DEFAULT:.1f}x current): "
              f"n_scored={s['n_scored']}, CSI={pooled['CSI']:.4f}, bias={pooled['bias']:.4f}, "
              f"HT={pooled['HT']:.4f}, FAR={pooled['FAR']:.4f}")

    per_tile_df = pd.DataFrame(per_tile_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    per_tile_out = base_dir / "friction_sweep_per_tile_metrics.csv"
    pooled_out = base_dir / "friction_sweep_pooled_metrics.csv"
    # Transient P:\ drive drops (SMB idle-session timeout/blip) happen on this
    # codebase's shared mount - retry the write rather than losing every
    # tile's already-done raster read over a momentary hiccup at the very end.
    retry_transient_io(per_tile_df.to_csv, per_tile_out, index=False)
    retry_transient_io(pooled_df.to_csv, pooled_out, index=False)
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
