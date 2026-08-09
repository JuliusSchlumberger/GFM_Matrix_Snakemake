"""Overlap-agreement diagnostic (hit/miss ratio) for one geographic region,
computed directly from that region's own simulated tiles - the region-scoped
counterpart to rules/postprocessing.smk's plot_overlap_continent_diagnostics,
for when only some regions are simulated so far and the global Snakemake
target can't be requested yet (it needs every chunk globally - see rule
postprocess_region's own docstring for why).

Reuses the EXACT same functions the real pipeline uses, just applied to one
region's tiles directly instead of reading pre-built chunk outputs:
  - src/regions.py:assign_regions / src/chunks.py:build_chunk_grid,
    chunk_tile_lookup, safe_chunks_for_region - to find which chunks are
    fully contained within the region (same "safe chunk" concept
    postprocess_region uses, so this never touches a tile outside the
    region even transiently).
  - src/merge.py:merge_tile_rasters_chunk - the actual function that reads
    per-tile waterdepth rasters and collects the per-cell (min, max) depth
    pairs across overlapping tiles (its real merged-raster output is
    written to a throwaway temp file per chunk here - not the
    merged_outputs/ production path - since only the returned mins/maxs
    arrays are used).
  - src/plotting.py:plot_overlap_continent_diagnostics - the actual
    plotting function, including the confirmed-flood/confirmed-no-flood/
    ambiguous ("hit/miss") classification logic - unchanged, not
    reimplemented here.

Usage:
    python snakemake_workflow/tests/plot_overlap_region_diagnostics.py \\
        --region Europe_West --return-period RP100 --waterlevel-name SLR_0 \\
        --output overlap_diagnostics_Europe_West_RP100_SLR_0.png
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from chunks import build_chunk_grid, chunk_tile_lookup, safe_chunks_for_region  # noqa: E402
from config_utils import load_config  # noqa: E402
from merge import merge_tile_rasters_chunk  # noqa: E402
from plotting import plot_overlap_continent_diagnostics  # noqa: E402
from regions import assign_regions  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(Path(__file__).resolve().parents[1] / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--region", required=True, help="region name from list_region_chunks.py, e.g. Europe_West")
    parser.add_argument("--return-period", required=True, help="e.g. RP100")
    parser.add_argument("--waterlevel-name", required=True, help="e.g. SLR_0")
    parser.add_argument("--output", required=True, help="output PNG path")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    tile_gdf = load_tile_grid(cfg["tile_grid"]["path"])

    bc_cfg = cfg["boundary_conditions"]
    n_slr = len(list(dict.fromkeys(bc_cfg["slr_scenarios"] + cfg["adaptation"]["slr_intensities"])))
    n_scenarios_per_tile = len(bc_cfg["return_periods"]) * n_slr

    tile_regions = assign_regions(tile_gdf, n_scenarios_per_tile)
    chunk_grid = build_chunk_grid(tile_gdf, cfg["postprocessing"]["chunk_size_deg"])
    lookup = chunk_tile_lookup(chunk_grid, tile_gdf)
    safe_chunk_ids, partial_chunk_ids = safe_chunks_for_region(lookup, tile_regions, args.region)

    if not safe_chunk_ids:
        raise SystemExit(
            f"No chunks are fully contained within region {args.region!r} - "
            f"check the region name against list_region_chunks.py's output."
        )
    print(
        f"Region {args.region}: {len(safe_chunk_ids)} safe chunk(s), "
        f"{len(partial_chunk_ids)} partial chunk(s) excluded (straddle another region).",
        flush=True,
    )

    chunk_bounds_by_id = dict(zip(chunk_grid["chunk_id"], chunk_grid["bounds"]))
    model_outputs = cfg["simulation"]["model_outputs"]
    scenario = f"{args.return_period}_{args.waterlevel_name}"

    # merge_tile_rasters_chunk expects a single raster_config dict combining
    # the shared raster_format options with the merge-specific thresholds -
    # same construction as scripts/merge_chunk.py's own.
    raster_config = {
        **cfg["raster_format"],
        "overlap_corr_max_samples": cfg["postprocessing"]["overlap_corr_max_samples"],
        "overlap_corr_seed": cfg["postprocessing"]["overlap_corr_seed"],
    }

    all_mins: list[np.ndarray] = []
    all_maxs: list[np.ndarray] = []
    total_overlap_cells = 0
    n_chunks_with_overlap = 0

    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, chunk_id in enumerate(safe_chunk_ids, start=1):
            tile_ids = lookup[chunk_id]
            tile_rasters = [
                f"{model_outputs}/{tid}/results/waterdepth_{scenario}.tif"
                for tid in tile_ids
            ]
            scratch_out = Path(tmpdir) / f"waterdepth_{chunk_id}_{scenario}.tif"
            t_chunk = time.time()
            mins, maxs, chunk_total = merge_tile_rasters_chunk(
                tile_rasters=tile_rasters,
                chunk_bounds=tuple(chunk_bounds_by_id[chunk_id]),
                waterdepth_output_path=scratch_out,
                block_size=cfg["postprocessing"]["block_size"],
                raster_config=raster_config,
            )
            elapsed_chunk = time.time() - t_chunk
            elapsed_total = time.time() - t_start
            eta_s = (elapsed_total / i) * (len(safe_chunk_ids) - i)
            print(
                f"[{i}/{len(safe_chunk_ids)}] chunk {chunk_id} "
                f"({len(tile_ids)} tiles): {len(mins):,} overlap samples "
                f"in {elapsed_chunk:.1f}s (total {elapsed_total:.0f}s, "
                f"ETA {eta_s:.0f}s)",
                flush=True,
            )
            if len(mins) == 0:
                continue
            all_mins.append(mins)
            all_maxs.append(maxs)
            total_overlap_cells += chunk_total
            n_chunks_with_overlap += 1

    mins = np.concatenate(all_mins) if all_mins else np.empty(0, dtype=np.float32)
    maxs = np.concatenate(all_maxs) if all_maxs else np.empty(0, dtype=np.float32)
    print(
        f"Pooled {len(mins):,} sampled overlap cells "
        f"({total_overlap_cells:,} true overlap cells before sub-sampling) "
        f"from {n_chunks_with_overlap} of {len(safe_chunk_ids)} safe chunk(s).",
        flush=True,
    )

    plot_cfg = cfg["postprocessing"]["plots"]
    plot_overlap_continent_diagnostics(
        mins=mins, maxs=maxs,
        threshold_m=cfg["exposure"]["exceedance_threshold_m"],
        output_path=args.output,
        continent_name=args.region,
        waterlevel_name=scenario,
        n_chunks=n_chunks_with_overlap,
        total_overlap_cells=total_overlap_cells,
        pie_colors=plot_cfg["overlap_pie_colors"],
        figsize=tuple(plot_cfg["overlap_continent_figsize"]),
        dpi=plot_cfg["dpi"],
    )
    print(f"Written: {args.output}", flush=True)


if __name__ == "__main__":
    main()
