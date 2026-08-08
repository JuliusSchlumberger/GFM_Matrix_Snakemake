"""Merge per-tile water depth rasters within one spatial chunk for a single return period and SLR scenario."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge import merge_tile_rasters_chunk  # noqa: E402

pp_cfg = snakemake.params.pp_cfg  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
chunk_bounds = tuple(snakemake.params.chunk_bounds)  # noqa: F821  (minx, miny, maxx, maxy)

# merge_tile_rasters_chunk expects a single raster_config dict combining the
# shared raster_format options with the merge-specific thresholds.
raster_config = {
    **snakemake.params.raster_config,  # noqa: F821
    "overlap_corr_max_samples": pp_cfg["overlap_corr_max_samples"],
    "overlap_corr_seed": pp_cfg["overlap_corr_seed"],
}

mins, maxs, total_overlap_cells = merge_tile_rasters_chunk(
    tile_rasters=list(snakemake.input.waterdepth_tiles),  # noqa: F821
    chunk_bounds=chunk_bounds,
    waterdepth_output_path=snakemake.output.waterdepth,  # noqa: F821
    provenance_output_path=snakemake.output.provenance,  # noqa: F821
    block_size=pp_cfg["block_size"],
    raster_config=raster_config,
)

# Persisted (not plotted here) so plot_overlap_continent_diagnostics.py can
# pool every chunk's samples by continent — see that script and
# merge_tile_rasters_chunk's docstring for why min/max-per-cell (not a
# single tile-pair) is what's collected. total_overlap_cells is the true
# pre-sub-sampling count (>= len(mins); equal to it only if this chunk never
# hit overlap_corr_max_samples) so the pooled continent-level plot can report
# how much of the real overlap population its Pearson r / pie chart actually
# sampled.
np.savez(
    snakemake.output.overlap_minmax,  # noqa: F821
    mins=mins,
    maxs=maxs,
    bounds=np.array(chunk_bounds, dtype="float64"),
    total_overlap_cells=np.array(total_overlap_cells, dtype="int64"),
)
