"""Merge per-tile water depth rasters within one spatial chunk for a single SLR scenario."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge import merge_tile_rasters_chunk  # noqa: E402
from plotting import plot_overlap_correlation  # noqa: E402

pp_cfg = snakemake.config["postprocessing"]  # noqa: F821
plot_cfg = pp_cfg["plots"]
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
chunk_bounds = tuple(snakemake.params.chunk_bounds)  # noqa: F821  (minx, miny, maxx, maxy)

# merge_tile_rasters_chunk expects a single raster_config dict combining the
# output format options with the merge-specific thresholds.
raster_config = {
    **pp_cfg["output_raster"],
    "flood_area_threshold_m": pp_cfg["flood_area_threshold_m"],
    "overlap_corr_max_samples": pp_cfg["overlap_corr_max_samples"],
}

overlap_pairs = merge_tile_rasters_chunk(
    tile_rasters=list(snakemake.input.waterdepth_tiles),  # noqa: F821
    chunk_bounds=chunk_bounds,
    count_output_path=snakemake.output.flood_count,  # noqa: F821
    waterdepth_output_path=snakemake.output.waterdepth,  # noqa: F821
    block_size=pp_cfg["block_size"],
    raster_config=raster_config,
)

# Correlation plot: only generate for the one designated scenario.
# For all other scenarios the output is touched (empty file) so Snakemake's
# output tracking is satisfied without writing a real plot.
corr_path = Path(snakemake.output.overlap_correlation_plot)  # noqa: F821
if plot_cfg["enabled"] and waterlevel_name == plot_cfg["correlation_scenario"]:
    plot_overlap_correlation(
        overlap_pairs=overlap_pairs,
        output_path=corr_path,
        waterlevel_name=waterlevel_name,
    )
else:
    corr_path.touch()
