"""Merge per-tile water depth rasters for a single SLR scenario into combined rasters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge import merge_tile_rasters  # noqa: E402
from plotting import plot_overlap_correlation  # noqa: E402

pp_cfg = snakemake.config["postprocessing"]  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

raster_config = {
    **pp_cfg["output_raster"],
    "flood_area_threshold_m": pp_cfg["flood_area_threshold_m"],
    "overlap_corr_max_samples": pp_cfg["overlap_corr_max_samples"],
}

overlap_pairs = merge_tile_rasters(
    tile_rasters=snakemake.input.waterdepth_tiles,  # noqa: F821
    count_output_path=snakemake.output.flood_count,  # noqa: F821
    waterdepth_output_path=snakemake.output.waterdepth,  # noqa: F821
    block_size=pp_cfg["block_size"],
    raster_config=raster_config,
)

plot_overlap_correlation(
    overlap_pairs=overlap_pairs,
    output_path=snakemake.output.overlap_correlation_plot,  # noqa: F821
    waterlevel_name=waterlevel_name,
)
