"""Merge per-tile water depth rasters within one spatial chunk for a single return period and SLR scenario."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from merge import merge_tile_rasters_chunk  # noqa: E402

pp_cfg = snakemake.params.pp_cfg  # noqa: F821
chunk_bounds = tuple(snakemake.params.chunk_bounds)  # noqa: F821  (minx, miny, maxx, maxy)

merge_tile_rasters_chunk(
    tile_rasters=list(snakemake.input.waterdepth_tiles),  # noqa: F821
    chunk_bounds=chunk_bounds,
    waterdepth_output_path=snakemake.output.waterdepth,  # noqa: F821
    provenance_output_path=snakemake.output.provenance,  # noqa: F821
    block_size=pp_cfg["block_size"],
    raster_config=snakemake.params.raster_config,  # noqa: F821
)
