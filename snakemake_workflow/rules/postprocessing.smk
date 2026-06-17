"""Rules for combining and visualising per-tile flood model results.

The study area is partitioned into spatial chunks (size: merge.chunk_size_deg).
Each chunk is merged independently (rule merge_chunk), then all chunks are
assembled into a GDAL VRT mosaic (rule build_mosaic_vrt) which is used for
the final plots (rule plot_merged_results).  This partitioning means each
merge job reads only the handful of tiles that overlap its chunk rather than
the full tile set, and produces smaller, faster-to-write output files.
"""

# Wildcard constraint so Snakemake does not try to match chunk_id against
# other wildcard patterns (e.g. waterlevel_name).
wildcard_constraints:
    chunk_id        = r"[NS]\d{2}[EW]\d{3}",
    waterlevel_name = r"SLR_\d+",


rule merge_chunk:
    """Merge per-tile water depths within one spatial chunk for one SLR scenario.

    All tiles that overlap the chunk are loaded into memory once (one read per
    tile).  The block loop then uses NumPy slicing with no further disk I/O
    until the output blocks are written.  A cross-tile correlation plot is
    also produced for cells where multiple tiles report flooding.
    """
    input:
        waterdepth_tiles=waterdepth_tiles_for_chunk,
    output:
        flood_count=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks",
            "flood_count_{chunk_id}_{waterlevel_name}.tif",
        ),
        waterdepth=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks",
            "waterdepth_{chunk_id}_{waterlevel_name}.tif",
        ),
        overlap_correlation_plot=os.path.join(
            config["postprocessing"]["merged_outputs"], "plots", "correlation",
            "overlap_correlation_{chunk_id}_{waterlevel_name}.png",
        ),
    params:
        chunk_bounds=lambda wildcards: _chunk_bounds_dict[wildcards.chunk_id],
    script:
        "../scripts/merge_chunk.py"


rule build_mosaic_vrt:
    """Assemble all chunk outputs into a GDAL VRT mosaic for one SLR scenario.

    The VRT is a lightweight XML file; no data is copied.  Downstream rules
    (plot_merged_results) open it with rasterio exactly like a regular raster.
    """
    input:
        flood_count=expand(
            rules.merge_chunk.output.flood_count,
            chunk_id=CHUNK_IDS, allow_missing=True,
        ),
        waterdepth=expand(
            rules.merge_chunk.output.waterdepth,
            chunk_id=CHUNK_IDS, allow_missing=True,
        ),
    output:
        flood_count_vrt=os.path.join(
            config["postprocessing"]["merged_outputs"], "flood_count_{waterlevel_name}.vrt",
        ),
        waterdepth_vrt=os.path.join(
            config["postprocessing"]["merged_outputs"], "waterdepth_{waterlevel_name}.vrt",
        ),
    script:
        "../scripts/build_mosaic_vrt.py"


rule plot_merged_results:
    """Plot the VRT-mosaicked flood-count and water-depth for one SLR scenario."""
    input:
        flood_count=rules.build_mosaic_vrt.output.flood_count_vrt,
        waterdepth=rules.build_mosaic_vrt.output.waterdepth_vrt,
    output:
        flood_count_plot=os.path.join(
            config["postprocessing"]["merged_outputs"], "plots",
            "flood_count_{waterlevel_name}.png",
        ),
        waterdepth_plot=os.path.join(
            config["postprocessing"]["merged_outputs"], "plots",
            "waterdepth_{waterlevel_name}.png",
        ),
    script:
        "../scripts/plot_merged_results.py"


rule plot_overlap_diagnostics:
    """Diagnostic figures comparing flood depths across tile overlap zones.

    For each of up to 6 focal tiles that overlap at least one other tile, one
    PNG is written to the output directory showing:
      - Whitesmoke land-polygon background.
      - Grey: cells where exactly one tile reports flooding (unique extent).
      - Red shades: cells where multiple tiles report flooding, coloured by the
        maximum depth difference.
      - Coloured rectangle outlines: each tile's model domain bbox.
    """
    input:
        waterdepth_tiles=expand(
            rules.run_aqueduct.output.waterdepth,
            tile_id=TILE_IDS, allow_missing=True,
        ),
    output:
        diagnostics=directory(os.path.join(
            config["postprocessing"]["merged_outputs"], "plots",
            "overlap_diagnostics_{waterlevel_name}",
        )),
    script:
        "../scripts/plot_overlap_diagnostics.py"
