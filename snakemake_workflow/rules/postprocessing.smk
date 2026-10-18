"""Rules for merging per-tile flood results into spatial chunks, a VRT mosaic and plots.

Chunks (merge.chunk_size_deg) are merged independently so each job reads only
the tiles overlapping it.
"""

# chunk_id's constraint lives in common.smk: Snakemake merges all
# wildcard_constraints globally, so redeclaring here would shadow those.
_merge_chunk_waterdepth_path = os.path.join(
    config["postprocessing"]["merged_outputs"], "chunks",
    "waterdepth_{chunk_id}_{return_period}_{waterlevel_name}.tif",
)
_merge_chunk_provenance_path = os.path.join(
    config["postprocessing"]["merged_outputs"], "chunks",
    "provenance_{chunk_id}_{return_period}_{waterlevel_name}.tif",
)


rule merge_chunk:
    """Merge per-tile water depths within one chunk for one RP and SLR scenario.

    The fine waterdepth is not temp(): the VRT only references chunk files, and
    validation/validate_country.py and validation/run_multi_rp_summary.py read
    these files directly after the pipeline finishes.
    """
    input:
        waterdepth_tiles=waterdepth_tiles_for_chunk,
    output:
        waterdepth=_merge_chunk_waterdepth_path,
        # Per-cell winning tile_id (int32) of the max-combine; debugging handle.
        provenance=_merge_chunk_provenance_path,
    params:
        chunk_bounds=lambda wildcards: _chunk_bounds_dict[wildcards.chunk_id],
        pp_cfg=config["postprocessing"],
        raster_config=config["raster_format"],
    script:
        "../scripts/merge_chunk.py"


rule compute_flood_fraction_chunk:
    """Compute the coarse flood-fraction raster for one (chunk, RP, SLR) scenario.

    Average-pools (depth > threshold) per fine pixel to the population raster's
    native ~1 km grid; values 0-1. Replaces the fine waterdepth for exposure.
    """
    input:
        waterdepth=rules.merge_chunk.output.waterdepth,
        population=lambda wildcards: expand(
            rules.prepare_exposure_grid_chunk.output.population,
            chunk_id=[wildcards.chunk_id],
        )[0],
    output:
        flood_fraction=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks", "flood_fraction",
            "flood_fraction_{chunk_id}_{return_period}_{waterlevel_name}.tif",
        ),
    params:
        threshold_m=config["exposure"]["exceedance_threshold_m"],
        block_size=config["postprocessing"]["block_size"],
    script:
        "../scripts/compute_flood_fraction_chunk.py"


rule build_mosaic_vrt:
    """Assemble all chunk outputs into a GDAL VRT mosaic for one RP and SLR scenario."""
    input:
        waterdepth=expand(
            rules.merge_chunk.output.waterdepth,
            chunk_id=CHUNK_IDS, allow_missing=True,
        ),
    output:
        waterdepth_vrt=os.path.join(
            config["postprocessing"]["merged_outputs"], "waterdepth_{return_period}_{waterlevel_name}.vrt",
        ),
    script:
        "../scripts/build_mosaic_vrt.py"


rule plot_merged_results:
    """Plot the VRT-mosaicked water-depth for one return period and SLR scenario."""
    input:
        waterdepth=rules.build_mosaic_vrt.output.waterdepth_vrt,
    output:
        waterdepth_plot=os.path.join(
            config["postprocessing"]["merged_outputs"], "plots",
            "waterdepth_{return_period}_{waterlevel_name}.png",
        ),
    params:
        pp_cfg=config["postprocessing"],
        threshold_m=config["exposure"]["exceedance_threshold_m"],
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        model_outputs=config["simulation"]["model_outputs"],
        tile_grid_path=config["tile_grid"]["path"],
    script:
        "../scripts/plot_merged_results.py"



rule prepare_exposure_grid_chunk:
    """Resolve and cache the population raster and geogunit IDs once per chunk.

    Geogunit IDs are on population's grid (nearest-neighbour). Both depend only
    on chunk bounds, so this avoids 100+ redundant fetches per chunk.
    """
    input:
        # Any (RP, SLR) works: only the chunk grid metadata is used.
        reference=lambda wildcards: expand(
            rules.merge_chunk.output.waterdepth,
            chunk_id=[wildcards.chunk_id],
            return_period=[RETURN_PERIODS[0]],
            waterlevel_name=[_PROTECTION_BASELINE_SLR],
        )[0],
    output:
        population=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks",
            "exposure_population_grid_{chunk_id}.tif",
        ),
        geogunit=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks",
            "exposure_geogunit_grid_{chunk_id}.tif",
        ),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        # Catalog keys: data_catalog_gfm.yml holds dataset identifiers.
        population_source="population",
        geogunit_source="geogunit_protection_units",
    script:
        "../scripts/prepare_exposure_grid_chunk.py"
