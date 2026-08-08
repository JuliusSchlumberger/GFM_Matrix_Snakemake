"""Rules for combining and visualising per-tile flood model results.

The study area is partitioned into spatial chunks (size: merge.chunk_size_deg).
Each chunk is merged independently (rule merge_chunk), then all chunks are
assembled into a GDAL VRT mosaic (rule build_mosaic_vrt) which is used for
the final plots (rule plot_merged_results).  This partitioning means each
merge job reads only the handful of tiles that overlap its chunk rather than
the full tile set, and produces smaller, faster-to-write output files.
"""

# Wildcard constraint so Snakemake does not try to match chunk_id against
# other wildcard patterns (e.g. waterlevel_name, return_period).
wildcard_constraints:
    chunk_id        = r"[NS]\d{2}[EW]\d{3}",
    waterlevel_name = r"SLR_\d+",
    return_period   = r"RP\d+",

# Checked directly here (not via the root Snakefile's own _plotting_enabled,
# which is only defined AFTER this file's `include:` line) since
# merge_chunk's own temp() marking below depends on it - see that rule's
# docstring for why.
_keep_merged_chunk_waterdepth = config["postprocessing"]["plots"]["enabled"]

_merge_chunk_waterdepth_path = os.path.join(
    config["postprocessing"]["merged_outputs"], "chunks",
    "waterdepth_{chunk_id}_{return_period}_{waterlevel_name}.tif",
)
_merge_chunk_provenance_path = os.path.join(
    config["postprocessing"]["merged_outputs"], "chunks",
    "provenance_{chunk_id}_{return_period}_{waterlevel_name}.tif",
)


rule merge_chunk:
    """Merge per-tile water depths within one spatial chunk for one return period and SLR scenario.

    The fine-resolution waterdepth output is marked temp() - auto-deleted by
    Snakemake once its declared consumers have run - ONLY when
    postprocessing.plots.enabled is false. When plots ARE enabled (the
    default), it must stay on disk: build_mosaic_vrt's output VRT is a
    lightweight XML reference to chunk file paths, not a data copy, so
    plot_merged_results (which reads real pixel data through that VRT - e.g.
    compute_flood_area_km2's windowed reads) needs the underlying chunk file
    to still exist when it runs.
    """
    input:
        waterdepth_tiles=waterdepth_tiles_for_chunk,
    output:
        waterdepth=(
            _merge_chunk_waterdepth_path if _keep_merged_chunk_waterdepth
            else temp(_merge_chunk_waterdepth_path)
        ),
        # Per-cell winning tile_id (int32) - the max-combine (2026-08,
        # replacing the previous valid-count-weighted mean) has no "average"
        # to inspect, so this is the practical debugging handle when a
        # merged value looks wrong. Same temp()-or-not lifetime as
        # waterdepth, since it's only useful alongside it.
        provenance=(
            _merge_chunk_provenance_path if _keep_merged_chunk_waterdepth
            else temp(_merge_chunk_provenance_path)
        ),
        overlap_minmax=os.path.join(
            config["postprocessing"]["merged_outputs"], "chunks", "overlap_samples",
            "overlap_minmax_{chunk_id}_{return_period}_{waterlevel_name}.npz",
        ),
    params:
        chunk_bounds=lambda wildcards: _chunk_bounds_dict[wildcards.chunk_id],
        pp_cfg=config["postprocessing"],
        raster_config=config["raster_format"],
    script:
        "../scripts/merge_chunk.py"


rule compute_flood_fraction_chunk:
    """Compute the coarse flood-fraction raster for one (chunk, RP, SLR) scenario.

    Reads the fine waterdepth from merge_chunk, evaluates depth > threshold per
    fine Aqueduct pixel, and average-pools the binary result to the population
    raster's native ~1 km resolution.

    Output: a tiny coarse raster (values 0–1) that replaces the large fine
    waterdepth for all downstream exposure analysis. The fine waterdepth is
    only deleted by Snakemake once this rule completes when
    postprocessing.plots.enabled is false (see merge_chunk's own docstring
    above) - otherwise it stays on disk for build_mosaic_vrt/
    plot_merged_results to read later.
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
    """Assemble all chunk outputs into a GDAL VRT mosaic for one return period and SLR scenario.

    The VRT is a lightweight XML file; no data is copied.  Downstream rules
    (plot_merged_results) open it with rasterio exactly like a regular raster.
    """
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
    """Resolve and cache the population raster and per-pixel geogunit IDs (on population's grid), once per chunk.

    Both depend only on this chunk's bounds, not on return_period,
    waterlevel_name or adaptation_scenario, so resolving them here - instead
    of inside rule compute_exposure_chunk, which runs 100+ times per chunk -
    avoids that many redundant data-catalog fetches and nearest-neighbour
    geogunit reprojections (exposure.prepare_exposure_grid_chunk).
    """
    input:
        # Use a fixed (RP, SLR) merge output just for the chunk bbox/grid metadata.
        # Any (RP, SLR) works since chunk extent depends only on chunk_id.
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
        # Catalog keys, not config values - data_catalog_gfm.yml is the
        # single place dataset identifiers live (see config.yml's header).
        population_source="population",
        geogunit_source="geogunit_protection_units",
    script:
        "../scripts/prepare_exposure_grid_chunk.py"



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
            "overlap_diagnostics_{return_period}_{waterlevel_name}",
        )),
    params:
        pp_cfg=config["postprocessing"],
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
    script:
        "../scripts/plot_overlap_diagnostics.py"


rule plot_overlap_continent_diagnostics:
    """Per-continent overlap-agreement diagnostics for one return period and SLR scenario.

    Pools every chunk's reservoir-sampled per-cell (min, max) depth across
    overlapping tiles (merge_chunk.output.overlap_minmax), groups chunks by
    continent (Natural Earth naturalearth_lowres), and writes one two-subplot
    PNG per continent: a min/max hexbin with Pearson r, and a pie chart
    classifying cells as confirmed-flood / confirmed-no-flood / ambiguous.
    """
    input:
        overlap_files=expand(
            rules.merge_chunk.output.overlap_minmax,
            chunk_id=CHUNK_IDS, allow_missing=True,
        ),
    output:
        diagnostics=directory(os.path.join(
            config["postprocessing"]["merged_outputs"], "plots",
            "overlap_diagnostics_continents_{return_period}_{waterlevel_name}",
        )),
    params:
        pp_cfg=config["postprocessing"],
        threshold_m=config["exposure"]["exceedance_threshold_m"],
    script:
        "../scripts/plot_overlap_continent_diagnostics.py"
