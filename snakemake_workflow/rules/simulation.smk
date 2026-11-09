"""Rules for running the eikonal flood model."""


rule run_eikonal:
    """Run the flood model for a single tile, return period and SLR scenario.

    `mem_mb` (eikonal_runner.estimate_eikonal_mem_mb) lets Snakemake throttle
    concurrency under `--resources mem_mb=<N>`; leave headroom (~80% of RAM).

    hop_distance == 0 tiles use station/IDW forcing (boundary_conditions.storm_tide_source); hop_distance >= 1 tiles
    are seeded from non-zero cells of strictly-lower-hop overlapping tiles for
    the SAME scenario. With no stations / no flooded seeds, an all-ZERO
    waterdepth is written ("definitely dry") and logged to
    `model_outputs/skipped_tiles/{tile_id}_{return_period}_{waterlevel_name}.txt`.

    On `MemoryError` the tile is marked in `model_outputs/oom_tiles/{tile_id}.txt`
    and an all-`EIKONAL_NODATA` placeholder ("not computed") is written, which
    `merge_results` ignores; other scenarios of that tile then skip the solve.
    """
    input:
        dem=rules.extract_dem.output.dem,
        mask=rules.extract_dem_mask.output.mask,
        friction=rules.compute_friction.output.friction,
        boundaries=rules.extract_boundaries.output.boundaries,
    output:
        waterdepth=os.path.join(
            config["simulation"]["model_outputs"], "{tile_id}", "results",
            "waterdepth_{return_period}_{waterlevel_name}.tif",
        ),
    params:
        model_outputs=config["simulation"]["model_outputs"],
        # Neighbour dem/mask for hop>=1 are preprocessing inputs (shared per
        # calibration group), not results; defaults to model_outputs.
        preprocessing_inputs_dir=config["simulation"].get("preprocessing_inputs_dir") or config["simulation"]["model_outputs"],
        raster_config=config["raster_format"],
        tile_grid_path=config["tile_grid"]["path"],
        # Same values as at tile-generation time (wet/dry classification).
        ocean_code=config["tile_generation"]["ocean_code"],
        river_code=config["tile_generation"]["river_code"],
        n_scenarios_per_tile=len(RETURN_PERIODS) * len(WATERLEVEL_NAMES),
        flooding_config=config["simulation"]["flooding"],
        # This tile's real DEM pixel size (local-input validation, TODO.md); DeltaDTM's
        # grid step for every tile without a dem_source_manifest. run_eikonal_cli.py: tile_pixel_size_m.
        pixel_size_m=lambda wildcards: _dem_source_meta_for_tile(wildcards.tile_id, "pixel_size_m",
                                                                DELTADTM_PIXEL_SIZE_M),
        static_level_inputs=static_level_inputs(config),
        hydrograph_path=str(storm_tide_hydrograph_path(config)),
    resources:
        # Deliberately generous; over-estimating only reduces concurrency.
        mem_mb=lambda wildcards, input: estimate_eikonal_mem_mb(
            input.dem,
            config["simulation"]["eikonal_mem_estimate"],
        ),
    script:
        "../scripts/run_eikonal.py"
