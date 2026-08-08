"""Rules for running the Aqueduct flood model."""


rule run_aqueduct:
    """Run the flood model for a single tile, return period and SLR scenario.

    The output filename includes `{return_period}_{waterlevel_name}` so it is
    clearly linked to the return period and sea level rise scenario used to
    produce it.

    Runs `flood_model.flood_depth_dense` in-process (see `run_aqueduct.py`) -
    validated bit-for-bit identical to the original Aqueduct/Julia reference
    implementation across 26 real tiles, see docs/python_vs_julia_qa.md.

    Multiple tiles run fully concurrently on one machine without issue - the
    only real local constraint is ordinary system memory. This rule's
    `mem_mb` resource (see `aqueduct_runner.estimate_aqueduct_mem_mb`) lets
    Snakemake's own scheduler run many small tiles concurrently while
    throttling around large ones, bounded by `--resources mem_mb=<N>` on
    the command line (`<N>` should leave headroom below total system RAM
    for the OS and other processes, e.g. ~80% of physical RAM).

    Wave-based hinterland forcing (2026-08): `tile_grid_path`'s `hop_distance`
    column (`tile_chunking.compute_run_order`) tells `run_aqueduct.py` which
    of two forcing paths a tile uses. `hop_distance == 0` (wave-0, has its
    own real ocean edge): the existing COAST-RP/IDW path - if `boundaries`
    has no stations at all (see `boundaries.select_stations_for_tile`),
    the solve is skipped and a real all-ZERO `waterdepth` is written directly
    (a confidently-computed "definitely dry" result, not an unknown - see
    `aqueduct_runner.write_zero_waterdepth` for why this differs from the
    OOM case below). `hop_distance >= 1` (hinterland, no ocean edge of its
    own): seeded directly from non-zero cells of already-simulated,
    strictly-lower-`hop_distance` overlapping tile(s)' own output for the
    SAME scenario (`boundaries.collect_neighbor_wave_seeds`) - if none are
    available yet or none show any flooding in the overlap, same real-zero
    fallback. Either skip case is logged to `model_outputs/skipped_tiles/
    {tile_id}_{return_period}_{waterlevel_name}.txt` for later mapping.

    If the solve raises `MemoryError` - a rare, tile-size-driven failure, or
    from `mem_mb` under-estimating concurrent memory pressure - the tile is
    marked in `model_outputs/oom_tiles/{tile_id}.txt` and this job's output
    falls back to an all-`AQUEDUCT_NODATA` placeholder instead - a genuine
    unknown ("not computed", unlike the confidently-zero fallbacks above -
    still logged to `skipped_tiles/` the same way) so `merge_results` ignores
    this tile rather than treating it as a real dry result. Once marked, all other
    `(return_period, waterlevel_name)` combinations for this `tile_id` skip
    the solve entirely and do the same, instead of also running out of memory.

    Each job also checks whether its own tile_id's output count just reached
    `n_scenarios_per_tile` (i.e. this was the last scenario for that tile);
    if so it scans model_outputs/ once and prints a running tally of tiles
    simulated/OOM'd/no-stations/still-running across the whole tile grid
    (see aqueduct_runner.print_simulation_progress) - a milestone per tile,
    not per job, to keep the O(n_tiles) scan infrequent.

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
        raster_config=config["raster_format"],
        tile_grid_path=config["tile_grid"]["path"],
        # Reused from tile_generation.* (2026-08 - same values already used at
        # tile-generation time, e.g. for wet/dry classification) rather than a
        # near-duplicate simulation-scoped config surface - see
        # flood_model.coastline_mask / run_aqueduct.py's hop_distance branching.
        ocean_code=config["tile_generation"]["ocean_code"],
        river_code=config["tile_generation"]["river_code"],
        n_scenarios_per_tile=len(RETURN_PERIODS) * len(WATERLEVEL_NAMES),
        flooding_config=config["simulation"]["flooding"],
    resources:
        # A deliberately generous upper bound (see aqueduct_mem_estimate in
        # config.yml) - a concurrency-throttling estimate, not a correctness
        # concern; only effect of over-estimating is somewhat fewer
        # concurrent jobs than strictly necessary.
        mem_mb=lambda wildcards, input: estimate_aqueduct_mem_mb(
            input.dem, config["simulation"]["aqueduct_mem_estimate"]
        ),
    script:
        "../scripts/run_aqueduct.py"
