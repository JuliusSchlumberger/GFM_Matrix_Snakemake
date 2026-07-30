"""Rules for running the Aqueduct flood model."""


rule run_aqueduct:
    """Run the flood model for a single tile, return period and SLR scenario.

    The output filename includes `{return_period}_{waterlevel_name}` so it is
    clearly linked to the return period and sea level rise scenario used to
    produce it.

    Which implementation actually runs is `simulation.engine` in config.yml
    ("python" - `flood_model.flood_depth_dense`, validated bit-for-bit
    identical to real Aqueduct output across 26 real tiles, see
    docs/python_vs_julia_qa.md - or "julia" - the original compiled
    `aqueduct_executable`, kept as a fallback/comparison option). The
    skip/OOM-handling and per-tile-progress logic below applies to either
    engine (see `run_aqueduct.py`); notes specific to Aqueduct/Julia's own
    failure mode are labelled as such.

    Aqueduct instances can run fully concurrently on one machine - a
    controlled test (2, 6, then 5 simultaneous instances, up to 140M-pixel
    tiles) never reproduced the `LLVM ERROR: Unable to allocate section
    memory!` JIT crash this rule used to serialize against. The real local
    constraint is ordinary system memory: running several large tiles at
    once can exhaust available RAM (one 140M-pixel tile that succeeded
    cleanly alone failed with a genuine Julia `OutOfMemoryError` only when
    4 other large tiles were running alongside it). This rule's `mem_mb`
    resource (see `aqueduct_runner.estimate_aqueduct_mem_mb`) lets
    Snakemake's own scheduler run many small tiles concurrently while
    throttling around large ones, bounded by `--resources mem_mb=<N>` on
    the command line (`<N>` should leave headroom below total system RAM
    for the OS and other processes, e.g. ~80% of physical RAM).

    If `boundaries` has no stations (the tile has no water level boundary
    points within it - see `boundaries.select_stations_for_tile`), Aqueduct
    is skipped entirely: an all-`AQUEDUCT_NODATA` placeholder is written for
    `waterdepth` (so `merge_results` ignores this tile) and the skip is
    logged to `model_outputs/skipped_tiles/{tile_id}_{return_period}_{waterlevel_name}.txt`
    for later mapping.

    If Aqueduct crashes with `OutOfMemoryError` - whether from the tile's
    own size (the memory cost of `component_indices` in `core/src/core.jl`
    scales with pixel count, not the return period or SLR scenario, so this
    is effectively a per-`tile_id` failure) or from `mem_mb` under-
    estimating concurrent memory pressure - the tile is marked in
    `model_outputs/oom_tiles/{tile_id}.txt` and this job's output falls back
    to the same all-`AQUEDUCT_NODATA` placeholder (logged to `skipped_tiles/`
    as above) instead of failing. Once marked, all other
    `(return_period, waterlevel_name)` combinations for this `tile_id` skip
    Aqueduct entirely and do the same, instead of also running out of memory.

    Each job also checks whether its own tile_id's output count just reached
    `n_scenarios_per_tile` (i.e. this was the last scenario for that tile);
    if so it scans model_outputs/ once and prints a running tally of tiles
    simulated/OOM'd/no-stations/still-running across the whole tile grid
    (see aqueduct_runner.print_simulation_progress) - a milestone per tile,
    not per job, to keep the O(n_tiles) scan infrequent.

    `dem`/`mask`/`friction` come from crop_flood_extent, not directly from
    extract_dem/extract_dem_mask/compute_friction - when
    simulation.flood_extent_crop.enabled is true this is a strictly smaller
    raster than the tile's full model_bbox extent (see src/flood_extent.py),
    which also makes `resources: mem_mb` below (computed from `input.dem`)
    automatically shrink along with it. `crop_info` records whether this
    tile has ANY cell that could possibly flood for ANY scenario - if not,
    Aqueduct is skipped the same way as the zero-boundary-stations case.
    """
    input:
        toml=rules.write_aqueduct_config.output.toml,
        dem=rules.crop_flood_extent.output.dem,
        mask=rules.crop_flood_extent.output.mask,
        friction=rules.crop_flood_extent.output.friction,
        boundaries=rules.extract_boundaries.output.boundaries,
        crop_info=rules.crop_flood_extent.output.crop_info,
    output:
        waterdepth=os.path.join(
            config["simulation"]["model_outputs"], "{tile_id}", "results",
            "waterdepth_{return_period}_{waterlevel_name}.tif",
        ),
    params:
        model_outputs=config["simulation"]["model_outputs"],
        aqueduct_executable=config["simulation"]["aqueduct_executable"],
        raster_config=config["raster_format"],
        tile_grid_path=config["tile_grid"]["path"],
        n_scenarios_per_tile=len(RETURN_PERIODS) * len(WATERLEVEL_NAMES),
        # "python" (flood_model.flood_depth_dense, validated bit-exact - see
        # docs/python_vs_julia_qa.md) or "julia" (aqueduct_executable above,
        # kept as a fallback/comparison option) - see run_aqueduct.py.
        engine=config["simulation"].get("engine", "python"),
        flooding_config=config["simulation"]["flooding"],
    resources:
        # Calibrated from real Aqueduct (Julia) memory behaviour - a
        # deliberately generous upper bound that also covers the "python"
        # engine, whose dense solver's actual footprint is lower and more
        # predictable (no equivalent of Julia's component_indices memory
        # scaling issue) - left as-is (not retuned for Python) since it's a
        # concurrency-throttling estimate, not a correctness concern; only
        # effect of over-estimating is somewhat fewer concurrent jobs than
        # strictly necessary.
        mem_mb=lambda wildcards, input: estimate_aqueduct_mem_mb(
            input.dem, config["simulation"]["aqueduct_mem_estimate"]
        ),
    script:
        "../scripts/run_aqueduct.py"
