"""Rules for running the Aqueduct flood model."""


rule run_aqueduct:
    """Run the Aqueduct flood model for a single tile, return period and SLR scenario.

    The output filename includes `{return_period}_{waterlevel_name}` so it is
    clearly linked to the return period and sea level rise scenario used to
    produce it.

    The Aqueduct executable is a Julia binary whose LLVM JIT cannot reliably
    allocate memory when multiple instances run at the same time (it crashes
    with `OutOfMemoryError`/`LLVM ERROR: Unable to allocate section memory!`).
    This rule therefore claims the whole `aqueduct_runs` resource pool, so
    pass `--resources aqueduct_runs=1` on the command line to run at most one
    Aqueduct instance at a time while preprocessing rules still use the
    remaining `--cores`.

    If `boundaries` has no stations (the tile has no water level boundary
    points within it - see `boundaries.select_stations_for_tile`), Aqueduct
    is skipped entirely: an all-`AQUEDUCT_NODATA` placeholder is written for
    `waterdepth` (so `merge_results` ignores this tile) and the skip is
    logged to `model_outputs/skipped_tiles/{tile_id}_{return_period}_{waterlevel_name}.txt`
    for later mapping.

    If Aqueduct crashes with `OutOfMemoryError` (the memory cost of
    `component_indices` in `core/src/core.jl` scales with the tile's pixel
    count, not the return period or SLR scenario, so this is effectively a
    per-`tile_id` failure), the tile is marked in
    `model_outputs/oom_tiles/{tile_id}.txt` and this job's output falls back
    to the same all-`AQUEDUCT_NODATA` placeholder (logged to `skipped_tiles/`
    as above) instead of failing. Once marked, all other
    `(return_period, waterlevel_name)` combinations for this `tile_id` skip
    Aqueduct entirely and do the same, instead of also running out of memory.
    """
    input:
        toml=rules.write_aqueduct_config.output.toml,
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
        aqueduct_executable=config["simulation"]["aqueduct_executable"],
        raster_config=config["simulation"]["input_raster"],
    resources:
        aqueduct_runs=1,
    script:
        "../scripts/run_aqueduct.py"
