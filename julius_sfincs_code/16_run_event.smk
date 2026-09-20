# Rule: run the main flood-event SFINCS simulation and its sanity checks.
#
# Unlike rule 14 (run_spinup), which writes its OWN shortened sfincs.inp into
# a spinup/ subdirectory, this rule runs the MAIN sfincs.inp that rule 13
# (build_sfincs) already wrote directly into sfincs_root -- it is already
# configured with tstart = spin-up end, tstop = end of the full forcing
# timeseries, and rstfile pointing at rule 14's restart file (see rule 13's
# "Phase 2 config" section). This rule only executes it, then produces the
# same kind of sanity-check diagnostics as rule 15 (sanity_checks) -- but for
# this event run's own sfincs_map.nc rather than the spin-up's -- plus a
# per-timestep flooded-area/flood-volume CSV.

rule run_event:
    input:
        sfincs_inp           = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
        rstart               = results_path("{basin_id}/spin_up/" + RST_FNAME),
        # Grid-aligned land mask (rule grid_align_landuse) -- plot background.
        land_mask_on_grid    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # Corrected sea/land classification (see rule run_spinup's own
        # comment) -- cells the final weir protects are cleared to "land",
        # not masked as open sea in this scenario's own flood diagnostics.
        sea_mask             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc              = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        plot_inundation_ratio      = results_path("{basin_id}/runs/{scenario}/visuals/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/runs/{scenario}/visuals/02_flood_animation.mp4"),
        flood_timeseries_csv       = results_path("{basin_id}/runs/{scenario}/visuals/flood_timeseries.csv"),
    params:
        sfincs_root                = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        # Subgrid reference raster for postprocessing lives in the skeleton
        # (rule build_sfincs_skeleton), not physically in sfincs_root --
        # this scenario's own model only references it via a relative path
        # in its own sfincs.inp (see 13_build_sfincs.py).
        skeleton_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        sfincs_exe                 = config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s                  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid            = config["sfincs"]["subgrid"]["enabled"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    # Claims the whole --cores budget -- see rule run_spinup's (14) threads
    # comment for the full rationale (SFINCS itself can multi-thread, but
    # Snakemake can't parallelize within one run, so give it the whole
    # machine rather than let other jobs compete with it for CPU).
    threads: workflow.cores
    log:
        "logs/{basin_id}/runs/{scenario}/16_run_event.log"
    script:
        "../scripts/16_run_event.py"
