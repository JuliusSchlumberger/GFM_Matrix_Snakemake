# Rule: run a short, basin-level SFINCS spin-up to produce a restart file,
# with the river at a fixed RP=1 and a calm sea (the level every event's own
# boundary lead-in starts at), entirely independent of any scenario's
# own design RP -- see 14_run_spinup.py's own module docstring for the
# full rationale and how it borrows geometry from the skeleton (rule
# build_sfincs_skeleton). No {scenario} wildcard: this runs ONCE per basin,
# not once per scenario, and every scenario's own event run (rule
# run_event, 16) references the SAME restart file via its own sfincs.inp
# (set in 13_build_sfincs.py).
#
# This is where the SFINCS solver is actually EXECUTED for the first time
# in the pipeline (rule build_sfincs_skeleton only assembles config/input
# files -- it never runs the executable, and neither does build_sfincs).

rule run_spinup:
    input:
        skeleton_inp         = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        land_mask_on_grid    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # Corrected sea/land classification (rule modelled_depth_estimation/
        # empirical_depth_estimation, whichever ran) -- cells the final weir
        # protects are cleared to "land" so they aren't masked as open sea in
        # this rule's own flood-diagnostic plot. Coarse, grid-aligned (rule
        # 09b's landuse_on_grid.tif is this file's own source) -- the SAME
        # file 13_build_sfincs_skeleton.py's own zsini reads. No separate
        # native-resolution sea_mask_corrected.tif anymore (removed
        # 2026-08-07b as pure duplication of this same boolean).
        sea_mask             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        surge_forcing        = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing        = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        grid_resolution      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
    output:
        rstart              = results_path("{basin_id}/spin_up/" + RST_FNAME),
        sfincs_map_nc       = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        plot_spinup         = results_path("{basin_id}/spin_up/validation_spinup.png"),
        plot_max_inundation = results_path("{basin_id}/spin_up/validation_max_inundation.png"),
    params:
        skeleton_root             = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        resolution                = lambda wildcards, input: json.load(open(input.grid_resolution))["resolution"],
        tref                      = config["sfincs"]["simulation"]["tref"],
        spinup_days               = config["sfincs"]["spinup"]["spinup_days"],
        sfincs_exe                = config["sfincs"]["simulation"]["sfincs_exe"],
        rst_fname                 = RST_FNAME,
        dtmapout_s                = config["sfincs"]["spinup"]["dtmapout_s"],
        dthisout_s                = config["sfincs"]["spinup"]["dthisout_s"],
        include_subgrid           = config["sfincs"]["subgrid"]["enabled"],
        timeout_s                 = config["sfincs"]["simulation"]["timeout_s"],
        waterlevel_buffer_m       = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        boundary_ramp_hours       = config["sfincs"]["spinup"]["boundary_ramp_hours"],
    # Claims the whole --cores budget: SFINCS itself can use multiple
    # threads (see run_sfincs_subprocess's OMP_NUM_THREADS handling), but
    # hydromt-sfincs/Snakemake can't parallelize *within* one run, so this
    # ensures no other job (this basin's or another's) competes with it for
    # CPU while it's executing -- the run gets the full machine instead.
    threads: workflow.cores
    log:
        "logs/{basin_id}/14_run_spinup.log"
    script:
        "../scripts/14_run_spinup.py"
