# Rule: build the scenario-DEPENDENT forcing on top of a basin's already-
# built SFINCS skeleton (rule build_sfincs_skeleton, 13_build_sfincs_
# skeleton.smk). Loads the skeleton in read mode, redirects writes to this
# scenario's own directory, and writes ONLY the forcing-related files
# (sfincs.bzs/.bnd/.dis/.src, its own hand-crafted sfincs.inp) -- see
# 13_build_sfincs.py's own module docstring for the full rationale and the
# HydroMT behavior (config path absolutization) that makes hand-crafting
# the .inp necessary rather than relying on sf.write().
#
# depends on the skeleton's own sfincs.inp (rather than nothing at all) so
# Snakemake still reruns this rule when the basin-level geometry genuinely
# changes -- but changing ONLY a scenario's own RP (surge_rp/river_rp in
# config/scenarios.yml) does not touch the skeleton's own inputs/outputs at
# all, so build_sfincs_skeleton itself is untouched, and (more importantly)
# rule run_spinup -- which now depends on the skeleton too, never on any
# scenario's own build -- does not need to re-run either.

rule build_sfincs:
    input:
        skeleton_inp      = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        grid_resolution   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
        # Only present when spin-up is enabled -- see include_rstart param.
        rstart = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/spin_up/" + RST_FNAME)
            if config["sfincs"]["spinup"]["enabled"] else []
        ),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
    params:
        depth_method       = config["river_processing"]["depth_method"],
        resolution         = lambda wildcards, input: json.load(open(input.grid_resolution))["resolution"],
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        include_rstart     = config["sfincs"]["spinup"]["enabled"],
        spinup_days        = config["sfincs"]["spinup"]["spinup_days"],
        rst_fname          = RST_FNAME,
        # "river_only" flat boundary level: set BELOW terrain.gebco_max_depth_m
        # (the deepest any clamped ocean bed cell can be), so the boundary is
        # guaranteed dry everywhere -- the model's behavior is then driven
        # entirely by river discharge, not by any water entering from the
        # coastal boundary. The extra 0.5 m is a safety margin below the clamp.
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        # Resolved per-scenario (see scenario_params in 00_common.smk): mode
        # is derived from which of surge_rp/river_rp are set for THIS
        # scenario's own entry in config/scenarios.yml -- "default" is not
        # special-cased, it is just the one scenario name that's required
        # to exist and gets used when target_scenarios isn't given.
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr    = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        # Per-scenario scaling factor on the built river discharge hydrograph
        # (config/scenarios.yml, see scenario_params in 00_common.smk;
        # default 1.0 = no-op), applied HERE against river_forcing.nc's own
        # built discharge -- deliberately NOT a param of rule
        # get_boundary_forcings (07) or rule modelled_depth_estimation (10),
        # so changing it only reruns this (cheap) per-scenario build + its
        # downstream event run, never rule 07, the weir/depth calibration,
        # or the skeleton build. Per-scenario (rather than global) so e.g.
        # river_500/river_only_500/compound_500 can be amplified to reach a
        # genuinely flood-inducing river discharge without also scaling
        # coast_500's own small RP=2 river component. See
        # src.river_forcing.build_design_discharge_matrix.
        discharge_multiplier = lambda wildcards: scenario_params(wildcards.scenario)["discharge_multiplier"],
        # Target global-mean SLR (m), applied HERE against surge_forcing.nc's
        # own dimensionless slr_fingerprint -- deliberately NOT a param of
        # rule get_boundary_forcings (07), so changing slr_m only reruns this
        # (cheap) per-scenario build + its downstream event run, never rule
        # 07 itself, rule 10's weir/depth calibration, or the skeleton build.
        # See src.surge.build_design_surge_matrix's own slr_m argument.
        slr_enabled        = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m              = config["boundary_forcings"]["surge"]["slr"]["slr_m"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        spin_up_root       = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
    log:
        "logs/{basin_id}/runs/{scenario}/13_build_sfincs.log"
    script:
        "../scripts/13_build_sfincs.py"
