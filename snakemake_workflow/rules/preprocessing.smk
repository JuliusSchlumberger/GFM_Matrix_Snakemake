"""Rules for preprocessing per-tile model inputs into `{preprocessing_inputs_dir}/{tile_id}/inputs/`.

`_preprocessing_inputs_dir` defaults to `simulation.model_outputs`. Calibration
sweeps whose points only change solver-runtime/exposure parameters can point it
at a group-level shared directory; model_outputs must stay per-run because
results/waterdepth_*.tif filenames carry no sweep-point tag.
"""

_preprocessing_inputs_dir = config["simulation"].get("preprocessing_inputs_dir") or config["simulation"]["model_outputs"]


rule compute_geoid_offset_raster:
    """Compute the global EGM2008 -> GOCO06s geoid-offset field once, as a GeoTIFF.

    Not tile-wildcarded: the field is smooth and global, so extract_dem reads a
    reprojected window of it instead of resynthesizing harmonics (pyshtools).
    """
    input:
        egm2008_gfc=_data_catalog.get_source("egm2008_geoid").path,
        goco06s_gfc=_data_catalog.get_source("goco06s").path,
    output:
        offset_raster=_vc_cfg["offset_raster_path"],
    script:
        "../scripts/compute_geoid_offset_raster.py"


rule compute_goco06s_absolute_raster:
    """Compute GOCO06s's own absolute geoid-height field once, as a GeoTIFF.

    Needed by a local DEM source's vertical-datum bridge (e.g. odn_osgm15), not by
    any deltadtm tile - see extract_dem's native_datum / vertical_datum.py.
    """
    input:
        goco06s_gfc=_data_catalog.get_source("goco06s").path,
    output:
        absolute_raster=_vc_cfg["goco06s_absolute_raster_path"],
    script:
        "../scripts/compute_goco06s_absolute_raster.py"


rule extract_tile_geometry:
    """Extract a single tile's geometry from the overlapping tile grid."""
    output:
        tile_geometry=os.path.join(_preprocessing_inputs_dir, "{tile_id}", "inputs", "tile_geometry.gpkg"),
    params:
        tile_grid_path=config["tile_grid"]["path"],
    script:
        "../scripts/extract_tile_geometry.py"


rule compute_model_bbox:
    """Compute a tight model domain bbox from DeltaDTM extent and ocean areas."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
    output:
        model_bbox=os.path.join(_preprocessing_inputs_dir, "{tile_id}", "inputs", "model_bbox.json"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        buffer_arcsec=config["simulation"]["model_bbox_buffer_arcsec"],
        elevation_threshold_m=config["tile_generation"]["elev_threshold_m"],
    script:
        "../scripts/compute_model_bbox.py"


rule extract_dem:
    """Clip the DEM to the model bbox and geoid-correct it to GOCO06s.

    `dem_source` is `deltadtm` (native_datum egm2008) for every tile unless
    `simulation.dem_source_manifest` names it otherwise - see Snakefile's
    `_dem_source_for_tile`/`_dem_source_meta_for_tile` (local-input
    validation, TODO.md).
    """
    input:
        model_bbox=rules.compute_model_bbox.output.model_bbox,
        geoid_offset_raster=rules.compute_geoid_offset_raster.output.offset_raster,
        gebco=_data_catalog.get_source("gebco").path,
    output:
        dem=os.path.join(_preprocessing_inputs_dir, "{tile_id}", "inputs", "dem.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        raster_config=config["raster_format"],
        dem_gap_fill_cfg=config["simulation"]["dem_gap_fill"],
        dem_source=lambda wc: _dem_source_for_tile(wc.tile_id),
        native_datum=lambda wc: _dem_source_meta_for_tile(wc.tile_id, "native_datum", "egm2008"),
        pixel_size_m=lambda wc: _dem_source_meta_for_tile(wc.tile_id, "pixel_size_m", 30.0),
        hybrid_grid_raster=lambda wc: _dem_source_meta_for_tile(wc.tile_id, "hybrid_grid_raster_path", None),
        goco06s_absolute_raster=_vc_cfg["goco06s_absolute_raster_path"],
        flat_fill_value=lambda wc: _dem_source_meta_for_tile(wc.tile_id, "flat_fill_value", None),
    script:
        "../scripts/extract_dem.py"


rule extract_dem_mask:
    """Clip the DEM-validity mask and reproject it onto the DEM grid for a single tile.

    A local dem_source (TODO.md local-input validation) derives its own mask
    from its raw nodata footprint instead - see `rasters.extract_local_dem_mask`.
    """
    input:
        dem=rules.extract_dem.output.dem,
        gebco=_data_catalog.get_source("gebco").path,
    output:
        mask=os.path.join(_preprocessing_inputs_dir, "{tile_id}", "inputs", "mask.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        raster_config=config["raster_format"],
        dem_source=lambda wc: _dem_source_for_tile(wc.tile_id),
        flat_fill_value=lambda wc: _dem_source_meta_for_tile(wc.tile_id, "flat_fill_value", None),
    script:
        "../scripts/extract_dem_mask.py"


rule compute_friction:
    """Compute the friction raster for a single tile from land use data."""
    input:
        dem=rules.extract_dem.output.dem,
    output:
        friction=os.path.join(_preprocessing_inputs_dir, "{tile_id}", "inputs", "friction.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        default_friction=config["simulation"]["flooding"]["default_friction"],
        raster_config=config["raster_format"],
    script:
        "../scripts/compute_friction.py"


# Resolved once at parse time: rebuilding a HydroMT catalog per boundaries job
# (~5 s each, ~45x per tile) only for this static path dominated tile runtime.
_deltadtm_mask_dir = os.path.dirname(_data_catalog.get_source("deltadtm_mask").path)

# Override (boundary_conditions.stations_cache_dir) for a scenario whose waterlevel_name
# values are not the shared global COAST-RP cache (e.g. a local-input validation run
# pre-seeding its own stations_*.gpkg under an isolated directory) - see TODO.md.
_stations_cache_dir = config["boundary_conditions"].get("stations_cache_dir") or os.path.join(
    config["paths"]["processed_inputs_dir"], "WL_scenarios_cache"
)


rule cache_waterlevel_stations:
    """Cache one scenario's global COAST-RP water-level stations as a GeoPackage.

    The source NetCDF has no tile dimension, so parsing it once replaces a
    per-tile re-read. Not temp(): reused across runs. Shared, non-tile output:
    on HPC it must be built in the build_shared_inputs step
    (generate_hpc_preprocess_job.py) before any --nolock batch starts, to
    avoid concurrent-write races.
    """
    output:
        stations_cache=os.path.join(
            _stations_cache_dir, "stations_{return_period}_{waterlevel_name}.gpkg",
        ),
    params:
        bc_cfg=config["boundary_conditions"],
    script:
        "../scripts/cache_waterlevel_stations.py"


rule extract_boundaries:
    """Extract water level boundary points for a single tile, return period and SLR scenario."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
        stations_cache=rules.cache_waterlevel_stations.output.stations_cache,
    output:
        boundaries=os.path.join(
            _preprocessing_inputs_dir, "{tile_id}", "inputs",
            "boundaries_{return_period}_{waterlevel_name}.gpkg",
        ),
    params:
        bc_cfg=config["boundary_conditions"],
        mask_dir=_deltadtm_mask_dir,
    script:
        "../scripts/extract_boundaries.py"
