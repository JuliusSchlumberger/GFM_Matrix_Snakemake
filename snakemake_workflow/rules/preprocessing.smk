"""Rules for preprocessing per-tile model inputs.

Each rule produces a single file in `model_outputs/{tile_id}/inputs/`. All
fixed parameters (data source names, raster output options, water level
scenario definitions) are read from `config/config.yml`.
"""


rule compute_geoid_offset_raster:
    """One-time computation of the global EGM2008 -> GOCO06s geoid-offset
    field (pyshtools spherical-harmonic synthesis, ~12s), cached to a small
    GeoTIFF. NOT tile_id-wildcarded: the offset field is smooth and global,
    so it only needs to be computed once regardless of how many tiles exist
    - every per-tile `extract_dem` job (below) reads a tiny reprojected
    window of this cached raster instead of resynthesizing the harmonics
    itself (see src/vertical_datum.py). Always in the DAG - extract_dem
    always depends on this rule's output (see below).
    """
    input:
        egm2008_gfc=_data_catalog.get_source("egm2008_geoid").path,
        goco06s_gfc=_data_catalog.get_source("goco06s").path,
    output:
        offset_raster=_vc_cfg["offset_raster_path"],
    script:
        "../scripts/compute_geoid_offset_raster.py"


rule extract_tile_geometry:
    """Extract a single tile's geometry from the overlapping tile grid."""
    output:
        tile_geometry=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "tile_geometry.gpkg"),
    params:
        tile_grid_path=config["tile_grid"]["path"],
    script:
        "../scripts/extract_tile_geometry.py"


rule compute_model_bbox:
    """Compute a tight model domain bbox from DeltaDTM extent and ocean areas."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
    output:
        model_bbox=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "model_bbox.json"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        buffer_arcsec=config["simulation"]["model_bbox_buffer_arcsec"],
        elevation_threshold_m=config["tile_generation"]["elev_threshold_m"],
    script:
        "../scripts/compute_model_bbox.py"


rule extract_dem:
    """Clip the DEM to the model domain bbox.

    Geoid-corrects the clipped elevation (EGM2008 -> GOCO06s) via the cached
    offset raster from compute_geoid_offset_raster above - see
    src/vertical_datum.py and rasters.extract_dem.
    """
    input:
        model_bbox=rules.compute_model_bbox.output.model_bbox,
        geoid_offset_raster=rules.compute_geoid_offset_raster.output.offset_raster,
    output:
        dem=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "dem.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        raster_config=config["raster_format"],
        dem_gap_fill_cfg=config["simulation"]["dem_gap_fill"],
    script:
        "../scripts/extract_dem.py"


rule extract_dem_mask:
    """Clip the DEM-validity mask and reproject it onto the DEM grid for a single tile."""
    input:
        dem=rules.extract_dem.output.dem,
    output:
        mask=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "mask.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        raster_config=config["raster_format"],
    script:
        "../scripts/extract_dem_mask.py"


rule compute_friction:
    """Compute the friction raster for a single tile from land use data."""
    input:
        dem=rules.extract_dem.output.dem,
    output:
        friction=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "friction.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        data_catalog_root=config["paths"]["root"],
        default_friction=config["simulation"]["flooding"]["default_friction"],
        raster_config=config["raster_format"],
    script:
        "../scripts/compute_friction.py"


# Resolved ONCE here at Snakefile-parse time (mirrors _data_catalog's own
# module-level construction in the root Snakefile), not per-job - unlike
# extract_dem/extract_dem_mask/compute_friction/compute_model_bbox (which
# genuinely need a live DataCatalog object for real raster reads via
# get_rasterdataset), extract_boundaries.py only ever used its own per-job
# get_data_catalog() call for ONE static path
# (deltadtm_mask's directory, for filter_stations_by_ocean_connectivity) that
# never varies across tiles/scenarios - rebuilding a whole HydroMT catalog
# (~5s, confirmed via real HPC job logs 2026-08-10: consecutive
# "Parsing data catalog" lines exactly 5s apart) JUST for that, ~45x per tile
# (once per return_period x waterlevel_name boundaries file), was costing
# ~3-4 minutes of pure overhead per tile - the dominant share of the ~5min/
# tile preprocessing throughput observed live on Hydrax.
_deltadtm_mask_dir = os.path.dirname(_data_catalog.get_source("deltadtm_mask").path)

_stations_cache_dir = os.path.join(config["paths"]["processed_inputs_dir"], "WL_scenarios_cache")


rule cache_waterlevel_stations:
    """Cache one (return_period, waterlevel_name) scenario's global COAST-RP
    water-level stations ONCE, as a small pre-parsed GeoPackage.

    The source NetCDF (boundary_conditions.nc_filename_template) is wildcarded
    ONLY by (return_period, waterlevel_name) - it has no tile dimension at
    all, so its station set is 100% identical for every tile that needs this
    scenario. Before this rule existed, every tile's own extract_boundaries
    job re-opened and re-parsed that same NetCDF via xarray independently
    (~5s each, confirmed against real tile 1013 data 2026-08-10) - up to
    ~2578x redundant re-reads of the same ~45 files. Same "compute once
    globally, read cheaply per tile" pattern as compute_geoid_offset_raster.

    NOT temp() - reused across every future preprocessing batch/invocation
    that needs this scenario, not just jobs within one Snakemake run. NOT
    tile-wildcarded, so - like compute_geoid_offset_raster's output - this is
    a shared, non-tile-specific output: on HPC, it must be built during the
    same build_shared_inputs phase-0 step (see generate_hpc_preprocess_job.py)
    BEFORE any --nolock batch starts, to avoid the exact concurrent-write
    race compute_geoid_offset_raster's own phase-0 build already exists to
    prevent (see that rule's/generate_hpc_preprocess_job.py's own docstrings).
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
            config["simulation"]["model_outputs"], "{tile_id}", "inputs",
            "boundaries_{return_period}_{waterlevel_name}.gpkg",
        ),
    params:
        bc_cfg=config["boundary_conditions"],
        mask_dir=_deltadtm_mask_dir,
    script:
        "../scripts/extract_boundaries.py"
