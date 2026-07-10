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
    itself (see src/vertical_datum.py). Only pulled into the DAG when
    vertical_datum_correction.enabled is true (off by default) - extract_dem
    only depends on this rule's output in that case (see below).
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
        buffer_arcsec=config["simulation"]["model_bbox_buffer_arcsec"],
    script:
        "../scripts/compute_model_bbox.py"


rule extract_dem:
    """Clip the DEM to the model domain bbox.

    Optionally geoid-corrects the clipped elevation (EGM2008 -> GOCO06s) via
    the cached offset raster from compute_geoid_offset_raster above, when
    vertical_datum_correction.enabled is true (off by default) - see
    src/vertical_datum.py and rasters.extract_dem.
    """
    input:
        model_bbox=rules.compute_model_bbox.output.model_bbox,
        geoid_offset_raster=(
            rules.compute_geoid_offset_raster.output.offset_raster
            if _vertical_datum_correction_enabled else []
        ),
    output:
        dem=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "dem.tif"),
    params:
        data_catalog=config["paths"]["hydromt_data_catalog"],
        raster_config=config["raster_format"],
        vertical_datum_correction_enabled=_vertical_datum_correction_enabled,
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
        default_friction=config["simulation"]["flooding"]["default_friction"],
        raster_config=config["raster_format"],
    script:
        "../scripts/compute_friction.py"


rule extract_boundaries:
    """Extract water level boundary points for a single tile, return period and SLR scenario."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
    output:
        boundaries=os.path.join(
            config["simulation"]["model_outputs"], "{tile_id}", "inputs",
            "boundaries_{return_period}_{waterlevel_name}.gpkg",
        ),
    params:
        bc_cfg=config["boundary_conditions"],
    script:
        "../scripts/extract_boundaries.py"


rule write_aqueduct_config:
    """Write the Aqueduct TOML configuration for a single tile, return period and SLR scenario."""
    input:
        dem=rules.extract_dem.output.dem,
        mask=rules.extract_dem_mask.output.mask,
        friction=rules.compute_friction.output.friction,
        boundaries=rules.extract_boundaries.output.boundaries,
    output:
        toml=os.path.join(
            config["simulation"]["model_outputs"], "{tile_id}", "inputs",
            "aqueduct_{return_period}_{waterlevel_name}.toml",
        ),
    params:
        flooding_config=config["simulation"]["flooding"],
    script:
        "../scripts/write_aqueduct_config.py"
