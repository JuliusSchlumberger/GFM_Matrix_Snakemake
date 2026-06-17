"""Rules for preprocessing per-tile model inputs.

Each rule produces a single file in `model_outputs/{tile_id}/inputs/`. All
fixed parameters (data source names, raster output options, water level
scenario definitions) are read from `config/config.yml`.
"""


rule extract_tile_geometry:
    """Extract a single tile's geometry from the overlapping tile grid."""
    output:
        tile_geometry=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "tile_geometry.gpkg"),
    script:
        "../scripts/extract_tile_geometry.py"


rule compute_model_bbox:
    """Compute a tight model domain bbox from DeltaDTM extent and ocean areas."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
    output:
        model_bbox=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "model_bbox.json"),
    script:
        "../scripts/compute_model_bbox.py"


rule extract_dem:
    """Clip the DEM to the model domain bbox."""
    input:
        model_bbox=rules.compute_model_bbox.output.model_bbox,
    output:
        dem=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "dem.tif"),
    script:
        "../scripts/extract_dem.py"


rule extract_dem_mask:
    """Clip the DEM-validity mask and reproject it onto the DEM grid for a single tile."""
    input:
        dem=rules.extract_dem.output.dem,
    output:
        mask=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "mask.tif"),
    script:
        "../scripts/extract_dem_mask.py"


rule compute_friction:
    """Compute the friction raster for a single tile from land use data."""
    input:
        dem=rules.extract_dem.output.dem,
    output:
        friction=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "friction.tif"),
    script:
        "../scripts/compute_friction.py"


rule extract_boundaries:
    """Extract water level boundary points for a single tile and SLR scenario."""
    input:
        tile_geometry=rules.extract_tile_geometry.output.tile_geometry,
    output:
        boundaries=os.path.join(
            config["simulation"]["model_outputs"], "{tile_id}", "inputs", "boundaries_{waterlevel_name}.gpkg"
        ),
    script:
        "../scripts/extract_boundaries.py"


rule write_aqueduct_config:
    """Write the Aqueduct TOML configuration for a single tile and SLR scenario."""
    input:
        dem=rules.extract_dem.output.dem,
        mask=rules.extract_dem_mask.output.mask,
        friction=rules.compute_friction.output.friction,
        boundaries=rules.extract_boundaries.output.boundaries,
    output:
        toml=os.path.join(config["simulation"]["model_outputs"], "{tile_id}", "inputs", "aqueduct_{waterlevel_name}.toml"),
    script:
        "../scripts/write_aqueduct_config.py"
