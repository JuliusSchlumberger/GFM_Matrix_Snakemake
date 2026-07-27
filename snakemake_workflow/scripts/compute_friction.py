"""Compute the friction raster for a single tile, reprojected onto the DEM grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import compute_friction, load_raster, save_raster  # noqa: E402

dem = load_raster(snakemake.input.dem)  # noqa: F821
bbox = list(dem.raster.bounds)  # model domain bbox encoded in the saved DEM

data_catalog = get_data_catalog(snakemake.params.data_catalog, root=snakemake.params.data_catalog_root)  # noqa: F821
friction = compute_friction(
    data_catalog,
    "land_use",
    "lu_to_roughness_lookup",
    bbox,
    dem,
    snakemake.params.default_friction,  # noqa: F821
)

save_raster(friction, snakemake.output.friction, snakemake.params.raster_config)  # noqa: F821
