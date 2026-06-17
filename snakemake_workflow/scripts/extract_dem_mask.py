"""Extract the DEM-validity mask for a single tile, reprojected onto the DEM grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import extract_dem_mask, load_raster, save_raster  # noqa: E402

dem = load_raster(snakemake.input.dem)  # noqa: F821
bbox = list(dem.raster.bounds)  # model domain bbox encoded in the saved DEM

data_catalog = get_data_catalog()
mask = extract_dem_mask(
    data_catalog,
    "deltadtm_mask",
    bbox,
    dem,
    land_polygons_source="land_polygons",
)

save_raster(mask, snakemake.output.mask, snakemake.config["simulation"]["input_raster"])  # noqa: F821
