"""Extract the DEM-validity mask for a single tile, reprojected onto the DEM grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import dem_bounds_4326, extract_dem_mask, extract_local_dem_mask, load_raster, save_raster  # noqa: E402

dem = load_raster(snakemake.input.dem)  # noqa: F821
bbox = dem_bounds_4326(dem)  # model domain bbox encoded in the saved DEM

data_catalog = get_data_catalog(snakemake.params.data_catalog, root=snakemake.params.data_catalog_root)  # noqa: F821
dem_source = snakemake.params.dem_source  # noqa: F821

if dem_source == "deltadtm":
    mask = extract_dem_mask(
        data_catalog,
        "deltadtm_mask",
        bbox,
        dem,
        gebco_source="gebco",
    )
else:
    mask = extract_local_dem_mask(
        data_catalog, dem_source, bbox, dem, flat_fill_value=snakemake.params.flat_fill_value,  # noqa: F821
    )

save_raster(mask, snakemake.output.mask, snakemake.params.raster_config)  # noqa: F821
