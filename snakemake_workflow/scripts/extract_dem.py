"""Extract the DEM for a single tile and save it as a raster file."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import extract_dem, save_raster  # noqa: E402

with open(snakemake.input.model_bbox) as f:  # noqa: F821
    bbox = json.load(f)

data_catalog = get_data_catalog(snakemake.params.data_catalog, root=snakemake.params.data_catalog_root)  # noqa: F821

gap_fill_cfg = snakemake.params.dem_gap_fill_cfg  # noqa: F821
dem_source = snakemake.params.dem_source  # noqa: F821

dem = extract_dem(
    data_catalog,
    dem_source,
    bbox,
    mask_source="deltadtm_mask",  # global; local tiles derive their own mask from dem_source instead
    geoid_offset_raster=snakemake.input.geoid_offset_raster,  # noqa: F821
    min_hard_fill_component_size_m2=gap_fill_cfg["min_hard_fill_component_size_m2"],
    interp_max_search_distance_m=gap_fill_cfg["interp_max_search_distance_m"],
    interp_smoothing_iterations=gap_fill_cfg["interp_smoothing_iterations"],
    land_fill_value_m=gap_fill_cfg["land_fill_value_m"],
    gebco_source="gebco",
    pixel_size_m=snakemake.params.pixel_size_m,  # noqa: F821
    native_datum=snakemake.params.native_datum,  # noqa: F821
    hybrid_grid_raster=snakemake.params.hybrid_grid_raster,  # noqa: F821
    goco06s_absolute_raster=snakemake.params.goco06s_absolute_raster,  # noqa: F821
    derive_mask_from_dem=(dem_source != "deltadtm"),
    flat_fill_value=snakemake.params.flat_fill_value,  # noqa: F821
)

save_raster(dem, snakemake.output.dem, snakemake.params.raster_config, dtype="int16")  # noqa: F821
