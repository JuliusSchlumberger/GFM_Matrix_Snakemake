"""Extract the DEM for a single tile and save it as a raster file."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import extract_dem, save_raster  # noqa: E402

with open(snakemake.input.model_bbox) as f:  # noqa: F821
    bbox = json.load(f)

data_catalog = get_data_catalog(snakemake.params.data_catalog)  # noqa: F821

# Only defined in input: when vertical_datum_correction.enabled (see
# preprocessing.smk) - the params flag mirrors that same condition, so
# snakemake.input.geoid_offset_raster is only ever accessed when it's real.
geoid_offset_raster = (
    snakemake.input.geoid_offset_raster  # noqa: F821
    if snakemake.params.vertical_datum_correction_enabled  # noqa: F821
    else None
)

gap_fill_cfg = snakemake.params.dem_gap_fill_cfg  # noqa: F821

dem = extract_dem(
    data_catalog,
    "deltadtm",
    bbox,
    mask_source="deltadtm_mask",
    geoid_offset_raster=geoid_offset_raster,
    min_hard_fill_component_size=gap_fill_cfg["min_hard_fill_component_size"],
    interp_max_search_distance=gap_fill_cfg["interp_max_search_distance"],
    interp_smoothing_iterations=gap_fill_cfg["interp_smoothing_iterations"],
    land_fill_value_m=gap_fill_cfg["land_fill_value_m"],
)

save_raster(dem, snakemake.output.dem, snakemake.params.raster_config)  # noqa: F821
