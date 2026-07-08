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
dem = extract_dem(
    data_catalog,
    "deltadtm",
    bbox,
    mask_source="deltadtm_mask",
)

save_raster(dem, snakemake.output.dem, snakemake.params.raster_config)  # noqa: F821
