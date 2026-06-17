"""Compute a tight model domain bounding box for a single tile."""

import json
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from rasters import compute_model_bbox, get_tile_bbox  # noqa: E402

tile = gpd.read_file(snakemake.input.tile_geometry)  # noqa: F821
tile_bbox = get_tile_bbox(tile)

data_catalog = get_data_catalog()
model_bbox = compute_model_bbox(
    data_catalog,
    "deltadtm",
    tile_bbox,
    buffer_arcsec=snakemake.config["simulation"]["model_bbox_buffer_arcsec"],  # noqa: F821
)

with open(snakemake.output.model_bbox, "w") as f:  # noqa: F821
    json.dump(model_bbox, f)
