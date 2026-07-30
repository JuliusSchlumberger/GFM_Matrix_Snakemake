"""Crop DEM/mask/friction to the region that could possibly flood for any
scenario of this tile - see src/flood_extent.py for the safety argument.

When `simulation.flood_extent_crop.enabled` is false, this still runs (so
the rest of the DAG never has to branch on the flag) but just copies
dem/mask/friction through unchanged, recording a full-tile window in
crop_info.json.
"""

import json
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import retry_transient_io  # noqa: E402
from flood_extent import compute_flood_candidate_window, crop_raster_to_window  # noqa: E402

crop_cfg = snakemake.params.crop_cfg  # noqa: F821
raster_config = snakemake.params.raster_config  # noqa: F821

with open(snakemake.input.max_waterlevel, encoding="utf-8") as f:  # noqa: F821
    max_waterlevel = json.load(f)["max_waterlevel"]

with retry_transient_io(rasterio.open, snakemake.input.dem) as src:  # noqa: F821
    orig_width, orig_height = src.width, src.height
    dem_arr = src.read(1)
with retry_transient_io(rasterio.open, snakemake.input.mask) as src:  # noqa: F821
    mask_arr = src.read(1)

empty_candidate = False
if not crop_cfg["enabled"]:
    window = Window(0, 0, orig_width, orig_height)
else:
    window = compute_flood_candidate_window(
        dem_arr, mask_arr, max_waterlevel,
        margin_px=crop_cfg["margin_px"],
        ocean_code=crop_cfg["ocean_code"],
    )
    if window is None:
        # Nothing can ever flood in this tile, for any scenario. A minimal
        # placeholder window is enough - run_aqueduct.py short-circuits on
        # crop_info.json's empty_candidate flag before these files are ever
        # read for real computation.
        empty_candidate = True
        window = Window(0, 0, min(2, orig_width), min(2, orig_height))

crop_raster_to_window(snakemake.input.dem, window, snakemake.output.dem, raster_config)  # noqa: F821
crop_raster_to_window(snakemake.input.mask, window, snakemake.output.mask, raster_config)  # noqa: F821
crop_raster_to_window(snakemake.input.friction, window, snakemake.output.friction, raster_config)  # noqa: F821

crop_info = {
    "enabled": crop_cfg["enabled"],
    "empty_candidate": empty_candidate,
    "window": [window.col_off, window.row_off, window.width, window.height],
    "original_width": orig_width,
    "original_height": orig_height,
    "cropped_width": window.width,
    "cropped_height": window.height,
    "original_pixels": orig_width * orig_height,
    "cropped_pixels": window.width * window.height,
}
with open(snakemake.output.crop_info, "w", encoding="utf-8") as f:  # noqa: F821
    json.dump(crop_info, f, indent=2)
