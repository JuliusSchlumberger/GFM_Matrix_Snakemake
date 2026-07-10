"""Plot the merged water-depth raster for a single return period and SLR scenario."""

import os
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from plotting import compute_flood_area_km2, plot_raster_with_coastlines  # noqa: E402

pp_cfg = snakemake.params.pp_cfg  # noqa: F821
plot_cfg = pp_cfg["plots"]
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_label = f"{return_period}, {waterlevel_name}"

with rasterio.open(snakemake.input.waterdepth) as src:  # noqa: F821
    bounds = src.bounds

data_catalog = get_data_catalog(snakemake.params.data_catalog)  # noqa: F821
coastlines_path = data_catalog.get_source("land_polygons").path
# Read directly with a bbox filter (uses the GeoPackage's spatial index) -
# data_catalog.get_geodataframe(..., bbox=...) reads the whole global dataset
# first, which takes minutes for this source.
coastlines = gpd.read_file(coastlines_path, layer="land_polygons", bbox=tuple(bounds))

oom_dir = os.path.join(snakemake.params.model_outputs, "oom_tiles")  # noqa: F821
if os.path.isdir(oom_dir):
    oom_tile_ids = {os.path.splitext(f)[0] for f in os.listdir(oom_dir) if f.endswith(".txt")}
    tile_grid = gpd.read_file(snakemake.params.tile_grid_path)  # noqa: F821
    tile_grid["tile_id"] = tile_grid["tile_id"].astype(str)
    oom_tiles = tile_grid[tile_grid["tile_id"].isin(oom_tile_ids)]
else:
    oom_tiles = None

threshold_m = snakemake.params.threshold_m  # noqa: F821
flood_area_km2 = compute_flood_area_km2(snakemake.input.waterdepth, threshold_m)  # noqa: F821
flood_annotation = (
    f"Flooded area ≥ {threshold_m * 100:.0f} cm:  {flood_area_km2:,.0f} km²"
)

plot_raster_with_coastlines(
    raster_path=snakemake.input.waterdepth,  # noqa: F821
    coastlines=coastlines,
    output_path=snakemake.output.waterdepth_plot,  # noqa: F821
    title=f"Merged water depth ({scenario_label})",
    label="Water depth (m)",
    cmap=plot_cfg["waterdepth_cmap"],
    resolution_arcsec=plot_cfg["merged_resolution_arcsec"],
    mask_value=0,
    oom_tiles=oom_tiles,
    annotation=flood_annotation,
    vmax_m=plot_cfg["waterdepth_vmax_m"],
    figsize=tuple(plot_cfg["merged_map_figsize"]),
    dpi=plot_cfg["dpi"],
)
