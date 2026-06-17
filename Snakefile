"""GFM Aqueduct preprocessing & simulation workflow.

For every tile in the overlapping tile grid (`paths.tile_grid`, see
`python/tile_mask_creation.py` and `select_tiles.py`), this workflow
preprocesses the DEM, DEM-validity mask, friction and water level boundary
inputs for each sea level rise (SLR) scenario, and runs the Aqueduct flood
model for each (tile, scenario) combination.

All fixed parameters are defined in `config/config.yml`.  `TILE_IDS` is the
static list of `tile_id` values read from `paths.tile_grid` at parse time —
run `select_tiles.py` beforehand to filter the tile grid down to tiles with
DEM coverage.

The postprocessing stage partitions the study area into spatial chunks of size
`merge.chunk_size_deg` degrees.  Each chunk is merged independently so only
the handful of tiles that overlap a chunk are ever read together.  All chunks
are mosaicked into a GDAL VRT for plotting.

Targets:
    preprocess:  generate all per-tile, per-scenario model inputs.
    simulate:    run the Aqueduct flood model for all tiles and SLR scenarios.
    postprocess: merge per-tile results into combined rasters and plots.
    all:         preprocess + simulate + postprocess (the default target).

Run with e.g. `snakemake all --cores 4 --resources aqueduct_runs=1` — the
`aqueduct_runs=1` resource limits the Aqueduct flood model to one instance at
a time while preprocessing rules use the remaining cores.
"""

import os

import geopandas as gpd
import numpy as np
from shapely.geometry import box as _shapely_box
from snakemake.utils import min_version

min_version("7.0")

configfile: "snakemake_workflow/config/config.yml"


# ── Tile list (read once; also used to build the chunk grid below) ───────────
_tile_gdf = gpd.read_file(config["tile_grid"]["path"])
_tile_gdf["tile_id"] = _tile_gdf["tile_id"].astype(str)
TILE_IDS = sorted(_tile_gdf["tile_id"].astype(int).tolist())
WATERLEVEL_NAMES = config["boundary_conditions"]["slr_scenarios"]


# ── Chunk grid (derived from tile_grid extent + chunk_size_deg) ──────────────
# Each chunk is identified by the N/S latitude and E/W longitude of its
# south-west corner, e.g. "S10E030" for the chunk at (lon=30°, lat=-10°).
# The grid is built programmatically — no external gpkg file needed.
# Changing `merge.chunk_size_deg` in config is all that is required to switch
# between e.g. 5° and 10° chunks.

def _build_chunk_grid(tile_gdf, chunk_size_deg):
    minx, miny, maxx, maxy = tile_gdf.total_bounds
    sz = chunk_size_deg
    xs = np.arange(np.floor(minx / sz) * sz, np.ceil(maxx / sz) * sz, sz)
    ys = np.arange(np.floor(miny / sz) * sz, np.ceil(maxy / sz) * sz, sz)
    rows = []
    for x in xs:
        for y in ys:
            cell = _shapely_box(x, y, x + sz, y + sz)
            if not tile_gdf.geometry.intersects(cell).any():
                continue
            xi, yi = int(round(x)), int(round(y))
            lat = f"N{yi:02d}" if yi >= 0 else f"S{-yi:02d}"
            lon = f"E{xi:03d}" if xi >= 0 else f"W{-xi:03d}"
            rows.append({
                "chunk_id": f"{lat}{lon}",
                "bounds": [x, y, x + sz, y + sz],
                "geometry": cell,
            })
    return gpd.GeoDataFrame(rows, crs=tile_gdf.crs)

_chunk_size_deg = config["postprocessing"]["chunk_size_deg"]
_chunk_grid = _build_chunk_grid(_tile_gdf, _chunk_size_deg)
CHUNK_IDS = _chunk_grid["chunk_id"].tolist()

# Fast lookup: chunk_id → [minx, miny, maxx, maxy]
_chunk_bounds_dict = {
    row["chunk_id"]: row["bounds"]
    for _, row in _chunk_grid.iterrows()
}

# Fast lookup: chunk_id → [tile_id, ...]
_chunk_tile_lookup = {
    row["chunk_id"]: _tile_gdf.loc[
        _tile_gdf.geometry.intersects(row["geometry"]), "tile_id"
    ].tolist()
    for _, row in _chunk_grid.iterrows()
}


def waterdepth_tiles_for_chunk(wildcards):
    """Return the waterdepth tile paths for a given chunk_id + waterlevel_name."""
    tile_ids = _chunk_tile_lookup[wildcards.chunk_id]
    return [
        os.path.join(
            config["simulation"]["model_outputs"],
            tid, "results",
            f"waterdepth_{wildcards.waterlevel_name}.tif",
        )
        for tid in tile_ids
    ]


include: "snakemake_workflow/rules/common.smk"
include: "snakemake_workflow/rules/preprocessing.smk"
include: "snakemake_workflow/rules/simulation.smk"
include: "snakemake_workflow/rules/postprocessing.smk"


_PREPROCESS_OUTPUTS = (
    expand(rules.extract_dem.output.dem, tile_id=TILE_IDS)
    + expand(rules.extract_dem_mask.output.mask, tile_id=TILE_IDS)
    + expand(rules.compute_friction.output.friction, tile_id=TILE_IDS)
    + expand(rules.extract_boundaries.output.boundaries, tile_id=TILE_IDS, waterlevel_name=WATERLEVEL_NAMES)
    + expand(rules.write_aqueduct_config.output.toml, tile_id=TILE_IDS, waterlevel_name=WATERLEVEL_NAMES)
)

_SIMULATION_OUTPUTS = expand(
    rules.run_aqueduct.output.waterdepth, tile_id=TILE_IDS, waterlevel_name=WATERLEVEL_NAMES
)

_plot_cfg = config["postprocessing"]["plots"]
_plotting_enabled = _plot_cfg["enabled"]
_corr_scenario = _plot_cfg["correlation_scenario"]  # None / null → no correlation plots

# VRT mosaics are always produced — they are the primary merged outputs even
# when plotting is disabled.
_POSTPROCESS_OUTPUTS = (
    expand(rules.build_mosaic_vrt.output.flood_count_vrt, waterlevel_name=WATERLEVEL_NAMES)
    + expand(rules.build_mosaic_vrt.output.waterdepth_vrt, waterlevel_name=WATERLEVEL_NAMES)
)

if _plotting_enabled:
    _POSTPROCESS_OUTPUTS += (
        expand(rules.plot_merged_results.output.flood_count_plot, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_merged_results.output.waterdepth_plot, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_overlap_diagnostics.output.diagnostics, waterlevel_name=WATERLEVEL_NAMES)
    )
    if _corr_scenario is not None:
        _POSTPROCESS_OUTPUTS += expand(
            rules.merge_chunk.output.overlap_correlation_plot,
            chunk_id=CHUNK_IDS, waterlevel_name=[_corr_scenario],
        )


rule preprocess:
    """Generate all per-tile, per-scenario model inputs."""
    input:
        _PREPROCESS_OUTPUTS,


rule simulate:
    """Run the Aqueduct flood model for all tiles and SLR scenarios."""
    input:
        _SIMULATION_OUTPUTS,


rule postprocess:
    """Merge per-tile results into combined, multi-tile rasters and plot them, for all SLR scenarios."""
    input:
        _POSTPROCESS_OUTPUTS,


rule all:
    """Run the full workflow: preprocessing, simulation and postprocessing for all tiles and SLR scenarios."""
    input:
        _PREPROCESS_OUTPUTS + _SIMULATION_OUTPUTS + _POSTPROCESS_OUTPUTS,
