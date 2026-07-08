"""GFM Aqueduct preprocessing & simulation workflow.

For every tile in the overlapping tile grid (`paths.tile_grid`, built by
`snakemake_workflow/preparation/run_preparation.py` — see that directory's
tile_mask_creation.py, select_tiles.py and merge_tiles.py for how it's
derived from DeltaDTM + COAST-RP coverage), this workflow preprocesses the
DEM, DEM-validity mask, friction and water level boundary inputs for each
(return period, sea level rise scenario) combination, and runs the Aqueduct
flood model for every (tile, return_period, waterlevel_name) combination.

All fixed parameters are defined in `config/config.yml`.  `TILE_IDS` is the
static list of `tile_id` values read from `paths.tile_grid` at parse time —
run `snakemake_workflow/preparation/run_preparation.py` beforehand to build
and filter the tile grid down to tiles with DEM coverage.

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

# Machine-local overrides (git-ignored, optional).  Create
# snakemake_workflow/config/config_local.yml to override any key without
# touching the committed config.yml — most commonly `paths.root` and
# `paths.code_root` for machines with different storage layouts.
_LOCAL_CFG = "snakemake_workflow/config/config_local.yml"
if os.path.exists(_LOCAL_CFG):
    configfile: _LOCAL_CFG


def _expand_paths(obj, substitutions: dict) -> object:
    """Recursively substitute {key} placeholders in all string config values."""
    if isinstance(obj, str):
        for key, val in substitutions.items():
            obj = obj.replace(f"{{{key}}}", val)
        return obj
    if isinstance(obj, dict):
        return {k: _expand_paths(v, substitutions) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_paths(v, substitutions) for v in obj]
    return obj


# Expand {root} / {code_root} throughout the whole config so every rule and
# script receives absolute paths without knowing about the substitution scheme.
_PATH_SUBS = {
    "root": config["paths"].get("root", ""),
    "code_root": config["paths"].get("code_root", config["paths"].get("root", "")),
}
config = _expand_paths(config, _PATH_SUBS)


# ── Tile list (read once; also used to build the chunk grid below) ───────────
_tile_gdf = gpd.read_file(config["tile_grid"]["path"])
_tile_gdf["tile_id"] = _tile_gdf["tile_id"].astype(str)
TILE_IDS = sorted(_tile_gdf["tile_id"].astype(int).tolist())
RETURN_PERIODS = [f"RP{rp}" for rp in config["boundary_conditions"]["return_periods"]]

# Adaptation design intensities: the SLR scenarios used as the protection
# standard for the adaptation measures.
ADAPTATION_SLR_INTENSITIES = config["adaptation"]["slr_intensities"]

# Full set of SLR scenarios to simulate: union of the base scenario list and
# the adaptation intensities. Using an ordered-set pattern (dict.fromkeys) to
# deduplicate while preserving declaration order (base list first, then any
# extra intensities not already listed).
WATERLEVEL_NAMES = list(dict.fromkeys(
    config["boundary_conditions"]["slr_scenarios"] + ADAPTATION_SLR_INTENSITIES
))


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
    """Return the waterdepth tile paths for a given chunk_id + return_period + waterlevel_name.

    Built with explicit "/"-joins, NOT os.path.join: this is the one place in
    the whole Snakefile where a rule's input is a plain list of path STRINGS
    (from a data-dependent lookup, not a `rules.X.output.Y` reference), which
    forces Snakemake to resolve the producing rule via regex matching against
    every rule's output pattern instead of a direct object reference. On
    Windows, os.path.join produces backslash-joined paths that do not
    textually match run_aqueduct's (forward-slash, `{root}`-substituted)
    declared output pattern for that regex match - every OTHER cross-rule
    dependency in this Snakefile uses `rules.X.output.Y` (no regex matching
    ever needed) so never hits this. Confirmed by dry-run: this caused a
    MissingInputException for every chunk, not just tile-grid edge cases.
    """
    tile_ids = _chunk_tile_lookup[wildcards.chunk_id]
    model_outputs = config["simulation"]["model_outputs"]
    return [
        f"{model_outputs}/{tid}/results/waterdepth_{wildcards.return_period}_{wildcards.waterlevel_name}.tif"
        for tid in tile_ids
    ]


_PROTECTION_BASELINE_SLR = config["protection"]["baseline_waterlevel_name"]



include: "snakemake_workflow/rules/common.smk"
include: "snakemake_workflow/rules/preprocessing.smk"
include: "snakemake_workflow/rules/simulation.smk"
include: "snakemake_workflow/rules/postprocessing.smk"


_PREPROCESS_OUTPUTS = (
    expand(rules.extract_dem.output.dem, tile_id=TILE_IDS)
    + expand(rules.extract_dem_mask.output.mask, tile_id=TILE_IDS)
    + expand(rules.compute_friction.output.friction, tile_id=TILE_IDS)
    + expand(
        rules.extract_boundaries.output.boundaries,
        tile_id=TILE_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
    )
    + expand(
        rules.write_aqueduct_config.output.toml,
        tile_id=TILE_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
    )
)

_SIMULATION_OUTPUTS = expand(
    rules.run_aqueduct.output.waterdepth,
    tile_id=TILE_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
)

_plot_cfg = config["postprocessing"]["plots"]
_plotting_enabled = _plot_cfg["enabled"]

# Primary postprocessing outputs:
#   - per-chunk coarse flood-fraction rasters (all RP × SLR × chunks)
#     → consumed by standalone compute_exposure_analysis.py
#   - VRT mosaics and plots (optional, for diagnostics)
# Fine-resolution waterdepth and flood-count rasters are marked temp() in
# merge_chunk and auto-deleted once compute_flood_fraction_chunk completes.
# The old compute_protection_height_chunk / compute_exposure_chunk /
# aggregate_exposure_statistics chain is replaced by the coarse-first approach.
_POSTPROCESS_OUTPUTS = expand(
    rules.compute_flood_fraction_chunk.output.flood_fraction,
    chunk_id=CHUNK_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
)

if _plotting_enabled:
    _POSTPROCESS_OUTPUTS += (
        expand(rules.plot_merged_results.output.flood_count_plot, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_merged_results.output.waterdepth_plot, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_overlap_diagnostics.output.diagnostics, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_overlap_continent_diagnostics.output.diagnostics, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
    )


rule preprocess:
    """Generate all per-tile, per-scenario model inputs."""
    input:
        _PREPROCESS_OUTPUTS,


rule simulate:
    """Run the Aqueduct flood model for all tiles, return periods and SLR scenarios."""
    input:
        _SIMULATION_OUTPUTS,


rule postprocess:
    """Merge per-tile results into combined, multi-tile rasters and plot them, for all return periods and SLR scenarios."""
    input:
        _POSTPROCESS_OUTPUTS,


rule all:
    """Run the full workflow: preprocessing, simulation and postprocessing for all tiles, return periods and SLR scenarios."""
    input:
        _PREPROCESS_OUTPUTS + _SIMULATION_OUTPUTS + _POSTPROCESS_OUTPUTS,
