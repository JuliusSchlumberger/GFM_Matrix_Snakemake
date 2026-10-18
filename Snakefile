"""GFM eikonal workflow: preprocess, simulate and merge per tile and scenario.

Run `preparation/run_preparation.py` first (builds the tile grid). Targets:
preprocess, simulate, postprocess, all. E.g. `snakemake all --cores 4
--resources mem_mb=8000`; set mem_mb to ~80% of physical RAM.
"""

import os
import sys
import time

import geopandas as gpd
from snakemake.utils import min_version

min_version("7.0")

# Patch os.makedirs to retry FileNotFoundError: a momentary SMB drop makes the
# shared data root vanish ([WinError 3]) and Snakemake's own output-dir creation does not
# retry. Path.mkdir() is not covered.
_ORIGINAL_MAKEDIRS = os.makedirs


def _retrying_makedirs(name, mode=0o777, exist_ok=False, _retries=5, _delay_s=15.0):
    last_err = None
    for attempt in range(1, _retries + 1):
        try:
            return _ORIGINAL_MAKEDIRS(name, mode, exist_ok)
        except FileNotFoundError as e:
            last_err = e
            if attempt < _retries:
                print(
                    f"[retry {attempt}/{_retries - 1}] os.makedirs({name!r}) failed: {e} "
                    f"- retrying in {_delay_s:.0f}s",
                    flush=True,
                )
                time.sleep(_delay_s)
    raise last_err


os.makedirs = _retrying_makedirs

sys.path.insert(0, os.path.join(workflow.basedir, "src"))
from eikonal_runner import estimate_eikonal_mem_mb  # noqa: E402
from config_utils import _expand_paths, get_data_catalog, merged_waterlevel_names  # noqa: E402
from tiles import build_chunk_grid  # noqa: E402

# GFM_CONFIG_PATH selects a materialized scenario config via the `configfile:`
# directive instead of --configfile, which failed reproducibly (unexplained) on HPC.
_CONFIG_PATH = os.environ.get("GFM_CONFIG_PATH", "snakemake_workflow/config/config.yml")
configfile: _CONFIG_PATH

# Machine-local overrides; keep in sync with config_utils.load_config(). Skipped
# with GFM_CONFIG_PATH: materialize_config() already folded it in, and reapplying
# could override scenario values.
_LOCAL_CFG = "snakemake_workflow/config/config_local.yml"
if "GFM_CONFIG_PATH" not in os.environ and os.path.exists(_LOCAL_CFG):
    configfile: _LOCAL_CFG


if not config["paths"].get("root"):
    raise ValueError("paths.root is not set - create snakemake_workflow/config/config_local.yml (copy config_local.yml.example) or pass a config that sets it (config_hpc.yml / a materialized scenario config)")
_PATH_SUBS = {
    "root": config["paths"]["root"],
    "code_root": config["paths"].get("code_root", config["paths"].get("root", "")),
}
# processed_inputs_dir is itself "{root}/...", so it is expanded first.
_PATH_SUBS["processed_inputs_dir"] = _expand_paths(config["paths"]["processed_inputs_dir"], _PATH_SUBS)
config = _expand_paths(config, _PATH_SUBS)


_tile_gdf = gpd.read_file(config["tile_grid"]["path"])
_tile_gdf["tile_id"] = _tile_gdf["tile_id"].astype(str)
TILE_IDS = sorted(_tile_gdf["tile_id"].astype(int).tolist())
RETURN_PERIODS = [f"RP{rp}" for rp in config["boundary_conditions"]["return_periods"]]

# Base SLR scenarios + adaptation design intensities, sorted (np.interp needs
# increasing values).
WATERLEVEL_NAMES = merged_waterlevel_names(config["boundary_conditions"], config["adaptation"])


# Chunk ids name the south-west corner, e.g. "S10E030" = (lon=30°, lat=-10°).
_chunk_size_deg = config["postprocessing"]["chunk_size_deg"]
_chunk_grid = build_chunk_grid(_tile_gdf, _chunk_size_deg)
CHUNK_IDS = _chunk_grid["chunk_id"].tolist()

_chunk_bounds_dict = {
    row["chunk_id"]: row["bounds"]
    for _, row in _chunk_grid.iterrows()
}

_chunk_tile_lookup = {
    row["chunk_id"]: _tile_gdf.loc[
        _tile_gdf.geometry.intersects(row["geometry"]), "tile_id"
    ].tolist()
    for _, row in _chunk_grid.iterrows()
}


def waterdepth_tiles_for_chunk(wildcards):
    """Return the waterdepth tile paths for a chunk and scenario.

    Joined with "/", not os.path.join: plain path strings are matched against
    run_eikonal's forward-slash output pattern, which backslashes break on Windows.
    """
    tile_ids = _chunk_tile_lookup[wildcards.chunk_id]
    model_outputs = config["simulation"]["model_outputs"]
    return [
        f"{model_outputs}/{tid}/results/waterdepth_{wildcards.return_period}_{wildcards.waterlevel_name}.tif"
        for tid in tile_ids
    ]


_PROTECTION_BASELINE_SLR = config["protection"]["baseline_waterlevel_name"]


# Resolved at parse time so the geoid rules can reference concrete .gfc paths.
# The EGM2008 -> GOCO06s correction is always applied.
_data_catalog = get_data_catalog(
    os.path.join(workflow.basedir, config["paths"]["hydromt_data_catalog"]), root=config["paths"]["root"]
)
_vc_cfg = config["vertical_datum_correction"]


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
)

_SIMULATION_OUTPUTS = expand(
    rules.run_eikonal.output.waterdepth,
    tile_id=TILE_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
)

# Must follow _PREPROCESS_OUTPUTS, which generate_eikonal_jobs uses.
include: "snakemake_workflow/rules/hpc_dispatch.smk"

_plot_cfg = config["postprocessing"]["plots"]
_plotting_enabled = _plot_cfg["enabled"]

# The fine merged waterdepth raster is temp() only when plots are disabled
# (see merge_chunk).
_POSTPROCESS_OUTPUTS = expand(
    rules.compute_flood_fraction_chunk.output.flood_fraction,
    chunk_id=CHUNK_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
)

if _plotting_enabled:
    _POSTPROCESS_OUTPUTS += expand(
        rules.plot_merged_results.output.waterdepth_plot, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
    )


rule preprocess:
    """Generate all per-tile, per-scenario model inputs."""
    input:
        _PREPROCESS_OUTPUTS,


rule simulate:
    """Run the eikonal flood model for all tiles, return periods and SLR scenarios."""
    input:
        _SIMULATION_OUTPUTS,


rule postprocess:
    """Merge per-tile results into chunk rasters and plots."""
    input:
        _POSTPROCESS_OUTPUTS,


rule all:
    """Run preprocessing, simulation and postprocessing."""
    input:
        _PREPROCESS_OUTPUTS + _SIMULATION_OUTPUTS + _POSTPROCESS_OUTPUTS,
