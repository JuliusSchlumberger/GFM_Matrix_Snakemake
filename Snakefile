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
`postprocessing.chunk_size_deg` degrees.  Each chunk is merged independently
so only the handful of tiles that overlap a chunk are ever read together.
All chunks are mosaicked into a GDAL VRT for plotting.

Targets:
    preprocess:  generate all per-tile, per-scenario model inputs.
    simulate:    run the Aqueduct flood model for all tiles and SLR scenarios.
    postprocess: merge per-tile results into combined rasters and plots.
    all:         preprocess + simulate + postprocess (the default target).

Run with e.g. `snakemake all --cores 4 --resources mem_mb=8000` — the
`mem_mb` resource lets Snakemake run several Aqueduct instances concurrently
while staying within a memory budget (see `run_aqueduct`'s docstring in
simulation.smk); set `<mem_mb>` to leave headroom below total system RAM for
the OS and other processes, e.g. ~80% of physical RAM.
"""

import os
import sys
import time

import geopandas as gpd
import numpy as np
from shapely.geometry import box as _shapely_box
from snakemake.utils import min_version

min_version("7.0")

# Monkey-patch os.makedirs to retry on a transient FileNotFoundError before
# any job runs. Snakemake's own io.py calls os.makedirs(dir, exist_ok=True)
# directly (not through pathlib) to prepare each job's output directory
# before running it - with model_outputs living on the P:\ network share, a
# momentary SMB drop (idle-session timeout, brief network blip) makes the
# whole drive briefly vanish from Windows' perspective, surfacing as
# `FileNotFoundError: [WinError 3] ... 'P:/'` even though the path is fine
# moments later. Snakemake doesn't retry this itself, so one blip anywhere
# in a multi-hour run previously aborted the whole invocation. Patching the
# shared os module here (rather than editing Snakemake's installed package,
# which would be wiped out by any env rebuild or Snakemake upgrade) makes
# every os.makedirs call in the process retry-safe, including Snakemake's
# own internals - `os.makedirs`'s own leaf-level `mkdir(name, mode)` call
# resolves against `os`'s module globals at call time, so this is visible
# to every caller already holding a reference to the `os` module (anything
# using `Path.mkdir()` instead has its own separate code path and is NOT
# covered by this patch).
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

sys.path.insert(0, os.path.join(workflow.basedir, "snakemake_workflow", "src"))
from aqueduct_runner import estimate_aqueduct_mem_mb  # noqa: E402
from config_utils import _expand_paths, get_data_catalog  # noqa: E402
from regions import assign_regions  # noqa: E402

configfile: "snakemake_workflow/config/config.yml"

# Machine-local overrides (git-ignored, optional).  Create
# snakemake_workflow/config/config_local.yml to override any key without
# touching the committed config.yml — most commonly `paths.root`,
# `paths.code_root`, and `paths.aqueduct_root` for machines with different
# storage layouts. Every standalone (non-Snakemake) entry-point script
# honors this same file via config_utils.load_config(), which mirrors this
# exact merge-then-expand sequence - keep the two in sync if this ever changes.
_LOCAL_CFG = "snakemake_workflow/config/config_local.yml"
if os.path.exists(_LOCAL_CFG):
    configfile: _LOCAL_CFG


# Expand {root} / {code_root} / {aqueduct_root} throughout the whole config so
# every rule and script receives absolute paths without knowing about the
# substitution scheme. aqueduct_root defaults to code_root, so it only needs
# to be set explicitly when the Aqueduct build lives somewhere else.
_PATH_SUBS = {
    "root": config["paths"].get("root", ""),
    "code_root": config["paths"].get("code_root", config["paths"].get("root", "")),
    "aqueduct_root": config["paths"].get("aqueduct_root", config["paths"].get("code_root", config["paths"].get("root", ""))),
}
# processed_inputs_dir itself needs root/code_root/aqueduct_root expanded
# first (it's declared as "{root}/processed_inputs"), so it's added as a
# second substitution pass rather than the single dict above - any config
# value using "{processed_inputs_dir}" (e.g. boundary_conditions.
# waterlevel_nc_dir) needs this to resolve to a real path at all.
_PATH_SUBS["processed_inputs_dir"] = _expand_paths(config["paths"]["processed_inputs_dir"], _PATH_SUBS)
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
    ever needed) so never hits this.
    """
    tile_ids = _chunk_tile_lookup[wildcards.chunk_id]
    model_outputs = config["simulation"]["model_outputs"]
    return [
        f"{model_outputs}/{tid}/results/waterdepth_{wildcards.return_period}_{wildcards.waterlevel_name}.tif"
        for tid in tile_ids
    ]


_PROTECTION_BASELINE_SLR = config["protection"]["baseline_waterlevel_name"]


# ── DeltaDTM vertical datum correction (rules compute_geoid_offset_raster /
#    extract_dem in preprocessing.smk) ── the data catalog is resolved once
#    at parse time, same as _tile_gdf above, so those rules' input:/output:
#    can reference concrete .gfc paths rather than re-querying the catalog
#    per invocation.
_data_catalog = get_data_catalog(
    os.path.join(workflow.basedir, config["paths"]["hydromt_data_catalog"]), root=config["paths"]["root"]
)
_vc_cfg = config.get("vertical_datum_correction", {})
_vertical_datum_correction_enabled = _vc_cfg.get("enabled", False)


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

# Geographic grouping of `simulate`'s targets (see src/regions.py and rule
# simulate_region below) - orchestration only, so a real-world area's worth
# of tiles can be requested and become analyzable together, instead of an
# arbitrary index range. Has no bearing on _SIMULATION_OUTPUTS above (still
# used by `simulate`/`all` exactly as before) or on where any output lands.
_TILE_REGIONS = assign_regions(_tile_gdf, len(RETURN_PERIODS) * len(WATERLEVEL_NAMES))
REGION_TILE_IDS: dict[str, list[int]] = {}
for _tile_id, _region_name in _TILE_REGIONS.items():
    REGION_TILE_IDS.setdefault(_region_name, []).append(_tile_id)
REGION_SIMULATION_OUTPUTS = {
    _region_name: expand(
        rules.run_aqueduct.output.waterdepth,
        tile_id=_tile_ids, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
    )
    for _region_name, _tile_ids in REGION_TILE_IDS.items()
}

# generate_aqueduct_jobs (HPC dispatch) needs _PREPROCESS_OUTPUTS, so this
# include comes after it's defined above, unlike the other rule files.
include: "snakemake_workflow/rules/hpc_dispatch.smk"

_plot_cfg = config["postprocessing"]["plots"]
_plotting_enabled = _plot_cfg["enabled"]
_plot_debug_enabled = _plot_cfg.get("debug", False)

# Primary postprocessing outputs:
#   - per-chunk coarse flood-fraction rasters (all RP × SLR × chunks)
#     → consumed by standalone compute_exposure_analysis.py
#   - VRT mosaic and plot (optional, for diagnostics)
# The fine-resolution waterdepth raster is marked temp() in merge_chunk (and
# auto-deleted once compute_flood_fraction_chunk completes) ONLY when
# postprocessing.plots.enabled is false - see merge_chunk's own docstring in
# postprocessing.smk for why it must persist when plots are on.
_POSTPROCESS_OUTPUTS = expand(
    rules.compute_flood_fraction_chunk.output.flood_fraction,
    chunk_id=CHUNK_IDS, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
)

if _plotting_enabled:
    _POSTPROCESS_OUTPUTS += (
        expand(rules.plot_merged_results.output.waterdepth_plot, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
        + expand(rules.plot_overlap_continent_diagnostics.output.diagnostics, return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES)
    )
    # Per-tile overlap diagnostic maps are a debugging aid (see
    # postprocessing.plots.debug in config.yml) - not requested by default,
    # unlike the per-continent correlation/agreement plot above.
    if _plot_debug_enabled:
        _POSTPROCESS_OUTPUTS += expand(
            rules.plot_overlap_diagnostics.output.diagnostics,
            return_period=RETURN_PERIODS, waterlevel_name=WATERLEVEL_NAMES,
        )


rule preprocess:
    """Generate all per-tile, per-scenario model inputs."""
    input:
        _PREPROCESS_OUTPUTS,


rule simulate:
    """Run the Aqueduct flood model for all tiles, return periods and SLR scenarios."""
    input:
        _SIMULATION_OUTPUTS,


rule simulate_region:
    """Run the Aqueduct flood model for one geographic region's tiles only
    (see src/regions.py) - a real, analyzable slice of global coverage
    (a continent, or half of an oversized one), for requesting/tracking via
    run_simulate_regions.sh instead of an arbitrary index range.

    Orchestration only: {region} selects WHICH of simulate's own targets get
    requested in this invocation, nothing else. Every output still lands at
    exactly the same model_outputs/{tile_id}/results/waterdepth_....tif path
    `simulate` itself would produce - postprocessing (merge_chunk, plots,
    exposure analysis) reads those same paths with no awareness this
    grouping exists, and the actual per-country/per-continent exposure
    results downstream (analysis/compute_exposure_analysis.py) are computed
    from the population/geogunit rasters at global-grid resolution, entirely
    independent of - and unaffected by - which region built a given tile.

    The output is a small marker file (not a data product) purely so this
    wildcarded rule can be targeted by a concrete path on the command line;
    it lives under model_outputs/_region_done/ alongside this codebase's
    other bookkeeping-only subfolders (oom_tiles/, skipped_tiles/), not
    mixed in with any tile's actual per-tile data.
    """
    input:
        lambda wildcards: REGION_SIMULATION_OUTPUTS[wildcards.region],
    output:
        # Deliberately NOT os.path.join: on Windows it would backslash-join
        # the "_region_done"/"{region}.done" segments, which then would not
        # textually match a forward-slash path built independently by
        # list_regions.py (used to pass this exact target on the command
        # line) - same documented pitfall as waterdepth_tiles_for_chunk
        # above, which needs the same workaround for the same reason.
        touch(config["simulation"]["model_outputs"] + "/_region_done/{region}.done"),


rule postprocess:
    """Merge per-tile results into combined, multi-tile rasters and plot them, for all return periods and SLR scenarios."""
    input:
        _POSTPROCESS_OUTPUTS,


rule all:
    """Run the full workflow: preprocessing, simulation and postprocessing for all tiles, return periods and SLR scenarios."""
    input:
        _PREPROCESS_OUTPUTS + _SIMULATION_OUTPUTS + _POSTPROCESS_OUTPUTS,
