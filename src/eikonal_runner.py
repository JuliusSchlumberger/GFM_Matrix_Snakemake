"""Functions for running the flood model.

`run_tile_scenario` is the per-tile skip/OOM/placeholder/solve logic shared by
both entry points, `snakemake_workflow/scripts/run_eikonal.py` (Snakemake rule) and
`snakemake_workflow/scripts/run_eikonal_cli.py` (HPC sbatch CLI), so they can't
drift apart; it calls `run_eikonal_python` (in-process `flood_model.flood_depth_dense`).
"""

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio

from boundaries import collect_neighbor_wave_seeds
from config_utils import path_ready, retry_transient_io
from flood_model import WATERLEVEL_EPSILON_M, flood_depth_dense
from rasters import (
    decode_dem_cm,
    decode_friction_int16,
    decode_waterlevel_cm,
    save_nodata_raster,
    save_waterdepth_raster,
)
from tiles import load_tile_grid


def run_eikonal_python(
    dem_path: str | Path,
    mask_path: str | Path,
    friction_path: str | Path,
    output_path: str | Path,
    resolution: float,
    k: int,
    variable: str,
    *,
    boundaries_path: str | Path | None = None,
    seed_rows: np.ndarray | None = None,
    seed_cols: np.ndarray | None = None,
    seed_values: np.ndarray | None = None,
    ocean_code: int = 1,
    river_code: int | None = None,
    obstacle_coupling: bool = False,
    max_outer_iterations: int = 3,
    max_rounds: int = 40,
    outer_convergence_pct: float = 0.01,
    waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M,
    friction_scale_factor: float = 1.0,
) -> dict:
    """Run `flood_model.flood_depth_dense` in-process and write its output.

    Exactly one of `boundaries_path` (wave-0: IDW-seeded from COAST-RP
    stations) or the `seed_rows`/`seed_cols`/`seed_values` triple (hop>=1:
    direct eikonal seeds) must be given - see `flood_depth_dense`'s own
    docstring, which performs the actual validation. `friction_scale_factor`
    defaults to 1.0 (a no-op) here; production's real value (9.0, from
    config.yml) is a required correction for `eikonal.py`'s discretisation
    treating friction as a cost-per-grid-step (×30, DeltaDTM's ~30m pixel)
    times a ×0.3 SFINCS-calibrated factor - not a free tuning knob. Memory
    exhaustion during the solve surfaces as a bare `MemoryError` (rare in
    practice); callers catch it to mark the tile via `mark_tile_oom`.

    Parameters
    ----------
    dem_path : str or Path
        Path to this tile's int16-cm-encoded DEM raster.
    mask_path : str or Path
        Path to this tile's land/water/boundary-code mask raster.
    friction_path : str or Path
        Path to this tile's int16-encoded friction raster.
    output_path : str or Path
        Path to write the resulting waterdepth raster to.
    resolution : float
        Pixel resolution in metres, forwarded to `flood_depth_dense`.
    k : int
        Number of nearest boundary stations used for IDW seeding.
    variable : str
        Name of the water-level variable/column to use.
    boundaries_path : str or Path or None, optional
        Boundary-stations vector file (wave-0 path). Mutually exclusive
        with `seed_rows`/`seed_cols`/`seed_values`.
    seed_rows : numpy.ndarray or None, optional
        Row indices of direct eikonal seed cells (hop>=1 path).
    seed_cols : numpy.ndarray or None, optional
        Column indices matching `seed_rows`.
    seed_values : numpy.ndarray or None, optional
        Water-level seed values in metres, matching `seed_rows`/`seed_cols`.
    ocean_code : int, default 1
        Mask code identifying ocean cells; only used on the
        `boundaries_path` path.
    river_code : int or None, optional
        Mask code identifying river cells; only used on the
        `boundaries_path` path.
    obstacle_coupling : bool, default False
        Whether to use the obstacle-coupling outer-iteration solve instead
        of the default round-based solve.
    max_outer_iterations : int, default 3
        Max outer iterations for the obstacle-coupling solve.
    max_rounds : int, default 40
        Max rounds for the round-based solve (used regardless of
        `obstacle_coupling`).
    outer_convergence_pct : float, default 0.01
        Convergence threshold (percent) for the obstacle-coupling solve.
    waterlevel_epsilon_m : float, default WATERLEVEL_EPSILON_M
        Water-level epsilon in metres (used regardless of
        `obstacle_coupling`).
    friction_scale_factor : float, default 1.0
        Multiplier applied to the decoded friction raster; 1.0 is a no-op
        here (see the note above for production's real value).

    Returns
    -------
    dict
        The `diagnostics` dict from `flood_depth_dense`.
    """
    # dem.tif/friction.tif/boundaries.gpkg's water-level column are all
    # int16-encoded on disk (see rasters.py's encode_dem_cm/
    # encode_friction_int16/encode_waterlevel_cm) - decode to float32
    # immediately on read so flood_depth_dense (and everything downstream
    # of it) sees the same float meters/friction values it always has.
    with retry_transient_io(rasterio.open, dem_path) as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
    with retry_transient_io(rasterio.open, mask_path) as src:
        mask = src.read(1)
    with retry_transient_io(rasterio.open, friction_path) as src:
        friction = decode_friction_int16(src.read(1))
    if friction_scale_factor != 1.0:
        friction = friction * friction_scale_factor

    if boundaries_path is not None:
        boundaries = retry_transient_io(gpd.read_file, boundaries_path)
        boundaries[variable] = decode_waterlevel_cm(boundaries[variable].to_numpy())
    else:
        boundaries = None

    waterdepth, diagnostics = flood_depth_dense(
        dem, mask, friction, transform,
        boundaries=boundaries,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        resolution=resolution, k=k, variable=variable,
        ocean_code=ocean_code, river_code=river_code,
        obstacle_coupling=obstacle_coupling,
        max_outer_iterations=max_outer_iterations,
        max_rounds=max_rounds,
        outer_convergence_pct=outer_convergence_pct,
        waterlevel_epsilon_m=waterlevel_epsilon_m,
    )
    save_waterdepth_raster(dem_path, waterdepth, output_path)
    return diagnostics


def estimate_eikonal_mem_mb(dem_path: str | Path, mem_estimate_cfg: dict[str, Any]) -> int:
    """Conservative peak-memory estimate (MB) for running the flood solve on one tile.

    Calibrated from 7 real tiles spanning 2.2M-140M DEM pixels (peak working
    set 800-7500MB); pixel count alone isn't an exact predictor (two tiles
    with near-identical pixel counts differed by ~900MB, likely flood-extent
    complexity), so this is deliberately generous. Used as run_eikonal's
    `resources: mem_mb` so Snakemake throttles concurrent jobs to fit a
    `--resources mem_mb=N` budget.

    Parameters
    ----------
    dem_path : str or Path
        Path to the tile's DEM raster; opened for width/height metadata
        only, not a full raster load.
    mem_estimate_cfg : dict of str to Any
        `simulation.eikonal_mem_estimate` config section, with keys
        `base_mb` and `bytes_per_pixel`.

    Returns
    -------
    int
        Estimated peak memory in MB, rounded up to the nearest 100.
    """
    with retry_transient_io(rasterio.open, dem_path) as ds:
        pixels = ds.width * ds.height
    raw_mb = mem_estimate_cfg["base_mb"] + mem_estimate_cfg["bytes_per_pixel"] * pixels / 1e6
    return int(math.ceil(raw_mb / 100.0) * 100)


def oom_marker_path(log_dir: str | Path, tile_id: str) -> Path:
    """Path to the OOM marker file for `tile_id` in `log_dir`.

    Parameters
    ----------
    log_dir : str or Path
        Directory the marker file lives in.
    tile_id : str
        The tile's `tile_id`.

    Returns
    -------
    Path
        Path to the marker file (not checked for existence).
    """
    return Path(log_dir) / f"{tile_id}.txt"


def tile_marked_oom(log_dir: str | Path, tile_id: str) -> bool:
    """Return True if `tile_id` was previously marked as too large for the flood solve.

    See `mark_tile_oom`, which writes the marker this checks for.

    Parameters
    ----------
    log_dir : str or Path
        Directory the marker file lives in.
    tile_id : str
        The tile's `tile_id`.

    Returns
    -------
    bool
        True if an OOM marker file exists for `tile_id`.
    """
    return oom_marker_path(log_dir, tile_id).exists()


def mark_tile_oom(log_dir: str | Path, tile_id: str, reason: str) -> None:
    """Record that `tile_id` ran out of memory in the flood solve.

    Once marked, other (return_period, waterlevel_name) scenarios for this
    `tile_id` skip the solve entirely (see `tile_marked_oom`), since they
    would run out of memory for the same reason (tile size).

    Parameters
    ----------
    log_dir : str or Path
        Directory to write the marker file to. Created if missing.
    tile_id : str
        The tile's `tile_id`.
    reason : str
        Short human-readable reason the tile was marked.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    oom_marker_path(log_dir, tile_id).write_text(reason)


def log_skipped_tile(log_dir: str | Path, tile_id: str, scenario_name: str, reason: str) -> None:
    """Record that a tile/scenario was skipped instead of being run through the eikonal solver.

    Writes one marker file per (tile_id, scenario_name) to `log_dir`, named
    `{tile_id}_{scenario_name}.txt`, so that skipped tiles can later be
    looked up by `tile_id` (e.g. to plot a map of skipped tiles against the
    tile grid). Writing one file per job avoids concurrent-write issues
    between parallel Snakemake jobs.

    Parameters
    ----------
    log_dir : str or Path
        Directory to write the marker file to. Created if missing.
    tile_id : str
        The tile's `tile_id`.
    scenario_name : str
        The `{return_period}_{waterlevel_name}` scenario identifier.
    reason : str
        Short human-readable reason the tile/scenario was skipped.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    (log_dir / f"{tile_id}_{scenario_name}.txt").write_text(reason)


# Substring written into a skipped_tiles/ marker by run_eikonal.py's
# `boundaries.empty` branch - distinguishes a tile with no water level
# boundary stations (never eligible to run, not a failure) from an
# OOM-driven skip when scanning skipped_tiles/ after the fact.
NO_STATIONS_REASON = "no water level boundary stations within tile"

# Substring written into a skipped_tiles/ marker by run_eikonal.py's
# hop_distance >= 1 branch when boundaries.collect_neighbor_wave_seeds
# finds no seed cells at all (2026-08 - no overlapping earlier-wave
# neighbour available yet, none overlap, or none have any flooding in the
# overlap for this scenario). Same "never eligible to run, not a failure"
# category as NO_STATIONS_REASON, just the hinterland-tile equivalent.
NO_UPSTREAM_FLOODING_REASON = "no non-zero water level found in any available overlapping earlier-wave tile"


def write_zero_waterdepth(dem_path: str | Path, output_path: str | Path) -> None:
    """Write a real, genuinely-computed all-zero waterdepth raster.

    For a tile confidently known to have no flooding (no boundary stations
    for a wave-0 tile, or no non-zero water level in any overlapping
    earlier-wave tile for a hop>=1 one - see NO_STATIONS_REASON/
    NO_UPSTREAM_FLOODING_REASON). Deliberately distinct from
    `rasters.save_nodata_raster` (used for OOM/too-large skips): a NODATA
    tile is treated by `merge.merge_tile_rasters_chunk` as "not computed"
    and ignored, but "confidently zero" is a real answer, not an unknown -
    writing it as nodata risks a real gap in merged output. Reuses
    `rasters.save_waterdepth_raster` unchanged rather than adding a
    parallel writer.

    Parameters
    ----------
    dem_path : str or Path
        Path to the tile's DEM raster (read for shape only).
    output_path : str or Path
        Path to write the all-zero waterdepth raster to.
    """
    with retry_transient_io(rasterio.open, dem_path) as src:
        shape = (src.height, src.width)
    save_waterdepth_raster(dem_path, np.zeros(shape, dtype=np.float32), output_path)


OOM_PREMARKED_REASON = "tile too large (out-of-memory on a previous return period/SLR scenario for this tile)"


def _stage_file_locally(src_path: str) -> str:
    """Copy `src_path` to node-local scratch and return the local copy's path.

    Added 2026-10-14: on a Hydrax compute node, GDAL/OGR's SQLite lock acquisition - needed
    just to OPEN a GeoPackage, even read-only, for a single non-concurrent read - can hang
    indefinitely on P:\\. A plain content copy (`shutil.copy2`, no GDAL/SQLite) does not hit
    that lock path. The caller deletes the returned file once done with it.
    """
    stage_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gfm_staged_inputs")
    os.makedirs(stage_dir, exist_ok=True)
    dst = os.path.join(stage_dir, f"{os.getpid()}_{os.path.basename(src_path)}")
    retry_transient_io(shutil.copy2, src_path, dst)
    return dst


def run_tile_scenario(
    tile_id: str,
    return_period: str,
    waterlevel_name: str,
    *,
    dem_path: str,
    mask_path: str,
    friction_path: str,
    boundaries_path: str,
    output_path: str,
    model_outputs: str,
    preprocessing_inputs_dir: str,
    tile_grid_path: str,
    raster_config: dict[str, Any],
    flooding_config: dict[str, Any],
    ocean_code: int,
    river_code: int,
    stage_boundaries: bool = False,
) -> dict[str, Any]:
    """Solve, skip or placeholder one (tile, return period, SLR scenario) and write its raster.

    The shared per-tile logic of both entry points, `run_eikonal.py` (Snakemake rule) and
    `run_eikonal_cli.py` (HPC sbatch CLI), so they cannot drift apart. In order:

    - Tile pre-marked OOM: nodata placeholder (`save_nodata_raster`), skip marker.
    - Forcing: a hop-0 tile reads its own `boundaries_path` stations; a hop>=1 tile collects
      eikonal seeds from already-simulated, strictly-earlier-wave overlapping tiles
      (`boundaries.collect_neighbor_wave_seeds`). A neighbour whose output does not exist
      yet counts as "no flooding there". No stations / no seeds: zero-depth raster, skip
      marker.
    - Otherwise the real solve (`run_eikonal_python`); a MemoryError marks the tile OOM for
      all its later scenarios and writes a nodata placeholder instead.

    Run-timing records are left to the caller: the two entry points record them differently.

    Parameters
    ----------
    tile_id : str
        Tile ID (`tile_grid.path`'s `tile_id`, as a string).
    return_period : str
        Return period name, e.g. "RP100".
    waterlevel_name : str
        SLR scenario name, e.g. "SLR_0"; also the boundaries' water-level column.
    dem_path, mask_path, friction_path : str
        This tile's preprocessed inputs.
    boundaries_path : str
        This scenario's `boundaries_{rp}_{slr}.gpkg` (read for hop-0 tiles only).
    output_path : str
        Destination `waterdepth_{rp}_{slr}.tif`.
    model_outputs : str
        `simulation.model_outputs`: root of per-tile results and of the `skipped_tiles/` and
        `oom_tiles/` marker directories.
    preprocessing_inputs_dir : str
        Root of per-tile `inputs/` (neighbour DEM/mask lookups for hop>=1 tiles).
    tile_grid_path : str
        Tile grid with `tile_id`, `hop_distance` and geometry.
    raster_config : dict
        `raster_format` config section (nodata placeholder encoding).
    flooding_config : dict
        `simulation.flooding` config section.
    ocean_code, river_code : int
        Mask class codes (`tile_generation.ocean_code` / `river_code`).
    stage_boundaries : bool, default False
        Copy the boundaries GeoPackage to node-local scratch before opening it (HPC only;
        see `_stage_file_locally`). The copy is always removed afterwards.

    Returns
    -------
    dict
        `{"outcome": "oom_premarked"}`, `{"outcome": "skip_no_inundation", "skip_reason": ...}`,
        `{"outcome": "oom"}`, or `{"outcome": "solved", "diagnostics": <run_eikonal_python's
        dict>, "solve_elapsed_s": <seconds spent in the solve>}`.
    """
    scenario_name = f"{return_period}_{waterlevel_name}"
    skipped_dir = os.path.join(model_outputs, "skipped_tiles")
    oom_dir = os.path.join(model_outputs, "oom_tiles")

    if tile_marked_oom(oom_dir, tile_id):
        log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=OOM_PREMARKED_REASON)
        save_nodata_raster(dem_path, output_path, raster_config)
        return {"outcome": "oom_premarked"}

    tile_grid = load_tile_grid(tile_grid_path)
    this_tile = tile_grid[tile_grid["tile_id"] == int(tile_id)]
    hop_distance = int(this_tile["hop_distance"].iloc[0])

    # Exactly one forcing set ends up populated (stations path or seeds), matching
    # flood_depth_dense's mutual-exclusivity contract.
    seed_rows = seed_cols = seed_values = None
    skip_reason = None
    read_boundaries_path = boundaries_path
    local_boundaries_path = None

    try:
        if hop_distance == 0:
            if stage_boundaries:
                local_boundaries_path = _stage_file_locally(boundaries_path)
                read_boundaries_path = local_boundaries_path
            boundaries = retry_transient_io(gpd.read_file, read_boundaries_path)
            if boundaries.empty:
                skip_reason = NO_STATIONS_REASON
        else:
            this_geom = this_tile.geometry.iloc[0]
            candidates = tile_grid[
                (tile_grid["hop_distance"] < hop_distance)
                & (tile_grid["tile_id"] != int(tile_id))
                & tile_grid.geometry.intersects(this_geom)
            ]
            available = []
            for cand_id in candidates["tile_id"]:
                cand_dem = os.path.join(preprocessing_inputs_dir, str(int(cand_id)), "inputs", "dem.tif")
                cand_mask = os.path.join(preprocessing_inputs_dir, str(int(cand_id)), "inputs", "mask.tif")
                cand_waterdepth = os.path.join(
                    model_outputs, str(int(cand_id)), "results", f"waterdepth_{scenario_name}.tif",
                )
                # path_ready, not bare os.path.exists (which swallows I/O errors and returns
                # False without retrying): on the P:\ share a transient blip would otherwise be
                # indistinguishable from "neighbour hasn't run yet", turning a flooded result
                # into a wrongly-confident real zero.
                if path_ready(cand_dem) and path_ready(cand_mask) and path_ready(cand_waterdepth):
                    available.append((cand_dem, cand_mask, cand_waterdepth))
            seed_rows, seed_cols, seed_values = collect_neighbor_wave_seeds(dem_path, available)
            if len(seed_rows) == 0:
                skip_reason = NO_UPSTREAM_FLOODING_REASON

        if skip_reason is not None:
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=skip_reason)
            write_zero_waterdepth(dem_path, output_path)
            return {"outcome": "skip_no_inundation", "skip_reason": skip_reason}

        oc_config = flooding_config.get("obstacle_coupling", {})
        start = time.perf_counter()
        try:
            diagnostics = run_eikonal_python(
                dem_path, mask_path, friction_path, output_path,
                resolution=flooding_config["resolution"], k=flooding_config["knn"],
                variable=waterlevel_name,
                boundaries_path=None if hop_distance >= 1 else read_boundaries_path,
                seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                ocean_code=ocean_code, river_code=river_code,
                obstacle_coupling=oc_config.get("enabled", False),
                max_outer_iterations=oc_config.get("max_outer_iterations", 3),
                max_rounds=flooding_config["max_rounds"],
                outer_convergence_pct=oc_config.get("outer_convergence_pct", 0.01),
                waterlevel_epsilon_m=flooding_config["waterlevel_epsilon_m"],
                friction_scale_factor=flooding_config.get("friction_scale_factor", 1.0),
            )
        except MemoryError:
            mark_tile_oom(oom_dir, tile_id, reason="MemoryError in flood_depth_dense - tile too large")
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason="MemoryError - tile too large")
            save_nodata_raster(dem_path, output_path, raster_config)
            return {"outcome": "oom"}
        return {"outcome": "solved", "diagnostics": diagnostics, "solve_elapsed_s": time.perf_counter() - start}
    finally:
        if local_boundaries_path is not None:
            try:
                os.remove(local_boundaries_path)
            except OSError:
                pass


def tile_output_complete(model_outputs_dir: str | Path, tile_id: int | str, n_scenarios_per_tile: int) -> bool:
    """Return True once every (return_period, waterlevel_name) output for `tile_id` is written.

    Counts `waterdepth_*.tif` files in the tile's results dir, written by
    every branch in run_eikonal.py (genuine run, OOM fallback, no-stations
    skip, or no-upstream-flooding skip), so this is agnostic to *how* each
    scenario finished.

    Parameters
    ----------
    model_outputs_dir : str or Path
        Root directory containing per-tile model output subdirectories.
    tile_id : int or str
        The tile's `tile_id`.
    n_scenarios_per_tile : int
        Expected number of (return_period, waterlevel_name) scenarios.

    Returns
    -------
    bool
        True if the tile's results dir has at least `n_scenarios_per_tile`
        `waterdepth_*.tif` files.
    """
    results_dir = Path(model_outputs_dir) / str(tile_id) / "results"
    if not results_dir.exists():
        return False
    return sum(1 for _ in results_dir.glob("waterdepth_*.tif")) >= n_scenarios_per_tile


def tile_had_no_stations(skipped_dir: str | Path, tile_id: int | str) -> bool:
    """Return True if `tile_id` was skipped for having no boundary stations.

    A tile's boundary-station selection depends only on its bbox, not on
    return_period/waterlevel_name, so if one scenario was skipped for this
    reason every scenario for this `tile_id` was - checking one marker
    suffices.

    Parameters
    ----------
    skipped_dir : str or Path
        Directory containing `log_skipped_tile` marker files.
    tile_id : int or str
        The tile's `tile_id`.

    Returns
    -------
    bool
        True if any skip marker for `tile_id` contains `NO_STATIONS_REASON`.
    """
    for marker in Path(skipped_dir).glob(f"{tile_id}_*.txt"):
        if NO_STATIONS_REASON in marker.read_text():
            return True
    return False


def log_run_timing(
    log_dir: str | Path,
    tile_id: str,
    return_period: str,
    waterlevel_name: str,
    elapsed_s: float,
    obstacle_coupling_diagnostics: dict | None = None,
) -> None:
    """Record one (tile_id, return_period, waterlevel_name) invocation's time and outcome.

    Written as a JSON file, one per invocation, with no stdout/terminal
    logging at all by design - this is the file-based record every real HPC
    dispatch job writes. One file per (tile_id, return_period,
    waterlevel_name) avoids concurrent-write corruption between parallel
    jobs, same reasoning as `log_skipped_tile`. Called from every outcome
    branch (genuine solve, skip, OOM) except the idempotent "already done"
    early return, whose own trivial elapsed time would otherwise overwrite
    a real record an earlier invocation already wrote.

    Parameters
    ----------
    log_dir : str or Path
        Directory to write the record file to. Created if missing.
    tile_id : str
        The tile's `tile_id`.
    return_period : str
        The run's `return_period` wildcard value.
    waterlevel_name : str
        The run's `waterlevel_name` wildcard value.
    elapsed_s : float
        Wall-clock seconds for this invocation. For a genuine solve called
        from `run_eikonal.py` this covers only `run_eikonal_python`
        itself; `run_eikonal_cli.py` instead times its entire invocation
        from argument parsing onward.
    obstacle_coupling_diagnostics : dict or None, optional
        For a genuine solve, the diagnostics dict `run_eikonal_python`
        returned, merged directly into the record. For a skip/OOM call
        from `run_eikonal_cli.py`, a small dict of its own instead, e.g.
        `{"outcome": "skip_no_inundation", "skip_reason": ...}` or
        `{"outcome": "oom"}`.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    record = {
        "tile_id": tile_id,
        "return_period": return_period,
        "waterlevel_name": waterlevel_name,
        "elapsed_s": elapsed_s,
        **(obstacle_coupling_diagnostics or {}),
    }
    path = log_dir / f"{tile_id}_{return_period}_{waterlevel_name}.json"
    path.write_text(json.dumps(record))


def print_simulation_progress(
    model_outputs_dir: str | Path,
    oom_dir: str | Path,
    skipped_dir: str | Path,
    tile_ids: list[int],
    n_scenarios_per_tile: int,
) -> None:
    """Print a running tally of tile outcomes across the whole tile grid.

    Only meaningful once called for a tile that just finished all its
    scenarios (see run_eikonal.py). Cheap per call, but still O(n_tiles),
    so callers should trigger it only on that per-tile milestone, not on
    every job.

    Parameters
    ----------
    model_outputs_dir : str or Path
        Root directory containing per-tile model output subdirectories.
    oom_dir : str or Path
        Directory containing `mark_tile_oom` marker files.
    skipped_dir : str or Path
        Directory containing `log_skipped_tile` marker files.
    tile_ids : list of int
        All tile ids in the grid, used to compute the tally.
    n_scenarios_per_tile : int
        Expected number of scenarios per tile, forwarded to
        `tile_output_complete`.
    """
    n_simulated = n_oom = n_no_stations = n_incomplete = 0
    for tile_id in tile_ids:
        if not tile_output_complete(model_outputs_dir, tile_id, n_scenarios_per_tile):
            n_incomplete += 1
        elif tile_marked_oom(oom_dir, tile_id):
            n_oom += 1
        elif tile_had_no_stations(skipped_dir, tile_id):
            n_no_stations += 1
        else:
            n_simulated += 1
    print(
        f"[tile progress] {n_simulated} simulated, {n_oom} OOM'd, "
        f"{n_no_stations} no-stations, {n_incomplete} still running "
        f"(of {len(tile_ids)} total tiles)",
        flush=True,
    )
