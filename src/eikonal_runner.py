"""Run the flood model per tile; `run_tile_scenario` is the shared entry point.

Used by both run_eikonal.py (Snakemake) and run_eikonal_cli.py (HPC).
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
    """Run `flood_model.flood_depth_dense` in-process, write depth, return diagnostics.

    Inputs are int16-encoded on disk (DEM and water levels in cm) and decoded
    to float32 m. `friction_scale_factor` (production 12.0, config.yml) is a
    required correction, not a tuning knob: the solver treats friction as
    cost per grid step (x30 for DeltaDTM's ~30 m pixel) times a x0.4
    SFINCS-calibrated factor (docs/methods_02). Raises MemoryError on OOM.
    """
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
    """Conservative peak-memory estimate (MB, rounded up to 100) for one tile solve.

    Calibrated on 7 tiles (2.2M-140M px, 800-7500 MB peak); same-size tiles
    differed by ~900 MB, hence generous. Used as Snakemake `mem_mb`.
    """
    with retry_transient_io(rasterio.open, dem_path) as ds:
        pixels = ds.width * ds.height
    raw_mb = mem_estimate_cfg["base_mb"] + mem_estimate_cfg["bytes_per_pixel"] * pixels / 1e6
    return int(math.ceil(raw_mb / 100.0) * 100)


def oom_marker_path(log_dir: str | Path, tile_id: str) -> Path:
    """Path to the OOM marker file for `tile_id` in `log_dir`."""
    return Path(log_dir) / f"{tile_id}.txt"


def tile_marked_oom(log_dir: str | Path, tile_id: str) -> bool:
    """Return True if `tile_id` has an OOM marker (see `mark_tile_oom`)."""
    return oom_marker_path(log_dir, tile_id).exists()


def mark_tile_oom(log_dir: str | Path, tile_id: str, reason: str) -> None:
    """Mark `tile_id` OOM so its remaining scenarios skip the solve."""
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    oom_marker_path(log_dir, tile_id).write_text(reason)


def log_skipped_tile(log_dir: str | Path, tile_id: str, scenario_name: str, reason: str) -> None:
    """Write a `{tile_id}_{scenario_name}.txt` skip marker (one file per job, no write races)."""
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    (log_dir / f"{tile_id}_{scenario_name}.txt").write_text(reason)


# Skip-marker substrings for "never eligible to run" (not failures), told apart
# from OOM skips when scanning skipped_tiles/.
NO_STATIONS_REASON = "no water level boundary stations within tile"

NO_UPSTREAM_FLOODING_REASON = "no non-zero water level found in any available overlapping earlier-wave tile"


def write_zero_waterdepth(dem_path: str | Path, output_path: str | Path) -> None:
    """Write an all-zero waterdepth raster for a tile known not to flood.

    Distinct from `rasters.save_nodata_raster` (OOM skips): the merge ignores
    nodata as "not computed", whereas zero is a real answer.
    """
    with retry_transient_io(rasterio.open, dem_path) as src:
        shape = (src.height, src.width)
    save_waterdepth_raster(dem_path, np.zeros(shape, dtype=np.float32), output_path)


OOM_PREMARKED_REASON = "tile too large (out-of-memory on a previous return period/SLR scenario for this tile)"


def _stage_file_locally(src_path: str) -> str:
    # On Hydrax, GDAL's SQLite lock to open a GeoPackage on P:\ can hang
    # indefinitely; a plain copy avoids it. Caller deletes the copy.
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

    Pre-marked OOM -> nodata placeholder. Hop-0 tiles use their stations;
    hop>=1 tiles seed from strictly-earlier-wave overlapping tiles (a missing
    neighbour output counts as no flooding). No stations/seeds -> zero raster.
    MemoryError -> OOM marker + nodata. `stage_boundaries` copies the
    GeoPackage to node-local scratch first (HPC). Timing is left to callers.
    Returns {"outcome": "oom_premarked" | "skip_no_inundation" | "oom" | "solved", ...}.
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
                # path_ready: os.path.exists would read a P:\ blip as "not run yet" -> false zero.
                if path_ready(cand_dem) and path_ready(cand_mask) and path_ready(cand_waterdepth):
                    available.append((cand_dem, cand_mask, cand_waterdepth))
            seed_rows, seed_cols, seed_values = collect_neighbor_wave_seeds(dem_path, available)
            if len(seed_rows) == 0:
                skip_reason = NO_UPSTREAM_FLOODING_REASON

        if skip_reason is not None:
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=skip_reason)
            write_zero_waterdepth(dem_path, output_path)
            return {"outcome": "skip_no_inundation", "skip_reason": skip_reason}

        # Indexed, not .get(): every config layers on config.yml, which sets all
        # of these, so a missing key is a broken config, not a reason to default.
        oc_config = flooding_config["obstacle_coupling"]
        start = time.perf_counter()
        try:
            diagnostics = run_eikonal_python(
                dem_path, mask_path, friction_path, output_path,
                resolution=flooding_config["resolution"], k=flooding_config["knn"],
                variable=waterlevel_name,
                boundaries_path=None if hop_distance >= 1 else read_boundaries_path,
                seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                ocean_code=ocean_code, river_code=river_code,
                obstacle_coupling=oc_config["enabled"],
                max_outer_iterations=oc_config["max_outer_iterations"],
                max_rounds=flooding_config["max_rounds"],
                outer_convergence_pct=oc_config["outer_convergence_pct"],
                waterlevel_epsilon_m=flooding_config["waterlevel_epsilon_m"],
                friction_scale_factor=flooding_config["friction_scale_factor"],
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
    """Return True once the tile has >= `n_scenarios_per_tile` waterdepth_*.tif outputs.

    Every outcome (solve, OOM, skips) writes one, so this ignores how each finished.
    """
    results_dir = Path(model_outputs_dir) / str(tile_id) / "results"
    if not results_dir.exists():
        return False
    return sum(1 for _ in results_dir.glob("waterdepth_*.tif")) >= n_scenarios_per_tile


def tile_had_no_stations(skipped_dir: str | Path, tile_id: int | str) -> bool:
    """Return True if `tile_id` was skipped for having no boundary stations.

    Station selection depends only on the bbox, so one marker covers all scenarios.
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
    """Write one JSON timing/outcome record per (tile, return period, SLR) invocation.

    Callers skip the "already done" early return so a trivial time never
    overwrites a real record. `elapsed_s` covers only the solve from
    run_eikonal.py but the whole invocation from run_eikonal_cli.py.
    `obstacle_coupling_diagnostics` is merged into the record.
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
    """Print a tally of tile outcomes across the grid; O(n_tiles), call per finished tile."""
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
