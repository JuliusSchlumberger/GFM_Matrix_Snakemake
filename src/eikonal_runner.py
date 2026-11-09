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
import rasterio.crs
import rasterio.transform
import rasterio.warp

from boundaries import (
    collect_neighbor_wave_seeds,
    static_level_inputs,
    static_station_levels,
    storm_tide_hydrograph_path,
)
from config_utils import path_ready, retry_transient_io, tile_pixel_size_m
from event_volume import DurationCorrection, load_coast_hg
from flood_model import FLOOD_MODELS, WATERLEVEL_EPSILON_M, SolveTrace, flood_depth_dense, static_level_grid
from rasters import (
    decode_dem_cm,
    decode_friction_int16,
    decode_waterlevel_cm,
    save_nodata_raster,
    save_waterdepth_raster,
)
from tiles import load_tile_grid


def run_eikonal_python(dem_path: str | Path, mask_path: str | Path, friction_path: str | Path,
                       output_path: str | Path, *args, **kwargs) -> dict:
    """Run `solve_tile`, write its depth to `output_path`, return the diagnostics."""
    waterdepth, diagnostics = solve_tile(dem_path, mask_path, friction_path, *args, **kwargs)
    save_waterdepth_raster(dem_path, waterdepth, output_path)
    return diagnostics


def flooding_solve_settings(flooding_config: dict[str, Any]) -> dict[str, Any]:
    """Return `solve_tile`'s solver settings from `simulation.flooding`."""
    # Indexed, not .get(): every config layers on config.yml, which sets all of these, so a
    # missing key is a broken config, not a reason to default.
    oc = flooding_config["obstacle_coupling"]
    return {
        "resolution": flooding_config["resolution"], "k": flooding_config["knn"],
        "idw_power": flooding_config["idw_power"],
        "obstacle_coupling": oc["enabled"], "max_outer_iterations": oc["max_outer_iterations"],
        "max_rounds": flooding_config["max_rounds"], "outer_convergence_pct": oc["outer_convergence_pct"],
        "waterlevel_epsilon_m": flooding_config["waterlevel_epsilon_m"],
        "friction_scale_factor": flooding_config["friction_scale_factor"],
        "duration_config": flooding_config["duration_correction"],
        "model": flooding_config["model"], "friction_source": flooding_config["friction_source"],
        "uniform_attenuation_m_per_km": flooding_config["uniform_attenuation_m_per_km"],
    }


def solve_wave0_tile(config: dict[str, Any], tile_id: int | str, return_period: str, waterlevel_name: str,
                     code_root: Path, trace: SolveTrace | None = None,
                     **overrides) -> tuple[np.ndarray, dict]:
    """Solve a hop-0 tile from its station boundaries as run_tile_scenario does, without writing.

    Same inputs and settings (two-component static level, correction B, the tile's pixel size);
    `overrides` replace `solve_tile` settings (the calibration studies' max_rounds, coupling,
    friction source). Returns (depth m, diagnostics).
    """
    sim = config["simulation"]
    inputs = Path(sim.get("preprocessing_inputs_dir") or sim["model_outputs"]) / str(tile_id) / "inputs"
    flooding = sim["flooding"]
    if flooding["slr_propagation"] not in SLR_PROPAGATION_MODES:
        raise ValueError(f"simulation.flooding.slr_propagation: {flooding['slr_propagation']!r} "
                         f"not in {SLR_PROPAGATION_MODES}")
    static_stations = None
    if flooding["slr_propagation"] == "two_component":
        static_stations = static_station_levels(**static_level_inputs(config), return_period=return_period,
                                                waterlevel_name=waterlevel_name)
    settings = {
        **flooding_solve_settings(flooding), "variable": waterlevel_name,
        "boundaries_path": inputs / f"boundaries_{return_period}_{waterlevel_name}.gpkg",
        "ocean_code": config["tile_generation"]["ocean_code"], "river_code": config["tile_generation"]["river_code"],
        "pixel_size_m": tile_pixel_size_m(config, str(tile_id), code_root), "static_stations": static_stations,
        "return_period": int(return_period.removeprefix("RP")),
        "hydrograph_path": str(storm_tide_hydrograph_path(config)), **overrides,
    }
    return solve_tile(inputs / "dem.tif", inputs / "mask.tif", inputs / "friction.tif", trace=trace, **settings)


def solve_tile(
    dem_path: str | Path,
    mask_path: str | Path,
    friction_path: str | Path,
    resolution: float,
    k: int,
    variable: str,
    *,
    idw_power: float,
    boundaries_path: str | Path | None = None,
    seed_rows: np.ndarray | None = None,
    seed_cols: np.ndarray | None = None,
    seed_values: np.ndarray | None = None,
    ocean_code: int = 1,
    river_code: int | None = None,
    obstacle_coupling: bool = False,
    max_outer_iterations: int,
    max_rounds: int = 40,
    outer_convergence_pct: float = 0.01,
    waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M,
    friction_scale_factor: float,
    pixel_size_m: float | None = None,
    static_stations: tuple[np.ndarray, np.ndarray] | None = None,
    duration_config: dict[str, Any] | None = None,
    return_period: int | None = None,
    hydrograph_path: str | Path | None = None,
    model: str = "eikonal",
    friction_source: str = "landcover",
    uniform_attenuation_m_per_km: float | None = None,
    trace: SolveTrace | None = None,
) -> tuple[np.ndarray, dict]:
    """Run `flood_model.flood_depth_dense` in-process; return (depth m, diagnostics).

    Inputs are int16-encoded on disk (DEM and water levels in cm) and decoded
    to float32 m. `friction_scale_factor` (production 0.25, config.yml) is a
    bare SFINCS-calibrated per-metre rate, not a tuning knob: the solver
    treats friction as cost per grid step, so it must be scaled by the
    tile's own grid step in metres (`pixel_size_m`; None = `resolution`)
    before use - `0.25 x 30` for a DeltaDTM tile (docs/methods_02). Raises
    MemoryError on OOM.
    `k` and `idw_power` set the station interpolation of both the coastline seeds and the
    static level.
    `static_stations` = (lonlat, level) from `boundaries.static_station_levels`
    switches on two-component SLR propagation; `duration_config`
    (`simulation.flooding.duration_correction`) with `enabled` switches on correction B (eikonal only);
    `return_period` (years) picks its hydrograph when the file holds one per RP;
    `hydrograph_path` (`boundaries.storm_tide_hydrograph_path`) replaces `coast_hg_path`.
    `model` is a `flood_model.FLOOD_MODELS` entry. `friction_source` "uniform" replaces the
    land-cover rate by `uniform_attenuation_m_per_km` everywhere (also the straight_line rate).
    """
    if friction_source not in FRICTION_SOURCES:
        raise ValueError(f"simulation.flooding.friction_source: {friction_source!r} not in {FRICTION_SOURCES}")
    cell_m = pixel_size_m if pixel_size_m is not None else resolution
    with retry_transient_io(rasterio.open, dem_path) as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
        crs = src.crs
    with retry_transient_io(rasterio.open, mask_path) as src:
        mask = src.read(1)
    uniform_m_per_m = None if uniform_attenuation_m_per_km is None else uniform_attenuation_m_per_km / 1000.0
    if friction_source == "uniform":
        friction = np.full(dem.shape, uniform_m_per_m * cell_m, dtype=np.float32)
    else:
        with retry_transient_io(rasterio.open, friction_path) as src:
            friction = decode_friction_int16(src.read(1))
        effective_friction_scale_factor = friction_scale_factor * cell_m
        if effective_friction_scale_factor != 1.0:
            friction = friction * effective_friction_scale_factor

    if boundaries_path is not None:
        boundaries = retry_transient_io(gpd.read_file, boundaries_path)
        boundaries[variable] = decode_waterlevel_cm(boundaries[variable].to_numpy())
    else:
        boundaries = None

    static_level = None
    if static_stations is not None:
        static_level = static_level_grid(dem.shape, transform, *static_stations, k=k, idw_power=idw_power)

    duration_correction = None
    if model == "eikonal" and duration_config is not None and duration_config["enabled"]:
        duration_correction = build_duration_correction(duration_config, dem.shape, transform, crs, return_period,
                                                        hydrograph_path)

    waterdepth, diagnostics = flood_depth_dense(
        dem, mask, friction, transform,
        boundaries=boundaries,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        resolution=cell_m, k=k, idw_power=idw_power, variable=variable,
        ocean_code=ocean_code, river_code=river_code,
        obstacle_coupling=obstacle_coupling,
        max_outer_iterations=max_outer_iterations,
        max_rounds=max_rounds,
        outer_convergence_pct=outer_convergence_pct,
        waterlevel_epsilon_m=waterlevel_epsilon_m,
        static_level=static_level,
        duration_correction=duration_correction,
        model=model,
        attenuation_m_per_m=uniform_m_per_m,
        trace=trace,
    )
    return waterdepth, diagnostics


def build_duration_correction(
    duration_config: dict[str, Any], shape: tuple[int, int], transform, crs=None, return_period: int | None = None,
    hydrograph_path: str | Path | None = None,
) -> DurationCorrection:
    """Build correction B for one tile from `simulation.flooding.duration_correction`.

    Hydrographs from `hydrograph_path`, else `coast_hg_path`; `return_period` (years) selects
    the hydrograph when the file holds one per RP.
    """
    xs, ys = rasterio.transform.xy(transform, [0, 0, shape[0] - 1, shape[0] - 1], [0, shape[1] - 1, 0, shape[1] - 1])
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    grid_crs = None if crs is None or rasterio.crs.CRS.from_user_input(crs).to_epsg() == 4326 else crs
    if grid_crs is not None:
        xs, ys = rasterio.warp.transform(grid_crs, "EPSG:4326", xs, ys)
    lonlat, series, dt_s = retry_transient_io(
        load_coast_hg, hydrograph_path or duration_config["coast_hg_path"], (min(xs), min(ys), max(xs), max(ys)),
        return_period=return_period,
    )
    return DurationCorrection(
        lonlat, series, dt_s, t_ref_s=duration_config["t_ref_h"] * 3600.0, alpha=duration_config["alpha"],
        max_multiplier=duration_config["max_multiplier"], crs=grid_crs,
    )


SLR_PROPAGATION_MODES = ("attenuated", "two_component")
FRICTION_SOURCES = ("landcover", "uniform")


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
    # On cluster compute nodes, GDAL's SQLite lock on a GeoPackage on the shared data root can hang
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
    pixel_size_m: float | None = None,
    static_level_inputs: dict[str, str] | None = None,
    hydrograph_path: str | None = None,
) -> dict[str, Any]:
    """Solve, skip or placeholder one (tile, return period, SLR scenario) and write its raster.

    Pre-marked OOM -> nodata placeholder. Hop-0 tiles use their stations;
    hop>=1 tiles seed from strictly-earlier-wave overlapping tiles. No
    stations, or no wet cell in any neighbour output -> zero raster. A missing
    neighbour dem/mask/waterdepth raises FileNotFoundError and writes nothing.
    MemoryError -> OOM marker + nodata. `stage_boundaries` copies the
    GeoPackage to node-local scratch first (HPC). `flooding_config["slr_propagation"]`
    "two_component" builds the static level from `static_level_inputs`
    (`boundaries.static_level_inputs`); "attenuated" ignores them. `pixel_size_m` is the
    tile's real DEM pixel size (a local high-res DEM source); None = flooding `resolution`.
    Timing is left to callers.
    Returns {"outcome": "oom_premarked" | "skip_no_inundation" | "oom" | "solved", ...}.
    """
    scenario_name = f"{return_period}_{waterlevel_name}"
    slr_propagation = flooding_config["slr_propagation"]
    if slr_propagation not in SLR_PROPAGATION_MODES:
        raise ValueError(f"simulation.flooding.slr_propagation: {slr_propagation!r} not in {SLR_PROPAGATION_MODES}")
    if slr_propagation == "two_component" and static_level_inputs is None:
        raise ValueError("slr_propagation 'two_component' needs static_level_inputs")
    if flooding_config["model"] not in FLOOD_MODELS:
        raise ValueError(f"simulation.flooding.model: {flooding_config['model']!r} not in {FLOOD_MODELS}")
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
            missing = []
            for cand_id in candidates["tile_id"]:
                cand_dem = os.path.join(preprocessing_inputs_dir, str(int(cand_id)), "inputs", "dem.tif")
                cand_mask = os.path.join(preprocessing_inputs_dir, str(int(cand_id)), "inputs", "mask.tif")
                cand_waterdepth = os.path.join(
                    model_outputs, str(int(cand_id)), "results", f"waterdepth_{scenario_name}.tif",
                )
                # path_ready: os.path.exists would read a shared-mount blip as "not run yet".
                cand_missing = [p for p in (cand_dem, cand_mask, cand_waterdepth) if not path_ready(p)]
                if cand_missing:
                    missing += cand_missing
                else:
                    available.append((cand_dem, cand_mask, cand_waterdepth))
            # A neighbour that has not run (or failed) is unknown, not dry: a zero
            # raster here would count as done and never be rerun. Raise without
            # writing output so a resume / rerun picks this triple up again.
            if missing:
                raise FileNotFoundError(
                    f"tile {tile_id} {scenario_name}: {len(missing)} lower-hop neighbour file(s) "
                    f"missing, nothing written: {missing}"
                )
            seed_rows, seed_cols, seed_values = collect_neighbor_wave_seeds(dem_path, available)
            if len(seed_rows) == 0:
                skip_reason = NO_UPSTREAM_FLOODING_REASON

        if skip_reason is not None:
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=skip_reason)
            write_zero_waterdepth(dem_path, output_path)
            return {"outcome": "skip_no_inundation", "skip_reason": skip_reason}

        static_stations = None
        if slr_propagation == "two_component":
            static_stations = static_station_levels(**static_level_inputs, return_period=return_period,
                                                    waterlevel_name=waterlevel_name)
        start = time.perf_counter()
        try:
            diagnostics = run_eikonal_python(
                dem_path, mask_path, friction_path, output_path,
                **flooding_solve_settings(flooding_config),
                variable=waterlevel_name,
                boundaries_path=None if hop_distance >= 1 else read_boundaries_path,
                seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                ocean_code=ocean_code, river_code=river_code,
                pixel_size_m=pixel_size_m,
                static_stations=static_stations,
                return_period=int(return_period.removeprefix("RP")),
                hydrograph_path=hydrograph_path,
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
