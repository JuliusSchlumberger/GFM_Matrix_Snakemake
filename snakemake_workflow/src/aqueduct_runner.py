"""Functions for running the flood model - either engine.

`run_aqueduct` (subprocess) and `run_aqueduct_python` (in-process) are the
two engines `simulation.engine` (config.yml) can select between - see
`scripts/run_aqueduct.py` and `scripts/run_aqueduct_cli.py`, which both
branch on that config value and share the helpers here so the two entry
points (Snakemake rule vs. HPC sbatch CLI) can't drift apart.
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio

from config_utils import retry_transient_io
from flood_model import flood_depth_dense
from rasters import save_waterdepth_raster

# Substring Aqueduct's Julia runtime prints (to stdout/stderr) on an
# unhandled out-of-memory crash, e.g. in `component_indices` during the
# flood-extent connected-component filter (see `core/src/core.jl`). This is
# distinct from the `LLVM ERROR: Unable to allocate section memory!` JIT
# crash that can occur when multiple Aqueduct instances run concurrently
# (see `resources: aqueduct_runs=1`) - that one is transient/concurrency
# related, not tile-size related, so it is intentionally not matched here.
OOM_SIGNATURE = "OutOfMemoryError"


def run_aqueduct(executable_path: str | Path, config_path: str | Path) -> None:
    """Run the Aqueduct flood model executable with a given TOML configuration.

    Args:
        executable_path: Path to the compiled Aqueduct executable.
        config_path: Path to the TOML configuration file for this tile/scenario run.

    Raises:
        subprocess.CalledProcessError: If the Aqueduct executable exits with
            a non-zero status code. The captured stdout/stderr are printed
            before the error is re-raised.
    """
    try:
        result = subprocess.run(
            [str(executable_path), str(config_path)],
            text=True,
            capture_output=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise


def run_aqueduct_python(
    dem_path: str | Path,
    mask_path: str | Path,
    friction_path: str | Path,
    boundaries_path: str | Path,
    output_path: str | Path,
    resolution: float,
    k: int,
    variable: str,
) -> None:
    """Run `flood_model.flood_depth_dense` in-process and write its output.

    The "python" engine's counterpart to `run_aqueduct` (Julia subprocess) -
    same inputs/output contract, no TOML/executable involved since this
    calls the validated Python port directly.

    Raises:
        MemoryError: if the solve exhausts available memory - the Python
            dense solver's counterpart to Julia's `OutOfMemoryError` (see
            `is_oom_error`), so callers can apply the same
            oom_tiles/skipped_tiles marking regardless of engine. In
            practice far less likely than Julia's failure mode (a
            358M-cell tile that OOM'd Julia ran fine here, ~5GB peak RSS) -
            this is a dormant safety net, not an expected code path.
    """
    with retry_transient_io(rasterio.open, dem_path) as src:
        dem = src.read(1)
        transform = src.transform
    with retry_transient_io(rasterio.open, mask_path) as src:
        mask = src.read(1)
    with retry_transient_io(rasterio.open, friction_path) as src:
        friction = src.read(1)
    boundaries = retry_transient_io(gpd.read_file, boundaries_path)

    waterdepth = flood_depth_dense(
        dem, mask, friction, boundaries, transform,
        resolution=resolution, k=k, variable=variable,
    )
    save_waterdepth_raster(dem_path, waterdepth, output_path)


def estimate_aqueduct_mem_mb(dem_path: str | Path, mem_estimate_cfg: dict[str, Any]) -> int:
    """Conservative peak-memory estimate (MB) for running Aqueduct on one tile.

    Calibrated from 7 real tiles spanning 2.2M-140M DEM pixels (peak working
    set 800-7500MB). Pixel count alone isn't an exact predictor - two tiles
    with near-identical pixel counts differed by ~900MB in practice, likely
    flood-extent/connected-component complexity - so this is deliberately
    generous, not a tight fit. Used as run_aqueduct's `resources: mem_mb` so
    Snakemake's own scheduler throttles concurrent Aqueduct jobs to fit
    within a `--resources mem_mb=N` budget (see that rule's docstring).

    Args:
        dem_path: Path to the tile's DEM raster (read for width/height only -
            a metadata-only open, not a full raster load).
        mem_estimate_cfg: `simulation.aqueduct_mem_estimate` config section,
            with keys `base_mb` and `bytes_per_pixel`.

    Returns:
        Estimated peak memory in MB, rounded up to the nearest 100.
    """
    with retry_transient_io(rasterio.open, dem_path) as ds:
        pixels = ds.width * ds.height
    raw_mb = mem_estimate_cfg["base_mb"] + mem_estimate_cfg["bytes_per_pixel"] * pixels / 1e6
    return int(math.ceil(raw_mb / 100.0) * 100)


def is_oom_error(error: subprocess.CalledProcessError) -> bool:
    """Return True if a failed Aqueduct run's captured output indicates an OutOfMemoryError.

    The memory cost of `component_indices` (`core/src/core.jl`) is dominated
    by the tile's total pixel count, not by the return period or SLR water
    level, so a tile that runs out of memory for one (return_period,
    waterlevel_name) combination will do so for (nearly) every combination.
    """
    return OOM_SIGNATURE in (error.stdout or "") or OOM_SIGNATURE in (error.stderr or "")


def oom_marker_path(log_dir: str | Path, tile_id: str) -> Path:
    """Path to the OOM marker file for `tile_id` in `log_dir`."""
    return Path(log_dir) / f"{tile_id}.txt"


def tile_marked_oom(log_dir: str | Path, tile_id: str) -> bool:
    """Return True if `tile_id` was previously marked as too large for Aqueduct (see `mark_tile_oom`)."""
    return oom_marker_path(log_dir, tile_id).exists()


def mark_tile_oom(log_dir: str | Path, tile_id: str, reason: str) -> None:
    """Record that `tile_id` ran out of memory in Aqueduct.

    Once marked, other (return_period, waterlevel_name) scenarios for this
    `tile_id` skip Aqueduct entirely (see `tile_marked_oom`), since they
    would run out of memory for the same reason (tile size).

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        reason: Short human-readable reason the tile was marked.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    oom_marker_path(log_dir, tile_id).write_text(reason)


def log_skipped_tile(log_dir: str | Path, tile_id: str, scenario_name: str, reason: str) -> None:
    """Record that a tile/scenario was skipped instead of being run through Aqueduct.

    Writes one marker file per (tile_id, scenario_name) to `log_dir`, named
    `{tile_id}_{scenario_name}.txt`, so that skipped tiles can later be
    looked up by `tile_id` (e.g. to plot a map of skipped tiles against the
    tile grid). Writing one file per job avoids concurrent-write issues
    between parallel Snakemake jobs.

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        scenario_name: The `{return_period}_{waterlevel_name}` scenario identifier.
        reason: Short human-readable reason the tile/scenario was skipped.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    (log_dir / f"{tile_id}_{scenario_name}.txt").write_text(reason)


# Substring written into a skipped_tiles/ marker by run_aqueduct.py's
# `boundaries.empty` branch - distinguishes a tile with no water level
# boundary stations (never eligible to run, not a failure) from an
# OOM-driven skip when scanning skipped_tiles/ after the fact.
NO_STATIONS_REASON = "no water level boundary stations within tile"


def tile_output_complete(model_outputs_dir: str | Path, tile_id: int | str, n_scenarios_per_tile: int) -> bool:
    """Return True once every (return_period, waterlevel_name) output for `tile_id` has been written.

    Counts `waterdepth_*.tif` files in the tile's results dir (written for
    every branch in run_aqueduct.py - genuine run, OOM fallback, or
    no-stations skip - so this is agnostic to *how* each scenario finished).
    """
    results_dir = Path(model_outputs_dir) / str(tile_id) / "results"
    if not results_dir.exists():
        return False
    return sum(1 for _ in results_dir.glob("waterdepth_*.tif")) >= n_scenarios_per_tile


def tile_had_no_stations(skipped_dir: str | Path, tile_id: int | str) -> bool:
    """Return True if `tile_id` was skipped for having no boundary stations.

    A tile's boundary-station selection depends only on its bbox, not on
    return_period/waterlevel_name, so if one scenario was skipped for this
    reason every scenario for this tile_id was - checking one marker suffices.
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
    cropped_pixels: int,
    original_pixels: int,
    crop_enabled: bool,
) -> None:
    """Record one successful Aqueduct invocation's wall-clock time and pixel counts.

    Written one file per (tile_id, return_period, waterlevel_name) - same
    reasoning as `log_skipped_tile`: avoids concurrent-write corruption
    between parallel Snakemake jobs. Only called for genuine `aqueduct.exe`
    runs (not the no-stations/no-candidate/OOM skip branches in
    run_aqueduct.py), since those don't reflect solver run time at all.

    Aggregate with `analysis/summarize_run_timings.py` to evaluate
    `simulation.flood_extent_crop`'s actual speedup - compare two runs (flag
    on vs. off) by pointing it at each run's own `run_timings/` directory.

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        return_period: The run's `return_period` wildcard value.
        waterlevel_name: The run's `waterlevel_name` wildcard value.
        elapsed_s: Wall-clock seconds spent inside `run_aqueduct` (the
            subprocess call to `aqueduct.exe`), excluding this script's own
            setup/skip-check overhead.
        cropped_pixels: Pixel count of the DEM actually passed to
            `aqueduct.exe` (from `crop_info.json` - equal to
            `original_pixels` when `flood_extent_crop.enabled` is false).
        original_pixels: Pixel count of the tile's full (pre-crop) DEM, for
            computing the achieved reduction ratio.
        crop_enabled: Whether `simulation.flood_extent_crop.enabled` was
            true for this run.
    """
    log_dir = Path(log_dir)
    retry_transient_io(log_dir.mkdir, parents=True, exist_ok=True)
    record = {
        "tile_id": tile_id,
        "return_period": return_period,
        "waterlevel_name": waterlevel_name,
        "elapsed_s": elapsed_s,
        "cropped_pixels": cropped_pixels,
        "original_pixels": original_pixels,
        "crop_enabled": crop_enabled,
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
    scenarios (see run_aqueduct.py) - cheap per-call (one glob per tile_id
    plus a couple of marker-file checks), but still O(n_tiles), so callers
    should only trigger it on that per-tile milestone, not on every job.
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
