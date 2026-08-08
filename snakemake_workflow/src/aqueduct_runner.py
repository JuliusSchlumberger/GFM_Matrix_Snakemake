"""Functions for running the flood model.

`run_aqueduct_python` (in-process, `flood_model.flood_depth_dense`) is
called from both `scripts/run_aqueduct.py` (Snakemake rule) and
`scripts/run_aqueduct_cli.py` (HPC sbatch CLI), which share the helpers here
so the two entry points can't drift apart.
"""

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio

from config_utils import retry_transient_io
from flood_model import flood_depth_dense
from rasters import (
    decode_dem_cm,
    decode_friction_int16,
    decode_waterlevel_cm,
    save_waterdepth_raster,
)


def run_aqueduct_python(
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
    max_outer_iterations: int = 5,
    max_rounds: int = 12,
    outer_convergence_pct: float = 0.01,
) -> dict:
    """Run `flood_model.flood_depth_dense` in-process and write its output.

    Exactly one of `boundaries_path` (wave-0: real/virtual COAST-RP
    stations, IDW-seeded onto this tile's own coastline) or the
    `seed_rows`/`seed_cols`/`seed_values` triple (hop>=1 hinterland: direct
    eikonal seeds, typically from `boundaries.collect_neighbor_wave_seeds`)
    must be given - see `flood_depth_dense`'s own docstring, which performs
    the actual validation.

    `ocean_code`/`river_code`: only used on the `boundaries_path` path (see
    `flood_model.coastline_mask`). `max_rounds` is forwarded regardless of
    `obstacle_coupling` (used by the default round-based solve either way -
    see `flood_depth_dense`'s own docstring, 2026-08); `obstacle_coupling`/
    `max_outer_iterations`/`outer_convergence_pct` are obstacle-coupling-
    specific. Defaults match `simulation.flooding` in config.yml
    (`max_rounds` top-level, the rest under `.obstacle_coupling`).

    Returns:
        The `diagnostics` dict from `flood_depth_dense` - see its docstring.

    Raises:
        MemoryError: if the solve exhausts available memory - callers apply
            the oom_tiles/skipped_tiles marking on this (see
            `mark_tile_oom`). In practice rare (a 358M-cell tile ran fine
            here at ~5GB peak RSS) - this is a dormant safety net, not an
            expected code path.
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
    )
    save_waterdepth_raster(dem_path, waterdepth, output_path)
    return diagnostics


def estimate_aqueduct_mem_mb(dem_path: str | Path, mem_estimate_cfg: dict[str, Any]) -> int:
    """Conservative peak-memory estimate (MB) for running the flood solve on one tile.

    Calibrated from 7 real tiles spanning 2.2M-140M DEM pixels (peak working
    set 800-7500MB). Pixel count alone isn't an exact predictor - two tiles
    with near-identical pixel counts differed by ~900MB in practice, likely
    flood-extent/connected-component complexity - so this is deliberately
    generous, not a tight fit. Used as run_aqueduct's `resources: mem_mb` so
    Snakemake's own scheduler throttles concurrent jobs to fit within a
    `--resources mem_mb=N` budget (see that rule's docstring).

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


def oom_marker_path(log_dir: str | Path, tile_id: str) -> Path:
    """Path to the OOM marker file for `tile_id` in `log_dir`."""
    return Path(log_dir) / f"{tile_id}.txt"


def tile_marked_oom(log_dir: str | Path, tile_id: str) -> bool:
    """Return True if `tile_id` was previously marked as too large for the flood solve (see `mark_tile_oom`)."""
    return oom_marker_path(log_dir, tile_id).exists()


def mark_tile_oom(log_dir: str | Path, tile_id: str, reason: str) -> None:
    """Record that `tile_id` ran out of memory in the flood solve.

    Once marked, other (return_period, waterlevel_name) scenarios for this
    `tile_id` skip the solve entirely (see `tile_marked_oom`), since they
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

# Substring written into a skipped_tiles/ marker by run_aqueduct.py's
# hop_distance >= 1 branch when boundaries.collect_neighbor_wave_seeds
# finds no seed cells at all (2026-08 - no overlapping earlier-wave
# neighbour available yet, none overlap, or none have any flooding in the
# overlap for this scenario). Same "never eligible to run, not a failure"
# category as NO_STATIONS_REASON, just the hinterland-tile equivalent.
NO_UPSTREAM_FLOODING_REASON = "no non-zero water level found in any available overlapping earlier-wave tile"


def write_zero_waterdepth(dem_path: str | Path, output_path: str | Path) -> None:
    """Write a real, genuinely-computed all-zero waterdepth raster.

    For a tile confidently known to have no flooding (no boundary stations
    at all for a wave-0 tile, or no non-zero water level in any available
    overlapping earlier-wave tile for a hop>=1 one - see NO_STATIONS_REASON/
    NO_UPSTREAM_FLOODING_REASON) - distinct from `rasters.save_nodata_raster`
    (used for OOM/too-large skips), which writes the NODATA sentinel:
    `merge.merge_tile_rasters_chunk` treats an all-nodata tile as "not
    computed" and ignores it when merging, but "confidently zero" is a real
    answer, not an unknown - writing it as nodata risks leaving a real gap
    in merged output wherever this tile is the only one covering an area.
    Reuses `rasters.save_waterdepth_raster` unchanged (a genuine computed
    result that happens to be all zeros), rather than adding a parallel
    writer.
    """
    with retry_transient_io(rasterio.open, dem_path) as src:
        shape = (src.height, src.width)
    save_waterdepth_raster(dem_path, np.zeros(shape, dtype=np.float32), output_path)


def tile_output_complete(model_outputs_dir: str | Path, tile_id: int | str, n_scenarios_per_tile: int) -> bool:
    """Return True once every (return_period, waterlevel_name) output for `tile_id` has been written.

    Counts `waterdepth_*.tif` files in the tile's results dir (written for
    every branch in run_aqueduct.py - genuine run, OOM fallback, no-stations
    skip, or no-upstream-flooding skip - so this is agnostic to *how* each
    scenario finished).
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
    obstacle_coupling_diagnostics: dict | None = None,
) -> None:
    """Record one successful flood-solve invocation's wall-clock time.

    Written one file per (tile_id, return_period, waterlevel_name) - same
    reasoning as `log_skipped_tile`: avoids concurrent-write corruption
    between parallel Snakemake jobs. Only called for genuine solve runs (not
    the no-stations/OOM skip branches in run_aqueduct.py), since those don't
    reflect solver run time at all.

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        return_period: The run's `return_period` wildcard value.
        waterlevel_name: The run's `waterlevel_name` wildcard value.
        elapsed_s: Wall-clock seconds spent inside `run_aqueduct_python`,
            excluding this script's own setup/skip-check overhead.
        obstacle_coupling_diagnostics: the dict returned by
            `run_aqueduct_python`/`flood_depth_dense`. Merged directly into
            the record so tiles that hit `max_outer_iterations` or
            `max_rounds` without converging are visible in production, not
            just in ad-hoc test scripts.
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
