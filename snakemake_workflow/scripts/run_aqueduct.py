"""Run the flood model for a single tile, return period and SLR scenario.

Runs `flood_model.flood_depth_dense` in-process (`run_aqueduct_python`,
round-based solve capped at `simulation.flooding.max_rounds`). See
`aqueduct_runner.py` for the shared helpers, and `scripts/run_aqueduct_cli.py`
for the equivalent HPC/sbatch entry point - keep the two in sync.

Wave-based hinterland forcing (2026-08): a tile's own `hop_distance`
(`tile_grid_path`, `tile_chunking.compute_run_order`) selects which of two
forcing paths this script uses - see the `hop_distance` branch below and
`rules/simulation.smk`'s own rule docstring for the full picture. Locally,
Snakemake's own DAG guarantees a hop>=1 tile's lower-hop neighbours are
already simulated before this script runs for it; on HPC, the equivalent
guarantee comes from `generate_aqueduct_jobs`'s per-wave SLURM dependency
barriers (see `hpc.md`) - either way, a genuinely not-yet-computed
neighbour is treated exactly like "no flooding there" (same real-zero
fallback as truly having no upstream flooding).
"""

import os
import sys
import time
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct_runner import (  # noqa: E402
    NO_STATIONS_REASON,
    NO_UPSTREAM_FLOODING_REASON,
    log_run_timing,
    log_skipped_tile,
    mark_tile_oom,
    print_simulation_progress,
    run_aqueduct_python,
    tile_marked_oom,
    tile_output_complete,
    write_zero_waterdepth,
)
from boundaries import collect_neighbor_wave_seeds  # noqa: E402
from config_utils import path_ready, retry_transient_io  # noqa: E402
from rasters import save_nodata_raster  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

model_outputs = snakemake.params.model_outputs  # noqa: F821
skipped_dir = os.path.join(model_outputs, "skipped_tiles")
oom_dir = os.path.join(model_outputs, "oom_tiles")
timings_dir = os.path.join(model_outputs, "run_timings")
tile_id = snakemake.wildcards.tile_id  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_name = f"{return_period}_{waterlevel_name}"
ocean_code = snakemake.params.ocean_code  # noqa: F821
river_code = snakemake.params.river_code  # noqa: F821

if tile_marked_oom(oom_dir, tile_id):
    log_skipped_tile(
        skipped_dir,
        tile_id,
        scenario_name,
        reason="tile too large (out-of-memory on a previous return period/SLR scenario for this tile)",
    )
    save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
else:
    tile_grid = load_tile_grid(snakemake.params.tile_grid_path)  # noqa: F821
    this_tile = tile_grid[tile_grid["tile_id"] == int(tile_id)]
    hop_distance = int(this_tile["hop_distance"].iloc[0])

    # Resolve this scenario's forcing: real/virtual boundary stations for a
    # wave-0 tile (own ocean edge), or direct eikonal seeds collected from
    # already-simulated, strictly-earlier-wave overlapping tile(s) for a
    # hop>=1 hinterland one. `boundaries_path`/`seed_rows`/`seed_cols`/
    # `seed_values` feed run_aqueduct_python below - exactly one set ends
    # up populated, matching flood_depth_dense's own mutual-exclusivity
    # contract.
    boundaries_path = None
    seed_rows = seed_cols = seed_values = None
    skip_reason = None

    if hop_distance == 0:
        boundaries = retry_transient_io(gpd.read_file, snakemake.input.boundaries)  # noqa: F821
        if boundaries.empty:
            skip_reason = NO_STATIONS_REASON
        else:
            boundaries_path = snakemake.input.boundaries  # noqa: F821
    else:
        this_geom = this_tile.geometry.iloc[0]
        candidates = tile_grid[
            (tile_grid["hop_distance"] < hop_distance)
            & (tile_grid["tile_id"] != int(tile_id))
            & tile_grid.geometry.intersects(this_geom)
        ]
        available = []
        for cand_id in candidates["tile_id"]:
            cand_dem = os.path.join(model_outputs, str(int(cand_id)), "inputs", "dem.tif")
            cand_mask = os.path.join(model_outputs, str(int(cand_id)), "inputs", "mask.tif")
            cand_waterdepth = os.path.join(
                model_outputs, str(int(cand_id)), "results", f"waterdepth_{scenario_name}.tif",
            )
            # A candidate whose output doesn't exist yet is simply omitted -
            # graceful degradation, since the DAG doesn't guarantee wave
            # ordering yet (see module docstring). path_ready (not bare
            # os.path.exists, which swallows I/O errors and returns False
            # without ever retrying) so a transient P:\ blip doesn't get
            # mistaken for "neighbour hasn't run yet".
            if path_ready(cand_dem) and path_ready(cand_mask) and path_ready(cand_waterdepth):
                available.append((cand_dem, cand_mask, cand_waterdepth))
        seed_rows, seed_cols, seed_values = collect_neighbor_wave_seeds(snakemake.input.dem, available)  # noqa: F821
        if len(seed_rows) == 0:
            skip_reason = NO_UPSTREAM_FLOODING_REASON

    if skip_reason is not None:
        log_skipped_tile(skipped_dir, tile_id, scenario_name, reason=skip_reason)
        write_zero_waterdepth(snakemake.input.dem, snakemake.output.waterdepth)  # noqa: F821
    else:
        start = time.perf_counter()
        succeeded = False
        obstacle_coupling_diagnostics = None

        flooding_config = snakemake.params.flooding_config  # noqa: F821
        oc_config = flooding_config.get("obstacle_coupling", {})
        try:
            obstacle_coupling_diagnostics = run_aqueduct_python(
                snakemake.input.dem, snakemake.input.mask, snakemake.input.friction,  # noqa: F821
                snakemake.output.waterdepth,  # noqa: F821
                resolution=flooding_config["resolution"], k=flooding_config["knn"],
                variable=waterlevel_name,
                boundaries_path=boundaries_path,
                seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                ocean_code=ocean_code, river_code=river_code,
                obstacle_coupling=oc_config.get("enabled", False),
                max_outer_iterations=oc_config.get("max_outer_iterations", 5),
                max_rounds=flooding_config["max_rounds"],
                outer_convergence_pct=oc_config.get("outer_convergence_pct", 0.01),
            )
        except MemoryError:
            mark_tile_oom(oom_dir, tile_id, reason="MemoryError in flood_depth_dense - tile too large")
            log_skipped_tile(skipped_dir, tile_id, scenario_name, reason="MemoryError - tile too large")
            save_nodata_raster(snakemake.input.dem, snakemake.output.waterdepth, snakemake.params.raster_config)  # noqa: F821
        else:
            succeeded = True

        if succeeded:
            log_run_timing(
                timings_dir, tile_id, return_period, waterlevel_name,
                elapsed_s=time.perf_counter() - start,
                obstacle_coupling_diagnostics=obstacle_coupling_diagnostics,
            )

# Only a per-tile milestone (not printed for every scenario job): once this
# tile's own output count reaches n_scenarios_per_tile, do the one O(n_tiles)
# scan across the whole grid and report the running totals.
n_scenarios_per_tile = snakemake.params.n_scenarios_per_tile  # noqa: F821
if tile_output_complete(model_outputs, tile_id, n_scenarios_per_tile):
    tile_ids = load_tile_grid(snakemake.params.tile_grid_path)["tile_id"].astype(int).tolist()  # noqa: F821
    print_simulation_progress(model_outputs, oom_dir, skipped_dir, tile_ids, n_scenarios_per_tile)
