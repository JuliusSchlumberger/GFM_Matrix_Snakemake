"""Run the flood model for a single tile, return period and SLR scenario.

Runs `flood_model.flood_depth_dense` in-process (`run_aqueduct_python`,
round-based solve capped at `simulation.flooding.max_rounds`). See
`aqueduct_runner.run_tile_scenario` for the per-tile logic shared with
`run_aqueduct_cli.py` (the HPC/sbatch entry point); this script only adds the
Snakemake wiring, its run-timing record and the per-tile progress line.

Wave-based hinterland forcing (2026-08): a tile's own `hop_distance`
(`tile_grid_path`, `connectivity_tiling.compute_hop_distances`) selects which of two
forcing paths is used - see the `hop_distance` branch in `run_tile_scenario` and
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aqueduct_runner import (  # noqa: E402
    log_run_timing,
    print_simulation_progress,
    run_tile_scenario,
    tile_output_complete,
)
from tiles import load_tile_grid  # noqa: E402

model_outputs = snakemake.params.model_outputs
tile_id = snakemake.wildcards.tile_id
return_period = snakemake.wildcards.return_period
waterlevel_name = snakemake.wildcards.waterlevel_name

result = run_tile_scenario(
    tile_id, return_period, waterlevel_name,
    dem_path=snakemake.input.dem, mask_path=snakemake.input.mask,
    friction_path=snakemake.input.friction, boundaries_path=snakemake.input.boundaries,
    output_path=snakemake.output.waterdepth,
    model_outputs=model_outputs,
    preprocessing_inputs_dir=snakemake.params.preprocessing_inputs_dir,
    tile_grid_path=snakemake.params.tile_grid_path,
    raster_config=snakemake.params.raster_config,
    flooding_config=snakemake.params.flooding_config,
    ocean_code=snakemake.params.ocean_code, river_code=snakemake.params.river_code,
)

# Locally only a genuine solve gets a run-timing record (elapsed = the solve itself).
if result["outcome"] == "solved":
    log_run_timing(
        os.path.join(model_outputs, "run_timings"), tile_id, return_period, waterlevel_name,
        elapsed_s=result["solve_elapsed_s"],
        obstacle_coupling_diagnostics=result["diagnostics"],
    )

# Only a per-tile milestone (not printed for every scenario job): once this
# tile's own output count reaches n_scenarios_per_tile, do the one O(n_tiles)
# scan across the whole grid and report the running totals.
n_scenarios_per_tile = snakemake.params.n_scenarios_per_tile
if tile_output_complete(model_outputs, tile_id, n_scenarios_per_tile):
    tile_ids = load_tile_grid(snakemake.params.tile_grid_path)["tile_id"].astype(int).tolist()
    print_simulation_progress(
        model_outputs, os.path.join(model_outputs, "oom_tiles"), os.path.join(model_outputs, "skipped_tiles"),
        tile_ids, n_scenarios_per_tile,
    )
