"""Snakemake wrapper running eikonal_runner.run_tile_scenario for one tile scenario.

A not-yet-computed hop>=1 neighbour fails the job without output. The rule does
not declare neighbour outputs as inputs, so run multi-hop grids wave by wave
(as run_delta_floodmaps.py and the HPC wave barriers do) or rerun until done.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eikonal_runner import (  # noqa: E402
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

# Locally only a genuine solve gets a run-timing record.
if result["outcome"] == "solved":
    log_run_timing(
        os.path.join(model_outputs, "run_timings"), tile_id, return_period, waterlevel_name,
        elapsed_s=result["solve_elapsed_s"],
        obstacle_coupling_diagnostics=result["diagnostics"],
    )

# Per-tile milestone only, to keep the O(n_tiles) grid scan infrequent.
n_scenarios_per_tile = snakemake.params.n_scenarios_per_tile
if tile_output_complete(model_outputs, tile_id, n_scenarios_per_tile):
    tile_ids = load_tile_grid(snakemake.params.tile_grid_path)["tile_id"].astype(int).tolist()
    print_simulation_progress(
        model_outputs, os.path.join(model_outputs, "oom_tiles"), os.path.join(model_outputs, "skipped_tiles"),
        tile_ids, n_scenarios_per_tile,
    )
