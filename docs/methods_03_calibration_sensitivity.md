# Method: Sensitivity Analysis for the Inner-Loop Round Cap and Outer-Loop Obstacle-Coupling Iterations

## 1. Overview

Two independent empirical calibration studies set the two round-cap
parameters in `simulation.flooding` (`config.yml`): the inner-loop sweep
round cap (`max_rounds`, §4.4 of `methods_02_flood_depth.md`) and the
outer-loop obstacle-coupling iteration cap (`obstacle_coupling.
max_outer_iterations`, §4.4a). Both studies sample the same population of
260 real coastal tiles and use production's own solver code paths
directly (seeding, friction scaling, sweep order, convergence check), not
a reimplementation.

## 2. Sweep-round calibration (`max_rounds`)

**Population**: 260 wave-0 (`hop_distance == 0`) tiles, ~10% of the ~2445
wave-0 tiles in the global domain, RP100/SLR_0 scenario. Selected by
`tests/select_calibration_tiles.py` from an over-provisioned ~300-tile
candidate pool, so a tile found dry at the first sweep can be replaced
without re-sampling.

**Method** (`tests/test_sweep_budget_calibration.py`): per tile, seeds and
friction are built exactly as in production (`flood_model.coastline_mask`
+ `_idw_seed_values`, `friction_scale_factor = 30`). Individual
`eikonal._dense_sweep` calls are then run in production's own
`_ORTHANT_ORDER`, one at a time, recording per-sweep metrics (max change,
flooded-cell count, depth stats). Every 4th sweep (one full round) is
checked against `waterlevel_epsilon_m = 0.03` - the same threshold and
check cadence `solve_eikonal_dense`'s own round loop uses - and the tile
stops early once a round's max change falls at or below it.

**Result** (confirmed complete run, `sweep_convergence_summary.csv`):
58.7% of the 260 tiles strictly converge (max per-cell change ≤ ε) within
40 rounds. Of the remaining 41.3%, 84% already have a frozen flood extent
by round 40 - the residual change is confined to depth still settling in
already-wet cells, not the wet/dry boundary. Production caps `max_rounds`
at 40 on this basis.

**Data**: `P:\11212688-004-global-floodmaps\modelling\calibration_260_tiles\`
- `sweep_budget/` holds one raw per-tile CSV, `figures/sweep_convergence_summary.csv`
the per-tile summary the percentage above is computed from directly.

**Supporting scripts**: `tests/test_sweep_budget_calibration.py` (per-tile
sweep traces, one CSV per tile) - `tests/aggregate_wet_tiles.py`
(combines per-tile results when the calibration run is split across
parallel HPC jobs) - `tests/plot_sweep_calibration_bands.py` (single
4-panel summary figure, `sweep_calibration_bands.png` - (a) ECDF of
rounds-to-converge (the direct evidence for the `max_rounds` cutoff, via
`tests/plot_sweep_budget_convergence.py`'s `collect()`, kept only as a
shared tile-convergence detector - it has no plotting of its own), (b)-(d)
solver-residual, depth-change, and newly-flooded-cell bands vs. round, each
with a vertical line at production's `max_rounds`; a tile's own trace ends
once it round-converges, so the sample each band is computed from shrinks
at higher round numbers - real data throughout, no carried-forward values
past a tile's own trace).

## 3. Obstacle-coupling outer-loop calibration (`max_outer_iterations`)

**Population**: the same 260-tile pool, restricted to the subset §2
already confirmed wet at the first sweep (`wet_tiles_selected.txt`) -
obstacle coupling has nothing to iterate on for a tile with no flooding at
all.

**Method** (`tests/test_obstacle_coupling_calibration.py`): per tile,
records an `n_outer = 0` baseline (a single unblocked solve, no
elevation-aware correction at all), then runs production's real
obstacle-coupling algorithm - static elevation pre-filter, then iterative
dynamic re-blocking with the blocked-cell set accumulated as a running
union across outer iterations (`methods_02_flood_depth.md` §4.4a) - up to
15 outer iterations, each iteration's inner solve capped at 40 rounds (the
§2 result) with the same ε = 0.03. A tile is counted as converged once an
outer iteration's own newly-blocked-cell count drops below
`outer_convergence_pct` of the tile's cells.

**Bug found and fixed during this study**: an earlier version of the
outer loop recomputed the blocked-cell set from scratch each iteration
instead of accumulating it as a running union. Under that version, 91% of
this study's non-converging tiles were caught in a stable, undamped
two-state oscillation between alternating blocked-cell configurations,
never settling regardless of how many outer iterations were allowed.
Production now accumulates the blocked-cell set monotonically
(`blocked = blocked | prev_blocked` in `flood_model.py`) specifically to
prevent this.

**Result, with the fix in place** (confirmed complete run, all 260/260
tiles converge, `obstacle_coupling_summary.csv`): 166 tiles (63.8%)
converge at the earliest mathematically possible outer iteration
(iteration 2 - the stopping check needs a previous iteration to compare
against, so it cannot fire any earlier); 89 more (34.2%) converge at
iteration 3; the remaining 5 tiles (1.9%) converge at iteration 4. No
tile in this study needs more than 4 outer iterations. `config.yml` caps
`max_outer_iterations` at 3 in production, covering 255/260 tiles
(98.1%) exactly; the remaining 5 are cut off one iteration short of full
convergence.

**Data**: `P:\11212688-004-global-floodmaps\modelling\calibration_260_tiles\`
- `sweep_budget/` and `obstacle_coupling/` hold one raw CSV per tile;
`figures/sweep_convergence_summary.csv` and
`figures/obstacle_coupling_summary.csv` hold the per-tile summary rows the
percentages above are computed from directly.

**Supporting scripts**: `tests/test_obstacle_coupling_calibration.py`
(per-tile outer-iteration traces) - `tests/plot_obstacle_coupling_calibration_bands.py`
(single 2-panel summary figure, `obstacle_coupling_calibration_bands.png` -
(a) converged-vs-still-active tile counts per outer iteration (stacked
bar), (b) `pct_newly_blocked` - the literal stopping-criterion metric -
vs. outer iteration with a horizontal line at `outer_convergence_pct`,
each panel also marking production's `max_outer_iterations`; a tile's own
trace ends once it converges, so the sample each band is computed from
shrinks at higher iteration numbers).

## 4. Open questions

1. **Do the `max_rounds = 40`/`max_outer_iterations = 3` conclusions hold
   for hop>=1 (hinterland) tiles?** Both studies sample only
   `hop_distance == 0` tiles. `config.yml`'s own comment on
   `obstacle_coupling` states the feature itself is "validated on both the
   wave-0 (real boundary stations) and hop>=1 (explicit seed cells)
   paths," so obstacle coupling does work for hop>=1 - but neither
   calibration study actually sampled a hop>=1 tile, so whether these
   specific round/iteration counts are still the right calibration for
   that population (seeded differently, and often smaller/simpler
   domains) is untested.
2. **Should the raw per-tile calibration data
   (`P:\...\modelling\calibration_260_tiles\`) be brought into version
   control, or otherwise archived somewhere more durable than the P:
   drive?** It currently exists only there, not in this git repository.
