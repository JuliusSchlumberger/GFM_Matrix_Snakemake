# Method: Sensitivity Analysis for the Inner-Loop Round Cap and Outer-Loop Obstacle-Coupling Iterations

## 1. Overview

Two independent empirical calibration studies set the two round-cap
parameters: the inner-loop sweep round cap (`max_rounds`) and the
outer-loop obstacle-coupling iteration cap (`max_outer_iterations`). Both
studies sample the same population of real coastal tiles and use
production's own solver code paths directly (seeding, friction scaling,
sweep order, convergence check), not a reimplementation.

## 2. Sweep-round calibration (`max_rounds`)

**Population**: 537 wet tiles out of 571 candidates stratified by size and
geography from the ~15% of the global domain's wave-0 (hop_distance == 0)
tiles sampled for this study, RP100/SLR_0 scenario.

**Method**: per tile, seeds and friction are built exactly as in
production. Individual sweep calls are then run one at a time, recording
per-sweep metrics (max change, flooded-cell count, depth stats). Every 4th
sweep (one full round) is checked against `waterlevel_epsilon_m = 0.03` -
and the tile stops early once a round's max change falls at or below it.

**Result**: 58.5% of the 537 wet tiles strictly converge (max per-cell
change ≤ ε) within 40 rounds. Of the remaining 41.5%, 97.8% already have a
frozen flood extent by round 40 - the residual change is confined to depth
still settling in already-wet cells, not the wet/dry boundary. Production
caps `max_rounds` at 40 on this basis.

**Supporting scripts**: `calibration_studies/test_sweep_budget_calibration.py` (per-tile
sweep traces, one CSV per tile) - `calibration_studies/aggregate_wet_tiles.py`
(combines per-tile results when the calibration run is split across
parallel HPC jobs) - `calibration_studies/plot_sweep_calibration_bands.py` (single
4-panel summary figure, `sweep_calibration_bands.png` - (a) ECDF of
rounds-to-converge (the direct evidence for the `max_rounds` cutoff, via
`calibration_studies/plot_sweep_budget_convergence.py`'s `collect()`, kept only as a
shared tile-convergence detector - it has no plotting of its own), (b)-(d)
solver-residual, depth-change, and newly-flooded-cell bands vs. round, each
with a vertical line at production's `max_rounds`; a tile's own trace ends
once it round-converges, so the sample each band is computed from shrinks
at higher round numbers - real data throughout, no carried-forward values
past a tile's own trace) - `calibration_studies/plot_sweep_time_vs_size.py`
(2026-10-08, single figure, `sweep_time_vs_size.png` - real wall-clock
seconds-to-converge vs. tile size in native cells, log-log, with a power-law
fit over the converged population for HPC capacity-planning estimates;
unlike the 4-panel figure above, uses the FULL tile population, not just the
wet-tile subset, since time-to-converge is a meaningful cost for a dry tile
too; a tile that never converges within `N_COMPLETE_ROUNDS` is plotted as a
distinct marker at its own elapsed time so far - a real lower bound, not a
value the fit is computed against).

## 3. Obstacle-coupling outer-loop calibration (`max_outer_iterations`)

**Population**: the same 571-candidate pool, restricted to the 500 tiles §2
already confirmed wet at the first sweep (`wet_tiles_selected.txt`) -
obstacle coupling has nothing to iterate on for a tile with no flooding at
all.

**Method**: per tile, records an `n_outer = 0` baseline (a single
unblocked solve, no elevation-aware correction at all), then runs
production's real obstacle-coupling algorithm - static elevation
pre-filter, then iterative dynamic re-blocking with the blocked-cell set
accumulated as a running union across outer iterations - up to 15 outer
iterations, each iteration's inner solve capped at 40 rounds (the §2
result) with the same ε = 0.03. A tile is counted as converged once an
outer iteration's own newly-blocked-cell count drops below
`outer_convergence_pct` of the tile's cells.

**Result**: 370 tiles (74.0%) converge after the earliest mathematically
possible number of outer iterations (2 - the stopping check needs a
previous iteration's blocking state to compare against, so it cannot fire
any earlier); 123 more (24.6%) need 3; the remaining 7 tiles (1.4%) need 4.
No tile in this study needs more than 4 - exactly matching production's
current `max_outer_iterations` cap. When a tile's delta since the previous
iteration is small enough to stop, that iteration's own `blocked` set is
discarded - never applied, never re-solved - and the returned flood map is
the *previous* iteration's output. A tile that "needs k iterations"
therefore has exactly (k-1) rounds of dynamic blocking baked into its
output, never k.

**Does production's missing "reached" guard change the actual flood-hazard
map?** Production's blocking check (`blocked = (waterlevel_b <= dem) |
static_blocked` in `flood_model.py`) has no guard against a cell still at
the solver's unseeded sentinel (`t=99`, `waterlevel=-99`) - unlike this
calibration script's own copy of the same logic, which adds `& reached`
(`reached = t_b[1:, 1:] < 99.0`) specifically so "not reached yet" can't be
misread as "definitely dry." This is a genuine, verified discrepancy
between the calibration script and what production actually runs.

Tested directly: ran production's real `flood_model.flood_depth_dense` (no
guard) against a faithful copy of its own outer loop with the guard added,
on several real tiles - including the most extreme case found in this
study (99.98% of cells "unreached" at the first outer iteration) and
tiles that need the full 4 outer iterations to converge. Result: zero
difference in every case - identical wet-cell count, identical wet-cell
set, max depth difference exactly 0.0 m.

Why: at each tile's first outer iteration, 92.5-99.0% of unreached cells
are already `static_blocked` (`dem > max_waterlevel`) - correctly excluded
for a reason wholly independent of the guard. But that leaves a real,
non-negligible population where the guard's own carve-out actually applies
(unreached, not already static-blocked): 90K-2.1M cells depending on the
tile. Despite that, the output still doesn't change, because of how
`solve_eikonal_dense`'s fast-sweep travel time behaves under added
friction: blocking a cell can only ever hold its neighbours' arrival times
the same or push them later, never earlier, so a cell unreached within
`max_rounds=40` from the *original*, mostly-unblocked friction field
(iteration 1) has no better chance of being reached within the same round
budget on a *later* iteration, guard or no guard - extra outer iterations
don't grow the round budget, each one independently re-solves capped at
40 rounds from the same seeds. A cell that stays unreached through every
outer iteration ends up with the same final classification either way:
explicitly blocked (no guard) or simply still at the sentinel water level,
which also fails `waterlevel > dem` (with the guard) - "never reached"
and "explicitly blocked" are different code paths to the same wet/dry
answer once no solve within the budget ever reaches a cell. The
discrepancy between the two code paths is real and still worth fixing for
self-documentation and for cells that genuinely get reached on a later
iteration, but has not been shown to affect any real output on the tiles
tested.

**Supporting scripts**: `calibration_studies/test_obstacle_coupling_calibration.py`
(per-tile outer-iteration traces) - `calibration_studies/plot_obstacle_coupling_calibration_bands.py`
(single 2-panel summary figure, `obstacle_coupling_calibration_bands.png` -
(a) converged-vs-still-active tile counts per outer iteration (stacked
bar), (b) `pct_newly_blocked` - the literal stopping-criterion metric -
vs. outer iteration with a horizontal line at `outer_convergence_pct`,
each panel also marking production's `max_outer_iterations`; a tile's own
trace ends once it converges, so the sample each band is computed from
shrinks at higher iteration numbers).