# Method: Sensitivity Analysis for the Inner-Loop Round Cap and Outer-Loop Obstacle-Coupling Iterations

## 1. Overview

Two independent empirical calibration studies set the two round-cap
parameters: the inner-loop sweep
round cap and the
outer-loop obstacle-coupling iteration cap. Both studies sample the same population of
260 real coastal tiles and use production's own solver code paths
directly (seeding, friction scaling, sweep order, convergence check), not
a reimplementation.

## 2. Sweep-round calibration (`max_rounds`)

**Population**: 260 tiles with exposure to the ocean, ~10% of the ~2445
tiles in the global domain to which that criterion applies, RP100/SLR_0 scenario. 

**Method**: per tile, seeds and
friction are built exactly as in production. Individual sweep calls are then run one at a time, recording per-sweep metrics (max change,
flooded-cell count, depth stats). Every 4th sweep (one full round) is
checked against `waterlevel_epsilon_m = 0.03` - and the tile
stops early once a round's max change falls at or below it.

**Result**:
58.7% of the 260 tiles strictly converge (max per-cell change ≤ ε) within
40 rounds. Of the remaining 41.3%, 84% already have a frozen flood extent
by round 40 - the residual change is confined to depth still settling in
already-wet cells, not the wet/dry boundary. Production caps `max_rounds`
at 40 on this basis.

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

**Method**: per tile,
records an `n_outer = 0` baseline (a single unblocked solve, no
elevation-aware correction at all), then runs production's real
obstacle-coupling algorithm - static elevation pre-filter, then iterative
dynamic re-blocking with the blocked-cell set accumulated as a running
union across outer iterations - up to
15 outer iterations, each iteration's inner solve capped at 40 rounds (the
§2 result) with the same ε = 0.03. A tile is counted as converged once an
outer iteration's own newly-blocked-cell count drops below
`outer_convergence_pct` of the tile's cells.

**Result**: 166 tiles (63.8%)
converge after the earliest mathematically possible number of inner
solves (2 - the stopping check needs a previous solve's blocking state to
compare against, so it cannot fire any earlier); 89 more (34.2%) need 3
solves; the remaining 5 tiles (1.9%) need 4. No tile in this study needs
more than 4. When a tile's
delta since the previous solve is small enough to stop, that
iteration's own `blocked` set is discarded - never applied, never
re-solved - and the returned flood map is the *previous* solve's output.
A tile that "needs k solves" therefore has exactly (k-1) rounds of dynamic
blocking baked into its output, never k.

**Does production's missing "reached" guard change the actual flood-hazard
map?** Production's blocking check (`blocked = (waterlevel_b <= dem) |
static_blocked` in `flood_model.py`) has no guard against a cell still at
the solver's unseeded sentinel (`t=99`, `waterlevel=-99`) - unlike this
calibration script's own copy of the same logic, which adds `& reached`
(`reached = t_b[1:, 1:] < 99.0`) specifically so "not reached yet" can't be
misread as "definitely dry." This was a genuine, verified discrepancy
between the calibration script and what production actually runs.

Tested directly (2026-09-28): ran production's
real `flood_model.flood_depth_dense` (no guard) against a faithful copy of
its own outer loop with the guard added, on 4 real tiles under production
settings at the time of the test (`max_rounds=40`, `max_outer_iterations=3`
- since bumped to 4, see above; `outer_convergence_pct=0.01`,
`waterlevel_epsilon_m=0.03`) - including tile 1490 (99.98% of cells
"unreached" at the first outer iteration, the most extreme case found
anywhere in the 260-tile study) and 3 of the 5 tiles that need a 4th solve
to converge. Result: zero difference in every case - identical wet-cell
count, identical wet-cell set, max depth difference exactly 0.0 m.

Confirmed directly why (2026-09-28): at each tile's first outer iteration,
92.5-99.0% of unreached cells are already `static_blocked` (`dem >
max_waterlevel`) - correctly excluded for a reason wholly independent of
the guard. But that leaves a real, non-negligible population where the
guard's own carve-out actually applies (unreached, not already
static-blocked): 90K-2.1M cells depending on the tile, not negligible.
Despite that, the output still doesn't change, because of how
`solve_eikonal_dense`'s fast-sweep travel time behaves under added
friction: blocking a cell can only ever hold its neighbours' arrival times
the same or push them later, never earlier, so a cell unreached within
`max_rounds=40` from the *original*, mostly-unblocked friction field
(iteration 1) has no better chance of being reached within the same
round budget on a *later* iteration, guard or no guard - extra outer
iterations don't grow the round budget, each one independently re-solves
capped at 40 rounds from the same seeds. A cell that stays unreached
through every outer iteration ends up with the same final classification
either way: explicitly blocked (no guard) or simply still at the sentinel
water level, which also fails `waterlevel > dem` (with the guard) - "never
reached" and "explicitly blocked" are different code paths to the same
wet/dry answer once no solve within the budget ever reaches a cell. The
discrepancy between the two code paths is real and still worth fixing for
self-documentation and for cells that genuinely get reached on a later
iteration, but has not been shown to affect any real output on the tiles
tested.

**Supporting scripts**: `tests/test_obstacle_coupling_calibration.py`
(per-tile outer-iteration traces) - `tests/plot_obstacle_coupling_calibration_bands.py`
(single 2-panel summary figure, `obstacle_coupling_calibration_bands.png` -
(a) converged-vs-still-active tile counts per outer iteration (stacked
bar), (b) `pct_newly_blocked` - the literal stopping-criterion metric -
vs. outer iteration with a horizontal line at `outer_convergence_pct`,
each panel also marking production's `max_outer_iterations`; a tile's own
trace ends once it converges, so the sample each band is computed from
shrinks at higher iteration numbers).