# ESP/FRA/NOR Model-Parameter Calibration Sweep — Design Doc

Status: **superseded/implemented (2026-09) by the HPC-runnable version of this sweep.** This doc's local-execution design (a Python driver script calling `snakemake` directly per parameter combination) was extended to run on Hydrax via the real SLURM dispatch machinery, with per-country return periods (RP100 for ESP/FRA, RP250 for NOR — COAST-RP has no native RP200 field) and a reusable scenario-config mechanism. The real implementation lives in `snakemake_workflow/scripts/calibration/` (`select_country_tiles.py`, `build_run_config.py`, `aggregate_calibration_results.py`) and `snakemake_workflow/config/calibration/` (group scenarios + sweep-point deltas) — most of the architectural facts/risks below (hop_distance closure requirement, output-path isolation, friction lever design) still apply and were carried forward unchanged; the tile-closure sizes were measured for real (78 tiles for ESP+FRA, 58 for NOR, zero pulled in as pure dependency beyond the benchmark-region core).

## Context

The core flood model (an in-process Fast Sweeping Method eikonal solver, `src/eikonal.py`/`flood_model.py`, invoked per-tile by `rule run_aqueduct`) currently runs with fixed default parameters — `default_friction`, `max_rounds` (eikonal round cap), `obstacle_coupling.max_outer_iterations`, `waterlevel_epsilon_m` (convergence tolerance) — that were never empirically checked against real benchmark HR/FAR/CSI. We already have real validation benchmarks and a working HR/FAR/CSI pipeline for three countries (ESP, FRA, NOR — see `validation/validate_country.py`). The goal is a controlled sensitivity study: rerun the model for just the tiles needed to correctly simulate ESP/FRA/NOR, sweep each parameter independently around its default, and see which ones actually move HR/FAR/CSI — without touching production data/config.

**Decisions already made (via user clarification):**
- Sweep design: **one-factor-at-a-time (OFAT)** — vary one parameter at a time around current defaults, not a full factorial grid. Cheaper; standard first pass for sensitivity screening.
- Friction lever: **add a `friction_scale_factor` code change** (small, in `aqueduct_runner.py`) rather than relying solely on the weak `default_friction` fallback (which only affects cells with no land-use class — most cells get friction from a land-use-derived raster instead).
- Tile scope: **full correctness** — include the transitive `hop_distance` dependency closure (upstream lower-hop neighbour tiles needed to correctly seed hinterland tiles), not just tiles inside the three countries.

## Key architectural facts established (real, verified against code)

1. **Model invocation**: `rule run_aqueduct` (`rules/simulation.smk`) → `scripts/run_aqueduct.py` → `aqueduct_runner.run_aqueduct_python` → `flood_model.flood_depth_dense` → `eikonal.solve_eikonal_dense`. Pure in-process Python/Numba — no external binary.
2. **Parameter locations** — all under `simulation.flooding` in `config.yml` (lines ~306-345):
   - `default_friction: 0.001` — fallback only, weak lever (see above).
   - `max_rounds: 12` — eikonal round cap (4 sweeps/round).
   - `waterlevel_epsilon_m: 0.03` — convergence tolerance, shared by both loops below.
   - `obstacle_coupling.enabled: true`, `max_outer_iterations: 5`, `outer_convergence_pct: 0.01` — the true "outer loop" (Kasmalkar et al. 2024 Flow-Tub mitigation).
3. **Critical water-depth threshold is NOT a solver parameter.** `flood_model.py` classifies `flood = (waterlevel > dem) & (mask != ocean_code)` — no threshold gate. The continuous depth raster (`waterdepth_{RP}_{SLR}.tif`) is always written in full; thresholding happens only downstream, in `validate_country.py`'s existing `depth_thresholds_m: [0.05, 0.10, 0.25, 0.50]` sweep. **This means the depth-threshold axis is already free** — no re-simulation needed, it's already computed per run by the existing validation code.
4. **Output naming has no parameter tag.** `{simulation.model_outputs}/{tile_id}/results/waterdepth_{return_period}_{waterlevel_name}.tif` — running the same tile/RP/SLR combo under different model parameters silently overwrites the previous run. Every parameter combination needs its own isolated output tree.
5. **`hop_distance` neighbour dependency has NO Snakemake DAG edge.** `rule run_aqueduct`'s `input:` only lists this tile's own `dem`/`mask`/`friction`/`boundaries` — the neighbour-seed lookup (`boundaries.collect_neighbor_wave_seeds`, called from `run_aqueduct.py`) is a raw runtime filesystem check (`path_ready()`) against `tile_grid[(hop_distance < this_hop) & intersects(this_geom)]`, invisible to Snakemake's scheduler. Production relies on either (a) the full-global single invocation's emergent ordering, or (b) HPC's explicit per-wave SLURM dependency barriers (`hpc_dispatch.smk`/`generate_aqueduct_jobs.py`). A missing/not-yet-built neighbour silently degrades to "no upstream flooding" rather than erroring — so **our subset run must explicitly stage waves in ascending `hop_distance` order**, mirroring the HPC pattern, or hop>=1 tiles could silently get wrong (underestimated) results.
6. **No existing tile-subsetting mechanism.** The four region/chunk-batching files found during this investigation (`run_postprocess_regions.sh`, `list_region_chunks.py`, `src/chunks.py`, `plot_overlap_region_diagnostics.py`) are already-committed but **dead code** — they import `src/regions.py`, which was deleted in an earlier rewrite (`4374c62`) before these files were even added (`55a44c7`). Not reusable as-is (see cleanup note below). `tests/select_calibration_tiles.py` is the closest real precedent (hand-rolled tile-selection script for an earlier solver-calibration study), restricted to `hop_distance == 0` tiles — but a code comment there claiming `obstacle_coupling` "isn't usable with explicit seed cells (hop>=1) yet" is now **stale**: `flood_model.py` line ~352 confirms `obstacle_coupling=True` has worked with the seeded (hop>=1) path since 2026-08.
7. **Paths are templated off `paths.root`** (`config.yml`, e.g. `simulation.model_outputs: "{root}/model_outputs"`, `postprocessing.merged_outputs: "{root}/merged_results"`, `validation.output_dir: "{root}/validation"`), with `config_local.yml` as the established per-machine override mechanism. **Do not blindly override `paths.root` wholesale** — real read-only inputs (tile grid, DEM, boundary conditions, data catalogs) are also resolved relative to it, so a blanket override would break input reads. Isolate calibration runs by overriding only the specific WRITE-side keys (`simulation.model_outputs`, `postprocessing.merged_outputs`, `validation.output_dir`, and any preprocessing-output paths reused unchanged from the base run) into a dedicated calibration subtree, while leaving all input-data paths untouched.

## Related cleanup (do first, separately)

**Resolved (verified 2026-09-18)** — this investigation surfaced dead/broken code that predated and was unrelated to this plan: the four files in point 6 above, which referenced a module (`src/regions.py`) and Snakemake rules (`simulate_region`, `postprocess_region`) that no longer existed. They have since been removed from the repository (confirmed absent, 2026-09-18) — no action needed.

## Implementation plan

### 1. `snakemake_workflow/tests/calibration/select_country_tiles.py`
- Loads the tile grid (`config["tile_grid"]["path"]`) and selects the "core" set: tiles overlapping ESP/FRA/NOR's benchmark regions (reuse the `regions:` bboxes already defined per-country in `data_catalog_validation.yml`, cross-checked against real territory via the WRI geogunit/ISO lookup already used by `validation.read_country_mask`/`load_iso_lookup`).
- Computes the transitive `hop_distance` closure: for any core tile with `hop_distance >= 1`, add every tile satisfying the **exact same query `run_aqueduct.py` itself uses** (`hop_distance < this_tile.hop_distance & geometry.intersects(this_geom)`), repeat until fixed point (a newly-added lower-hop tile may itself be hop>=1 and need further upstream neighbours).
- Reports: core count vs. full closure count, breakdown by `hop_distance` and by "inside the 3 countries" vs. "pulled in as a pure dependency" — flag prominently if the closure is much larger than the core (e.g. >2x), since this is a genuine unknown until computed for real.
- Writes `tile_ids_by_wave.json` (`{hop_distance: [tile_id, ...]}`) for the sweep driver to consume, ensuring wave-ordered execution.
- **This script's output is a hard gate**: report the real closure size to the user before committing to the full OFAT sweep, since cost scales with it.

### 2. Friction-scale-factor code change
- Add `simulation.flooding.friction_scale_factor: 1.0` (default no-op) to `config.yml`.
- Apply as a multiplier where the friction raster is loaded in `aqueduct_runner.run_aqueduct_python`, right before it's passed to `flood_depth_dense` — **not** by regenerating `compute_friction.py`'s own preprocessing output. This keeps the (expensive, land-use-derived) friction raster computed once and reused unscaled across every combination that varies a *different* parameter — only the friction-sweep combinations themselves pay for the scaling, and that's a cheap runtime multiply, not a re-preprocess.

### 3. `snakemake_workflow/tests/calibration/run_calibration_sweep.py`
- Defines the OFAT grid: `friction_scale_factor` (e.g. 0.5/1/2), `max_rounds` (e.g. 4/8/12/20), `obstacle_coupling.max_outer_iterations` (e.g. 1/3/5/10, plus an `enabled: false` baseline), `waterlevel_epsilon_m` (e.g. 0.01/0.03/0.10) — each varied independently, everything else held at current defaults.
- Per combination: deep-copies the base config, patches `simulation.flooding.*` (the swept parameter) plus the isolated write-side output paths under `{calibration_root}/{run_tag}/`, writes a temp override YAML.
- Executes in **ascending-`hop_distance` waves** using `tile_ids_by_wave.json` — one Snakemake invocation (explicit `waterdepth_{RP100}_{SLR_0}` target paths, not the `simulate` aggregate target) per wave, waiting for each wave to fully complete before starting the next, so hop>=1 tiles always see their real upstream neighbours.
- Runs the needed postprocessing (`merge_chunk`, `compute_flood_fraction_chunk`) for touched chunks, then `validate_country.py --country ESP/FRA/NOR` against the run's own isolated `output_dir`.
- Parses the resulting `metrics_{country}_RP100_SLR_0.csv`, tags every row with `run_tag` + swept-parameter name + value, appends to a master `calibration_results.csv` at `{calibration_root}`.
- The depth-threshold axis needs no separate driver logic — `validate_country.py` already sweeps `depth_thresholds_m` per invocation, so every row already carries results for all 4 thresholds.

### 4. `snakemake_workflow/tests/calibration/plot_calibration_sensitivity.py`
- Loads `calibration_results.csv`; for each swept parameter, plots HR/FAR/CSI vs. parameter value, faceted by country and by `threshold_m` — small matplotlib script following existing `plotting.py` conventions.

## Execution ownership

- Claude builds all four scripts/changes above and smoke-tests the wiring end-to-end (1-2 tiles, 1 parameter combination, isolated scratch calibration path) to confirm nothing errors and no production path is touched.
- Given real per-tile cost and closure size are both currently unknown, and this is real, possibly substantial compute, **the actual full OFAT sweep should be run by the user on Hydrax** (matching established practice for production-scale runs) using the driver script — exact commands handed over rather than run by Claude.
- After script 1 runs for real, report the actual closure size (and a timing sample from the smoke test) so the user can approve/adjust the sweep's scope before the full grid is launched.

## Verification

1. Confirm `select_country_tiles.py`'s neighbour-closure query is character-for-character equivalent to `run_aqueduct.py`'s own candidate query — a mismatch here would silently produce an incomplete or incorrect tile set.
2. Smoke-test the override-config mechanism: run one tile through simulate → postprocess → validate into a scratch calibration path at **default** parameter values, and confirm its `waterdepth`/metrics output is identical to a normal run — proves the override plumbing itself introduces no behavior change before trusting any swept-parameter result.
3. Before any real invocation, diff the generated override config against the base config and confirm only the intended write-side keys changed, and that none of the new values collide with real production paths.
4. After a real (even partial) sweep, spot-check one `calibration_results.csv` row against `validate_country.py`'s own printed summary for that run, to confirm the aggregation parses correctly.

## Known risks / open items to flag to the user

- Full `hop_distance` closure size for ESP/FRA/NOR is genuinely unknown until script 1 runs for real — could be a modest border effect or could expand further; it's a hard gate, not an assumption.
- No existing documented per-tile runtime numbers anywhere in the repo — real sweep cost is unknown until measured via the smoke test.
- `friction_scale_factor`/threshold sweeps are cheap (no full solver re-run, or none at all for thresholds); `max_rounds`/`obstacle_coupling`/`waterlevel_epsilon_m` sweeps each require a full new eikonal solve per tile — the expensive, unavoidable part of the study.
- The wave-staged local execution pattern (explicit per-hop-distance Snakemake invocations) is a new usage pattern for this codebase's local (non-HPC) path — needs real verification, since production has never relied on it locally before (only on HPC via SLURM barriers).
