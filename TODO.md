# TODO

Open items after the cleanup on branch `cleanup` (not merged into `main`). Everything in
the code is done; what remains needs you, the HPC cluster or `<data_root>`.

## Next steps

1. **Merge the latest `cleanup` commits into `main`** (the earlier merge is done; `git log origin/main..cleanup`).
2. **On the cluster:** back up the `*_materialized.yml` files in `<code_root>/snakemake_workflow/config/`,
   `git pull` on `main`, create the gitignored site files
   (`cp sfincs_tiles/site_env.example.sh sfincs_tiles/site_env.sh` and
   `cp snakemake_workflow/config/config_hpc.example.yml snakemake_workflow/config/config_hpc.yml`;
   the examples hold the example site's values), then `python snakemake_workflow/scripts/materialize_configs.py` (the cluster
   checkout is not under `<data_root>`, so it cannot be done from the local machine). Configs with the old keys
   `slr_scenario` / `slr_scenarios` fail with KeyError.
3. **On the cluster:** `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal` once, to check the
   shell drivers (only tested locally with stubbed sbatch/python). Paths live in
   `sfincs_tiles/site_env.sh`; keep its roots in sync with `config_hpc.yml`.
4. **Cluster `gfm` env:** `mamba install -n gfm pytest=8.4.2 ruff=0.12.0`.
5. **`<data_root>` cleanup:** `sfincs_smoke_test/` is no longer used; delete it if you like. When
   calibration stage 1 runs again, pass `--overwrite` so screen / pre-flight error records cached by
   earlier runs are recomputed.
Global deltas are flood-hazard only; their CSV and figures were regenerated on 2026-10-21 from the
existing ssp126/ssp245 simulations (`flooded_area_km2` = the former unprotected value; old files in
`.cleanup_baseline/deltas_before_hazard_only/`).

Done on 2026-10-18 (outputs regenerated with the new code, values identical to the previous files;
old copies in `.cleanup_baseline/p_backup_regen/`): flood totals (cluster job), Bangkok case exposure,
delta flood extent + SLR-year column, delta design selection and `deltas_design_materialized.yml`
(design runs since removed, item 6),
England FZ3 / FZ2 metrics, calibration_500_tiles batch jobs, the validation summary tables, and the
materialized configs on this machine. The 2026-10-15 production `resolved_config.yml` was moved to
`model_outputs/hpc_jobs/resolved_config_2026-10-15.yml`.

Snakemake: the config-key rename re-triggers `cache_waterlevel_stations`, `extract_boundaries` and
`run_eikonal` the next time a scenario runs (+25 jobs for bangkok_chao_phraya). Accepted.

Renamed CLI flags (old spellings now rejected): `--outdir` -> `--out-dir`, `--figdir` -> `--fig-dir`,
`--output` / `--out-csv` -> `--out`, `--dir` -> `--timings-dir`, `--threads` -> `--workers`,
`--max-outer` -> `--max-outer-iterations`, `--inner-max-rounds` -> `--max-rounds`,
`--tiles` -> `--mask-files`, `--skip-done` / `--no-force` -> `--skip-existing`,
`--n-tiles` -> `--n-panel-tiles` (plots), `--cpus` -> `--cpus-per-task`,
`--flood-totals-cpus` -> `--flood-totals-cpus-per-task`; positional arguments of the calibration
study scripts and `build_tile_subset_by_id.py` are options now (`--out-dir`, `--n-candidates`,
`--fig-dir`, `--tile-ids`, `--tile-ids-file`, `--out`). Notebooks or scripts outside the repo that
import renamed modules (pipeline_plots, analysis_figures, exposure_grid, ...) need the same update.

## Reviewer critiques (branch `review-critiques`)

Evidence: `review_critiques/` scripts; results, figures and the summary page source in
`.cleanup_baseline/review_critiques/`.

Decided and implemented:
- **SLR propagation (critique 1a):** `simulation.flooding.slr_propagation: two_component` is the
  default. The static level S = MDT + SLR fingerprint spreads by coast connectivity without
  attenuation and acts as sea; only the storm tide on top attenuates. Bangkok tiles: 1.8-2.7x the
  attenuated flooded area from +0.5 m SLR, identical at SLR 0.
- **Duration correction B (critique 3):** `simulation.flooding.duration_correction` on, alpha 1,
  t_ref 12 h, COAST-HG shape of the nearest station per cell, second solve (~2x runtime). On SFINCS
  RP100 tiles: pooled CSI 0.742 -> 0.763 on 498 random tiles (median tile unchanged; the gain comes
  from a minority of overflooded basins), false alarms down; t_ref 6-24 h changes CSI by < 0.01. The
  weir volume cap (A) changed nothing and stays in `src/event_volume.py` only for the experiments.
- **Land-cover friction (critique 2):** kept, described as a land-cover-indexed attenuation rate
  (docs/methods_02 §2.2, code comments; config keys and file names unchanged). Land-cover and
  uniform attenuation are within 0.005 pooled CSI on 498 random tiles, with and without B.
- **friction_scale_factor with B:** 7.5 (498 random RP100 tiles, pooled CSI 0.763, bias 1.08;
  9.0 gives 0.757, bias 0.98). Without B the optimum stays 12.0 (CSI 0.742). Results in
  `.cleanup_baseline/review_critiques/calibration/representative/`, figures R1/R2.
- **Straight-line baseline** (level = nearest seed level - rate x Euclidean distance, no solver;
  `--stage distance`): best pooled CSI 0.713 coast-connected (0.10 m/km), 0.668 without
  connectivity, against 0.739 eikonal uniform and 0.763 eikonal + B. Median tile within 0.006;
  worse by > 0.2 CSI on 3-4% of tiles (sills, basins, ridges). Figure R3.
- **Outputs:** new results go to `{root}/model_outputs_v2`, `merged_results_v2`, `figures_v2`,
  `validation_v2`; per-tile inputs stay in `{root}/model_outputs` (`preprocessing_inputs_dir`).
  Bangkok, deltas and calibration_500_tiles pin `attenuated` and B off in their `*_pins.yml` and
  set `preprocessing_inputs_dir: null` (own inputs).
- **Protection rule:** `protection.rule: flood_fraction | water_level` (implemented, default
  flood_fraction; `minimum_rp` 2 applies to the water-level rule only).
- **Permanent inundation:** to be counted in EAI by anchoring the exceedance curve at p = 1 with
  the permanently inundated exposure, and reported as a separate metric (not implemented yet, see
  Open).

Open:
- **Bathtub baselines: review all implementations and make them consistent (to do before any
  bathtub vs eikonal comparison goes into the paper).** Each was written for a different study, and
  each makes its own choices, so their scores are not directly comparable:
  - HC-bathtub: `sfincs_tiles/run_calibration_models.compute_bathtub_depth`. One flat level, the
    tile's highest station value, not interpolated. The depth file is unpruned; it is pruned only at
    scoring (`sfincs_tiles/compute_calibration_result.py` via
    `flood_agreement.prune_to_ocean_connected`).
  - Straight-line attenuated bathtub: `review_critiques/calibration_experiments.run_tile_distance`.
    Level = the nearest production seed (IDW of 15 stations) minus rate x Euclidean distance.
    Pruned with `flood_model.prune_to_coast_connected` on the seed cells, on the effective DEM.
  - Static inundation: `flood_model.static_inundation`, a connected flat-level bathtub at
    S = MDT + SLR inside the two-component solve. Land cells only; rivers and lakes are not
    conduits.
  - Spill level: `event_volume.spill_level`, a priority-flood bathtub (minimax elevation from the
    sea) used by correction B.
  - Naming: the eikonal solver is labelled "EA-bathtub" in `sfincs_tiles/` (calibration_results,
    plots, docs/methods_04a) and in `analysis/summarize_run_timings.py`. Literature references:
    Vafeidis et al. 2019 and the Flow-tub rate (0.2 m/km) in docs/methods_02.

  They differ in:
  - the water-level source (tile maximum, nearest seed, or IDW);
  - the connectivity rule: two pruning functions, with different coastline definitions and
    river cells as conduits or not;
  - the DEM (raw or effective, no-data handling);
  - whether pruning is applied to the depth raster or only to the score;
  - the names used for them.

  Goal: one shared implementation, e.g. `src/bathtub.py`, with:
  - a level field: flat, nearest-seed or IDW, optionally minus rate x distance;
  - one connectivity rule;
  - the same DEM, seeds and wet threshold as the eikonal solver.

  Then define one set of names (HC-bathtub, attenuated bathtub, eikonal / eikonal + B), use them
  in `sfincs_tiles/`, `review_critiques/` and `validation/`, and re-score every baseline on the
  same tiles. `sfincs_tiles/` changes need your go-ahead, and none while an HPC batch runs.
- **fsf grid between 7.5 and 9:** the optimum may lie between the tested values (bias crosses 1
  near 8.7). Refine only if the bias matters more than the last 0.005 CSI.
- **Pinned scenarios re-run on the next Snakemake call:** the new `flooding` keys and the
  `static_level_inputs` param change run_eikonal's params, so bangkok_chao_phraya (98 jobs in the
  dry run) and the other scenarios re-run run_eikonal with identical results. Use
  `--rerun-triggers mtime` to skip that.
- **HPC calibration tooling uses neither change:** `sfincs_tiles/run_calibration_models.py` calls
  `flood_depth_dense` without the static level or B, so stage-1/2 calibration results describe the
  old physics. Wire it when `sfincs_tiles/` may be touched.
- **`run_eikonal_dag.py` OOM tile split:** `tile_split.split_tile` reads the mask from
  `model_outputs/<tile>/inputs`; with `preprocessing_inputs_dir` set it raises FileNotFoundError
  instead of splitting. Pass the inputs dir if splits are needed in production.
- **Permanent inundation p = 1 anchor:** implement with the static-flood extent of the
  two-component solve (flood_depth_dense diagnostic `n_static_wet_cells` only counts it now).
- **Tide in the static level:** decide later whether S = MDT + SLR (current choice) should also
  include a tidal high water (MHHW). Needs a tide dataset (GTSM tide statistics or FES2014); COAST-RP
  cannot separate tide from surge.
- **`protection.default_rp: 5`** for geogunits without FLOPROS values vs the RP2 minimum: keep 5?
- **Flood-fraction rule never exposes saturated cells:** where the design event already floods the
  whole cell (ff = 1), no larger event or SLR can exceed the design flood fraction, so the cell stays
  protected forever. Bangkok case: 33% of the population in the design floodplain sits in such cells.
  Production uses this rule; the water-level rule does not have the flaw.
- **Water-level rule + binary breach is harsh under SLR:** over all COAST-RP stations the median RP100
  standard drops to "fails every year" by ~0.25 m SLR (RP1000 by ~0.4 m), and a defence then counts
  as fully breached. Bangkok: protect at the +1 m design equals no adaptation from +1.5 m. Freeboard
  or an overtopping volume would soften this before the water-level rule is adopted.
- **SLR fingerprints scale vertical land motion with global SLR (production issue, left as is for
  now):** `prepare_boundary_conditions.compute_slr_fingerprints` takes the nearest AR6 point within
  1 deg (tide gauges included), whose 2100 total relative SLR contains VLM, and scales it by T / GMSL.
  Upper Gulf of Thailand stations (13.46 N, 100.2-100.7 E) get 5.16 m at T = 1.5 m; the global
  0.1-99.9 percentiles are -3.9 / +4.3 m. Affects both propagation variants.
- **Materialized configs on the cluster:** re-materialize after pulling (new keys `protection.rule`,
  `protection.minimum_rp`, `flooding.slr_propagation`, `flooding.duration_correction`, new paths).
- **Cluster runs (not planned for now):** SFINCS at +1 m / +2 m SLR (benchmark for 1a) and at
  RP10 / RP1000 on ~100 tiles (transferability, critique 1b, now with B on; RP100 COAST-HG shape
  shifted to each RP peak).

## Known behaviour

- Outputs at `friction_scale_factor` 9.0 (deltas, bangkok_chao_phraya) and 30.0
  (calibration_500_tiles) are kept as computed; each config says so. Production is 7.5 (with B).
- A hop>=1 triple whose neighbour output is missing fails without writing output. Local multi-hop
  `snakemake simulate` / `run_eikonal_dag.py` can therefore fail when Snakemake schedules a hop-1
  tile before its neighbours (the rule does not declare neighbour inputs); rerun, or run wave by
  wave as `run_delta_floodmaps.py` does.

## Git history rewrite (after the merge)

History holds ~1.2 GB of old files that are no longer tracked (209 MB packed): `sfincs_tiles/*.npy`
(tile306 / tile1669 arrays, up to 102 MB each), `batch_logs/`, `batch_logs_simulate_regions/`,
`run_timings.csv`, `run_postprocess.log`, `preparation/logs/`, `adaptation_validation_output/`,
`tests/*/output/`, `tests/*/plots/`. Rewriting changes every commit hash and needs a force push.

1. Merge `cleanup` into `main`, push, and make sure no clone has unpushed work (local repo,
   cluster `<code_root>`).
2. Install the tool: `pip install git-filter-repo`.
3. Work on a fresh mirror, not your working repo:
   ```
   git clone --mirror <repo_url> gfm-mirror.git
   cd gfm-mirror.git
   git filter-repo --invert-paths \
     --path-glob 'sfincs_tiles/*.npy' --path batch_logs/ --path batch_logs_simulate_regions/ \
     --path run_timings.csv --path run_postprocess.log --path preparation/logs/ \
     --path adaptation_validation_output/ --path-glob 'tests/*/output/*' --path-glob 'tests/*/plots/*'
   git count-objects -vH        # check size-pack dropped
   git log --oneline -5 main    # spot-check
   git push --force --mirror
   ```
4. Re-clone every working copy (copy the gitignored files across: `config_local.yml`,
   `*_materialized.yml`, `.cleanup_baseline/`), or `git fetch && git reset --hard origin/<branch>`.
5. Old hashes in docs and messages change; `.git/filter-repo/commit-map` in the mirror maps old to
   new (e.g. for AUDIT.md's `git show 89e130e:AUDIT.md`).
