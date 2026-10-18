# TODO

Open items after the cleanup on branch `cleanup` (not merged into `main`). Everything in
the code is done; what remains needs you, Hydrax or P:\.

## Next steps

1. **Review and merge `cleanup` into `main`** (`git log main..cleanup`).
2. **Before pulling on Hydrax:** back up the `*_materialized.yml` files there (a pull removes
   files that are now untracked); restore them afterwards.
3. **After pulling, on every machine:** re-materialize the scenario configs
   (`python snakemake_workflow/scripts/materialize_configs.py`). Config keys were renamed:
   `boundary_conditions.slr_scenario` -> `slr_projection`, `slr_scenarios` -> `waterlevel_names`;
   configs with the old keys now fail with KeyError. `deltas_design_materialized.yml` is written
   by `analysis/prepare_delta_design_runs.py` (it still has the old keys and the removed
   `aqueduct_mem_estimate`): regenerate it before the next delta design run.
4. **On Hydrax:** `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal` once, to check the
   shell drivers (only tested locally with stubbed sbatch/python). Paths live in
   `sfincs_tiles/hydrax_env.sh`; keep its roots in sync with `config_hpc.yml`.
5. **Hydrax `gfm` env:** `mamba install -n gfm pytest=8.4.2 ruff=0.12.0`.
6. **Local `config_local.yml`:** delete the leftover `paths.aqueduct_root` line (unused). This
   machine has `paths.sfincs_exe` set for the SFINCS smoke test; add it on other machines if wanted.
7. **P:\ cleanup:** `modelling/sfincs_smoke_test/` is no longer used (the smoke test now builds in a
   temp folder); delete it if you like. When calibration stage 1 runs again, pass `--overwrite`
   so screen / pre-flight error records cached by earlier runs are recomputed.

## Before reusing existing outputs

Old formats are no longer accepted; regenerate before the next step that reads them:
- `merged_results/exposure/flood_totals/flood_totals_*.csv` (int `return_period`): `analysis/compute_flood_totals.py`.
- `bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv`: `analysis/compute_bangkok_case_exposure.py`.
- `deltas_floodmaps/delta_flood_extent.csv` (column `slr_scenario`): `analysis/compute_delta_flood_extent.py`,
  then `analysis/add_slr_year_column.py`.
- England FZ3 / FZ2 results (now `{validation.output_dir}/GBR/metrics_GBR_england_ea_floodzone{3,2}_sea_*.csv`):
  `validation/rescore_england_fz3.py` and `validation/validate_england_fz2_band.py`, before
  `compare_against_published.py` / `build_overview_table.py`.
- Calibration-study batch jobs in `calibration_500_tiles/hpc_jobs/` pass the output dir
  positionally: regenerate with `calibration_studies/generate_calibration_batch_jobs.py` before
  resubmitting.
- Snakemake: the config-key rename re-triggers `cache_waterlevel_stations`, `extract_boundaries` and
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

## Known behaviour

- Outputs at `friction_scale_factor` 9.0 (deltas, bangkok_chao_phraya) and 30.0
  (calibration_500_tiles) are kept as computed; each config says so. Production is 12.0.
- A hop>=1 triple whose neighbour output is missing fails without writing output. Local multi-hop
  `snakemake simulate` / `run_eikonal_dag.py` can therefore fail when Snakemake schedules a hop-1
  tile before its neighbours (the rule does not declare neighbour inputs); rerun, or run wave by
  wave as `run_delta_floodmaps.py` does.
- `generate_eikonal_jobs.generate_wave_dispatch` still takes an unused `tile_ids` parameter.

## Git history rewrite (after the merge)

History holds ~1.2 GB of old files that are no longer tracked (209 MB packed): `sfincs_tiles/*.npy`
(tile306 / tile1669 arrays, up to 102 MB each), `batch_logs/`, `batch_logs_simulate_regions/`,
`run_timings.csv`, `run_postprocess.log`, `preparation/logs/`, `adaptation_validation_output/`,
`tests/*/output/`, `tests/*/plots/`. Rewriting changes every commit hash and needs a force push.

1. Merge `cleanup` into `main`, push, and make sure no clone has unpushed work (Windows repo,
   Hydrax `~/gfm_code`).
2. Install the tool: `pip install git-filter-repo`.
3. Work on a fresh mirror, not your working repo:
   ```
   git clone --mirror https://github.com/JuliusSchlumberger/GFM_Matrix_Snakemake.git gfm-mirror.git
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
