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

Done on 2026-10-18 (outputs regenerated with the new code, values identical to the previous files;
old copies in `.cleanup_baseline/p_backup_regen/`): flood totals (cluster job), Bangkok case exposure,
delta flood extent + SLR-year column, delta design selection and `deltas_design_materialized.yml`,
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

## Known behaviour

- Outputs at `friction_scale_factor` 9.0 (deltas, bangkok_chao_phraya) and 30.0
  (calibration_500_tiles) are kept as computed; each config says so. Production is 12.0.
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
