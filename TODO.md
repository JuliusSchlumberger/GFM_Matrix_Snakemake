# TODO: real changes found during the cleanup

The cleanup phases (AUDIT.md §8) are behaviour-preserving. Anything that needs
a real (behaviour- or result-changing) fix is logged here instead of being done
in-phase.

## Start here (status 2026-10-18)

All cleanup phases (AUDIT.md section 8, Phases 0-7) are done on branch `cleanup` and pushed; nothing
is merged into `main` yet. Each phase was verified locally (ruff, pytest, subset dry runs, and
AST/parity checks); nothing ran on Hydrax or wrote to P:\. Suggested order for you:

1. **Review / merge `cleanup`** into `main` (one commit per phase, `git log main..cleanup`).
2. **Before pulling on Hydrax:** back up any `*_materialized.yml` there - a pull deletes files
   that Phase 1 untracked; restore afterwards. (`config_hpc.yml` is tracked again since 2026-10-18.)
3. **On Hydrax after pulling:** `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal`.
   (Stale job files: done 2026-10-18 from Windows, see the Phase 7a/7b notes below.)
4. **Snakemake "code changed" flags**: solved by `profiles/default/config.yaml` (auto-loaded,
   `rerun-triggers: [mtime, params, input, software-env]`). No `--touch` needed.
5. **Decide:** re-materialize the scenario configs or not (item below); the `mdt_sign` test fixture;
   the near-duplicates left in Phase 5d.
6. Hydrax `gfm` env: `mamba install -n gfm pytest=8.4.2 ruff=0.12.0` (Phase 0 tooling).

## Found during the docstring trim (code, not docs - not changed)

- **Decide: outputs computed with `friction_scale_factor` != 12.0** (production is 12.0 since
  2026-10-13; the global run's `model_outputs/hpc_jobs/resolved_config.yml` has 12.0). Still at 9.0:
  deltas_ssp126 / deltas_ssp245 / deltas_design (base configs set 9.0, "0.3x production's 30"),
  bangkok_chao_phraya (run 2026-10-08).
  Still at 30.0: calibration_500_tiles (solver studies, 2026-10-05). Re-run at 12.0
  = change the base config + its `*_pins.yml`, re-materialize, re-run.
- `preparation/select_country_tiles.py`: GBR comment still says only Wales/Scotland benchmarks (England
  FZ2/FZ3 exist now).
- `src/population_growth.interpolate_growth_factor` docs say it returns `default` outside the year range;
  the code clamps. `src/tile_split.choose_split` docs say it raises; it falls back with a warning.
- `sfincs_tiles/retry_io.retry_transient_io`: `retries` is total attempts, not retries.
- `sfincs_tiles/calibration_paths.decode_waterdepth_cm`: only nodata becomes NaN, 0 cm stays 0.0 -
  check that dry is always encoded as nodata where callers assume NaN = dry.
- `src/connectivity_tiling._domain_touches_ocean` reads DEM tiles via `dem_index` and ignores the result
  (wasted read). `sfincs_tiles/build_sfincs_tile._compute_zsini_array`: `grid_coords` unused.
- Rename candidate: `boundary_lines_geometry` attribute `land_eroded` holds land grown outward.
- Strings with stale history/claims (string literals, so untouched): argparse help of
  `build_sfincs_tile.py --no-rotated`; assert messages with dates in `tests/flood_model_memory_validation`
  and `tests/eikonal_kernel_validation`; `generate_hpc_postprocess_job.py` Usage names
  `submit_postprocess.sh` (writes `submit_postprocess_and_exposure.sh`); `generate_eikonal_solver_examples.py`
  prints "by round 40" for a 15-round loop; `build_overview_table._FZ2_NOTE` says "see session notes".
- `tests/sfincs_subgrid_nearest_validation`: `test_reproject_nearest_transform_matches_scaled_source`
  tests a misaligned target - name does not match.
- Git history holds ~1 GB of `sfincs_tiles/tile306_*.npy` / `tile1669_*.npy` (commit c0f84d6,
  2026-09-23; up to 97 MB each, GitHub warns on push). Removing them needs a history rewrite.

## Open

- **Snakemake "code has changed" on locally produced simulation outputs** (Phase 5b): `run_eikonal.py`
  was rewritten into a thin wrapper, so e.g. bangkok_chao_phraya's 405 run_eikonal outputs dry-run as
  "Code has changed since last execution" (with `--rerun-triggers mtime params input software-env`
  there is nothing to do, so inputs and outputs are untouched). Clear it once after the cleanup (item D,
  with the Phase 7a renames) via `snakemake --touch` - it writes Snakemake metadata next to P:\ outputs.
  The earlier "local path lacks the hop-0 fix" item was wrong (465dd93 only staged the file), so the
  deltas do not need a re-run.
  Also covered by the same `--touch`: after re-materializing (2026-10-18, old values pinned in
  `config/<name>_pins.yml`), `merge_chunk` reports "Params have changed" because its `pp_cfg`
  param is the whole `postprocessing` section, which lost ~12 keys config.yml retired (e.g.
  `overlap_corr_*`; no code reads them). No simulation is re-triggered.
- **`generate_eikonal_jobs.generate_wave_dispatch` has 3 unused parameters** (`model_outputs`,
  `hop_by_tile`, `raster_config`; docstring says so). Left in place in Phase 7a on purpose:
  dropping them also means dropping the `generate_eikonal_jobs` rule's params, which re-triggers
  that rule. Remove together with the rule params when you next regenerate HPC dispatch anyway.
  (The audit's other "unused parameter", `connectivity_tiling`'s `dem_index`, is in fact used.)
- **Hydrax: check the shell-driver changes once after pulling Phase 6** (I could only test them
  locally against a fake tile tree with fake sbatch/python, where both branches matched the old
  behaviour). All Hydrax paths now live in `sfincs_tiles/hydrax_env.sh` (same values as before,
  overridable via GFM_DATA_ROOT / GFM_CODE_ROOT / GFM_PYTHON / GFM_SFINCS_PYTHON). The two rerun
  scripts are merged into `sfincs_tiles/rerun_calibration.sh --part eikonal|sfincs`; the old names
  still work as wrappers. Suggested check: `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal`.
- **Remaining machine-specific paths not moved to config** (deliberately): `run_bangkok_case_study.bat`'s
  `STUDY_ROOT` (case-study data on P:\), and `tests/sfincs_synthetic_end_to_end/test_sfincs_smoke.py`'s
  default SFINCS exe path (the test SKIPs when it's absent).
- **Phase 7a renamed aqueduct -> eikonal** (files, rules, functions, config key). After pulling on
  Hydrax: (1) job files - checked 2026-10-18: no file in `model_outputs/hpc_jobs` calls the old
  names, nothing to regenerate; (2) item D: Snakemake will flag
  locally produced outputs of the renamed rule/scripts as "code changed" - outputs are unchanged,
  clear it with `snakemake --touch` (or run with `--rerun-triggers mtime params input software-env`).
  `simulation.aqueduct_mem_estimate` was renamed `eikonal_mem_estimate`; rules/simulation.smk still
  falls back to the old key for configs materialized/resolved before the rename - drop the fallback
  once those are regenerated. `paths.aqueduct_root` was removed (no config used `{aqueduct_root}`);
  the leftover line in your local config_local.yml is harmless, delete it by hand.
- **Phase 7b renamed modules and scripts** (src/plotting -> pipeline_plots, visualization ->
  analysis_figures, exposure -> exposure_grid, flood_extent folded into flood_model,
  validation.tri_domain_mask -> rasters.polygon_domain_mask, validation/_rescore_england_fz3 ->
  rescore_england_fz3, calibration_studies test_*_calibration -> run_*_study and
  select_calibration_tiles -> select_study_candidate_tiles, scripts/calibration/select_country_tiles
  -> preparation/select_country_tiles). Code outside this repo that imports the old module names
  (notebooks, personal scripts) needs the same rename. The 60 calibration-study sbatch files in
  `calibration_500_tiles/hpc_jobs` (2026-10-05) were fixed in place on 2026-10-18: only the script path
  was rewritten (`tests/test_<study>_calibration.py` -> `calibration_studies/run_<study>_study.py`, same
  CLI); batches, --time and resolved_config.yml are as they ran. Backup: `.cleanup_baseline/calib500_hpc_jobs_backup/`.
