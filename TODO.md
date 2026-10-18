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
3. **On Hydrax after pulling:** regenerate any HPC job files written before Phase 7a/7b (they call
   old script names); then `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal`.
4. **Snakemake "code changed" flags** (renamed rules/scripts): clear once with `snakemake --touch`
   on the configs whose outputs exist (writes Snakemake metadata next to P:\ outputs).
5. **Decide:** re-materialize the scenario configs or not (item below); the `mdt_sign` test fixture;
   the near-duplicates left in Phase 5d.
6. Hydrax `gfm` env: `mamba install -n gfm pytest=8.4.2 ruff=0.12.0` (Phase 0 tooling).

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
- **`config.yml` default paths left as they are** (`paths.root: D:/GFM`, `code_root:
  C:/Users/Schlu005/GFM`): changing them could break a machine that relies on them without a
  config_local.yml. New `config_local.yml.example` documents the override. Replace the defaults
  with neutral placeholders if no machine uses them.
- **Remaining machine-specific paths not moved to config** (deliberately): `run_bangkok_case_study.bat`'s
  `STUDY_ROOT` (case-study data on P:\), and `tests/sfincs_synthetic_end_to_end/test_sfincs_smoke.py`'s
  default SFINCS exe path (the test SKIPs when it's absent).
- **Phase 7a renamed aqueduct -> eikonal** (files, rules, functions, config key). After pulling on
  Hydrax: (1) **regenerate any HPC job files written before the rename** that haven't run yet -
  they call `run_aqueduct_cli.py` / `snakemake generate_aqueduct_jobs`, which no longer exist
  (rerun the `generate_hpc_*` / `generate_eikonal_jobs` step); (2) item D: Snakemake will flag
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
  (notebooks, personal scripts) needs the same rename. Calibration-study sbatch files generated
  before the rename call the old script names: regenerate them with generate_calibration_batch_jobs.py.
- **`sfincs_tiles/build_station_boundary_lines.py` (1254 lines) was not split in Phase 7c** on
  purpose: 38 tightly coupled functions sharing 35 module constants, 7 importers (partly via
  `bsl.<name>` attribute access), no tests, and no way to run SFINCS stage 1 end to end locally
  (needs P:\ data + Hydrax). A split would be unverifiable. If wanted later: first add a
  regression test that builds boundary lines for 1-2 stored tiles and compares the output gpkg,
  then split (lines / conflict resolution / plotting / CLI), keeping re-exports for `bsl.X` users.
