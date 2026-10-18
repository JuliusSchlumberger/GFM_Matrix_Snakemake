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

- **`tests/sfincs_mdt_sign_validation/validate_mdt_sign.py` fails (found in Phase 0, 2026-10-17).**
  `test_elevation_mdt_is_added_not_subtracted` expects the ocean cell to be
  GEBCO + MDT (-19.5664 m), but `sfincs_tiles/build_elevation.build_combined_elevation`
  returns 5.0 m, the land value. This is neither the added nor the subtracted value.
  It already fails at baseline (`.cleanup_baseline/validate_mdt_sign.log`), in hydromt-sfincs-dev.
  *Probable cause, not verified:* `_fill_coastal_transition_elevation` (added in
  283b07c, 2026-09-24) blends the coastal ocean band towards land elevation. On the
  test's 4×4 grid with a single ocean column, every ocean cell lies inside that band.
  If so, the fixture needs a wider ocean region, or the MDT sign should be asserted
  on `info`/`gebco_shifted` before the transition fill. That is a test fix, not a
  sign bug, but confirm that before changing anything.
- **FZ2 band figures disagree between the tracked CSVs (found in Phase 2b).**
  The note in `validation/comparison_vs_published.csv` says bias 2.4x and HR 0.066;
  `validation/overview_extent_metrics.csv` says bias 1.99 and HR 0.060. One of them is
  stale. Regenerate both (`build_overview_table.py`, `compare_against_published.py`)
  after the next England FZ2 run and check which is right. methods_04b currently cites
  only the overview CSV's bias.
- **England catalog comment is out of date.** The header comment of
  `england_ea_floodzone3_sea` / `_floodzone2_sea` in `data_catalog_validation.yml` says
  the data were filtered to `flood_source="sea"` only. `fetch_england_flood_zones.py`
  later added `river and sea`. Confirm which filter the current `fz*_sea_modelled.gpkg`
  were built with, then fix the comment.
- **`validation/validate_sf_bay.py` docstring** refers to `validate_country_depth_raster()`,
  which doesn't exist. `_rescore_england_fz3.py`'s docstring implies it uses
  `postprocessing.block_size`, but it hard-codes 8192 px. Both are docstring-only fixes
  (Phase 4/4b).
- **`fetch_england_flood_zones.py` and `build_overview_table.py` have no CLI** (no
  argparse). Any `--help` runs them for real, and fetch writes to P:\. Add argparse in
  Phase 6, when the hard-coded P:\ paths move to config.
- **Two console messages disappeared with the partial path.** "flood_totals missing →
  model_total_wet_km2 NaN" and "No coastal benchmark found for X" were only printed by
  the deleted `validate_country()`. The national path now prints just "No extent rows
  produced." Re-add them to the national path if you want them.
- **Snakemake "code has changed" on locally produced simulation outputs** (Phase 5b): `run_eikonal.py`
  was rewritten into a thin wrapper, so e.g. bangkok_chao_phraya's 405 run_eikonal outputs dry-run as
  "Code has changed since last execution" (with `--rerun-triggers mtime params input software-env`
  there is nothing to do, so inputs and outputs are untouched). Clear it once after the cleanup (item D,
  with the Phase 7a renames) via `snakemake --touch` - it writes Snakemake metadata next to P:\ outputs.
  The earlier "local path lacks the hop-0 fix" item was wrong (465dd93 only staged the file), so the
  deltas do not need a re-run.
- **SF Bay validation reads one hard-coded model chunk.** `validation/validate_sf_bay.py:100`
  points at `merged_results/chunks/waterdepth_N35W125_RP100_SLR_0.tif`. If the Bay domain
  spans more than that chunk, or the chunk grid changes, part of the Bay is silently not
  scored. Phase 6 moves the path into config; picking up overlapping chunks automatically
  (as `validate_country.py` does with `_chunks_overlapping_bbox`) would be a behaviour
  change, so it's logged here. Bringing SF Bay into `run_validation.py` is deferred (AUDIT.md §9.4 Q2).
- **`openpyxl` in environment.yml has no user** (found in Phase 4). Its only consumer,
  a calibration xlsx report script, was removed earlier; nothing imports openpyxl or
  calls `to_excel`/`ExcelWriter`. Dropping the pin is an env change, so it's logged here.
- **`src/flood_model.prune_to_coast_connected` docstring describes a "pre-solve candidate
  mask" use** (found in Phase 4b), but every caller uses it post-solve on the flood mask,
  and `flood_depth_dense` says the dense solver has no pre-solve candidate step. This is
  probably a leftover from an older (sparse) solver. Confirm, then trim the docstring.
- **`src/eikonal_runner.py` module docstring** names `scripts/run_eikonal.py` /
  `run_eikonal_cli.py`; the real paths are under `snakemake_workflow/scripts/`. Fix it with
  the Phase 7a renames (these files become `run_eikonal*.py`).
- **`generate_eikonal_jobs.generate_wave_dispatch` has 3 unused parameters** (`model_outputs`,
  `hop_by_tile`, `raster_config`; docstring says so). Left in place in Phase 7a on purpose:
  dropping them also means dropping the `generate_eikonal_jobs` rule's params, which re-triggers
  that rule. Remove together with the rule params when you next regenerate HPC dispatch anyway.
  (The audit's other "unused parameter", `connectivity_tiling`'s `dem_index`, is in fact used.)
- **`generate_hpc_postprocess_job.py --help` crashes** (found in Phase 5a, pre-existing at HEAD):
  argparse raises `TypeError: %d format: a real number is required, not dict` because a
  `help=` string contains an unescaped `%`. Only `--help` is affected; normal runs parse fine.
  Fix: escape it as `%%`.
- **Near-duplicate helpers left unmerged in Phase 5d** (they differ in behaviour, so merging
  would change results or output; unify deliberately if wanted):
  - `_first_crossing`: compute_tipping_points_summary.py vs plot_tipping_points.py (the latter
    defaults `years=YEARS`).
  - `_people`: plot_extended_pathways.py vs plot_slr_vs_population.py (the latter handles
    negative values with a sign).
  - `_read_band` / `_read_band_and_transform`: compute_flood_totals.py (via `_open_and_read`)
    vs analysis/_common.py / compute_bangkok_case_exposure.py.
  - `_run` (subprocess step runner): analysis/run_analysis.py, validation/run_validation.py,
    run_delta_floodmaps.py: three different variants (printing, timing, fail-fast handling).
- **Materialized scenario configs are older snapshots than config.yml** (found in Phase 6).
  `snakemake_workflow/scripts/materialize_configs.py` (new; replaces build_delta_run_configs.py)
  rebuilds `<name>_materialized.yml` from `config.yml` + `config_local.yml` + `<name>.yml`, but a
  fresh build differs from the files on disk (39-65 keys each), because config.yml changed after
  they were written. Examples: bangkok_chao_phraya/thailand_bangkok `adaptation.slr_intensities`
  lose SLR_1500; gbr_wales_scotland_friction9 `hpc.n_nodes` 35 -> 40; analysis switches;
  removed/added hpc keys. I did NOT overwrite them (that would change what those scenarios run).
  Decide per scenario whether to re-materialize (`python snakemake_workflow/scripts/materialize_configs.py <name>`);
  to inspect first: `... --out-dir <tmp>` and diff. A fresh clone gets the new-config version.
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
- **`check_preprocess_progress.py` still counts `aqueduct_*.toml` files** per tile: Julia-era Aqueduct
  TOML configs that no current code writes, so that column always reports 0 of N. Not renamed in
  Phase 7a (it is a file-name pattern on disk). Remove the TOML column or point it at a real output.
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
- **A second chunk-ID formatter exists**: `validation/_chunks._postprocess_chunk_id(x, y, size)`
  re-implements the "N00E000" naming of `src/tiles.build_chunk_grid` (the single grid definition
  since Phase 5a). Same rule today; they could drift. Make `_postprocess_chunk_id` call a shared
  naming helper in tiles.py (behaviour-preserving; pin with tests/chunk_grid_validation).
