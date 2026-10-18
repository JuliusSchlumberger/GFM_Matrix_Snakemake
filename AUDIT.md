# Repository audit: GFM_Matrix_Snakemake

Audit date: 2026-10-17 · HEAD `f8eda9e` (main) · 105 commits since 2026-06-17.
Read-only audit: the only file written is this one. Uncommitted edits at audit time: `analysis/compute_custom_pathways.py`, `analysis/plot_custom_pathways.py`, `analysis/plot_slr_vs_population.py` (treated as active work).

**Entry points**, inferred because the prompt's placeholders were left empty:

| Entry point | Drives |
|---|---|
| `Snakefile` (targets `preprocess`, `simulate`, `postprocess`, `all`; `generate_aqueduct_jobs` is requested explicitly) | main eikonal DAG |
| `snakemake_workflow/scripts/generate_hpc_{preprocess,postprocess}_job.py`, `generate_hpc_simulation_jobs.py` | SLURM dispatch (hpc.md) |
| `preparation/run_preparation.py` | tile grid + boundary conditions (runs before the DAG) |
| `analysis/run_analysis.py` | exposure/adaptation figures |
| `validation/run_validation.py` | eikonal vs **national hazard maps** (benchmark HR/FAR/CSI) |
| `sfincs_tiles/generate_calibration_jobs.py stage1/stage2` (top level), `rerun_{eikonal,sfincs}_calibration.sh` (partial re-run drivers), `run_final_tile.sh` (per-tile SLURM worker, not called by hand) | eikonal vs **SFINCS** (friction-factor calibration) on Hydrax |
| `calibration_studies/generate_calibration_batch_jobs.py {sweep_budget,obstacle_coupling}` → `test_sweep_budget_calibration.py` (inner sweep rounds), then `aggregate_wet_tiles.py`, then `test_obstacle_coupling_calibration.py` (outer iterations), then `plot_sweep_*.py` / `plot_obstacle_coupling_calibration_bands.py`. Tile pool comes from `calibration_studies/select_calibration_tiles.py` | **eikonal solver sensitivity** (inner rounds / outer loops; docs/methods_03). *Added 2026-10-17 at the user's request.* |
| `run_pipeline.py` (to be renamed, see Q1), `run_delta_floodmaps.py`, `run_bangkok_case_study.bat`, `run_thailand_bangkok.{bat,ps1}` | wrappers / case studies |

**Do the validation and SFINCS-calibration rows (the two benchmark entry points) overlap?** No. They compare the eikonal solver against different references: official hazard maps (fractional coverage, soft scoring) versus SFINCS model runs (binary depth rasters, hard scoring). What they share is only the metric formulas, which are copied (see §4.1.3). Inside the SFINCS row there is overlap: the two `rerun_*.sh` scripts share their path setup, reset loop and stage-2 submit. They differ only in which artefacts they delete and whether a post job is queued, so they can become one `rerun_calibration.sh --part {eikonal,sfincs}`.

**Environments**: `gfm` (`environment.yml`: Python 3.10.13, snakemake 7.32.4, hydromt 0.9.3, no hydromt_sfincs, no pytest) is the main environment. `hydromt-sfincs-dev` is used for sfincs_tiles model building/running (matplotlib broken); no env file describes it. `gfm_python_preprocessing` is the **same** `gfm` env under an outdated name on the Windows machine; Hydrax calls it `gfm` (user, 2026-10-17, correcting an earlier note).

**Existing tooling**: no pyproject/setup.cfg, no pytest config, no pre-commit, no ruff config, no CI (`.github/` absent). Ruff is mentioned only as an optional comment in `environment.yml`.

---

## 1. Map

| Directory | Holds |
|---|---|
| `/` (root) | `Snakefile`, `run_pipeline.py` (OOM-retry wrapper), `check_*_progress.py` monitors, `run_delta_floodmaps.py`, three `.bat`/`.ps1` case-study launchers, **plus stray tracked data**: `run_timings.csv` (9.5 MB), `triplet_comparison_CLEAN.csv`, `sfincs.log` |
| `src/` | Shared library for everything (flat; reached via `sys.path.insert`). Solver (`eikonal`, `flood_model`), I/O (`rasters`, `tiles`, `boundaries`, `merge`), config, exposure/analysis helpers, plotting |
| `snakemake_workflow/rules/` | 5 `.smk` rule files included by `Snakefile` |
| `snakemake_workflow/scripts/` | Rule `script:` targets, SLURM job generators and their sbatch payloads, plus a few manual/one-off tools |
| `snakemake_workflow/config/` | `config.yml` (source of truth), data catalogs, per-study override ymls, **tracked** `*_materialized.yml`, **tracked** `config_hpc.yml`, `calibration/sweep/*.yml` (orphaned) |
| `preparation/` | Pre-DAG steps run by `run_preparation.py`, manual subset/ingest tools, `logs/` (72k lines of tracked historical logs) |
| `analysis/` | Post-DAG exposure, pathways and tipping-point figures, plus the deltas deliverable, the Bangkok case study and one-off experiments (mixed concerns) |
| `validation/` | National-map benchmark validation: GBR (Wales, Scotland, England) + CAN (New Brunswick) via `run_validation.py`; England FZ2/FZ3 and SF Bay via standalone scripts (scope since 2026-10-17, §9) and tracked result CSVs |
| `sfincs_tiles/` | SFINCS calibration/validation sub-pipeline (separate env, Hydrax `.sh` drivers) |
| `calibration_studies/` | Eikonal round-cap / obstacle-coupling sensitivity studies (methods_03) |
| `julius_sfincs_code/` | Imported external basin-scale SFINCS workflow (reference only, not runnable here) |
| `tests/` | Standalone assert scripts (`validate_*.py`, some `test_*.py`), plus one-off diagnostics (`norway_diagnostics/`) and archived prototype output |
| `docs/` | Methods docs 01–04b, upstream bug reports, an outdated runbook, figures plus their generator script |
| `scratch/` | One 13-line one-off script |
| `adaptation_validation_output/`, `batch_logs/` | Tracked output/log dirs (regenerable or stale) |
| `.claude/`, `.snakemake/`, `.vscode/`, `__pycache__/`, `batch_logs_simulate/` | Untracked/ignored local state |

---

## 2. Dependency graph

### 2a. Snakemake rule graph

```
all ─┬─ preprocess ── extract_tile_geometry → compute_model_bbox → extract_dem → extract_dem_mask
     │                compute_geoid_offset_raster, compute_friction
     │                cache_waterlevel_stations → extract_boundaries
     ├─ simulate ──── run_aqueduct            (scripts/run_aqueduct.py)
     └─ postprocess ─ merge_chunk → compute_flood_fraction_chunk ← prepare_exposure_grid_chunk
                      build_mosaic_vrt, plot_merged_results   (only if postprocessing.plots.enabled)
generate_aqueduct_jobs (hpc_dispatch.smk; not under `all`, requested explicitly)
```

### 2b. HPC (SLURM) graph, outside Snakemake's DAG

```
generate_hpc_preprocess_job.py ──sbatch──> generate_hpc_simulation_jobs.py ──imports──> generate_aqueduct_jobs.generate_wave_dispatch
                                                                           └─sbatch──> run_simulation_batch.py ──subprocess──> run_aqueduct_cli.py
generate_hpc_postprocess_job.py ──imports──> generate_exposure_jobs.py ──sbatch──> run_exposure_pass1/2.py, reduce_exposure_shares.py, reduce_exposure_write.py
                                └─imports──> generate_hpc_flood_totals_and_plots_job.py ──sbatch──> analysis/compute_flood_totals.py, analysis/run_analysis.py
```

The HPC exposure path imports `analysis/compute_exposure_analysis.py`, so `analysis/` is **not** independent of the pipeline, contrary to the README.

### 2c. src/ import graph (consumers outside src/)

| Module | Reached from |
|---|---|
| `config_utils` | ~100 files (everything) |
| `rasters`, `tiles`, `boundaries`, `merge`, `aqueduct_runner`, `vertical_datum`, `exposure`, `protection` | DAG rule scripts |
| `flood_model` → `eikonal`, `flood_extent` | DAG via `aqueduct_runner`; also sfincs_tiles and calibration_studies |
| `plotting` | DAG (`plot_merged_results`), validation, analysis |
| `exposure_analysis`, `population_growth`, `population_extension`, `pathways`, `visualization` | analysis/ (+HPC exposure through `compute_exposure_analysis`) |
| `validation` | validation/, analysis (deltas) |
| `map_style` | analysis, sfincs_tiles, validation plots |
| `connectivity_tiling` | only `preparation/build_tile_manifest.py` |
| `tile_split` | only `run_pipeline.py` |
| `__init__.py` | nothing (empty; flat layout) |

### 2d. sfincs_tiles graph

```
rerun_{eikonal,sfincs}_calibration.sh → generate_calibration_jobs.py stage1/stage2
   stage1 → select_calibration_tiles.py → preflight_sfincs_mask, build_station_boundary_lines
   stage2 → sbatch run_final_tile.sh → regenerate_dem_mask → build_elevation(mdt) → build_roughness
            → build_line_forcing(connected_forcing) → build_sfincs_tile → run_calibration_models
            → run_sfincs_tile(sfincs_run) → compute_calibration_result(flood_agreement, calibration_paths)
rerun_eikonal_calibration.sh → compute_calibration_result_dedup → run_calibration_sweep_analysis
            → compute_friction_sweep_metrics, calibration_results, plot_validation_results,
              plot_sfincs_tile_diagnostics, plot_worst_tiles_panel
shared: gfm_config, retry_io, sfincs_grid
```

### 2e. Not reachable from any entry point, wrapper or documented manual step

- `snakemake_workflow/scripts/debug_avoid_ssp.py`
- `julius_sfincs_code/*` (all 10 files)
- `sfincs_tiles/scan_nodata_frame_leak.py`
- `scratch/dtm_seeger_pronk.py`
- `tests/_list_tiles_by_size.py`
- `preparation/promote_world_domains.py`, referenced only by an archive README
- `analysis/compare_fundy_extreme_forcing.py`, `plot_coastrp_stations_map.py`, `plot_landuse_map.py`, `plot_waterlevel_by_rp.py`
- `analysis/plot_adaptation_pathways.py`, `compute_tipping_points_summary.py`, `compute_custom_pathways.py`, `plot_custom_pathways.py`, `plot_delta_*`: manual only, with no orchestrator switch
- `validation/_scratch_check_corrupt_exposure_grids.py`
- `snakemake_workflow/config/calibration/sweep/*.yml` (14 files): no code reads that path
- Dead functions:
  - `src/exposure_analysis.py:205 _agg_by_country`
  - `src/map_style.py:111 land_water_legend_handles`
  - `src/merge.py:66 _bounds_intersect`
  - `sfincs_tiles/sfincs_run.py` `parse_sfincs_inp`, `forward_geometry_files`
- ~~Called only from tests: `src/rasters.py:404 resolve_offshore_mask_gaps_via_gebco`, `src/rasters.py:493 relabel_small_isolated_ocean_patches`~~ **Correction (Phase 3, 2026-10-18): wrong. Both are called unconditionally by `extract_dem` (rasters.py:661) and `extract_dem_mask` (:731, :734), i.e. by every production preprocessing run. They are CORE and stay; their tests stay.**

---

## 3. Classification table

The *Reach* column says how a file is reached:

| Code | Meaning |
|---|---|
| DAG | Snakemake `script:` target |
| HPC | SLURM payload or job generator |
| PREP | `run_preparation.py` |
| ANA | `run_analysis.py` |
| VAL | `run_validation.py` |
| SF | sfincs_tiles `.sh` chain |
| WRAP | root wrapper / `.bat` |
| imp | imported by a reachable module |
| manual | documented manual CLI |
| — | none |

"Last" is the date of the file's last commit.

### Root

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| Snakefile | 315 | 10-09 | entry | CORE | includes all 5 rule files |
| run_pipeline.py | 201 | 09-14 | entry | SUPPORT | local OOM-retry wrapper; only user of `src/tile_split` |
| check_preprocess_progress.py | 118 | 09-14 | manual | SUPPORT | monitor; duplicates `_count_progress`/`_print_report` with check_simulation |
| check_simulation_progress.py | 122 | 10-07 | manual | SUPPORT | as above |
| check_postprocess_progress.py | 136 | 09-14 | manual | SUPPORT | contains 3rd copy of `_build_chunk_grid` |
| run_delta_floodmaps.py | 230 | 10-16 | entry | SUPPORT | deltas deliverable; subprocesses snakemake + run_preparation |
| run_bangkok_case_study.bat | 94 | 10-09 | entry | SUPPORT | Chao Phraya case study (distinct from thailand pair) |
| run_thailand_bangkok.bat | 61 | 10-02 | entry | SUPPORT | header: "Safe to delete once this project is done" |
| run_thailand_bangkok.ps1 | 56 | 10-02 | entry | DUPLICATE | twin of `run_thailand_bangkok.bat` (same 4 stages, same paths) |
| run_timings.csv | 202,681 | 10-15 | — | STALE (output) | written by `analysis/summarize_run_timings.py --csv`; 9.5 MB |
| triplet_comparison_CLEAN.csv | 202 | 09-23 | — | UNKNOWN | no writer in tracked code |
| sfincs.log | 21 | 09-20 | — | STALE | stray SFINCS run log |

### src/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| src/__init__.py | 0 | 09-14 | — | STALE | empty; flat sys.path layout never imports `src` as package |
| src/aqueduct_runner.py | 361 | 10-07 | DAG imp | CORE | imported by Snakefile, run_aqueduct(_cli) |
| src/boundaries.py | 355 | 10-07 | DAG imp | CORE | extract_boundaries, cache_waterlevel_stations, run_aqueduct |
| src/config_utils.py | 327 | 10-08 | all | CORE | ~100 importers |
| src/connectivity_tiling.py | 734 | 10-07 | PREP imp | CORE | only `build_tile_manifest.py`; unused param `dem_index` (:590) |
| src/eikonal.py | 394 | 10-07 | DAG imp | CORE | via flood_model |
| src/exposure.py | 78 | 10-15 | DAG imp | CORE | only prepare_exposure_grid_chunk; name clashes with exposure_analysis |
| src/exposure_analysis.py | 732 | 10-13 | ANA/HPC imp | CORE | analysis + HPC exposure |
| src/flood_extent.py | 24 | 09-14 | DAG imp | CORE | only `effective_dem`, `LAND_CODE`; name misleading (docstring: extent code "removed") |
| src/flood_model.py | 446 | 10-12 | DAG imp | CORE | aqueduct_runner, sfincs_tiles, calibration |
| src/map_style.py | 132 | 10-15 | ANA/SF imp | SUPPORT | figure styling; 1 dead fn |
| src/merge.py | 339 | 10-16 | DAG imp | CORE | merge_chunk; 1 dead fn |
| src/pathways.py | 34 | 10-13 | ANA imp | SUPPORT | 7 analysis scripts |
| src/plotting.py | 273 | 10-09 | DAG imp | CORE | plot_merged_results, validation |
| src/population_extension.py | 57 | 10-13 | ANA imp | SUPPORT | 7 analysis scripts |
| src/population_growth.py | 199 | 10-13 | ANA imp | SUPPORT | analysis + exposure_analysis |
| src/protection.py | 34 | 09-14 | DAG imp | CORE | via exposure.py |
| src/rasters.py | 1087 | 10-07 | DAG imp | CORE | many; 2 fns only used by tests |
| src/tile_split.py | 188 | 10-07 | WRAP imp | SUPPORT | only run_pipeline.py |
| src/tiles.py | 446 | 10-07 | DAG imp | CORE | extract_tile_geometry etc. |
| src/validation.py | 1037 | 10-02 | VAL imp | CORE | validation/, analysis; metrics duplicated in sfincs_tiles/flood_agreement.py |
| src/vertical_datum.py | 173 | 09-14 | DAG imp | CORE | compute_geoid_offset_raster |
| src/visualization.py | 804 | 10-16 | ANA imp | SUPPORT | analysis figures |

### snakemake_workflow/rules

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| rules/common.smk | 8 | 09-14 | DAG | CORE | wildcard constraints |
| rules/preprocessing.smk | 187 | 10-08 | DAG | CORE | 8 rules |
| rules/simulation.smk | 94 | 10-07 | DAG | CORE | run_aqueduct |
| rules/postprocessing.smk | 169 | 10-09 | DAG | CORE | 5 rules |
| rules/hpc_dispatch.smk | 101 | 10-07 | DAG | CORE | generate_aqueduct_jobs |

### snakemake_workflow/scripts

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| extract_tile_geometry.py | 13 | 09-14 | DAG | CORE | rule script |
| compute_geoid_offset_raster.py | 14 | 09-14 | DAG | CORE | rule script |
| compute_model_bbox.py | 28 | 09-14 | DAG | CORE | rule script |
| extract_dem.py | 32 | 09-23 | DAG | CORE | rule script |
| extract_dem_mask.py | 23 | 09-23 | DAG | CORE | rule script |
| compute_friction.py | 24 | 09-14 | DAG | CORE | rule script |
| cache_waterlevel_stations.py | 35 | 09-14 | DAG | CORE | rule script |
| extract_boundaries.py | 72 | 10-07 | DAG | CORE | rule script |
| run_aqueduct.py | 160 | 10-05 | DAG | DUPLICATE | twin of `run_aqueduct_cli.py`; 70/121 lines shared; CLI has drifted (465dd93, 10-14) |
| run_aqueduct_cli.py | 260 | 10-14 | HPC | CORE | called by run_simulation_batch.py:56 |
| merge_chunk.py | 20 | 10-09 | DAG | CORE | rule script |
| compute_flood_fraction_chunk.py | 155 | 10-16 | DAG | CORE | rule script |
| prepare_exposure_grid_chunk.py | 20 | 09-14 | DAG | CORE | rule script |
| build_mosaic_vrt.py | 18 | 07-10 | DAG | CORE | only if plots enabled |
| plot_merged_results.py | 63 | 10-09 | DAG | CORE | only if plots enabled |
| generate_aqueduct_jobs.py | 606 | 10-15 | DAG/HPC | CORE | hpc_dispatch rule; imported by generate_hpc_simulation_jobs |
| generate_hpc_preprocess_job.py | 638 | 10-07 | manual (hpc.md) | CORE | HPC step 1 |
| generate_hpc_simulation_jobs.py | 284 | 10-13 | HPC | CORE | sbatch from preprocess job (:501) |
| generate_hpc_postprocess_job.py | 634 | 10-15 | manual (hpc.md) | CORE | HPC step 3 |
| generate_exposure_jobs.py | 495 | 10-07 | HPC imp | CORE | imported by postprocess job :89 |
| generate_hpc_flood_totals_and_plots_job.py | 179 | 10-13 | HPC imp | CORE | imported by postprocess job :90 |
| run_simulation_batch.py | 133 | 10-13 | HPC | CORE | sbatch payload; also .bat |
| run_exposure_pass1.py | 56 | 09-14 | HPC | CORE | generate_exposure_jobs.py:177 |
| run_exposure_pass2.py | 63 | 10-13 | HPC | CORE | generate_exposure_jobs.py:191 |
| reduce_exposure_shares.py | 62 | 08-10 | HPC | CORE | generate_exposure_jobs.py:207 |
| reduce_exposure_write.py | 67 | 09-14 | HPC | CORE | generate_exposure_jobs.py:220 |
| collect_hpc_failures.py | 133 | 09-14 | manual (hpc.md:204) | SUPPORT | step 2b |
| build_delta_run_configs.py | 43 | 10-05 | manual | SUPPORT | generates deltas_*_materialized.yml (run_delta_floodmaps.py:166) |
| calibration/select_country_tiles.py | 190 | 10-08 | manual | UNKNOWN | only a comment in bangkok_chao_phraya.yml:14; docstring says its study "is now complete" |
| debug_avoid_ssp.py | 100 | 09-14 | — | STALE | no references; "One-off diagnostic"; hard-coded P:\ (:12-13) |

### snakemake_workflow/config and docs

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| config/config.yml | 839 | 10-15 | all | CORE | source of truth; but :9-10 default paths `D:/GFM`, `C:/Users/Schlu005/GFM` |
| config/data_catalog_gfm.yml | 549 | 10-13 | all | CORE | :9 `D:/GFM` |
| config/data_catalog_validation.yml | 935 | 10-16 | VAL | CORE | :6 `P:/` path |
| config/config_hpc.yml | 38 | 08-08 | HPC | SUPPORT | tracked, contains personal `/p/…`, `/u/schlumbe/…`; `.example` twin exists |
| config/config_hpc.yml.example | 44 | 08-08 | manual | SUPPORT | template referenced by README:84 |
| config/{bangkok_chao_phraya,thailand_bangkok,calibration_500_tiles,deltas_ssp126,deltas_ssp245,gbr_wales_scotland_friction9}.yml | 40–107 | 10-07..10-09 | WRAP/manual | SUPPORT | study overrides |
| config/*_materialized.yml (7) | ~250 each | 10-07..10-16 | WRAP | STALE (generated) | deltas_* from build_delta_run_configs / prepare_delta_design_runs; other 4 have **no generator**; gbr_* and thailand_* older than their source yml; bake in machine paths |
| config/calibration/sweep/*.yml (14) | 1–10 | 10-08 | — | STALE | no reader; the generator `scripts/calibration/build_run_config.py` cited in .gitignore:5 and Snakefile:117 does not exist |
| snakemake_workflow/hpc.md | 261 | 10-07 | doc | SUPPORT | consistent with code |

### preparation/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| run_preparation.py | 236 | 10-07 | entry | CORE | `ALL_STEPS` orchestrator |
| sync_deltadtm.py | 216 | 10-07 | PREP | CORE | step sync_deltadtm |
| fix_ocean_mask_with_osm_land.py | 163 | 10-05 | PREP | CORE | step fix_ocean_mask |
| build_deltadtm_vrt.py | 441 | 09-14 | PREP | CORE | step build_deltadtm_vrt |
| build_tile_manifest.py | 296 | 10-07 | PREP | CORE | step tile_generation |
| prepare_boundary_conditions.py | 550 | 09-14 | PREP | CORE | step boundary_conditions |
| plot_domain_maps.py | 161 | 10-15 | PREP imp | SUPPORT | imported by build_tile_manifest:84 |
| build_delta_tile_subset.py | 96 | 10-05 | manual | SUPPORT | run_delta_floodmaps, deltas_ssp126.yml |
| build_tile_subset_by_id.py | 55 | 10-05 | manual | DUPLICATE | same skeleton as build_delta_tile_subset.py:45-80; no references |
| convert_norway_stormflo.py | 414 | 10-02 | — | STALE | Norway dropped from validation scope (§9); nothing imports it |
| promote_world_domains.py | 119 | 10-05 | — | STALE | superseded by tile_generation → connectivity_tiling; input gpkg untracked |
| logs/ (11 .log) | 72,358 | 09-14 | — | STALE | historical run logs (Aug), not regenerable, not read by code |

### analysis/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| run_analysis.py | 264 | 10-14 | entry | CORE | orchestrator; also HPC flood_totals job :145 |
| compute_exposure_analysis.py | 1224 | 10-13 | ANA/HPC | CORE | step 1; imported by reduce_exposure_write :31 |
| compute_flood_totals.py | 267 | 10-09 | ANA/HPC | CORE | step 1b |
| plot_burning_ember.py | 451 | 10-14 | ANA | CORE | step 2 |
| plot_tipping_points.py | 250 | 10-16 | ANA | CORE | step 4b |
| plot_extended_pathways.py | 230 | 10-16 | ANA | CORE | step 4c |
| plot_slr_vs_population.py | 458 | 10-14 | ANA | CORE | step 4d (uncommitted edits) |
| plot_timeseries.py | 312 | 10-13 | ANA (off) | SUPPORT | `plot_timeseries: false` |
| plot_world_map.py | 118 | 09-14 | ANA (off) | SUPPORT | disabled in config |
| plot_adaptation_bars.py | 98 | 09-14 | ANA (off) | STALE | disabled; untouched since 09-14; may predate protect_avoid |
| compute_adaptation_pathways.py | 306 | 10-16 | imp | SUPPORT | imported by compute_custom_pathways :35 |
| compute_custom_pathways.py | 196 | 10-16 | manual | SUPPORT | uncommitted edits |
| plot_custom_pathways.py | 299 | 10-16 | manual | SUPPORT | uncommitted edits; supersedes plot_adaptation_pathways |
| plot_adaptation_pathways.py | 171 | 10-16 | manual | DUPLICATE | superseded by plot_custom_pathways (shares `chain_segments`, `plot_entity_panel`, `_block_parts`) |
| compute_tipping_points_summary.py | 285 | 10-16 | manual | SUPPORT | table companion to plot_tipping_points |
| extract_slr_trajectories.py | 143 | 09-14 | manual | SUPPORT | one-time prep (data_catalog_gfm.yml) |
| extract_country_population.py | 191 | 10-07 | manual | SUPPORT | one-time prep |
| compute_delta_flood_extent.py | 532 | 10-16 | manual/imp | SUPPORT | deltas; imported by 2 scripts |
| prepare_delta_design_runs.py | 156 | 10-16 | manual | SUPPORT | run_delta_floodmaps.py:165 |
| add_slr_year_column.py | 108 | 10-07 | manual | SUPPORT | deltas CSV enrichment |
| add_protection_standard_column.py | 127 | 10-07 | manual | SUPPORT | deltas CSV enrichment |
| plot_delta_flood_map.py | 136 | 10-16 | manual | SUPPORT | deltas figure, no caller |
| plot_delta_affected_fraction_timeseries.py | 161 | 10-16 | manual | SUPPORT | deltas figure, no caller |
| compute_bangkok_case_exposure.py | 156 | 10-09 | WRAP | SUPPORT | run_bangkok_case_study.bat |
| plot_bangkok_case_flood_map.py | 178 | 10-09 | WRAP | SUPPORT | .bat |
| plot_bangkok_eai_timeseries.py | 110 | 10-09 | WRAP | SUPPORT | .bat |
| plot_bangkok_exposure_vs_rp.py | 54 | 10-09 | WRAP | SUPPORT | .bat |
| plot_coastrp_stations_map.py | 119 | 10-02 | — | UNKNOWN | no caller; likely methods figure |
| plot_landuse_map.py | 114 | 10-02 | — | UNKNOWN | no caller; likely methods figure |
| plot_waterlevel_by_rp.py | 101 | 10-02 | — | UNKNOWN | no caller; likely methods figure |
| compare_fundy_extreme_forcing.py | 305 | 10-07 | — | STALE | docstring "One-off experiment" |
| validate_adaptation_measures.py | 393 | 10-13 | manual | SUPPORT | diagnostic; writes adaptation_validation_output/ (misplaced under analysis/) |
| summarize_run_timings.py | 67 | 10-15 | manual | SUPPORT | writes run_timings.csv |

### validation/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| run_validation.py | 140 | 10-09 | entry | CORE | orchestrator |
| validate_country.py | 1421 | 10-08 | VAL | CORE (trim) | imported by 5 scripts; ~530 lines (partial + depth-band paths) are ESP/FRA-only → §9 |
| plot_agreement_map.py | 712 | 10-02 | VAL | CORE (trim) | unless --skip-plots; unit-dots + depth_agreement are ESP/FRA-only → §9 |
| run_multi_rp_summary.py | 174 | 10-02 | manual | SUPPORT (trim) | docs/methods_04b, postprocessing.smk:38; calls the partial path (:110) → §9 |
| validate_england_fz2_band.py | 171 | 10-16 | manual | SUPPORT | feeds build_overview_table |
| _rescore_england_fz3.py | 204 | 10-16 | manual | SUPPORT | load-bearing despite `_` prefix (feeds overview/compare tables) |
| _scratch_check_corrupt_exposure_grids.py | 41 | 10-16 | — | STALE | one-off for race fixed 10-15 |
| fetch_england_flood_zones.py | 170 | 10-16 | manual | SUPPORT | one-time download |
| build_sf_bay_benchmark_mosaic.py | 74 | 10-15 | manual | SUPPORT | SF Bay benchmark |
| validate_sf_bay.py | 271 | 10-15 | manual | SUPPORT | SF Bay |
| compare_against_published.py | 174 | 10-16 | manual | SUPPORT | reporting |
| build_overview_table.py | 152 | 10-16 | manual | SUPPORT | reporting |
| *.csv (4) | 3–27 | 10-15..10-16 | — | SUPPORT (output) | regenerable outputs of the 3 reporting scripts; small, arguably keep as results |

### sfincs_tiles/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| run_final_tile.sh | 151 | 10-12 | entry | CORE | per-tile chain |
| rerun_eikonal_calibration.sh | 104 | 10-16 | entry | CORE | batch rerun + dedup + sweep analysis |
| rerun_sfincs_calibration.sh | 60 | 10-15 | entry | CORE | batch rerun |
| generate_calibration_jobs.py | 195 | 10-12 | SF | CORE | stage1/stage2 writer |
| select_calibration_tiles.py | 272 | 10-13 | SF | CORE | stage1 |
| preflight_sfincs_mask.py | 109 | 10-13 | SF imp | CORE | stage1 |
| build_station_boundary_lines.py | 1254 | 10-13 | SF imp | CORE | stage1 + 6 importers |
| regenerate_dem_mask.py | 80 | 10-13 | SF | CORE | run_final_tile.sh |
| build_elevation.py | 284 | 10-13 | SF | CORE | run_final_tile.sh |
| mdt.py | 89 | 10-02 | SF imp | CORE | copy of prepare_boundary_conditions `_load_mdt`/`_nearest_valid_grid` (deliberate, documented) |
| build_roughness.py | 79 | 10-13 | SF | CORE | run_final_tile.sh |
| build_line_forcing.py | 120 | 10-15 | SF | CORE | run_final_tile.sh |
| connected_forcing.py | 131 | 10-15 | SF imp | CORE | build_line_forcing / build_sfincs_tile |
| build_sfincs_tile.py | 561 | 10-15 | SF | CORE | run_final_tile.sh |
| run_calibration_models.py | 253 | 10-16 | SF | CORE | run_final_tile.sh |
| run_sfincs_tile.py | 222 | 10-13 | SF | CORE | run_final_tile.sh |
| sfincs_run.py | 136 | 10-02 | SF imp | CORE | trimmed copy of julius_sfincs_code/sfincs_run.py; 2 unused fns |
| sfincs_grid.py | 82 | 10-13 | SF imp | CORE | own `parse_sfincs_inp` (differs from sfincs_run's) |
| compute_calibration_result.py | 194 | 10-13 | SF | CORE | run_final_tile.sh |
| compute_calibration_result_dedup.py | 253 | 10-13 | SF | CORE | not a twin: Voronoi overlap correction, imports helpers from the original |
| compute_friction_sweep_metrics.py | 59 | 10-13 | SF imp | CORE | sweep analysis |
| calibration_results.py | 135 | 10-13 | SF imp | CORE | |
| flood_agreement.py | 153 | 10-13 | SF imp | CORE | `confusion_counts`/`metrics_from_counts` have drifted from src/validation.py |
| calibration_paths.py | 37 | 10-13 | SF imp | CORE | `decode_waterdepth_cm` diverges from src/rasters.py |
| run_calibration_sweep_analysis.py | 175 | 10-16 | SF | CORE | rerun_eikonal .sh |
| plot_validation_results.py | 353 | 10-16 | SF imp | CORE | also imported by analysis/ |
| plot_sfincs_tile_diagnostics.py | 580 | 10-15 | SF imp | CORE | |
| plot_worst_tiles_panel.py | 241 | 10-15 | SF imp | CORE | |
| gfm_config.py | 44 | 10-02 | SF imp | CORE | 20+ importers |
| retry_io.py | 33 | 10-02 | SF imp | DUPLICATE | near-copy of `config_utils.retry_transient_io` (2 lines differ) |
| check_calibration_progress.py | 114 | 10-12 | manual | SUPPORT | monitor |
| build_gebco_vrt.py | 104 | 10-02 | manual | SUPPORT | one-time; data_catalog_gfm.yml |
| scan_nodata_frame_leak.py | 119 | 10-16 | — | SUPPORT (diagnostic) | no caller; tile-631 leak diagnostic (now fixed) |

### calibration_studies/, julius_sfincs_code/, scratch/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| calibration_studies/test_obstacle_coupling_calibration.py | 534 | 10-06 | manual | SUPPORT | study CLI, not a pytest test; cited by methods_03 |
| calibration_studies/test_sweep_budget_calibration.py | 399 | 10-06 | manual | SUPPORT | study CLI, not a pytest test |
| calibration_studies/generate_calibration_batch_jobs.py | 268 | 10-08 | manual | SUPPORT | writes sbatch for test_*; stale comment :209 |
| calibration_studies/select_calibration_tiles.py | 135 | 10-06 | manual | SUPPORT | name collision with sfincs_tiles version (different algorithm) |
| calibration_studies/aggregate_wet_tiles.py | 80 | 10-06 | manual | SUPPORT | methods_03 |
| calibration_studies/plot_sweep_budget_convergence.py | 71 | 10-06 | manual | SUPPORT | methods_03 |
| calibration_studies/plot_sweep_calibration_bands.py | 255 | 10-06 | manual | SUPPORT | methods_03 |
| calibration_studies/plot_sweep_time_vs_size.py | 126 | 10-08 | manual | SUPPORT | methods_03 |
| calibration_studies/plot_obstacle_coupling_calibration_bands.py | 190 | 10-06 | manual | SUPPORT | methods_03 |
| calibration_studies/diagnose_large_residual.py | 159 | 10-06 | manual | SUPPORT | one-off; cited in flood_model docstring |
| julius_sfincs_code/13_build_sfincs.{py,smk} | 536/88 | 09-20 | — | STALE | external basin workflow; imports nonexistent `src.log`; config keys absent |
| julius_sfincs_code/13_build_sfincs_skeleton.{py,smk} | 728/130 | 09-20 | — | STALE | same |
| julius_sfincs_code/14_run_spinup.{py,smk} | 343/61 | 09-20 | — | STALE | same |
| julius_sfincs_code/16_run_event.{py,smk} | 201/50 | 09-20 | — | STALE | same |
| julius_sfincs_code/sfincs_run.py | 158 | 09-20 | — | STALE | upstream of sfincs_tiles/sfincs_run.py |
| julius_sfincs_code/sfincs_restart.py | 161 | 09-20 | — | STALE | no references |
| scratch/dtm_seeger_pronk.py | 13 | 09-14 | — | STALE | one-off download; no docstring |

### tests/

| path | lines | last | reach | cat | evidence |
|---|---|---|---|---|---|
| aqueduct_runner_validation/validate_aqueduct_runner.py | 176 | 09-21 | test | SUPPORT | aqueduct_runner, rasters; synthetic |
| boundary_station_search_validation/validate_station_search.py | 92 | 09-14 | test | SUPPORT | boundaries; synthetic |
| boundary_waterlevel_encoding_validation/validate_waterlevel_encoding.py | 109 | 09-14 | test | SUPPORT | boundaries, rasters; synthetic |
| coastline_mask_validation/validate_coastline_mask.py | 100 | 09-14 | test | SUPPORT | flood_model; synthetic |
| eikonal_kernel_validation/validate_eikonal_kernel.py | 293 | 10-07 | test | SUPPORT | eikonal; synthetic |
| flood_depth_dense_seed_path_validation/validate_seed_path.py | 168 | 09-14 | test | SUPPORT | flood_model; synthetic |
| flood_model_memory_validation/validate_flood_model_memory.py | 92 | 09-21 | test | SUPPORT | flood_model, eikonal |
| neighbor_wave_seeding_validation/validate_neighbor_wave_seeds.py | 217 | 09-14 | test | SUPPORT | boundaries, merge, rasters |
| rasters_encoding_validation/validate_rasters_encoding.py | 308 | 10-02 | test | SUPPORT | rasters (12 cases) |
| tiles_clamp_window_validation/validate_clamp_window.py | 77 | 09-21 | test | SUPPORT | tiles |
| flood_extent_validation/test_metrics.py | 192 | 10-02 | test | SUPPORT | validation; pytest-collectable |
| flood_extent_validation/test_synthetic_end_to_end.py | 292 | 10-02 | test | SUPPORT | validation, plotting; pytest-collectable |
| sfincs_antimeridian_validation/… | 83 | 10-13 | test | SUPPORT | build_sfincs_tile; needs hydromt_sfincs |
| sfincs_landmask_doublecheck_validation/… | 106 | 09-21 | test | SUPPORT | run_sfincs_tile |
| sfincs_reprojection_validation/… | 121 | 09-21 | test | SUPPORT | run_sfincs_tile |
| sfincs_roughness_range_validation/… | 121 | 09-21 | test | SUPPORT | build_roughness |
| sfincs_subgrid_nearest_validation/… | 164 | 09-21 | test | SUPPORT | build_sfincs_tile |
| sfincs_mdt_sign_validation/… | 170 | 10-13 | test | SUPPORT | build_elevation, build_line_forcing |
| sfincs_hydrograph_truncation_validation/validate_truncation_window.py | 125 | 10-13 | test | SUPPORT | needs P:\ (:74); docstring cites deleted build_boundary_forcing.py |
| sfincs_synthetic_end_to_end/test_sfincs_smoke.py | 155 | 10-13 | test | SUPPORT | needs P:\ + SFINCS exe despite the name |
| slr_fingerprint_scenario_comparison/*.py (+png) | 130 | 09-28 | manual | SUPPORT | analysis script, no asserts |
| _list_tiles_by_size.py | 46 | 09-14 | — | STALE | hard-coded `D:/GFM/model_outputs` |
| norway_diagnostics/ (12 py + csv/json) | ~1,600 | 09-14 | — | STALE | Norway dropped from validation scope (§9) → delete, with its comment pointers |
| connectivity_tiling_prototype/output/ | ~8,050 | 10-05 | — | STALE (delete) | prototype scripts already removed |

### docs/

| path | lines | last | cat | evidence |
|---|---|---|---|---|
| methods_01_tile_processing_and_waterlevels.md | 211 | 10-07 | SUPPORT | refs resolve |
| methods_02_flood_depth.md | 496 | 10-13 | SUPPORT | refs resolve |
| methods_03_calibration_sensitivity.md | 129 | 10-08 | SUPPORT | refs resolve |
| methods_04a_SFINCSvalidation.md | 120 | 10-13 | SUPPORT | omits dedup + rerun_*.sh; `sfincs_calibration_bline`, `pct_within_0` not in code |
| methods_04b_MapsValidation.md | 161 | 10-02 | SUPPORT | OK |
| sfincs_validation_pipeline_preparation.md | 228 | 09-27 | STALE | 11 referenced scripts deleted in 7b6f7f1 (10-13) |
| python_vs_julia_qa.md | 215 | 07-30 | STALE | deleted in Phase 4 (user decision); its evidence citations in code removed |
| hydromt_sfincs_bug_*.md (2) | 25/51 | 09-24 | SUPPORT | upstream bug reports |
| generate_eikonal_solver_examples.py | 471 | 10-08 | SUPPORT | generates eikonal_example_*.png |
| eikonal_example_{rounds,block_sweep,obstacle_coupling}.png | — | 09-25..10-08 | SUPPORT | generated |
| _zoom_panel_a1.png, ridge_gap_test_figure.png | — | 08-08, 09-27 | STALE | referenced by nothing |

### Other tracked non-source

| path | cat | evidence |
|---|---|---|
| adaptation_validation_output/ (17 png/pdf) | STALE (output) | written by validate_adaptation_measures.py:286 |
| batch_logs/ (20 logs, 07-27) | STALE | no writer in current code |
| environment.yml | SUPPORT | `gfm` env; cites missing pixi.toml, memory.md, tests/river_mouth_tile_validation, tests/build_calibration_report_xlsx.py |

---

## 4. Code quality findings (ranked by impact)

### 4.1 Duplicated logic (correctness risk)
1. **`run_aqueduct.py` vs `run_aqueduct_cli.py`** (`snakemake_workflow/scripts/`): duplicated per-tile solve/skip/OOM/placeholder logic. The headers say "keep the two in sync", but they have already drifted: the CLI gained `_stage_file_locally` and changed the hop-0 boundary read in 465dd93 (10-14); `run_aqueduct.py` was last touched 10-05. Local and HPC runs can therefore produce different results.
2. **`_build_chunk_grid` ×3**: `Snakefile:171`, `generate_hpc_postprocess_job.py`, `check_postprocess_progress.py`. If one copy drifts, chunk IDs stop matching between the DAG and HPC.
3. **Metric copies across environments**: `sfincs_tiles/flood_agreement.py` `confusion_counts`/`metrics_from_counts` vs `src/validation.py`; `sfincs_tiles/calibration_paths.py decode_waterdepth_cm` vs `src/rasters.py`; `sfincs_tiles/retry_io.py` vs `src/config_utils.py retry_transient_io`. *Checked 2026-10-17 (Q5): no numeric divergence.* The HR/FAR/CSI/bias formulas are identical. The drift is in the interface and outputs only:
   - **Return shape**: a 3-tuple without `tn`, vs a 4-tuple.
   - **Hit-rate label**: `HT` vs `HR`.
   - **Extra metrics**: `EB`/`EB_ratio` are absent from the sfincs copy.
   - **Nodata decoding**: NaN vs `AQUEDUCT_NODATA`. Both read as dry.
   - **Retry**: the sfincs copy catches plain `OSError`, the src one a curated exception tuple.

   Still worth consolidating, so the two can't drift into a real difference later.
4. **HPC sbatch helpers copy-pasted**:
   - `_stage_configfile_lines` ×4 (`generate_aqueduct_jobs`, `generate_hpc_flood_totals_and_plots_job`, `generate_hpc_postprocess_job`, `generate_hpc_preprocess_job`)
   - `_account_line` ×4
   - `_retry_wrapper_lines` ×2
   - `_stage_tile_grid_lines` ×2
   - `_stage_file_locally` ×2 (`generate_hpc_postprocess_job`, `run_aqueduct_cli`)
   - Paired `generate_wave_dispatch`/`generate_resume_dispatch` (`generate_aqueduct_jobs.py:253/436`) and `generate_exposure_dispatch`/`generate_exposure_resume_dispatch` (`generate_exposure_jobs.py:91/285`) are structurally duplicated.
5. **analysis helpers**:
   - `_read_band` ×3 (compute_exposure_analysis, compute_flood_totals, compute_bangkok_case_exposure)
   - `_read_band_and_transform`, `_round_val` ×2
   - `_growth_matrix_path` ×3 (compute_adaptation_pathways, compute_tipping_points_summary, plot_tipping_points)
   - `_growth_path`, `_eai_from_interp`, `_first_crossing`, `_people` ×2 each
   - `chain_segments`, `plot_entity_panel`, `_block_parts` in both plot_adaptation_pathways and plot_custom_pathways
6. **Orchestrator `_run`** ×3: `analysis/run_analysis.py`, `validation/run_validation.py`, `run_delta_floodmaps.py`.
7. **Progress monitors**: `_count_progress`/`_print_report` duplicated in `check_preprocess_progress.py` and `check_simulation_progress.py`; `_bar` ×3.
8. **SFINCS input parsing**: `parse_sfincs_inp` ×3 (`sfincs_tiles/sfincs_grid.py`, `sfincs_tiles/sfincs_run.py`, `julius_sfincs_code/sfincs_run.py`); the two sfincs_tiles versions differ.
9. **calibration_studies**: `load_tile` ×3; `_full_depth_array`, `_band_stats`, `_plot_band` ×2.
10. **preparation**: `build_tile_subset_by_id.py:27-45` and `build_delta_tile_subset.py:45-80` share the same skeleton.
11. **MDT helpers**: `sfincs_tiles/mdt.py` `_load_mdt`/`_nearest_valid_grid` copy `preparation/prepare_boundary_conditions.py`. Deliberate (separate env) and documented.
12. **sys.path boilerplate**: 37 of 47 files in analysis/validation repeat `sys.path.insert(... "src")`.

### 4.2 Hard-coded paths
- **Tracked config defaults**:
  - `snakemake_workflow/config/config.yml:9-10` (`D:/GFM`, `C:/Users/Schlu005/GFM`, a different username)
  - `data_catalog_gfm.yml:9` (`D:/GFM`)
  - `data_catalog_validation.yml:6` (`P:/…`)
  - `config_hpc.yml` (`/p/11212688…`, `/u/schlumbe/gfm_code`), tracked
  - all 7 `*_materialized.yml` bake in `P:/` and `C:/Users/schlumbe/` paths
- **Hydrax paths repeated 5×**:
  - `sfincs_tiles/generate_calibration_jobs.py:42-45`
  - `sfincs_tiles/run_final_tile.sh:31-37`
  - `sfincs_tiles/rerun_eikonal_calibration.sh:30-34`
  - `sfincs_tiles/rerun_sfincs_calibration.sh:26-28`
  - `calibration_studies/generate_calibration_batch_jobs.py:74`
- **validation/ inputs pinned to P:\ instead of `validation.output_dir`**:
  - `build_overview_table.py:24-28` and `compare_against_published.py:27-30` (the same constants duplicated)
  - `validate_sf_bay.py:69,96,100` (one is a specific chunk tif)
  - `build_sf_bay_benchmark_mosaic.py:28-29`
  - `fetch_england_flood_zones.py:49`
  - `_scratch_check_corrupt_exposure_grids.py:18-19`
- **`all_deltas.shp` path outside config**:
  - `analysis/compute_delta_flood_extent.py:121` (`DELTAS_PATH = P:\…`; re-imported by add_protection_standard_column.py:54 and prepare_delta_design_runs.py)
  - `preparation/build_delta_tile_subset.py:41` (a separate copy)
- **Other code**:
  - `preparation/plot_domain_maps.py:55,57,58`
  - `preparation/convert_norway_stormflo.py:97`
  - `snakemake_workflow/scripts/debug_avoid_ssp.py:12-13`
  - `tests/sfincs_hydrograph_truncation_validation/validate_truncation_window.py:74`
  - `tests/_list_tiles_by_size.py:12`
- **Launchers**:
  - `run_bangkok_case_study.bat:16-18`
  - `run_thailand_bangkok.bat:15-17`
  - `run_thailand_bangkok.ps1:19-21`
  - These hard-code `C:\Users\schlumbe\…\python.exe`/`snakemake.exe`; the thailand pair also hard-codes the repo path (`%~dp0` would do).
- **Docstrings**: `sfincs_tiles/build_sfincs_tile.py:9`, `run_sfincs_tile.py:7` (Windows user env path in usage text).

### 4.3 Functions over ~100 lines
| file:line | function | lines |
|---|---|---|
| snakemake_workflow/scripts/generate_hpc_preprocess_job.py:202 | main | 433 |
| validation/validate_country.py:412 | validate_country (no docstring) | 346 |
| analysis/compute_exposure_analysis.py:907 | main | 315 |
| validation/validate_country.py:945 | validate_country_national_coverage | 290 |
| snakemake_workflow/scripts/generate_hpc_postprocess_job.py:353 | main | 278 |
| sfincs_tiles/build_sfincs_tile.py:277 | build_sfincs_tile | ~246 |
| src/flood_model.py:222 | flood_depth_dense | 225 |
| snakemake_workflow/scripts/generate_hpc_simulation_jobs.py:85 | main | 196 |
| snakemake_workflow/scripts/generate_exposure_jobs.py:91 | generate_exposure_dispatch | 192 |
| src/visualization.py:316 | plot_burning_ember | 189 |
| analysis/run_analysis.py:74 | main (8 copy-paste step blocks) | 188 |
| validation/validate_country.py:759 | validate_country_depth_bands | 185 |
| snakemake_workflow/scripts/generate_aqueduct_jobs.py:253 | generate_wave_dispatch | 181 |
| validation/plot_agreement_map.py:527 | plot_country | 167 |
| snakemake_workflow/scripts/run_aqueduct_cli.py:94 | main | 163 |
| src/merge.py:184 | merge_tile_rasters_chunk | 156 |
| snakemake_workflow/scripts/generate_aqueduct_jobs.py:436 | generate_resume_dispatch | 156 |
| analysis/plot_extended_pathways.py:73 | main | 155 |
| snakemake_workflow/scripts/generate_exposure_jobs.py:285 | generate_exposure_resume_dispatch | 155 |
| analysis/plot_tipping_points.py:100 | main | 148 |
| analysis/compute_adaptation_pathways.py:159 | main | 145 |
| src/rasters.py:543 | extract_dem | 141 |
| sfincs_tiles/build_station_boundary_lines.py:877 | build_tile | ~134 |
| analysis/plot_burning_ember.py:316 | main | 133 |
| preparation/build_tile_manifest.py:166 | run | 131 |
| preparation/prepare_boundary_conditions.py:249 | compute_slr_fingerprints | 131 |
| analysis/compute_custom_pathways.py:64 | main | 130 |
| analysis/compute_tipping_points_summary.py:153 | main | 130 |
| src/eikonal.py:270 | solve_eikonal_dense | 125 |
| calibration_studies/test_obstacle_coupling_calibration.py:298 | run_tile_trace | ~125 |
| run_pipeline.py:74 | main | 124 |
| calibration_studies/generate_calibration_batch_jobs.py:145 | main | ~122 |
| analysis/compute_delta_flood_extent.py:411 | main | 119 |
| preparation/build_deltadtm_vrt.py:299 | _build_vrt | 119 |
| snakemake_workflow/scripts/compute_flood_fraction_chunk.py:31 | compute_flood_fraction | 118 |
| analysis/compute_flood_totals.py:148 | main | 117 |
| src/boundaries.py:241 | collect_neighbor_wave_seeds | 115 |
| validation/validate_england_fz2_band.py:57 | main | 112 |
| src/rasters.py:205 | compute_model_bbox | 111 |
| analysis/validate_adaptation_measures.py:281 | main | 110 |
| preparation/run_preparation.py:127 | main | 110 |
| src/boundaries.py:116 | filter_stations_by_ocean_connectivity | 107 |
| src/connectivity_tiling.py:152 | _mosaic_nearest_coarse | 107 |
| src/aqueduct_runner.py:28 | run_aqueduct_python | 106 |
| validation/_rescore_england_fz3.py:58 | _score_one_rp | 106 |
| validation/run_multi_rp_summary.py:67 | main | 105 |
| analysis/compute_exposure_analysis.py:731 | pass2_all_tasks | 103 |
| calibration_studies/test_sweep_budget_calibration.py:297 | main | ~101 |

Module size: `sfincs_tiles/build_station_boundary_lines.py` (1254), `validation/validate_country.py` (1421), `analysis/compute_exposure_analysis.py` (1224), `src/rasters.py` (1087), `src/validation.py` (1037).

### 4.4 Missing or misleading docstrings
- **Coverage**: module docstrings are excellent almost everywhere. Exceptions: `src/__init__.py` (empty), `scratch/dtm_seeger_pronk.py`.
- **Missing on public functions**: mostly CLI `main()` (≈19 in snakemake_workflow/scripts) and about 40 in sfincs_tiles:
  - `build_sfincs_tile`, `build_tile`, `compute_result`, `cmd_stage1/2`, `cmd_pool/screen/select`, `preflight_tile`, `regenerate_dem_mask`, `scan_tile`, `build_manning_n`, `build_gebco_vrt`, `reproject_to_4326`, plot_*
- **Notable gaps elsewhere**:
  - `validation/validate_country.py:412 validate_country`
  - `snakemake_workflow/scripts/collect_hpc_failures.py` (3 fns)
  - `src/connectivity_tiling.py` `ConnectivityConfig`, `Domain`, `bbox_to_polygon`
  - `analysis/plot_custom_pathways.py:107,145,210,236`
  - `analysis/plot_adaptation_pathways.py:57,91`
  - `analysis/compute_exposure_analysis.py:156 ChunkData`
  - `run_delta_floodmaps.py:158 run_scenario`
  - `validation/build_overview_table.py:61,111`
  - `validation/fetch_england_flood_zones.py:84`
  - preparation: build_tile_manifest.py:87,92,126,166; sync_deltadtm.py:151,181; build_deltadtm_vrt.py:299,418; convert_norway_stormflo.py:205,213,364
- **Misleading**:
  - `src/config_utils.py:141`: "Identical logic to the root Snakefile's own `_expand_paths`", but the Snakefile imports this function (Snakefile:88).
  - `src/flood_extent.py`: the module name implies flood-extent code that has been removed.
  - Usage lines still point to `snakemake_workflow/analysis/…` / `snakemake_workflow/validation/…` / `snakemake_workflow/preparation/…` (pre-flat layout): `analysis/run_analysis.py:23`, `validation/run_validation.py:10`, `validation/plot_agreement_map.py:54`, `analysis/plot_adaptation_bars.py:11`, `plot_burning_ember.py:29`, `plot_timeseries.py:16`, `plot_world_map.py:13`, `extract_country_population.py:30`, `extract_slr_trajectories.py:18`, `preparation/run_preparation.py:62`.
  - `preparation/run_preparation.py:58`: `extract_boundaries.py` location is stale.
  - `analysis/run_analysis.py` docstring: lists plot_adaptation_bars as a normal step and omits the pathways/tipping-summary scripts.
  - `calibration_studies/generate_calibration_batch_jobs.py:209`: references deleted `generate_calibration_array_job.py`.
  - `tests/sfincs_hydrograph_truncation_validation/validate_truncation_window.py`: docstring cites deleted `build_boundary_forcing.py`.

### 4.5 Dead code and branches
- No `if False`, commented-out blocks or TODO/FIXME markers found anywhere.
- Dead functions (§2e): `src/exposure_analysis.py:205`, `src/map_style.py:111`, `src/merge.py:66`, `sfincs_tiles/sfincs_run.py` (`parse_sfincs_inp`, `forward_geometry_files`).
- ~~Test-only functions: `src/rasters.py:404`, `:493` (no config key enables them).~~ **Correction (Phase 3, 2026-10-18): wrong. Both are called unconditionally by `extract_dem` (rasters.py:661) and `extract_dem_mask` (:731, :734), i.e. by every production preprocessing run. They are CORE and stay; their tests stay.**
- Unused imports:
  - `snakemake_workflow/scripts/generate_aqueduct_jobs.py:65 os`
  - `src/exposure.py:12 numpy`
  - `src/merge.py:30 from_bounds`
  - `src/rasters.py:22 shapely_box`
  - `src/visualization.py:34 cm`
- Unused parameter: `src/connectivity_tiling.py:590 dem_index` (its docstring says so).
- Steps switched off in config but kept in the orchestrator: `analysis/run_analysis.py` plot_adaptation_bars / plot_timeseries / plot_world_map.
- `generate_hpc_preprocess_job.py:526` emits `_retry_wrapper_lines()` "harmless if unused".
- Retry logic exists in three forms: `Snakefile:67` (`_retrying_makedirs`, monkeypatches `os.makedirs`), `src/config_utils.retry_transient_io`, `sfincs_tiles/retry_io.py`.

### 4.6 Inconsistent naming
- **"aqueduct" names for the Python eikonal solver**: `run_aqueduct`, `aqueduct_runner`, `generate_aqueduct_jobs`.
- **Job-generator names**: `generate_hpc_*_job` (singular) vs `generate_hpc_simulation_jobs` / `generate_exposure_jobs` (plural, exposure has no `hpc_`).
- **Similar module names**:
  - `src/exposure.py` vs `src/exposure_analysis.py`
  - `plotting.py` / `visualization.py` / `map_style.py` (pipeline plots / analysis figures / style)
- **Duplicate filename `select_calibration_tiles.py`**: `sfincs_tiles/` and `calibration_studies/` hold different algorithms under the same name.
- **`test_*.py` that aren't tests**: `calibration_studies/test_*.py` are study CLIs that pytest would collect from the repo root, and they import heavy dependencies.
- **Underscore prefix used both ways**: `validation/_rescore_england_fz3.py` is load-bearing; `validation/_scratch_*` is throwaway.
- **Test folder names**: `tests/sfincs_synthetic_end_to_end/` needs real P:\ data and a SFINCS exe.
- **Config usernames**: `config.yml` uses `Schlu005`; everything else uses `schlumbe`.

---

## 5. Documentation state

**What exists.** `README.md` (110 lines), `snakemake_workflow/hpc.md` (consistent with code), and docs/methods_01–04b (references resolve, except as noted). Module docstrings are present and detailed in nearly every .py. There is no README in `sfincs_tiles/`, `analysis/`, `validation/`, `preparation/`, `calibration_studies/`, `tests/` or `julius_sfincs_code/`.

**Outdated relative to the code.**
- **README.md**
  - :40 and :109 cite `snakemake_workflow/memory.md`, which does not exist. `environment.yml` cites it too.
  - :98 cites `tests/martinique_diagnostics/`, which does not exist.
  - :42 says calibration studies live in `tests/`; they are in top-level `calibration_studies/`.
  - The layout section omits `sfincs_tiles/`, `julius_sfincs_code/`, `calibration_studies/`, `run_delta_floodmaps.py` and the `.bat`/`.ps1` launchers.
  - It says `analysis/` is "NOT part of the DAG". The HPC exposure path imports `analysis/compute_exposure_analysis.py`.
  - It says `snakemake_workflow/` holds "ONLY what Snakemake itself touches". `scripts/` also holds manual and one-off tools.
  - :32/:74: validation is described as `--countries ESP FRA NOR`. The current GBR/England, SF Bay and CAN work runs through standalone scripts that README doesn't mention.
  - :38-39 vs :84: unclear whether `config_hpc.yml` or `.example` is authoritative.
- **`docs/sfincs_validation_pipeline_preparation.md`**: references 11 scripts deleted in 7b6f7f1:
  - `select_validation_tiles.py`, `select_test_tiles.py`, `generate_validation_batch_jobs.py`
  - `submit_*.sh`, `run_one_tile.sh`
  - `build_boundary_forcing.py`, `run_models_sequential.py`, `postprocess_sequential.py`, `postprocess_tile_summary.py`, `run_eikonal_on_sfincs_subgrid.py`

  It is nevertheless the only place the `hydromt-sfincs-dev` env is described.
- **`docs/methods_04a_SFINCSvalidation.md`**: does not cover `compute_calibration_result_dedup.py` (overlap correction) or the `rerun_*.sh` drivers. `sfincs_calibration_bline` and `pct_within_0` appear nowhere in code.
- **`docs/python_vs_julia_qa.md`**: cites `julia_vs_python_timing.html`, `python_full_pipeline_timing.csv` and `julia_timing_vs_size.csv` (absent). It predates 6 solver commits (block sweep, obstacle coupling), so the timing numbers are likely stale.
- **`environment.yml`**: "Mirrors pixi.toml", but pixi.toml is absent. It cites `tests/river_mouth_tile_validation`, `tests/build_calibration_report_xlsx.py` and `memory.md`, all absent.
- **`.gitignore:5` and `Snakefile:117`**: cite `scripts/calibration/build_run_config.py`, which is absent.

**What a new contributor could not run.**
1. **Anything in `sfincs_tiles/`** or its 8 tests: `hydromt_sfincs` is not in `environment.yml`, and no env file exists for `hydromt-sfincs-dev`. Its interpreter paths are hard-coded to `/u/schlumbe/.conda/envs/…`.
2. **The environment naming**: nothing documents that the `gfm` env is called `gfm_python_preprocessing` on the Windows machine, or that sfincs_tiles plotting must run in it because hydromt-sfincs-dev's matplotlib is broken.
3. **HPC dispatch with their own account or paths**: `config_hpc.yml` is tracked with personal paths, and the sfincs `.sh` scripts hard-code Hydrax paths in 5 places.
4. **The default config off this machine**: `config.yml` defaults to `D:/GFM` and `C:/Users/Schlu005/GFM`, so a contributor must create `config_local.yml`. The README explains this but gives no template.
5. **The tests as a suite**: no pytest in the env, no config, and only 3 files match `test_*.py`.
6. **Validation reporting and the deltas analysis**: both need the P:\ share at hard-coded paths.
7. **Re-materializing 4 of the 7 `*_materialized.yml`**: no generator script exists.

---

## 6. Test coverage

**State.** There is no harness. Tests are standalone scripts with plain `assert` and a `__main__` runner. Most `validate_*.py` files already contain `test_*` functions, so pytest would collect them with `python_files = test_*.py validate_*.py`. pytest could not be run during the audit: it is not in the env, and the PATH `python` is the Windows Store stub.

| Area | Tested (by) | Untested |
|---|---|---|
| src/ CORE | aqueduct_runner, boundaries (3), rasters (4), flood_model (3), eikonal (2), merge, tiles, validation (2), plotting | **connectivity_tiling, exposure, protection, vertical_datum**; config_utils and flood_extent are only exercised indirectly |
| src/ SUPPORT | — | exposure_analysis, population_growth, population_extension, pathways, visualization, map_style, tile_split |
| snakemake_workflow/scripts | — | **all**, including run_aqueduct vs run_aqueduct_cli parity, `_build_chunk_grid` consistency, and the sbatch generators |
| preparation/ | prepare_boundary_conditions (indirectly, via the slr comparison; no asserts) | build_tile_manifest, sync_deltadtm, build_deltadtm_vrt, fix_ocean_mask |
| sfincs_tiles/ CORE | build_sfincs_tile, run_sfincs_tile, sfincs_run, gfm_config, build_elevation, build_line_forcing, build_roughness (all need hydromt_sfincs) | **flood_agreement, compute_calibration_result(_dedup)**, build_station_boundary_lines, connected_forcing, mdt, calibration_paths, calibration_results, compute_friction_sweep_metrics, preflight_sfincs_mask, regenerate_dem_mask, select_calibration_tiles, run_calibration_* |
| analysis/, validation/ scripts | — | all (validate_country is only exercised via src/validation tests) |

**Tests that need external data**: `validate_truncation_window.py` (P:\, SKIPs otherwise), `test_sfincs_smoke.py` (P:\ + SFINCS exe), `slr_fingerprint_scenario_comparison`, `norway_diagnostics/`. Everything else is synthetic.

**Highest-value gaps**: `flood_agreement` vs `src/validation` metric parity; `compute_calibration_result`; `connectivity_tiling` (it defines the tile set); `exposure`/`protection` (they feed the EAI numbers); `run_aqueduct` vs `run_aqueduct_cli` parity.

---

## 7. Questions: answered 2026-10-17

| # | Topic | Decision | Consequence for the plan |
|---|---|---|---|
| 1 | Entry points | Incomplete: the eikonal sensitivity studies were missing; `run_pipeline.py` needs a descriptive name | Entry-point table updated (top of file). Rename `run_pipeline.py` to `run_eikonal_dag.py` (see open question A). Merge the two `rerun_*.sh` (B) |
| 2 | `julius_sfincs_code/` | **Delete** | Phase 2. Replace the reference in `sfincs_tiles/sfincs_run.py`'s docstring with a provenance line naming the deleting commit |
| 3 | `run_aqueduct.py` vs `run_aqueduct_cli.py` | Keep both entry points, unify the logic | Phase 5b |
| 4 | Hop-0 boundary read (465dd93) | **Meant for both** | **Correction (Phase 5b, 2026-10-18):** 465dd93 only copies the boundaries `.gpkg` to node-local scratch before opening it (a GDAL/SQLite lock hang on Hydrax compute nodes). It reads the same file, so local and HPC results never differed. Phase 5b gave both entry points the shared `aqueduct_runner.run_tile_scenario`; staging stays HPC-only (`stage_boundaries=True` in the CLI) |
| 5 | Metric drift | Unintentional | See the finding below the table |
| 6 | `calibration/sweep/*.yml` | Delete | Phase 2 |
| 7 | `*_materialized.yml` | Gitignore + one generic `materialize_configs.py` | Phase 1 (untrack) + Phase 6 (generator) |
| 8 | `config_hpc.yml` | Untrack, keep `.example` | Phase 1 |
| 9 | Tracked outputs/logs | Untrack all listed; keep `validation/*.csv` | Phase 1 |
| 10 | `triplet_comparison_CLEAN.csv` | **Untrack and delete** | Phase 1 (`git rm`, not `--cached`) |
| 11 | `plot_adaptation_pathways.py`, `plot_adaptation_bars.py` | Retire after `plot_custom_pathways.py` is committed | Phase 3. Precondition: you commit your 3 modified analysis files first |
| 12 | 3 uncalled analysis figures | Move to a figures folder | Phase 2: move to `docs/figure_scripts/` together with `docs/generate_eikonal_solver_examples.py` |
| 13 | One-off diagnostics | **Delete** (not archive) | Phase 2. Covers `debug_avoid_ssp.py`, `compare_fundy_extreme_forcing.py`, `_scratch_check_corrupt_exposure_grids.py`, `scan_nodata_frame_leak.py` (nothing is running), `tests/_list_tiles_by_size.py`, `scratch/dtm_seeger_pronk.py` (so `scratch/` goes too), `promote_world_domains.py` |
| 14 | `norway_diagnostics/`, `connectivity_tiling_prototype/output/` | ~~Move to `archive/`~~ → **delete both** (no archive dir, 2026-10-17). **Superseded for norway_diagnostics** by the validation scope reduction (§9): Norway is dropped, so delete it | Phase 2: delete `connectivity_tiling_prototype/output/`. Phase 2b: delete `norway_diagnostics/` and its comment pointers (config.yml:789, data_catalog_gfm.yml:233, README:99) |
| 15 | `select_calibration_tiles.py` collision, `select_country_tiles.py` | Rename the calibration_studies one; move the other to `preparation/` | Phase 7 |
| 16 | 2 test-only `rasters.py` functions | ~~Delete, with their test cases~~ **Kept.** The audit finding was wrong: `extract_dem` / `extract_dem_mask` call both in production (found in Phase 3) | none |
| 17 | pytest + ruff + minimal `pyproject.toml` | Yes | Phase 0 |
| 18 | Environments | Add `environment_sfincs.yml`. `gfm_python_preprocessing` is the local (Windows) name of the `gfm` env; Hydrax calls it `gfm` | Phase 0 (env file). Phase 4: docs call the env `gfm` and mention the local alias. Phase 6: launchers read the interpreter from an env var instead of hard-coding either name. Local checks use `C:/Users/schlumbe/AppData/Local/miniforge3/envs/gfm_python_preprocessing/python.exe` |
| 19 | Renames | In scope, **including `aqueduct` → `eikonal`** | New Phase 7a (below). See open questions C and D |
| 20 | Running work | Nothing running | No freeze on `sfincs_tiles/` |

**Q5: which results came from which metric implementation?**

None of the published numbers depend on which copy computed them. The HR/HT, FAR, CSI and bias formulas in `sfincs_tiles/flood_agreement.py` and `src/validation.py` are identical; the drift is interface only (§4.1.3). The `decode_waterdepth_cm` and `retry_transient_io` copies don't change any numbers either.

The scoring does differ by output tree, but by design:

| Output directory (P:\…/modelling/) | Produced by | Scoring |
|---|---|---|
| `calibration_roughness_SFINCS/`, `calibration_roughness_SFINCS_dedup/` | `sfincs_tiles/compute_calibration_result(_dedup).py` → `flood_agreement` | Hard counts: eikonal vs SFINCS, both binarised at `WET_THRESHOLD_M`. Hit rate is labelled **`HT`** |
| `validation/{CAN,DNK,ESP,FIN,FRA,GBR,NOR}/`, `validation/*.csv` | `validation/validate_country.py` etc. → `src/validation` | **Soft** counts (`confusion_counts_soft`, benchmark coverage fraction as credit). Hit rate is labelled **`HR`**; EB/EB_ratio included |

There is one real "different method" risk. Soft scoring replaced hard τ=0.5 scoring in national-map validation; the docstring dates this to 2026-09-30, and it was committed in `26d1f1a` on 2026-10-02 11:40. Two files on P:\ predate that commit: `validation/FRA/metrics_FRA_RP250_SLR_0.csv` (2026-09-30 21:17) and `depth_metrics_FRA_RP250_SLR_0.csv` (21:23). The commit is later than the files, so the code that wrote them may or may not have had soft scoring yet. Every other validation metrics CSV is dated 2026-10-09 or later and is soft-scored. If FRA RP250 feeds `multi_rp_summary.csv` (dated 10-16), re-run FRA RP250 to be safe. The docstring says soft scoring reproduces τ=0.5 "almost exactly", so any difference should be small.

### Follow-up questions A–E: answered 2026-10-17

- **A. Rename `run_pipeline.py` → `run_eikonal_dag.py`.** Yes; done in Phase 7a.
- **B. Merge the two `rerun_*.sh` into `rerun_calibration.sh --part {eikonal,sfincs}`.** Yes; done in Phase 6.
- **C. Config keys: is "rename, then pull on Hydrax" enough?** Yes, with one gotcha (in bold below):
  - `simulation.aqueduct_mem_estimate` exists only in the tracked `config.yml`, so a pull carries the rename.
  - `paths.aqueduct_root` is **dead**. It is a leftover from the external `aqueduct.exe` build, and the `{aqueduct_root}` placeholder is never substituted in any config value. Delete it rather than rename it: from `Snakefile:113-134` and `src/config_utils.py:173-208`. The line left in your gitignored `config_local.yml:19-23` is harmless; remove it by hand.
  - `config_hpc.yml` on Hydrax mentions "aqueduct" only in comments.
  - No fallback or deprecation warning is needed.
  - **Phase 1 gotcha:** `git rm --cached` a tracked file, commit it, then pull on another clone, and git **deletes** that clone's working copy. Before pulling the Phase 1 commit on Hydrax, back up `config_hpc.yml` (and any `*_materialized.yml` or logs you still need there), then restore them as untracked files. This applies only if `/u/schlumbe/gfm_code` is a git clone; the rerun scripts say "sync", so check whether it is rsync'd instead.
- **D. Snakemake code-change re-run after the rule rename.** Deferred until after the cleanup; see the "After the cleanup" list at the end of §8.
- **E. Local runs without the hop-0 fix.** ~~You ran the deltas locally on 2026-10-16 (`run_delta_floodmaps.py` → `snakemake` → rule `run_aqueduct` → `run_aqueduct.py`, which lacks 465dd93). So `P:\…\modelling\deltas_floodmaps\` and everything derived from it must be **re-run after Phase 5b**: the deltas flood-extent CSVs, `add_slr_year_column`/`add_protection_standard_column` outputs, and the deltas figures. See the "After the cleanup" list.~~ **Correction (Phase 5b, 2026-10-18):** 465dd93 only copies the boundaries `.gpkg` to node-local scratch before opening it (a GDAL/SQLite lock hang on Hydrax compute nodes). It reads the same file, so local and HPC results never differed. **The deltas do not need a re-run for this reason.**

---

## 8. Proposed cleanup plan (revised 2026-10-17)

Each phase is one commit or a small PR and can be verified independently. **Precondition**: commit your 3 modified analysis files (done: 4403f0c). **Branch**: `cleanup`.

**Dry-run check used by every phase.** Never build the full DAG locally: production splits it into parallel jobs on HPC, and `snakemake -n all` over the whole tile grid does not finish in reasonable time (killed after 6+ min on 2026-10-17). "`snakemake -n` equals the baseline" means both subset dry runs below give the same per-rule job counts as `.cleanup_baseline/snakemake_n_all_{thailand_bangkok,bangkok_chao_phraya}.txt`:
```bash
E=/c/Users/schlumbe/AppData/Local/miniforge3/envs/gfm_python_preprocessing
for c in thailand_bangkok bangkok_chao_phraya; do
  GFM_CONFIG_PATH="$PWD/snakemake_workflow/config/${c}_materialized.yml" PATH="$E:$E/Scripts:$E/Library/bin:$PATH" \
    $E/Scripts/snakemake.exe -n all --cores 1 | sed -n '/^Job stats:/,/^$/p'
done
```
Baseline (Phase 0): thailand_bangkok 453 jobs (run_aqueduct 90, extract_boundaries 90, merge_chunk 90, compute_flood_fraction_chunk 90, build_mosaic_vrt 45, plot_merged_results 45, prepare_exposure_grid_chunk 2, all 1); bangkok_chao_phraya 455 jobs (merge_chunk 180, compute_flood_fraction_chunk 180, build_mosaic_vrt 45, plot_merged_results 45, prepare_exposure_grid_chunk 4, all 1). After Phase 1 untracks the `*_materialized.yml`, they stay on disk (gitignored), so this check keeps working. **Since Phase 5b**, bangkok_chao_phraya's 405 locally produced `run_aqueduct` outputs dry-run as "Code has changed" (the script was rewritten; outputs unchanged, item D). From then on that config is checked with `--rerun-triggers mtime params input software-env`, which must report "Nothing to be done" (it does on the pre-5b code too).

**Phase 0: Tooling only.** ✅ *done, ce7540e (2026-10-17)*
- Add `pytest` and `ruff` to `environment.yml`. Add `environment_sfincs.yml` (hydromt-sfincs-dev).
- Add a minimal `pyproject.toml`:
  - `python_files = ["test_*.py", "validate_*.py"]`, `testpaths = ["tests"]`
  - markers `needs_pdrive`, `needs_sfincs`
  - ruff `F`, `E9`
- Add `CLAUDE.md` (appendix).
- *Verify*: `pytest tests -m "not needs_pdrive and not needs_sfincs"` passes. `ruff check .` is reported only, not fixed.

**Phase 1: Untrack generated artifacts.** ✅ *done 2026-10-17 (58 files untracked, `triplet_comparison_CLEAN.csv` deleted). Until Phase 6 adds `materialize_configs.py`, a fresh clone has no `*_materialized.yml`, so copy them from an existing checkout.*
- `git rm --cached` these: `run_timings.csv`, `sfincs.log`, `batch_logs/`, `preparation/logs/`, `adaptation_validation_output/`, `config_hpc.yml`, all `*_materialized.yml`. Extend `.gitignore`.
- `git rm` (also deletes the local file) `triplet_comparison_CLEAN.csv`.
- **Before pulling this commit on Hydrax**, back up the files there that are about to be untracked (`config_hpc.yml`, any `*_materialized.yml` still needed). A pull deletes files that another clone untracked with `git rm --cached`; restore them afterwards as untracked files (see Q-C).
- *Verify*: `git status` is clean; the untracked files still exist on disk; `snakemake -n all` equals the baseline.

**Phase 2: Delete unreachable code.** ✅ *done 2026-10-17; also removed the empty, unused `src/__init__.py` (STALE in §3). 3 unused imports in the moved figure scripts were fixed early so ruff passes on touched files.* *No `archive/` dir (user, 2026-10-17): every STALE/DUPLICATE item is deleted; git history keeps it.*
- Delete:
  - `julius_sfincs_code/`, `scratch/`
  - `config/calibration/sweep/`
  - the Q13 one-offs
  - `docs/_zoom_panel_a1.png`, `docs/ridge_gap_test_figure.png`
  - `tests/connectivity_tiling_prototype/`
  - `docs/sfincs_validation_pipeline_preparation.md`. Before deleting it, carry its only unique content (the `hydromt-sfincs-dev` env notes) into `environment_sfincs.yml` (done in Phase 0) and methods_04a.
- Move to `docs/figure_scripts/`: the 3 uncalled analysis figures plus `generate_eikonal_solver_examples.py`. Fix their `sys.path` depth.
- Update the reference in `sfincs_run.py`'s docstring (provenance line for the deleted julius code).
- *Verify*: `git grep` finds no reference to any deleted or moved path; `python docs/figure_scripts/<each>.py --help`; `pytest`; `snakemake -n all` equals the baseline.

**Phase 2b: Validation scope reduction** (§9; delete only, one commit). ✅ *done 2026-10-18. Verified: a local GBR+CAN `run_validation.py` on the pre-2b commit vs on the trimmed code gave identical metrics CSVs (GBR 9×35, CAN 3×35, summary 12×35), pixel-identical agreement rasters (8) and byte-identical PNGs (11). The multi-RP summary and England scripts were not re-run (user decision; they share the verified code path). `known_caveats` values reworded (user, 2026-10-17); untracked leftovers under tests/ deleted.*
- Delete the files: `preparation/convert_norway_stormflo.py`, `tests/norway_diagnostics/`.
- Delete the catalog entries: the 10 DROP entries in `data_catalog_validation.yml` (§9.2).
- Delete the code ranges in §9.1:
  - in `validate_country.py`: the partial and depth-band paths and the DNK raster branch
  - in `src/validation.py`: the partial-domain, depth-band and raster-fraction helpers
  - in `plot_agreement_map.py`: unit dots and `depth_agreement`
  - the partial call in `run_multi_rp_summary.py`
  - the DNK raster test in `tests/flood_extent_validation/test_synthetic_end_to_end.py`
- Delete the dead `validation:` config keys: `eval_domain`, `vector_simplify_tolerance_m`, `buffer_quad_segs`, `depth_band_open_ended_min_m`.
- Change the `BenchmarkSpec.coverage` default from `"partial"` to `"national"`, or make it raise when missing.
- Trim the dropped-country justification text in config, catalog and docstrings. Rewrite methods_04b for the 5 kept regions, adding England FZ2/FZ3 and SF Bay.
- **Keep** `tri_domain_mask`, which 4 analysis scripts use; Phase 7b renames it to `polygon_domain_mask`. Also keep `_chunks_overlapping_bbox` and the subregion machinery.
- *Verify*:
  - `git grep -niE '\b(FRA|ESP|FIN|NOR|DNK)\b|france|spain|finland|norway|denmark'` matches only provenance one-liners.
  - `pytest`.
  - `python validation/run_validation.py --countries GBR CAN` reproduces the current `P:\…\validation\{GBR,CAN}` metrics CSVs exactly. Snapshot them first.
  - `python validation/validate_england_fz2_band.py`, `_rescore_england_fz3.py`, `validate_sf_bay.py` and `run_multi_rp_summary.py --countries GBR` all run and give unchanged CSVs.

**Phase 3: Delete dead code inside live modules.** ✅ *done 2026-10-18; `ruff check .` is clean repo-wide. The 2 "test-only" rasters.py functions were kept (the audit finding was wrong, see Q16).*
- Remove the unused imports (and the other ruff F401/F541/F841 hits listed in `.cleanup_baseline/ruff_all.txt`: 39 at Phase 0), the 3 dead functions, the 2 unused `sfincs_run.py` functions. ~~and the 2 test-only `rasters.py` functions~~ (kept: they are production code, see Q16).
- Remove `plot_adaptation_pathways.py` and `plot_adaptation_bars.py`, plus the `run_analysis.py` step and config switch.
- *Verify*: `ruff check --select F401,F811,F841` on the touched files; `pytest`; `snakemake -n all`; `python analysis/run_analysis.py --help`.

**Phase 4: Documentation drift** (text only). ✅ *done 2026-10-18. Verified by an AST comparison: every changed .py differs from HEAD only in docstrings/comments, apart from one SKIP message string. `docs/python_vs_julia_qa.md` deleted (user, 2026-10-18), and the Python-vs-Julia evidence comments citing it removed from src/eikonal.py, src/flood_model.py, rules/simulation.smk and config.yml (design notes in those sentences kept).*
- README:
  - layout, including the eikonal-sensitivity and SFINCS-calibration entry points
  - remove the memory.md and martinique references
  - correct the claim that `analysis/` is outside the DAG
  - list the standalone validation scripts
- Fix the stale usage paths in docstrings (§4.4), the `config_utils.py:141` docstring, `.gitignore:5`/`Snakefile:117` and the `environment.yml` references.
- Docs and docstrings: call the env `gfm`, noting "(`gfm_python_preprocessing` on the Windows machine)". Add dedup and the rerun scripts to methods_04a.
- *Verify*: a script that checks every repo path mentioned in `*.md`, `*.py` docstrings and `*.sh` comments exists; `snakemake -n all` is unchanged.

**Phase 4b: NumPy-style docstrings** (own phase, user 2026-10-17; text only). ✅ *done 2026-10-18: 71 CORE modules, 250 public functions/classes, all NumPy-style. Docstring-only verified by AST comparison (docstrings stripped) against HEAD.*
- Convert the public functions/classes of every CORE module (§3) to short NumPy-style docstrings (Parameters / Returns / Raises where useful) that say what and why, not a restatement of the code. Keep the existing explanatory prose where it carries the *why*.
- One subagent per module, one commit per directory (`src/`, `snakemake_workflow/scripts/`, `sfincs_tiles/`, `preparation/`, `analysis/`, `validation/`).
- Other phases leave docstring style alone.
- *Verify*: `ruff check` on the touched files; `pytest`; `git diff --stat` shows only docstring and comment lines (check with `git diff -U0 | grep '^[+-]' | grep -v '^[+-]\s*\(#\|"""\|$\)'`, reviewed by hand).

**Phase 5: Consolidate duplicated logic** (one PR each). ✅ *done 2026-10-18 (5a-5e).*
- **5a.** ✅ *done 2026-10-18: `src/tiles.build_chunk_grid`; parity vs all 3 old copies on production (949 chunks) + 2 subset grids: IDs, bounds, geometry, CRS identical; new tests/chunk_grid_validation.* One `_build_chunk_grid` in `src/`, used by the Snakefile, `generate_hpc_postprocess_job` and `check_postprocess_progress`. *Verify*: a unit test asserting identical chunk IDs; `snakemake -n postprocess` equals the baseline.
- **5b.** ✅ *done 2026-10-18: `aqueduct_runner.run_tile_scenario` (+ `_stage_file_locally`); both entry points are thin wrappers keeping their own timing/progress behaviour. Parity: synthetic 5-tile harness (all branches) old vs new, local + CLI: all rasters, skip/OOM markers and timing records identical; real tiles (Thailand 31/499 x RP100_SLR_0 + RP1000_SLR_1000, inputs from P:, outputs local) old vs new CLI: all 4 rasters bit-identical. Snakemake now flags locally produced simulation outputs as "code changed" (`run_aqueduct.py` was rewritten); outputs are unchanged, see D.* Move the shared tile solve into `src/aqueduct_runner.py` (renamed in 7a), **including the hop-0 boundary-read change for the local path** (Q4). *Verify*: parity test on a synthetic tile (both entry points give bit-identical output); real-tile spot-check of 2–3 tiles against existing HPC outputs on P:\.
- **5c.** ✅ *done 2026-10-18: in `src/hpc_common.py` (not scripts/: every importer already has src/ on its path, incl. the Snakemake `script:` generate_aqueduct_jobs.py). 4 helpers x 12 copies -> 1. Verified: each generator's AST equals HEAD minus the removed defs plus one import (so every call site and generated sbatch line is unchanged); helper outputs identical to HEAD copies; all 6 generators import. `_stage_file_locally` (1 copy left in generate_hpc_postprocess_job) not merged: it stages on the login node, aqueduct_runner's on compute nodes; merging would make the generator import the solver stack.* HPC sbatch helpers go into `snakemake_workflow/scripts/hpc_common.py`. *Verify*: first (before any 5c edit) run each `generate_hpc_*` with a temp copy of `config_hpc.yml` whose job/output dirs point into `.cleanup_baseline/sbatch_before/` (never the real Hydrax/P: job dirs; deferred from Phase 0 for that reason); after the change, regenerate into `sbatch_after/` and diff: byte-identical.
- **5d.** ✅ *done 2026-10-18: `analysis/_common.py` holds `_read_band`, `_round_val`, `_growth_matrix_path`, `_growth_path`, `_eai_from_interp` (11 copies in 6 scripts -> 1; `_growth_matrix_path` in plot_tipping_points differed only in a type hint). Verified: each file's AST equals HEAD minus the removed defs plus the import lines; all affected scripts (incl. the 4 HPC exposure scripts importing compute_exposure_analysis) pass `--help`. Not merged (behaviour differs): `_first_crossing`, `_people`, flood_totals' `_read_band`/`_read_band_and_transform`, and the 3 `_run` variants - see TODO.md. No run_analysis.py rerun: its outputs live on P:\ and the moved code is AST-identical.* Analysis helpers go into `analysis/_common.py`, and the triplicated `_run` goes into `src/`. *Verify*: rerun `run_analysis.py` on one country subset and diff the CSVs.
- **5e.** ✅ *done 2026-10-18, differently than planned: sfincs_tiles cannot import src/ in hydromt-sfincs-dev (`src/config_utils` needs hydromt 0.9's `hydromt.log.setuplog`; confirmed ImportError), so the copies stay. New tests/metric_parity_validation/validate_sfincs_src_parity.py pins them to the src originals (confusion counts, HR/HT-FAR-CSI-bias incl. NaN cases, waterdepth decoding + constants, retry semantics); each copy got a 'twin of src/...' comment. No calibration code changed, so no calibration_result.json recompute needed.* ~~`sfincs_tiles` imports `confusion_counts`/`metrics_from_counts`/`retry_transient_io` from `src/`, which is pure numpy and works in both environments. Keep an `HT` alias column so existing calibration CSVs stay readable. *Verify*: parity pytest on synthetic arrays; recompute one tile's `calibration_result.json` and diff it, expecting identical HR/FAR/CSI/bias.~~

**Phase 6: Hard-coded paths into config.** ✅ *done 2026-10-18. New key `paths.deltas_shapefile`; all other paths derive from existing keys (validation.benchmark_root/output_dir, postprocessing.merged_outputs, tile_grid.path, ...). 11 replaced constants verified to resolve to the old values on this machine. Hydrax: `sfincs_tiles/hydrax_env.sh` (sourced by shell, parsed by `gfm_config.hydrax_paths()`), `rerun_calibration.sh --part` (old names = wrappers; both branches exercised against a fake tree). Launchers use `GFM_ENV` (default = old env path) + `%~dp0`; .ps1 deleted. `materialize_configs.py` replaces build_delta_run_configs.py, but existing materialized files are older snapshots and were NOT regenerated (TODO.md). config.yml defaults kept (TODO.md).*
- Move to config: `DELTAS_PATH`, the validation reporting inputs, and the Hydrax paths (one sourced `sfincs_tiles/hydrax_env.sh`, plus the merged `rerun_calibration.sh` per B).
- Launchers: use `%~dp0` and an env var for the interpreter, targeting `gfm`. Delete `run_thailand_bangkok.ps1` (duplicate of the .bat).
- Add `config_local.yml.example`, and `materialize_configs.py` (generic, replacing `build_delta_run_configs.py`).
- *Verify*: `git grep -nE '[A-Z]:[\\/](Users|GFM)|/p/1121|/u/schlumbe'` returns only `*.example` and docs; `snakemake -n all` equals the baseline; `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal` on Hydrax (you run this).

**Phase 7: Renames and structural tidy.**
- **7a. aqueduct → eikonal.** ✅ *done 2026-10-18. Files: eikonal_runner.py, run_eikonal(_cli).py, generate_eikonal_jobs.py, run_eikonal_dag.py, generate_hpc_exposure_jobs.py, tests/eikonal_runner_validation/. Rules run_eikonal / generate_eikonal_jobs; `run_eikonal_python`, `estimate_eikonal_mem_mb`, `EIKONAL_NODATA`; config `eikonal_mem_estimate` (old key still read as fallback); `{aqueduct_root}` removed. Provenance mentions (external Aqueduct model, WRI Aqueduct data) kept. Verified: AST of every .py equals HEAD with the rename map applied (docstrings aside), except 2 message strings and the aqueduct_root removal; ruff, pytest (66), --help, dry run (run_aqueduct 90 -> run_eikonal 90). generate_hpc_*_job(s) singular/plural left as is (only generate_exposure_jobs renamed). Hydrax follow-ups in TODO.md.* Covers about 260 mentions in 50 files, excluding logs and provenance mentions.
  - Modules: `aqueduct_runner.py` → `eikonal_runner.py`, `run_aqueduct{,_cli}.py` → `run_eikonal{,_cli}.py`, `generate_aqueduct_jobs.py` → `generate_eikonal_jobs.py`.
  - Rules: `run_aqueduct` → `run_eikonal`, `generate_aqueduct_jobs` → `generate_eikonal_jobs`.
  - Config keys, with fallback (C).
  - `run_pipeline.py` → `run_eikonal_dag.py` (A).
  - Unify `generate_hpc_*_job(s)` naming; `generate_exposure_jobs` → `generate_hpc_exposure_jobs`.
  - *Verify*: `git grep -i aqueduct` shows only provenance and the deprecation fallback; `pytest`; `snakemake -n all` job counts equal the baseline (then D: `--list-code-changes`, and `--touch` if needed); the generated sbatch files differ from the baseline only in the renamed script names.
- **7b. Other renames.**
  - `src/flood_extent.py` folds into `flood_model.py`.
  - `src/validation.tri_domain_mask` → `polygon_domain_mask`, probably moved to `rasters.py`. It is a generic polygon-to-raster mask, used by the deltas and Bangkok analysis, not by validation.
  - `_rescore_england_fz3.py` → `rescore_england_fz3.py`.
  - `calibration_studies/select_calibration_tiles.py` → `select_study_candidate_tiles.py`; `calibration_studies/test_*_calibration.py` → `run_*_study.py` (update the references in `generate_calibration_batch_jobs.py` and methods_03).
  - Move `select_country_tiles.py` to `preparation/`.
  - Rename the `exposure`/`exposure_analysis` and `plotting`/`visualization` modules to clarify the split.
  - *Verify*: `pytest`; `python <script> --help` for every renamed CLI; `snakemake -n all`.
- **7c. Splits.**
  - `build_station_boundary_lines.py` into lines, conflict and plot modules.
  - `validate_country.py`: after Phase 2b it is ~890 lines with one scoring path (national coverage), so splitting it is optional; at most, move the chunk/mosaic helpers (`_chunks_overlapping_bbox` etc.) into a `validation/_chunks.py` that the England scripts import.
  - `run_analysis.py` into a STEPS table.
  - *Verify*: `pytest`; one tile through `run_final_tile.sh` on Hydrax; `run_validation.py --countries GBR CAN` and a diff of the metrics CSVs.


**After the cleanup (deferred, long-running; you run these):**
1. **Snakemake code-change check (D).** Run `snakemake -n all --list-code-changes`. If the rule/script rename makes simulation outputs look stale, run `snakemake --touch` once rather than re-simulating. This writes Snakemake metadata next to the P:\ outputs, so approve it explicitly.
2. ~~**Re-run the deltas (E).**~~ Not needed: see the correction under E (465dd93 never changed results).
3. **FRA RP250 validation:** closed. FRA is dropped and its outputs are deleted. `validation/multi_rp_summary.csv` was checked 2026-10-17 and has GBR rows only (also no CAN rows; see §9.4 Q3).

---

## 9. Validation scope reduction (added 2026-10-17)

**Decision**: national-map validation now targets only **Scotland, Wales, England (GBR), New Brunswick (CAN) and California / SF Bay (USA)**, plus their subregions. **Dropped**: France, Spain, Finland, Norway, Japan, Germany, Australia, Denmark. The output dirs for the dropped countries have already been deleted on P:\; only `validation/{CAN,GBR}` remain there. The benchmark *input* data under `P:\…\inputs\validation\{ESP,FRA,NOR,DNK,FIN}` is untouched; whether to delete it is your call, outside the repo.

**What this frees up.** Every kept catalog benchmark is `coverage: national`, `variable: extent` and vector. Spain and France were the only users of the partial-coverage path; France was the only user of the depth-band path; Denmark was the only raster benchmark. All three code paths can go. SF Bay runs through the standalone `validate_sf_bay.py`, which has no catalog entry and imports only `config_utils`. Japan, Germany and Australia have no code or data, only a "pending" comment at `data_catalog_validation.yml:14`. The JPN entries at `config.yml:637,662` are *analysis* country lists, so keep them.

### 9.1 Code and config to remove or edit

| path[:lines] | what | country | action | evidence |
|---|---|---|---|---|
| validation/validate_country.py:412-758 | `validate_country()`, the partial-coverage path (segments/perimeter, TRI, connectivity, units CSV) | ESP/FRA | DELETE range | skips every non-`partial` entry (:452); no kept entry is partial |
| validate_country.py:759-944 | `validate_country_depth_bands()` | FRA | DELETE range | only the FRA `variable: depth` entries |
| validate_country.py:1046-1050, 1113-1121 | RasterDataset branch | DNK | DELETE | the only raster benchmark |
| validate_country.py:1360-1363, 1397-1418 | `main()` calls to the partial/depth paths, depth_metrics write | ESP/FRA | EDIT | |
| validate_country.py:1-71, 116, 332, 953-988, 1133, 1254, 1277 | ESP/FRA/NOR examples, `--country ESP` | — | EDIT text | |
| validate_country.py:124-411, 945-1328 | chunk/mosaic helpers, `_find_benchmark_keys`, `_blend_regions`, national path, `_write_agreement_raster` | shared | **KEEP** | England scripts import `_chunks_overlapping_bbox`; plot_agreement_map, run_multi_rp_summary |
| validation/run_multi_rp_summary.py:57-61, 110 | imports/calls `validate_country` (partial) | ESP/FRA | EDIT, in the same commit | otherwise it breaks |
| run_multi_rp_summary.py:2, 16, 25, 31-34, 43 | Norway text, `--countries NOR` | NOR | EDIT text | |
| validation/plot_agreement_map.py:486-526, 684-692 | `plot_unit_csi_dots` (units CSVs exist only for partial) | ESP/FRA | DELETE | |
| plot_agreement_map.py:48-55, 531, 652, 700-702 | `--metric depth_agreement` | FRA | DELETE | |
| plot_agreement_map.py:5-6, 20, 419, 435, 659-665 | text, DNK raster skip | — | EDIT text | |
| validation/run_validation.py:12 | `--countries ESP FRA` usage | — | EDIT text | discovery is catalog-driven |
| src/validation.py:323-396, 995-1037 | `rasterize_depth_bands`, `depth_band_counts`/`_metrics` | FRA | DELETE | only the depth path |
| src/validation.py:199-215, 402-501, 530-607 | `region_for_point`, `build_evaluation_clusters`, `units_with_extent_coverage`, `connectivity_domain_mask`, `connectivity_components` | ESP/FRA | DELETE | only the partial path |
| src/validation.py:668-736 | `read_benchmark_raster_fraction` | DNK | DELETE | plus its test (next row) |
| tests/flood_extent_validation/test_synthetic_end_to_end.py:207-272 | DNK raster test | DNK | DELETE range | rest of the file and test_metrics.py: keep |
| src/validation.py:187-198 + `exclude_bbox` field | `_exclude_bbox_mask` | none | DELETE | no entry uses it |
| src/validation.py `BenchmarkSpec` | `ht_min_col`/`ht_max_col`, `study_area`, `wet_values`/`depth_threshold_m`; `coverage` default `"partial"` | FRA/ESP/DNK | EDIT; default → `"national"` or raise | otherwise an entry without `coverage` silently yields no rows |
| src/validation.py:502-529 | `tri_domain_mask` | (was FRA) | **KEEP** (rename in 7b) | 4 analysis scripts (deltas, Bangkok, protection) |
| src/validation.py docstrings :76, 115, 635, 678, 763, 847 | soft-scoring justification from DNK/FIN/NOR/FRA | — | EDIT text, keep functions | |
| preparation/convert_norway_stormflo.py | Norway ingest (414 lines) | NOR | DELETE file | nothing imports it |
| tests/norway_diagnostics/ | Norway investigation | NOR | DELETE dir | pointers only: config.yml:789, data_catalog_gfm.yml:233, README:99 |
| snakemake_workflow/config/config.yml:763-786, 822-831 | `eval_domain`, `vector_simplify_tolerance_m`, `buffer_quad_segs`, `depth_band_open_ended_min_m`, an orphan Copernicus comment | ESP/FRA | DELETE | read only by the deleted paths |
| config.yml:747-749, 754-762, 788-797, 805-811 | NOR/ESP/DNK/FIN/FRA justification text | — | EDIT text | keep the `benchmark_wet_fraction` key (national path) |
| data_catalog_gfm.yml:233 | norway_diagnostics pointer | NOR | EDIT | |
| snakemake_workflow/scripts/calibration/select_country_tiles.py:1-18 | ESP/FRA/NOR history | — | EDIT docstring | GROUPS is already GBR/CAN only |
| gbr_wales_scotland_friction9.yml:15, rules/preprocessing.smk:10, generate_hpc_preprocess_job.py:263, run_aqueduct_cli.py:108, fetch_england_flood_zones.py:19, analysis/compute_flood_totals.py:67 | historical "ESP/FRA/NOR" or Norway/DNK comments | — | EDIT to one-line provenance | |
| docs/methods_04b_MapsValidation.md:25, 31-41, 44, 50-55, 79, 92, 140-147, 155-156 | partial/depth methods, dropped status rows, Norway caveats | all | REWRITE for the kept regions | the doc also lacks England FZ2/FZ3 and SF Bay |
| README.md:32, 74, 98-99 | `(ESP/FRA/NOR)`, `--countries ESP FRA NOR`, martinique/norway pointers | — | EDIT | |
| validation/build_overview_table.py, compare_against_published.py, the 4 tracked validation CSVs | — | — | KEEP | already GBR/CAN/USA only |

### 9.2 `data_catalog_validation.yml` entries

| entry | country | lines | action |
|---|---|---|---|
| file header | — | 1-14 | EDIT (:9-14 Australia "pending" line) |
| spain_arpsi_marina_segments | ESP | 16-36 | DROP |
| spain_zi_marina_q100 | ESP | 38-83 | DROP |
| france_tri_perimeters | FRA | 85-107 | DROP |
| france_inondable_02moy | FRA | 109-154 | DROP |
| france_inondable_reunion_03mcc | FRA | 156-190 | DROP |
| france_inondable_ht_02moy (depth) | FRA | 192-243 | DROP |
| france_inondable_ht_reunion_03mcc (depth) | FRA | 245-278 | DROP |
| norway_stormflo_200ar | NOR | 280-369 | DROP |
| denmark_stormsurge_national (raster) | DNK | 371-453 | DROP |
| wales_nrw_floodzone_seas | GBR | 455-545 | KEEP (comments cite NOR/FIN/DNK: edit :464, 513) |
| finland_stormsurge_national | FIN | 547-635 | DROP |
| scotland_sepa_coastal_m | GBR | 637-715 | KEEP (edit :710) |
| england_ea_floodzone3_sea | GBR | 717-778 | KEEP |
| england_ea_floodzone2_sea | GBR | 780-816 | KEEP |
| new_brunswick_present_day_1in100yr | CAN | 818-935 | KEEP (edit :835) |

That is about 590 of 935 lines removed.

### 9.3 Minimal set that stays

- **validation/**: `run_validation.py`, a trimmed `validate_country.py` (national path plus chunk helpers), `plot_agreement_map.py` (agreement metric only), `run_multi_rp_summary.py`, `validate_england_fz2_band.py`, `_rescore_england_fz3.py`, `fetch_england_flood_zones.py`, `validate_sf_bay.py`, `build_sf_bay_benchmark_mosaic.py`, `build_overview_table.py`, `compare_against_published.py`.
- **src/validation.py**: `BenchmarkSpec`, `fix_catalog_meta_encoding`, `load_benchmark_spec`, `primary_region_name`, `load_benchmark_full`, `filter_benchmark` (Wales `attribute_filter`), `benchmark_fraction_from_vector`, `wet_mask_from_fraction`, `model_domain_mask`, `permanent_water_mask`, `read_permanent_water_mask`, `load_iso_lookup`, `read_country_mask`, `confusion_counts`, `confusion_counts_soft`, `confusion_counts_tolerant`, `_safe_div`, `metrics_from_counts`, plus `tri_domain_mask` (for analysis).
- **Subregions**: keep `regions:`, `_blend_regions`, `primary_region_name` and `plot_subregions_grid`. GBR and CAN use them.

### 9.4 Questions: answered

1. **Keep the raster-benchmark path?** Deleted in Phase 2b. A non-GeoDataFrame entry now raises `ValueError`.
2. **Bring SF Bay into the catalog / `run_validation.py`?** Not in this cleanup. It stays standalone, its scripts are untouched by 2b, and it is documented in methods_04b. Its hard-coded single-chunk input is logged in TODO.md, and its paths move to config in Phase 6.
3. **No CAN rows in `multi_rp_summary.csv`?** By design. `new_brunswick_present_day_1in100yr` has no `comparison_return_periods`, and the multi-RP summary only scores benchmarks that do.

---

## Appendix: Draft CLAUDE.md (for approval; not yet written to disk)

```markdown
# CLAUDE.md

## Project
GFM (Global Flood Model): tile-based global coastal flood hazard pipeline extending
Deltares' Aqueduct Coastal Flooding method. Per DeltaDTM tile it prepares DEM/mask/
friction/water-level boundaries, runs a Python Fast-Sweeping eikonal flood solver per
(tile, return period, SLR scenario), merges to spatial chunks, then runs exposure/
adaptation analysis and benchmark validation. A separate SFINCS sub-pipeline
(`sfincs_tiles/`) calibrates/validates the eikonal solver against SFINCS on Hydrax.

## Layout & entry points
- `preparation/run_preparation.py`: run first; builds tile grid + boundary inputs.
- `Snakefile`: targets `preprocess`, `simulate`, `postprocess`, `all`.
  Rules in `snakemake_workflow/rules/`, rule scripts in `snakemake_workflow/scripts/`.
- HPC/SLURM: `snakemake_workflow/scripts/generate_hpc_{preprocess,postprocess}_job.py`,
  `generate_hpc_simulation_jobs.py` (see `snakemake_workflow/hpc.md`).
- `analysis/run_analysis.py`: exposure/adaptation figures (HPC exposure jobs also
  import `analysis/compute_exposure_analysis.py`).
- `validation/run_validation.py`: eikonal vs national hazard maps (soft-scored HR/FAR/CSI).
  Scope is ONLY Scotland, Wales, England, New Brunswick and California/SF Bay (+ subregions);
  SF Bay and England FZ2/FZ3 run via standalone scripts in `validation/`. Do not add
  other countries back without asking.
- `sfincs_tiles/generate_calibration_jobs.py stage1|stage2`, `rerun_*_calibration.sh`:
  eikonal vs SFINCS friction calibration, Hydrax only (`run_final_tile.sh` is the
  per-tile SLURM worker).
- `calibration_studies/generate_calibration_batch_jobs.py sweep_budget|obstacle_coupling`:
  eikonal solver sensitivity (inner sweep rounds / outer iterations), docs/methods_03.
- `src/`: shared library, flat layout; scripts add it via `sys.path.insert`. Follow the
  sibling file's pattern; do not introduce packaging/relative imports.

## Config
- `snakemake_workflow/config/config.yml` is the single source of truth for every pipeline.
- Machine-specific paths go in `config_local.yml` (gitignored, auto-loaded). Never put
  per-machine values in `config.yml`.
- HPC: `config_hpc.yml` (gitignored; copy from `config_hpc.yml.example`), passed via
  `GFM_CONFIG_PATH`.
- `*_materialized.yml` are generated; never hand-edit them.

## Environments
- `gfm` (environment.yml): main pipeline, analysis, validation, and all plotting
  (including sfincs_tiles plot scripts). Snakemake pinned to 7.32.4.
- `hydromt-sfincs-dev` (environment_sfincs.yml): sfincs_tiles model build/run only.
  Its matplotlib is broken; never plot in it.
- On the Windows machine this env is named `gfm_python_preprocessing` (outdated name, same env):
  `C:/Users/schlumbe/AppData/Local/miniforge3/envs/gfm_python_preprocessing/python.exe`.
  The PATH `python` there is the Windows Store stub; always use the full interpreter path.

## Commands
- Dry run: `snakemake -n all`  (always run after touching Snakefile/rules/scripts)
- Local run: `python run_pipeline.py`  or  `snakemake all --cores 4 --resources mem_mb=8000`
- Tests: `pytest tests -m "not needs_pdrive and not needs_sfincs"`
- Lint: `ruff check .`
- Progress monitors (`check_*_progress.py`, `sfincs_tiles/check_calibration_progress.py`):
  the user runs these themselves, usually on Hydrax. Give the command, don't run it.

## Data
Inputs/outputs live on the P:\ share (`/p/11212688-004-global-floodmaps/` on Hydrax).
Never write to, move, or delete anything on P:\ or the model output directories.

## Do not touch without asking
- Output path patterns and wildcard constraints in `snakemake_workflow/rules/*.smk`:
  changing them orphans existing outputs on P:\.
- `snakemake_workflow/config/config.yml` defaults, `data_catalog_*.yml`.
- Pinned versions in `environment.yml`.
- `sfincs_tiles/` while a Hydrax batch is running (ask whether one is).

## Conventions
- Keep `run_aqueduct.py` (local) and `run_aqueduct_cli.py` (HPC) behaviourally identical
  until they are unified.
- Rich module docstrings explaining *why*; match the surrounding comment density.
- One-off diagnostics are deleted once their question is answered, not left in the tree.
```
