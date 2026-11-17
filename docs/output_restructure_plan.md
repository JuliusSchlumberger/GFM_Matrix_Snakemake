# Output folder restructure plan

Plan only. Nothing under `<data_root>` or in the code has been moved, renamed or edited.
`R` = `<data_root>` (`paths.root`): `P:/11212688-004-global-floodmaps/modelling` locally,
`/p/11212688-004-global-floodmaps/modelling` on the HPC (same storage). `PI` =
`R/processed_inputs`, `MO` = `simulation.model_outputs`, `PPI` =
`simulation.preprocessing_inputs_dir`, `MR` = `postprocessing.merged_outputs`, `JD` =
`hpc.jobs_dir`, `TG` = `tile_grid.path`.

---

## 1. Inventory

### 1.1 How paths are defined

- **Templates in config.** All path keys are templates in `snakemake_workflow/config/config.yml`.
  They are expanded by plain `str.replace` in two passes:
  1. `{root}` and `{code_root}`
  2. `{processed_inputs_dir}` and `{figures_root}`

  The expansion code sits in two places: `Snakefile:54-75` and `src/config_utils.py:105,143-150`
  (`load_config`, plus `load_hpc_config` at :168).
- **Overlays.** The 11 scenario overlays override whole keys.
  `snakemake_workflow/scripts/materialize_configs.py` writes `*_materialized.yml` via
  `config_utils.materialize_config` (:201).
- **Variants.** `src/flood_variants.py:68-90` appends `__<variant>` to each output key separately.
- **Literals in rules.** The `.smk` rules hard-code everything below each root: `{tile_id}/inputs/...`,
  `{tile_id}/results/...`, `chunks/`, `chunks/flood_fraction/` and `plots/`. No rule has a
  `log:` or `benchmark:` entry, and none uses `temp()`, `directory()` or `protected()`.
- **SFINCS and calibration studies bypass config.** `sfincs_tiles/` builds paths as
  `R/<--base-dir-name>` (24 scripts default to `calibration_roughness_SFINCS`).
  `calibration_studies/` builds `R/<--study-dir-name>` (default `calibration_500_tiles`).
- **HPC data root.** On the HPC, the shell drivers take the data root from `GFM_DATA_ROOT` in
  `sfincs_tiles/site_env.sh`.

### 1.2 Directories on disk vs code

Status values:
- **ok**: code and disk agree.
- **mismatch**: code points elsewhere than the data on disk.
- **orphan**: no code writes or reads it.
- **dup**: overlaps another directory.

| on disk (`R/…`) | written by | read by | config key / literal | status |
|---|---|---|---|---|
| `model_outputs/<tid>/inputs/` (~4504 tiles × 50 files) | preprocessing rules (`preprocessing.smk:46-189`), HPC preprocess jobs | `run_eikonal`, `eikonal_runner.py:76-93,374`, `check_preprocess_progress.py:72`, **SFINCS hard-coded** (`run_final_tile.sh:27`, `generate_calibration_jobs.py:112,114`, `select_calibration_tiles.py:152`, `build_station_boundary_lines.py:355`, `preflight_sfincs_mask.py:65`), `calibration_studies/select_study_candidate_tiles.py:33,85` | `PPI` = `{root}/model_outputs` (`config.yml:196`) | ok; shared by v2, the SFINCS comparison and parameter adjustment |
| `model_outputs/<tid>/results/` (45 rasters per tile) | `run_eikonal` (`simulation.smk:47-51`), `run_eikonal_cli.py` | `merge_chunk`, `validation/_chunks.py:72`, `validate_country.py:231`, `merge_england_local_tiles.py` | `MO` = `{root}/model_outputs_v2` (`config.yml:195`) | **mismatch**: v1 results are under `model_outputs`, config targets the not-yet-run `model_outputs_v2` |
| `model_outputs/{run_timings (202 680), skipped_tiles (34 734)}` | `eikonal_runner.py:339-340,480`, `run_eikonal.py:46,56`, `run_eikonal_cli.py:73` | `summarize_run_timings.py`, `collect_hpc_failures.py:92`, `plot_domain_maps.py:132`, `tile_output_complete` (:439-450) | literal under `MO` | ok (v1) |
| `model_outputs/hpc_jobs/` (1040 files, `logs/` 1711, `exposure/`) | `generate_hpc_*`, `hpc_common.py:74-75` | `collect_hpc_failures.py`, `check_postprocess_progress.py:48`, resume logic | `JD` = `{root}/model_outputs_v2/hpc_jobs` (`config.yml:300`) | **mismatch** (v1 under `model_outputs`); also holds `resolved_config_2026-10-15.yml` |
| `merged_results/{chunks (87 309, flat), exposure/{*.csv, flood_totals/}}` | `postprocessing.smk:9-121`, `generate_hpc_exposure_jobs.py:95`, `compute_flood_totals.py:96` | `run_analysis.py:84-87`, validation scripts, `compute_exposure_analysis.py:501` | `MR` = `{root}/merged_results_v2` (`config.yml:340`); `validation.flood_totals_dir` = literal `{root}/merged_results_v2/exposure/flood_totals` (:584) | **mismatch**; `flood_totals_dir` does not derive from `MR`, and no overlay overrides it |
| `processed_inputs/*.nc`, `*.csv` | `prepare_boundary_conditions.py:375-431`, `extract_*` | `boundaries.py:157-211` | `PI` (`config.yml:11`) | ok. SLR ssp126 files are from 2026-10-05 and ssp245 from 2026-11-07; the ssp126 ones may be stale |
| `processed_inputs/WL_scenarios/`, `WL_scenarios_cache/` | `prepare_boundary_conditions`, `cache_waterlevel_stations` (`preprocessing.smk:143-172`) | `extract_boundaries` | `waterlevel_nc_dir` (:97), `stations_cache_dir` (default literal at `preprocessing.smk:144`) | ok |
| `processed_inputs/storm_tide_product/{…, cache/, gauges/}` | `prepare_storm_tide_product.py:56-84` | `run_eikonal` (hydrographs) | `storm_tide_product_dir` (:112), `cache_dir` (:180) | ok |
| `processed_inputs/mask/domain_tiles_global.gpkg`, `00_…04_*.gpkg`, checkpoint, `domains_final*.png` | `build_tile_manifest.py:123-232` (intermediates go to `TG.parent`), `plot_domain_maps.py:131` | all production rules, `tile_split.py:137` (**rewrites TG in place**) | `TG` (:84); hard-coded copies in `generate_calibration_batch_jobs.py:159` and `select_study_candidate_tiles.py:84` | ok |
| `processed_inputs/mask/domain_tiles_global_pre_connectivity_migration.gpkg` | none | none | none | **orphan** (old tiling) |
| `processed_inputs/mask/{bangkok_chao_phraya,thailand_bangkok,deltas_global,gbr_wales_scotland_friction9}_tiles.gpkg`, `calibration/` (5 files) | `build_tile_subset_by_id.py`, `build_delta_tile_subset.py:75`, `select_country_tiles.py:120,166` | overlays' `TG` | overlays | `thailand_bangkok_tiles.gpkg` is **dup?** of the Bangkok grid; `gbr_wales_scotland_friction9` is **orphan?** (no overlay uses it) |
| `processed_inputs/mask/validation/` (12 files: `*_fine_tiles*.gpkg`, `*_dem_source_manifest*.csv`) | `build_{england,wales,sf_bay}_fine_tiles.py` | overlays' `TG`, `simulation.dem_source_manifest` (`config_utils.py:297`) | overlays | ok |
| `processed_inputs/mask/tile_generation_debug/` | none | none | none | **empty** |
| `calibration_500_tiles/{grid.gpkg, candidate_tiles.txt, resolved_config.yml, hpc_jobs/logs (empty), model_outputs/(574, own hpc_jobs, run_timings, skipped_tiles)}` | `generate_calibration_batch_jobs.py:145,175-209`, Snakemake with the C500 overlay | `calibration_studies/*` | `--study-dir-name` default (:104); overlay `calibration_500_tiles.yml:8,11,39` | ok. The C500 overlay does **not** override `MR`, `visualization.output_dir` or `validation.output_dir`, so its postprocess would write into production |
| `calibration_roughness_SFINCS/` (≈1000 `<tid>/{inputs,sfincs_model,outputs}`, `boundary_lines/`, `candidates/`, `hpc_jobs_stage{1,2}/`, CSVs) | `sfincs_tiles/*` (see 1.4) | `sfincs_tiles/*`, tests (3 files), `boundary_test/extract_gtsm_series.py:38` | `--base-dir-name` literal (24 scripts), `run_final_tile.sh:13`, `rerun_calibration.sh:34` | ok. `metrics_overview_table_fsf12.csv` is 2 bytes |
| `calibration_roughness_SFINCS_dedup/` (`<tid>/outputs/calibration_result.json` + CSVs) | `compute_calibration_result_dedup.py:78-98` | `run_calibration_models.py`, `exposure_depth.py:485`, `rerun_calibration.sh:51` | `<base>_dedup` literal | ok (meant to sit next to its base) |
| `sfincs_calibration_bline/` (579 tiles, sampled `sfincs_model/` empty) | none | none | none | **orphan** |
| `bangkok_chao_phraya/{model_outputs, merged_results, figures, dispatch, domain.gpkg, resolved_config.yml}` | Snakemake with the BKK overlay, `run_bangkok_case_study.bat` | `analysis/*bangkok*` | overlay; **`.bat:30,37-45,85,90` hard-codes `STUDY_ROOT`** | ok |
| `deltas_floodmaps/{ssp126,ssp245}/{model_outputs, merged_results, WL_scenarios, delta_merge_scratch}` | Snakemake with the deltas overlays, `run_delta_floodmaps.py` | `compute_delta_flood_extent.py:90-202` | overlays | ok |
| `deltas_floodmaps/{delta_flood_extent.csv, figures/}` | analysis scripts hard-coding `root/"deltas_floodmaps"`: `compute_delta_flood_extent.py:194`, `add_slr_year_column.py:59,66`, `plot_delta_affected_fraction_timeseries.py:103-113`, `plot_delta_flood_map.py:87-94` | same | **literal** | ok |
| `deltas_floodmaps/previous_without_protection/` | none | none | none | **dup/orphan** (2026-10-15 copy) |
| `england_local_input/{model_outputs(16), merged_results/{chunks,local_native}, processed_inputs/WL_scenarios_cache, validation/GBR, figures, agreement_*.png}` | ELI overlay, `merge_england_local_tiles.py:30`, `plot_local_vs_global_agreement.py:351` (writes to `validation.output_dir.parent`) | validation scripts | overlay | **mismatch**: the overlay now points to `england_local_input/all_england/…`. The data on disk is the earlier domain (`england_fine_tiles.gpkg`, per the stale materialized config) |
| `england_global_input/{model_outputs(15), merged_results/chunks, validation/GBR}` | EGI overlay | validation | overlay | **mismatch** (same `all_england` shift) |
| `england_hop_test/{model_outputs(53), processed_inputs/WL_scenarios_cache}` | EHT overlay | – | overlay | ok (test) |
| `wales_{local,global}_input/…`, `scotland_local_input/processed_inputs/WL_scenarios_cache`, `sf_bay_local_input/{model_outputs(56), merged_results/local_native, validation/USA_*}` | overlays, `validate_sf_bay.py` | validation | overlays | ok. Scotland has a station cache only, no run yet. `sf_bay_local_input/validation/` holds 4 variant folders (`_global_baseline`, `_masked_fix`, `_usgs10m`, `_sample`) |
| `sf_bay_global_input/model_outputs` (4 tiles; tile 1 has inputs only) | aborted run (TODO.md:1437-1444) | none | no overlay | **orphan** |
| `fundy_extreme_forcing_experiment/{571,891,csi_comparison.csv}` | no code in the repo | none | none | **orphan** (one-off; New Brunswick) |
| `figures/main_production/` | analysis scripts (`visualization.output_dir`) | – | `config.yml:421` | ok (v1 figures) |
| `figures/validation/{CAN,GBR,USA,*.csv}` | `run_validation.py`, `validate_country.py`, `rescore_*`, `run_multi_rp_summary.py` | `build_overview_table.py:132`, `compare_against_published.py:58` | `validation.output_dir` (:560) | ok, but this is tables/metrics, not figures |
| `figures/sfincs_runs/calibration_roughness_SFINCS{,_dedup}/` | `sfincs_tiles/gfm_config.read_figures_root` (:36-42) | – | literal `sfincs_runs/<base>` | ok |
| `figures/input_forcing_qa/` | `input_analysis/make_boundary_forcing_figures.py:2135` | – | literal | ok |
| `figures/500_tiles/` | `calibration_studies/plot_*` (:129, :158, …), `compare_slr_fingerprint_scenarios.py:65` | – | literal | not on disk now; `hydrax_rerun.yml:68` deletes and rebuilds it |
| `hpc_harness/{hydrax_rerun, smoke}/` | `hpc/harness.py:287-323` | harness | `state_dir` in `hpc/plans/*.yml` (hard-coded `/p/…`) | ok; `smoke` is test state |
| `input_analysis/boundary_forcing/{*.md, figures/}` | manual | – | none | ok (notes) |
| `inputs/` (27 entries) | downloads and sync scripts; **also generated**: `ICGM/geoid_*.tif` (rule outputs, `preprocessing.smk:12-36`), `deltadtm.vrt`, `DeltaDTM/_zips`; `DeltaDTM_masks` is rewritten in place by `fix_ocean_mask_with_osm_land.py:77,89` | everything | catalog, `config.yml:12-30,173-177,282,559` | ok. Two odd entries: `fes2014.zip` (10.5 GB) sits next to the extracted `fes2014/`, and `Storm-tide` is a 10 KB file with no extension |
| not on disk: `model_outputs_v2`, `merged_results_v2`, `local_vs_global_exposure` | – | – | config defaults; `hydrax_rerun.yml:220` | expected (v2 not run yet) |
| repo, not `R`: `adaptation_validation_output/`, `validation/overview_*.csv`, `run_timings.csv`, `batch_logs*/`, `sfincs.log` | `validate_adaptation_measures.py:208`, `build_overview_table.py:154`, manual redirects | – | literals | out of scope; listed so they aren't forgotten |

### 1.3 Duplicated path definitions in code

These are the places the migration must keep consistent:

- **Solver output pattern** `<MO>/<tid>/results/waterdepth_{rp}_{wl}.tif`: `simulation.smk:49-50`,
  `Snakefile:115`, `run_delta_floodmaps.py:73`, `run_eikonal_cli.py:70`,
  `generate_hpc_preprocess_job.py:63`.
- **Per-tile input names**: `preprocessing.smk:46-189` and `generate_hpc_preprocess_job.py:41-44`.
- **Chunk names**: `postprocessing.smk:9-121` and `generate_hpc_postprocess_job.py:251-262`.
- **Production tile grid**: `TG` vs the literals in `generate_calibration_batch_jobs.py:159` and
  `select_study_candidate_tiles.py:84`.
- **Root pairs that resolve to the same folder**:
  - `jobs_dir` vs `<MO>/hpc_jobs`
  - `flood_totals_dir` vs `<MR>/exposure/flood_totals`
  - `PPI` vs `MO` (in every overlay)

### 1.4 HPC-only path handling

- **SFINCS shell drivers.** `sfincs_tiles/site_env(.example).sh:9-15` sets `GFM_DATA_ROOT`. The
  drivers hard-code the layout:
  - `run_final_tile.sh:13,22,27-36`: `$BASE/<tile>/…`, `$DATA_ROOT/model_outputs/$TILE_ID/inputs`
  - `rerun_calibration.sh:34-51,88`: `${BASE}_dedup`, `hpc_jobs_stage2`
- **Generated job files.** `generate_calibration_jobs.py:29-30,91-219` writes Linux paths into the
  sbatch files and `resolved_config.yml`.
- **Harness plans.** `hpc/plans/hydrax_rerun.yml` hard-codes about 25 paths (:13, 30, 68-147, 191,
  212-226). `smoke.yml:7-8` hard-codes the harness state dir.
- **Existing job files are frozen.** Every existing `hpc_jobs*/` folder holds sbatch files,
  `submit_*.sh` and `resolved_config.yml` with absolute old paths baked in.
- **Simulation waves use existence checks only.** They bypass Snakemake:
  `generate_hpc_simulation_jobs.py --resume` globs the results, and `run_eikonal_cli.py:30` skips
  outputs that exist.
- **Preprocess and postprocess use Snakemake on the nodes.** They run
  `snakemake --rerun-triggers=mtime --rerun-incomplete` from `code_root`. Their metadata therefore
  lives in `/u/schlumbe/gfm_code/.snakemake` (keys `/p/…`). Local metadata is separate:
  `<repo>/.snakemake/metadata` holds 151 706 records keyed `P:/…`.
- **`*_materialized.yml` bake in the local root.** They contain `P:/` and `C:/Users/…` (lines 5-9)
  and must be regenerated on each machine.

---

## 2. Proposed layout

### 2.1 Tree

```
R/
├── inputs/                     raw downloads (unchanged; geoid/VRT products that already live here stay)
├── processed_inputs/           [ADD] derived inputs shared by several runs
│   ├── *.nc, *.csv, WL_scenarios/, WL_scenarios_cache/, storm_tide_product/   (unchanged)
│   └── tilings/
│       └── <tiling_id>/                    e.g. connectivity_v1, later fixed_<size>
│           ├── tile_grid.gpkg              (= TG)
│           ├── dem_source_manifest.csv     (fine tilings only)
│           ├── build/                      00_…04_*.gpkg, checkpoint, histograms, domain maps
│           ├── subsets/<subset>.gpkg       bangkok_chao_phraya, deltas_global, calibration_500_tiles, calibration/<group>
│           └── tile_inputs/<tid>/inputs/   per-tile dem/mask/friction/boundaries (= PPI)
├── main_production/
│   └── <release>/                          v1 (existing), v2 (planned rerun)      ← run root
├── parameter_adjustment/                   ← run root (500-tile eikonal sensitivity)
│   └── <study>_<friction>/                 sweep_budget_*, obstacle_coupling_* (study outputs)
├── sfincs_comparison/
│   ├── roughness_calibration/              (was calibration_roughness_SFINCS, internal layout unchanged)
│   └── roughness_calibration_dedup/        (was calibration_roughness_SFINCS_dedup)
├── case_studies/
│   └── <case>/<domain>/<input_variant>[__<test_variant>]/      ← run root
│       e.g. england/all_england/local_input, wales/severn_estuary/global_input,
│            san_francisco_bay/alameda/local_input, scotland/firth_of_forth/local_input,
│            england/wash/local_input__hop_test, new_brunswick/… (see Q4)
├── figures/
│   └── <same relative path as the run root>/   main_production/v2, parameter_adjustment,
│                                               sfincs_comparison/roughness_calibration, case_studies/…,
│                                               input_forcing_qa, input_analysis
└── hpc_harness/<plan>/         [ADD] harness state (cross-run operations)
```

Every run root has the same sub-structure. The relative paths below each root stay exactly what
the rules write today, so **no `.smk` output pattern changes**:

```
<run_root>/
├── model_outputs/<tid>/results/…          MO   (+ run_timings/, skipped_tiles/, oom_tiles/)
├── model_outputs/<tid>/inputs/…           only when the run builds its own inputs (PPI: null)
├── merged_results/{chunks/, exposure/, local_native/, plots/, *.vrt}    MR
├── validation/<ISO>/…                     validation.output_dir
├── hpc_jobs/{*.sbatch, resolved_config.yml, logs/, exposure/}            JD (sibling of MO, no longer inside it)
├── processed_inputs/WL_scenarios_cache/   run-specific station caches (local water levels)
└── scratch/                               run intermediates (e.g. delta_merge_scratch)
```

### 2.2 Naming convention

- **Folder names.** All folders are lowercase `snake_case`, with no dates and no `_v2`/`_new`/`_old`
  suffixes. The only version labels are `main_production/<release>` (`v1`, `v2`) and
  `tilings/<tiling_id>`.
- **Level order.**
  - Case studies: `<product>/<case>/<domain>/<input_variant>`.
  - Domains: reuse the names already in the tile-grid files (`all_england`, `severn_estuary`,
    `alameda`, `firth_of_forth`, `wash`).
  - `<input_variant>` is one of `local_input` or `global_input`.
- **Test and sensitivity variants** get a `__<variant>` suffix on the run root (the separator
  `flood_variants` already uses), never a new level.
- **Figures** mirror the run root under `figures/`. Validation metrics are tables, so they stay in
  the run root.
- **One template key.** The config gains `paths.run` (run root relative to `R`), and every output key
  derives from it:

  ```yaml
  paths:
    run: "main_production/v2"
    run_root: "{root}/{run}"
  simulation:
    model_outputs: "{run_root}/model_outputs"
    preprocessing_inputs_dir: "{processed_inputs_dir}/tilings/connectivity_v1/tile_inputs"
  hpc:
    jobs_dir: "{run_root}/hpc_jobs"
  postprocessing:
    merged_outputs: "{run_root}/merged_results"
  visualization:
    output_dir: "{figures_root}/{run}"
  validation:
    output_dir: "{run_root}/validation"
    flood_totals_dir: "{run_root}/merged_results/exposure/flood_totals"   # see Q6
  tile_grid:
    path: "{processed_inputs_dir}/tilings/connectivity_v1/tile_grid.gpkg"
  ```

  An overlay then sets only `paths.run`, `tile_grid.path` and, if needed,
  `preprocessing_inputs_dir: null`. That replaces 4-6 keys per overlay today and removes the C500
  leak into production.

### 2.3 Where things go

- **Logs.** SLURM logs go to `<run_root>/hpc_jobs/logs/`. No rule writes a local log; Snakemake's
  own log stays in `<checkout>/.snakemake/log`. Harness logs go to `hpc_harness/<plan>/logs/`.
- **Run timings and skip/OOM markers.** These stay in `<run_root>/model_outputs/{run_timings,
  skipped_tiles,oom_tiles}/`. Moving them out of `MO` would touch the resume logic in ~8 files
  (`eikonal_runner.py:339-340,439-480`, `run_eikonal.py:46,56`, `tile_split.py:140`,
  `collect_hpc_failures.py:92`, `check_simulation_progress.py:22`, `run_eikonal_dag.py:27`,
  `plot_merged_results.py:31`) for cosmetic gain. Not recommended now.
- **Resolved configs.** `<run_root>/hpc_jobs/resolved_config.yml` (as now). Dated copies such as
  `resolved_config_2026-10-15.yml` go next to it.
- **Test variants.** These are siblings with the `__<variant>` suffix, e.g.
  `case_studies/england/wash/local_input__hop_test` or `main_production/v2__no_correction_b`.
  Throw-away tests (`hpc_harness/smoke`) are deleted when done.
- **Intermediates.**
  - Shared caches stay in `processed_inputs/storm_tide_product/cache/`.
  - Run caches go to `<run_root>/processed_inputs/` and scratch to `<run_root>/scratch/`.
  - Node-local staging (`$TMPDIR/gfm_*`) is unchanged.

### 2.4 Is the top level missing something?

Yes, three things:

1. **`processed_inputs/`** is shared by production, parameter adjustment, the SFINCS comparison and
   the global-input case studies. It belongs to no single product, and `inputs/` should stay raw.
   It also gives the central tilings a home.
2. **`hpc_harness/`** holds operational state across runs. An alternative is to nest it under a
   `_ops/` folder.
3. **A home for Bangkok, the global deltas, `local_vs_global_exposure` and the Fundy experiment.**
   None of them is a validation case. See Q4.

---

## 3. Open questions

**Q1. Should tile information (manifest, masks, tile geometry) live in one central folder used by
production and validation runs?**

Recommendation: yes, as `processed_inputs/tilings/<tiling_id>/`, at two levels.

- **(a) Tile definitions, always central.** This covers the grid, the DEM-source manifest, build
  intermediates, domain maps and subsets.
  - They are small, every run reads them, and today they are scattered across `PI/mask/`,
    `PI/mask/validation/`, `PI/mask/calibration/` and `calibration_500_tiles/`.
  - The fine case-study grids become tilings of their own (`england_fine_all_england`,
    `wales_fine_severn_estuary`, …), because their tile ids differ from the global grid.
  - Subsets of the global grid share its ids, so they go into `subsets/`.
- **(b) Per-tile prepared inputs: central only where they are actually shared.** That is the global
  DeltaDTM + COAST-RP set, which v2 production, parameter adjustment and the SFINCS comparison all
  read. The SFINCS comparison already reads them through a hard-coded `model_outputs`.
  - Case studies with local DEMs or local water levels keep their inputs inside their run root
    (`PPI: null`), because those inputs belong to that run only.
  - So do Bangkok and the deltas runs, which computed their own.

How this works for both tilings:
- **Connectivity domains vs fixed tiles.** The tile id is only meaningful together with
  `<tiling_id>`, so both coexist. A fixed-tile grid becomes `tilings/fixed_<size>/` with its own
  `tile_inputs/`. Old runs keep pointing at `connectivity_v1`.
- **Numeric ids.** Keep tile ids numeric within each tiling. The wildcard constraint
  `tile_id = \d+` (`rules/common.smk:4`, protected) then needs no change.
- **Chunks are unaffected.** Merged chunks use a fixed 5° grid (`N60W005`), independent of the
  tiling.

Trade-offs:
- **Shared mutable state.** Rewriting a central file changes it for every consumer. `run_eikonal`
  takes `TG` as input, so rewriting `tile_grid.gpkg` in place makes all downstream outputs of every
  run on that tiling look stale (mtime trigger). `tile_split.py:137` does exactly that, as does
  re-running `build_tile_manifest`.
  - Rule: a tiling is write-once. Changes produce a new `<tiling_id>`.
  - `tile_split` needs a fix before it is used on a central grid (TODO).
- **Production inputs move twice.** Splitting v1 `model_outputs/<tid>/{inputs,results}` takes one
  rename per tile: about 9000 `mv` calls, minutes on the HPC.
  - The alternative is to move `model_outputs` whole to `main_production/v1/model_outputs` and
    point v2's `PPI` there.
  - That is one rename, but v2, parameter adjustment and the SFINCS comparison then depend on a v1
    folder, so v1 results can never be dropped independently.

**Q2. One run root per v2 production release, or overwrite in place?**
- Recommendation: `main_production/v1` (existing) and `main_production/v2` (new).
- Trade-off: v1 stays readable for comparisons at the cost of disk space. v1's ~87 k chunks and
  ~200 k rasters can be deleted later in one step.

**Q3. Validation metrics in the run root or under `figures/`?**
- Recommendation: run root (`<run_root>/validation/`). Scripts already read and write CSVs there
  (`build_overview_table.py`, `compare_against_published.py`). Only deliverable figures go to
  `figures/`.
- Trade-off: the validation PNGs written by `plot_agreement_map.py` move with the metrics, so
  validation figures are not all under `figures/`. Splitting them needs a second key and code
  changes in `plot_agreement_map.py:386-472` and `plot_sf_bay.py:85-98`.

**Q4. Where do non-validation case runs go?** These are Bangkok, the global deltas,
`local_vs_global_exposure` and `fundy_extreme_forcing_experiment`.
- Recommendation: everything under `case_studies/`, with the same convention:
  - `case_studies/bangkok/chao_phraya/global_input`
  - `case_studies/global_deltas/{ssp126,ssp245}`, with the cross-SSP `delta_flood_extent.csv`
    and figures at `case_studies/global_deltas/`
  - `case_studies/new_brunswick/bay_of_fundy/extreme_forcing_experiment`
  - `case_studies/england_wales/local_vs_global_exposure`
- Trade-off: `case_studies` then means "regional runs", not strictly validation cases. The
  alternative is a seventh top-level folder, `applications/`.

**Q5. New Brunswick has no run of its own.**
- Today CAN is scored from the production run (`validation.output_dir/CAN`).
- Recommendation: the production-scored metrics stay in `main_production/<release>/validation/CAN`.
  `case_studies/new_brunswick/` holds only the Fundy experiment until a local-input run exists.
- Trade-off: one validation case lives in two places.
- The same split applies to GBR: production-scored metrics for England, Wales and Scotland sit in
  `main_production/…/validation/GBR`, while the local/global comparison runs sit in `case_studies`.

**Q6. `validation.flood_totals_dir`.**
- Today it is the literal `{root}/merged_results_v2/exposure/flood_totals`, so every case-study run
  reads production totals.
- Recommendation: derive it from `run_root`. For production the value is unchanged; for case
  studies the scored totals change, which is a **result change**.
- The parity alternative is to pin it to `{root}/main_production/v2/merged_results/exposure/flood_totals`
  and log the bug in TODO.

**Q7. SFINCS comparison internal layout.**
- Recommendation: keep it unchanged. Only the base name becomes
  `sfincs_comparison/roughness_calibration`, and the dedup sibling becomes
  `sfincs_comparison/roughness_calibration_dedup`.
  - The code builds `root / base_dir_name` and `<base>_dedup`, so a nested `--base-dir-name` should
    work without dedup changes. To be confirmed by a dry run of the generators.
- Moving the ~1000 tile folders into a `tiles/` subfolder would be tidier but touches ~12 scripts and
  `run_final_tile.sh:28-36`.

**Q8. Orphans.** These are candidates for deletion after the move:
- `sfincs_calibration_bline/`
- `sf_bay_global_input/`
- `deltas_floodmaps/previous_without_protection/`
- `processed_inputs/mask/domain_tiles_global_pre_connectivity_migration.gpkg`
- `processed_inputs/mask/thailand_bangkok_tiles.gpkg`
- `processed_inputs/mask/tile_generation_debug/`
- `hpc_harness/smoke/`
- `inputs/fes2014.zip`

Recommendation: delete only after the restructure has been verified, and list each with its size first.

---

## 4. Change list

### 4.1 Old → new data paths

| old (`R/…`) | new (`R/…`) |
|---|---|
| `model_outputs/<tid>/inputs/` | `processed_inputs/tilings/connectivity_v1/tile_inputs/<tid>/inputs/` (Q1 (b); else stays with v1) |
| `model_outputs/<tid>/results/`, `model_outputs/{run_timings,skipped_tiles,oom_tiles}/` | `main_production/v1/model_outputs/…` |
| `model_outputs/hpc_jobs/` | `main_production/v1/hpc_jobs/` |
| `merged_results/` | `main_production/v1/merged_results/` |
| `figures/main_production/` | `figures/main_production/v1/` |
| `figures/validation/` | `main_production/v1/validation/` |
| `processed_inputs/mask/domain_tiles_global.gpkg` | `processed_inputs/tilings/connectivity_v1/tile_grid.gpkg` |
| `processed_inputs/mask/{00_…04_*.gpkg, *checkpoint*.jsonl, *histogram*.png, domains_final*.png}` | `processed_inputs/tilings/connectivity_v1/build/` |
| `processed_inputs/mask/{bangkok_chao_phraya,deltas_global,gbr_wales_scotland_friction9}_tiles.gpkg` | `processed_inputs/tilings/connectivity_v1/subsets/<name>.gpkg` |
| `processed_inputs/mask/calibration/*.gpkg` | `processed_inputs/tilings/connectivity_v1/subsets/calibration/` |
| `calibration_500_tiles/calibration_500_tiles_grid.gpkg` | `processed_inputs/tilings/connectivity_v1/subsets/calibration_500_tiles.gpkg` (verify its ids are global ids) |
| `processed_inputs/mask/validation/<case>_fine_tiles_<domain>.gpkg` (+ manifest) | `processed_inputs/tilings/<case>_fine_<domain>/{tile_grid.gpkg, dem_source_manifest.csv}` |
| `processed_inputs/mask/validation/england_fine_tiles.gpkg` (domain of the existing England runs) | `processed_inputs/tilings/england_fine_<domain>/` (domain name: decision D6) |
| `calibration_500_tiles/` (rest) | `parameter_adjustment/` (`model_outputs/hpc_jobs` → `parameter_adjustment/hpc_jobs`) |
| `figures/500_tiles/` | `figures/parameter_adjustment/` |
| `calibration_roughness_SFINCS/` | `sfincs_comparison/roughness_calibration/` |
| `calibration_roughness_SFINCS_dedup/` | `sfincs_comparison/roughness_calibration_dedup/` |
| `figures/sfincs_runs/calibration_roughness_SFINCS{,_dedup}/` | `figures/sfincs_comparison/roughness_calibration{,_dedup}/` |
| `england_local_input/` | `case_studies/england/<domain>/local_input/` |
| `england_global_input/` | `case_studies/england/<domain>/global_input/` |
| `england_hop_test/` | `case_studies/england/wash/local_input__hop_test/` |
| (config only) `england_{local,global}_input/all_england/` | `case_studies/england/all_england/{local,global}_input/` |
| `wales_{local,global}_input/` | `case_studies/wales/severn_estuary/{local,global}_input/` |
| `scotland_local_input/` | `case_studies/scotland/firth_of_forth/local_input/` |
| `sf_bay_local_input/` | `case_studies/san_francisco_bay/alameda/local_input/` |
| `bangkok_chao_phraya/` | `case_studies/bangkok/chao_phraya/global_input/` (Q4) |
| `deltas_floodmaps/ssp{126,245}/` | `case_studies/global_deltas/ssp{126,245}/` (`delta_merge_scratch` → `scratch/`) |
| `deltas_floodmaps/{delta_flood_extent.csv, figures/}` | `case_studies/global_deltas/…` (figures → `figures/case_studies/global_deltas/`) |
| `fundy_extreme_forcing_experiment/` | `case_studies/new_brunswick/bay_of_fundy/extreme_forcing_experiment/` |
| `input_analysis/boundary_forcing/` | `figures/input_analysis/boundary_forcing/` |
| `<case>/figures/`, `<case>/agreement_*.png` | `figures/case_studies/<case>/<domain>/<input>/`; the agreement PNGs follow `validation.output_dir.parent` = run root |

### 4.2 Config (protected: needs go-ahead)

- **Snakefile placeholders.** `Snakefile:54-75` gets `{run}` and `{run_root}` in the `_PATH_SUBS`
  passes. `{run_root}` must expand before the keys that use it.
- **config_utils.** `src/config_utils.py:105,143-150` gets the same expansion. `materialize_config`
  (:201-216) keeps templates; the fact that it bakes in `config_local.yml` is a separate TODO.
- **config.yml keys.** Lines 84, 195, 196, 300, 340, 421, 560 and 584 get the derived values from
  2.2. Add `paths.run`. Remove the comments at :13-15 that list `figures/` subfolders.
- **Overlays (11 files).** Replace `simulation.model_outputs`, `hpc.jobs_dir`,
  `postprocessing.merged_outputs`, `visualization.output_dir` and `validation.output_dir` with
  `paths.run`. Repoint `tile_grid.path`, `simulation.dem_source_manifest` and
  `boundary_conditions.stations_cache_dir` to `{run_root}/processed_inputs/WL_scenarios_cache`.
  - `bangkok_chao_phraya.yml:9-22`
  - `calibration_500_tiles.yml:8-39`
  - `deltas_ssp126.yml`, `deltas_ssp245.yml` (also `waterlevel_nc_dir` → `{run_root}/WL_scenarios`)
  - `england_{local,global}_input.yml`, `england_hop_test.yml`
  - `wales_{local,global}_input.yml`, `scotland_local_input.yml`, `sf_bay_local_input.yml`
- **Regenerate.** Rebuild all `*_materialized.yml` with `materialize_configs.py`, on each machine.
- **Variants.** `src/flood_variants.py:68-90` appends `__<variant>` to `paths.run` instead of to
  each output key. This also changes the test `tests/flood_models_validation/validate_flood_models.py:98-115`.

### 4.3 Hard-coded paths in code

All of these move to config-derived values: one constant or config key, no literals.

**sfincs_tiles**
- One default for the base name: `sfincs_comparison.base_dir_name` in config.yml, read by
  `gfm_config.py`. Replace the `--base-dir-name` default in 24 scripts:
  `build_elevation.py:204`, `build_line_forcing.py:120`, `build_roughness.py:65`,
  `build_sfincs_tile.py:418`, `build_station_boundary_lines.py:348`, `calibration_split.py:72`,
  `check_calibration_progress.py:94`, `compute_calibration_result.py:247`,
  `compute_calibration_result_dedup.py:77-78`, `compute_variant_metrics.py:32`,
  `exposure_depth.py:485`, `exposure_depth_pass.py:177`, `generate_calibration_jobs.py:237,254`,
  `plot_depth_accuracy.py:107`, `plot_sfincs_tile_diagnostics.py:489`,
  `plot_validation_results.py:351`, `plot_worst_tiles_panel.py:182`, `preflight_sfincs_mask.py:64`,
  `regenerate_dem_mask.py:65`, `run_calibration_models.py:220`,
  `run_calibration_sweep_analysis.py:49`, `run_sfincs_tile.py:163`, `select_calibration_tiles.py:301`.
- `gfm_config.py:36-42`: `figures_root/sfincs_runs/<base>` → `figures_root/<base>`.
- Tile-input source `model_outputs` → `PPI`:
  - `build_station_boundary_lines.py:355`
  - `preflight_sfincs_mask.py:65`
  - `select_calibration_tiles.py:152`
  - `generate_calibration_jobs.py:112,114`
  - `run_final_tile.sh:27`: becomes `$GFM_TILE_INPUTS`, a new variable in `site_env.example.sh`
    that must match `PPI`; add it to the `check_site_roots` comparison.
- `run_final_tile.sh:13`, `rerun_calibration.sh:34`: default `BASE_DIR_NAME`.
- `plot_sfincs_tile_diagnostics.py:1`: docstring path.

**calibration_studies**
- `generate_calibration_batch_jobs.py:104,145,159`: study dir and grid from config, not literals.
- `select_study_candidate_tiles.py:33,84,85`
- `figures_root/"500_tiles"` in `plot_obstacle_coupling_calibration_bands.py:111,129`,
  `plot_sweep_calibration_bands.py:141,158`, `plot_sweep_time_vs_size.py:73`,
  `compare_slr_fingerprint_scenarios.py:9,60,65`.

**analysis**
- `root/"deltas_floodmaps"` → a `deltas.analysis_dir` key:
  - `compute_delta_flood_extent.py:194-195`
  - `add_slr_year_column.py:59,66`
  - `plot_delta_affected_fraction_timeseries.py:103-113`
  - `plot_delta_flood_map.py:87-94`
- Docstrings mentioning `<merged_results>/exposure` (10 files, listed in 1.2) need no change: the
  placeholder stays valid.

**preparation**
- `build_tile_manifest.py:127,168,222-232`: intermediates go to `TG.parent/build` (behaviour:
  path only).
- `build_delta_tile_subset.py:75`, `select_country_tiles.py:120,166`,
  `build_{england,wales,sf_bay}_fine_tiles.py:162/169/167`, `plot_domain_maps.py:131`: write into
  `tilings/<id>/…`.
- `merge_england_local_tiles.py:30`: no change (uses `MR`).

**Launchers**
- `run_bangkok_case_study.bat:30,37-45,85,90`: derive `STUDY_ROOT` from the materialized
  `paths.run` instead of hard-coding it.
- `run_delta_floodmaps.py:73`: no change (it uses `MO`), but it is a duplicate of the pattern.

**src**
- `pipeline_plots.py:26,57`: unchanged (`inputs/` is fixed).
- `tile_split.py:111` reads `MO/<tid>/inputs/mask.tif` instead of `PPI`. This is an existing bug and
  must be fixed before central tile inputs are used; goes to TODO. Same for `tile_split.py:137`,
  which rewrites `TG` in place.

**Other code**
- `validation/rescore_1km_fp_buffer.py:291-293`: help text.
- `boundary_test/extract_gtsm_series.py:38`
- `review_critiques/calibration_experiments.py`, `review_critiques/plot_calibration_experiments.py`
- Tests: `test_sfincs_smoke.py:30,45,83`, `validate_truncation_window.py:25,29,58-83`,
  `validate_antimeridian_handling.py:23,29-32`.
- `hpc/plans/hydrax_rerun.yml:13,30,68-147,191,212-226` and `smoke.yml:7-8`: rewrite against the
  new layout.

**Docs**
- `README.md:17-21,54,61` (already stale: `<data_root>/figures`, `<data_root>/validation`)
- `docs/running.md:42-224` (≈20 lines)
- `docs/hpc.md:27-186` (≈11 lines)
- `docs/methods_02_flood_depth.md:89`
- `docs/methods_04a_SFINCSvalidation.md:12-131`
- `docs/methods_04b_MapsValidation.md:5-115`
- `docs/data.md:66-85`
- `docs/figure_scripts/plot_*` defaults
- `AUDIT.md:37,72-84`
- `TODO.md`: ~25 path mentions, plus the `figures_v2`/`validation_v2` plan at :78
- `CLAUDE.md` Layout section (`calibration_500_tiles` name)

**HPC-specific**
- `sfincs_tiles/site_env.sh` and `site_env.example.sh`: add `GFM_TILE_INPUTS`. `GFM_DATA_ROOT`
  is unchanged.
- `config_hpc.yml`: no output keys, no change.
- Regenerate all job folders. Never resubmit an old `submit_*.sh` or `resume_*.sh`, since they
  contain absolute old paths.
- The HPC checkout's `.snakemake/` needs the same metadata migration as the local one (4.4,
  step 3).

### 4.4 Migration order (outputs stay valid, nothing recomputed)

**Mechanism** (checked in Snakemake 7.32.4):
- `persistence.py:413-440` fires the params/input/code/software-env triggers only when a metadata
  record exists for the output path. So a moved output with no record is judged on mtime alone.
- A same-filesystem `mv` keeps mtimes. Directory outputs are timed by an internal
  `.snakemake_timestamp` (`io.py:505-525`), which moves with them.

**Two traps follow from that:**
1. **A plain move silently clears pending reruns.** TODO.md:33 records accepted param-triggered
   reruns (+25 jobs for Bangkok). After a move those records are gone, so the jobs no longer show
   up. Parity requires migrating the metadata, not dropping it.
2. **Copying instead of moving resets mtimes.** Copying any input (above all `tile_grid.gpkg`, an
   input of every `run_eikonal`) makes everything downstream stale.

   Rule: always `mv` within the same filesystem, never `cp`. Check first with
   `stat -c %d <src> <dst-parent>`.

**Order:**

1. **Preconditions.**
   - Confirm no HPC batch and no local Snakemake run is active.
   - Confirm `.snakemake/incomplete/` is empty in both checkouts.
   - Record baselines in `.cleanup_baseline/restructure/`, for every materialized subset config:
     - the dry-run job count (Bangkok: 480)
     - `--list-params-changes` and `--list-input-changes`
   - Record a file count per directory to be moved, and a sample of 100 file mtimes per tree.
   - The user runs the same on the HPC checkout.
2. **Code, values unchanged (maintain).**
   - Introduce `{run}`/`{run_root}` and replace all literals from 4.3 with config-derived values.
   - Every key still resolves to today's path.
   - Proof: materialize all configs before and after and compare the expanded values (identical),
     then `ruff check .`, `pytest -m "not needs_data"`, and the Bangkok dry-run count = 480.
3. **Metadata migration tool.** A script with a dry-run mode, run once per checkout:
   - Rename each `.snakemake/metadata` record from old to new path (urlsafe-b64 file names, split
     into `@` dirs as in `persistence.py:598-610`).
   - Rewrite the old→new prefixes inside each record's `input` and `params` fields.
   - Apply the longest prefix first.
4. **Move tree by tree, smallest first.** Each tree is one commit. For each tree:
   1. `mv` on the HPC.
   2. Migrate metadata in both checkouts.
   3. Commit the config switch for that tree.
   4. Re-materialize on both machines.
   5. Verify: dry-run count and list-changes equal the baseline, the mtime sample is unchanged,
      the file count is equal, and the user runs the progress checks.

   Order of trees:
   1. figures-only folders (`500_tiles`, `sfincs_runs`, `input_forcing_qa`, `input_analysis`)
   2. `sfincs_comparison` (not Snakemake; verify with `check_calibration_progress.py`, run by the user)
   3. `parameter_adjustment`
   4. Bangkok (pilot; baseline 480)
   5. the deltas
   6. the Wales, Scotland, SF Bay and England case studies
   7. the tilings (`PI/mask` → `PI/tilings`, in the same commit as all `tile_grid.path` values)
   8. production v1 last: per-tile split of `model_outputs` into `tile_inputs` and
      `main_production/v1`, then `merged_results`, `hpc_jobs` and `figures/main_production`
5. **Docs, harness plans, TODO** updated in one commit.
6. **Orphans** deleted on approval (Q8), listing sizes first.

---

## 5. Decisions needed

- [ ] **D1 Top level.** Add `processed_inputs/` and `hpc_harness/` as top-level folders (2.4)?
- [ ] **D2 Central tiles (Q1).** Definitions central: yes or no. Production per-tile inputs:
  (a) split into `processed_inputs/tilings/connectivity_v1/tile_inputs` (~9000 renames, v1 results
  independent), or (b) leave them inside `main_production/v1/model_outputs` (one rename, v2 depends
  on the v1 folder)?
- [ ] **D3 Releases (Q2).** `main_production/v1` + `v2` side by side?
- [ ] **D4 Validation location (Q3).** Run root (recommended) or `figures/`?
- [ ] **D5 Non-validation runs (Q4).** Bangkok, the deltas, local-vs-global and Fundy under
  `case_studies/`, or a new `applications/`?
- [ ] **D6 England domain.** What is the domain name of the existing `england_{local,global}_input`
  runs (grid `england_fine_tiles.gpkg`): `wash` or something else? Does `all_england` replace it or
  sit beside it?
- [ ] **D7 New Brunswick/GBR (Q5).** Production-scored metrics stay with the production run?
- [ ] **D8 `flood_totals_dir` (Q6).** Derive it per run (a result change for case studies) or pin it
  to production (parity, bug logged)?
- [ ] **D9 SFINCS internal layout (Q7).** Keep the tile folders at base level (recommended)?
- [ ] **D10 Orphans (Q8).** Approve the deletion list after verification.
- [ ] **D11 Protected changes.** Go-ahead to edit the `config.yml` defaults and the overlays, plus
  `site_env.example.sh` (new `GFM_TILE_INPUTS`).
- [ ] **D12 Timing.** Restructure before the v2 rerun starts (recommended: v2 then writes straight
  into `main_production/v2`), and is any HPC batch running now?
