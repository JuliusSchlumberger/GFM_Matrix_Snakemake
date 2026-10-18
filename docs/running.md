# Running the pipeline

Commands run from `<code_root>` in the `gfm` env ([install.md](install.md)).
Preparation, the Snakemake DAG, analysis and validation are separate runs; none triggers the next.
The full global grid runs on SLURM ([hpc.md](hpc.md)); this page covers the local entry points.

## 1. Preparation

Run once before the DAG, and again whenever DeltaDTM or boundary inputs change:

```
python preparation/run_preparation.py [STEP ...] [--config <config.yml>] [--skip-existing] [--fail-fast]
```

Steps, in order (also the names of the `preparation.*` switches in `config.yml`):

| Step | Does |
|---|---|
| `sync_deltadtm` | download the DeltaDTM DEM and mask tiles |
| `fix_ocean_mask` | reclassify ocean cells inside OSM land polygons, in place |
| `build_deltadtm_vrt` | build the DEM and mask VRT mosaics |
| `tile_generation` | build the connectivity-first tile grid (`tile_grid.path`), scenario-independent |
| `boundary_conditions` | write the COAST-RP + SLR scenario water-level NetCDFs |

Without a STEP it runs the steps enabled under `preparation.*` (default: all); naming steps runs exactly those.
A failed step is reported and the rest continue unless `--fail-fast`.

## 2. Scenario configs

A scenario is an override file `snakemake_workflow/config/<scenario>.yml` (plus optional `<scenario>_pins.yml`) layered on `config.yml` + `config_local.yml`.
Build the merged config, then point `GFM_CONFIG_PATH` at it:

```
python snakemake_workflow/scripts/materialize_configs.py [<scenario> ...]
export GFM_CONFIG_PATH=snakemake_workflow/config/<scenario>_materialized.yml
```
```
python snakemake_workflow/scripts/materialize_configs.py bangkok_chao_phraya
export GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml
```

Without arguments it rebuilds every known scenario (`bangkok_chao_phraya`, `calibration_500_tiles`, `deltas_ssp126`, `deltas_ssp245`).
`*_materialized.yml` are generated: rerun the script after editing `config.yml`, `config_local.yml` or the override, never hand-edit them.
Without `GFM_CONFIG_PATH`, Snakemake uses `config.yml` + `config_local.yml`.

## 3. Snakemake DAG

| Target | Runs |
|---|---|
| `preprocess` | per-tile DEM, mask, friction and boundary inputs |
| `simulate` | the eikonal solve per (tile, return period, SLR scenario) |
| `postprocess` | merge to chunks, exposure grid, flood fractions, plots |
| `all` | all three |

```
snakemake <target> --cores 4 --resources mem_mb=8000
```

`run_eikonal_dag.py` wraps the same call (default target `simulate`) and, when tiles run out of memory, splits them into overlapping halves and reruns, up to `tile_split.max_retries` times:

```
python run_eikonal_dag.py [--target all] [--cores 4] [--mem-mb 8000] [--config <config.yml>]
```

`profiles/default/config.yaml` (loaded automatically) limits rerun triggers to `mtime, params, input, software-env`: editing code never reruns finished outputs, so after a logic change delete the affected outputs or force them (`--forcerun <rule>`).

### Hop ordering

A tile with `hop_distance >= 1` is seeded from the outputs of its lower-hop neighbours for the same scenario, and that dependency is not in the DAG.
All hop-0 tiles must finish before hop-1 tiles start, and so on.
A local multi-hop `simulate` can therefore start a hop-1 job too early; that job fails without output, and rerunning `simulate` completes it.
To avoid the failures, run the waves one by one as `run_delta_floodmaps.py` does (one Snakemake call per `hop_distance`, waiting for each).

### Dry runs

Never dry-run the full grid locally (it does not finish); use a subset config:

```
GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml snakemake -n all --cores 1
```

### Progress monitors

Read-only, safe to run any time (locally or on the cluster):

```
python check_preprocess_progress.py [--config <config.yml>] [--watch SECONDS]
python check_simulation_progress.py [--config <config.yml>] [--watch SECONDS]
python check_postprocess_progress.py [--config <config.yml>] [--watch SECONDS]
python sfincs_tiles/check_calibration_progress.py [--list-failed] [--watch SECONDS]
```

## 4. Bangkok / Chao Phraya case study

Nine tiles: core tiles 1026, 1048, 1195 (hop 0) and 3930, 3933, 3943 (hop 1), plus hop-0 seeding neighbours 1092, 1194, 1199.
Outputs go to `<data_root>/bangkok_chao_phraya/`.

1. Build the tile subset from the global tile grid:
   ```
   printf '%s\n' 1026 1048 1092 1194 1195 1199 3930 3933 3943 > bangkok_tile_ids.txt
   python preparation/build_tile_subset_by_id.py --tile-ids-file bangkok_tile_ids.txt \
       --out <data_root>/processed_inputs/mask/bangkok_chao_phraya_tiles.gpkg
   ```
   The neighbour list is the lower-hop closure of the core tiles (`_hop_distance_closure` in `preparation/select_country_tiles.py`); recompute it if the tile grid changes.
2. Materialize the config and preprocess:
   ```
   python snakemake_workflow/scripts/materialize_configs.py bangkok_chao_phraya
   export GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml
   snakemake preprocess --cores 3
   ```
3. Simulate hop 0 first, then hop 1 (rerun `snakemake simulate --cores 3` until it completes, see [Hop ordering](#hop-ordering)).
4. Postprocess, case-polygon exposure and plots:
   ```
   snakemake postprocess --cores 3
   python analysis/compute_bangkok_case_exposure.py --config $GFM_CONFIG_PATH \
       --case-polygon-gpkg <data_root>/bangkok_chao_phraya/bangkok_chao_phraya_domain.gpkg --case-polygon-layer bangkok_tile
   python analysis/plot_bangkok_exposure_vs_rp.py \
       --exposure-csv <data_root>/bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv \
       --out-dir <data_root>/bangkok_chao_phraya/figures
   python analysis/plot_bangkok_case_flood_map.py --config $GFM_CONFIG_PATH \
       --case-polygon-gpkg <data_root>/bangkok_chao_phraya/bangkok_chao_phraya_domain.gpkg --case-polygon-layer bangkok_tile \
       --return-period RP100
   python analysis/plot_bangkok_eai_timeseries.py --config $GFM_CONFIG_PATH \
       --exposure-csv <data_root>/bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv
   ```

On Windows, `run_bangkok_case_study.bat` runs steps 3-4 in one go, stopping at the first failure:

```
set "GFM_ENV=<env>"
run_bangkok_case_study.bat
```
```
set "GFM_ENV=C:\Users\me\miniforge3\envs\gfm"
run_bangkok_case_study.bat
```

`GFM_ENV` is the env directory holding `python.exe` and `Scripts\snakemake.exe`.
The batch file simulates with `run_simulation_batch.py` and expects, besides the materialized config, `<data_root>/bangkok_chao_phraya/resolved_config.yml` (the path-expanded config) and `dispatch/wave0_triples.csv` / `wave1_triples.csv` (one `tile_id,return_period,waterlevel_name` per line, no header) there; set `STUDY_ROOT` to use another study directory.

## 5. Analysis

Exposure and adaptation analysis and figures, after `postprocess`; the steps run are the `analysis.*` switches in the config:

```
python analysis/run_analysis.py [--config <config.yml>] [--expdir <dir>] [--fig-dir <dir>] \
    [--only-exposure | --only-plots] [--skip-exposure] [--skip-flood-totals] [--skip-world-maps] [--fail-fast]
```

## 6. Validation

Soft-scored HR/FAR/CSI against the national hazard maps of Wales, Scotland, England (GBR) and New Brunswick (CAN), after `postprocess`:

```
python validation/run_validation.py [--config <config.yml>] [--countries GBR CAN] [--skip-plots] [--fail-fast]
```

Standalone scripts, all taking `--config <config.yml>`:

| Script | Does |
|---|---|
| `validation/fetch_england_flood_zones.py` | download the EA Flood Zones 2/3 benchmark (modelled coastal sources) |
| `validation/build_sf_bay_benchmark_mosaic.py` | mosaic the CoSMoS county depth rasters for SF Bay |
| `validation/validate_sf_bay.py` | score SF Bay depth against CoSMoS (RP100, no SLR) |
| `validation/plot_sf_bay.py` | plot the SF Bay agreement and depth-difference maps |
| `validation/rescore_england_fz3.py` | score England Flood Zone 3 in bounded memory |
| `validation/validate_england_fz2_band.py` | score England Flood Zone 2 against the model's RP100-RP1000 band |
| `validation/compare_against_published.py` | tabulate our metrics next to published studies |
| `validation/build_overview_table.py` | build the overview tables from the CSVs above |

## 7. Global deltas

RP100 flood maps for the global deltas under ssp126 and ssp245; runs boundary conditions, preprocess, wave-by-wave simulate and postprocess per scenario:

```
python preparation/build_delta_tile_subset.py
python run_delta_floodmaps.py [--scenarios ssp126 ssp245] [--stages ...] [--cores 4] [--mem-mb <MB>] [--skip-boundary-conditions]
```

Design-event floods of the FLOPROS-protected deltas: `python analysis/prepare_delta_design_runs.py`, then `python run_delta_floodmaps.py --scenarios design --stages boundary_conditions preprocess simulate`.
