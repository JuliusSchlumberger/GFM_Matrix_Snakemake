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
Only Snakemake reads `GFM_CONFIG_PATH`; without it, Snakemake uses `config.yml` + `config_local.yml`, and the Python scripts take the config as `--config <config.yml>`.
A scenario writes its outputs where its override sets `simulation.model_outputs`, `postprocessing.merged_outputs` and `visualization.output_dir`.
`<scenario>_pins.yml` fixes the values the scenario's published outputs were computed with (e.g. `friction_scale_factor: 9.0` for Bangkok): keep it to reproduce them, or delete its keys to follow `config.yml`.

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

Set `mem_mb` to about 80% of the machine's RAM; Snakemake uses it to limit how many tiles solve at once.
`run_eikonal_dag.py` wraps the same call (default target `simulate`) and, when tiles run out of memory, splits them into overlapping halves and reruns, up to `tile_split.max_retries` times:

```
python run_eikonal_dag.py [--target all] [--cores 4] [--mem-mb 8000] [--config <config.yml>]
```

`profiles/default/config.yaml` (loaded automatically) limits rerun triggers to `mtime, params, input, software-env`: editing code never reruns finished outputs, so after a logic change delete the affected outputs or force them (`--forcerun <rule>`).

### Scenario names

The solve writes `<model_outputs>/<tile_id>/results/waterdepth_<return_period>_<waterlevel_name>.tif`.
`return_period` is `RP<years>` for each entry of `boundary_conditions.return_periods` (default `RP2` ... `RP1000`).
`waterlevel_name` is `SLR_<mm>` for each entry of `boundary_conditions.waterlevel_names` plus `adaptation.slr_intensities` (default `SLR_0`, `SLR_500`, `SLR_1000`, `SLR_1500`, `SLR_2000`).

### Hop ordering

A tile with `hop_distance >= 1` is seeded from the outputs of its lower-hop neighbours for the same scenario, and that dependency is not in the DAG.
All hop-0 tiles must finish before hop-1 tiles start, and so on.
A multi-hop `snakemake simulate` can start a hop-1 job too early; that job fails without output, Snakemake then starts no new jobs (unless `--keep-going`), and rerunning `simulate` completes it.

To run one hop at a time, request its output files as targets instead of `simulate`, and wait for each call to finish (`run_delta_floodmaps.py` does this per `hop_distance`).
Each tile's `hop_distance` is a column of the tile grid (`tile_grid.path`):

```
snakemake --cores <N> <model_outputs>/{<tile_id>,...}/results/waterdepth_RP{<years>,...}_SLR_{<mm>,...}.tif
```
```
MO=<data_root>/bangkok_chao_phraya/model_outputs
snakemake --cores 3 $MO/{1026,1048,1092,1194,1195,1199}/results/waterdepth_RP{2,5,10,25,50,100,250,500,1000}_SLR_{0,500,1000,1500,2000}.tif
snakemake --cores 3 $MO/{3930,3933,3943}/results/waterdepth_RP{2,5,10,25,50,100,250,500,1000}_SLR_{0,500,1000,1500,2000}.tif
```

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

Nine tiles: core tiles 1026, 1048, 1195 (hop 0) and 3930, 3933, 3943 (hop 1), plus hop-0 seeding neighbours 1092, 1194, 1199; 9 return periods x 5 SLR levels each.
Outputs go to `<data_root>/bangkok_chao_phraya/`.
Runtime, memory and disk figures for this case are not documented; ask the project maintainers (`<issue_tracker>`, see [data.md](data.md)).

1. Run the preparation once with the default config (no `--config`): its steps are global, and Bangkok overrides none of their settings.
   ```
   python preparation/run_preparation.py
   ```
2. Build the tile subset from the global tile grid (`tile_grid.path` of `config.yml` + `config_local.yml`), at the `tile_grid.path` that `bangkok_chao_phraya.yml` sets:
   ```
   printf '%s\n' 1026 1048 1092 1194 1195 1199 3930 3933 3943 > bangkok_tile_ids.txt
   python preparation/build_tile_subset_by_id.py --tile-ids-file bangkok_tile_ids.txt \
       --out <data_root>/processed_inputs/mask/bangkok_chao_phraya_tiles.gpkg
   ```
   Tile IDs are assigned after sorting the domains by `hop_distance` and connected component, so they match when the global grid is built from the same DeltaDTM v1.1 tiles, OSM land polygons and `tile_generation` settings.
   If your grid differs, take the tiles covering the case polygon plus their lower-hop closure (`_hop_distance_closure` in `preparation/select_country_tiles.py`; there is no command for it).
3. Materialize the config and preprocess:
   ```
   python snakemake_workflow/scripts/materialize_configs.py bangkok_chao_phraya
   export GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml
   snakemake preprocess --cores 3 --resources mem_mb=8000
   ```
4. Simulate hop 0, then hop 1, with the two target commands under [Hop ordering](#hop-ordering) (or rerun `snakemake simulate --cores 3` until it completes).
   `python check_simulation_progress.py --config $GFM_CONFIG_PATH` then reports all 405 waterdepth files.
5. Postprocess, case-polygon exposure and plots:
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

No step writes the case polygon `bangkok_chao_phraya_domain.gpkg` (layer `bangkok_tile`, one polygon in EPSG:4326 spanning lon 99.79-101.23, lat 13.03-14.77); obtain it from the project maintainers or supply your own single-polygon layer.

On Windows, `run_bangkok_case_study.bat` runs steps 4-5 in one go, stopping at the first failure:

```
set "GFM_ENV=<env>"
run_bangkok_case_study.bat
```
```
set "GFM_ENV=C:\Users\me\miniforge3\envs\gfm"
run_bangkok_case_study.bat
```

`GFM_ENV` is the env directory holding `python.exe` and `Scripts\snakemake.exe`.
The batch file simulates with `run_simulation_batch.py` and expects, besides the materialized config, `<data_root>/bangkok_chao_phraya/resolved_config.yml` (the fully path-expanded config) and `dispatch/wave0_triples.csv` / `wave1_triples.csv` (one `tile_id,return_period,waterlevel_name` per line, no header) there; set `STUDY_ROOT` to use another study directory.
No command in the repo writes these three files; without them, use steps 4-5 above.

## 5. Analysis

Exposure and adaptation analysis and figures, after `postprocess`; the steps run are the `analysis.*` switches in the config:

```
python analysis/run_analysis.py [--config <config.yml>] [--expdir <dir>] [--fig-dir <dir>] \
    [--only-exposure | --only-plots] [--skip-exposure] [--skip-flood-totals] [--skip-world-maps] [--fail-fast]
```

## 6. Validation

Soft-scored HR/FAR/CSI against the national hazard maps of Wales, Scotland, England (GBR) and New Brunswick (CAN), after `postprocess`.
It needs model output over the benchmark regions, which no subset scenario covers, so it runs on the global grid's outputs:

```
python validation/run_validation.py [--config <config.yml>] [--countries GBR CAN] [--skip-plots] [--fail-fast]
```

Standalone scripts, all taking `--config <config.yml>`:

| Script | Does |
|---|---|
| `validation/fetch_england_flood_zones.py` | download the EA Flood Zones 2/3 benchmark (modelled coastal sources) |
| `validation/build_sf_bay_benchmark_mosaic.py` | mosaic the CoSMoS county depth rasters for SF Bay ([data.md](data.md#inputs-outside-the-catalogs)) |
| `validation/validate_sf_bay.py` | score SF Bay depth against CoSMoS (RP100, no SLR) |
| `validation/plot_sf_bay.py` | plot the SF Bay agreement and depth-difference maps |
| `validation/rescore_england_fz3.py` | score England Flood Zone 3 in bounded memory |
| `validation/validate_england_fz2_band.py` | score England Flood Zone 2 against the model's RP100-RP1000 band |
| `validation/compare_against_published.py` | tabulate our metrics next to published studies |
| `validation/build_overview_table.py` | build the overview tables from the CSVs above |

## 7. Global deltas

RP100 flood maps for the global deltas under ssp126 and ssp245, from the delta polygons of `paths.deltas_shapefile` ([data.md](data.md#inputs-outside-the-catalogs)); runs boundary conditions, preprocess, wave-by-wave simulate and postprocess per scenario:

```
python preparation/build_delta_tile_subset.py
python run_delta_floodmaps.py [--scenarios ssp126 ssp245] [--stages ...] [--cores 4] [--mem-mb <MB>] [--skip-boundary-conditions]
```

The delta results are flood hazard only: no protection standards are applied. From the simulated tiles, per delta, scenario and SLR level (all RP100):

```
python analysis/compute_delta_flood_extent.py [--scenarios ssp126 ssp245] [--delta-ids <id> ...] [--out <csv>]
python analysis/add_slr_year_column.py [--csv <csv>]
python analysis/plot_delta_affected_fraction_timeseries.py [--csv <csv>] [--out <png>]
python analysis/plot_delta_flood_map.py [--csv <csv>] [--out-dir <dir>]
```

| Script | Writes (default under `<data_root>/deltas_floodmaps/`) |
|---|---|
| `compute_delta_flood_extent.py` | `delta_flood_extent.csv`: `flooded_area_km2` (merged depth > 0 inside the delta polygon) and `pct_flooded_of_modeled_area` (of the tile-covered part of the polygon, `delta_area_km2_modeled`) |
| `add_slr_year_column.py` | adds `year_slr_reached_median` to the CSV in place (NaN beyond 2150) |
| `plot_delta_affected_fraction_timeseries.py` | `figures/delta_affected_fraction_timeseries.png`: % flooded over time on the median AR6 SLR trajectory |
| `plot_delta_flood_map.py` | `figures/delta_flood_pct_<ssp>.png`: world maps of % flooded per SLR level |
