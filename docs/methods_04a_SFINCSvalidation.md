# SFINCS calibration and validation of the eikonal flood model

## Purpose

The GFM production pipeline estimates coastal inundation with an eikonal-equation solver: a friction-cost attenuation model that propagates storm-tide water levels inland from the coast, trading physical completeness for global-scale runtime. This pipeline compares it against SFINCS, a hydrodynamic (shallow-water) model, on 1,500 tiles of the production tile grid. It does two things at once:

- **calibrates** each flood-model variant's parameter (friction factor, uniform rate, duration exponent) on the calibration tiles, picking the value with the best pooled agreement with SFINCS;
- **validates** every variant at its selected value on the other tiles, which took no part in the fit.

The variants (bathtub, connected bathtub, straight-line attenuation, eikonal with a uniform or a land-cover rate, each eikonal with or without the duration correction) are listed under [Flood-model variants](#flood-model-variants). All run directly on SFINCS's own subgrid DEM/roughness to isolate physics/numerics differences from any grid/projection/elevation-source difference; SFINCS itself is the reference.

The run lives in `{root}/calibration_roughness_SFINCS` (overlap-corrected results in `{root}/calibration_roughness_SFINCS_dedup`, see Comparison); all scripts are in `sfincs_tiles/`.

## SFINCS boundary: the station boundary line

Earlier SFINCS set-ups forced the model along the open-sea tile edge. Because offshore bathymetry is floored at -10 m (`build_elevation.py`'s `MIN_BATHYMETRY_M`), the tide travelling from that edge to the coast over a shallow, flat sea was amplified (quarter-wavelength resonance), and SFINCS water levels at the coast rose well above the COAST-RP forcing. The final set-up instead forces SFINCS along a **boundary line ~1 km offshore that follows the coast**, built from the tile's own COAST-RP stations (`build_station_boundary_lines.py`); the sea beyond the line is not modelled.

**Stations.** The line uses exactly the stations the production eikonal model uses for the tile: its `inputs/boundaries_RP100_SLR_0.gpkg` (production's search box - the tile bbox + `boundary_conditions.station_search_buffer_deg` = 1°, at least `station_search_min_size_deg` = 2° wide - minus stations in separate water bodies). Before building, stations not within 2 km of open ocean are dropped, as are stations in a bay whose mouth closes when the ocean is shrunk by 2 km (a cut-off lagoon). The coastline is traced from the global DeltaDTM mask over the same search box.

**Construction** (rules in the module docstring):
1. Each station's node is moved ≥ 1 km out to open sea (at most 5 km).
2. Candidate links between each station and its 5 nearest neighbours either follow the coastline shifted seaward or run straight, at increasing seaward offsets (0.5-4 km), keeping ≥ 700 m from land. The coastline shape is used unless it is rugged (> 1.5× the straight distance), so bay mouths and rugged coasts get straight links.
3. Links are accepted cheapest first, at most two per station and no loops, giving one or more lines per tile.
4. Line ends inside the tile are extended to the tile edge (along the coast where not rugged), lines are untangled and shortcut where the coast-following path is much longer than a land-clearing straight link, and open-sea stretches of the tile edge that no line closes off become boundary themselves.
5. The lines split the tile: faces that are mostly ocean become **inactive** (the sea side), the rest stays active.

**Suitability.** A tile's line is *suitable* when the split has no conflict - at most 2 % of the tile's land falls in inactive faces, at most 50 % of its ocean stays active (the sea is closed off) and no line ends inside the tile - and at least 2 stations on the line have a hydrograph within 0.5°. Conflicting tiles are rebuilt "straight-first" once; if still conflicting they are excluded. Every tile gets an inspection figure in `boundary_lines/figures/` (suitable) or `figures_unsuitable/`.

## Tile selection

Selection runs on the HPC as stage 1 of the pipeline (`select_calibration_tiles.py`), as a funnel:

| step | criterion | tiles |
|---|---|---|
| pool | production grid tiles with `hop_distance == 0` (an own ocean boundary), not crossing the antimeridian | 3,761 of 4,504 |
| screen | production inputs present in `model_outputs/` (`mask.tif`, `tile_geometry.gpkg`) and ≥ 1 COAST-RP station | 3,632 |
| boundary line | line built and suitable (above); 158 tiles had < 2 usable stations, 40 failed to build | 2,090 |
| pre-flight | the SFINCS grid + mask builds from the line with ≥ 1 boundary cell (`preflight_sfincs_mask.py`) | 2,088 |
| sample | spatially stratified: round-robin over shuffled 15° lon/lat bins, seed 0 | **1,500** |

No size, river-fraction, ocean-fraction or latitude filter is applied. The funnel is recorded per candidate in `candidates/candidate_funnel.csv`; the selection in `tile_ids.txt`, `tile_selection_metadata.csv` and `tile_selection.gpkg`; `figures/tile_selection_map.png` maps every candidate by funnel stage and `figures/tile_selection_histograms.png` compares the selected, eligible and screened populations.

## Building each tile's SFINCS model

Inputs are copied (not linked) from `model_outputs/<tile>/inputs/`; `regenerate_dem_mask.py` then rebuilds `dem.tif`/`mask.tif` with the current extraction code.

### Elevation

SFINCS needs a genuine offshore bathymetric surface (the eikonal model's own DEM zero-fills every non-land cell). `build_elevation.py` combines DeltaDTM land elevation with GEBCO bathymetry (MDT-corrected to DeltaDTM's vertical reference) on ocean cells, clipped to roughly -10 m to 0 m (deeper water is never reached by the storm-tide forcing; GEBCO's coarse native resolution produces implausible above-sea-level spikes near the DeltaDTM/GEBCO transition otherwise). Lake/river cells get elevation re-derived from nearby land via nearest-neighbour interpolation plus a local spatial minimum filter, instead of DeltaDTM's flat 0 m fill (trivially easy to flood with no channel resistance). Land elevation is floored at -15 m, 5 m above `hydromt_sfincs`'s hardcoded -20 m subgrid volume-table clamp (a land cell below -20 m otherwise produces a spurious flood from the mismatch between the model's clamped dry-state reference and the true elevation used for reported depth).

Coverage gaps in the DeltaDTM validity mask default to land - correct for a data-sparse inland gap, wrong for one that is simply offshore. A gap cell is reclassified as ocean when GEBCO independently reads negative there and it is connected, through other such cells, to confirmed ocean in the same tile - guarding isolated below-sea-level inland basins from misclassification.

### Roughness

`build_roughness.py` decodes the eikonal model's friction raster (Manning's n / 100, int16-scaled) back into a Manning's n GeoTIFF, checked against a physical plausibility envelope.

### Boundary forcing

`build_line_forcing.py` places one forcing point per station used on the line, at the station's offshore node. Each point gets the RP100 hydrograph of the nearest hydrograph station (≤ 0.5°), shifted so that its peak equals the station's MDT-corrected RP100 level (`offset = level - hydrograph max`, added to the whole series). Levels and hydrographs follow `boundary_conditions.storm_tide_source`: the storm-tide product's levels and per-RP hydrographs (`gfm_product`), or COAST-RP's levels with COAST-HG's average-tide hydrograph (`coast_rp`). The product's hydrographs are cut to COAST-HG's ~149 h axis with the peak at 74.5 h. SFINCS interpolates between neighbouring points along the boundary. The hydrograph is truncated to the 40-110 h window around the storm peak.

### Model assembly

`build_sfincs_tile.py` builds a 120 m grid with a 30 m subgrid table:
- **Grid:** a rotated UTM grid - hydromt's minimum rotated rectangle around the tile polygon. The rotation (typically a fraction of a degree) is the meridian convergence of the lon/lat tile in UTM; the rectangle's corners outside the tile have no data and are inactive.
- **Mask:** active = the tile minus the line's inactive faces; waterlevel boundary = the active cells touching a 1.5-cell buffer around the line (`all_touched`, so a diagonal line gives a continuous boundary). Invalid line geometries are repaired (`make_valid`) first.
- **Elevation/roughness** are pre-reprojected with nearest-neighbour resampling before being handed to `hydromt_sfincs` (which otherwise forces bilinear resampling regardless of the method requested).
- **Forcing points** are passed in the model's UTM CRS: `water_level.create()` selects them with the polygonized boundary cells, and reprojecting those to lon/lat produced invalid geometry for tiles near ±180°. The selection buffer is derived from the stations' distance to the grid (+ 25 km).
- **Initial water levels** (`zsini`) are IDW-interpolated from the forcing points onto ocean cells only; every other cell starts dry.

SFINCS v2.4.0 then runs in its container on node-local scratch, and `run_sfincs_tile.py --skip-run` downscales the peak water level onto the 30 m subgrid (`hmax_subgrid.tif`, land only).

## Eikonal and bathtub on the SFINCS subgrid

`run_calibration_models.py` runs the variants on SFINCS's own 30 m subgrid (`dep_subgrid.tif`, `manning_subgrid.tif`), with the native mask reprojected onto it:
- **Coastline and seeds as in production:** the coastline is `flood_model.coastline_mask`'s rule (ocean within 1 cell of land/river, in an ocean body reaching the domain edge - on the rotated grid also the no-data corners outside the tile), and each coastline cell gets the inverse-distance average (`knn`, `idw_power` of `simulation.flooding`) of its nearest **ocean-connected** stations (`flood_model.idw_seed_values_at`, the function production uses). Every variant starts from these seeds.
- **Solver settings from the main config** (`simulation.flooding`): `max_rounds`, `waterlevel_epsilon_m`, obstacle coupling with `max_outer_iterations`. Land-cover friction = subgrid Manning's n / 100 × friction_scale_factor × the subgrid cell size; uniform friction = rate × cell size. Outputs: `<run key>_waterdepth_RP100_SLR_0.tif` (`flood_variants.run_key`, e.g. `eikonal_landcover_fsf0.25`), run settings and solver diagnostics in `model_runs.json`.
- **Duration correction:** from the hydrographs SFINCS was forced with (`corrected_hydrographs.csv` at `matched_boundary_points.gpkg`, its simulated window).

## Comparison

`compute_calibration_result.py` compares every model with SFINCS on the 30 m subgrid and writes one file per tile, `outputs/calibration_result.json`: domain areas, SFINCS flooded area and depth stats, and for the bathtub and every fsf the flooded area, depth stats, agreement with SFINCS and (eikonal) the solver diagnostics.

- **Scoring area:** land in the native mask that is **active in SFINCS**. Land the boundary line cut off as inactive is left out for every model; the model rasters themselves are not masked.
- **Extent:** a cell is wet above `WET_THRESHOLD_M` = 0.10 m; per tile, three areas - matched (both wet), model-only, SFINCS-only. HT/FAR/CSI/bias are only ever computed from areas summed across tiles (`flood_agreement.metrics_from_counts`), never as per-tile ratios averaged afterwards, which would let a small, lightly flooded tile weigh as much as a large, heavily flooded one.
- **Depth:** at every mutually wet cell, pooled the same way: `r`, `bias_m`, `rmse_m` from summed sufficient statistics; `median_error_m` and `pct_within_0.2m` (share of cells with |error| ≤ 0.2 m; the key is built as `pct_within_{band}m` with the default band 0.2 m) from a pooled 2D histogram (`flood_agreement.depth_error_metrics_from_pooled`) - robust to the long tail of a few very deep cells that bias/RMSE are sensitive to.

**Overlap correction.** Neighbouring production tiles overlap (the deliberate boundary-condition buffer), so pooling the per-tile areas counts any (mis)agreement in an overlap zone once per overlapping tile; for the selected tiles the summed footprints exceed their union by a factor 1.237. `compute_calibration_result_dedup.py` re-derives every `calibration_result.json` as production merges tiles: SFINCS and every flood-model run first take, per cell, the **maximum over all calibration tiles whose model grid covers it** (production's per-cell maximum merge, `src/merge.py`; the neighbours' rasters are reprojected onto the tile's grid), and the scoring area is then intersected with the tile's **Voronoi core** - the cells closer to its own centre than to any other selected tile's centre (centroids from `tile_selection.gpkg`, in EPSG:6933; all selected tiles take part, so a tile without a result still claims its core). Every place thus counts once, with production's merged value; the neighbours may belong to either group. The results go to a separate base directory, `calibration_roughness_SFINCS_dedup`, in the same `<tile>/outputs/calibration_result.json` layout (with `core_fraction_of_active_land` added to the domain stats); the raw directory is not touched.

## Flood-model variants

Seven variants (`src/flood_variants.py`) are run on the same seeds, coastline, solver settings
and subgrid, and scored as written:

| Variant | Propagation | Parameter |
|---|---|---|
| `bathtub` | IDW of the seed levels against the DEM | - |
| `bathtub_connected` | the same, kept where connected to the coastline | - |
| `straight_line` | max over seeds of level - rate x straight-line distance, connected | uniform rate (m/km) |
| `eikonal_uniform` | eikonal with obstacle coupling, uniform rate | uniform rate (m/km) |
| `eikonal_landcover` | eikonal with obstacle coupling, land-cover rate | friction_scale_factor (per m) |
| `eikonal_*_duration` | the above plus the duration correction, from the SFINCS run's own boundary hydrographs | the base variant's parameter, alpha |

The tiles are split by a hash of the tile id into a **calibration** group (30 %) and a
**validation** group (70 %), `calibration_split.py`; a tile never changes group when tiles are
added, and `calibration_split_balance.csv` compares the groups by latitude band, size, ocean and
river share and SFINCS flood area. `run_calibration_models.py` runs in two phases, each on every
tile of both groups:

1. `sweep` - the bathtubs, `straight_line` and `eikonal_uniform` over the uniform rates and
   `eikonal_landcover` over the friction factors (`calibration_paths`).
2. `duration` - both duration variants over the alphas, each at the parameter its base variant
   was selected with.

After each phase a post job (`generate_calibration_jobs.py stage2`) runs `compute_variant_metrics.py`,
which pools every run per group and writes `variant_selection.json`: per variant the
calibration-group run with the best pooled CSI (or |bias - 1|), its validation-group score, and
the best variant. Only the calibration group picks the runs; the validation group has the same
per-run table, so the selected run's validation score and the shape of each sweep are out of
sample.

## Analysis

`run_calibration_sweep_analysis.py` is the single entry point; it reads only the per-tile result files (`calibration_results.py`), so it can be rerun at any time on whatever tiles have finished:

1. `compute_variant_metrics.py` - every run vs SFINCS per tile and pooled per group (`variant_runs_per_tile.csv`, `variant_runs_pooled.csv`, extent area- and population-weighted) and `variant_selection.json`; `variant_runs_depth_accuracy.csv` reports each run's depth agreement, which does not enter the selection. It is pooled from exact 1 cm histograms of eikonal minus SFINCS depth (`calibration_depth_accuracy.npz` per tile, area and WorldPop population weights, per SFINCS depth class): the share within 10/20/30/50 cm, bias, MAE, RMSE and quantiles of the absolute error, on three sets of cells - both models wet, every SFINCS-wet cell (a model-dry cell at its own sub-threshold depth) and every cell either model floods. `plot_depth_accuracy.py` draws these, with CSI, against each swept variant's parameter (`figures/depth_accuracy_by_run.png`).
2. The best variant's selected run is plotted (`--select-by bias`: |bias - 1| closest to 0).
3. `plot_worst_tiles_panel.py` - 3×3 panels of the lowest-CSI tiles of that run (union area ≥ 50 km²), split by which model over-predicts, mapped over SFINCS-active land.
4. `plot_validation_results.py` - the run against the selected connected bathtub: extent scatter, CSI histogram (per-tile and pooled), depth scatter, global agreement map, misalignment vs tile land area, pooled depth correlation, depth-bin alignment heatmap, and `metrics_overview_table_<run>.csv` (every selected run vs SFINCS, extent + depth); `--group calibration|validation` restricts to one group.
5. **Forcing checks** for every tile (`plot_sfincs_tile_diagnostics.py`, into `figures/forcing_checks/`, named `{overprediction,underprediction,agreement}_<tile>_...` from the run's point of view): the boundary cells, forcing points and hydrographs (`_boundary_forcing.png`), and the COAST-RP forcing comparison (`_forcing_comparison.png`). The latter maps the station levels and the eikonal coastline seeds, and plots - for each station inside the tile with ocean within 2 km - the station level, the eikonal seed and SFINCS's peak water level, both read at the same point: the coastline cell nearest the station, if within 10 km. By default only the tiles of the worst-tile panels (`worst_tiles_selection_<run>.csv`) get these; `--forcing-checks-all` does every scored tile.

The analysis is run twice, once per base directory, and each run writes its CSVs, tables and `figures/` into its own directory. On `calibration_roughness_SFINCS` all metrics count the overlap zones once per tile; on `calibration_roughness_SFINCS_dedup` (with `--raster-base-dir-name calibration_roughness_SFINCS`) the per-tile and pooled metrics, the variant selection (which the duration phase and the local comparisons use), the validation figures and the overview table are overlap-merged, while the map-based figures (worst-tile panels, forcing checks) read their rasters and inputs from the raw directory, which is the only one that holds them.

## HPC pipeline

Two stages, chained on Slurm (`generate_calibration_jobs.py`), all jobs on the partition set by `hpc.calibration_sbatch.partition`:

- **Stage 1** - 10 jobs, each one interleaved shard of the candidate pool: screen → boundary lines (4 worker processes) → SFINCS mask pre-flight.
- **Bridge** - one job that starts when every stage-1 job has ended: collects the line results, runs the tile selection (+ maps) and submits stage 2.
- **Stage 2** - 20 jobs over the 1,500 tiles, balanced by an estimated runtime from the pre-flight's active cell count (largest tiles first in every job). Per tile, `run_final_tile.sh`: copy inputs → DEM/mask, elevation, roughness → line forcing → SFINCS model build → flood-model runs of one phase (`CAL_PHASE`) → SFINCS run → postprocessing → `calibration_result.json`.

Every step is skipped when its output exists, so a resubmitted tile resumes where it stopped. A tile records its current step in `outputs/calibration_run.step`, its success in `calibration_run.done` and failures in `calibration_run.failed`; `check_calibration_progress.py` reports both stages per job.

`rerun_calibration.sh` redoes part of the run in place after a code change, relying on this skip logic: per selected tile it deletes the done/failed/step markers (which re-queues the tile) plus the artefacts to be redone, then write and submit the stage-2 jobs via `generate_calibration_jobs.py stage2`. `--part models` (model-side changes) runs one phase (`PHASE`); existing runs are kept unless `RESET_RUNS=1` deletes every `*_waterdepth_RP100_SLR_0*.tif`, `model_runs.json` and `calibration_result.json` first; a post job (after all stage-2 jobs) then runs `compute_calibration_result_dedup.py` and the analysis on both base directories, which writes the selection the next phase needs. `--part sfincs` (forcing/mask changes) deletes `sfincs.inp`, so line forcing → SFINCS model build → SFINCS run → postprocessing → `calibration_result.json` are redone, followed by the sweep phase and its post job. Two conda environments are called by full interpreter path: `gfm` (flood models, lines, selection) and `hydromt-sfincs-dev` (SFINCS build/postprocessing, pre-flight, comparison); their specs are `environment.yml` and `environment_sfincs.yml`.

## Structural limitations (not bugs)

- **Flat, low-relief floodplains:** eikonal's friction-cost model charges a toll per cell traversed regardless of terrain flatness, so it caps out a few km from the coast on a wide, flat, low-gradient plain, while SFINCS's hydrodynamics fill the same connected low-lying basin over the storm duration with near-zero head loss. Gaps of 10 km+ between the two models' flood extent on a single tile are possible this way; a single global fsf cannot fix this without over-flooding tiles where friction does real work.
- **Selection by line suitability:** only tiles with a suitable boundary line are eligible (2,090 of 3,632 screened). Tiles whose coast the automatic line cannot close off - complex archipelagos, long enclosed bays, coasts with too few stations - are therefore under-represented in the sample.
- **Sea beyond the line is not modelled:** land and water on the inactive side of the line are outside the comparison; by the suitability rule this is at most 2 % of a tile's land.
- **Eikonal convergence at low friction:** at fsf 3 a quarter of the solves stop at the 40-round cap before converging (at fsf ≥ 12, ≥ 95 % converge), so the low-friction end of the sweep is less reliable than the region around the optimum.
