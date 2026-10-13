# SFINCS calibration and validation of the eikonal flood model

## Purpose

The GFM production pipeline estimates coastal inundation with an eikonal-equation solver: a friction-cost attenuation model that propagates storm-tide water levels inland from the coast, trading physical completeness for global-scale runtime. This pipeline compares it against SFINCS, a hydrodynamic (shallow-water) model, on 1,000 tiles of the production tile grid. It does two things at once:

- **calibrates** the eikonal's global `friction_scale_factor` (fsf): the eikonal is run at fsf = 3, 6, …, 30 and the value with the best pooled agreement with SFINCS is selected;
- **validates** the model at that value, against SFINCS and against a plain bathtub.

Three models are compared against SFINCS on every tile:
- **HC-bathtub** (hydraulically-connected bathtub): `max(station water level) - elevation`, no friction or propagation, restricted to flooding with an ocean-connected path (`flood_agreement.prune_to_ocean_connected`) so interior basins with no hydraulic path to the sea don't count.
- **EA-bathtub** (eikonal-attenuated bathtub): the production eikonal solver, run directly on SFINCS's own subgrid DEM/roughness to isolate physics/numerics differences from any grid/projection/elevation-source difference - once per fsf.
- **SFINCS** itself, the reference.

The run lives in `{root}/calibration_roughness_SFINCS`; all scripts are in `sfincs_tiles/`.

## SFINCS boundary: the station boundary line

Earlier SFINCS set-ups forced the model along the open-sea tile edge. Because offshore bathymetry is floored at -10 m (`build_elevation.py`'s `MIN_BATHYMETRY_M`), the tide travelling from that edge to the coast over a shallow, flat sea was amplified (quarter-wavelength resonance), and SFINCS water levels at the coast rose well above the COAST-RP forcing. The final set-up instead forces SFINCS along a **boundary line ~1 km offshore that follows the coast**, built from the tile's own COAST-RP stations (`build_station_boundary_lines.py`); the sea beyond the line is not modelled.

**Stations.** The line uses exactly the stations the production eikonal model uses for the tile: its `inputs/boundaries_RP100_SLR_0.gpkg` (production's search box - the tile bbox + `boundary_conditions.station_search_buffer_deg` = 1°, at least `station_search_min_size_deg` = 2° wide - minus stations in separate water bodies). Before building, stations not within 2 km of open ocean are dropped, as are stations in a bay whose mouth closes when the ocean is shrunk by 2 km (a cut-off lagoon). The coastline is traced from the global DeltaDTM mask over the same search box.

**Construction** (rules in the module docstring):
1. Each station's node is moved ≥ 1 km out to open sea (at most 5 km).
2. Candidate links between each station and its 5 nearest neighbours either follow the coastline shifted seaward or run straight, at increasing seaward offsets (0.5-4 km), keeping ≥ 700 m from land. The coastline shape is used unless it is rugged (> 1.5× the straight distance), so bay mouths and rugged coasts get straight links.
3. Links are accepted cheapest first, at most two per station and no loops, giving one or more lines per tile.
4. Line ends inside the tile are extended to the tile edge (along the coast where not rugged), lines are untangled and shortcut where the coast-following path is much longer than a land-clearing straight link, and open-sea stretches of the tile edge that no line closes off become boundary themselves.
5. The lines split the tile: faces that are mostly ocean become **inactive** (the sea side), the rest stays active.

**Suitability.** A tile's line is *suitable* when the split has no conflict - at most 2 % of the tile's land falls in inactive faces, at most 50 % of its ocean stays active (the sea is closed off) and no line ends inside the tile - and at least 2 stations on the line have a COAST-HG hydrograph within 0.5°. Conflicting tiles are rebuilt "straight-first" once; if still conflicting they are excluded. Every tile gets an inspection figure in `boundary_lines/figures/` (suitable) or `figures_unsuitable/`.

## Tile selection

Selection runs on the HPC as stage 1 of the pipeline (`select_calibration_tiles.py`), as a funnel:

| step | criterion | tiles |
|---|---|---|
| pool | production grid tiles with `hop_distance == 0` (an own ocean boundary), not crossing the antimeridian | 3,761 of 4,504 |
| screen | production inputs present in `model_outputs/` (`mask.tif`, `tile_geometry.gpkg`) and ≥ 1 COAST-RP station | 3,632 |
| boundary line | line built and suitable (above); 158 tiles had < 2 usable stations, 40 failed to build | 2,090 |
| pre-flight | the SFINCS grid + mask builds from the line with ≥ 1 boundary cell (`preflight_sfincs_mask.py`) | 2,088 |
| sample | spatially stratified: round-robin over shuffled 15° lon/lat bins, seed 0 | **1,000** |

As in the earlier `sfincs_calibration` study, no size, river-fraction, ocean-fraction or latitude filter is applied. The funnel is recorded per candidate in `candidates/candidate_funnel.csv`; the selection in `tile_ids.txt`, `tile_selection_metadata.csv` and `tile_selection.gpkg`; `figures/tile_selection_map.png` maps every candidate by funnel stage and `figures/tile_selection_histograms.png` compares the selected, eligible and screened populations.

## Building each tile's SFINCS model

Inputs are copied (not linked) from `model_outputs/<tile>/inputs/`; `regenerate_dem_mask.py` then rebuilds `dem.tif`/`mask.tif` with the current extraction code.

### Elevation

SFINCS needs a genuine offshore bathymetric surface (the eikonal model's own DEM zero-fills every non-land cell). `build_elevation.py` combines DeltaDTM land elevation with GEBCO bathymetry (MDT-corrected to DeltaDTM's vertical reference) on ocean cells, clipped to roughly -10 m to 0 m (deeper water is never reached by the storm-tide forcing; GEBCO's coarse native resolution produces implausible above-sea-level spikes near the DeltaDTM/GEBCO transition otherwise). Lake/river cells get elevation re-derived from nearby land via nearest-neighbour interpolation plus a local spatial minimum filter, instead of DeltaDTM's flat 0 m fill (trivially easy to flood with no channel resistance). Land elevation is floored 5 m above `hydromt_sfincs`'s hardcoded -20 m subgrid volume-table clamp (a land cell below -18 m otherwise produces a spurious flood from the mismatch between the model's clamped dry-state reference and the true elevation used for reported depth).

Coverage gaps in the DeltaDTM validity mask default to land - correct for a data-sparse inland gap, wrong for one that is simply offshore. A gap cell is reclassified as ocean when GEBCO independently reads negative there and it is connected, through other such cells, to confirmed ocean in the same tile - guarding isolated below-sea-level inland basins from misclassification.

### Roughness

`build_roughness.py` decodes the eikonal model's friction raster (Manning's n / 100, int16-scaled) back into a Manning's n GeoTIFF, checked against a physical plausibility envelope.

### Boundary forcing

`build_line_forcing.py` places one forcing point per station used on the line, at the station's offshore node. Each point gets the average-tide hydrograph of the nearest COAST-HG station (≤ 0.5°), shifted so that its peak equals the station's MDT-corrected COAST-RP RP100 level (`offset = level - hydrograph max`, added to the whole series). SFINCS interpolates between neighbouring points along the boundary. The hydrograph is truncated from ~149 h to the 40-110 h window around the storm peak.

### Model assembly

`build_sfincs_tile.py` builds a 120 m grid with a 30 m subgrid table:
- **Grid:** a rotated UTM grid - hydromt's minimum rotated rectangle around the tile polygon. The rotation (typically a fraction of a degree) is the meridian convergence of the lon/lat tile in UTM; the rectangle's corners outside the tile have no data and are inactive.
- **Mask:** active = the tile minus the line's inactive faces; waterlevel boundary = the active cells touching a 1.5-cell buffer around the line (`all_touched`, so a diagonal line gives a continuous boundary). Invalid line geometries are repaired (`make_valid`) first.
- **Elevation/roughness** are pre-reprojected with nearest-neighbour resampling before being handed to `hydromt_sfincs` (which otherwise forces bilinear resampling regardless of the method requested).
- **Forcing points** are passed in the model's UTM CRS: `water_level.create()` selects them with the polygonized boundary cells, and reprojecting those to lon/lat produced invalid geometry for tiles near ±180° (fixed 2026-10-13 after 5 tiles failed). The selection buffer is derived from the stations' distance to the grid (+ 25 km).
- **Initial water levels** (`zsini`) are IDW-interpolated from the forcing points onto ocean cells only; every other cell starts dry.

SFINCS v2.4.0 then runs in its container on node-local scratch, and `run_sfincs_tile.py --skip-run` downscales the peak water level onto the 30 m subgrid (`hmax_subgrid.tif`, land only).

## Eikonal and bathtub on the SFINCS subgrid

`run_calibration_models.py` runs both on SFINCS's own 30 m subgrid (`dep_subgrid.tif`, `manning_subgrid.tif`), with the native mask reprojected onto it:
- **Coastline and seeds as in production:** the coastline is `flood_model.coastline_mask`'s rule (ocean within 1 cell of land/river, in an ocean body reaching the domain edge - on the rotated grid also the no-data corners outside the tile), and each coastline cell gets the inverse-distance-squared average of its `knn` = 15 nearest **ocean-connected** stations (`flood_model.idw_seed_values_at`, the function production uses).
- **Solver settings from the main config** (`simulation.flooding`): `max_rounds` 40, `waterlevel_epsilon_m` 0.03, obstacle coupling with `max_outer_iterations` 4. Friction = subgrid Manning's n × fsf, for fsf = 3, 6, …, 30. Outputs: `eikonal_on_subgrid_waterdepth_RP100_SLR_0_fsf<X>.tif`, solver diagnostics per fsf in `eikonal_solver_diagnostics.json`.
- **Bathtub:** the tile's highest station level minus elevation on land.

## Comparison

`compute_calibration_result.py` compares every model with SFINCS on the 30 m subgrid and writes one file per tile, `outputs/calibration_result.json`: domain areas, SFINCS flooded area and depth stats, and for the bathtub and every fsf the flooded area, depth stats, agreement with SFINCS and (eikonal) the solver diagnostics.

- **Scoring area:** land in the native mask that is **active in SFINCS**. Land the boundary line cut off as inactive is left out for every model; the model rasters themselves are not masked.
- **Extent:** a cell is wet above `WET_THRESHOLD_M` = 0.10 m; per tile, three areas - matched (both wet), model-only, SFINCS-only. HT/FAR/CSI/bias are only ever computed from areas summed across tiles (`flood_agreement.metrics_from_counts`), never as per-tile ratios averaged afterwards, which would let a small, lightly flooded tile weigh as much as a large, heavily flooded one.
- **Depth:** at every mutually wet cell, pooled the same way: `r`, `bias_m`, `rmse_m` from summed sufficient statistics; `median_error_m` and `pct_within_0.2m` from a pooled 2D histogram (`flood_agreement.depth_error_metrics_from_pooled`) - robust to the long tail of a few very deep cells that bias/RMSE are sensitive to.

## Analysis

`run_calibration_sweep_analysis.py` is the single entry point; it reads only the per-tile result files (`calibration_results.py`), so it can be rerun at any time on whatever tiles have finished:

1. `compute_friction_sweep_metrics.py` - eikonal vs SFINCS per tile and pooled per fsf (`friction_sweep_per_tile_metrics.csv`, `friction_sweep_pooled_metrics.csv`).
2. The fsf with the best pooled CSI is chosen (`--select-by bias`: |bias - 1| closest to 0).
3. `plot_worst_tiles_panel.py` - 3×3 panels of the lowest-CSI tiles at that fsf (union area ≥ 1 km²), split by which model over-predicts, mapped over SFINCS-active land.
4. `plot_validation_results.py` - extent scatter, CSI histogram (per-tile and pooled), depth scatter, global agreement map, misalignment vs tile land area, pooled depth correlation, depth-bin alignment heatmap, and `metrics_overview_table_fsf<X>.csv` (HC-bathtub and EA-bathtub vs SFINCS, extent + depth).
5. **Forcing checks** for every tile (`plot_sfincs_tile_diagnostics.py`, into `figures/forcing_checks/`, named `{overprediction,underprediction,agreement}_<tile>_...` from the eikonal's point of view): the boundary cells, forcing points and hydrographs (`_boundary_forcing.png`), and the COAST-RP forcing comparison (`_forcing_comparison.png`). The latter maps the station levels and the eikonal coastline seeds, and plots - for each station inside the tile with ocean within 2 km - the station level, the eikonal seed and SFINCS's peak water level, both read at the same point: the coastline cell nearest the station, if within 3 km.

## HPC pipeline

Two stages, chained on Slurm (`generate_calibration_jobs.py`), all jobs on the 4vcpu partition:

- **Stage 1** - 10 jobs, each one interleaved shard of the candidate pool: screen → boundary lines (4 worker processes) → SFINCS mask pre-flight.
- **Bridge** - one job that starts when every stage-1 job has ended: collects the line results, runs the tile selection (+ maps) and submits stage 2.
- **Stage 2** - 20 jobs over the 1,000 tiles, balanced by an estimated runtime from the pre-flight's active cell count (largest tiles first in every job). Per tile, `run_final_tile.sh`: copy inputs → DEM/mask, elevation, roughness → line forcing → SFINCS model build → bathtub + eikonal sweep → SFINCS run → postprocessing → `calibration_result.json`.

Every step is skipped when its output exists, so a resubmitted tile resumes where it stopped. A tile records its current step in `outputs/calibration_run.step`, its success in `calibration_run.done` and failures in `calibration_run.failed`; `check_calibration_progress.py` reports both stages per job. Two conda environments are called by full interpreter path: `gfm` (eikonal, bathtub, lines, selection) and `hydromt-sfincs-dev` (SFINCS build/postprocessing, pre-flight, comparison).

## Changes from the earlier studies

- **`validation_sfincs` v1-v6:** tiles filtered by size, river fraction, ocean fraction (≥ 5 %) and latitude (|lat| ≤ 55°); SFINCS forced along the open-sea tile edge from the k nearest stations; eikonal seeded by planar IDW over all coastline-adjacent ocean cells; the fsf sweep run as a separate step. Superseded.
- **`sfincs_calibration`** (1,126 tiles, hop 0, stratified, no other filters): same open-sea forcing, later resampled to points every 25 km along the boundary; this exposed the tidal amplification described above. **`sfincs_calibration_bline`** re-ran 573 of its tiles with the boundary line and confirmed that the overshoot disappears; the final pipeline adopts it, with stations from the tile's production selection instead of every COAST-RP station within 0.5°, and the production eikonal coastline and station weighting instead of planar IDW.

## Structural limitations (not bugs)

- **Flat, low-relief floodplains:** eikonal's friction-cost model charges a toll per cell traversed regardless of terrain flatness, so it caps out a few km from the coast on a wide, flat, low-gradient plain, while SFINCS's hydrodynamics fill the same connected low-lying basin over the storm duration with near-zero head loss. Gaps of 10 km+ between the two models' flood extent on a single tile are possible this way; a single global fsf cannot fix this without over-flooding tiles where friction does real work.
- **Selection by line suitability:** only tiles with a suitable boundary line are eligible (2,090 of 3,632 screened). Tiles whose coast the automatic line cannot close off - complex archipelagos, long enclosed bays, coasts with too few stations - are therefore under-represented in the sample.
- **Sea beyond the line is not modelled:** land and water on the inactive side of the line are outside the comparison; by the suitability rule this is at most 2 % of a tile's land.
- **Eikonal convergence at low friction:** at fsf 3 a quarter of the solves stop at the 40-round cap before converging (at fsf ≥ 12, ≥ 95 % converge), so the low-friction end of the sweep is less reliable than the region around the optimum.
