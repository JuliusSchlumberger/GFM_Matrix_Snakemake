# SFINCS validation of the eikonal flood model

## Purpose

The GFM production pipeline estimates coastal inundation with an eikonal-equation solver: a friction-cost attenuation model that propagates storm-tide water levels inland from the coast, trading physical completeness for global-scale runtime. This pipeline validates it against SFINCS, a genuinely hydrodynamic (shallow-water) model, on a representative sample of tiles from the production tile grid.

Two models are compared against SFINCS on every tile:
- **HC-bathtub** (hydraulically-connected bathtub): `max(boundary water level) - elevation`, no friction or propagation, restricted to cells with an ocean-connected path (`flood_agreement.prune_to_ocean_connected`) so interior basins with no real hydraulic path to the sea don't count as flooded.
- **EA-bathtub** (Eikonal-attenuated bathtub): the production eikonal solver directly on SFINCS's own subgrid DEM/roughness to isolate physics/numerics differences from any grid/projection/elevation-source difference.

## Tile selection

`select_validation_tiles.py` draws `--n-tiles` (default 500) tiles from the production tile grid, restricted to `hop_distance == 0` (a real ocean edge). Eligibility (shared with `select_test_tiles.py`): existing `mask.tif` under a cell-count cap, ≤1% river fraction, a real COAST-HG hydrograph station within search radius. From the eligible pool: `--min-ocean-frac` (default 0.05) floor, `--max-abs-lat-deg` (default 55) latitude band, antimeridian-straddling tiles excluded (their geometry breaks `hydromt_sfincs`'s `water_level.create()` with a GEOS topology exception), spatially stratified across a coarse global grid for continental spread. No area-based pre-filter.

Also writes `tile_locations_map.png`, `tile_selection_histograms.png`, `tile_selection_statistics.xlsx`.

## Pipeline architecture

Two conda environments, switched by invoking each stage's interpreter directly by full binary path (shell `conda activate` inside a non-interactive nested `bash` silently falls back to the base environment instead of raising):
- `gfm` / `gfm_python_preprocessing`: `src/rasters.py`, `src/flood_model.py` (older hydromt pin) - DEM/mask regeneration, bathtub/eikonal.
- `hydromt-sfincs-dev`: SFINCS model build/run/postprocess (newer hydromt pin).

Every per-tile stage is idempotent, gated on its own expected output file. The batch is dispatched as N independent Slurm scripts (`generate_validation_batch_jobs.py`), each looping sequentially through its own tile-ID slice and calling `run_one_tile.sh` per tile; no dependency between the N scripts.

### `run_one_tile.sh` stages

1. Copy `tile_geometry.gpkg`, `model_bbox.json`, `dem.tif`, `mask.tif`, `friction.tif`, `boundaries_RP100_SLR_0.gpkg` from `model_outputs/<tile>/inputs/`.
2. `regenerate_dem_mask.py` (gfm env, non-fatal) - rebuilds `dem.tif`/`mask.tif` with the current `extract_dem`/`extract_dem_mask`, rather than trusting the copied version.
3. `build_elevation.py`, `build_roughness.py`, `build_boundary_forcing.py`, `build_sfincs_tile.py` (hydromt-sfincs-dev env, fatal on failure).
4. `run_eikonal_on_sfincs_subgrid.py --models bathtub eikonal` (gfm env, non-fatal).
5. SFINCS itself: `apptainer exec ... sfincs`, staged to local scratch.
6. `run_sfincs_tile.py --skip-run` (hydromt-sfincs-dev env) - postprocess only (`hmax.tif`/`flood_extent.tif`), if `sfincs_map.nc` exists.
7. `postprocess_tile_summary.py` (hydromt-sfincs-dev env) - always runs, writes `summary_{model}.json`.

## Building each tile's SFINCS model

### Elevation

SFINCS needs a genuine offshore bathymetric surface (the eikonal model's own DEM zero-fills every non-land cell). `build_elevation.py` combines DeltaDTM land elevation with GEBCO bathymetry (MDT-corrected to DeltaDTM's vertical reference) on ocean cells, clipped to roughly -10m to 0m (deeper water never reached by the storm-tide forcing; GEBCO's coarse native resolution produces implausible above-sea-level spikes near the DeltaDTM/GEBCO transition otherwise). Lake/river cells get elevation re-derived from nearby land via nearest-neighbour interpolation plus a local spatial minimum filter, instead of DeltaDTM's flat 0m fill (trivially easy to flood with no real channel resistance). Land elevation is floored 5m above `hydromt_sfincs`'s own hardcoded -20m subgrid volume-table clamp (a genuinely below -18m land cell otherwise produces a spurious flood from the mismatch between the model's clamped dry-state reference and the true elevation used for reported depth).

Coverage gaps in the DeltaDTM validity mask (a tile bounding box extending beyond real mask coverage) default to land - correct for a genuinely data-sparse inland gap, wrong for a gap that's simply offshore, since SFINCS only places its boundary on ocean cells reaching the domain's outer edge. A gap cell is reclassified from land to ocean when GEBCO independently reads negative there and it's connected, through other such cells, to an already-confirmed ocean cell in the same tile - guarding genuinely isolated below-sea-level inland basins from misclassification.

### Roughness

`build_roughness.py` decodes the eikonal model's friction raster (Manning's n / 100, int16-scaled) back into a genuine Manning's n GeoTIFF, checked against a physical plausibility envelope.

### Boundary forcing

`build_boundary_forcing.py` reconstructs a full hydrograph (SFINCS's transient solver needs one, not a single peak value): each COAST-RP boundary point is matched to its nearest COAST-HG station, and the offset between the boundary point's MDT-corrected peak and the matched hydrograph's raw peak is added across the whole series. Only the nearest `k` (default 20) of the eikonal model's own boundary points are used - its ~100km search radius is appropriate for cost-distance seeding, not for blending distant, decorrelated stations into a small SFINCS domain.

### Model assembly

`build_sfincs_tile.py` builds a 120m coarse grid with a 30m subgrid table. Elevation and roughness are pre-reprojected with nearest-neighbour resampling before being handed to `hydromt_sfincs` (which otherwise silently forces bilinear resampling regardless of the method requested). The forcing hydrograph is truncated from ~149h to a 70h window bracketing the storm peak. The boundary-inclusion buffer radius is computed from the real distance between a tile's matched stations and its grid, rather than a fixed guess (a fixed buffer can exclude a tile's only station and crash model construction). Initial water levels (`zsini`) are IDW-interpolated from matched stations onto ocean cells only, every other cell left dry via `hydromt_sfincs`'s bed-level-fallback sentinel.

## Running and comparing the three models

All three run on the same fine-resolution UTM subgrid. Agreement is quantified per tile as three area counts (`postprocess_tile_summary.py`, at `WET_THRESHOLD_M=0.10m`, `flood_agreement.py`): matched (both wet), model-only, SFINCS-only. Ratios (HT/FAR/CSI/bias) are only ever computed on counts already summed across tiles (`flood_agreement.metrics_from_counts`) - never as a per-tile ratio averaged after, which would let a small, lightly-flooded tile dominate the average as much as a large, heavily-flooded one.

Cell-level depth agreement (at every mutually-wet cell) is pooled the same way: `r`, `bias_m`, `rmse_m` from summed sufficient statistics; `median_error_m`, `pct_within_0.2m` from a pooled 2D histogram (`flood_agreement.depth_error_metrics_from_pooled`) - median error and the within-band percentage are robust to the long right tail of a few very deep cells that bias/RMSE are sensitive to.

## Aggregation and reporting

- `compute_metrics_overview_table.py` (`build_and_write_table`) - the canonical pooled table: HT/FAR/CSI/bias and the five depth-error metrics, one row per model, written to `metrics_overview_table.csv` (+ `metrics_overview_per_tile.csv`). Run automatically at the end of `plot_validation_results.py`'s `main()`.
- `plot_validation_results.py` - the main figure set: extent scatter, CSI histogram (per-tile unweighted + area-weighted pooled), depth scatter, global agreement map, misalignment-vs-tile-size, pooled depth correlation, depth-bin alignment heatmap.
- `plot_worst_tiles_panel.py` - 3x3 panels of the worst-CSI tiles (union area ≥ `--min-union-km2`), split by which model over-predicts.
- `plot_eikonal_disagreement_extremes.py` - N-worst under/over-prediction grids.
- `plot_sfincs_tile_diagnostics.py` - per-tile build sanity-check plots (mask, boundary/forcing, biggest local disagreement).
- `aggregate_tile_summaries.py` - globs every `summary_{model}.json` into one master CSV.

## Batch dispatch

`setup_batch_inputs.py` bulk-copies `model_outputs/` inputs into the batch dir up front; `generate_validation_batch_jobs.py` generates the N sbatch scripts calling `run_one_tile.sh`. Local/manual alternatives (one machine, no Slurm): `run_sfincs_tiles.py` (full chain) and `run_models_sequential.py` (bathtub/eikonal only).

## Structural limitations (not bugs)

- **Flat, low-relief floodplains**: eikonal's friction-cost model charges a toll per cell traversed regardless of terrain flatness, so it caps out a few km from the coast on a wide, flat, low-gradient plain, while SFINCS's real hydrodynamics fill the same connected low-lying basin over the storm duration with near-zero head loss. Gaps of 10km+ between the two models' flood extent on a single tile are possible this way; a single global `friction_scale_factor` cannot fix this without over-flooding tiles where friction is doing real, physically meaningful work elsewhere.
