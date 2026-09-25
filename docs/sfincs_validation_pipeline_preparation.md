# SFINCS validation pipeline: tile preparation

Describes the per-tile preparation pipeline that produces a runnable SFINCS
model, a bathtub solve, and an eikonal solve for each tile in an SFINCS-vs-
eikonal validation batch (`sfincs_tiles/`). Covers tile selection through
per-tile model build, run, and postprocessing.

## 1. Tile selection

`select_validation_tiles.py` selects the tiles used for a validation batch.

- Starting population: every tile with `hop_distance == 0` (a real ocean
  edge) in `processed_inputs/mask/domain_tiles_global.gpkg`.
- Excludes antimeridian-crossing tiles (bbox longitude span >60 deg, the
  signature of a tile whose polygon wraps +-180 deg).
- Excludes tiles with `|lat_centroid| > 55` deg.
- Excludes tiles with no `model_outputs/<tile_id>/inputs/mask.tif` yet.
- Eligibility (via `select_test_tiles.py`'s `evaluate_tile`): tile cell
  count under a max, river fraction under a max, and a real match against a
  COAST-HG boundary-forcing station.
- From the eligible pool, keeps tiles with `ocean_frac >= 0.05`, then draws
  a spatially stratified sample (round-robin across coarse lon/lat bins)
  up to the requested tile count, with no area-based pre-filter.

Outputs, under `{root}/{base_dir_name}/`:
- `tile_ids.txt` - one tile ID per line, the batch's own tile roster.
- `tile_selection_metadata.csv` - per-tile metadata for the selected sample.
- `eligible_tile_pool.csv` - the full eligible pool, for audit.
- `tile_locations_map.png`, `tile_selection_histograms.png`,
  `tile_selection_statistics.xlsx` - selection diagnostics.

## 2. Batch job generation

`generate_v2_batch_jobs.py` reads `tile_ids.txt` and writes N independent
SLURM sbatch scripts (`{prefix}_batch_NNN.sbatch`) plus one
`submit_{prefix}es.sh` that submits all of them. Each sbatch script loops
sequentially through its own slice of tile IDs, calling the per-tile runner
script (`run_one_tile_v3.sh` or `run_one_tile_v2.sh`) once per tile, with
`BASE_DIR_NAME=<base_dir_name>` set explicitly on every invocation. Tiles
are split evenly across `--n-nodes` batches; batches are fully independent
(no SLURM job dependency between them). Also writes `resolved_config.yml`
(the Linux-path view of `config.yml`), which every per-tile Python step
reads via `--config`.

## 3. Per-tile pipeline (`run_one_tile_v3.sh`)

Runs on Hydrax, one invocation per tile, `bash run_one_tile_v3.sh <tile_id>
[--models bathtub,eikonal,sfincs] [--max-rounds N]`. Each stage is gated on
its own output file already existing, so a re-run after a partial failure
only redoes what's missing. Steps:

### 3.1 Copy inputs

Copies `tile_geometry.gpkg`, `model_bbox.json`, `dem.tif`, `mask.tif`,
`friction.tif`, `boundaries_RP100_SLR_0.gpkg` from
`model_outputs/<tile_id>/inputs/` into this batch's own
`<base_dir_name>/<tile_id>/inputs/`.

### 3.2 Regenerate `dem.tif` / `mask.tif`

`regenerate_dem_mask.py` calls `src/rasters.py`'s `extract_dem` /
`extract_dem_mask` directly (the same functions the main preprocessing
pipeline uses), rebuilding `dem.tif`/`mask.tif` under today's code rather
than trusting whatever was on disk in `model_outputs/`. This includes
`resolve_offshore_mask_gaps_via_gebco`: where `deltadtm_mask` has no
coverage at all, a nodata cell is reclassified to ocean if GEBCO reads
negative there and it is 4-connected to a cell already confirmed ocean -
otherwise it stays land. Overwrites the copied `dem.tif`/`mask.tif` in
place.

### 3.3 Build SFINCS inputs (env: `hydromt-sfincs-dev`)

**`build_elevation.py`** builds `sfincs_model/elevation_combined.tif`, a
single elevation surface at `dem.tif`'s native grid:

- Land cells: `dem.tif`'s own DeltaDTM value, floored at -15m (works
  around `hydromt_sfincs`'s subgrid-table builder, which clamps any
  subgrid pixel below -20m to exactly -20m).
- Lake/river cells: re-derived from nearby land cells via nearest-valid-
  neighbour fill, then a 500m rolling-minimum filter along the water body,
  floored at 0m.
- Ocean cells: GEBCO bathymetry, bilinear-reprojected onto `dem.tif`'s
  grid, shifted by the local MDT offset (`H_GOCO06s = H_MSL + MDT`) so it
  shares DeltaDTM's own vertical reference, then clipped to [-10, 0] m.
  Within 200m of the coast, GEBCO's own value is discarded and
  interpolated instead, since GEBCO's ~450m native resolution means the
  nearest real sample to the coastline can already read as a clamped-deep
  value with no real sample to interpolate a gradual approach from.

**`build_roughness.py`** builds `sfincs_model/manning_n.tif` by decoding
`friction.tif`'s `Manning's_n / 100` on-disk convention back into a real
Manning's n (`decoded_value * 100`), with a `[0.001, 1.0]` physical-
plausibility check.

**`build_boundary_forcing.py`** matches the tile's own COAST-RP boundary
points (`boundaries_RP100_SLR_0.gpkg`) to COAST-HG hydrograph stations
(nearest within `MAX_MATCH_DIST_DEG=0.5` deg), keeps only the 20 nearest
of those matched points to the tile itself, and computes an empirical MDT
offset per station (`offset = boundary_value_m - hydrograph_max_m`),
applied additively to that station's full hydrograph. Writes
`sfincs_model/matched_boundary_points.gpkg` and
`sfincs_model/corrected_hydrographs.csv`.

**`build_sfincs_tile.py`** assembles the runnable SFINCS model:

1. Coarse computational grid at 120m resolution, UTM (one zone per tile).
2. Active-cell mask over the whole tile; waterlevel boundary cells wherever
   the tile's own ocean polygon touches the active domain
   (`all_touched=True`, needed for a continuous boundary line across a
   coastline running diagonally through the tile's rotated UTM grid).
3. A subgrid table (30m, 4x refinement, 20 hypsometric levels) built from
   `elevation_combined.tif` and `manning_n.tif`, both nearest-neighbour
   pre-reprojected onto the exact subgrid grid before `subgrid.create()`
   runs (avoids `hydromt_sfincs`'s own forced-bilinear resampling, and any
   gap between the source grid and the rotated UTM subgrid footprint is
   filled from the nearest valid neighbour).
4. Water-level forcing from the matched, MDT-corrected hydrographs,
   truncated to the 40-110h window around the storm peak (every COAST-HG
   hydrograph in this pipeline shares the same synthetic time axis, peaking
   at t=74.5h). The forcing buffer radius is computed per tile from the
   real distance of the farthest matched station to the grid's own bbox,
   not a fixed guess.
5. Initial water level (`zsini`): IDW-interpolated from the matched
   stations' first timestep, kept only on cells the tile's native mask
   marks as ocean; every other cell is left at `hydromt_sfincs`'s own dry
   sentinel (-9999.0).
6. Output config: `dtmaxout` spans the whole simulated duration (so
   `zsmax` is one true run-wide maximum), `dtmapout=1800s`.
7. Writes the model to `sfincs_model/`.

### 3.4 Bathtub + eikonal solve (env: `gfm`)

`run_eikonal_on_sfincs_subgrid.py` runs the production eikonal flood solver
(`src/flood_model.py::flood_depth_dense`) and/or a bathtub fill directly on
the SFINCS subgrid's own DEM/roughness (`elevation_combined_subgrid_src.tif`
/ `manning_n_subgrid_src.tif`), instead of the eikonal model's separate
lon/lat DeltaDTM grid. Running on the shared subgrid means both models see
the same isotropic UTM grid, the same 30m resolution, and the same
`effective_dem()` land/ocean/lake/river handling. Boundary seeding uses the
same planar IDW as `zsini` (`idw_interpolate_to_grid`), not the haversine
IDW the eikonal model's own hop>=1 hinterland-tile path uses (that would be
wrong on a projected UTM grid). Writes
`outputs/bathtub_waterdepth_RP100_SLR_0.tif` and/or
`outputs/eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif`
(int16, cm, on the SFINCS subgrid's own UTM grid). Obstacle coupling is not
enabled for this solve.

### 3.5 SFINCS run

Stages the built model to node-local scratch, runs the `sfincs-cpu`
apptainer image (`OMP_NUM_THREADS=4`, timeout 14400s), and copies
`sfincs_map.nc` back to `sfincs_model/`.

### 3.6 Postprocess (env: `hydromt-sfincs-dev`)

**`run_sfincs_tile.py`** computes max inundation depth on the model's fine
subgrid resolution via `hydromt_sfincs`'s `downscale_floodmap()`
(`zsmax` downscaled onto the real subgrid DEM, `hmin=0.05`), masked to
land cells by both `downscale_floodmap()`'s own `gdf_mask` and an
independent reprojection of the native `mask.tif` onto the subgrid (both
checks must agree a cell is land). Reprojects the result to EPSG:4326 with
an explicit latitude-corrected target resolution (so the output pixel is
square in real ground distance, not in degrees), writing
`outputs/hmax.tif` and `outputs/flood_extent.tif`.

**`postprocess_tile_summary.py`** writes `outputs/summary.json`: flooded
area (km2) and depth stats for bathtub, eikonal, and SFINCS; SFINCS-vs-
bathtub and SFINCS-vs-eikonal agreement (matched / model-only / SFINCS-only
km2, via a shared wet threshold); and SFINCS boundary-cell distance to real
land (mean/median/max km). A whole batch's summaries are merged with a glob
+ concat (`aggregate_tile_summaries.py`).

## 4. Local sequential eikonal fill-in

`run_eikonal_v4_sequential.py` runs `run_eikonal_on_sfincs_subgrid.py
--models eikonal` for every tile in a batch's `tile_ids.txt`, one tile at a
time on a local machine rather than via an HPC batch. For each tile it
checks that `sfincs_model/elevation_combined_subgrid_src.tif` exists
(skipping tiles whose SFINCS build hasn't landed yet) and that the eikonal
output doesn't already exist (skipping tiles already done), so it is safe
to interrupt and re-run. Passes `--max-rounds 40`
(`simulation.flooding.max_rounds` in production `config.yml`) by default.

## Environments

- `gfm` (or the local `gfm_python_preprocessing` env): `regenerate_dem_mask.py`,
  `run_eikonal_on_sfincs_subgrid.py` - needs `src/config_utils.py`'s
  `hydromt.DataCatalog` (hydromt 0.9.3).
- `hydromt-sfincs-dev`: `build_elevation.py`, `build_roughness.py`,
  `build_boundary_forcing.py`, `build_sfincs_tile.py`, `run_sfincs_tile.py`,
  `postprocess_tile_summary.py` - needs `hydromt_sfincs` (hydromt 1.4.1);
  does not import `src/config_utils.py` at all (incompatible with this
  env's hydromt version), using `sfincs_tiles/gfm_config.py`'s
  `read_root`/`resolve_catalog_path` instead.

## Output layout

```
{root}/{base_dir_name}/
  tile_ids.txt, tile_selection_metadata.csv, eligible_tile_pool.csv, ...
  resolved_config.yml
  hpc_jobs/
    {prefix}_batch_NNN.sbatch, submit_{prefix}es.sh, logs/
  {tile_id}/
    inputs/            dem.tif, mask.tif, friction.tif, tile_geometry.gpkg,
                        model_bbox.json, boundaries_RP100_SLR_0.gpkg
    sfincs_model/       elevation_combined.tif, manning_n.tif,
                        matched_boundary_points.gpkg, corrected_hydrographs.csv,
                        sfincs.inp, sfincs_map.nc, subgrid/, ...
    outputs/            bathtub_waterdepth_RP100_SLR_0.tif,
                        eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif,
                        hmax.tif, flood_extent.tif, summary.json
```
