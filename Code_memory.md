# GFM — Claude Code Reference Memory

Project-level orientation doc for future Claude Code sessions. Covers the whole
repo: `core/` (Julia flood model), `snakemake_workflow/` (Python/Snakemake
preprocessing pipeline), `python/` (tile-grid generation + legacy
watershed-based prep), and `Boundary_conditions_waterlevels/` (legacy water
level scenario prep). Update this file when the pipeline structure, configs,
or module responsibilities change materially.

## 1. Project Overview

- GFM (Global Flood Model) extends the
  [Aqueduct Coastal Flooding](https://github.com/Deltares-research/aqueduct-coastal-flooding)
  project to **global, tile-based** coastal flood simulation (as opposed to a
  per-delta/per-basin model like GCFM_UU).
- Two halves:
  - `core/` — the Julia package `Aqueduct` implementing the actual flood model
    (`flood_depth`), compiled into a standalone executable
    `build/aqueduct/aqueduct.exe` via PackageCompiler + a Rust CLI wrapper
    (`build/cli/`).
  - `snakemake_workflow/` — Python/Snakemake pipeline that prepares per-tile
    inputs (DEM, DEM-validity mask, friction, water-level boundary points) for
    `aqueduct.exe`, runs it per (tile, SLR scenario), and merges/plots results.
- Environment is fully managed by **pixi** (`pixi.toml` / `pixi.lock`): conda
  deps (hydromt, geopandas, rasterio, snakemake-minimal, xarray, ...) plus
  Julia (via `juliaup`, dev feature) and Rust (for the CLI build, dev feature).
- `pixi run build` → `julia --project=build build/build.jl` → compiles
  `core/` + `build/cli/` → `build/aqueduct/aqueduct.exe`.
- Processing is **tile-based**: the world is split into an overlapping tile
  grid (`python/tile_mask_creation.py`), filtered/merged
  (`snakemake_workflow/select_tiles.py`, `merge_tiles.py`), and every tile is
  run through the Snakemake pipeline independently, then results are merged
  back into combined rasters/plots.

## 2. Pipeline Structure

### 2a. Tile grid preparation (one-off, manual — NOT in the Snakemake DAG)

Run in order; each step writes a new GeoPackage consumed by the next:

1. **`python/tile_mask_creation.py`** — reads the **root** `config.yml`
   (`paths.hydromt_lib`, `paths.tile_mask_grid`). Splits each 5°x5° tile of
   the `five_deg_grid` source into four 2.5°x2.5° quadrants (SW=0, SE=1,
   NW=2, NE=3), then scales each by `SCALE_FACTOR=1.5` around its center →
   3.75°x3.75° tiles overlapping neighbors by 0.625° on each side.
   `tile_id = parent_id * 10 + quadrant_id`. Output → `paths.tile_mask_grid`.
2. **`snakemake_workflow/select_tiles.py`** — reads
   `snakemake_workflow/config/config.yml`. Filters tiles down to those with
   DEM (`data_sources.dem`) coverage via `src/tiles.has_dem_coverage` /
   `filter_tiles_with_dem_coverage` (coarse `tile_selection.sample_size`-px
   read per tile; tiles without coverage are logged and dropped). Output →
   `<input>_filtered.gpkg`.
3. **`snakemake_workflow/merge_tiles.py`** — merges tiles with too little
   ocean *or* land coverage (`tile_merging.min_fraction`, default `0.05`)
   into a neighbor, via `src/tiles.compute_land_ocean_fractions` /
   `merge_undersized_tiles`. Output → `<input>_merged.gpkg`.
4. Point `paths.tile_grid` in `snakemake_workflow/config/config.yml` at the
   final merged grid. `Snakefile` computes
   `TILE_IDS = sorted(gpd.read_file(tile_grid)["tile_id"].astype(int).tolist())`
   **at parse time** — re-run `snakemake` (don't expect it to pick up grid
   changes mid-run).

### 2b. Snakemake workflow (`snakemake_workflow/`)

Entry point `Snakefile`, config `config/config.yml`, rules split across
`rules/{common,preprocessing,simulation,postprocessing}.smk`.

**Preprocessing** — outputs under `{model_outputs}/{tile_id}/inputs/`:

| Rule | Input | Output | Script | Key src functions |
|---|---|---|---|---|
| `extract_tile_geometry` | — | `tile_geometry.gpkg` | `extract_tile_geometry.py` | `tiles.get_tile_geometry`, `save_tile_geometry` |
| `compute_model_bbox` | `tile_geometry.gpkg` | `model_bbox.json` | `compute_model_bbox.py` | `rasters.compute_model_bbox`, `get_tile_bbox` |
| `extract_dem` | `model_bbox.json` | `dem.tif` | `extract_dem.py` | `rasters.extract_dem` |
| `extract_dem_mask` | `dem.tif` | `mask.tif` | `extract_dem_mask.py` | `rasters.extract_dem_mask` |
| `compute_friction` | `dem.tif` | `friction.tif` | `compute_friction.py` | `rasters.compute_friction` |
| `extract_boundaries` *(per waterlevel_name)* | `tile_geometry.gpkg` | `boundaries_{waterlevel_name}.gpkg` | `extract_boundaries.py` | `boundaries.load_waterlevel_stations`, `select_stations_for_tile`, `save_boundary_points` |
| `write_aqueduct_config` *(per waterlevel_name)* | `dem`, `mask`, `friction`, `boundaries` | `aqueduct_{waterlevel_name}.toml` | `write_aqueduct_config.py` | `aqueduct_config.build_aqueduct_config`, `write_aqueduct_config` |

DAG: `extract_tile_geometry → compute_model_bbox → extract_dem → {extract_dem_mask, compute_friction}`.
`extract_boundaries` runs in parallel from `tile_geometry.gpkg`; `write_aqueduct_config` depends on all four preprocessed inputs.

**Simulation** — output under `{model_outputs}/{tile_id}/results/`:

| Rule | Output | Script | Key src functions |
|---|---|---|---|
| `run_aqueduct` *(per waterlevel_name)* | `waterdepth_{waterlevel_name}.tif` | `run_aqueduct.py` | `aqueduct_runner.run_aqueduct` (subprocess → `aqueduct.exe <toml>`) |

`run_aqueduct` declares `resources: aqueduct_runs=1` — **the Julia LLVM JIT
cannot reliably run multiple `aqueduct.exe` instances concurrently** (crashes
with `OutOfMemoryError` / `LLVM ERROR: Unable to allocate section memory!`).
Always pass `--resources aqueduct_runs=1`; preprocessing rules still use the
remaining `--cores`.

**Skipped tiles (2026-06-12)**: `run_aqueduct.py` reads `boundaries`
(`boundaries_{waterlevel_name}.gpkg`) first; if it has **zero stations**
(possible now that `select_stations_for_tile` no longer buffers beyond the
tile bbox), Aqueduct is not run for that (tile_id, waterlevel_name). Instead:
`rasters.save_nodata_raster` writes an all-`AQUEDUCT_NODATA` placeholder as
`waterdepth_{waterlevel_name}.tif` (so `merge_tile_rasters` ignores the tile),
and `aqueduct_runner.log_skipped_tile` writes a marker file to
`{model_outputs}/skipped_tiles/{tile_id}_{waterlevel_name}.txt`. Glob that
directory to build a map of skipped tiles against the tile grid.

**OOM tile skip (2026-06-15)**: `aqueduct.exe` can crash with
`OutOfMemoryError()` inside `component_indices` (`core/src/core.jl:67`,
called from `flood_depth`'s connected-component filter). Its memory cost is
dominated by the tile's total pixel count (label_components partitions
**every** pixel, including the background label, into per-component index
arrays), not by `waterlevel_name` - so a tile that OOMs for one SLR scenario
will OOM for (nearly) all of them. To avoid repeatedly burning ~3-4min per
scenario on tiles that can never succeed:
- `run_aqueduct.py` catches `CalledProcessError` and checks
  `aqueduct_runner.is_oom_error` (searches captured stdout/stderr for
  `"OutOfMemoryError"`, distinct from the transient/concurrency
  `LLVM ERROR: Unable to allocate section memory!` crash covered by
  `aqueduct_runs=1`).
- On a real OOM, `aqueduct_runner.mark_tile_oom` writes
  `{model_outputs}/oom_tiles/{tile_id}.txt`, and the job falls back to the
  same `save_nodata_raster` + `log_skipped_tile` placeholder as the
  zero-boundary-stations case above (so the job succeeds and
  `merge_tile_rasters` ignores this tile/scenario).
- Before running Aqueduct at all, `run_aqueduct.py` checks
  `aqueduct_runner.tile_marked_oom(oom_dir, tile_id)`; if the tile was
  already marked (by an earlier scenario, this run or a previous one), it
  skips straight to the nodata placeholder without invoking `aqueduct.exe`.
- Net effect: the first OOM for a given `tile_id` still costs one real
  (failing) Aqueduct run, but all other `waterlevel_name` scenarios for that
  tile become near-instant nodata placeholders, both within the current
  `snakemake` invocation and on future re-runs (the marker persists on disk).
- **Tradeoff**: tiles marked OOM contribute **no flood data at all** to the
  merged outputs for any SLR scenario. As of 2026-06-15, the largest tiles by
  pixel count are 16782 (497M px, 25279x19662), 12723 (434M px,
  30897x14044), 17770 (355M px, 25280x14044) and 15581 (276M px,
  19662x14044) - all candidates for OOM. The root cause
  (`component_indices` building per-component index arrays for the whole
  raster, including the huge background label) could instead be fixed in
  `core/src/core.jl` by replacing it with a `Set`-membership broadcast over
  `labels` (avoiding `component_indices` entirely), which would let large
  tiles produce real results - not yet implemented (requires a
  `pixi run build` rebuild + retest).

**Postprocessing** — outputs under `{merged_outputs}/`:

| Rule | Output | Script | Key src functions |
|---|---|---|---|
| `merge_results` *(per waterlevel_name)* | `flood_count_{waterlevel_name}.tif`, `waterdepth_{waterlevel_name}.tif` | `merge_results.py` | `merge.merge_tile_rasters` |
| `plot_merged_results` *(per waterlevel_name)* | `plots/flood_count_{waterlevel_name}.png`, `plots/waterdepth_{waterlevel_name}.png` | `plot_merged_results.py` | `plotting.plot_raster_with_coastlines` |

**Aggregate targets** (in `Snakefile`): `preprocess`, `simulate`,
`postprocess`, `all` (= preprocess + simulate + postprocess, the default).

**Run**: `snakemake all --cores 4 --resources aqueduct_runs=1`

## 3. Key Data Sources (HydroMT catalogs)

### `python/config/data_catalog_gfm.yml`
The **active** catalog, referenced via `paths.hydromt_data_catalog` in
`snakemake_workflow/config/config.yml` and loaded via
`config_utils.get_data_catalog(config)`:

- **`deltadtm`** — DeltaDTM DEM, VRT mosaic, EPSG:4326. → `data_sources.dem`.
  Native nodata sentinel `-3.4028235e+38`.
- **`deltadtm_mask`** — DEM-validity mask, VRT, different native
  resolution/grid than the DEM, nodata sentinel `255`. → `data_sources.dem_mask`.
- **`osm_coastlines`** — OSM coastline water polygons (GeoPackage,
  `osm_coastlines-db.gpkg`). Still in the catalog but no longer referenced by
  the active Snakemake pipeline; superseded by `land_polygons` below.
- **`land_polygons`** — OSM land polygons (GeoPackage, `land_polygons.gpkg`,
  global 1°×1° tiles). **Active coastlines source** → `data_sources.coastlines`.
  Used in three places: (1) `compute_model_bbox` — ocean extent = tile area
  minus land polygons; (2) `extract_dem` / `extract_dem_mask` — authoritative
  land/ocean boundary for filling nodata cells; (3) `plot_merged_results` —
  plotting context. The GeoPackage has **two layers**: `land_polygons` (used)
  and `marine_buffer` (default layer when no layer name is given — does not
  cover inland areas, do not use). All `gpd.read_file` calls therefore pass
  `layer="land_polygons"` explicitly.
- **`coastrp_rp100_slr0`** — example COAST-RP_EWL water-level GeoDataset
  (RP100, SLR_0), produced by the legacy `Boundary_conditions_waterlevels/`
  pipeline. The actual sources used at runtime are the
  `COAST-RP_EWL_{return_period}_{waterlevel_name}.nc` files under
  `paths.waterlevel_nc_dir` (see §6), read directly with `xarray`, not via the
  catalog.
- **`hybas_NW_eu_lev03_v1c`**, **`hybas_global_v1c`** — HydroBASINS watershed
  polygons, used by the legacy per-watershed pipeline
  (`python/preprocessing.py`, §13).
- **`five_deg_grid`** — global 5°x5° tile index (derived from the
  DiluviumDEM tile index), source for `tile_mask_creation.py`.
- **`land_use`** — Copernicus Global Land Cover 100m (`Copernicus_LandUse.tif`,
  362880x141120 px, ~110m res, bounds -180..180 lon / -60..80 lat). Class code
  `200` = ocean/open water, nodata = `255`. → `data_sources.land_use`. Used
  both for friction computation (§4) and for tile ocean/land fractions in
  `merge_tiles.py` (§5).
- **`lu_to_roughness_lookup`** — CSV lookup, index column
  `copernicus_worldcover` (Copernicus land-use class code), value column
  `manning_n` (Manning's n roughness coefficient; `-999.0` = invalid/nodata
  marker). → `data_sources.roughness_lookup`.

### `python/config/preprocessing_catalog.yml`
Separate catalog used only by the **legacy** `Boundary_conditions_waterlevels/`
notebooks (§13):

- **`mdt_hybrid_cnes_cls22_cmems2020`** — Mean Dynamic Topography
  (MDT-HYBRID-CNES-CLS22-CMEMS2020), 1/8° global, used for datum correction of
  water levels (notebook `01_retrieve_MDT_correction.ipynb`).
- **`ipcc_ar6_slr_projections`** — IPCC AR6 sea-level-rise projections
  (FACTS model output), organized by SSP scenario / confidence level / type
  (values vs rates) (notebook `02_get_SLR_fingerprint.ipynb`).

## 4. Topography, DEM Mask & Friction Processing (`src/rasters.py`)

Tiles in the overlapping grid are **exact rectangles**, so a tile's bounding
box == its geometry — rasters are clipped with `bbox=...` only, no
`geometry_mask` step is needed anywhere in this module.

### Helpers

- **`get_tile_bbox(tile)`** → `[minx, miny, maxx, maxy]` from a single-row
  tile GeoDataFrame.
- **`load_raster(path)`** → `xr.DataArray` (single band, via rioxarray, with
  `.raster` accessor). Used by scripts that load a previously saved DEM to
  derive the model-domain bbox from `dem.raster.bounds`.
- **`_land_raster(data_catalog, land_polygons_source, bbox, dem)`** → `bool
  ndarray` shaped `(height, width)`, True where the OSM `land_polygons` layer
  covers each pixel. All callers pass `layer="land_polygons"` to avoid the
  `marine_buffer` default layer. Returns all-False if no land features intersect
  `bbox` (e.g. a purely oceanic tile).
- **`_DEM_LAND_FILL = 9999.0`** — elevation written for land cells without
  DeltaDTM coverage. High enough that these cells never flood.

### Domain computation

- **`compute_model_bbox(data_catalog, dem_source, land_polygons_source, tile_bbox)`**
  → `[minx, miny, maxx, maxy]` — computes a tight model domain by taking the
  union of (a) the bounding box of valid DeltaDTM pixels and (b) the bounding
  box of the ocean area within the tile (= tile area minus land polygons), then
  clipping the result to `tile_bbox`. This reduces the tile domain to only the
  coastal strip that needs to be processed. Written to `model_bbox.json`; all
  downstream rules derive their bbox from the saved DEM's bounds.

### Per-tile extraction

- **`extract_dem(data_catalog, dem_source, bbox, mask_source)`** — clips the
  DEM to `bbox`. **No nodata cells remain** in the output: the fill value for
  missing DEM cells is decided entirely from `mask_source` (DeltaDTM's own
  validity mask, reprojected onto the DEM's grid) — NOT from a separately
  sourced OSM land polygon dataset (removed 2026-07 - it could misalign with
  DeltaDTM's own coastline). Missing cells where the mask says land (0) OR
  where the mask has no coverage at all (255, nodata) → `_DEM_LAND_FILL`
  (9999.0) — areas outside DeltaDTM's own coverage are treated as irrelevant,
  definitely-dry land, not an unknown for another dataset to arbitrate.
  Missing cells where the mask says ocean (1), lake (2) or river (3) → `0.0`.
  Ocean cells get 0.0 because Aqueduct overwrites them anyway via
  `dem[.!landmask] .= 0.0`; lake/river cells get 0.0 so flooding can still
  propagate through them; land (and mask-uncovered) cells get 9999 so they
  are never spuriously flooded. The output raster has no declared nodata value.
- **`extract_dem_mask(data_catalog, mask_source, bbox, dem, nodata_sentinel=255)`**
  — clips the DeltaDTM validity mask, reprojects (nearest) onto `dem`'s grid.
  Cells with no mask coverage at all (`nodata_sentinel`) are set to land (0)
  — consistent with `extract_dem`'s own rule, and no longer land-polygon
  dependent. Valid DeltaDTM values (0=land, 1=ocean, 2=lake, 3=river) are kept
  unchanged. All cells have a valid value; no nodata cells remain.
- **`compute_friction(data_catalog, lulc_source, lookup_source, bbox, dem,
  default_friction)`** — clips Copernicus land use, reprojects (mode) onto
  `dem`'s grid, builds a vectorized lookup array from `lu_to_roughness_lookup`
  (indexed by `copernicus_worldcover`, `manning_n` → renamed to `N`,
  `-999.0` = invalid), computes `friction = N / 100`. Cells with no valid
  land-use classification (ocean, outside LULC coverage, unmapped codes) are
  filled with `default_friction` (`flooding.default_friction` in config, default
  `0.001` — matches the Aqueduct core's `coalesce(friction, 0.001)` hardcoded
  default). All cells have a valid value; no nodata cells remain.

### Raster IO

- **`save_raster(da, output_path, raster_config, dtype="float32")`** — writes a
  single-band raster using the `raster` config section (`driver`="COG",
  `compression`="zstd", `predictor`=3, `nodata`=-9999).
- **`save_nodata_raster(reference_path, output_path, raster_config)`** — writes
  an all-`AQUEDUCT_NODATA` placeholder on the same grid as `reference_path`.
  Used for skipped/OOM tiles so `merge_tile_rasters` ignores them.

## 5. Tile Grid Generation & Merging (`src/tiles.py`)

- **`load_tile_grid`**, **`get_tile_geometry`**, **`save_tile_geometry`** —
  basic GeoPackage IO for single tiles.
- **`has_dem_coverage(dem_path, bbox, sample_size=128)`** /
  **`filter_tiles_with_dem_coverage`** — used by `select_tiles.py` (§2a step
  2). Reads a coarse `sample_size`x`sample_size` window per tile to cheaply
  check for any valid DEM pixel.
- **`_fraction_for_window(src, bounds, ocean_code, nodata)`** — full-resolution
  windowed read (`boundless=True, fill_value=nodata`) of the land-use raster
  for one bbox; returns `(ocean_fraction, land_fraction)` (both 0 if no valid
  pixels). Per-tile reads of `Copernicus_LandUse.tif` are cheap (~0.1s/tile)
  even at full resolution — deliberately **not** downsampled.
- **`compute_land_ocean_fractions(tile_grid, land_use_path, ocean_code, nodata=255)`**
  — adds `ocean_fraction`/`land_fraction` columns to every tile.
- **`merge_undersized_tiles(tile_grid, land_use_path, ocean_code, min_fraction, nodata=255)`**
  — the core of `merge_tiles.py` (§2a step 3). A tile is "bad" if
  `ocean_fraction < min_fraction` OR `land_fraction < min_fraction`. For each
  bad tile, in order:
  0. **Drop pure-ocean tiles**: if `land_fraction == 0` (100% ocean, no land
     at all), the tile is **dropped from the grid outright** — not merged
     with anything, since there's nothing to flood-model there and merging
     would only dilute a neighbor's fractions.
  1. **Bad-bad unionize**: look for a "bad" **cardinal** (N/S/E/W only — see
     `_is_cardinal_neighbor`) neighbor such that the bbox-of-union of both
     geometries meets `min_fraction` for *both* fractions. If found, the two
     tiles merge into one "good" tile (bbox-of-union geometry); preferred
     because it resolves two bad tiles without growing any existing tile.
  2. **Bad-good absorption**: otherwise, absorb into a "good" cardinal
     neighbor (`_pick_merge_partner`: prefer same-area neighbors, else
     smallest "good" neighbor, tie-broken by max overlap area). Surviving
     geometry = bbox-of-union; the bad tile is dropped.
  3. If none of the above applies (e.g. a small island surrounded by ocean),
     the tile is **kept unchanged**.
- **`_is_cardinal_neighbor(geom_a, geom_b, threshold=0.5)`** — for two
  axis-aligned bbox rectangles, returns True only if their bounds overlap and
  that overlap spans ≥`threshold` of the smaller tile's width (N/S
  relationship) or height (E/W relationship). Diagonal/corner-touching tiles
  return False. `threshold=0.5` is hardcoded (not config-exposed).
- **`_pick_merge_partner(candidates, own_size)`** — candidates are
  `(other_id, other_size, overlap_area, ...)` tuples; prefers
  `np.isclose(other_size, own_size)`, else the smallest-size subset, then
  picks the max-overlap candidate.

**Validated on the Africa grid** (427 DEM-filtered tiles, `min_fraction=0.05`):
289 final tiles, 138 bad→good absorptions, 51 islands kept unchanged, max
single-tile growth 3x (down from 11x before the cardinal-neighbor
restriction — see §11).

## 6. Boundary Forcings & Flood Model Core

### Boundary points (`src/boundaries.py`)
- **`load_waterlevel_stations(nc_path, variable, x_var, y_var, column_name)`**
  — reads one `COAST-RP_EWL_{return_period}_{waterlevel_name}.nc` file
  (`paths.waterlevel_nc_dir`), builds a point GeoDataFrame
  (EPSG:4326) with a `column_name` water-level column. **Drops stations with
  NaN water level** (`stations[stations[column_name].notna()]`) — the
  Aqueduct flood model cannot handle missing boundary values.
- **`select_stations_for_tile(stations, tile)`** — selects stations
  intersecting the tile's bounding box (no buffer; `2026-06-12` removed the
  earlier `station_buffer_deg` expansion per user direction — only stations
  within the tile itself are used). Simple bbox intersect — no
  `min_stations`/fallback logic.
- **`save_boundary_points`** — writes the selection to GeoPackage.

### Aqueduct config schema (`core/src/config.jl`, written by `src/aqueduct_config.py`)
TOML structure (via `Configurations.jl`):
```toml
input_dir = "."
results_dir = "../results"
[input]
dem = "dem.tif"
mask = "mask.tif"
friction = "friction.tif"
boundaries = "boundaries_{waterlevel_name}.gpkg"
[output]
waterdepth = "waterdepth_{waterlevel_name}.tif"
[flooding]
resolution = 30      # flooding.resolution
debug = false        # flooding.debug
[waterlevels]
knn = 15             # flooding.knn
name = "{waterlevel_name}"  # must match the column name in `boundaries`
```
`input` paths resolve relative to the TOML's directory + `input_dir`;
`output.waterdepth` resolves relative to TOML dir + `results_dir`
(`config/Config` provides `input_path`/`results_path` helpers).

### Flood model core (`core/src/core.jl`, `flood_depth`)
1. Reads boundary point coordinates (`x`, `y`) and the `variable` column
   (= `waterlevels.name`) from the `boundaries` GeoDataFrame.
2. Builds a coastline mask: `coastlinemask = dilate(landmask) != mask` (cells
   on the land/water boundary).
3. Builds a `BallTree` (Haversine distance) over the boundary station
   coordinates.
4. **`k = min(length(x), k)`** — clamps the configured KNN `k`
   (`flooding.knn`/`waterlevels.knn`) to the number of available boundary
   stations (fix from a prior session: avoids KNN errors when a tile has
   fewer boundary stations than `knn`).
5. **`idw!`** — inverse-distance-squared weighting interpolates boundary
   water levels onto every coastline cell → `initial`.
6. **FastSweeping** (Eikonal.jl) propagates `initial` inland over the
   `friction` field (used as the eikonal "speed"/cost), `epsilon = min(friction) / (resolution * 10)`
   → `waterlevel` raster.
7. `flood = (waterlevel > dem) & (mask != 1)` — `mask = 1` is ocean (see
   `extract_dem_mask` in §4); the condition `mask != 1` restricts flooding to
   land (0), lake (2), and river (3) cells. Ocean cells are explicitly excluded.
8. Connected-component filter: only flood components touching the (dilated)
   coastline mask are kept — removes isolated inland depressions with no
   hydraulic connection to the sea.
9. `waterdepth = waterlevel - dem` where flooded, else `0`.

`cover2friction` (ESA land-cover code → friction dict) is a legacy
alternative to the Python `compute_friction` lookup-table approach; not used
by the current Snakemake pipeline.

### Entry points (`core/src/main.jl`, `core/src/lib.jl`, `core/src/Aqueduct.jl`)
- `Aqueduct.main(ARGS)` — CLI entry: parses the TOML config, sets up a
  `ConsoleLogger` (Debug if `flooding.debug` else Info), runs `flood_depth`,
  writes `output.waterdepth` as COG/zstd/predictor=3 with `nodata=0.0`.
- `core/src/lib.jl` — `@ccallable execute(toml_path::Cstring)::Cint` for the
  `libaqueduct` shared library (built by `build/build.jl` via
  PackageCompiler's `create_library`).
- `Aqueduct.jl` — module root, `include`s `core.jl`, `config.jl`, `main.jl`,
  `lib.jl`; deps: LocalFilters, ImageMorphology, Geomorphometry, Eikonal,
  GeoArrays, GeoDataFrames, Distances, NearestNeighbors, GeoInterface.

## 7. Simulation, Build & Testing

- **Build**: `pixi run build` → `julia --project=build build/build.jl` →
  `PackageCompiler.create_library("../core", "aqueduct"; lib_name="libaqueduct", precompile_execution_file="precompile.jl", ...)`
  + `cargo build --release` in `build/cli/` → copies the resulting
  `aqueduct(.exe)` into `build/aqueduct/`. Also writes `Build.toml`,
  `Project.toml`/`Manifest.toml` copies, and a version-stamped `README.md`
  into `build/aqueduct/`. If the build fails to find files on Windows, enable
  long paths (`build/README.md`).
- **`core/test/{region}/`** — per-region test fixtures/configs for the Julia
  core (`britain`, `england`, `iberia`, `ireland`, `larochelle`, `mozambique`,
  `scotland`, `southmyanmar`, `westfrance`, `westitaly`). Predates the
  tile-based Snakemake pipeline; the README's "Ireland" usage example
  (`aqueduct_rp1_slr_500.toml`) refers to one of these. Useful as small,
  self-contained `aqueduct.exe` smoke-test inputs.
- **`core/scripts/example.ipynb`** — example notebook driving `Aqueduct.jl`
  directly (Julia).
- **`validation/aqueduct_floodmaps.ipynb`** — notebook for visually
  inspecting/validating output flood-depth rasters.
- **pixi invocation gotcha**: `pixi run ...` / `pixi run -e dev ...` fail in
  this checkout (tries to solve for `osx-arm64`/`default` env, errors on
  `snakemake-minimal >=8.0`). **Workaround**: call the dev env's Python
  directly, e.g. `.pixi/envs/dev/python.exe -c "..."` or
  `.pixi/envs/dev/python.exe snakemake_workflow/select_tiles.py`. This
  bypasses pixi's environment resolution entirely.

## 8. Postprocessing & Plotting

### `src/merge.py`
Tiles in the overlapping grid share a common pixel grid (all clipped from the
same source DEM mosaic at the same resolution), so merging needs **no
resampling** — each tile is read with a window aligned to the combined output
grid.

- `AQUEDUCT_NODATA = np.finfo(np.float32).max` — sentinel the Aqueduct model
  writes for cells **outside the area it computed** (distinct from `0.0` =
  "computed, no flooding").
- `compute_union_grid(raster_paths)` → `(transform, width, height, crs)`
  covering the union of all tile bounds (assumes shared CRS/resolution/pixel
  alignment).
- `merge_tile_rasters(tile_rasters, count_output_path, waterdepth_output_path, block_size, raster_config)`
  — processes the combined grid in `block_size`x`block_size` blocks (bounded
  memory). Per cell:
  - `flood_count` (uint8) = number of tiles with valid (`< AQUEDUCT_NODATA`)
    and `> 0` water depth at that cell.
  - `waterdepth` (float32) = **IDW (1/distance) weighted mean** of all tiles'
    valid water depths, where distance is the Euclidean distance in degrees
    from the pixel to each tile's bounding-box centroid. In single-tile areas
    the result equals that tile's value exactly; in overlap zones it transitions
    smoothly. `raster_config["nodata"]` (-9999) where no tile has valid data.
    Uses `np.divide(..., where=weight_sum > 0)` to avoid a RuntimeWarning from
    0/0 in unoccupied cells.

### `src/plotting.py`
- `plot_raster_with_coastlines(raster_path, coastlines, output_path, title, label, cmap, max_size, mask_value=None, oom_tiles=None)`
  — reads the raster downsampled to at most `max_size` px/side
  (`Resampling.average`), masks `nodata` and (optionally) `mask_value`
  (e.g. 0, so zero-depth cells are transparent and the whitesmoke land
  background shows through). No explicit ocean-pixel masking via
  `geometry_mask` — flooding only occurs on land so ocean pixels are already 0
  or nodata. `coastlines` is drawn as a whitesmoke background for context only.
  Optionally draws OOM-skipped tiles as a grey overlay. `vmax =
  min(float(masked.max()), 10)` caps the colorbar at 10 m (uses Python's
  built-in `min()`, not `np.min()` which takes `axis` as its second arg).
- `plot_merged_results.py` reads `land_polygons` **directly via
  `gpd.read_file(path, layer="land_polygons", bbox=...)`** rather than
  `data_catalog.get_geodataframe(...)` — the latter reads the whole global
  dataset first (minutes), while `gpd.read_file` uses the GeoPackage's spatial
  index.

## 9. Configuration

There are **two separate, unrelated `config.yml` files** — don't confuse them:

### Root `c:\Users\Schlu005\GFM\config.yml` (legacy)
Used by `python/tile_mask_creation.py` and the legacy
`python/preprocessing.py` / `Boundary_conditions_waterlevels/` pipelines.
Sections:
- `paths`: `boundary_conditions`, `processed_inputs`, `MDT_correction_file`,
  `SLR_directory`, `WL_scenarios`, `hydromt_lib` (→
  `python/config/data_catalog_gfm.yml`), `domain_mask`, `masks`, `run_output`,
  `tile_mask_grid`.
- `choices`: `SLR_scenario` (e.g. `"ssp245"`, options ssp126/ssp245/ssp585),
  `confidence_level` (`"medium"`, options low/medium).
- Many path values carry `# original: 'p:/11210264-004-global-flood-modellin/...'`
  comments — these are the original Deltares P-drive locations, now mirrored
  under `D:/GFM/...`.

### `snakemake_workflow/config/config.yml` (active workflow config)
Everything the Snakemake workflow's rules/scripts need — **nothing should be
hardcoded in `rules/` or `scripts/`**. Sections:
- `paths`: `hydromt_data_catalog`, `tile_grid` (→ final merged tile grid, see
  §2a), `model_outputs`, `waterlevel_nc_dir`, `aqueduct_executable`,
  `merged_outputs`.
- `data_sources`: `dem`, `dem_mask`, `land_use`, `roughness_lookup`,
  `coastlines` — names into `data_catalog_gfm.yml`. `coastlines` maps to the
  `land_polygons` catalog entry (the `land_polygons` layer of `land_polygons.gpkg`).
- `tile_selection`: `sample_size` (128) — for `select_tiles.py`.
- `tile_merging`: `min_fraction` (0.05), `ocean_landcover_code` (200) — for
  `merge_tiles.py` (§2a, §5).
- `waterlevels`: `return_period` ("RP100"), `names` (`SLR_0`...`SLR_3000`,
  7 scenarios), `nc_filename_template`, `nc_variable_template`,
  `station_x_var`/`station_y_var`.
- `flooding`: `resolution` (30), `knn` (15), `debug` (false) — written into
  each tile/scenario's Aqueduct TOML. `default_friction` (0.001) — fill value
  for cells with no valid LULC classification in `compute_friction`; mirrors
  Aqueduct core's `coalesce(friction, 0.001)` hardcoded default.
- `raster`: `driver`="COG", `compression`="zstd", `predictor`=3,
  `nodata`=-9999 — shared by DEM/mask/friction outputs.
- `merge`: `block_size` (2048), `n_overlap_locations` (10), `driver`="GTiff",
  `compression`="zstd", `predictor`=3, `nodata`=-9999, `plot_max_size` (2000),
  `count_cmap`="viridis", `waterdepth_cmap`="Blues". No `aggregation` key —
  IDW weighting is the only merge strategy implemented.

`rules/common.smk` sets `wildcard_constraints`: `tile_id=r"\d+"`,
`waterlevel_name="|".join(config["waterlevels"]["names"])`.

## 10. Src Module Responsibilities (`snakemake_workflow/src/`)

| Module | Responsibility |
|---|---|
| `config_utils.py` | `get_data_catalog(config, logger_name=...)` → `hydromt.DataCatalog` from `paths.hydromt_data_catalog`. |
| `tiles.py` | Tile grid IO, DEM-coverage filtering, ocean/land fraction computation, undersized-tile merging (§5). |
| `rasters.py` | Per-tile domain reduction (`compute_model_bbox`), DEM/mask/friction extraction, raster IO (`load_raster`, `save_raster`, `save_nodata_raster`). Internal `_land_raster` helper provides OSM land polygon rasterisation used by DEM and mask extraction. See §4. |
| `boundaries.py` | Water-level station loading + per-tile selection (§6). |
| `aqueduct_config.py` | Builds/writes the per-tile/scenario Aqueduct TOML (§6). |
| `aqueduct_runner.py` | `run_aqueduct(executable_path, config_path)` — subprocess wrapper around `aqueduct.exe`, raises on non-zero exit (prints captured stdout/stderr first). `log_skipped_tile(log_dir, tile_id, waterlevel_name, reason)` — marker file for skipped tiles (§6). `is_oom_error`, `mark_tile_oom`, `tile_marked_oom` — OOM-tile detection/skip (§6). |
| `merge.py` | Combines per-tile result rasters into flood-count/water-depth mosaics (§8). |
| `plotting.py` | Raster + OSM-coastline plotting (§8). |

`snakemake_workflow/scripts/*.py` are thin Snakemake `script:` shims: each
reads `snakemake.input`/`snakemake.config`/`snakemake.wildcards`, calls the
corresponding `src/` function(s), writes `snakemake.output`. No logic lives
in `scripts/` beyond wiring.

## 11. Known Issues / Design Decisions

- **pixi invocation**: see §7 — use `.pixi/envs/dev/python.exe` directly
  instead of `pixi run`/`pixi run -e dev` in this checkout.
- **Cardinal-neighbor tile merging (Phase D)**: an earlier version of
  `merge_undersized_tiles` considered *any* overlapping "good" neighbor
  (including diagonal/corner-touch), which let one tile (16413) absorb a
  chain of 7 neighbors and grow 11x. Per explicit user direction, merging is
  now restricted to immediate top/bottom/left/right (`_is_cardinal_neighbor`,
  threshold=0.5) neighbors, and a "bad-bad unionize" path was added (two bad
  cardinal neighbors merge into one good tile if their union meets
  `min_fraction`). Re-tested on the Africa grid: max growth dropped from 11x
  to 3x. The bad-bad path is implemented and correct but was not triggered by
  the Africa dataset (0 occurrences) — worth re-checking once exercised on a
  dataset where it fires.
- **Drop pure-ocean tiles (Phase E, 2026-06-12)**: `merge_undersized_tiles`
  now drops any bad tile with `land_fraction == 0` (100% ocean) immediately,
  before attempting bad-bad unionize or bad-good absorption. Per explicit
  user direction — such tiles have nothing to flood-model and shouldn't be
  merged into a neighbor (that would only dilute the neighbor's ocean/land
  balance). Not yet re-validated against a real tile grid to confirm how many
  tiles this drops in practice.
- **No-buffer station selection (2026-06-12)**: `select_stations_for_tile`
  no longer expands the tile bbox (`station_buffer_deg` removed from config
  and the function signature) — per user direction, only stations within a
  tile's own bounds feed `core.jl`'s `k=15` KNN for that tile. Watch for
  small/merged tiles with **fewer than `flooding.knn` (15) stations in their
  bbox**: `flood_depth` clamps `k = min(length(x), k)`, so this degrades
  gracefully to fewer neighbors, but a tile with **zero** stations in its
  bbox would pass an empty boundary set to `flood_depth` and likely error or
  produce NaN/Inf water levels.
- **DEM nodata handling**: DeltaDTM's native nodata (`-3.4028235e+38`) is
  replaced with meaningful fill values rather than `raster.nodata`: land cells
  without DeltaDTM elevation → `9999.0` (never floods); ocean cells without
  elevation → `0.0` (Aqueduct overwrites these anyway). The output DEM has no
  nodata cells at all.
- **Mask nodata handling**: The DeltaDTM mask has no nodata cells (its nodata
  attr is None), but its 0/1 values can contradict OSM land polygons — notably
  DeltaDTM marks many nearshore ocean cells as land (0). `extract_dem_mask`
  overrides 0/1/255 cells with OSM land polygon (land=0, ocean=1) while
  preserving lake=2 and river=3. The output mask has no nodata cells.
- **Friction nodata handling**: Land-use lookup's invalid marker (`-999.0`) and
  unmapped LULC codes are filled with `flooding.default_friction` (0.001)
  rather than a nodata sentinel, so Aqueduct never encounters a missing friction
  value and `coalesce(friction, 0.001)` in Julia is effectively a no-op.
- **NaN boundary stations**: `load_waterlevel_stations` drops stations with
  NaN water levels — the Aqueduct model cannot handle them as boundary
  conditions.
- **KNN clamp**: `core/src/core.jl`'s `flood_depth` clamps
  `k = min(length(x), k)` so tiles with fewer boundary stations than
  `flooding.knn` don't error.
- **Land-use full-resolution reads**: `_fraction_for_window` reads
  `Copernicus_LandUse.tif` at full resolution per tile (not downsampled) —
  confirmed cheap (~0.1s/tile) and simpler than a windowed/overview approach.
- **Julia JIT concurrency**: `aqueduct.exe`'s LLVM JIT cannot run multiple
  instances concurrently (`resources: aqueduct_runs=1` in
  `rules/simulation.smk`).
- **Stale data (unresolved, low priority)**: tiles processed before the
  `rasters.py` fixes (land polygon layer, DEM fill, mask override) have
  incorrect `dem.tif`, `mask.tif`, and `model_bbox.json`. Since Snakemake
  does not detect source-code changes as invalidating outputs, delete
  `model_outputs/{tile_id}/inputs/model_bbox.json` (and downstream files)
  for affected tiles and re-run. 12 boundary files (tiles 15220, 16430, all
  SLR scenarios) also have stale NaN values predating the
  `load_waterlevel_stations` NaN-drop fix.
- **`flood_depth` mask polarity** (§6 step 7): resolved. `extract_dem_mask`
  writes `0`=land, `1`=ocean, `2`=lake, `3`=river. Aqueduct's `mask != 1`
  correctly restricts flooding to non-ocean cells; `flood[mask .== 1] .= false`
  enforces ocean cells are not flooded.

## 12. Output Structure

```
{model_outputs}/                       # paths.model_outputs, e.g. D:/GFM/model_outputs
├── {tile_id}/
│   ├── inputs/
│   │   ├── tile_geometry.gpkg
│   │   ├── model_bbox.json              # tight domain bbox [minx,miny,maxx,maxy]
│   │   ├── dem.tif
│   │   ├── mask.tif
│   │   ├── friction.tif
│   │   ├── boundaries_{waterlevel_name}.gpkg   # one per SLR scenario
│   │   └── aqueduct_{waterlevel_name}.toml     # one per SLR scenario
│   └── results/
│       └── waterdepth_{waterlevel_name}.tif    # one per SLR scenario
├── skipped_tiles/                       # {tile_id}_{waterlevel_name}.txt, see §6
├── oom_tiles/                           # {tile_id}.txt, see §6
└── _merged/                            # paths.merged_outputs
    ├── flood_count_{waterlevel_name}.tif
    ├── waterdepth_{waterlevel_name}.tif
    └── plots/
        ├── flood_count_{waterlevel_name}.png
        └── waterdepth_{waterlevel_name}.png
```

`{waterlevel_name}` ∈ `waterlevels.names` (`SLR_0`, `SLR_500`, ..., `SLR_3000`
— 7 scenarios at `waterlevels.return_period`="RP100").

## 13. Archive / Legacy Components

These predate or run parallel to the Snakemake tile pipeline and are not part
of the `Snakefile` DAG. Useful for context but shouldn't be extended without
checking whether the equivalent Snakemake/`src/` code (§4-§8) should be
updated instead.

- **`python/preprocessing.py`** + **`python/src/preprocessing_functions.py`**
  — original per-watershed (HydroBASINS `HYBAS_ID`) preprocessing pipeline:
  `mask_creation` (watershed mask minus DEM-invalid areas, optional coastline
  distance filter), `process_datasets`/`clip_data_watershed`/`process_mask`/
  `process_lulc` (clip+reproject+friction per watershed), `waterlevel_scenarios`,
  `write_toml`/`create_base_toml`. Reads root `config.yml`. The `area_name`,
  `hybas_ids` etc. are hardcoded "USER defined" variables at module scope —
  recent commits (`chore: Update area_name and mask_path for ...`) just edit
  these for different regions (England, Myanmar, ...) to regenerate
  per-watershed test outputs.
- **`python/run_gfm.py`** — `run_aqueduct_simulation(exe_path, toml_config_path)`
  + `check_files` (validates `[input]` paths in a TOML exist). README's
  documented "Alternative to run GFM" entry point; superseded for the tile
  pipeline by `src/aqueduct_runner.run_aqueduct` (no `check_files` there).
- **`python/watershed_mask_creation.py`** — thin CLI wrapper around
  `mask_creation` with hardcoded P-drive paths (England example, HYBAS_IDs
  2050048790/2050052960).
- **`python/group_nc_files.py`**, **`python/visualize_tif.py`**,
  **`python/vrt_creation.bat`** — standalone utility scripts (not read in
  detail this session); likely for combining NetCDF outputs, quick raster
  visualization, and building VRT mosaics (e.g. for `deltadtm`/`deltadtm_mask`).
- **`Boundary_conditions_waterlevels/`** — notebook pipeline that produces the
  `COAST-RP_EWL_{return_period}_{waterlevel_name}.nc` files consumed via
  `paths.waterlevel_nc_dir`:
  - `00_select_water_level_dataset.ipynb`
  - `01_retrieve_MDT_correction.ipynb` (uses `mdt_hybrid_cnes_cls22_cmems2020`)
  - `02_get_SLR_fingerprint.ipynb` (uses `ipcc_ar6_slr_projections`,
    `choices.SLR_scenario`/`confidence_level` from root `config.yml`)
  - `03_combine_wl_data_scenarios.ipynb` — presumably combines base water
    levels + MDT correction + SLR fingerprint into the final `SLR_*` scenario
    NetCDFs.
  - `retrieve_boundary_conditions.py` — CDS API helpers:
    `get_GTSM_CDS_tide` (GTSM tidal indicators, 1985-2014 historical),
    `get_GTSM_ERA5_CDS_EWL` (GTSM extreme water level return periods),
    `get_SLR_data` (loads AR6 SLR fingerprint NetCDFs, splits into gauge vs.
    grid locations by `locations < 1e9`).
