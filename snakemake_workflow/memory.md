================================================================================
GFM (Global Flood Model) — Claude Code Reference Memory
Last updated: 2026-08-07
================================================================================

This file describes the CURRENT system — what each part of the codebase does
and why, not how it got there. Design decisions that aren't obvious from the
code alone keep a one-line "why" note; pure development history has been
deliberately left out (git log/history is the place for that).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project:   GFM — Global Flood Model (extends Deltares' Aqueduct Coastal
           Flooding methodology to global, tile-based coastal flood
           simulation).
Location:  C:\Users\Schlu005\GFM   (repo root; also `paths.code_root`)
Data root: D:/GFM                  (`paths.root` — all model I/O lives here)
User:      Julius Schlumberger (julius.schlumberger@gmail.com), researcher.

Purpose: for every tile in a global tile grid, prepare DEM / DEM-mask /
friction / water-level-boundary inputs, run the flood model per (tile,
return_period, SLR scenario), merge per-tile results into spatial chunks,
compute flood-fraction rasters, and run a standalone exposure/adaptation
analysis (population exposed to flooding, with FLOPROS protection standards
and SSP population-growth/adaptation scenarios) on top.

The flood model is a pure-Python port (`src/flood_model.py` +
`src/eikonal.py`) of the original Aqueduct methodology — there is no
Julia/compiled-executable dependency anywhere in the active pipeline. The
port was validated bit-for-bit identical to the original Julia reference
implementation across 26 real tiles (100.000% Jaccard, 0.0m RMSE/mean-error/
max-diff) before the Julia path was removed — see `docs/python_vs_julia_qa.md`
for that validation record.

Repo layout:
  `snakemake_workflow/` — the active pipeline (this file describes it).
    `preparation/` — one-off scripts run BEFORE the Snakemake DAG (tile-grid
      + boundary-condition prep), orchestrated by `run_preparation.py`.
    `rules/`, `scripts/`, `src/` — the Snakemake DAG itself and its shared
      library code.
    `analysis/` — standalone exposure/adaptation analysis, run AFTER the DAG,
      orchestrated by `run_analysis.py`.
    `tests/` — calibration studies and standalone regression/validation
      scripts (not a formal pytest suite) — see section 10.
  `core/` — the original Julia package. No longer used by the active
    pipeline; not part of any current workflow.
  Other repo dirs not part of the active pipeline: `python/` (old
    watershed-based preprocessing), `Boundary_conditions_waterlevels/` (old
    notebook pipeline), `validation/` (a notebook), `old_code_Gundula/`.

Environment: managed via `pixi` (`pixi.toml`/`pixi.lock`). Several
older/alternate env files exist at repo root from earlier setup attempts —
`pixi.toml` is the source of truth.

NOTE: `config.yml` at the **repo root** is a separate, legacy config used
only by old scripts under `python/` — unrelated to the active workflow
config, which lives at `snakemake_workflow/config/config.yml`.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PIPELINE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three stages, run in order:

  0. Preparation (`snakemake_workflow/preparation/run_preparation.py`, NOT a
     Snakemake DAG) — builds the tile grid and boundary-condition scenario
     files that everything else reads.
  1. Snakemake DAG (`Snakefile` at repo root, `include:`s rule files from
     `snakemake_workflow/rules/`) — per-tile preprocessing, simulation, and
     spatial-chunk postprocessing.
  2. Analysis (`snakemake_workflow/analysis/run_analysis.py`, NOT a Snakemake
     DAG) — exposure/adaptation analysis on top of the DAG's flood-fraction
     output.

Snakefile targets (defined at the bottom of the root `Snakefile`):
  - `preprocess`  → all per-tile, per-scenario model inputs.
  - `simulate`    → all flood-model waterdepth runs (`run_aqueduct` rule).
  - `postprocess` → per-chunk flood-fraction rasters (+ optional VRT/plots).
  - `all`         → preprocess + simulate + postprocess (the default target).
  - `generate_aqueduct_jobs` → HPC/SLURM sbatch dispatch (see section 11).

Run with: `snakemake all --cores 4 --resources mem_mb=8000`, or prefer
`python snakemake_workflow/run_pipeline.py` — same invocation wrapped in a
retry loop that auto-splits any tile the solve runs out of memory on (see
section 9's OOM HANDLING) instead of silently giving up on it. Multiple
tiles run fully concurrently on one machine without issue; the `mem_mb`
resource (`aqueduct_runner.estimate_aqueduct_mem_mb`) lets Snakemake's own
scheduler run many small tiles concurrently while throttling around large
ones — `<mem_mb>` should leave headroom below total system RAM for the OS
and other processes (e.g. ~80% of physical RAM).

`_PREPROCESS_OUTPUTS` / `_SIMULATION_OUTPUTS` / `_POSTPROCESS_OUTPUTS` in the
Snakefile are the canonical output lists; any new rule output that should be
part of a stage MUST be added there.

Machine-local overrides: create `snakemake_workflow/config/config_local.yml`
(git-ignored) to override `paths.root`/`paths.code_root` (or `tile_grid.path`)
without touching the committed `config.yml`. Loaded automatically by the
Snakefile if present, and by every standalone script via
`config_utils.load_config()`.

RULE SUMMARY (per tile_id unless noted — see section 5 for full detail):
  compute_geoid_offset_raster — one-time, global EGM2008→GOCO06s geoid-offset
                               field, cached to a GeoTIFF. Always in the DAG.
  extract_tile_geometry     — clip a single tile from `tile_grid.path` → tile_geometry.gpkg
  compute_model_bbox        — tight model-domain bbox → model_bbox.json
  extract_dem                — clip/fill/geoid-correct DEM → dem.tif
  extract_dem_mask           — clip/reproject DEM-validity mask onto DEM grid → mask.tif
  compute_friction            — Copernicus LULC → Manning's-n friction raster → friction.tif
  extract_boundaries        (per return_period × waterlevel_name)
                               — select water-level stations within tile bbox → boundaries_{rp}_{slr}.gpkg
  run_aqueduct               (per tile_id × return_period × waterlevel_name; simulation.smk)
                               — run the flood model in-process; resource `mem_mb` (estimated per tile) → waterdepth_{rp}_{slr}.tif
  merge_chunk                (per chunk_id × return_period × waterlevel_name; postprocessing.smk)
                               — merge per-tile waterdepth within one 5°×5° chunk (temp() outputs)
  compute_flood_fraction_chunk (per chunk_id × rp × slr)
                               — threshold + average-pool waterdepth to ~1 km flood-fraction raster
  build_mosaic_vrt / plot_merged_results (per rp × slr; only if plots.enabled)
                               — GDAL VRT mosaic of all chunks + PNGs
  prepare_exposure_grid_chunk (per chunk_id, once)
                               — cache population + geogunit-ID rasters per chunk
  plot_overlap_diagnostics    (per rp × slr; only if plots.enabled/debug)
                               — up to `n_overlap_locations` diagnostic PNGs of cross-tile overlap
  plot_overlap_continent_diagnostics (per rp × slr; only if plots.enabled)
                               — pools merge_chunk's overlap_minmax .npz across all chunks,
                                 grouped by Natural Earth continent; one hexbin+pie PNG/continent
  generate_aqueduct_jobs      — HPC dispatch (see section 11); not part of `all`.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. WILDCARDS & SCENARIO MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tile_id          — `\d+`, read from `tile_grid.path` at Snakefile parse time
                   into the static list `TILE_IDS`. Run `preparation/
                   run_preparation.py` beforehand to build the tile grid.
                   Tile IDs are assigned in wave order by tile generation
                   (see section 4) — lower IDs tend to be wave-0 (coastal)
                   tiles, though this is a byproduct, not a guarantee; the
                   authoritative wave indicator is the `hop_distance` column
                   on the tile grid itself, not the numeric tile_id.
return_period    — `RP{n}`, n ∈ boundary_conditions.return_periods
                   (currently 5, 10, 25, 50, 100, 250, 500, 1000).
waterlevel_name  — `SLR_{mm}`, the union (ordered, deduped) of
                   boundary_conditions.slr_scenarios and
                   adaptation.slr_intensities (the latter declared separately
                   so they can be listed once and reused as both simulation
                   scenarios and adaptation design intensities).
chunk_id         — `[NS]\d{2}[EW]\d{3}` (e.g. `S20E035`), built
                   PROGRAMMATICALLY in the Snakefile (`_build_chunk_grid`)
                   from the tile grid's total bounds and
                   `postprocessing.chunk_size_deg` (currently 5°) — no
                   external chunk-grid file needed.

So the full simulated matrix is TILE_IDS × RETURN_PERIODS × WATERLEVEL_NAMES
(preprocessing + simulation), and CHUNK_IDS × RETURN_PERIODS ×
WATERLEVEL_NAMES for postprocessing.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. PREPARATION (snakemake_workflow/preparation/*.py — runs BEFORE the DAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Orchestrated by `run_preparation.py`: loads `config.yml` once, runs steps in
sequence (`sync_deltadtm` → `tile_generation` → `boundary_conditions`), each
gated by `preparation.<step>` in config.yml. Step modules expose `run(config)`
and are not standalone entry points (running one directly exits with an error
telling you to use `run_preparation.py`). Usage:
`python run_preparation.py [step ...]` — name specific steps to run only
those, or give none to use `preparation.*` in config.yml.

  sync_deltadtm.py — downloads DeltaDTM v1.1 DEM (per-continent zips) +
    mask tiles directly from 4TU.ResearchData (hardcoded URLs at the top of
    the script) and extracts them into the data catalog's `deltadtm`/
    `deltadtm_mask` source directories. Also downloads 4TU's pre-built DEM
    VRT mosaic and patches it to point at local tile paths, and builds a
    mask VRT locally (no pre-built one exists upstream). Idempotent —
    skips anything already downloaded/extracted.

  tile_generation → `build_tile_manifest.py`, delegating to
    `src/tile_chunking.py`'s 13-stage pipeline (see section 5's
    `tile_chunking.py` entry for the algorithm). Frozen-geometry DAG: this
    stage only depends on the DEM/mask and `tile_generation.elev_threshold_m`
    — never on scenario (COAST-RP station values, SLR, return period) — so
    the tile manifest is computed once and reused for every scenario.
    Writes `tile_grid.path` (`{processed_inputs_dir}/mask/
    domain_tiles_global.gpkg`), with a `hop_distance` column (0 = wave-0,
    has its own real ocean edge; ≥1 = hinterland, seeded from an
    already-simulated lower-hop neighbour — see section 5's `run_aqueduct.py`
    entry). Also writes `river_mouth_seeds.gpkg` and, if
    `tile_generation.write_debug_gpkg` is true, one numbered GeoPackage per
    pipeline stage under `debug_gpkg_dir` for QGIS inspection.

  boundary_conditions → `prepare_boundary_conditions.py`. Generates the
    per-(return_period, SLR) water-level NetCDFs consumed by
    `extract_boundaries.py`: drops Antarctic COAST-RP stations, subtracts
    the AVISO MDT at each station (re-referencing COAST-RP from local MSL to
    the GOCO06s geoid — always applied, matching the DEM's own geoid
    correction, see section 8's VERTICAL DATUM CORRECTION), computes
    per-station IPCC AR6 SLR fingerprints scaled to each target global-mean
    SLR, and combines them: `total_wl = (storm_tide(RP) - MDT) +
    SLR_fingerprint(target_slr)`. Reads its RP/SLR lists from config.yml so
    it always matches the Snakemake wildcard domain.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. SNAKEMAKE DAG — RULES, SCRIPTS, SRC MODULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All `scripts/*.py` are thin Snakemake `script:` shims — logic lives in
`src/`. Listed in execution order.

── Preprocessing (rules/preprocessing.smk) ──────────────────────────────────

  extract_tile_geometry.py   — selects one tile by tile_id, saves geometry.
  compute_model_bbox.py      — tight bbox via `src/rasters.compute_model_bbox`
                                (DeltaDTM valid-cell extent + buffer).
  extract_dem.py             — clips/fills/geoid-corrects the DEM via
                                `src/rasters.extract_dem` (see section 8's DEM
                                GAP-FILL entry for the fill-value logic).
  extract_dem_mask.py        — clips/reprojects the DEM-validity mask via
                                `src/rasters.extract_dem_mask`.
  compute_friction.py        — Manning's-n friction raster via
                                `src/rasters.compute_friction`.
  extract_boundaries.py      — loads one (RP, SLR) water-level NetCDF,
                                selects stations within tile bbox + a search
                                buffer (`boundary_conditions.
                                station_search_buffer_deg`/
                                `station_search_min_size_deg` — whichever
                                gives the larger box), then filters by a
                                coarse ocean-connectivity check
                                (`boundaries.filter_stations_by_ocean_
                                connectivity`) so a station on the "wrong
                                side" of a landmass isn't used. Small/
                                isolated tiles can end up with zero stations
                                even with the buffer — an empty boundaries
                                file is a real, expected outcome (see
                                `run_aqueduct.py` below).

── Simulation (rules/simulation.smk, 1 rule: run_aqueduct) ─────────────────

  run_aqueduct.py — runs `flood_model.flood_depth_dense` in-process via
    `aqueduct_runner.run_aqueduct_python` (`scripts/run_aqueduct_cli.py` is
    the equivalent HPC/sbatch entry point — keep the two in sync).

    Wave-based hinterland forcing: a tile's `hop_distance` (from
    `tile_grid_path`) selects which of two forcing paths this script uses.
    `hop_distance == 0` (wave-0, has its own real ocean edge): the existing
    COAST-RP/IDW path — if `boundaries` has no stations at all, the solve is
    skipped and a real all-ZERO waterdepth is written directly (a
    confidently-computed "definitely dry" result, not an unknown — see
    `write_zero_waterdepth`). `hop_distance >= 1` (hinterland, no ocean edge
    of its own): seeded directly from non-zero cells of already-simulated,
    strictly-lower-hop_distance overlapping tile(s)' own output for the SAME
    scenario (`boundaries.collect_neighbor_wave_seeds`) — if none are
    available yet or none show flooding in the overlap, same real-zero
    fallback. Either skip case is logged to `model_outputs/skipped_tiles/`.

    This script itself just reads whatever's on disk right now and treats a
    not-yet-computed neighbour exactly like "confirmed no flooding there" —
    locally, Snakemake's tile_id-ordered scheduling happens to make a
    not-yet-computed neighbour rare in practice. For genuinely distributed
    (HPC) execution, the ordering guarantee lives one level up, in the HPC
    dispatch script's wave-barrier SLURM dependencies — see section 11.

    If the solve raises `MemoryError` (rare, tile-size-driven), the tile is
    marked in `model_outputs/oom_tiles/{tile_id}.txt` and this job's output
    falls back to an all-NODATA placeholder — a genuine unknown, distinct
    from the confidently-zero fallbacks above, so `merge_chunk` ignores it
    rather than treating it as a real dry result. Once marked, all other
    scenarios for that tile_id skip the solve entirely.

    Each job also checks whether its tile_id's output count just reached
    `n_scenarios_per_tile`; if so it scans `model_outputs/` once and prints a
    running tally of tiles simulated/OOM'd/no-stations/still-running.

── Postprocessing (rules/postprocessing.smk) ────────────────────────────────

  merge_chunk.py               — merges per-tile waterdepth within one chunk
                                (`src/merge.merge_tile_rasters_chunk`, max-
                                combine across overlapping tiles, per-cell
                                winning-tile-id provenance raster); always
                                persists a reservoir-sampled per-cell
                                (min, max) depth-across-overlapping-tiles
                                sample to `overlap_minmax_*.npz` for
                                `plot_overlap_continent_diagnostics.py`.
  compute_flood_fraction_chunk.py — two-pass block-wise reprojection (binary
                                exceedance × domain mask) computing
                                `ff = flooded_fine_px / total_fine_px` per
                                coarse (~1 km) cell — this is what every
                                downstream exposure computation consumes.
  build_mosaic_vrt.py          — GDAL VRT mosaic of all chunks. Only if
                                `postprocessing.plots.enabled`.
  plot_merged_results.py       — plots the VRT-mosaicked waterdepth with an
                                OSM land-polygon background. Only if
                                plots.enabled.
  plot_overlap_diagnostics.py  — grey/red composite of unique vs. overlapping
                                flood cells + tile bbox outlines, for up to
                                `n_overlap_locations` focal tiles. Only if
                                plots.debug.
  plot_overlap_continent_diagnostics.py — loads every chunk's overlap_minmax
                                .npz, looks up each chunk's continent via
                                point-in-polygon against Natural Earth
                                (falling back to nearest continent by
                                centroid distance for offshore chunks),
                                pools mins/maxs per continent, plots one
                                hexbin+pie PNG per continent. Only if
                                plots.enabled.
  prepare_exposure_grid_chunk.py — caches population + geogunit-ID rasters
                                once per chunk (`src/exposure.
                                prepare_exposure_grid_chunk`) — avoids
                                repeating the fetch per RP×SLR scenario.
                                Feeds `analysis/`, not postprocess's own
                                outputs — runs in parallel with the chain
                                above.

── src/ module responsibilities ─────────────────────────────────────────────

  config_utils.py    — `get_data_catalog()` (HydroMT catalog wrapper with
                       retry-wrapped reads); `load_config()` (config.yml +
                       config_local.yml layering + `{root}`/`{code_root}`
                       path expansion, for standalone scripts — mirrors the
                       Snakefile's own merge-then-expand sequence);
                       `retry_transient_io()` (retries a transient I/O error,
                       e.g. a momentary network-share drop); `merged_slr_
                       scenarios()` (dedup+sort union of SLR scenario lists).

  tiles.py           — tile-grid I/O (`load_tile_grid`, `get_tile_geometry`,
                       `save_tile_geometry`) and DeltaDTM mask/DEM mosaic
                       window helpers (`_mosaic_mask_for_trim`,
                       `mosaic_water_fraction_downsampled`,
                       `mosaic_mask_dem_coarse`) used by `tile_chunking.py`
                       and `boundaries.py` to read an arbitrary bbox window
                       across however many native 1°×1° DeltaDTM source
                       tiles it spans (handling their native-resolution
                       mismatch — y is a constant 1 arcsec, x coarsens at
                       high latitude — via nearest-neighbour resampling onto
                       the finest resolution found among the overlapping
                       files).

  tile_chunking.py   — the tile-generation pipeline (13 stages): tile
                       indexing → floodability filtering → greedy maximal-
                       rectangle chunk covering (seeded at river-mouth deltas
                       first) → overlap reduction/repair → connector chunks
                       → exposure-based shaving → redundancy dropping →
                       oversized-chunk splitting → overlap-density capping →
                       hop-distance run-order computation
                       (`compute_run_order`). A chunk with any ocean-mask
                       edge is "wave 0" (`hop_distance = 0`); every other
                       chunk's `hop_distance` is its BFS distance (over the
                       chunk-adjacency graph) to the nearest wave-0 chunk —
                       a chunk may only draw forcing from a neighbour at
                       STRICTLY lower hop_distance. `compute_run_order`'s
                       output order becomes the tile_id sequence (a
                       provenance-set/group_id grouping also identifies
                       fully mutually-independent tile sets, though only
                       `hop_distance` is persisted to the production
                       manifest — see section 11). Uses `ProcessPoolExecutor`
                       with per-worker globals for its exposure/wet-edge
                       classification stages — a real global run takes
                       roughly an hour.

  tile_split.py      — OOM recovery: splits a tile into two smaller,
                       overlapping sub-tiles by choosing whichever axis best
                       balances land-pixel count (never leaving either half
                       with zero land). Called by `run_pipeline.py`'s retry
                       loop, never by the main DAG.

  rasters.py         — the per-tile raster I/O layer, used by nearly every
                       script. `compute_model_bbox`; `extract_dem`/
                       `extract_dem_mask`/`compute_friction`; the int16
                       encode/decode scheme for DEM/friction/waterdepth/
                       waterlevel (see section 8); `save_raster`/
                       `save_nodata_raster`/`save_waterdepth_raster`.

  boundaries.py      — `load_waterlevel_stations` (drops NaN stations);
                       `select_stations_for_tile`; `filter_stations_by_
                       ocean_connectivity` (coarse long-range connectivity
                       pre-filter, rejects a station on the "wrong side" of
                       a landmass); `save_boundary_points`;
                       `collect_neighbor_wave_seeds` (the hop≥1 forcing
                       mechanism — reads a neighbour's own DEM + this same
                       scenario's waterdepth raster, restricted to the
                       target tile's bounds, keeps only genuinely-flooded
                       cells, snaps each to the target grid, combines
                       multiple sources via `np.maximum.at`).

  aqueduct_runner.py — `run_aqueduct_python` (runs `flood_model.
                       flood_depth_dense` in-process and writes its output —
                       shared by `scripts/run_aqueduct.py` and
                       `scripts/run_aqueduct_cli.py` so the two entry points
                       can't drift apart); `estimate_aqueduct_mem_mb`;
                       OOM/skip bookkeeping (`mark_tile_oom`/
                       `tile_marked_oom`/`log_skipped_tile`/
                       `write_zero_waterdepth`/`tile_output_complete`/
                       `print_simulation_progress`).

  flood_model.py     — the flood-depth solve itself. `coastline_mask`
                       (ocean cells adjacent to land — the wave-0 seeding
                       fringe); `_idw_seed_values` (k-nearest-station IDW
                       onto that fringe); `prune_to_coast_connected`
                       (drops flooded regions not connected to the coast/
                       seed set); `flood_depth_dense` (the main entry point
                       — see section 8's EIKONAL SOLVE / OBSTACLE COUPLING
                       entry for the algorithm).

  eikonal.py         — the Fast Sweeping Method eikonal solver
                       (`solve_eikonal_dense`), Numba-JIT'd. Pure numerical
                       kernel, no I/O, no shared state — see section 8.

  merge.py           — `AQUEDUCT_NODATA` (float32 nodata sentinel for
                       merged/analysis-facing rasters); `decode_waterdepth_
                       array`/`_read_waterdepth_patch` (int16-centimetre →
                       float32-metres decode); `merge_tile_rasters_chunk`
                       (block-wise, per-cell max-combine across overlapping
                       tiles + provenance raster + reservoir-sampled
                       min/max overlap diagnostics).

  vertical_datum.py  — EGM2008→GOCO06s geoid conversion for the DEM
                       (`compute_geoid_offset_grid`/`write_geoid_offset_
                       raster`, one-time global spherical-harmonic
                       synthesis cached to a small raster;
                       `sample_geoid_offset`, cheap per-tile resample).

  plotting.py        — `compute_flood_area_km2`; `plot_overlap_continent_
                       diagnostics`; `plot_raster_with_coastlines`.

  protection.py      — `load_geogunit_ids` (nearest-neighbour reprojection
                       of the WRI geogunit-protection-units raster);
                       `GEOGUNIT_INVALID = -1`.

  exposure.py        — `prepare_exposure_grid_chunk` (caches population +
                       geogunit-ID rasters once per chunk).

  exposure_analysis.py — the coarse-resolution exposure math (see section 8's
                       EXPOSURE/ADAPTATION MATH entry).

  population_growth.py — `load_ssp_growth_factors` (Excel → country-name-to-
                       ISO-3 mapping); `interpolate_growth_factor`;
                       `get_geogunit_growth_series`.

  visualization.py   — plotting layer for `analysis/` (burning-ember,
                       timeseries, world-map, growth-matrix parsing).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. CONFIGURATION (snakemake_workflow/config/config.yml)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Grouped by pipeline stage; every section has a `# Used by:` comment pointing
at its consumer(s). Key sections:

  paths            — `root`/`code_root` (override via config_local.yml),
                     `hydromt_data_catalog`, `processed_inputs_dir`.
  sync_deltadtm    — download staging dir.
  vertical_datum_correction — geoid-offset raster cache path. Always applied
                     (no toggle) — see section 8.
  preparation      — which of the 3 preparation steps to run.
  tile_generation  — the tile_chunking.py pipeline's ~20 tunable parameters
                     (thresholds, resolutions, output paths) — each has an
                     inline comment explaining what it controls and, where
                     relevant, how it was tuned.
  tile_grid.path   — the final tile manifest location; both the preparation
                     pipeline's target and the Snakemake DAG's starting point.
  tile_split       — OOM-recovery split parameters (fraction, max depth/
                     retries).
  boundary_conditions — COAST-RP/SLR scenario generation + per-tile station
                     search parameters. `mdt_correction` is always applied
                     (no toggle) — matches the DEM's geoid correction.
  raster_format    — GTiff/zstd output settings shared by every raster write.
  simulation       — `model_outputs` dir; `flooding.*` (resolution, knn,
                     default_friction, `max_rounds` — the eikonal solve's
                     round cap, currently 12, see section 8; `obstacle_
                     coupling.*` — currently `enabled: true`, see section 8's
                     EIKONAL SOLVE entry for what this does); `dem_gap_fill.*`
                     (DEM nodata handling, see section 8); `aqueduct_mem_estimate`
                     (peak-memory model for the `mem_mb` resource).
  hpc              — SLURM dispatch parameters (n_nodes = max node batches
                     PER WAVE, sbatch template) — see section 11 for the
                     wave-barrier design.
  postprocessing   — chunk size, block size, overlap-diagnostic sampling,
                     all plot appearance settings.
  protection, exposure, adaptation, population_growth, analysis,
  visualization    — exposure/adaptation-analysis parameters, see section 8.

`config_hpc.yml.example` — a template for a git-ignored `config_hpc.yml`
override used by `generate_aqueduct_jobs.py` to resolve Linux HPC paths
(`paths.root` + `paths.code_root` only — no separate executable path, see
section 11).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. END-TO-END DATA FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stage 0 (preparation, manual): DeltaDTM sync → tile-grid generation
    (frozen geometry) → boundary-condition NetCDFs (per RP × SLR).
  Stage 1 (Snakemake `preprocess`): per tile — geometry → bbox → DEM (geoid-
    corrected, gap-filled) → mask → friction; per (tile, RP, SLR) — boundary
    stations.
  Stage 2 (Snakemake `simulate`): per (tile, RP, SLR) — flood solve, wave-
    ordered (wave-0 tiles use real station forcing; hop≥1 tiles are seeded
    from an already-simulated lower-hop neighbour).
  Stage 3 (Snakemake `postprocess`): per (chunk, RP, SLR) — merge tiles →
    flood fraction (feeds exposure analysis) → optional VRT/plots.
  Stage 4 (analysis, manual, `analysis/run_analysis.py`): per-chunk exposure
    grids (population + geogunit) → chunk-streamed EAI computation
    (baseline/protect/retreat/avoid) → burning-ember/timeseries/world-map
    plots.

Stage 4's chunk-streaming design: `compute_exposure_analysis.py` never
materializes a global-resolution array. Pass 1 streams every chunk once,
accumulating per-country sums (`exposure_analysis.country_sums`); pass 2
resolves final per-country EAI values and scatters them back onto cells only
where a spatial output is actually needed (`apply_country_shares`/
`scatter_country_values`) — this is what keeps a global run's peak memory
bounded regardless of tile/chunk count.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. KEY DOMAIN LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── INT16 RASTER ENCODING ─────────────────────────────────────────────────────
DEM, friction, waterdepth, and waterlevel are all stored on disk as int16 at
a fixed scale (`rasters.py`'s `encode_*`/`decode_*` functions), not float32:
real flood depths/elevations never approach int16's range at the chosen
scale (cm for DEM/waterdepth, so ±327.67m; a large fixed multiplier for
friction), so this halves file size with no meaningful precision loss and
avoids ever writing an incompressible float mantissa. A dedicated int16
sentinel value marks "not computed" for each (e.g. `WATERDEPTH_NODATA_INT16
= np.iinfo(np.int16).max`), decoded to `merge.AQUEDUCT_NODATA` (float32) by
`merge.decode_waterdepth_array` wherever waterdepth is read for merging/
analysis.

── EIKONAL SOLVE / OBSTACLE COUPLING (src/flood_model.py, src/eikonal.py) ────
`flood_depth_dense` seeds either real/virtual boundary stations (wave-0,
IDW-interpolated onto the coastline fringe via `coastline_mask`) or explicit
seed cells (hop≥1, from `boundaries.collect_neighbor_wave_seeds`) — exactly
one of the two — then solves the eikonal equation over the FULL dense
domain (land, ocean, lake — every cell, no candidate-set restriction) via
Fast Sweeping (`eikonal.solve_eikonal_dense`): friction-weighted geodesic
propagation from the seeds, `round_max_change <= epsilon` (4 sweeps/round)
as the convergence check, capped at `flooding.max_rounds` (currently 12,
i.e. up to 48 sweeps). A cell no seed's influence ever reaches keeps the
solver's default: `t = +99` (waterlevel = `-t` = `-99m`) — comfortably below
any real elevation, deliberately in the OPPOSITE direction from `rasters.
DEM_NODATA_M`/`land_fill_value_m` (`+99m`, "definitely dry") — so an
unreached cell reads as "never flooded," not "flooded at exactly sea
level." (A below-sea-level DEM cell with genuinely zero real forcing — e.g.
a tile with no boundary stations at all — would otherwise spuriously read
as flooded against the old `t=0` default.) `flood(x) = waterlevel(x) >
dem(x)` for non-ocean cells, then `prune_to_coast_connected` drops any
flooded region not connected back to the seed set.

`obstacle_coupling` (`simulation.flooding.obstacle_coupling.enabled`,
currently `true`) mitigates a structural weakness Fast Sweeping shares with
cost-distance flood models: a friction-cheap shortcut through terrain higher
than the locally-attenuated water level can produce an illegitimate high
potential value beyond it. When enabled: a static pre-filter blocks any cell
with `dem > max(station/seed values)` (friction → effectively infinite),
then an outer loop re-solves, blocks any additional cell where
`waterlevel <= dem` after each solve, and repeats until the % of newly-
blocked cells drops below `outer_convergence_pct` or `max_outer_iterations`
is hit. Calibration studies (`snakemake_workflow/tests/`, see section 10)
are ongoing to validate/refine `max_rounds`/`max_outer_iterations` at scale.
`obstacle_coupling=True` works on the hop≥1 explicit-seed path too (2026-08
— the earlier `NotImplementedError` guard was untested-not-unsupported;
`station_values` is aliased to `seed_values`, so the static pre-filter's
threshold just uses the seed values' own max). Validated on a real hop≥1
tile (2494, seeded from neighbour 1482's real solve): coupled/uncoupled
produced identical flooded-cell counts and a small (<1m) depth difference,
consistent with wave-0 tiles' own coupling/non-coupling agreement.

**Bug found and fixed during that validation**: `boundaries.
collect_neighbor_wave_seeds` reconstructed a source cell's absolute water
level as `raw_dem + depth`. `flood_depth_dense` itself saves `depth` as
`waterlevel - effective_dem` (`flood_extent.effective_dem` zeroes DEM on
ocean/lake/river cells, mirroring `core.jl`'s `dem[.!landmask] .= 0.0`), so
for a non-land cell `depth`
already IS the full water level - adding the raw (un-zeroed) DEM on top
double-counted that cell's real terrain elevation. Caught on a real river
cell (raw dem 29.89m, true water level ~5.74m) that produced a seed value
of 35.63m, which propagated into a near-total spurious flood of the seeded
tile. Fixed by using `effective_dem(dem, mask)` instead of the raw DEM;
`collect_neighbor_wave_seeds` now also takes each source's mask path
(`source_paths` is `[(dem_path, mask_path, waterdepth_path), ...]`).

── DEM NODATA / GAP FILL (src/rasters.py's extract_dem) ──────────────────────
Every DEM cell gets a real value — no nodata survives to the encoded output.
The fill value is decided from the DeltaDTM validity mask (reprojected onto
the DEM's own grid), never a separately-sourced land polygon dataset:
  - Ocean(1)/lake(2)/river(3) cells with no DEM reading: filled with the
    GOCO06s-geoid-corrected offset (0.0 if `vertical_datum_correction`'s
    offset happens to be exactly 0 at that point — see VERTICAL DATUM
    CORRECTION below). Ocean cells' own DEM value is functionally inert
    either way — `flood_model.py`'s flood check excludes `mask == ocean_code`
    entirely, and DEM never feeds the eikonal propagation itself (only
    friction does) — but lake/river cells' fill value directly affects
    whether/how much they're reported as flooded, so the geoid-consistent
    value matters there.
  - Land(0) or no-mask-coverage(255) cells with no DEM reading: small,
    isolated gaps (< `min_hard_fill_component_size` connected cells) are IDW-
    interpolated from surrounding valid DeltaDTM elevation; large gaps are
    hard-filled to `land_fill_value_m` (99m — comfortably above DeltaDTM's
    real ≤30m validity envelope, so it reads as implausibly-high,
    never-floods terrain, while fitting the int16-centimetre encoding).
    Investigated 2026-08: the large-gap case is dominated by genuine holes in
    DeltaDTM's own raw source data (confirmed directly against the raw VRT,
    at the exact pixel level, for both a tile-edge case and an interior
    single-pixel-line case) — not a per-tile extraction artifact, and not
    recoverable from an overlapping neighbour tile (every tile reads the
    same single shared source file, so a neighbour touching the same
    geography sees the identical gap). The 99m hard-fill is the correct
    behaviour for this — a large gap really does mean "no data anywhere,"
    so the model should not assume flow there.

── VERTICAL DATUM CORRECTION ─────────────────────────────────────────────────
DEM elevations are geoid-corrected EGM2008 → GOCO06s (`extract_dem`'s
`geoid_offset_raster` param, always supplied — no toggle), and COAST-RP
storm-tide levels have MDT (mean dynamic topography) always subtracted
(`prepare_boundary_conditions.py`'s `mdt_correction`, always applied — no
toggle), re-referencing them from local MSL to the same GOCO06s geoid. Both
sides of the model must be on the same vertical reference for results to be
physically meaningful — this is why the two corrections are wired together
rather than independently toggleable. The synced DEM data
(`sync_deltadtm.py`) is the raw, EGM2008-referenced 4TU.ResearchData
release; the geoid correction is applied on-the-fly per tile in
`extract_dem` (`src/vertical_datum.py`'s cached global offset field,
resampled cheaply per tile).

── OOM HANDLING (src/aqueduct_runner.py, src/tile_split.py) ──────────────────
If the flood solve raises `MemoryError` for a (tile, RP, SLR) job, the tile
is marked in `oom_tiles/{tile_id}.txt` and every other scenario for that
tile skips the solve, writing an all-NODATA placeholder instead (distinct
from the "confidently zero" skip case — see `write_zero_waterdepth`'s
docstring). `merge_chunk` treats an all-nodata tile as "not computed" and
ignores it, rather than treating it as a real dry result. Separately,
`run_pipeline.py`'s retry loop scans for OOM markers after each Snakemake
invocation and splits each marked tile into two smaller overlapping halves
(`tile_split.split_tile`, choosing whichever axis best balances land-pixel
count), then re-runs — up to `tile_split.max_retries` times, each tile split
at most `tile_split.max_depth` times.

── EXPOSURE/ADAPTATION MATH (src/exposure_analysis.py) ───────────────────────
Four scenarios, all built on the same coarse (~1km, population-grid-
resolution) flood-fraction input:
  - baseline: raw exposure, no protection.
  - protect: binary protection — a cell/return-period combination below the
    per-geogunit FLOPROS-snapped protection standard (at a given design SLR)
    is treated as fully protected (`build_protection_fraction`/
    `protect_exposure_grid`).
  - retreat: redistributes floodplain population proportionally to safe
    capacity per country (`compute_retreat_capacity`/`compute_retreat`) —
    capacity depends only on the slr_intensity design grid, computed once
    per slr_intensity and reused across every (RP, SLR) scenario.
  - avoid: redirects FUTURE population growth away from the floodplain
    (`compute_avoid_redirected`/`compute_avoid`) — the redistributed grid
    depends only on (slr_intensity, SSP, year), computed once and reused.

Each scenario produces 3 output files: File 1 (`_base.csv`, discrete
`SLR_{mm}` columns, no growth applied), File 2 (`_growth_matrix.csv`, dense
SLR × growth-rate grid via `interpolate_eai_linear` — the burning-ember
plot's background), File 3 (`_ssp.csv`, one `EAI_{SSP}_{year}` column per
real SSP trajectory, resolved via `resolve_ssp_scenario_eai`). `avoid` has
no File 1 (`_base.csv`) — its growth-rate axis isn't scenario-neutral the
way the other three's is (each growth-rate step genuinely changes WHERE
people are, not just how many), so File 2 is recomputed directly rather
than derived from a shared baseline grid.

`compute_country_eai` does vectorized trapezoidal EAI integration over
return periods, aggregated by ISO country, using precomputed per-cell→
country index arrays (`_build_iso_index`) reused across many calls.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

D:/GFM/model_outputs/{tile_id}/
├── inputs/
│   ├── tile_geometry.gpkg
│   ├── model_bbox.json
│   ├── dem.tif
│   ├── mask.tif
│   ├── friction.tif
│   └── boundaries_{return_period}_{waterlevel_name}.gpkg
└── results/
    └── waterdepth_{return_period}_{waterlevel_name}.tif
D:/GFM/model_outputs/skipped_tiles/{tile_id}_{rp}_{slr}.txt   ← skip log
D:/GFM/model_outputs/oom_tiles/{tile_id}.txt                  ← OOM marker
D:/GFM/model_outputs/run_timings/{tile_id}_{rp}_{slr}.json     ← per-run timing + obstacle_coupling diagnostics
D:/GFM/model_outputs/hpc_jobs/                                 ← generate_aqueduct_jobs.py output (see section 11)

D:/GFM/merged_results/
├── chunks/
│   ├── waterdepth_{chunk}_{rp}_{slr}.tif           (temp unless plots.enabled)
│   ├── provenance_{chunk}_{rp}_{slr}.tif           (winning tile_id per cell; same lifetime as waterdepth)
│   ├── overlap_samples/overlap_minmax_{chunk}_{rp}_{slr}.npz
│   ├── flood_fraction/flood_fraction_{chunk}_{rp}_{slr}.tif   ← kept; feeds analysis/
│   ├── exposure_population_grid_{chunk}.tif
│   └── exposure_geogunit_grid_{chunk}.tif
├── waterdepth_{rp}_{slr}.vrt      (if plots.enabled)
├── exposure/                                    ← written by analysis/compute_exposure_analysis.py
│   ├── exposure_baseline_base.csv               ← File 1: discrete SLR_{mm} cols, no growth
│   ├── exposure_baseline_growth_matrix.csv      ← File 2: dense SLR x growth-rate grid (ember bg)
│   ├── exposure_baseline_ssp.csv                ← File 3: one EAI_{SSP}_{year} col, pre-resolved
│   ├── exposure_protect_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   ├── exposure_retreat_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   └── exposure_avoid_{slr}_growth_matrix.csv / _ssp.csv   ← no _base.csv, see section 8
└── plots/                                        (if plots.enabled)
    ├── waterdepth_{rp}_{slr}.png
    ├── overlap_diagnostics_continents_{rp}_{slr}/
    └── overlap_diagnostics_{rp}_{slr}/            (if plots.debug)

D:/GFM/figures/                                    ← analysis/ visualization output_dir
├── burning_ember/
├── adaptation_bars/
├── timeseries/
└── world_maps/                                    (only if analysis.plot_world_maps)

D:/GFM/processed_inputs/
├── WL_scenarios/COAST-RP_EWL_{rp}_{slr}.nc         ← boundary_conditions output
├── mask/domain_tiles_global.gpkg                    ← tile_grid.path (tile_generation output)
├── mask/river_mouth_seeds.gpkg                       ← tile_generation.river_mouth_seeds_path
└── mask/tile_generation_debug/                        ← per-stage debug GeoPackages (if write_debug_gpkg)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. TESTS / CALIBRATION TOOLING (snakemake_workflow/tests/, repo-root tests/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Not a formal pytest suite — a mix of reusable calibration tooling and
standalone regression scripts, each run directly (`python <script>.py`).

`snakemake_workflow/tests/`:
  - `select_calibration_tiles.py` — picks a geographically/size-stratified
    candidate pool of wave-0 tiles.
  - `test_sweep_budget_calibration.py` — traces the non-coupling solve's
    per-sweep convergence across the candidate pool, to calibrate
    `flooding.max_rounds`.
  - `test_obstacle_coupling_calibration.py` — traces the obstacle_coupling
    outer loop across the candidate pool, to calibrate `max_outer_
    iterations`/`outer_convergence_pct`/inner `max_rounds`.
  - `build_calibration_report_xlsx.py` — post-hoc XLSX report generator over
    a calibration run's per-tile CSVs (currently pinned to one specific
    run's output directory — would need generalizing to reuse across runs).
  - Calibration studies to date: a 7-tile pilot, then a 40-tile study
    (finished, has a full XLSX report), then a 100-tile study (candidate
    selection + partial run, currently paused). None of this has yet driven
    a change to the production defaults already in config.yml
    (`max_rounds: 12`, `max_outer_iterations: 5`) beyond what the 7-tile
    pilot already established — it's ongoing validation at larger scale.
  - `diagnose_large_residual.py`, `_list_tiles_by_size.py` — smaller one-off/
    reusable diagnostic helpers.

Repo-root `tests/` — standalone regression scripts, each validating a
specific current `src/` function against known-good behaviour:
`boundary_station_search_validation/`, `boundary_waterlevel_encoding_
validation/`, `coastline_mask_validation/`, `flood_depth_dense_seed_path_
validation/`, `neighbor_wave_seeding_validation/`, `tile_dedup_validation/`,
`tile_run_order_validation/`, `tile_shave_split_validation/`,
`river_mouth_tile_validation/`. Also holds calibration-study output data
(`obstacle_coupling_calibration_40/` etc., see above).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. HPC/SLURM DISPATCH — WAVE-BARRIER DESIGN (2026-08)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

`rules/hpc_dispatch.smk`'s `generate_aqueduct_jobs` rule + `scripts/
generate_aqueduct_jobs.py` group tiles by `hop_distance` FIRST (the tile
grid's real distribution: hop0 = majority, hop1 = 81, hop2 = 32, hop3 = 17,
hop4 = 6 tiles — max hop 4), then split each wave into up to `hpc.n_nodes`
node batches (whole-tile granularity, capped at the wave's own tile count
so a small wave never gets more batches than tiles). Output per run:
`wave{H}_batch_{id}.sbatch` per (wave, batch), a Linux-path-resolved
`resolved_config.yml`, and a `submit_waves.sh` driver. Each generated job
calls `scripts/run_aqueduct_cli.py` once per (tile, RP, SLR), which shares
its core logic with `run_aqueduct.py`.

Both open gaps from the pre-2026-08 flat-dispatch design are now closed:

1. **Wave/hop_distance awareness.** `submit_waves.sh` submits wave 0 with
   no dependency, captures every job ID via `sbatch --parsable`, then
   submits each later wave with `--dependency=afterany:<every prior-wave
   job id, colon-joined>` — a hop≥1 tile's job can now only start once
   EVERY wave-(hop−1) job has reached a terminal state, not "whichever
   neighbour happened to finish first." `afterany` (not `afterok`) is
   deliberate: one failed/OOM tile in a wave shouldn't block the next
   wave, since that tile's downstream neighbours already fall back to a
   real-zero result when a neighbour is missing or dry.

2. **Julia-era exclusion bug, found during the rewrite.** The old script's
   generation-time exclusion check (skip + write nodata placeholder if
   `boundaries_{scenario}.gpkg` is empty) was applied to EVERY tile, wave-0
   or not. A hinterland (hop≥1) tile has no nearby COAST-RP stations by
   definition, so its boundaries file is always empty — meaning every
   hop≥1 tile was silently excluded from HPC dispatch entirely, never
   scheduled at all. The rewrite makes this check wave-0-only; hop≥1
   tiles are always scheduled, and `run_aqueduct_cli.py`'s own live
   neighbour-seed check (already correct — reads `hop_distance` per tile,
   falls back to a real-zero result if no seeds are available) is what
   actually decides whether a given hop≥1 job produces flooding.

3. **Julia per-machine JIT constraint removed.** `hpc.md` and
   `config_hpc.yml.example` no longer reference `aqueduct_executable`/
   `paths.aqueduct_root` — the Python engine needs only `paths.root` +
   `paths.code_root` on the HPC side. Per-node concurrency (multiple
   tiles at once via `cpus_per_task`, now unblocked since there's no
   single-instance constraint) is NOT implemented — each node still runs
   its batch's jobs sequentially — and remains a legitimate future
   optimization, not a correctness gap.

4. **P:\ network-mount latency (found during the rewrite).** The
   hop≥1 neighbour-availability check in both `run_aqueduct.py` and
   `run_aqueduct_cli.py` used bare `os.path.exists()` on the candidate
   neighbour's dem/mask/waterdepth paths — `os.path.exists` catches
   `OSError` internally and just returns `False`, so it never triggers a
   retry; a momentary P:\ blip on the HPC's network mount would look
   identical to "neighbour genuinely hasn't run yet" and silently produce
   a wrongly-confident real-zero result. Fixed with a new
   `config_utils.path_ready()` helper (`os.stat`, which DOES raise, routed
   through the existing `retry_transient_io`). Both call sites updated.

5. **Failure reporting (`scripts/collect_hpc_failures.py`, new).** Each
   node's `run_job()` already logs a failed `(tile, rp, slr)` to its own
   `logs/wave{H}_batch_{id}_failures.txt` and keeps going (`set -uo
   pipefail`, not `-e`) — this script compiles every batch's failures.txt
   PLUS `model_outputs/skipped_tiles/*.txt` into one `failure_report.csv`,
   keeping genuine job errors (need re-running) separate from
   confidently-resolved skips (no stations / no upstream flooding / OOM —
   real results, not failures).

6. **Bundled preprocessing + dispatch (`scripts/generate_hpc_preprocess_job.py`,
   new, rewritten to be node-parallel 2026-08).** Stage 1 (preprocessing +
   `generate_aqueduct_jobs`) doesn't have to run on the local Windows
   machine, and doesn't have to run on a single node either — preprocessing
   has no wave/hop_distance ordering constraint (every tile is
   independent), so this script splits all tiles' target files
   (dem/mask/friction + every (return_period, waterlevel_name) boundaries
   file - real scale: 2,578 tiles x 48 targets/tile = 123,744 total, at
   this repo's current RP/SLR config) evenly across `hpc.n_nodes` (same
   even-split as simulation batching) and writes one
   `preprocess_batch_{id}.sbatch` per node (`snakemake --cores N $(cat
   ...targets.txt)`, using new `hpc.preprocess_sbatch.*` config) plus a
   `preprocess_batch_{id}_targets.txt` per batch (target lists run into
   the thousands of paths - read at runtime via `$(cat ...)`, not
   inlined). `submit_preprocess_and_dispatch.sh` submits every batch with
   NO dependency between them (fully parallel - unlike simulation waves),
   then submits one more `generate_jobs_and_dispatch.sbatch`
   (`generate_aqueduct_jobs` + `submit_waves.sh`) with
   `--dependency=afterany:<every batch job id>` - the same afterany-join
   pattern already used between simulation waves, just one more phase in
   front of wave 0. Uses `set -euo pipefail` throughout (unlike the
   per-tile `run_job()` wrapper) since each script is either one Snakemake
   invocation across many tiles or one monolithic dispatch step, nothing
   sensible to chain into on failure. Works whether generated from the
   local Windows machine (dual-path resolution via `config_hpc.yml`, same
   mechanism as `generate_aqueduct_jobs.py`) or run natively ON Hydrax
   (`config_hpc.yml` unnecessary in that case — `config_local.yml` alone
   already points at Hydrax's own paths, so local view == Linux view).
   Target file paths are reconstructed directly as f-strings (not via
   `rules.X.output.Y`, since this is a standalone script) - mirrors the
   same approach `generate_aqueduct_jobs.py`'s own local-view boundaries
   check already uses.

7. **Tile-size partition routing (2026-08).** Hydrax's regular partitions
   scale RAM at a fixed 8GB/vCPU (confirmed from the real partition table:
   `1vcpu`=8GB ... `60vcpu`=480GB). Since the pipeline's own
   `aqueduct_mem_estimate` model puts the largest known tile (~207M pixels,
   from the calibration study) at ~10-17GB, `hpc_dispatch.smk` now splits
   each wave's tiles by estimated pixel count (bbox-area proxy, computable
   before preprocessing - `area_deg2 * 3600**2`, same proxy as
   `tests/select_calibration_tiles.py`): tiles at/above
   `hpc.large_tile_pixel_threshold` (default 70M, chosen to leave headroom
   under `sbatch`'s partition RAM) go to `hpc.sbatch_large` (a bigger
   partition, e.g. `16vcpu`=128GB) instead of `hpc.sbatch` (e.g. `1vcpu` -
   by far the most-provisioned partition, so the right default for the
   common case). Batch filenames became `wave{H}_{small|large}_batch_{id}.
   sbatch`; `submit_waves.sh` still only barriers on WAVE, submitting both
   size classes of a wave together (size only picks a batch's partition,
   never its ordering).

Real Deltares Hydrax specifics (confirmed via `henrique/General attention
points for Hydrax users.docx`, 2026-08): partitions are `1vcpu`/`4vcpu`/
`16vcpu`/`24vcpu`/`44vcpu`/`60vcpu`/`gpu`/`test` (30 min cap); `--time` is
MANDATORY (an unset job silently gets a 1-minute default and is hard-killed
at the limit regardless of progress, format `days-hours:minutes:seconds`);
`--account` is NOT required on Hydrax (Henrique's own working sbatch
scripts never set it) — `hpc.sbatch.account` in `config.yml` is now
optional (empty string omits the `#SBATCH --account` line entirely, see
`generate_aqueduct_jobs.py`); env activation is `module load miniconda`
+ `eval "$(conda shell.bash hook)"` + `conda activate gfm`, not a single
module load. The `gfm` conda env itself has so far only been validated on
Windows (via miniforge) — it still needs building+validating fresh on
Hydrax (Linux) directly, since conda environments don't transfer across OS.
