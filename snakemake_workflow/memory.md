================================================================================
GFM (Global Flood Model) — Claude Code Reference Memory
Last updated: 2026-07-08
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project:   GFM — Global Flood Model (extends Deltares' Aqueduct Coastal
           Flooding model to global, tile-based coastal flood simulation)
Location:  C:\Users\Schlu005\GFM   (repo root; also `paths.code_root`)
Data root: D:/GFM                  (`paths.root` — all model I/O lives here)

Purpose: for every tile in an overlapping global tile grid, prepare DEM /
DEM-mask / friction / water-level-boundary inputs, run the compiled Julia
"Aqueduct" flood model per (tile, return_period, SLR scenario), merge the
per-tile results into spatial chunks, and run a standalone exposure/
adaptation analysis (population exposed to flooding, with FLOPROS protection
standards and SSP population-growth/adaptation scenarios) on top.

Current study area: whatever `tile_grid.path` points at in config.yml — this
gets repointed often during active development/testing (regional test grids
under `inputs/mask/`, e.g. `domain_tiles_WesternEurope.gpkg`,
`domain_tiles_TestRegion.gpkg`) before a full global run against
`merge_tiles.py`'s global output. Check config.yml directly rather than
trusting a specific filename recorded here — it goes stale fast.

User: Julius Schlumberger (julius.schlumberger@gmail.com), researcher.

Repo is two halves:
  - `core/`  — Julia package `Aqueduct` (flood model itself: `core/src/core.jl`
    `flood_depth`, config schema in `core/src/config.jl`, CLI entry in
    `core/src/main.jl`). Compiled via PackageCompiler + a Rust CLI wrapper
    (`build/cli/`) into a standalone executable `build/aqueduct/aqueduct.exe`.
    Rebuild with `pixi run build` (→ `julia --project=build build/build.jl`).
  - `snakemake_workflow/` — the active Python/Snakemake pipeline described
    below, which prepares inputs, invokes `aqueduct.exe`, and post-processes
    results. Also contains two standalone (non-Snakemake) subdirectories of
    one-off scripts orchestrated by their own `run_*.py` entry point:
    `preparation/` (tile grid + boundary condition prep, runs BEFORE the DAG)
    and `analysis/` (exposure/adaptation analysis, runs AFTER the DAG).

Other repo dirs (legacy / superseded, not part of the active DAG):
  `python/` — old watershed-based preprocessing + tile-grid utilities.
  `Boundary_conditions_waterlevels/` — old notebook pipeline for boundary-
    condition prep, now consolidated into `snakemake_workflow/preparation/
    prepare_boundary_conditions.py`.
  `validation/` — `aqueduct_floodmaps.ipynb` notebook.
  `old_code_Gundula/` — archived.

Environment: managed via `pixi` (`pixi.toml`/`pixi.lock` — conda deps + Julia
via juliaup + Rust). Several older/alternate env files exist at repo root
(`environment.yml`, `environment_julius.yml`, `requirements.txt`, etc.) from
earlier setup attempts — prefer `pixi.toml` as the source of truth.

NOTE: `config.yml` at the **repo root** is a separate, legacy config (used
only by old scripts under `python/`) — unrelated to the active workflow
config, which lives at `snakemake_workflow/config/config.yml`. Don't confuse
the two when editing configuration.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PIPELINE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Snakefile:    repo root `Snakefile` (NOT inside `snakemake_workflow/`) —
              this is the actual DAG entry point; it `include:`s rule files
              from `snakemake_workflow/rules/`.
Rule files:   snakemake_workflow/rules/{common,preprocessing,simulation,
              postprocessing}.smk
Scripts:      snakemake_workflow/scripts/*.py  (thin `script:` shims)
Shared src:   snakemake_workflow/src/  (see section 5)
Analysis:     snakemake_workflow/analysis/  (standalone, run manually after
              the DAG — see section 7)

Named aggregate targets (defined at the bottom of the root `Snakefile`):
  - `preprocess`  → all per-tile, per-scenario model inputs (rules 1-6 below)
  - `simulate`    → all Aqueduct waterdepth runs (rule `run_aqueduct`)
  - `postprocess` → per-chunk flood-fraction rasters (+ optional VRT/plots)
  - `all`         → preprocess + simulate + postprocess (the **default** target)

Run with: `snakemake all --cores 4 --resources aqueduct_runs=1`
The `aqueduct_runs=1` resource is REQUIRED — Aqueduct's Julia LLVM JIT
crashes (`OutOfMemoryError` / "Unable to allocate section memory!") if more
than one instance runs concurrently; the resource pool caps that rule at 1
concurrent job while preprocessing rules still use the remaining `--cores`.
Prefer `python snakemake_workflow/run_pipeline.py` over calling `snakemake`
directly — it wraps the exact same invocation in a retry loop that
auto-splits any tile Aqueduct runs out of memory on (see KEY DOMAIN LOGIC
NOTES → AUTOMATIC TILE SPLITTING ON OOM) instead of silently giving up on
it. Direct `snakemake` invocation still works for debugging (it just won't
retry/split).

`_PREPROCESS_OUTPUTS` / `_SIMULATION_OUTPUTS` / `_POSTPROCESS_OUTPUTS` in the
Snakefile are the canonical output lists; any new rule output that should be
part of a stage MUST be added there.

Machine-local overrides: create `snakemake_workflow/config/config_local.yml`
(git-ignored, see `config_local.yml.example`) to override `paths.root`/
`paths.code_root` (or `tile_grid.path`) without touching the committed
`config.yml`. Loaded automatically by the Snakefile if present.

RULE SUMMARY (per tile_id unless noted):
  extract_tile_geometry     — clip a single tile from `tile_grid.path` → tile_geometry.gpkg
  compute_model_bbox        — tight model-domain bbox = DeltaDTM valid-cell
                               extent (in tile) + `model_bbox_buffer_arcsec` → model_bbox.json
  extract_dem                — clip/fill DEM to bbox → dem.tif
  extract_dem_mask           — clip/reproject DEM-validity mask onto DEM grid → mask.tif
  compute_friction            — Copernicus LULC → Manning's-n friction raster → friction.tif
  extract_boundaries       (per return_period × waterlevel_name)
                               — select water-level stations within tile bbox → boundaries_{rp}_{slr}.gpkg
  write_aqueduct_config    (per return_period × waterlevel_name)
                               — build per-tile/scenario Aqueduct TOML → aqueduct_{rp}_{slr}.toml
  run_aqueduct              (per tile_id × return_period × waterlevel_name; simulation.smk)
                               — invoke `aqueduct.exe`; resource `aqueduct_runs=1` → waterdepth_{rp}_{slr}.tif
  merge_chunk                (per chunk_id × return_period × waterlevel_name; postprocessing.smk)
                               — merge per-tile waterdepth within one 5°×5° chunk (temp() outputs)
  compute_flood_fraction_chunk (per chunk_id × rp × slr)
                               — threshold + average-pool waterdepth to ~1 km flood-fraction raster
  build_mosaic_vrt / plot_merged_results (per rp × slr; only if plots.enabled)
                               — GDAL VRT mosaic of all chunks + PNGs
  prepare_exposure_grid_chunk (per chunk_id, once)
                               — cache population + geogunit-ID rasters per chunk
  plot_overlap_diagnostics    (per rp × slr; only if plots.enabled)
                               — up to `n_overlap_locations` diagnostic PNGs of cross-tile overlap
  plot_overlap_continent_diagnostics (per rp × slr; only if plots.enabled)
                               — pools merge_chunk's overlap_minmax .npz across all chunks,
                                 grouped by Natural Earth continent; one hexbin+pie PNG/continent


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. WILDCARDS & SCENARIO MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tile_id          — `\d+`, read from `tile_grid.path` at Snakefile parse time
                   into the static list `TILE_IDS` (run `preparation/
                   run_preparation.py` beforehand to build/filter the tile
                   grid to DEM-covered tiles).
return_period    — `RP{n}`, n ∈ boundary_conditions.return_periods
                   = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]  (10 values)
waterlevel_name  — `SLR_{mm}`, the UNION (ordered, deduped via
                   `dict.fromkeys`) of boundary_conditions.slr_scenarios
                   ([SLR_0 … SLR_1400 in 200 mm steps, 8 values]) and
                   adaptation.slr_intensities ([SLR_250, SLR_500, SLR_1000] —
                   declared separately so they can be listed once and reused
                   as both simulation scenarios and adaptation design
                   intensities). Currently 9 unique values are simulated.
chunk_id         — `[NS]\d{2}[EW]\d{3}` (e.g. `S20E035`), built
                   PROGRAMMATICALLY in the Snakefile (`_build_chunk_grid`)
                   from the tile grid's total bounds and
                   `postprocessing.chunk_size_deg` (currently 5°) — no
                   external chunk-grid file needed. Changing `chunk_size_deg`
                   alone is enough to re-tile the postprocessing stage.

So the full simulated matrix is TILE_IDS × RETURN_PERIODS × WATERLEVEL_NAMES
(preprocessing + simulation), and CHUNK_IDS × RETURN_PERIODS ×
WATERLEVEL_NAMES for postprocessing.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. SCRIPTS (chronological: preparation runs first, then the Snakemake DAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4a. PREPARATION SCRIPTS (snakemake_workflow/preparation/*.py — one-off, NOT
in the Snakemake DAG, run manually BEFORE the Snakemake pipeline; grouped in
their own subdirectory the same way analysis/*.py is, since neither is a
Snakemake rule/script but both are still part of the overall pipeline).
Orchestrated by run_preparation.py, mirroring analysis/run_analysis.py:

  run_preparation.py           — single entry point; runs sync_deltadtm →
                                tile_mask_creation → select_tiles →
                                merge_tiles → prepare_boundary_conditions.py
                                in sequence as subprocesses (each step gated
                                by a `preparation.*` config.yml switch,
                                overridable via `--only-tile-grid`/
                                `--only-boundary-conditions`/
                                `--skip-sync-deltadtm`; `--force` forwards to
                                prepare_boundary_conditions.py only — the
                                tile-grid steps have no cache to bypass).
                                `--fail-fast` aborts on first failure,
                                otherwise later steps still run (though the
                                tile-grid chain is a real dependency chain —
                                each step reads the previous one's output
                                file, so a failure there will cascade into a
                                clear "file not found" in the next step
                                rather than being silently skipped). Every
                                step accepts `--config`; run_preparation.py
                                forwards its own `--config` to all five (only
                                prepare_boundary_conditions.py received it
                                before 2026-07 — fixed so a machine-local
                                config_local.yml-style override actually
                                reaches every step, not just the last one).
  sync_deltadtm.py             — syncs/verifies DeltaDTM tiles from a local
                                source dir against a manifest CSV. Must run
                                before tile_mask_creation.py (which reads the
                                deltadtm_mask catalog VRT). Config-driven as
                                of 2026-07 (`sync_deltadtm.source`/`target` in
                                config.yml — previously hardcoded absolute
                                paths in the script itself); `target` is
                                deliberately NOT expressed via `{root}` since
                                the manifest-driven tile cache doesn't
                                necessarily live under `paths.root`.
  tile_mask_creation.py       — step 1 of tile-grid prep, three sub-steps:
                                (a) build_five_deg_grid_from_deltadtm bins
                                every 1°×1° tile referenced by the
                                `deltadtm_mask` VRT into its enclosing 5°
                                cell, keeping cells with ≥1 DeltaDTM tile
                                (860 cells, lat -70°..85° — replaces the
                                DiluviumDEM-derived `five_deg_grid` catalog
                                source, which only reached ±60°, since
                                DiluviumDEM's own 1° tile index doesn't
                                extend that far). (b) filter_grid_by_coastrp
                                additionally requires ≥1 non-Antarctic
                                COAST-RP station within the cell (buffered by
                                the quadrant-scaling overflow, 0.625°) —
                                COAST-RP's raw station coverage is
                                near-global (-84.7°..83.65°), but
                                prepare_boundary_conditions.py drops every
                                station south of boundary_conditions.
                                coastrp_min_lat (config.yml, -60° by
                                default — a practical round-number cutoff,
                                not a cited threshold; see the config
                                comment for the reasoning) before it reaches
                                extract_boundaries.py, so cells that far
                                south would never get boundary forcing
                                regardless of DeltaDTM coverage; this drops
                                them (and any DeltaDTM cell with no nearby
                                station at all, e.g. remote Arctic interior)
                                up front rather than only discovering it via
                                an empty-boundaries skip in run_aqueduct.py.
                                Reads `boundary_conditions.station_x_var`/
                                `station_y_var` from config for the COAST-RP
                                coordinate variable names (fixed 2026-07 —
                                previously hardcoded "station_x_coordinate"/
                                "station_y_coordinate" independently of the
                                config keys extract_boundaries.py already
                                honored, a latent inconsistency risk if those
                                were ever changed). 797 of the 860 DeltaDTM
                                cells survive this filter. Result written to
                                one_off_edits.five_deg_grid_deltadtm.gpkg.
                                (c) splits each surviving 5°×5° cell into 4
                                quadrants, scales each 1.5× around its
                                centroid → overlapping 3.75°×3.75° tiles
                                (tile_id = parent_id*10 + quadrant_id — see
                                section 8's TILE ID SCHEME for why the
                                quadrant digit 0-3 matters beyond naming).
  select_tiles.py             — step 2, TWO sequential filters (both new
                                2026-07 in their current form):
                                (1) filter_tiles_by_dem_mask — keeps tiles
                                with any land/lake/river DeltaDTM mask
                                coverage, as before.
                                (2) filter_tiles_by_exposure — of the
                                survivors, keeps tiles with any positive
                                population (`exposure.population_source`,
                                same WorldPop raster prepare_exposure_grid_
                                chunk.py uses) within the tile bbox; a tile
                                with zero population anywhere (or entirely
                                outside WorldPop's own coverage) has nobody
                                for a flood to expose, so it's dropped too.
                                Prints a summary: how many tiles excluded by
                                each check, out of how many survived the
                                previous one. Discarded tiles from BOTH
                                checks are written as GeoPackages (full rows
                                with geometry — `tiles_without_dem.gpkg`/
                                `tiles_without_exposure.gpkg`, alongside the
                                output), not plain tile_id text files as
                                before, specifically so they can be opened
                                directly in QGIS to visually confirm the
                                filters are excluding the right tiles.
  merge_tiles.py               — step 3, REDESIGNED 2026-07 (see section 8's
                                TILE TRIMMING AND MERGE REDESIGN for the full
                                reasoning/history). Five stages, in order:
                                (1) compute_tile_fractions — ocean_fraction/
                                land_fraction/mask_fraction/nodata_fraction
                                per tile, from DeltaDTM mask files, on each
                                tile's ORIGINAL nominal geometry (cached to
                                `<output>_fractions.csv`, reused on rerun
                                unless deleted).
                                (2) compute_trimmed_geometries — shaves each
                                tile's bbox down to its land-containing
                                extent + `trim_buffer_arcsec` margin; this
                                OVERWRITES `geometry` in place (no separate
                                "trimmed" column) and becomes every tile's
                                real working shape for the rest of the
                                script — the fraction columns from step 1
                                stay tied to the pre-trim nominal footprint
                                and are what step 3 actually judges tiles by.
                                (3) merge_undersized_tiles — TWO phases:
                                water-deficient tiles (ocean_fraction <
                                min_coast_fraction) each unionize with
                                whichever cardinal neighbour (any status) has
                                the HIGHEST ocean_fraction; then land-
                                deficient tiles (land_fraction <
                                min_coast_fraction, excluding land_fraction
                                == 0 tiles dropped outright in a step 0) each
                                unionize with whichever cardinal neighbour
                                has the highest land_fraction. Both phases
                                share one `max_merge_count` cap across both.
                                No `min_mask_fraction` any more (removed —
                                mask_fraction is diagnostic-only now, see
                                section 8).
                                (4) compute_trimmed_geometries again — re-
                                tightens any diagonal slack left by a
                                bbox-union merge.
                                (5) deduplicate_overlapping_tiles — a final
                                pass over ALL tiles (not just ones that were
                                "bad") that consolidates any pair whose
                                (already-trimmed) geometries have
                                intersection-over-union ≥ `dedup_iou_threshold`
                                — catches two tiles that each individually
                                passed the fraction thresholds but happen to
                                trim down to the same physical feature (step
                                3 never compares two already-good tiles to
                                each other). One more trim pass afterward
                                keeps the consolidated geometry tight.
                                Output is what `tile_grid.path` should point
                                at. `--config` supported (previously only
                                `--input`/`--output`).
  prepare_boundary_conditions.py — generates the per-(RP, SLR) water-level
                                NetCDFs consumed by extract_boundaries.py:
                                drops Antarctic COAST-RP stations, computes
                                per-station IPCC AR6 SLR fingerprints scaled to
                                each target global-mean SLR, and combines them:
                                total_wl = storm_tide(RP) + SLR_fingerprint.
                                No MDT correction (see section 8 — COAST-RP and
                                the DeltaDTM v1.1 DEM in use are both already
                                MSL-referenced). Reads its RP/SLR lists straight
                                from config.yml so it always matches the
                                Snakemake wildcard domain. Its `--config` default
                                and run_preparation.py both point at
                                snakemake_workflow/config/config.yml (single
                                config, own local `{root}` _expand() helper since
                                it's a standalone script, not Snakemake-invoked) —
                                the legacy repo-root `config.yml` (paths.
                                processed_inputs/WL_scenarios, choices.*) is no
                                longer read by this script. FIXED 2026-07: the
                                SLR-fingerprint cache filenames
                                (`SLR_base_*.nc`/`SLR_fingerprints_*.nc`) now
                                bake in `slr_scenario`/`confidence_level`
                                (were hardcoded to literal "ssp245"/"2100"
                                regardless of the actual configured scenario
                                — switching `slr_scenario` without `--force`
                                silently reused the stale cache under the old
                                naming). The per-scenario NetCDF writer now
                                builds its filename/variable name from
                                `boundary_conditions.nc_filename_template`/
                                `nc_variable_template` (same templates
                                extract_boundaries.py reads with) instead of
                                a separately hardcoded f-string that happened
                                to produce the same pattern — changing either
                                config template now actually changes what
                                gets written, not just what gets read.


4b. SNAKEMAKE DAG SCRIPTS (snakemake_workflow/scripts/*.py)

All are thin Snakemake `script:` shims — logic lives in `src/`. Listed in
execution order (preprocessing, per tile; then simulation; then
postprocessing, per chunk).

  extract_tile_geometry.py   — selects one tile by tile_id, saves geometry.
  compute_model_bbox.py      — computes tight bbox via src/rasters.compute_model_bbox.
  extract_dem.py             — clips DEM, fills missing cells via
                                src/rasters.extract_dem. REDESIGNED 2026-07:
                                the fill value now comes purely from the
                                DeltaDTM mask (reprojected onto the DEM's own
                                grid) — NOT from OSM land_polygons any more
                                (see section 8's DEM/MASK NODATA FILL entry
                                for the full reasoning and empirical
                                verification behind this change).
  extract_dem_mask.py        — clips/reprojects DEM-validity mask via
                                src/rasters.extract_dem_mask. REDESIGNED
                                2026-07 alongside extract_dem.py — mask cells
                                with no DeltaDTM coverage at all resolve to
                                land(0) directly, no land_polygons involved.
  compute_friction.py        — Manning's-n friction raster via
                                src/rasters.compute_friction.
  extract_boundaries.py      — loads one (RP, SLR) water-level NetCDF,
                                selects stations within tile bbox + a buffer
                                (`boundary_conditions.station_search_buffer_deg`,
                                added 2026-07 alongside the tile-trimming
                                work above — tiles getting tighter shrinks
                                the candidate station pool for Aqueduct's
                                k-nearest-neighbour IDW interpolation
                                (`flooding.knn`), so a buffer keeps that pool
                                from shrinking along with the tile; was a
                                plain unbuffered bbox intersect before).
                                Small/merged/trimmed tiles can still end up
                                with zero stations even with the buffer,
                                which is the trigger for run_aqueduct's skip
                                path.
  write_aqueduct_config.py   — builds/writes the per-tile/scenario TOML
                                (schema must match core/src/config.jl).
  run_aqueduct.py            — runs aqueduct.exe; handles empty-boundaries
                                skip, OOM-marked-tile skip, and
                                OutOfMemoryError → mark + nodata fallback
                                (see section 9).
  merge_chunk.py             — merges per-tile waterdepth within one chunk
                                (src/merge.merge_tile_rasters_chunk); always
                                persists the chunk's reservoir-sampled per-cell
                                (min, max) depth-across-overlapping-tiles pairs
                                to an `overlap_minmax_*.npz` (mins, maxs, bounds)
                                for plot_overlap_continent_diagnostics.py.
  compute_flood_fraction_chunk.py — two-pass block-wise reprojection
                                (binary exceedance × domain mask) computing
                                `ff = flooded_fine_px / total_fine_px` per
                                coarse (~1 km) cell — robust to out-of-domain
                                nodata (see section 9). Runs right after
                                merge_chunk.py on the same chunk; this is
                                what every downstream exposure computation
                                consumes.
  build_mosaic_vrt.py        — GDAL VRT mosaic of all chunks (osgeo.gdal.BuildVRT).
                                Only runs if `postprocessing.plots.enabled`.
  plot_merged_results.py     — plots the VRT-mosaicked flood-count/water-depth
                                with an OSM land-polygon background (still
                                land_polygons' job — only the DEM/mask
                                NODATA FILL logic moved away from it, see
                                above; plotting never used it for that).
                                Only runs if plots.enabled.
  plot_overlap_diagnostics.py — grey/red composite of unique vs. overlapping
                                flood cells + tile bbox outlines, for up to
                                `n_overlap_locations` focal tiles. Only runs
                                if plots.enabled.
  plot_overlap_continent_diagnostics.py — loads every chunk's overlap_minmax
                                .npz, looks up each chunk's continent via a
                                point-in-polygon test against Natural Earth
                                naturalearth_lowres (chunk centroid; falls back
                                to nearest continent by centroid distance if no
                                polygon contains it — coastal chunks can sit
                                just offshore), pools mins/maxs per continent
                                (plain np.concatenate — each chunk is already
                                capped at overlap_corr_max_samples, no further
                                sampling needed), and calls
                                plotting.plot_overlap_continent_diagnostics once
                                per continent. Replaces the old single-scenario
                                per-chunk overlap-correlation plot. Only runs
                                if plots.enabled.
  prepare_exposure_grid_chunk.py — caches population + geogunit-ID rasters
                                once per chunk via src/exposure.prepare_exposure_grid_chunk
                                (avoids repeating the fetch per RP×SLR scenario).
                                Independent of the flood-fraction chain above
                                (feeds analysis/, not postprocess's own
                                outputs) — runs in parallel with it.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. SRC MODULE RESPONSIBILITIES (snakemake_workflow/src/*.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config_utils.py       — get_data_catalog(): thin wrapper around hydromt.DataCatalog.
tiles.py               — tile-grid IO (load/get/save tile geometry);
                         filter_tiles_by_dem_mask (select_tiles.py check 1:
                         land/lake/river DeltaDTM mask coverage);
                         filter_tiles_by_exposure (select_tiles.py check 2,
                         added 2026-07: any positive population within the
                         tile bbox, via the same `exposure.population_source`
                         raster prepare_exposure_grid_chunk.py uses — a
                         lookup failure, e.g. bbox entirely outside WorldPop's
                         coverage, is treated the same as zero exposure);
                         compute_tile_fractions (ocean_fraction/land_fraction/
                         mask_fraction/nodata_fraction per tile from DeltaDTM
                         mask files, on each tile's ORIGINAL nominal
                         footprint — nodata_fraction added 2026-07,
                         diagnostic only); compute_trimmed_bbox /
                         compute_trimmed_geometries (added 2026-07: mosaics a
                         tile's overlapping 1°×1° DeltaDTM mask files into one
                         array, via `_mosaic_mask_for_trim` — a SEPARATE
                         implementation from `_compute_fractions_from_tiles`,
                         not shared, because routing the fraction computation
                         through the same mosaic introduced a ~1e-4
                         discrepancy vs. the original per-file-loop
                         implementation — see section 8 for why — then shaves
                         pure-ocean/nodata edge rows/columns inward until the
                         first row/column with land/lake/river is hit, keeps
                         `trim_buffer_arcsec` of margin beyond that, and
                         handles the empirically-confirmed fact that DeltaDTM
                         mask tiles do NOT all share one native resolution —
                         y is a constant 1 arcsec but x coarsens at high
                         latitude to compensate for longitude convergence —
                         via `out_shape`+nearest-neighbour resampling onto the
                         finest resolution found among the overlapping
                         files); merge_undersized_tiles (REDESIGNED 2026-07,
                         see section 8's TILE TRIMMING AND MERGE REDESIGN —
                         two phases, water-deficient tiles unionize with
                         their highest-ocean_fraction cardinal neighbour, then
                         land-deficient tiles unionize with their highest-
                         land_fraction cardinal neighbour, both sharing one
                         `max_merge_count` cap; operates directly on
                         `geometry`, expected to already be trimmed by the
                         caller — no more `mask_dir`/`adjacency_geometry_col`
                         parameters, no more `min_mask_fraction`);
                         deduplicate_overlapping_tiles (added 2026-07: a
                         separate post-merge pass consolidating any two tiles
                         whose trimmed geometries have intersection-over-
                         union ≥ `dedup_iou_threshold` — catches duplicates
                         merge_undersized_tiles structurally can't, since it
                         only ever compares a deficient tile against
                         candidates, never two already-good tiles against
                         each other).
tile_split.py           — split_depth (counts trailing {8,9} digits in a
                         tile_id to determine how many times it's already
                         been split — see section 8's TILE ID SCHEME);
                         build_split_candidates (both candidate overlapping
                         halves, lat-split and lon-split, at `tile_split.
                         fraction`); count_land_pixels (reads a tile's own
                         already-extracted mask.tif); choose_split (picks
                         whichever axis best balances land-pixel count
                         between the two halves, never an axis leaving either
                         half with zero land — see section 8's AUTOMATIC TILE
                         SPLITTING ON OOM); split_tile (the actual mutation:
                         computes the two children via choose_split, updates
                         tile_grid.path with the new rows in place of the
                         parent, deletes the parent's stale model_outputs
                         directory, and removes its OOM/skip markers — called
                         by run_pipeline.py's retry loop, never by
                         merge_tiles.py).
rasters.py             — compute_model_bbox (buffered DeltaDTM valid-cell
                         extent); extract_dem (fills missing DEM cells purely
                         from the DeltaDTM mask, reprojected onto the DEM's
                         own grid, NOT from OSM land_polygons (removed
                         2026-07 - could misalign with DeltaDTM's own
                         coastline): mask=land(0) or no-coverage(255)→9999,
                         mask=ocean(1)/lake(2)/river(3)→0); extract_dem_mask
                         (reprojects the mask onto the DEM grid; no-coverage
                         cells (255)→land(0), valid 0/1/2/3 kept unchanged);
                         compute_friction (Copernicus LULC → Manning's-n /
                         100 lookup, default_friction fallback); save_raster;
                         save_nodata_raster (AQUEDUCT_NODATA placeholder for
                         skipped/OOM tiles).
boundaries.py           — load_waterlevel_stations (drops NaN stations —
                         Aqueduct can't handle missing boundary values);
                         select_stations_for_tile (plain bbox intersect, no
                         buffer); save_boundary_points.
aqueduct_config.py      — build_aqueduct_config / write_aqueduct_config,
                         TOML schema matching core/src/config.jl.
aqueduct_runner.py      — run_aqueduct (subprocess wrapper); is_oom_error /
                         mark_tile_oom / tile_marked_oom / log_skipped_tile
                         (OOM/skip bookkeeping, see section 9).
merge.py                — AQUEDUCT_NODATA = np.finfo(np.float32).max;
                         merge_tile_rasters_chunk: block-wise, CHUNK-SCOPED
                         merge → flood_count (uint8) + simple valid-count-
                         weighted mean waterdepth, plus a bounded reservoir-
                         sampled per-cell (min, max) depth across ALL
                         overlapping tiles (>=2 valid, any flood status —
                         deliberately includes disagreeing/"ambiguous" cells,
                         unlike the old first-two-flooding-tiles pairing) for
                         plot_overlap_continent_diagnostics.py.
plotting.py             — compute_flood_area_km2 (latitude-corrected pixel
                         area); plot_overlap_continent_diagnostics (per-
                         continent hexbin of min/max depth w/ Pearson r + pie
                         chart of confirmed-flood/confirmed-no-flood/ambiguous
                         cell counts, split on flood_area_threshold_m);
                         plot_raster_with_coastlines (downsampled raster + OSM
                         land background + optional OOM-tile overlay).
protection.py           — load_geogunit_ids: nearest-neighbour reprojection
                         of the WRI geogunit-protection-units raster onto a
                         reference grid; GEOGUNIT_INVALID = -1.
exposure.py              — prepare_exposure_grid_chunk: caches population +
                         geogunit-ID rasters once per chunk.
exposure_analysis.py    — core coarse-resolution exposure math:
                         build_protection_fraction / build_adapt_protection_fraction
                         (per-geogunit FLOPROS-snapped RP → design-event
                         protection flood fraction at a given design SLR;
                         baseline and protect both call this, just with
                         different design SLRs — see KEY DOMAIN LOGIC NOTES),
                         protect_exposure_grid (binary; used for BOTH baseline
                         and protect), compute_retreat_capacity / compute_retreat
                         (redistributes floodplain population proportionally to
                         safe capacity per country; capacity depends only on the
                         slr_intensity design grid, not on (RP, SLR), so callers
                         compute it once per slr_intensity via
                         compute_retreat_capacity and reuse it across every
                         scenario in compute_retreat), similarly
                         compute_avoid_redirected / compute_avoid (redirects
                         future population growth away from the floodplain; the
                         redistributed grid depends only on (slr_intensity, SSP,
                         year), computed once via compute_avoid_redirected and
                         reused across every scenario in compute_avoid — see KEY
                         DOMAIN LOGIC NOTES for the exact formula),
                         _redistribute_by_country (per-country proportional
                         redistribution via np.bincount, shared by both — now
                         composed from two reusable halves, country_sums
                         (per-country sums, called once per chunk in
                         compute_exposure_analysis.py's chunk-streaming
                         pass 1) and apply_country_shares (scatters a
                         finished per-ISO share dict back onto cells, called
                         in pass 2) — split specifically so a chunk-streaming
                         caller can run the two halves as separate passes;
                         _redistribute_by_country's own signature/behavior is
                         unchanged for existing callers), _build_iso_index /
                         scatter_country_values
                         (precomputed per-cell→country index arrays, reused
                         across many calls; scatter_country_values paints one
                         value per ISO onto a (H,W) grid in O(H×W) — used e.g.
                         to build the per-cell SSP growth-factor grid),
                         compute_country_eai (vectorized trapezoidal EAI
                         integration over return periods, aggregated by ISO
                         country; accepts an optional precomputed iso_index),
                         interpolate_eai_linear (linear, NOT PCHIP —
                         np.interp-based per-ISO SLR-curve densification, used
                         for File 2's dense grid), resolve_ssp_scenario_eai
                         (linearly resolves File 1 to a single EAI value per
                         (SSP, year) at that SSP's real trajectory SLR, scaled
                         by real growth — produces File 3 for baseline/protect/
                         retreat), apply_growth_rates_to_eai (generic
                         scenario-neutral growth-rate axis, same scalar for
                         every ISO — produces File 2 for baseline/protect/
                         retreat; avoid's File 2 is genuinely recomputed
                         instead, see KEY DOMAIN LOGIC NOTES).
population_growth.py    — load_ssp_growth_factors (Excel → country-name-to-
                         ISO-3 mapping via Natural Earth + manual overrides);
                         interpolate_growth_factor (linear interp, 5-yr grid);
                         get_geogunit_growth_series; _build_name_to_iso (also
                         reused directly by plot_burning_ember.py, inverted,
                         for ISO→country-name plot titles).
visualization.py         — plotting layer for analysis/: load_growth_matrix_csv
                         / build_growth_matrix_from_grid (parse File 2's
                         EAI_SLR_{mm}_g{pct} columns into a matrix — no
                         interpolation needed, the SLR axis is already dense
                         from file-creation time), ssp_rcp_label ("SSP2" ->
                         "SSP2-RCP4.5", shared by plot_timeseries.py's panel
                         titles and plot_burning_ember's uncertainty-subplot
                         labels), plot_burning_ember (1 + N stacked subplots:
                         the ember heatmap+contours+SSP trajectories on top,
                         then one P17-P83 SLR-uncertainty subplot PER SSP
                         below it — y-axis label = ssp_rcp_label(ssp), y-tick
                         labels = the highlight years. All subplots share one
                         GridSpec column so their plot areas are identical
                         widths; the heatmap's colorbar lives in a SEPARATE
                         dedicated GridSpec column (row 0 only) rather than
                         via `fig.colorbar(im, ax=ax)`, which would shrink
                         only that one subplot and break x-axis alignment
                         with the sharex'd subplots below it — a real bug hit
                         in practice), plot_adaptation_bars,
                         build_geo109_to_iso_lookup, plot_world_map. A handful
                         of functions predate the current File 1/2/3
                         architecture and are unused by any analysis/ script
                         (kept as advertised standalone API / not worth the
                         churn to remove): plot_timeseries (function —
                         analysis/plot_timeseries.py has its own, much
                         simpler _timeseries_figure that reads File 3
                         directly instead), aggregate_to_country,
                         aggregate_to_continent, save_per_country_figures,
                         save_per_continent_figures, build_ssp_growth_for_entity
                         (plot_burning_ember.py has its own local
                         _build_ssp_growth instead).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. CONFIGURATION (snakemake_workflow/config/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config.yml — every value used by a rule or script lives here; nothing is
hardcoded in rules/ or scripts/. Rules read config into a `params:` block;
scripts read `snakemake.params.*` (not `snakemake.config` directly) so each
rule's config dependency is explicit. Dataset identifiers (DEM, land use,
…) are catalog keys from data_catalog_gfm.yml — no separate alias layer.
All `{root}` / `{code_root}` placeholders are expanded once at Snakefile
parse time (`_expand_paths`) so every rule/script gets absolute paths.

Key sections (active values as of 2026-07-08):
  paths.root / code_root:            D:/GFM / C:/Users/Schlu005/GFM
  sync_deltadtm.source / target:     {root}/inputs/DeltaDTM /
                                       D:/GCFM_UU/raw_data/DeltaDTM  (added
                                       2026-07 — sync_deltadtm.py was
                                       previously hardcoded, see section 4a)
  preparation.*:                     sync_deltadtm/tile_mask_creation/
                                       select_tiles/merge_tiles/
                                       boundary_conditions = true (all steps
                                       of preparation/run_preparation.py)
  tile_grid.path:                    whatever regional/global tile grid is
                                       currently under test — see PROJECT
                                       OVERVIEW, changes often
    min_coast_fraction:               0.05  (merge_tiles.py's water-
                                       deficient/land-deficient threshold —
                                       `min_mask_fraction` REMOVED 2026-07,
                                       mask_fraction is diagnostic-only now)
    max_merge_count:                  4  (cap shared across both merge
                                       phases, was per-phase implicitly before)
    trim_buffer_arcsec:               600  (~1/6°, ~18.5km at the equator —
                                       coastal margin kept by compute_trimmed_
                                       bbox beyond a tile's land extent; added
                                       2026-07)
    dedup_iou_threshold:              0.8  (deduplicate_overlapping_tiles —
                                       added 2026-07; confirmed on a regional
                                       test grid that the one genuine
                                       duplicate pair found had IoU=1.000
                                       while the next-highest legitimate
                                       neighbour pair topped out at 0.645, a
                                       wide margin below this default)
  tile_split.fraction / max_depth / max_retries: 0.667 / 2 / 5  (run_pipeline.py's
                                       OOM-recovery retry loop — SIMPLIFIED
                                       2026-07, poll_interval_seconds/
                                       oom_settle_seconds/
                                       graceful_stop_grace_seconds REMOVED
                                       along with the live-watching mechanism
                                       that used them, see section 8)
  boundary_conditions.waterlevel_nc_dir: {root}/processed_inputs/WL_scenarios
    processed_inputs_dir: {root}/processed_inputs  (prepare_boundary_conditions.py's
                            intermediate COAST-RP_preprocessed.nc / SLR_base_*.nc /
                            SLR_fingerprints_all.nc, reused across runs unless --force —
                            cache filenames now bake in slr_scenario/confidence_level,
                            see section 4a)
    slr_scenario / confidence_level: ssp245 / medium  (IPCC AR6 regional SLR
                            projection choice for prepare_boundary_conditions.py)
    return_periods:  1,2,5,10,25,50,100,250,500,1000
    slr_scenarios:   SLR_0 … SLR_1400 (200 mm steps, 8 values)
    coastrp_min_lat: -60  (Antarctic COAST-RP station cutoff — practical
                            round-number choice, not a cited threshold;
                            read by both prepare_boundary_conditions.py and
                            tile_mask_creation.py — see section 8)
    station_search_buffer_deg: 1.0  (added 2026-07 — extract_boundaries.py's
                            select_stations_for_tile buffer around a tile's
                            bbox when picking candidate COAST-RP stations;
                            needed once tile_grid.trim_buffer_arcsec started
                            shrinking tiles, so the candidate pool for
                            Aqueduct's knn=15 IDW interpolation doesn't
                            shrink along with them. Was an unbuffered plain
                            bbox intersect before.)
  simulation.model_outputs:          {root}/model_outputs
    aqueduct_executable:             {code_root}/build/aqueduct/aqueduct.exe
    model_bbox_buffer_arcsec:        3   (~1 DeltaDTM pixel of coastal buffer)
    flooding: resolution=30 m, knn=15 (nearest water-level stations), default_friction=0.001
    input_raster: GTiff (NOT COG - see section 8) / zstd / predictor=3, nodata=-9999
  postprocessing.merged_outputs:     {root}/merged_results
    chunk_size_deg:                  5  (also bounds peak memory of the
                                          per-chunk geogunit lookup, which
                                          scales with chunk_size_deg²)
    block_size:                      2048
    flood_area_threshold_m:          0.05
    plots.enabled:                   false  (VRTs + GeoTIFFs always written;
                                               PNGs/diagnostics are opt-in)
    overlap_corr_max_samples:        50000  (per chunk, for the per-continent
                                               overlap correlation/agreement diagnostics)
  protection.baseline_waterlevel_name: SLR_0   (fixed reference scenario for
                                       prepare_exposure_grid_chunk's chunk
                                       grid metadata; also read into
                                       compute_exposure_analysis.py)
    geogunit_source / flopros_source:  geogunit_protection_units / flopros_protection_standards
    default_rp:                        5  (fallback when FLOPROS has no
                                             Coastal or Riverine standard)
  exposure.population_source:        population (WorldPop 2020, ~1 km, count)
    exceedance_threshold_m:          0.10
  adaptation.slr_intensities:        SLR_250, SLR_500, SLR_1000 (design
                                       intensities for protect/retreat; union'd
                                       into WATERLEVEL_NAMES by the Snakefile)
  population_growth:                 SSP1/2/3/5 growth factors, output years
                                       2025–2100, from an Excel workbook
  analysis.*:                        compute_exposure/plot_burning_ember/
                                       plot_adaptation_bars/plot_timeseries = true;
                                       plot_world_maps = false (slow)
    avoid_worker_memory_budget_gb:    8  (caps AVOID's ProcessPoolExecutor
                                       worker count by shared-payload size —
                                       see section 8)
  visualization.*:                   growth_rates (scenario-neutral ember
                                       y-axis, also the File 2 growth-matrix
                                       axis), slr_interp (File 2's dense SLR
                                       grid, min/max/n_points — linearly
                                       interpolated, not PCHIP), SSP↔RCP code
                                       mapping (126/245/585 — used for both
                                       plot_timeseries.py's panel titles and
                                       extract_slr_trajectories.py), colour
                                       palettes

config_local.yml.example — template for a git-ignored, machine-local
`config_local.yml` overriding `paths.root`/`paths.code_root` (and optionally
`tile_grid.path`); loaded by the Snakefile after config.yml if present.

data_catalog_gfm.yml — HydroMT DataCatalog, `root: "D:/GFM"`. Key entries:
  coast_rp                — COAST-RP storm-tide NetCDF (~22,670 stations
                             after Antarctic removal); boundary-condition source.
                             MSL-referenced — see section 8.
  mdt_hybrid_cnes_cls22_cmems2020 — AVISO MDT. NOT used by
                             prepare_boundary_conditions.py (see section 8);
                             kept in the catalog for other diagnostic uses.
  ipcc_ar6_slr_projections — AR6 regional SLR fingerprints.
  five_deg_grid             — DiluviumDEM-derived 5° tile index; NOT used by
                             tile_mask_creation.py anymore (superseded by a
                             DeltaDTM-mask-derived 5° grid built inline —
                             see the tile_mask_creation.py entry above).
  deltadtm / deltadtm_mask  — DeltaDTM v1.1 (MSL-referenced, GOCO06s geoid +
                             MDT correction — see section 8) 1-arcsec DEM +
                             validity mask (VRTs).
  land_polygons             — OSM land polygons; layer **must** be specified
                             as `land_polygons` (not the default `marine_buffer`).
                             As of 2026-07 used ONLY for plotting coastline
                             reference lines (plot_merged_results.py,
                             plot_overlap_diagnostics.py) — extract_dem/
                             extract_dem_mask no longer use it at all (see
                             section 8's DEM/MASK NODATA FILL entry); don't
                             assume it still feeds DEM-fill decisions.
  land_use                  — Copernicus 100 m LULC.
  geogunit_protection_units / geogunit_country_units / geogunit_country_list
                             — WRI geogunit-107 (sub-national) / geogunit-109
                             (country) rasters + lookup table.
  flopros_protection_standards — FLOPROS coastal/riverine design RPs by geogunit-107.
  ssp_population_growth_factors — Excel workbook of per-country SSP growth factors.
  population                — WorldPop 2020, ~1 km, population COUNT (not density).
  lu_to_roughness_lookup     — Manning's-n lookup by Copernicus LULC class.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. END-TO-END DATA FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 0 — Preparation (manual, one-off, NOT in the Snakemake DAG; all in
  snakemake_workflow/preparation/, run via `python preparation/
  run_preparation.py` or individually — see section 4a for full detail on
  every step below):
  Tile grid prep: sync_deltadtm.py (ensure DeltaDTM tiles are present)
    → tile_mask_creation.py (DeltaDTM+COAST-RP-derived 5° grid → 3.75°
      overlapping tiles)
    → select_tiles.py (two checks: DeltaDTM coverage, then population
      exposure — drops tiles with nobody there for a flood to expose)
    → merge_tiles.py (trim each tile to its land extent + buffer; merge
      water-deficient tiles with their highest-ocean neighbour, then
      land-deficient tiles with their highest-land neighbour; re-trim;
      deduplicate any tiles that converge on the same physical feature)
    → point `tile_grid.path` at the result.
  Boundary condition prep (independent of the tile-grid chain above):
    prepare_boundary_conditions.py combines COAST-RP storm-tide return
    periods + IPCC AR6 SLR fingerprints (no MDT correction — see section 8)
    into one COAST-RP_EWL_{RP}_{SLR}.nc file per (return_period,
    waterlevel_name) under `boundary_conditions.waterlevel_nc_dir`.

Stage 1 — Preprocessing (`snakemake preprocess`, per tile):
  extract tile geometry → compute tight model bbox (DeltaDTM valid-cell
  extent + buffer, clipped to tile) → extract DEM (fill missing cells purely
  from the DeltaDTM mask — land/no-coverage→9999, ocean/lake/river→0; NOT
  OSM land_polygons any more, see section 8) → [in parallel] extract
  DEM-validity mask (same mask-only no-coverage→land(0) rule) and compute
  friction (Copernicus LULC → Manning's-n) → extract boundary points per
  (RP, SLR), buffered by `station_search_buffer_deg` → write Aqueduct TOML
  per (RP, SLR).

Stage 2 — Simulation (`snakemake simulate`):
  run `aqueduct.exe <toml>` per (tile, RP, SLR) — the Julia flood model
  (core/src/core.jl: flood_depth) builds a coastline mask, IDW-interpolates
  boundary water levels onto coastline cells via KNN (`flooding.knn=15`) +
  BallTree, propagates inland over the friction field, floods cells where
  waterlevel > DEM elevation and mask != ocean, filters to components
  connected to the coast, and computes waterdepth. Empty-boundary or
  OOM-marked tiles get a nodata placeholder instead (see section 9).

Stage 3 — Postprocessing (`snakemake postprocess`, chunk-based):
  merge per-tile waterdepth into 5°×5° chunks (merge_chunk, temp() outputs)
  → immediately threshold + average-pool to coarse (~1 km) flood-fraction
  rasters (compute_flood_fraction_chunk — this is what all downstream
  exposure math consumes) → optionally (plots.enabled) build VRT mosaics +
  PNGs + overlap diagnostics. In parallel, prepare_exposure_grid_chunk caches
  population + geogunit rasters once per chunk (not once per scenario).

Stage 4 — Exposure/adaptation analysis (`analysis/`, standalone — run
  manually after the Snakemake DAG via `run_analysis.py`, NOT wired into the
  Snakefile):
  compute_exposure_analysis.py streams every chunk directly (NO global
  mosaic — see CHUNK STREAMING below, this replaced an earlier
  rio_merge-based mosaic architecture that could not complete at global
  scale), snaps each geogunit's FLOPROS protection standard to the nearest
  simulated return period, and computes baseline / protect / retreat / avoid
  Expected Annual Impact (EAI) per country (with SSP population-growth
  scaling applied via population_growth.py) → CSVs → visualized by
  plot_burning_ember.py / plot_adaptation_bars.py / plot_timeseries.py /
  plot_world_map.py.
  Run with: `python snakemake_workflow/analysis/run_analysis.py
  [--config ...] [--expdir ...] [--figdir ...] [--fail-fast]` — switches read
  from config.yml `analysis.*`, overridable via `--only-exposure`/
  `--only-plots`/`--skip-*` CLI flags.

  CHUNK STREAMING (why, and how — see section 8's "CHUNK-STREAMING EXPOSURE
  ANALYSIS" entry for the full derivation): an earlier version mosaicked
  every chunk into ONE dense array per (RP, SLR) via rasterio.merge, holding
  all of them (`flood_fractions: dict[(rp,slr)->ndarray]`) in memory at
  once. Measured empirically: at global scale (real tile domain spans
  -60.6..85.6 lat, full 360° lon, 30 arcsec/px) that's a ~758M-pixel
  bounding box, ~6GB per grid, ~600GB for 10 RPs × 10 SLRs — crashes during
  the mosaic-loading step on a 16.9GB machine, before any EAI computation
  begins, regardless of speed. A per-country batching idea was tried first
  and rejected: measured directly, Mozambique's own bounding box is only
  ~2.7x smaller than a 13-chunk multi-country test region (elongated
  coastal countries don't shrink much when cropped to their bounding
  *rectangle*). The fix: read each ~600×600 px chunk file directly
  (`_load_chunk`/`ChunkData`), never mosaicking — peak memory is one
  chunk's arrays (~1-2 MB) regardless of study-area size or (RP,SLR) count.
  Baseline/protect/retreat/avoid all funnel through `_stream_eai(chunk_ids,
  ..., exposure_fn)`, which loops chunks, calls `exposure_fn` per (RP,SLR),
  and SUMS each chunk's `compute_country_eai` DataFrame (`.add(...,
  fill_value=0.0)`) — exact, not approximate, because `_trapezoid_eai` is
  linear (fixed integration weights): `Σ_chunks trapezoid(chunk_grid) ==
  trapezoid(Σ_chunks chunk_grid)`. Verified via exact-reproduction test
  against the old mosaic path on real 13-chunk data: baseline/retreat/avoid
  all matched to ~1e-9 absolute / ~1e-15 relative (floating-point
  summation-order noise only).
  Retreat/avoid's country-wide redistribution (`_redistribute_by_country`)
  was split into `country_sums`/`apply_country_shares` (src/exposure_
  analysis.py; `_redistribute_by_country` now just composes them,
  unchanged behavior/signature, existing call sites untouched) precisely
  because it already reduced to two per-country SUMS before any cell gets
  its answer — sums are associative, so `pass1_shares` accumulates them
  chunk-by-chunk (one streaming pass per `slr_intensity`) before any
  exposure grid is computed. Retreat's `share_retreat[iso] =
  Σ(apf·population)/Σ(1-apf)` is reused by EVERY avoid task at that
  intensity too: since a growth factor is always a per-country scalar,
  `share_avoid[iso] = share_retreat[iso] * max(0, g_iso-1)` — a cheap dict
  comprehension, no extra chunk streaming needed for avoid's share.
  GOTCHA (cost real debugging time, check this pattern anywhere nodata
  matters): a chunk file's OWN nodata value must be read via
  `rasterio.open(...).read(1, masked=True).filled(fill)`, NOT a plain
  `.read(1)` — `exposure_population_grid_*.tif`'s nodata sentinel is
  `np.finfo(np.float32).min` (≈-3.4e38), a FINITE value, so a naive
  `~np.isfinite(arr)` cleanup (which is all the old mosaic-based code
  needed, since `rio_merge` already silently nodata-masked each source
  before this defensive check ever ran) does NOT catch it on a raw
  single-file read — corrupts every downstream computation for any chunk
  containing it (surfaced immediately by the exact-reproduction test above
  as ~1e39-magnitude garbage in MOZ/ZAF's EAI).
  AVOID worker payload: the ProcessPoolExecutor initializer (
  `_init_avoid_worker`) now sends `chunk_ids`/`chunks_dir`/`flood_frac_dir`
  paths + small dicts (iso_lookup, rp_applied, per-slr_intensity
  `share_retreat`) — each worker calls `_load_chunk` itself per task,
  exactly like the main process does for baseline/protect/retreat.
  Payload dropped from hundreds of GB (old: full pop_grid/geo_ids/
  flood_fractions pickled to every worker) to a few KB — `analysis.
  avoid_worker_memory_budget_gb` capping still exists (harmless) but
  essentially never binds anymore (kept the config knob rather than
  removing it, in case some other constraint reappears).
  KNOWN TRADEOFF (not fixed, acknowledged): each `_load_chunk` call opens
  ~100 small flood_fraction files + population + geogunit; baseline/
  protect/retreat/every avoid task each stream through all chunks
  independently (no cross-scenario chunk-data caching) — trades "hundreds
  of GB of RAM" for "more file-open syscalls." Acceptable: memory
  exhaustion is a hard failure, extra I/O is just slower, and avoid's
  existing ProcessPoolExecutor parallelism already spreads its share of
  this cost across `--cores`. A `ThreadPoolExecutor` over the per-chunk
  loop (I/O-bound, GDAL releases the GIL, same pattern the old
  `_load_flood_fractions` used) would reduce wall-clock further if this
  ever becomes the bottleneck — not implemented, flagged as a possible
  follow-up only.
  MEASURED (real data, this machine): `_load_chunk` costs ~5.2s/chunk
  (13-chunk test domain, 3 repeated passes: 5221/5240/5366 ms/chunk —
  stable, no slowdown trend, confirming this is genuine per-chunk I/O cost
  rather than a leak). Peak RSS stayed flat at ~315-340MB across all 3
  passes (vs the old mosaic path's memory growing linearly with (RP,SLR)
  scenario count until it crashed) — this is the actual verification that
  chunk streaming achieves its goal. At ~1401 chunks globally, one full
  streaming pass ≈ 2 hours; scaling this against the ~46 pass-equivalents
  a full run needs (baseline + 3 protect + 3×2 retreat + 144 avoid tasks
  ÷4 cores) lands close to the earlier ~93-hour hypothetical CPU-time
  estimate for this stage (see the pipeline-efficiency review earlier this
  session) — the exposure-analysis stage is a minor contributor next to
  the ~130-day simulation stage either way, so this tradeoff needs no
  further optimization right now.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. KEY DOMAIN LOGIC NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROTECTION / FLOPROS — BASELINE IS "PROTECT AT SLR_0":
  src/protection.py resolves geogunit IDs (nearest-neighbour reprojection)
  onto the population grid; `default_rp`/`geogunit_source`/`flopros_source`
  are read in analysis/compute_exposure_analysis.py, which snaps each
  geogunit's FLOPROS "Coastal" RP (falling back to "Riverine", then
  `protection.default_rp`=5) to the nearest simulated return period ≥ that
  standard.
  Baseline and protect share the EXACT SAME mechanism
  (`exposure_analysis.protect_exposure_grid`, binary: cells with
  `ff <= adapt_prot_frac` are fully protected -> 0; cells with
  `ff > adapt_prot_frac` are fully exposed -> `ff x population`) — the only
  difference is which design SLR calibrates `adapt_prot_frac`
  (`exposure_analysis.build_adapt_protection_fraction`, i.e.
  `flood_fraction[(RP_applied(geogunit), design_slr)]`). Baseline uses
  `protection.baseline_waterlevel_name` (SLR_0); protect uses each
  `adaptation.slr_intensities` entry (SLR_250/500/1000). This guarantees
  protect_SLR_x can NEVER show more exposure than baseline: its design ff
  (calibrated at a higher SLR) is always >= baseline's (calibrated at
  SLR_0), by construction, for every cell.
  HISTORY: baseline used to compare the scenario's return period directly
  against each cell's FLOPROS return period (SLR-independent — exposure only
  grew via ff's own magnitude increasing at already-exposed RPs, never via
  new RPs becoming exposed), justified at the time by the same "ff ≈ 1.0 at
  all RPs in a narrow domain" concern that motivated protect/retreat/avoid's
  ff-vs-ff design. That RP-only baseline threshold is what let
  protect_SLR_250 show MORE EAI than doing nothing once real SLR passed
  ~300mm (protect's ff-anchored threshold could newly expose frequent,
  high-EAI-weight low-RP events that baseline's SLR-blind RP cutoff would
  never count) — fixed by giving baseline the SAME SLR-anchored threshold
  mechanism as every adaptation measure, not by softening protect's math.
  (An intermediate fix tried softening protect/avoid to continuous partial
  exceedance instead — reverted; binary is the intended design.)

EXPOSURE FORMULAS (exposure_analysis.py):
  - baseline / protect: `protect_exposure_grid(ff, adapt_prot_frac, population)`
    — binary, see above. `population` is always E0 (2020, unscaled).
  - retreat: `compute_retreat_capacity` redistributes `adapt_prot_frac x
    population` proportionally to `(1 - adapt_prot_frac)` capacity across a
    country (unchanged). `compute_retreat` exposure =
    `[(ff - adapt_prot_frac) / (1 - adapt_prot_frac)] x eff_pop` for
    ff > adapt_prot_frac. The `/(1 - adapt_prot_frac)` normalization matters:
    retreat fully vacates the design floodplain, so `eff_pop` is concentrated
    entirely in the remaining `(1 - adapt_prot_frac)` "safe" share of the
    cell, not spread over the whole cell — the exposed share of that
    concentrated population must be measured against the safe area, not the
    whole cell. (An earlier version omitted this normalization.)
  - avoid: `compute_avoid_redirected(adapt_prot_frac, population, growth_factor,
    ...)` computes `avoiding = adapt_prot_frac x population x max(0, g-1)`
    (only growth ABOVE 2020 gets redirected) redistributed the same way as
    retreat's capacity — unchanged. `compute_avoid(ff, adapt_prot_frac,
    population, redirected, growth_factor)` (population = E0, unscaled)
    computes, for ff > adapt_prot_frac:
      `exposure = E0 x ff`                                                   [baseline term]
        `+ [(ff-adapt_prot_frac)/(1-adapt_prot_frac)] x E0x(1-adapt_prot_frac)xg`  [organic-growth term]
        `+ [(ff-adapt_prot_frac)/(1-adapt_prot_frac)] x redirected`          [redirected-growth term]
    First term is EXACTLY protect_exposure_grid's binary formula applied to
    the unscaled 2020 population (avoid provides no MORE protection to the
    existing population than plain protect would). Second and third terms
    both use retreat's normalized marginal exceedance: second applies it to
    the population that already lived outside the design floodplain, grown
    in place by `growth_factor` (that population was never displaced, so it
    keeps growing where it is and stays exposed to the safe area's own
    marginal risk); third applies it to the redirected-growth population.
    HISTORY: an earlier version omitted the organic-growth term entirely
    (only baseline + redirected), which silently treated everyone outside
    the (usually small) design floodplain as permanently safe from any
    exceedance beyond the design standard even as they kept growing —
    understating avoid's exposure, increasingly so at higher growth/longer
    horizons. Fixed by adding growth_factor as a compute_avoid parameter and
    the organic_grown term (using the growth factor `g` directly, per the
    paper's literal equation - NOT `g-1`, unlike Y_hat_k/compute_avoid_
    redirected's redirected-growth term); verified against a hand-derived
    2-cell example matching the paper's Eq. 4 exactly.
    IMPORTANT CAVEAT (confirmed intentional with the paper's author, not a
    bug): because the organic-growth term uses `g` (not `g-1`), avoid's
    exposure NO LONGER reduces to protect's exposure at growth_factor=1 (no
    growth) - Term 1 (E0 x ff, the full 2020 cell population exposed to the
    TOTAL flood fraction) and Term 2 (E0x(1-apf)xg, the outside-floodplain
    population exposed to the MARGINAL fraction) both include that same
    outside-floodplain 2020 population, so at g=1 the two terms are additive
    on top of each other rather than the marginal term vanishing. This
    differs from the retreat/redirected-growth pattern (which uses `g-1` and
    correctly vanishes at no growth) - the asymmetry is deliberate per the
    paper's literal equations, not an oversight, so don't "fix" it to match
    retreat's pattern without checking with the user first.
  All four are genuinely NON-linear in growth for avoid (the redirected term
  has a max(0, g-1) clamp) but exactly linear in a UNIFORM growth scalar for
  baseline/protect/retreat (population scales through every term
  proportionally) — this asymmetry drives the File 2 growth-matrix
  generation strategy below.
  Verified end-to-end against real flood-fraction rasters (trimmed
  RP/SLR/SSP subset, not the full ~5h run): protect_SLR_250 <= baseline and
  protect_SLR_500 <= protect_SLR_250 at every SLR for every country (max
  violation 0.0); retreat_SLR_250 <= protect_SLR_250 everywhere; avoid's
  growth-matrix values are clearly non-linear (g150/g0 ratio ~1.002 at one
  SLR, far from the 2.5x a linear model would predict).

TILE TRIMMING AND MERGE REDESIGN (2026-07, preparation/merge_tiles.py, src/tiles.py):
  MOTIVATION (validated before implementing, not assumed): a controlled
  same-tile A/B padding experiment showed Aqueduct's own runtime scales with
  TOTAL pixel count, not just land pixel count — tile 16083's real rasters
  (12.1M px, 11.3s) padded with a 0.75x-per-side nodata ocean margin to
  75.5M px (6.2x area, IDENTICAL land content/coastline/boundaries) took
  56.8s. This resolved an earlier correlational puzzle across different real
  tiles (16071 had fewer land px than 15013 but more total px and took LESS
  time) as cross-tile flood-pattern-complexity noise, not a real land-px
  relationship. Confirms trimming ocean-only margins is a genuine wall-clock
  win (176,100 Aqueduct invocations at global scale), not just a disk-space
  one.
  SHAVE, NOT BUFFER-AND-CLIP: `compute_trimmed_bbox` mosaics a tile's
  overlapping 1°×1° DeltaDTM mask files (`_mosaic_mask_for_trim`) and peels
  pure-ocean/nodata rows/columns off each of the four edges inward until the
  first row/column containing land/lake/river is hit, then keeps
  `trim_buffer_arcsec` of additional margin beyond that edge — naturally
  bounded by the mosaic's own extent, so the result can never exceed the
  input bbox (no separate clip-back step needed, unlike an earlier
  buffer-then-clip design that was tried and rejected as conceptually wrong
  by the user). Nodata/uncovered pixels are treated the same as ocean for
  shaving purposes (confirmed: not treating them specially is both simpler
  and correct, given DEM/MASK NODATA FILL below).
  RESOLUTION MISMATCH BUG (found and fixed): DeltaDTM mask tiles do NOT all
  share one native resolution — y is a constant 1 arcsec but x coarsens at
  high latitude to compensate for longitude convergence (confirmed
  empirically: 3 arcsec at 76-77°N vs 5 arcsec at 80-81°N for real files).
  `_mosaic_mask_for_trim` handles this via `out_shape`+nearest-neighbour
  resampling onto the finest resolution found among the overlapping files —
  the original naive implementation (assuming one uniform resolution) crashed
  with a shape-mismatch `ValueError` the first time a tile's bbox spanned a
  resolution boundary.
  WHY THE FRACTION COMPUTATION STAYS SEPARATE: an early version tried
  routing `_compute_fractions_from_tiles` through the same mosaic helper as
  the trim function, for code-sharing. Rejected after empirical comparison:
  it introduced a consistent ~1e-4 discrepancy vs. the original per-file-loop
  implementation (likely `rasterio.merge`/window-rounding differences), which
  is small but a real, unexplained bias not worth risking on a computation
  that drives real global merge/drop decisions. `_compute_fractions_from_tiles`
  was reverted to its original, verified-byte-identical implementation;
  `_mosaic_mask_for_trim` is a separate, independent mosaic used only by the
  shave algorithm, where sub-pixel imprecision is inconsequential against the
  600-arcsec margin.
  TWO-PHASE MERGE (replaces the old single "bad tile" concept): every merge
  decision now depends on which side of `min_coast_fraction` a tile fails —
  water-deficient (ocean_fraction too low) tiles unionize with their
  highest-ocean_fraction cardinal neighbour; land-deficient tiles unionize
  with their highest-land_fraction cardinal neighbour. Fraction thresholds
  are always evaluated on each tile's ORIGINAL nominal footprint (confirmed
  requirement — trimming first would trivially inflate a tile's own fraction
  and defeat the threshold's purpose), while adjacency/cardinal-neighbour
  detection uses the TRIMMED geometry (a tile whose real land content doesn't
  reach its nominal shared border with a neighbour shouldn't register as
  adjacent to it). `min_mask_fraction` REMOVED entirely — it never actually
  guarded against the "tile has no water" failure mode it was assumed to
  (that's `min_coast_fraction`'s job), and removing the land-polygon-
  independent classification simplified the merge function down to pure
  geometry/tabular logic (no more `mask_dir` parameter at all).
  BUG FOUND AND FIXED (nominal-footprint dilution): the union-viability check
  for a candidate merge originally re-tested fractions on the union of the
  two tiles' FULL NOMINAL bboxes. For two overlapping-grid tiles whose only
  shared content is one small island, this massively dilutes the fraction
  with ocean from each tile's own unrelated nominal footprint, wrongly
  rejecting an obviously-good merge (confirmed on real data: tile 6961/6970
  both trim down to an IDENTICAL small island bbox, yet their nominal union's
  land_fraction was 0.0018, far below threshold). Fixed: when adjacency was
  determined via a dedicated trimmed-geometry column, the union is accepted
  without re-testing thresholds — trimmed-geometry adjacency already proves
  both tiles have real land (`compute_trimmed_bbox` only ever shrinks a tile
  when it finds some).
  DEDUP PASS (deduplicate_overlapping_tiles, added after the above fix
  revealed a second, structurally different gap): two tiles that are EACH
  individually "good" (pass fraction thresholds on their own) but happen to
  trim down to the same physical feature are never compared to each other by
  the merge loop at all, since it only ever compares a deficient tile against
  candidates. A separate post-merge pass consolidates any pair whose trimmed
  geometries have intersection-over-union ≥ `dedup_iou_threshold` (0.8).
  Uses IoU specifically (not a one-sided overlap ratio) because the
  overlapping-tile-grid design deliberately gives many legitimately-distinct
  neighbouring tiles high one-sided overlap — confirmed on real data: the
  genuine duplicate pair found had IoU=1.000, the next-highest legitimate
  neighbour pair topped out at 0.645.
  Verified end-to-end on regional test grids (Mozambique-area, China-coast):
  a real inland tile (3630, originally 100% land, zero water, would have
  been permanently stuck with "no suitable neighbour" under the old
  algorithm) now correctly merges into a combined tile with ocean_fraction
  ≈0.16 under the new water-deficient phase.

DEM/MASK NODATA FILL (2026-07, src/rasters.extract_dem / extract_dem_mask):
  REMOVED the OSM `land_polygons` dataset from both functions entirely — the
  fill decision for a missing DEM/mask cell now comes purely from the
  DeltaDTM mask itself (reprojected onto the DEM's own grid), not a
  separately-sourced, independently-aligned dataset. Two things drove this,
  investigated in order:
  (1) User's original concern: land_polygons isn't well aligned with
  DeltaDTM's own coastline, so using it to decide "is this missing-DEM cell
  land or ocean" is unreliable at the boundary.
  (2) A DIFFERENT, larger issue found while verifying that concern
  empirically (real crosstab, DeltaDTM's own mask vs. its own DEM raster,
  10 test tiles): the overwhelming majority of DEM nodata is over LAND, not
  ocean (65-98% per tile) — the OPPOSITE of what was initially assumed. This
  is not a data error: DeltaDTM is a coastal-STRIP elevation product, not a
  global DEM, and these nominal 2.5-3.75° tiles extend well inland beyond its
  real measured coverage — most of a tile's inland interior is legitimately
  "land the DEM never measured," not "ocean/gap." (Separately verified: the
  existing lake/river override already worked correctly on real data — every
  lake/river nodata pixel checked got the intended 0.0m fill, not 9999 — so
  that specific mechanism had no bug, contrary to an initial worry that
  rivers/lakes, critical paths for inland flood propagation, might be
  getting wrongly blocked.)
  NEW RULE (both functions, single source of truth): missing DEM cell where
  the DeltaDTM mask says land (0) OR has no coverage at all (255, nodata) →
  `_DEM_LAND_FILL` (9999m, so Julia never floods it); mask says ocean (1),
  lake (2), or river (3) → 0.0m. `extract_dem_mask` applies the matching
  rule to the mask output itself: no-coverage (255) → land (0), valid
  0/1/2/3 kept unchanged. `_land_raster` (the old land_polygons geometry-mask
  helper) is now dead code, removed. `land_polygons` itself is NOT removed
  from the data catalog — it's still used for plotting coastline reference
  lines (plot_merged_results.py, plot_overlap_diagnostics.py), just no
  longer for this.
  Verified empirically on all 10 real test tiles after the change: zero
  leftover nodata in either output, land/ocean/lake/river/no-coverage cells
  all filled exactly as intended (100% match against the expected rule on
  every category, including the "not covered by mask at all" case that
  motivated the fix).

FRICTION:
  src/rasters.compute_friction clips/reprojects Copernicus LULC (mode
  resampling) onto the DEM grid, looks up Manning's-n via
  `lu_to_roughness_lookup`, divides by 100, and fills unclassified/ocean
  cells with `simulation.flooding.default_friction` (0.001) — matching
  Aqueduct core's hardcoded `coalesce(friction, 0.001)`.

BOUNDARY CONDITIONS:
  src/boundaries.load_waterlevel_stations drops NaN-valued stations (Aqueduct
  can't handle missing boundary values). select_stations_for_tile buffers the
  tile bbox by `boundary_conditions.station_search_buffer_deg` (1.0° default)
  before selecting candidate stations — CHANGED 2026-07, was a plain
  unbuffered bbox intersect before. The buffer was added specifically because
  merge_tiles.py's new trim step (see TILE TRIMMING AND MERGE REDESIGN below)
  makes tiles tighter around their coastline, which would otherwise shrink
  the candidate station pool available to Aqueduct's k-nearest-neighbour
  (`flooding.knn=15`) IDW interpolation along with the tile. Small/merged/
  trimmed tiles can still legitimately end up with zero stations even with
  the buffer, which triggers the run_aqueduct.py skip path (see below).
  prepare_boundary_conditions.py guarantees NaN-free output at every stage
  via an explicit fallback (SLR fingerprint → global-mean if no nearby valid
  cell). No MDT correction is applied — see VERTICAL DATUM below.

VERTICAL DATUM (DEM ↔ SURGE FORCING):
  DEM (deltadtm catalog entry) and COAST-RP storm-tide forcing are BOTH
  referenced to local mean sea level (MSL) — verified directly against the
  files in use, not just catalog metadata:
    - DeltaDTM: the tiles actually loaded (`D:\GCFM_UU\raw_data\DeltaDTM\
      deltadtm.vrt`) are named `DeltaDTM_v1_1_..._GOCO06s_MDT.tif` and are
      sourced from the WUR YODA "sea-level-referenced-coastal-elevation"
      vault — the v1.1 MSL-referenced release (Seeger & Minderhoud, 2025),
      re-referenced from the original EGM2008 geoid: geoid converted to
      GOCO06s, then MDT subtracted.
    - COAST-RP: documented as MSL-referenced by its own source paper
      (Dullaart et al. 2021).
  Because both already share the same reference, prepare_boundary_conditions.py
  does NOT apply an MDT correction to COAST-RP — `total_wl = storm_tide(RP)
  + SLR_fingerprint(target_slr)`, no `+ MDT` term. src/rasters.extract_dem
  also does no vertical adjustment of its own; it passes the DeltaDTM v1.1
  elevations through unchanged (only horizontal clip/fill).
  HISTORY: an earlier version of prepare_boundary_conditions.py added
  `+ MDT` to COAST-RP, based on data_catalog_gfm.yml metadata (now corrected)
  that mislabeled the DEM as EGM2008-geoid-referenced (the DOI/version fields
  were stale, copied from the original DeltaDTM v1.0 — Pronk et al. 2024 —
  rather than the v1.1 MSL release actually in use). That would have shifted
  the boundary water levels away from the DEM's frame by the local MDT
  magnitude (up to ~1-2 m in places, spatially varying) instead of aligning
  them. Removed 2026-07-02 after confirming the DEM's actual provenance.

SLR SCENARIO ORDERING (must be sorted for interpolation):
  `list(dict.fromkeys(boundary_conditions.slr_scenarios +
  adaptation.slr_intensities))` does NOT produce a numerically sorted list —
  the adaptation intensities (SLR_250, SLR_500) get appended after the base
  list, giving e.g. [...,SLR_1400, SLR_250, SLR_500]. `config_utils.
  merged_slr_scenarios(bc_cfg, adapt_cfg)` is the canonical, SORTED
  replacement — used everywhere this union is built (compute_exposure_analysis.py,
  prepare_boundary_conditions.py, plot_timeseries.py, plot_burning_ember.py,
  plot_adaptation_bars.py, plot_world_map.py). This matters beyond cosmetics:
  several plotting functions feed the parallel mm-values list positionally
  into `np.interp`/`pchip_interpolate`, both of which require strictly
  increasing x. `np.interp` doesn't error on unsorted input — it silently
  returns nonsense (confirmed: with the unsorted list, SLR_500 sits last, so
  np.interp clamped EVERY target ≥500mm to the SLR_500 value, silently
  discarding all higher-SLR data — this was the root cause of a
  counter-intuitive timeseries plot where weaker adaptation designs appeared
  to outperform stronger ones). `pchip_interpolate` raises `ValueError: x
  must be strictly increasing sequence` on unsorted input instead.
  The Snakefile's own `WATERLEVEL_NAMES = list(dict.fromkeys(...))` is NOT
  affected (only used for Snakemake `expand()` wildcard generation, which is
  order-independent) and was intentionally left as-is.

THREE-FILE OUTPUT ARCHITECTURE (compute_exposure_analysis.py):
  Each scenario (baseline, protect_{slr_int}, retreat_{slr_int} — NOT avoid,
  see below) gets three CSVs, each derived from the one before it:
    File 1 — exposure_{label}_base.csv: per-country EAI at each DISCRETE
      MODELLED SLR scenario (plain `SLR_{mm}`-named columns, straight from
      `compute_country_eai`, no "EAI_" prefix), 2020 population, no growth.
      The basis both other files derive from.
    File 2 — exposure_{label}_growth_matrix.csv: File 1 linearly
      interpolated (`exposure_analysis.interpolate_eai_linear`, np.interp —
      NOT PCHIP) onto the DENSE `visualization.slr_interp` grid (default 100
      points, 0-1500mm), then scaled by the scenario-neutral
      `visualization.growth_rates` axis (same factor for every ISO) via
      `apply_growth_rates_to_eai`. Columns: `EAI_SLR_{mm}_g{pct}`. Feeds
      plot_burning_ember.py's heatmap/contour background — the SLR axis is
      ALREADY dense from file-creation time, so
      `visualization.build_growth_matrix_from_grid` does no further
      interpolation at plot time, just aggregate + reshape.
    File 3 — exposure_{label}_ssp.csv: one column per (SSP, year) —
      `EAI_{SSP}_{year}` — via `exposure_analysis.resolve_ssp_scenario_eai`,
      which for each (SSP, year) linearly interpolates
      `visualization.slr_trajectories_csv` to that year's real global-mean
      SLR (SLR assumed to rise linearly between the two nearest modelled
      trajectory years — the trajectory CSV is on a fixed 10-year grid while
      population_growth.output_years has irregular 5/10/15/25-year gaps),
      linearly interpolates File 1's SLR-EAI curve to that SLR, then scales
      by that country's real SSP/year population growth factor. Both
      interpolation steps happen at FILE-CREATION time — plot_timeseries.py
      and plot_world_map.py just read the column directly, no further
      SLR/growth interpolation needed downstream.
  AVOID has no File 1: its "redirected growth" term is inherently
  growth-dependent (see EXPOSURE FORMULAS above), so there's no
  growth-free curve to serve as a shared basis. It gets File 2 and File 3
  independently, each genuinely recomputed (not derived from a shared base):
    - File 2: `_avoid_growth_matrix_worker_task` recomputes avoid at the
      DISCRETE modelled SLR points for each (slr_intensity, growth_rate)
      pair (growth_rate is a plain float, not a per-country grid —
      compute_avoid_redirected/compute_avoid already broadcast a scalar
      internally, no scatter_country_values needed), THEN linearly
      densifies that discrete-SLR result onto visualization.slr_interp's
      grid via interpolate_eai_linear — same "compute at few points then
      density-interpolate" pattern as baseline/protect/retreat, just
      applied to avoid's OWN (already growth-scaled) curve.
    - File 3: `_avoid_ssp_worker_task` recomputes avoid at the discrete
      modelled SLR points for each (slr_intensity, SSP, year) using a real
      per-country growth grid (scatter_country_values +
      interpolate_growth_factor, unchanged from before), then resolves that
      discrete-SLR result to a SINGLE value at the real trajectory SLR for
      that year (same inline linear-interpolation pattern
      resolve_ssp_scenario_eai uses, just done directly in the worker since
      avoid has no File 1 to call that function on).
    Both worker types run in the SAME ProcessPoolExecutor session, sharing
    the same worker-memory-budget-capped pool and _AVOID_CTX payload.
  Downstream scripts glob file suffixes precisely (`exposure_*_base.csv`,
  `exposure_*_growth_matrix.csv`, `exposure_*_ssp.csv`) — a plain
  `exposure_*.csv` glob would pick up ALL THREE per scenario with
  incompatible schemas.

  Country-level ember plots use a fixed vmax=2,000,000 colorbar (comparable
  across countries); global ember plots keep a dynamic per-figure vmax
  (different order of magnitude). plot_burning_ember.py's SSP trajectory
  overlay lines are independent of File 1/2/3 — built directly from real SSP
  growth factors (population_growth.interpolate_growth_factor) and
  slr_trajectories_csv, same as before.

MERGE:
  src/merge.merge_tile_rasters_chunk is CHUNK-SCOPED (not a single global
  merge) — block-wise reads keep memory bounded, produces a simple
  valid-count-weighted mean waterdepth (not distance-weighted IDW) plus a
  flood_count raster (both still strictly excluding AQUEDUCT_NODATA cells —
  unaffected by the diagnostic-only assumption below), and separately
  harvests a bounded-size (`overlap_corr_max_samples`=50000 per chunk)
  reservoir sample of per-cell (min, max) depth across ALL tiles whose
  FOOTPRINT covers the cell (>=2 covering tiles, regardless of flood status)
  — persisted to overlap_minmax_*.npz by merge_chunk.py. A tile that
  geographically covers a cell but never computed it (AQUEDUCT_NODATA — e.g.
  an OOM/skipped tile) is treated as reporting 0.0 ("no flooding") for THIS
  diagnostic sampling only (tracked via a separate `footprint_count`, not the
  `valid_count` used by the merged raster) — a deliberate "silence means dry"
  assumption, distinct from a tile whose footprint simply doesn't reach that
  cell at all (still excluded, NaN). _PairSamples (the reservoir sampler)
  trims to `overlap_corr_max_samples` immediately once exceeded
  (`_OVERFLOW_FACTOR=1`) rather than letting the buffer grow to a multiple of
  it first, keeping peak buffered memory to roughly one incoming block's
  batch beyond the cap.
  plot_overlap_continent_diagnostics.py later pools these per Natural-Earth
  continent (chunk centroid point-in-polygon) into one hexbin+pie diagnostic
  PNG per continent per (RP, SLR); this replaced the old per-chunk
  plot_overlap_correlation (single tile-pair, single designated scenario,
  and — critically — it silently dropped cells where tiles disagreed about
  flood status, which is exactly what the new pie chart's "ambiguous"
  category needs to see).

FLOOD FRACTION:
  compute_flood_fraction_chunk.py's `compute_flood_fraction` is the bridge
  from fine (~30 m Aqueduct) resolution to coarse (~1 km population-grid)
  resolution used by ALL downstream exposure math. Uses a two-pass
  reprojection trick (binary exceedance-with-nodata-preserved × domain-mask-
  without-nodata) so `ff = flooded_fine_pixels / total_fine_pixels_in_cell`,
  correctly treating out-of-domain pixels as non-flooded rather than
  excluding them from the denominator (an explicit fix — a naive near-zero
  threshold approach would make ff ≈ 1.0 everywhere).

OOM / SKIP HANDLING (src/aqueduct_runner.py, scripts/run_aqueduct.py):
  Distinguishes the tile-size-driven `OutOfMemoryError` (persistent — the
  memory cost of `component_indices` in core/src/core.jl scales with the
  tile's pixel count, not RP/SLR, so it's effectively a per-tile_id failure;
  marks `model_outputs/oom_tiles/{tile_id}.txt`, all future (RP, SLR) combos
  for that tile then skip Aqueduct immediately) from the transient
  concurrency error "LLVM ERROR: Unable to allocate section memory!"
  (handled instead by the `aqueduct_runs=1` resource constraint, not treated
  as OOM). Empty-boundaries tiles and OOM-marked tiles both get an
  all-`AQUEDUCT_NODATA` placeholder written for `waterdepth` (so merge_chunk
  ignores them) and are logged to `model_outputs/skipped_tiles/`. Crucially,
  none of this fails the Snakemake rule — `snakemake` always exits 0 for an
  OOM'd tile, the only signal is the marker file's existence.

  IDEAL TILE SIZE (investigated, decided NOT to increase the default):
  `core/src/core.jl::flood_depth` holds ~8-9 full same-shape arrays
  simultaneously (dem, initial, mask, friction, landmask, coastlinemask,
  flood, labels, waterdepth, plus the fast-sweeping solver's internal
  state) — a computable fixed floor of ~39 bytes/pixel from known dtypes,
  PLUS `component_indices`' genuinely data-dependent cost (scales with how
  fragmented the flood pattern is for that specific tile+scenario, not just
  pixel count — confirmed by reading core.jl directly, not knowable from
  tile geometry alone). Real data from this project: tile 15013 (8426×8427 =
  71.0M px) succeeded; tile 16421 (8427×12528 = 105.6M px, ~1.49x more,
  asymmetric shape from merge_tiles.py's undersized-tile absorption, not the
  base quadrant split) already hit OutOfMemoryError (marker still on disk,
  never split — this project's real runs so far have used direct
  `snakemake` invocations, not run_pipeline.py). The two data points imply
  ~100-150 bytes/pixel in practice. This machine has 16.9GB total RAM
  (confirmed via `Get-CimInstance Win32_ComputerSystem`) and IS the
  production machine (not just this dev/test domain) — reserving headroom
  for the OS and concurrent non-Aqueduct jobs under `--cores`, a
  conservative pixel budget lands close to TODAY'S EXISTING tile size
  (2×2 quadrant split, ~3.75° final, ~71M px for a "regular" tile), not
  meaningfully above it. A whole undivided 5° cell (7.5° final, ~4x pixel
  count, ~284M px) would push the majority of tiles past the observed
  105.6M-px failure point — decided NOT to change `tile_mask_creation.py`'s
  quadrant split; today's default is already close to this machine's
  ceiling. If this ever moves to a machine with substantially more RAM,
  redo this calculation before assuming bigger tiles would help.

  RASTER FORMAT FIX (`simulation.input_raster.driver`): was `"COG"`, changed
  to `"GTiff"` (same zstd/predictor=3/nodata=-9999) — COG's embedded
  overview/pyramid generation is pure overhead for files read exactly once
  by Aqueduct and never served as web map tiles. Verified concretely:
  re-encoding tile 15013's real friction.tif (identical CRS/transform/
  compression/predictor, only driver changed) as GTiff gives 10.76MB vs the
  original COG file's 42.51MB (~4x smaller), pixel-values identical. Real
  disk/IO win (matters for wall-clock — I/O time is part of every one of the
  ~90 (RP×SLR) Aqueduct invocations per tile); does NOT raise Aqueduct's
  memory ceiling (overviews aren't loaded into the in-memory array Aqueduct
  computes with). `src/rasters.py::save_raster` now explicitly sets
  `tiled=True, blockxsize=512, blockysize=512` (COG implied this
  automatically; plain GTiff defaults to striped layout otherwise, which
  would hurt the small-windowed-read access patterns merge.py/tile_split.py
  rely on) — `save_nodata_raster` needs no separate fix since it copies
  `ref.profile` from the (now-fixed) reference DEM, inheriting the same
  tiling settings.

AUTOMATIC TILE SPLITTING ON OOM (run_pipeline.py, src/tile_split.py):
  Since OOM is tile-size-driven, `run_pipeline.py` is a retry-loop
  orchestrator (NOT a Snakemake `checkpoint` — none exists anywhere in this
  codebase; the tile_id wildcard domain is read once, eagerly, at
  Snakefile-parse time from tile_grid.path — `Snakefile:75-77` — so a
  mid-run DAG mutation isn't possible without one).

  SIMPLIFIED 2026-07 (reverted at user's request): originally used a
  live-watching design — `_run_watched`/`_graceful_stop` launched `snakemake`
  via `Popen` (Windows: `CREATE_NEW_PROCESS_GROUP`), polled
  `model_outputs/oom_tiles/*.txt` every `poll_interval_seconds` while it ran,
  and interrupted it early (`CTRL_BREAK_EVENT`/SIGINT, hard-`kill()` fallback
  after `graceful_stop_grace_seconds`) as soon as a new OOM marker appeared,
  rather than waiting for the whole DAG to finish — since Aqueduct is fully
  serialized (`aqueduct_runs=1`), an OOM discovered early in a run would
  otherwise sit idle with its nodata placeholder for the rest of that run.
  Verified working (fake long-sleeping subprocess, ~2s detection-to-interrupt
  latency) but judged too complex for the value: replaced with a plain
  blocking `subprocess.run()` that lets each `snakemake` invocation finish
  normally before scanning for OOM markers. Trades away the early-interrupt
  latency win for much simpler code — no live subprocess/signal management,
  no `--rerun-incomplete` workaround for a hard-kill leaving jobs mid-write
  (nothing hard-kills anymore, so that concern is gone too). The
  `tile_split.poll_interval_seconds`/`oom_settle_seconds`/
  `graceful_stop_grace_seconds` config keys were removed along with it (only
  the now-gone watching loop used them); `fraction`/`max_depth`/`max_retries`
  are unaffected and still used exactly as before.

  After each `snakemake` invocation returns, the retry-loop body scans
  `model_outputs/oom_tiles/*.txt` for tile_ids still present in the CURRENT
  tile_grid.path (guards against stale markers from an already-split tile),
  splits each one via `tile_split.split_tile`, and re-invokes `snakemake`.
  This works with almost no other pipeline changes because two things are
  NOT cached/precomputed the way you'd expect: chunk membership (which
  tiles feed which merge_chunk output) is a live spatial join against
  tile_grid.path recomputed on every Snakemake invocation (`Snakefile:
  100-137`, `_build_chunk_grid`/`_chunk_tile_lookup`), and the entire
  per-tile preprocessing chain (extract_tile_geometry → compute_model_bbox →
  extract_dem/extract_dem_mask/compute_friction → extract_boundaries →
  write_aqueduct_config → run_aqueduct) has no tile_id cache other than
  tile_grid.path itself. So two new rows in tile_grid.path + a fresh
  `snakemake` invocation regenerates the full input chain for the new
  tile_ids automatically, no rule changes needed.

  BUG FIXED: `waterdepth_tiles_for_chunk` (Snakefile, merge_chunk's
  `input.waterdepth_tiles`) used to build its returned paths with
  `os.path.join(config["simulation"]["model_outputs"], tid, "results", ...)`
  — on Windows this produces a MIXED-separator string (forward slashes from
  the `{root}`-substituted prefix, backslashes from os.path.join's
  Windows-native joins for the rest). This is the ONLY place in the whole
  Snakefile where a rule's input is a plain list of path STRINGS from a
  data-dependent lookup (every other cross-rule dependency uses
  `rules.X.output.Y`, a direct object reference needing no regex matching)
  — so it's the only place that forces Snakemake to resolve the producing
  rule (`run_aqueduct`) via regex-matching the string against every rule's
  output pattern, and the mixed-separator string didn't textually match.
  Symptom: `MissingInputException in rule merge_chunk` for EVERY chunk
  (confirmed via `snakemake -n` targeting individual merge_chunk outputs
  directly - reproduced on both a normal chunk and the domain's antimeridian
  edge case below), even though targeting the exact same tile's waterdepth
  file directly as a CLI target built a valid plan fine (direct CLI targets
  don't go through this same input-function regex-matching path). Fixed by
  building the path with an explicit `f"{model_outputs}/{tid}/results/..."`
  (forward slashes only) instead of `os.path.join`. Verified: `snakemake -n`
  on individual chunks resolves correctly post-fix.

  SEPARATE, PRE-EXISTING FINDING (not a bug, a real property of the tile
  grid, surfaced while debugging the above): `tile_mask_creation.py`'s
  `scale_factor`=1.5 overlap buffer means tiles near the antimeridian
  legitimately extend past ±180° longitude (confirmed: domain total_bounds
  = [-180.625, -60.625, 180.625, 85.625]; 11 tiles have minx<-180, 15 have
  maxx>180 - e.g. tile 7590 spans -180.625..-176.875). `_build_chunk_grid`
  treats longitude as a flat linear axis (no antimeridian wraparound), so
  this produces real chunk_ids like "S50W185" (longitude -185, past the
  normal -180..180 range) that a naive reader might mistake for a bug — it
  isn't one on its own (chunk_id's wildcard_constraint regex only checks
  digit COUNT, not value range, so "W185" matches fine), just a
  not-obviously-named consequence of not handling antimeridian wraparound.
  Not fixed (no evidence yet that it causes incorrect results, only that it
  creates a legitimately-named-but-confusing edge-case chunk) - flag for
  follow-up if it ever causes double-counting or missing coverage right at
  the date line.

  Tile ID scheme: existing tile_ids are `parent_5deg_id*10+quadrant_id`
  (quadrant_id always 0-3, tile_mask_creation.py:208) — no original tile_id
  can end in a digit 4-9. A split child's id is `parent_tile_id*10+{8,9}`,
  collision-free by construction, keeps tile_id a plain int everywhere (an
  "a"/"b" string suffix would break `Snakefile:77`'s
  `TILE_IDS=...astype(int)` and every `int(wildcards.tile_id)` call), and
  encodes split depth in the digits themselves — `tile_split.split_depth`
  just counts trailing digits in {8,9} — no separate tracking file needed
  (42→depth 0; 428/429→depth 1; 4288/4289→depth 2).

  Split axis selection (`tile_split.choose_split`): tries both the
  lat-split (north/south halves) and lon-split (west/east halves) at
  `tile_split.fraction` (default 2/3, giving ~33% overlap in the middle —
  this IS the "overlapping" requirement, not a bug), counts land/lake/river
  pixels (mask value in {0,2,3}, same convention as merge_tiles.py) in each
  half from the ALREADY-EXTRACTED `model_outputs/{tile_id}/inputs/mask.tif`
  (DEM/mask extraction always completes before Aqueduct can OOM on a tile),
  rejects an axis if either half has zero land (a purely-oceanic sub-tile
  defeats the point of splitting), and among valid axes picks whichever
  minimizes the land-count difference between the two halves — this matters
  because splitting along the WRONG axis could leave nearly all the
  land/coastal content (and hence memory cost) in one half, not actually
  reducing OOM risk there. Recursion is capped at `tile_split.max_depth`
  (default 2); a tile still OOM-ing at max depth keeps today's
  give-up/nodata-placeholder behaviour, with a warning printed.

  Split children deliberately SKIP `merge_tiles.py`'s min_coast_fraction
  validation — that function is one-shot/pre-DAG only
  (`preparation/merge_tiles.py`, never re-invoked by run_pipeline.py) and
  the land-coverage-balance constraint above already prevents the
  degenerate case it exists to catch.


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
│   ├── boundaries_{return_period}_{waterlevel_name}.gpkg
│   └── aqueduct_{return_period}_{waterlevel_name}.toml
└── results/
    └── waterdepth_{return_period}_{waterlevel_name}.tif
D:/GFM/model_outputs/skipped_tiles/{tile_id}_{rp}_{slr}.txt   ← skip log
D:/GFM/model_outputs/oom_tiles/{tile_id}.txt                  ← OOM marker

D:/GFM/merged_results/
├── chunks/
│   ├── flood_count_{chunk}_{rp}_{slr}.tif        (temp)
│   ├── waterdepth_{chunk}_{rp}_{slr}.tif         (temp)
│   ├── flood_fraction/flood_fraction_{chunk}_{rp}_{slr}.tif   ← kept; feeds analysis/
│   ├── exposure_population_grid_{chunk}.tif
│   └── exposure_geogunit_grid_{chunk}.tif
├── flood_count_{rp}_{slr}.vrt / waterdepth_{rp}_{slr}.vrt      (if plots.enabled)
├── exposure/                                    ← written by analysis/compute_exposure_analysis.py
│   ├── exposure_baseline_base.csv               ← File 1: discrete SLR_{mm} cols, no growth
│   ├── exposure_baseline_growth_matrix.csv      ← File 2: dense SLR x growth-rate grid (ember bg)
│   ├── exposure_baseline_ssp.csv                ← File 3: one EAI_{SSP}_{year} col, pre-resolved
│   ├── exposure_protect_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   ├── exposure_retreat_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   └── exposure_avoid_{slr}_growth_matrix.csv / _ssp.csv   ← no _base.csv, see KEY DOMAIN LOGIC NOTES
└── plots/                                        (if plots.enabled)
    ├── flood_count_{rp}_{slr}.png / waterdepth_{rp}_{slr}.png
    ├── overlap_diagnostics_{rp}_{slr}/
    └── correlation/overlap_correlation_{chunk}_{rp}_{slr}.png

D:/GFM/figures/                                    ← analysis/ visualization output_dir
├── burning_ember/
├── adaptation_bars/
├── timeseries/
└── world_maps/                                    (only if analysis.plot_world_maps)

D:/GFM/processed_inputs/WL_scenarios/COAST-RP_EWL_{rp}_{slr}.nc  ← boundary_conditions
D:/GFM/inputs/mask/                                              ← preparation/ tile-grid outputs
├── five_deg_grid_deltadtm.gpkg                    ← tile_mask_creation.py step (a)/(b)
├── tiles_2_5deg_with_overlap.gpkg                  ← tile_mask_creation.py step (c), pre-filter
├── tiles_2_5deg_with_overlap_clean.gpkg            ← select_tiles.py output (both checks passed)
├── tiles_without_dem.gpkg / tiles_without_exposure.gpkg  ← select_tiles.py discards, per check
│                                                       (GPKGs since 2026-07, for visual QGIS
│                                                        confirmation — were plain tile_id .txt before)
├── <name>_fractions.csv                            ← merge_tiles.py's per-tile ocean/land/mask/
│                                                       nodata fractions, cached (reused unless deleted)
└── <name>.gpkg                                      ← merge_tiles.py output = active tile_grid.path
                                                        (filename varies — currently under active
                                                        regional testing, see PROJECT OVERVIEW)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. KNOWN ISSUES / STALE DOCS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

`Code_memory.md` at the REPO ROOT is a prior Claude-authored reference doc
(dated ~2026-06-17) but is now MATERIALLY OUT OF DATE relative to the current
tree: `merge_results.py` (old global merge) was replaced by `merge_chunk.py`
+ `compute_flood_fraction_chunk.py`; `preprocessing_data.yml` was deleted;
the whole `protection.py`/`exposure.py`/`exposure_analysis.py`/
`population_growth.py`/`visualization.py` + `analysis/` subsystem was added
after that doc was written. Prefer THIS file (and the current source tree)
over Code_memory.md where they conflict.

`config.yml` at repo root vs. `snakemake_workflow/config/config.yml`: two
DIFFERENT files with the same name — root one is legacy (`python/` scripts
only), the workflow one is authoritative.

`aqueduct_runs=1` resource: forgetting `--resources aqueduct_runs=1` on the
snakemake command line lets multiple Aqueduct/Julia instances race for JIT
memory allocation and crash — always include it for `simulate`/`all` runs.

`analysis/` is NOT wired into the Snakemake DAG — after `snakemake all`
completes, `run_analysis.py` must be run manually as a separate step.

OSM land polygons: same gotcha as sibling projects — the file has both a
`land_polygons` and (implicitly, per hydromt catalog convention) other
layers; always read the `land_polygons` layer explicitly, never rely on the
default first layer. As of 2026-07 this dataset is used ONLY by the
plotting scripts (plot_merged_results.py, plot_overlap_diagnostics.py) —
extract_dem/extract_dem_mask no longer touch it at all, see section 8's
DEM/MASK NODATA FILL entry. Don't assume a `land_polygons` reference
anywhere still means DEM-fill logic.

`postprocessing.chunk_size_deg` was lowered 10° → 5° after repeated OOM
crashes in the geogunit lookup inside `prepare_exposure_grid_chunk`/
`protection.load_geogunit_ids`, which resolves the WRI geogunit raster onto
the FULL chunk extent at once (not block-wise) — that step's memory scales
with chunk_size_deg².
