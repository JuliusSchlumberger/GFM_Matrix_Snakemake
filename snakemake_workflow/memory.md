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

Run with: `snakemake all --cores 4 --resources mem_mb=8000`
Aqueduct instances run fine concurrently on one machine — a controlled test
(2, 6, then 5 simultaneous instances up to 140M pixels) never reproduced the
previously-assumed `LLVM ERROR: Unable to allocate section memory!` JIT
crash. The real local constraint is ordinary system memory: `run_aqueduct`'s
`mem_mb` resource (`aqueduct_runner.estimate_aqueduct_mem_mb`, calibrated
from real tiles) lets Snakemake run many small tiles concurrently while
throttling around large ones, bounded by `--resources mem_mb=<N>` (leave
headroom below total system RAM for the OS/other processes).
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
`paths.code_root`/`paths.aqueduct_root` (or `tile_grid.path`) without
touching the committed `config.yml`. Loaded automatically by the Snakefile
if present, and by every standalone script via `config_utils.load_config()`.

RULE SUMMARY (per tile_id unless noted):
  compute_geoid_offset_raster (NOT per tile_id — one-time, only in the DAG
                               when vertical_datum_correction.enabled)
                               — pyshtools EGM2008/GOCO06s spherical-harmonic
                               synthesis → cached geoid-offset GeoTIFF; see
                               section 8's VERTICAL DATUM CORRECTION
  extract_tile_geometry     — clip a single tile from `tile_grid.path` → tile_geometry.gpkg
  compute_model_bbox        — tight model-domain bbox = DeltaDTM valid-cell
                               extent (in tile) + `model_bbox_buffer_arcsec` → model_bbox.json
  extract_dem                — clip/fill DEM to bbox → dem.tif (optionally
                               geoid-corrected per tile, see
                               compute_geoid_offset_raster above)
  extract_dem_mask           — clip/reproject DEM-validity mask onto DEM grid → mask.tif
  compute_friction            — Copernicus LULC → Manning's-n friction raster → friction.tif
  extract_boundaries       (per return_period × waterlevel_name)
                               — select water-level stations within tile bbox → boundaries_{rp}_{slr}.gpkg
  write_aqueduct_config    (per return_period × waterlevel_name)
                               — build per-tile/scenario Aqueduct TOML → aqueduct_{rp}_{slr}.toml
  run_aqueduct              (per tile_id × return_period × waterlevel_name; simulation.smk)
                               — invoke `aqueduct.exe`; resource `mem_mb` (estimated per tile) → waterdepth_{rp}_{slr}.tif
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
                                merge_tiles → boundary_conditions in
                                sequence. REDESIGNED 2026-07 (in-process, not
                                subprocess): config.yml is loaded ONCE
                                (config_utils.load_config) and passed as an
                                already-loaded dict directly to each step
                                module's `run(config, ...)` function - the 5
                                step modules (sync_deltadtm.py,
                                tile_mask_creation.py, select_tiles.py,
                                merge_tiles.py, prepare_boundary_conditions.py)
                                are no longer standalone entry points; each
                                lost its own `argparse`/`load_config()`/
                                `if __name__ == "__main__":` block and gained
                                a plain `if __name__ == "__main__": sys.exit(
                                "...no longer a standalone entry point...")`
                                guard instead, so running one directly fails
                                loudly (exit 1, clear message) rather than
                                silently doing nothing (a real risk: with no
                                argparse left at all, `python select_tiles.py
                                --config X` would otherwise just import the
                                module, execute no top-level code, and exit
                                0 having done nothing). Old subprocess-based
                                isolation (crash in one step doesn't kill the
                                others) is now done via try/except per step
                                in `_run_step()` instead of a subprocess
                                return code - same UX (banner, timing,
                                [OK]/[FAIL] icon - NOT ✓/✗, see cp1252 note
                                below), same `--fail-fast` semantics.
                                Step selection: positional STEP arguments
                                (`python run_preparation.py select_tiles
                                merge_tiles`) replace the old `--only-tile-
                                grid`/`--only-boundary-conditions`/
                                `--skip-sync-deltadtm` flags entirely - name
                                the step(s) you want (from ALL_STEPS =
                                sync_deltadtm/tile_mask_creation/
                                select_tiles/merge_tiles/
                                boundary_conditions, matching config.yml's
                                `preparation.*` keys exactly), or give none
                                to fall back to `preparation.*` in
                                config.yml (today's real default: only
                                sync_deltadtm is false, the other 4 run).
                                Validated manually (not via argparse
                                `choices=`) because `choices=` combined with
                                `nargs="*"` on a positional has a real
                                argparse bug: with zero STEP args given, it
                                incorrectly validates the empty-list default
                                itself against `choices`, raising a spurious
                                "invalid choice: []" - confirmed by test,
                                worked around by validating `args.steps`
                                against `ALL_STEPS` by hand and calling
                                `parser.error(...)` for real. `--force`
                                still forwards to boundary_conditions only
                                (`run(config, force=True)`) - the tile-grid
                                steps have no cache to bypass.
                                CP1252 CONSOLE CRASH FIXED 2026-07 (found
                                while testing this redesign): the banner/icon
                                print()s used `═` (U+2550) and `✓`/`✗`
                                (U+2713/2717) - NOT in cp1252's printable
                                range (unlike em-dash, which is) - so
                                run_preparation.py would crash outright on a
                                plain Windows console the moment any step
                                actually ran (not just at --help time, like
                                the same class of bug found earlier this
                                session in docstrings). This was a PRE-
                                EXISTING bug carried over verbatim from the
                                old subprocess-based `_run()`'s own banner
                                style, never actually triggered before.
                                Replaced with plain `=`/`[OK]`/`[FAIL]`.
                                Verified via a stubbed-function test harness
                                (monkeypatches each step module's `run` to a
                                no-op/failing stub before calling `main()`,
                                so the orchestration logic - step selection,
                                config-loading-once, --force forwarding,
                                --fail-fast abort-before-next-step - is
                                exercised without touching real project
                                files or running the actual (slow,
                                side-effecting) pipeline logic): explicit
                                positional steps, config-default fallback,
                                `--force` reaching only boundary_conditions,
                                and `--fail-fast` correctly aborting before
                                the next step runs, all confirmed passing.
  sync_deltadtm.py             — no longer a standalone entry point as of
                                2026-07 (exposes `run(config)`, see
                                run_preparation.py above for the full
                                redesign). REWRITTEN 2026-07: downloads
                                DeltaDTM v1.1
                                DEM (per-continent zips) + mask_tiles.zip
                                directly from 4TU.ResearchData (hardcoded
                                DEM_ZIP_URLS/MASK_ZIP_URL dict at the top of
                                the script — one-time copy-paste from the
                                4TU dataset page, see script docstring) and
                                extracts their .tif files straight into the
                                data catalog's `deltadtm`/`deltadtm_mask`
                                source directories (the parent dir of each
                                source's `path:` in data_catalog_gfm.yml) via
                                get_data_catalog() — NOT a separate
                                `sync_deltadtm.source`/`target` pair (that
                                older manifest-CSV-sync design is gone).
                                `sync_deltadtm.zip_download_dir` (config.yml)
                                is only a temp staging dir for the .zip
                                downloads before extraction; download/extract
                                are both idempotent (skips a file already
                                fully downloaded/extracted), safe to re-run.
                                Must run before tile_mask_creation.py (which
                                reads the deltadtm_mask catalog VRT).
  tile_mask_creation.py       — no longer a standalone entry point as of
                                2026-07 (exposes `run(config)`, see
                                run_preparation.py above). step 1 of tile-grid prep, three sub-steps:
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
  select_tiles.py             — exposes `run(config)`, called from
                                run_preparation.py (see above). step 2, TWO
                                sequential filters:
                                (1) filter_tiles_by_dem_mask — keeps tiles
                                with any land/lake/river DeltaDTM mask
                                coverage.
                                (2) filter_tiles_by_exposure — of the
                                survivors, keeps tiles with any positive
                                population (`"population"` catalog source,
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
  merge_tiles.py               — no longer a standalone entry point as of
                                2026-07 (exposes `run(config)`, dropped its
                                old `--input`/`--output` override flags in
                                the process - unused by run_preparation.py,
                                see run_preparation.py above). step 3, REDESIGNED 2026-07 (see section 8's
                                TILE TRIMMING AND MERGE REDESIGN for the full
                                reasoning/history). Also runs, FIRST and
                                unconditionally checked (disabled by default -
                                see section 8's VERTICAL DATUM CORRECTION),
                                an optional EGM2008 -> GOCO06s DEM datum
                                correction (`vertical_datum_correction` in
                                config.yml, src/vertical_datum.py) - a no-op
                                until raw EGM2008-referenced DeltaDTM tiles
                                actually exist. Five further stages, in order:
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
  prepare_boundary_conditions.py — no longer a standalone entry point as of
                                2026-07 (exposes `run(config, force=False)`,
                                `logging.basicConfig()` moved out to
                                run_preparation.py's own `main()` since
                                calling it more than once in-process is a
                                silent no-op past the first call - see
                                run_preparation.py above). generates the per-(RP, SLR) water-level
                                NetCDFs consumed by extract_boundaries.py:
                                drops Antarctic COAST-RP stations, OPTIONALLY
                                (`boundary_conditions.mdt_correction.enabled`,
                                default false — see section 8's VERTICAL DATUM
                                CORRECTION) subtracts the AVISO MDT at each
                                station to re-reference from local MSL to
                                GOCO06s (compute_mdt_correction — nearest valid
                                grid cell + `fallback_search_deg` window
                                search, cached to
                                MDT_mapped_on_coastal_points.nc), computes
                                per-station IPCC AR6 SLR fingerprints scaled to
                                each target global-mean SLR, and combines them:
                                total_wl = storm_tide(RP) [- MDT] +
                                SLR_fingerprint. MDT correction OFF by default
                                (see section 8 — COAST-RP and the DeltaDTM
                                v1.1 DEM currently in use are both already
                                MSL-referenced in that configuration). Reads its RP/SLR lists straight
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
                         filter_tiles_by_exposure (select_tiles.py check 2:
                         any positive population within the tile bbox, via
                         the same `"population"` catalog raster
                         prepare_exposure_grid_chunk.py uses — a lookup
                         failure, e.g. bbox entirely outside WorldPop's
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
                         merge → simple valid-count-weighted mean waterdepth
                         (flood_count output REMOVED 2026-07 — no plot
                         consumed it, see section 8), plus a bounded
                         reservoir-sampled per-cell (min, max) depth across
                         ALL overlapping tiles (>=2 valid, any flood status —
                         deliberately includes disagreeing/"ambiguous" cells,
                         unlike the old first-two-flooding-tiles pairing) for
                         plot_overlap_continent_diagnostics.py.
plotting.py             — compute_flood_area_km2 (latitude-corrected pixel
                         area); plot_overlap_continent_diagnostics (per-
                         continent hexbin of min/max depth w/ Pearson r + pie
                         chart of confirmed-flood/confirmed-no-flood/ambiguous
                         cell counts, split on exposure.exceedance_threshold_m);
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
hardcoded in rules/ or scripts/, EXCEPT data-catalog key names (e.g.
`"deltadtm"`, `"population"`, `"geogunit_protection_units"`,
`"flopros_protection_standards"`, `"ssp_population_growth_factors"`), which
are literal strings at each call site rather than threaded through
config.yml — data_catalog_gfm.yml is the single place dataset identifiers
live, config.yml holds only pipeline parameters. Rules read config into a
`params:` block; scripts read `snakemake.params.*` (not `snakemake.config`
directly) so each rule's config dependency is explicit. All `{root}` /
`{code_root}` / `{aqueduct_root}` / `{processed_inputs_dir}` placeholders
are expanded once at Snakefile parse time (`_expand_paths`) so every
rule/script gets absolute paths (`processed_inputs_dir` needs a second
expansion pass since it's itself declared as `"{root}/processed_inputs"`).
Every standalone (non-Snakemake) script does the same via
`config_utils.load_config()`, which also honors a git-ignored
`config_local.yml` sibling for machine-local path overrides.

Key sections:
  paths.root / code_root / aqueduct_root: machine-specific; aqueduct_root
                                       defaults to code_root if unset.
    processed_inputs_dir: {root}/processed_inputs (prepare_boundary_
                                       conditions.py's intermediate NetCDFs;
                                       also the base for
                                       boundary_conditions.waterlevel_nc_dir
                                       and visualization.slr_trajectories_csv)
    hydromt_data_catalog: snakemake_workflow/config/data_catalog_gfm.yml
  sync_deltadtm.zip_download_dir:    {root}/inputs/DeltaDTM/_zips (temp .zip
                                       staging; final DEM/mask tile dirs come
                                       from data_catalog_gfm.yml's
                                       deltadtm/deltadtm_mask sources).
                                       delete_zips_after_extract defaults
                                       false. sync_deltadtm.py also
                                       downloads and patches 4TU's pre-built
                                       global DEM VRT mosaic
                                       (DEM_VRT_URL constant,
                                       download_and_patch_vrt) — rewrites
                                       every <SourceFilename> to this
                                       machine's absolute tile paths so the
                                       VRT works regardless of which
                                       continents were downloaded.
  preparation.*:                     sync_deltadtm/tile_mask_creation/
                                       select_tiles/merge_tiles/
                                       boundary_conditions switches for
                                       preparation/run_preparation.py
  vertical_datum_correction.enabled: DEM EGM2008 -> GOCO06s geoid
                                       correction, applied PER TILE inside
                                       rule extract_dem (preprocessing.smk).
                                       .gfc paths come from
                                       data_catalog_gfm.yml's
                                       `egm2008_geoid`/`goco06s` entries, not
                                       config. offset_raster_path is the
                                       cache for the ONE-TIME global geoid-
                                       offset raster (rule
                                       compute_geoid_offset_raster) — a
                                       pipeline-generated file, not a data-
                                       catalog candidate.
  raster_format:                     driver/compression/predictor/nodata
                                       shared by every GTiff the pipeline
                                       writes (preprocessing inputs,
                                       run_aqueduct's nodata placeholder,
                                       merge_chunk's output) - one format
                                       throughout instead of separate
                                       simulation.input_raster/
                                       postprocessing.output_raster copies.
  tile_grid.path:                    whatever regional/global tile grid is
                                       currently under test — see PROJECT
                                       OVERVIEW, changes often
    min_coast_fraction:               merge_tiles.py's water-deficient/
                                       land-deficient threshold
    max_merge_count:                  cap shared across both merge phases
    trim_buffer_arcsec:               coastal margin compute_trimmed_bbox
                                       keeps beyond a tile's land extent;
                                       must stay comfortably above
                                       simulation.model_bbox_buffer_arcsec
                                       (which needs real slack to add its
                                       own buffer without being clamped back
                                       to the tile bbox) and, at the coarsest
                                       DeltaDTM x-resolution found near the
                                       poles (~5"/px around 80-83N), several
                                       times that resolution to survive
                                       pixel-rounding.
    dedup_iou_threshold:              deduplicate_overlapping_tiles's
                                       minimum IoU to consolidate two tiles
                                       covering the same physical feature —
                                       0.8 sits with a wide margin between a
                                       confirmed duplicate pair (IoU=1.000)
                                       and the next-highest legitimate
                                       neighbour pair (IoU=0.645) on a
                                       regional test grid.
    cardinal_neighbor_overlap_threshold: minimum bbox-overlap fraction (of
                                       the smaller tile's width/height) for
                                       merge_undersized_tiles to treat two
                                       tiles as cardinal (not diagonal)
                                       neighbours.
  tile_split.fraction / max_depth / max_retries: run_pipeline.py's OOM-
                                       recovery retry loop around
                                       `snakemake` (each invocation runs to
                                       completion; OOM'd tiles are split and
                                       the DAG re-run from a fresh, mostly-
                                       cached state, up to max_retries times).
  boundary_conditions.waterlevel_nc_dir: {processed_inputs_dir}/WL_scenarios
    slr_scenario / confidence_level: IPCC AR6 regional SLR projection choice
                            for prepare_boundary_conditions.py
    return_periods / slr_scenarios: RP and SLR wildcard domains
    coastrp_min_lat: Antarctic COAST-RP station cutoff (practical round
                            number, not a cited threshold) — read by both
                            prepare_boundary_conditions.py and
                            tile_mask_creation.py
    mdt_correction.enabled: COAST-RP local-MSL -> GOCO06s MDT subtraction;
                            must be enabled together with
                            vertical_datum_correction.enabled so COAST-RP
                            and the DEM share one vertical reference
    station_search_buffer_deg: extract_boundaries.py's buffer around a
                            tile's bbox when picking candidate COAST-RP
                            stations, so tile_grid.trim_buffer_arcsec
                            shrinking tiles doesn't also shrink the
                            candidate pool for Aqueduct's knn IDW
                            interpolation
  simulation.model_outputs:          {root}/model_outputs
    aqueduct_executable:             {aqueduct_root}/build/aqueduct/aqueduct.exe
    model_bbox_buffer_arcsec:        buffer (arcsec) added around the DEM's
                                       own valid-cell extent, clamped to the
                                       tile bbox, for the flood model's
                                       coastal entry point - continuous
                                       degree-space, not pixel-quantized.
    dem_gap_fill:                    min_hard_fill_component_size (connected-
                                       component size above which a missing-
                                       land DEM gap hard-fills to
                                       land_fill_value_m instead of IDW-
                                       interpolating), interp_max_search_
                                       distance, interp_smoothing_iterations,
                                       land_fill_value_m
    flooding: resolution/knn/debug/default_friction for each tile's Aqueduct TOML config
  postprocessing.merged_outputs:     {root}/merged_results
    chunk_size_deg:                  also bounds peak memory of the
                                       per-chunk geogunit lookup, which
                                       scales with chunk_size_deg²
    block_size:                      write-block side length (pixels) for
                                       merge_chunk/compute_flood_fraction_
                                       chunk's block-by-block I/O loops -
                                       memory tiling only, no resampling
    overlap_corr_max_samples / overlap_corr_seed: reservoir-sampling cap and
                                       RNG seed for the per-continent overlap
                                       correlation/agreement diagnostics
                                       (src/merge.py's _PairSamples)
    plots.*:                         enabled/debug switches, dpi, and per-
                                       plot figsize/colormap/vmax/threshold
                                       values for plot_merged_results.py,
                                       plot_overlap_diagnostics.py,
                                       plot_overlap_continent_diagnostics.py
  protection.baseline_waterlevel_name: fixed reference scenario for
                                       prepare_exposure_grid_chunk's chunk
                                       grid metadata; also read into
                                       compute_exposure_analysis.py
    default_rp:                        fallback when FLOPROS has no Coastal
                                       or Riverine standard
  exposure.exceedance_threshold_m:   the SINGLE flooded-depth threshold used
                                       throughout (compute_flood_fraction_
                                       chunk, plot_merged_results' flood-area
                                       annotation, plot_overlap_continent_
                                       diagnostics' pie chart) - passed
                                       explicitly as a threshold_m rule param
                                       everywhere it's needed.
  adaptation.slr_intensities:        design intensities for protect/retreat;
                                       union'd into WATERLEVEL_NAMES by the
                                       Snakefile
  population_growth:                 ssps/output_years switches; the growth-
                                       factors Excel path/sheet come from the
                                       ssp_population_growth_factors catalog
                                       entry, not config
  analysis.*:                        compute_exposure/plot_burning_ember/
                                       plot_adaptation_bars/plot_timeseries/
                                       plot_world_maps switches
    avoid_worker_memory_budget_gb:    caps AVOID's ProcessPoolExecutor
                                       worker count by shared-payload size
  visualization.*:                   growth_rates (scenario-neutral ember
                                       y-axis, also the File 2 growth-matrix
                                       axis), slr_interp (File 2's dense SLR
                                       grid — linearly interpolated, not
                                       PCHIP), highlight_years, country_eai_
                                       vmax (shared fixed cap for per-country
                                       ember/timeseries plots), n_ember_
                                       contours, per-plot figsize keys,
                                       world_map_lat_range/downsample,
                                       geo109_subsample (lookup-table build,
                                       distinct from world_map_downsample's
                                       display-render subsampling), SSP↔RCP
                                       code mapping (used by plot_timeseries.
                                       py's panel titles and extract_slr_
                                       trajectories.py), colour palettes,
                                       level_linestyles/baseline_linestyle
                                       (plot_timeseries.py's per-intensity
                                       linestyle encoding)

config_local.yml.example — template for a git-ignored, machine-local
`config_local.yml` overriding `paths.root`/`paths.code_root`/
`paths.aqueduct_root` (and optionally `tile_grid.path`); loaded by the
Snakefile after config.yml if present, and by every standalone script via
`config_utils.load_config()`.

data_catalog_gfm.yml — HydroMT DataCatalog, `root: "D:/GFM"`. Key entries:
  coast_rp                — COAST-RP storm-tide NetCDF (~22,670 stations
                             after Antarctic removal); boundary-condition source.
                             MSL-referenced — see section 8.
  mdt_cnes_cls22            — AVISO MDT-HYBRID-CNES-CLS22-CMEMS2020. Used by
                             prepare_boundary_conditions.py's
                             compute_mdt_correction ONLY when
                             boundary_conditions.mdt_correction.enabled is
                             true (off by default — see section 8). Sole MDT
                             entry.
  ipcc_ar6_slr_projections — AR6 regional SLR fingerprints (NetCDF, directory-
                             root path completed at read time by scenario/
                             confidence choice) — prepare_boundary_conditions.py.
  ipcc_ar6_slr_wg1_csv      — Global-mean SLR percentile CSVs, one per SSP/RCP
                             code (SLR_{code}_wg1.csv, directory-root path
                             completed by visualization.ssp_rcp_codes) —
                             extract_slr_trajectories.py. Provenance/citation
                             not yet confirmed, unlike the regional entry above.
  egm2008_geoid / goco06s   — ICGEM spherical-harmonic .gfc coefficient files,
                             read directly by pyshtools (src/vertical_datum.py),
                             not through HydroMT's typed readers. Used to
                             compute the EGM2008 -> GOCO06s geoid-offset
                             raster when vertical_datum_correction.enabled.
  five_deg_grid             — DiluviumDEM-derived 5° tile index; NOT used by
                             tile_mask_creation.py (superseded by a DeltaDTM-
                             mask-derived 5° grid built inline).
  deltadtm / deltadtm_mask  — DeltaDTM v1.1 1-arcsec DEM + validity mask
                             (VRTs) — see section 8 for vertical datum.
  land_polygons             — OSM land polygons; layer **must** be specified
                             as `land_polygons` (not the default `marine_buffer`).
                             Used ONLY for plotting coastline reference lines
                             (plot_merged_results.py, plot_overlap_diagnostics.py)
                             — extract_dem/extract_dem_mask don't use it.
  land_use                  — Copernicus 100 m LULC.
  geogunit_protection_units / geogunit_country_units / geogunit_country_list
                             — WRI geogunit-107 (sub-national) / geogunit-109
                             (country) rasters + lookup table.
  flopros_protection_standards — FLOPROS coastal/riverine design RPs by geogunit-107.
  ssp_population_growth_factors — Excel workbook of per-country SSP growth
                             factors, read via catalog.get_source(...).path
                             (not a config.yml path/sheet pair).
  population                — WorldPop 2020, ~1 km, population COUNT (not density).
  lu_to_roughness_lookup     — Manning's-n lookup by Copernicus LULC class.

All dataset identifiers are literal catalog-key strings at their call sites
(e.g. `catalog.get_dataframe("flopros_protection_standards")`) — config.yml
holds pipeline parameters only, never a dataset's catalog key name.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. END-TO-END DATA FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 0 — Preparation (manual, one-off, NOT in the Snakemake DAG; all in
  snakemake_workflow/preparation/, run via `python preparation/
  run_preparation.py [STEP ...]` — as of 2026-07 the individual step
  scripts below are no longer standalone entry points, only callable
  through run_preparation.py (positionally naming one or more steps, or
  omitting STEP entirely to use config.yml's preparation.* switches) — see
  section 4a for full detail on every step below):
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
  onto the population grid using the `"geogunit_protection_units"` catalog
  raster. analysis/compute_exposure_analysis.py reads the `"flopros_
  protection_standards"` catalog table and snaps each geogunit's FLOPROS
  "Coastal" RP (falling back to "Riverine", then `protection.default_rp`)
  to the nearest simulated return period ≥ that standard.
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

DEM/MASK NODATA FILL (src/rasters.extract_dem / extract_dem_mask):
  The fill decision for a missing DEM/mask cell comes purely from the
  DeltaDTM mask itself (reprojected onto the DEM's own grid) — the OSM
  `land_polygons` dataset is not used here, since it isn't well aligned with
  DeltaDTM's own coastline. `land_polygons` is still used elsewhere, for
  plotting coastline reference lines (plot_merged_results.py,
  plot_overlap_diagnostics.py).

  DeltaDTM is a coastal-STRIP elevation product, not a global DEM, and the
  tile grid's nominal footprints extend well inland beyond its measured
  coverage — most of a tile's inland interior is legitimately "land the DEM
  never measured," so most DEM nodata is over land, not ocean.

  Rule (both functions, single source of truth): missing DEM cell where the
  DeltaDTM mask says land (0) OR has no coverage at all (255, nodata) →
  `land_fill_value_m` (default 9999m, so Julia never floods it); mask says
  ocean (1), lake (2), or river (3) → 0.0m. `extract_dem_mask` applies the
  matching rule to the mask output itself: no-coverage (255) → land (0),
  valid 0/1/2/3 kept unchanged.

  GAP-SIZE-DEPENDENT LAND FILL refines the flat-fill rule above: a lone
  nodata pixel surrounded by real elevation is more likely a measurement
  gap than genuinely unmeasured terrain, so `rasters.extract_dem` splits
  missing-LAND cells (missing-WATER cells are unaffected, always 0.0
  regardless of gap size) by the size of their connected (4-connectivity,
  `scipy.ndimage.label`) group of missing-land cells:
    - group size < `simulation.dem_gap_fill.min_hard_fill_component_size`
      → INTERPOLATED from surrounding valid DeltaDTM elevation via
      `rasterio.fill.fillnodata` (IDW; `interp_max_search_distance`/
      `interp_smoothing_iterations`, both config.yml-driven).
    - group size >= threshold → flat `land_fill_value_m` fill.
  `fillnodata` runs ONCE over the whole tile using only originally-valid
  DeltaDTM cells as interpolation anchors, producing a candidate value for
  every originally-missing cell (land AND water); only the small-gap-land
  subset of those candidates is actually used - large-gap-land and all
  water cells get their fixed value regardless of what the interpolation
  computed there. Connectivity is computed on missing-LAND cells only
  (ocean/lake/river nodata cells never participate in a land gap's
  component, don't inflate its size, and don't themselves become
  interpolation targets).

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
  DEFAULT STATE (both new switches below OFF): DEM (deltadtm catalog entry)
  and COAST-RP storm-tide forcing are BOTH referenced to local mean sea
  level (MSL) — verified directly against the files in use, not just catalog
  metadata:
    - DeltaDTM: the tiles actually loaded (`D:\GCFM_UU\raw_data\DeltaDTM\
      deltadtm.vrt`) are named `DeltaDTM_v1_1_..._GOCO06s_MDT.tif` and are
      sourced from the WUR YODA "sea-level-referenced-coastal-elevation"
      vault — the v1.1 MSL-referenced release (Seeger & Minderhoud, 2025),
      re-referenced from the original EGM2008 geoid: geoid converted to
      GOCO06s, then MDT subtracted (over land, via extrapolation — this
      extrapolation step is exactly what the 2026-07 change below avoids).
    - COAST-RP: documented as MSL-referenced by its own source paper
      (Dullaart et al. 2021).
  With both switches off, prepare_boundary_conditions.py does NOT apply an
  MDT correction to COAST-RP — `total_wl = storm_tide(RP) +
  SLR_fingerprint(target_slr)`, no `- MDT` term — and src/rasters.extract_dem
  does no vertical adjustment of its own; it passes the DeltaDTM v1.1
  elevations through unchanged (only horizontal clip/fill).
  HISTORY: an earlier version of prepare_boundary_conditions.py added
  `+ MDT` to COAST-RP (wrong sign for this direction of conversion, and
  gated on stale/mislabeled DEM metadata) — removed 2026-07-02 after
  confirming the DEM's actual (MSL-referenced) provenance. That removal is
  what left both switches below off by default.

  VERTICAL DATUM CORRECTION (2026-07, config-gated, OFF by default): a
  DELIBERATE, correctly-signed reintroduction of a datum correction, this
  time avoiding the WUR-YODA release's MDT-over-land extrapolation entirely.
  Two independent switches, meant to be toggled TOGETHER (enabling only one
  leaves DEM and forcing on different vertical references):
    - `vertical_datum_correction.enabled` (config.yml) — rules
      compute_geoid_offset_raster / extract_dem in preprocessing.smk,
      src/vertical_datum.py, rasters.extract_dem. REDESIGNED 2026-07 (twice
      — see history below) into its final form: applied PER MODEL TILE,
      folded directly into extract_dem's own per-tile_id clip step, NOT a
      separate preparation/*.py pass over the whole raw DeltaDTM release.
      The expensive part (spherical-harmonic geoid synthesis via pyshtools +
      boule from the data catalog's `egm2008_geoid`/`goco06s` .gfc sources,
      ~12-38s including the cache write) runs exactly ONCE, in the new
      non-tile_id-wildcarded rule `compute_geoid_offset_raster`, cached to a
      small (~2.7 MB) GeoTIFF at `vertical_datum_correction.
      offset_raster_path` (default `{root}/inputs/ICGM/
      geoid_offset_egm2008_goco06s.tif`). Every per-tile `extract_dem` job
      then just reprojects a tiny window of that cached raster onto its own
      DEM grid (`sample_geoid_offset`, ~0.05s) and ADDS it to every DEM cell
      with valid DeltaDTM elevation ONLY — nodata cells are left untouched
      (still nodata, so extract_dem's existing 9999/0 fill logic downstream
      is unaffected by the correction). NO MDT term is ever applied to the
      DEM (mirrors GCFM_UU/workflow/scripts/05a_get_elevation.py's own DEM
      correction exactly - MDT is only ever meaningful at sea). Because
      `extract_dem` now has a REAL Snakemake dependency edge on
      `compute_geoid_offset_raster.output.offset_raster` (via `input:`, not
      just `params:`), enabling the switch requires zero manual
      `_PREPROCESS_OUTPUTS` wiring in the Snakefile — Snakemake pulls the
      one-time rule in automatically, exactly once (verified via dry-run:
      count=1 for compute_geoid_offset_raster, extract_dem's own count
      unchanged from the disabled state). Still: the `deltadtm` catalog
      `path` must ALREADY point at the ORIGINAL EGM2008-referenced release
      before this switch is turned on (applying it to the already-MSL-
      referenced WUR-YODA `_GOCO06s_MDT` release would double-correct) -
      that catalog repoint is still the user's own manual step, not done by
      this pipeline. `deltadtm.path` was separately relocated 2026-07 from
      the absolute `D:\GCFM_UU\raw_data\DeltaDTM\deltadtm.vrt` to
      `{root}/inputs/DeltaDTM/deltadtm.vrt` (same 7553 `_GOCO06s_MDT` tiles,
      just moved under the GFM project root) - verified consistent (not a
      dangling/broken reference).
      DESIGN HISTORY (two iterations before this final one, same 2026-07
      session): (1) originally a step bolted onto preparation/merge_tiles.py
      (a standalone, non-Snakemake script) per an early literal instruction
      to "add it to merge_tiles" - corrected whole raw 1x1deg DeltaDTM
      source tiles up front, writing `_GOCO06s`-suffixed copies of all
      ~7500+ tiles. (2) then pulled into an actual Snakemake `checkpoint
      correct_deltadtm_datum` rule (preprocessing.smk) for real dependency
      tracking, still whole-raw-tile-granularity, gated into
      `_PREPROCESS_OUTPUTS` manually since nothing consumed its output via a
      real file dependency. (3) THIS final version: per-model-tile,
      cached-offset-raster design above - triggered by the user clarifying
      "I wanted the correction to happen per tile" and confirming (over
      compute_dem_tile_dir's whole-file-glob approach) that folding it into
      extract_dem was the intended granularity. `src/vertical_datum.py`'s
      old `correct_dem_tile`/`correct_deltadtm_tiles_dir` (whole-raw-tile
      functions) and `scripts/correct_deltadtm_datum.py` were deleted;
      `compute_geoid_offset_grid` was kept (still the core pyshtools call)
      and `write_geoid_offset_raster`/`sample_geoid_offset` added.
      Verified end-to-end: `write_geoid_offset_raster` against the real
      .gfc files (~38s incl. write, physically plausible field);
      `sample_geoid_offset` (~0.05s per tile-sized window); the full
      extract_dem correction logic against a synthetic HydroMT-accessor
      DataArray (valid cell correctly shifted by the sampled offset, nodata
      cell provably untouched); Snakefile dry-run in both states (disabled:
      no compute_geoid_offset_raster job, extract_dem count unchanged;
      enabled: compute_geoid_offset_raster count=1, extract_dem count=44
      unchanged from disabled).
      CATALOG SCHEMA BUG FOUND + FIXED 2026-07 (still applies): when `egm2008_geoid`/
      `goco06s` were first added to data_catalog_gfm.yml they used
      `file_path:`/`file_format:`/top-level `description:`/`attributes:`
      (a schema HydroMT does NOT recognize — it requires `path:` +
      `data_type:` at minimum) - this broke `get_data_catalog()` for the
      ENTIRE pipeline (every script that loads the catalog), not just this
      feature. Fixed by converting both to the standard
      `data_type: DataFrame` / `driver: csv` / `path:` / `meta:` schema
      already used throughout this file (data_type/driver are inert here —
      both files are read directly by pyshtools, never through HydroMT's own
      DataFrame reader; they exist only so the catalog parses and `.path`
      resolves). A third, similarly-broken `mdt_cnes_cls22` entry was found
      alongside them and fixed the same way (also fixed: `mdt_cnes_cls22`
      initially appeared to duplicate a pre-existing
      `mdt_hybrid_cnes_cls22_cmems2020` entry at a different path — the user
      subsequently REMOVED that older duplicate entry and confirmed
      `mdt_cnes_cls22` is the canonical one, so all script references
      (prepare_boundary_conditions.py's compute_mdt_correction call,
      src/vertical_datum.py's docstring, data_catalog_gfm.yml's `coast_rp`
      cross-reference) were updated to `mdt_cnes_cls22` — verified it
      resolves to a real file on disk, `D:/GFM/inputs/AVISO/
      mdt_hybrid_cnes_cls22_cmems2020_global.nc`, and the old key correctly
      raises KeyError now).
    - `boundary_conditions.mdt_correction.enabled` (config.yml) —
      preparation/prepare_boundary_conditions.py's compute_mdt_correction.
      SUBTRACTS (not adds) the AVISO MDT (`mdt_cnes_cls22`,
      already in the data catalog) at each COAST-RP station (nearest valid
      grid cell, falling back to a `fallback_search_deg` window search for
      NaN cells - ~27/22670 stations still end up NaN even with the
      fallback, same known gap as the legacy notebook, left unchanged;
      treated as 0 correction for those stations) to bring COAST-RP from
      local MSL to GOCO06s — the sign convention is ported from
      GCFM_UU/workflow/src/surge.apply_mdt_correction
      (`rp_level = rp_level_raw - mdt`), NOT the `+ MDT` addition previously
      used in Boundary_conditions_waterlevels/03_combine_wl_data_scenarios.
      ipynb (`total_wl = storm_tide + MDT + SLR`), which was the wrong sign
      for this direction of the conversion. Cached to
      `MDT_mapped_on_coastal_points.nc` in `processed_inputs_dir`.
  `_nearest_valid_grid`/`compute_mdt_correction` and
  `compute_geoid_offset_grid`/`correct_dem_tile` were both verified against
  real inputs this session: the geoid offset (real .gfc files) produced a
  physically plausible global field (-4.9 to +4.0 m, smooth 0.3° grid,
  ~12s); the MDT nearest-valid+fallback logic was verified against synthetic
  data with a deliberate NaN band, matching expected values exactly
  (direct hit, polar edge, and NaN-gap fallback all correct); the raw/
  corrected/WUR-YODA filename filtering in correct_deltadtm_tiles_dir was
  verified against a synthetic tile directory (correctly skips WUR-YODA
  `_GOCO06s_MDT` tiles and already-corrected tiles, only processes genuinely
  raw ones). Not yet verified end-to-end on real DeltaDTM/MDT data, since
  neither the raw EGM2008 tiles nor the local MDT NetCDF are downloaded yet.

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
  File 3's SSP task list (`ssp_tasks`) is built only from SSPs actually
  present in the SLR-trajectory CSV's columns (`available_ssps`), not the
  full configured `population_growth.ssps` list: a configured SSP with no
  trajectory data always returns `None` from `_avoid_ssp_worker_task`, and
  `_maybe_write_avoid`'s completion check (`len(ssp_results) >=
  expected_ssp`) would otherwise wait forever for a result that never
  arrives - stalling the CSV write for every slr_intensity, not just the
  one missing SSP, even though every other task completed.
  Downstream scripts glob file suffixes precisely (`exposure_*_base.csv`,
  `exposure_*_growth_matrix.csv`, `exposure_*_ssp.csv`) — a plain
  `exposure_*.csv` glob would pick up ALL THREE per scenario with
  incompatible schemas.

  Country-level ember/timeseries plots use a fixed `visualization.
  country_eai_vmax` colorbar/y-axis cap (comparable across countries);
  global ember plots keep a dynamic per-figure vmax. The burning-ember
  figure has two stacked subplots sharing the SLR x-axis: the impact-matrix
  heatmap, and (when slr_uncertainty data exists) a single uncertainty
  subplot with one y-tick per highlight year - each year's row shows every
  SSP's P17-P83 error bar side by side (small offsets), coloured to match
  the heatmap's SSP legend, with only the left spine and no x tick marks
  (see src/visualization.plot_burning_ember's docstring for the full
  layout). plot_burning_ember.py's SSP trajectory overlay lines are
  independent of File 1/2/3 — built directly from real SSP growth factors
  (population_growth.interpolate_growth_factor) and slr_trajectories_csv.

MERGE:
  src/merge.merge_tile_rasters_chunk is CHUNK-SCOPED (not a single global
  merge) — block-wise reads keep memory bounded, produces a simple
  valid-count-weighted mean waterdepth (not distance-weighted IDW, strictly
  excluding AQUEDUCT_NODATA cells — unaffected by the diagnostic-only
  assumption below). The flood_count (uint8 tile-overlap-count) raster and
  its VRT/plot were REMOVED 2026-07 — unused by any downstream step, the
  plot was purely diagnostic and the user didn't need it. Separately
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
  `OutOfMemoryError` (the memory cost of `component_indices` in
  core/src/core.jl scales with the tile's pixel count, not RP/SLR, so it's
  effectively a per-tile_id failure regardless of whether it's caused by the
  tile's own size or by `mem_mb` under-estimating concurrent memory
  pressure — see `run_aqueduct`'s docstring in simulation.smk) marks
  `model_outputs/oom_tiles/{tile_id}.txt`; all future (RP, SLR) combos for
  that tile then skip Aqueduct immediately. Empty-boundaries tiles and
  OOM-marked tiles both get an
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

  RASTER FORMAT (`raster_format` in config.yml, shared by preprocessing
  inputs, run_aqueduct's nodata placeholder, and merge_chunk's output):
  plain `"GTiff"`, not `"COG"` - COG's embedded overview/pyramid generation
  is pure overhead for files read exactly once by Aqueduct and never served
  as web map tiles, and inflates file size ~4x for no benefit (does not
  raise Aqueduct's own memory ceiling, since overviews aren't loaded into
  the in-memory array Aqueduct computes with). `src/rasters.py::save_raster`
  explicitly sets `tiled=True, blockxsize=512, blockysize=512` (plain GTiff
  defaults to striped layout otherwise, which would hurt the small-
  windowed-read access patterns merge.py/tile_split.py rely on) —
  `save_nodata_raster` needs no separate handling since it copies
  `ref.profile` from the reference DEM, inheriting the same tiling settings.

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
  rather than waiting for the whole DAG to finish — under the (since-revised,
  see "Run with" above) assumption that Aqueduct was fully serialized
  (`aqueduct_runs=1`), an OOM discovered early in a run would otherwise sit
  idle with its nodata placeholder for the rest of that run.
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
│   ├── waterdepth_{chunk}_{rp}_{slr}.tif         (temp)
│   ├── flood_fraction/flood_fraction_{chunk}_{rp}_{slr}.tif   ← kept; feeds analysis/
│   ├── exposure_population_grid_{chunk}.tif
│   └── exposure_geogunit_grid_{chunk}.tif
├── waterdepth_{rp}_{slr}.vrt      (if plots.enabled; flood_count_*.vrt REMOVED 2026-07)
├── exposure/                                    ← written by analysis/compute_exposure_analysis.py
│   ├── exposure_baseline_base.csv               ← File 1: discrete SLR_{mm} cols, no growth
│   ├── exposure_baseline_growth_matrix.csv      ← File 2: dense SLR x growth-rate grid (ember bg)
│   ├── exposure_baseline_ssp.csv                ← File 3: one EAI_{SSP}_{year} col, pre-resolved
│   ├── exposure_protect_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   ├── exposure_retreat_{slr}_base.csv / _growth_matrix.csv / _ssp.csv
│   └── exposure_avoid_{slr}_growth_matrix.csv / _ssp.csv   ← no _base.csv, see KEY DOMAIN LOGIC NOTES
└── plots/                                        (if plots.enabled)
    ├── waterdepth_{rp}_{slr}.png                 (flood_count_*.png REMOVED 2026-07)
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

`mem_mb` resource: forgetting `--resources mem_mb=<N>` on the snakemake
command line for `simulate`/`all` runs leaves Aqueduct's memory-budgeted
concurrency unbounded, risking real OOM crashes under heavy concurrent load
(no per-instance JIT crash risk — that was disproven, see "Run with" above).

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

`postprocessing.chunk_size_deg` bounds peak memory of the geogunit lookup
inside `prepare_exposure_grid_chunk`/`protection.load_geogunit_ids`, which
resolves the WRI geogunit raster onto the FULL chunk extent at once (not
block-wise) — that step's memory scales with chunk_size_deg², so raising
this value risks OOM there.
