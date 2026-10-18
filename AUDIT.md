# Repository state and final-sweep brief

Handover for a last review pass over the repo. It says what the repo is, what the
cleanup already did (so the sweep does not redo it), the rules any change must
follow, how to verify, and where the remaining work probably is. Open real
(behaviour-changing) issues live in `TODO.md`; standing instructions in `CLAUDE.md`.
The original audit (map, classification, phased plan) is in git history:
`git show 89e130e:AUDIT.md`.

## 1. What the repo is

GFM (Global Flood Model): tile-based global coastal flood hazard pipeline. Per
DeltaDTM tile it prepares DEM/mask/friction/water-level boundaries, runs a Python
fast-sweeping eikonal solver per (tile, return period, SLR scenario), merges tiles
into 5 deg chunks, then runs exposure/adaptation analysis and validation against
national hazard maps. `sfincs_tiles/` calibrates the eikonal friction against SFINCS
on the Hydrax HPC.

Size: ~31,600 lines of Python/Snakemake/shell in 189 tracked files.

| Directory | Role | Entry points |
|---|---|---|
| `src/` (23) | Shared library, flat layout, imported via `sys.path.insert` | - |
| `Snakefile`, `snakemake_workflow/` (33) | DAG: preprocess, simulate, postprocess; HPC job generators | `snakemake <target>`, `run_eikonal_dag.py`, `generate_hpc_*.py`, `generate_eikonal_jobs.py` |
| `preparation/` (10) | One-off global inputs: tile grid, boundary conditions, DeltaDTM VRTs | `run_preparation.py` |
| `analysis/` (28) | Exposure, adaptation pathways, figures, delta case | `run_analysis.py`, `run_delta_floodmaps.py` (top level) |
| `validation/` (13) | Eikonal vs national maps (GBR, CAN) and SF Bay | `run_validation.py`, standalone `validate_sf_bay.py` + `plot_sf_bay.py`, `rescore_england_fz3.py`, `validate_england_fz2_band.py` |
| `sfincs_tiles/` (36) | SFINCS calibration, Hydrax only | `generate_calibration_jobs.py stage1/stage2`, `rerun_calibration.sh --part`, `run_final_tile.sh` |
| `calibration_studies/` (10) | Solver sensitivity (sweep budget, obstacle coupling) | `generate_calibration_batch_jobs.py` |
| `tests/` (26) | pytest suites (`validate_*.py`, `test_*.py`) | `pytest` |
| `docs/` | `methods_01..04b` method write-ups, `figure_scripts/` | - |

Scenario configs in `snakemake_workflow/config/`: `config.yml` (single source of
truth), `config_local.yml` (gitignored, machine paths), `config_hpc.yml` (Hydrax),
`<scenario>.yml` + `<scenario>_pins.yml` -> generated `<scenario>_materialized.yml`
(bangkok_chao_phraya, calibration_500_tiles, deltas_ssp126, deltas_ssp245;
deltas_design is written by `analysis/prepare_delta_design_runs.py`).
`profiles/default/config.yaml` sets Snakemake rerun triggers (see section 4).

## 2. Already done (do not redo)

All on branch `cleanup` (~50 commits ahead of `main`, not merged).

- Phases 0-7 of the original plan: tooling (ruff, pytest), stale/duplicate files
  deleted, out-of-scope validation removed, doc drift fixed, near-duplicate helpers
  unified (`src/step_runner.py`, `analysis/_common.py`, `validation/_chunks.py`,
  `tiles.chunk_id_from_corner`), hard-coded paths moved to config /
  `sfincs_tiles/hydrax_env.sh`, aqueduct -> eikonal and module renames, large
  modules split (`run_analysis.py` STEPS table, `build_station_boundary_lines.py`
  -> `boundary_lines_geometry.py` + `boundary_lines_plot.py`).
- Documentation trim over all 188 files (CLAUDE.md "Documentation standard"):
  docstring+comment lines 15,723 -> 4,309; ~40 doc/code contradictions resolved
  or logged in TODO.md.
- Removed scenarios: thailand_bangkok, gbr_wales_scotland_friction9; references to
  the earlier SFINCS studies (validation_sfincs*, sfincs_calibration*) removed from
  code, comments and docs.
- `friction_scale_factor` 12.0 documented as production (30 x 0.4); calibration
  scripts share `flood_model.OBSTACLE_BLOCK_FRICTION`; `build_overview_table.py`
  reads the published values from `compare_against_published.py`.
- Data catalogs: one `meta` schema per entry (description, source_author,
  source_url, source_version, source_license, paper_ref/paper_doi), sources and
  licences filled from the providers' pages; unused entry removed.
- SF Bay now writes agreement and depth-difference rasters and maps
  (`validate_sf_bay.py`, `plot_sf_bay.py`).

## 3. Rules for any change

- **Behaviour-preserving by default.** A change that alters results, outputs, file
  names or CLI behaviour is logged in `TODO.md`, not made, unless the user asks.
- **Never write to, move or delete anything on `P:\`** (`/p/11212688-004-global-floodmaps/`
  on Hydrax). Reading is fine. Local test outputs go to `.cleanup_baseline/` (gitignored).
- **Do not change without asking:** output path patterns and wildcard constraints in
  `rules/*.smk` (orphans outputs on P:\), `config.yml` defaults, `data_catalog_*.yml`,
  pinned versions in `environment*.yml`, `sfincs_tiles/` while a Hydrax batch runs.
- **Stale code is deleted** (git keeps it), never moved to an archive folder.
- **Documentation standard** (CLAUDE.md): one-line imperative docstrings plus only
  non-inferable facts (units, datums, sign conventions, references); no history,
  dates or phase references in code or docs; no new `.md` files unless asked.
- **Validation scope:** Wales, Scotland, England, New Brunswick, SF Bay only.
- **Never run `snakemake -n all` on the full grid** (does not finish); never run a
  script's `--help` without checking it uses argparse (some run for real and write
  to P:\); progress monitors (`check_*_progress.py`) are run by the user.
- **Git:** commit per logical step with a clear message; do not push unless asked.
- Flat `src/` layout via `sys.path.insert`; no packaging or relative imports.

## 4. Verification toolkit

Interpreters (Windows): `gfm` env at
`C:/Users/schlumbe/AppData/Local/miniforge3/envs/gfm_python_preprocessing/python.exe`
(the PATH `python` is a Store stub); `hydromt-sfincs-dev` for `sfincs_tiles` model
build/run and `tests/sfincs_*` (its matplotlib is broken, never plot there; its
hydromt 1.x cannot import `src/config_utils`).

| Check | Command | Current result |
|---|---|---|
| Lint | `python -m ruff check .` | clean |
| Tests (gfm) | `python -m pytest -q -m "not needs_pdrive" tests` | 68 passed |
| Tests (sfincs) | in hydromt-sfincs-dev: `python -m pytest -q -m "not needs_pdrive" tests/sfincs_*` (without `-m`, the smoke test rebuilds `P:\...\sfincs_smoke_test`) | 17 passed without `-m` (count with `-m` not yet re-measured) |
| Dry run | `GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml snakemake -n all --cores 1` | 455 jobs, matches `.cleanup_baseline/snakemake_n_all_bangkok_chao_phraya_profile.txt` |

The dry-run job count depends on `profiles/default/config.yaml`
(`rerun-triggers: [mtime, params, input, software-env]`); without it, any script
edit flags its rule as "code changed".

Methods that worked for proving a refactor changes nothing:
- **Docs/comments-only edits:** compare the token stream (minus COMMENT/NL tokens and
  docstring statements) and the docstring-stripped AST against `git show HEAD:<file>`.
- **Moves/splits:** every moved def/constant AST-identical to HEAD, defined once.
- **Behaviour parity:** run old code (a `git worktree` of HEAD) and new code on the
  same inputs (synthetic, or real tiles read from P:\ with outputs to
  `.cleanup_baseline/`) and compare outputs (raster hashes, gpkg geometry/attribute
  hashes, CSV/DataFrame equality). Example: 4 real tiles for the boundary-line split.
- **Config edits:** materialize to a scratch dir and compare parsed YAML by value.

## 5. Known open items (details in TODO.md)

Decisions for the user, not sweep work: merge `cleanup`; Hydrax checks after pulling
(`DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal`); re-run the
delta / bangkok_chao_phraya / calibration_500_tiles outputs that used
`friction_scale_factor` 9.0 or 30.0; history rewrite for ~1 GB of `.npy` blobs.

Small code issues already logged (fix only with the user's go-ahead):
- `population_growth.interpolate_growth_factor` / `tile_split.choose_split`: docs vs
  behaviour (clamps / warns instead of default / raise).
- `retry_io.retry_transient_io`: `retries` counts attempts.
- `calibration_paths.decode_waterdepth_cm`: 0 cm stays 0.0, only nodata -> NaN.
- Unused parameters: `generate_eikonal_jobs.generate_wave_dispatch` (3, tied to rule
  params), `build_sfincs_tile._compute_zsini_array(grid_coords)`; wasted DEM read in
  `connectivity_tiling._domain_touches_ocean`.
- Misnomer `boundary_lines_geometry.land_eroded` (holds land grown outward).
- String literals with history or stale claims (argparse help, assert messages,
  one Usage line, one print), and one test whose name does not match its check.
- `rules/simulation.smk` fallback to the old `aqueduct_mem_estimate` key.
- Lookup column `copernicus_worldcover` holds Copernicus LC100 codes (label only).

## 6. Where a final sweep should look

Not yet systematically checked:
1. **Dead code at function level:** public functions, constants and CLI flags with no
   caller (the cleanup removed dead files, not dead functions). Check callers across
   scripts, `.smk` files, shell drivers and generated job text.
2. **Config keys:** keys in `config.yml` / scenario configs that no code reads, and
   keys read in code with a `.get(..., default)` that hides a missing setting.
3. **Docs vs code:** README.md, `snakemake_workflow/hpc.md`, `docs/methods_*.md`
   (commands, file names, defaults, numbers) against the current code and config.
   The trim covered docstrings and comments, not these files.
4. **String literals:** argparse help, print/log messages and generated sbatch text
   with stale names, history or wrong defaults (deliberately outside the trim).
5. **Remaining duplication:** helpers copied between `src/`, `sfincs_tiles/`
   (separate env, cannot import `src/`) and scripts; flag rather than merge across
   the env boundary.
6. **Error handling:** bare `except Exception` that hides real failures; retries
   around non-transient errors; exit codes of drivers.
7. **Tests:** modules no test imports (`src/exposure_analysis.py`, `exposure_grid.py`,
   `pathways.py`, `population_growth.py`, `connectivity_tiling.py`, all of `analysis/`);
   tests that only run with P:\ data; misnamed tests.
8. **Shell drivers** (`sfincs_tiles/*.sh`): only syntax-checked locally; review
   quoting, `set -uo pipefail` use and path handling for Hydrax.
9. **Naming consistency** across scripts (CLI flag names, output file patterns,
   `rp`/`return_period`, `slr`/`waterlevel_name`).

A sweep prompt should state, per dimension: the scope (directories), what counts as
a finding, report-only vs fix (and which fixes are allowed under section 3), the
verification required before each commit, and the report format (file:line, issue,
evidence, proposed fix, behaviour impact).
