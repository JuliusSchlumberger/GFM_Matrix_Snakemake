# GFM — Global Flood Model

A tile-based coastal flood model pipeline, extending Deltares' Aqueduct Coastal
Flooding methodology to global scale. For every tile in a global tile grid it
prepares DEM/mask/friction/water-level-boundary inputs, runs a Fast Sweeping
Method eikonal flood solver per (tile, return period, sea-level-rise
scenario), merges results into spatial chunks, and runs exposure/adaptation
and benchmark-validation analyses on top. A separate sub-pipeline
(`sfincs_tiles/`) calibrates/validates the eikonal solver against SFINCS.

## Repo layout

The Snakemake DAG covers preprocessing → simulation → postprocessing; the
other folders are standalone pipelines that share the same library code:

```
Snakefile                  entry point for the Snakemake DAG
run_pipeline.py            wraps `snakemake` with an OOM-retry loop
check_{preprocess,simulation,postprocess}_progress.py
                            read-only progress monitors
run_delta_floodmaps.py     global-deltas deliverable (ssp126/ssp245 configs)
run_bangkok_case_study.bat, run_thailand_bangkok.{bat,ps1}
                            local case-study launchers

src/                       shared library, imported by everything below
                            (sys.path.insert, not a pip-installed package)

preparation/               run_preparation.py: builds the tile grid +
                            boundary-condition inputs the DAG reads, run first

analysis/                  run_analysis.py: exposure/adaptation analysis and
                            figures. Not a Snakemake rule target, but the HPC
                            exposure jobs import compute_exposure_analysis.py
                            and the HPC flood-totals job runs
                            compute_flood_totals.py + run_analysis.py

validation/                run_validation.py: HR/FAR/CSI vs national hazard
                            maps (Wales/Scotland/England, New Brunswick);
                            England FZ2/FZ3 and SF Bay via standalone scripts

sfincs_tiles/              eikonal vs SFINCS calibration/validation (Hydrax):
                            generate_calibration_jobs.py stage1|stage2,
                            rerun_{eikonal,sfincs}_calibration.sh;
                            run_final_tile.sh is the per-tile SLURM worker

calibration_studies/       eikonal solver sensitivity (inner sweep rounds /
                            outer obstacle-coupling iterations):
                            generate_calibration_batch_jobs.py
                            sweep_budget|obstacle_coupling (docs/methods_03)

snakemake_workflow/
  rules/                     rule definitions (*.smk)
  scripts/                   the rules' `script:` targets, plus HPC/SLURM job
                              generators and their payloads (generate_hpc_*,
                              run_exposure_pass1/2, reduce_exposure_*) and a
                              few manual tools
  config/                    config.yml (single source of truth), subset
                              configs, data catalogs
  hpc.md                     HPC/SLURM setup and dispatch

tests/                     pytest suite (regression scripts, also runnable
                            standalone)
docs/                      methods docs; figure_scripts/ holds the standalone
                            scripts that make their figures

environment.yml            `gfm` env (main pipeline)
environment_sfincs.yml     `hydromt-sfincs-dev` env (SFINCS build/run only)
pyproject.toml             pytest + ruff config only (not a package)
TODO.md, AUDIT.md          ongoing cleanup: audit, plan, deferred changes
```

## Environments

- `gfm` (`environment.yml`): everything — DAG, preparation, analysis,
  validation, and all plotting. On the Windows dev machine the same env is
  named `gfm_python_preprocessing` (outdated name); on Hydrax it is `gfm`.
- `hydromt-sfincs-dev` (`environment_sfincs.yml`): `sfincs_tiles/` model
  build/run only. Its matplotlib crashes on `savefig`, so run the
  `sfincs_tiles/` plotting scripts in `gfm`.

## Running things

Four separate things get run, none of them automatically triggering another:

1. **Preparation** (once, before the DAG, and whenever DeltaDTM/boundary
   inputs change):
   ```
   python preparation/run_preparation.py [STEP...] [--config snakemake_workflow/config/config.yml]
   ```
2. **The Snakemake DAG** (preprocessing → simulation → postprocessing):
   ```
   python run_pipeline.py
   # or directly:
   snakemake all --cores 4 --resources mem_mb=8000
   ```
   The full grid runs on HPC as split jobs (see `snakemake_workflow/hpc.md`).
   Monitor with `check_preprocess_progress.py` / `check_simulation_progress.py`
   / `check_postprocess_progress.py` (read-only, safe to run anytime).
3. **Analysis** (exposure/adaptation, after the DAG):
   ```
   python analysis/run_analysis.py
   ```
4. **Validation** (benchmark HR/FAR/CSI comparison, after the DAG):
   ```
   python validation/run_validation.py --countries GBR CAN
   ```
   SF Bay and England Flood Zone 2/3 run via the standalone scripts in `validation/`.

**Dry runs:** never `snakemake -n all` on the full grid locally (the DAG is
split into HPC jobs and a full local dry run does not finish). Use a subset
config instead:
```
GFM_CONFIG_PATH=snakemake_workflow/config/thailand_bangkok_materialized.yml snakemake -n all --cores 1
# or bangkok_chao_phraya_materialized.yml
```

## Configuration

`snakemake_workflow/config/config.yml` is the single source of truth for
every pipeline above (not just the Snakemake DAG). Override machine-specific
values (`paths.root`, `paths.code_root`, ...) in
`snakemake_workflow/config/config_local.yml` — gitignored, auto-loaded if
present, never touch the committed `config.yml` for a per-machine value.
`*_materialized.yml` and `config_hpc.yml` are generated/untracked: never
hand-edit the former; create the latter by copying `config_hpc.yml.example`.

## Import convention

Every entry point uses `sys.path.insert(0, ...)` to reach `src/` (and, where
relevant, `analysis/`) — this is a flat script layout, not an installable
Python package. If you add a new script, follow the existing pattern in a
sibling file rather than introducing a different import mechanism.

## Tests

```
pytest -m "not needs_pdrive"                    # in gfm
pytest tests/sfincs_* -m "not needs_pdrive"     # in hydromt-sfincs-dev
```
`tests/conftest.py` skips the `tests/sfincs_*` tests unless hydromt_sfincs
2.x is installed. Each script still runs standalone with `python <script>.py`.

## Further reading

- `docs/methods_01_tile_processing_and_waterlevels.md` — the data-preparation
  pipeline (elevation/water mask, tiling, water-level boundaries).
- `docs/methods_02_flood_depth.md` — the friction-weighted eikonal flood-solver
  methodology.
- `docs/methods_03_calibration_sensitivity.md` — sensitivity studies for the
  inner-loop round cap and outer obstacle-coupling iterations.
- `docs/methods_04a_SFINCSvalidation.md` — SFINCS calibration and validation
  of the eikonal model.
- `docs/methods_04b_MapsValidation.md` — validation against official national
  flood hazard maps.
- `snakemake_workflow/hpc.md` — HPC/SLURM setup and dispatch.
