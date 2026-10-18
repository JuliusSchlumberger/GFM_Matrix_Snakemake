# CLAUDE.md

## Project
GFM (Global Flood Model): tile-based global coastal flood hazard pipeline extending
Deltares' Aqueduct Coastal Flooding method. Per DeltaDTM tile it prepares DEM/mask/
friction/water-level boundaries, runs a Python Fast-Sweeping eikonal flood solver per
(tile, return period, SLR scenario), merges to spatial chunks, then runs exposure/
adaptation analysis and benchmark validation. A separate SFINCS sub-pipeline
(`sfincs_tiles/`) calibrates/validates the eikonal solver against SFINCS on Hydrax.

## Layout & entry points
- `preparation/run_preparation.py`: run first; builds tile grid + boundary inputs.
- `Snakefile`: targets `preprocess`, `simulate`, `postprocess`, `all`.
  Rules in `snakemake_workflow/rules/`, rule scripts in `snakemake_workflow/scripts/`.
- HPC/SLURM: `snakemake_workflow/scripts/generate_hpc_{preprocess,postprocess}_job.py`,
  `generate_hpc_simulation_jobs.py` (see `snakemake_workflow/hpc.md`).
- `analysis/run_analysis.py`: exposure/adaptation figures (HPC exposure jobs also
  import `analysis/compute_exposure_analysis.py`).
- `validation/run_validation.py`: eikonal vs national hazard maps (soft-scored HR/FAR/CSI).
  Scope is ONLY Scotland, Wales, England, New Brunswick and California/SF Bay (+ subregions);
  SF Bay and England FZ2/FZ3 run via standalone scripts in `validation/`. Do not add
  other countries back without asking.
- `sfincs_tiles/generate_calibration_jobs.py stage1|stage2`, `rerun_*_calibration.sh`:
  eikonal vs SFINCS friction calibration, Hydrax only (`run_final_tile.sh` is the
  per-tile SLURM worker).
- `calibration_studies/generate_calibration_batch_jobs.py sweep_budget|obstacle_coupling`:
  eikonal solver sensitivity (inner sweep rounds / outer iterations), docs/methods_03.
- `src/`: shared library, flat layout; scripts add it via `sys.path.insert`. Follow the
  sibling file's pattern; do not introduce packaging/relative imports.

## Config
- `snakemake_workflow/config/config.yml` is the single source of truth for every pipeline.
- Machine-specific paths go in `config_local.yml` (gitignored, auto-loaded). Never put
  per-machine values in `config.yml`.
- HPC: `config_hpc.yml` (tracked, Hydrax values; `config_hpc.yml.example` for other clusters), passed via
  `GFM_CONFIG_PATH`.
- `*_materialized.yml` are generated; never hand-edit them.

## Environments
- `gfm` (environment.yml): main pipeline, analysis, validation, and all plotting
  (including sfincs_tiles plot scripts). Snakemake pinned to 7.32.4.
- `hydromt-sfincs-dev` (environment_sfincs.yml): sfincs_tiles model build/run only.
  Its matplotlib is broken; never plot in it.
- On the Windows machine this env is named `gfm_python_preprocessing` (outdated name, same env):
  `C:/Users/schlumbe/AppData/Local/miniforge3/envs/gfm_python_preprocessing/python.exe`.
  The PATH `python` there is the Windows Store stub; always use the full interpreter path.

## Commands
- Dry run: never on the full DAG (it is split into parallel HPC jobs; a full local `-n all`
  does not finish). Use the subset config instead, via
  `GFM_CONFIG_PATH=snakemake_workflow/config/bangkok_chao_phraya_materialized.yml`
  `snakemake -n all --cores 1` (~10 s). Baseline: `.cleanup_baseline/snakemake_n_all_bangkok_chao_phraya_profile.txt`.
- `profiles/default/config.yaml` (auto-loaded) sets `rerun-triggers: [mtime, params, input, software-env]`:
  code/comment edits never re-run outputs. A real logic change needs its outputs removed or re-run explicitly.
- Local run: `python run_eikonal_dag.py`  or  `snakemake all --cores 4 --resources mem_mb=8000`
- Tests: `pytest -m "not needs_pdrive"` in `gfm` (sfincs_* tests are skipped automatically);
  `pytest tests/sfincs_* -m "not needs_pdrive"` in `hydromt-sfincs-dev`.
- Lint: `ruff check .` must stay clean.
- Progress monitors (`check_*_progress.py`, `sfincs_tiles/check_calibration_progress.py`):
  the user runs these themselves, usually on Hydrax. Give the command, don't run it.

## Data
Inputs/outputs live on the P:\ share (`/p/11212688-004-global-floodmaps/` on Hydrax).
Never write to, move, or delete anything on P:\ or the model output directories.

## Do not touch without asking
- Output path patterns and wildcard constraints in `snakemake_workflow/rules/*.smk`:
  changing them orphans existing outputs on P:\.
- `snakemake_workflow/config/config.yml` defaults, `data_catalog_*.yml`.
- Pinned versions in `environment.yml` / `environment_sfincs.yml`.
- `sfincs_tiles/` while a Hydrax batch is running (ask whether one is).

## Conventions
- The per-tile solve lives in `src/eikonal_runner.run_tile_scenario`; `run_eikonal.py`
  (Snakemake) and `run_eikonal_cli.py` (HPC) are thin wrappers - change behaviour there, not in them.
- Docstrings and comments follow the documentation standard below.
- Stale code and one-off diagnostics are deleted (git history keeps them), not archived.

## Documentation standard

Documentation is for future readers of the code, not a record of this session.

- Never write change history into code or docs: no "previously", "updated to",
  "now handles", "refactored", "fixed", phase/step references, or dates.
  Git is the changelog. If a summary of changes is needed, put it in the
  commit message or say it in chat.
- Docstrings: one summary line in the imperative, max ~80 chars. Add further
  lines ONLY for things not inferable from the signature: units, datums and
  sign conventions, assumptions, valid ranges, the equation or paper the
  function implements, non-obvious side effects or failure modes.
- No Parameters/Returns sections that only repeat the type hints. Document a
  parameter only when its name and type do not make it obvious.
- No docstrings on private helpers, trivial wrappers, dunder methods, or tests
  unless a non-obvious assumption is involved.
- Inline comments explain WHY, never WHAT. Delete comments that paraphrase the
  code. Keep comments that encode a physical/numerical assumption or a
  reference.
- Module docstring: one or two lines on purpose and main entry point. No file
  listings, no author/date headers.
- README per package: purpose, how to run, key config options. Target under
  40 lines. Top-level README: under 120 lines. No feature lists, no "overview"
  prose that repeats the directory names.
- Do not create new .md files (NOTES, SUMMARY, CHANGES, DESIGN, etc.) unless
  asked. Report in chat instead.
- Prefer self-explanatory names over comments. If a comment is needed to
  explain a name, rename.
- When unsure whether to document something: leave it out. Missing docs are
  cheap to add; noise is expensive to remove.

## Cleanup state
`AUDIT.md` describes the current repo state, the rules for changes, the verification
toolkit and the areas still to sweep. Changes stay behaviour-preserving; real
(result-changing) issues go to `TODO.md` instead of being made. `.cleanup_baseline/`
(gitignored) holds reference outputs and local test outputs.
