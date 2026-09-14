# GFM — Global Flood Model

A tile-based coastal flood model pipeline, extending Deltares' Aqueduct Coastal
Flooding methodology to global scale. For every tile in a global tile grid it
prepares DEM/mask/friction/water-level-boundary inputs, runs a Fast Sweeping
Method eikonal flood solver per (tile, return period, sea-level-rise
scenario), merges results into spatial chunks, and runs exposure/adaptation
and benchmark-validation analyses on top.

## Repo layout

The repo separates what Snakemake actually orchestrates from three standalone
pipelines that just happen to share the same library code:

```
Snakefile                  entry point for the Snakemake DAG
run_pipeline.py            wraps `snakemake` with an OOM-retry loop
check_{preprocess,simulation,postprocess}_progress.py
                            read-only progress monitors

src/                       shared library, imported by everything below
                            (sys.path.insert, not a pip-installed package)

preparation/                standalone, NOT part of the Snakemake DAG
  run_preparation.py         builds the tile grid + boundary-condition inputs
                              the DAG reads, run this first

analysis/                   standalone, NOT part of the Snakemake DAG
  run_analysis.py             exposure/adaptation analysis on the DAG's output

validation/                 standalone, NOT part of the Snakemake DAG
  run_validation.py           benchmark HR/FAR/CSI comparison (ESP/FRA/NOR)

snakemake_workflow/         ONLY what Snakemake itself touches
  rules/                     rule definitions (*.smk)
  scripts/                   the rules' `script:` targets + HPC/SLURM dispatch helpers
  config/                    config.yml (single source of truth for every
                              pipeline above), config_local.yml (gitignored
                              per-machine overrides), config_hpc.yml, data catalogs
  hpc.md, memory.md          HPC setup notes / running engineering-notes log

tests/                      calibration studies, standalone regression
                              scripts, one-off diagnostic folders (no pytest
                              harness configured anywhere in this repo)

scratch/                    unreferenced one-off scripts, kept for reference only

docs/                       design docs and validation methodology notes
```

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
   Monitor with `check_preprocess_progress.py` / `check_simulation_progress.py`
   / `check_postprocess_progress.py` (read-only, safe to run anytime).
3. **Analysis** (exposure/adaptation, after the DAG):
   ```
   python analysis/run_analysis.py
   ```
4. **Validation** (benchmark HR/FAR/CSI comparison, after the DAG):
   ```
   python validation/run_validation.py --countries ESP FRA NOR
   ```

## Configuration

`snakemake_workflow/config/config.yml` is the single source of truth for
every pipeline above (not just the Snakemake DAG). Override machine-specific
values (`paths.root`, `paths.code_root`, ...) in
`snakemake_workflow/config/config_local.yml` — gitignored, auto-loaded if
present, never touch the committed `config.yml` for a per-machine value. See
`snakemake_workflow/config/config_hpc.yml.example` for the HPC/SLURM path.

## Import convention

Every entry point uses `sys.path.insert(0, ...)` to reach `src/` (and, where
relevant, `analysis/`) — this is a flat script layout, not an installable
Python package. If you add a new script, follow the existing pattern in a
sibling file rather than introducing a different import mechanism.

## Tests

`tests/` holds standalone regression scripts (one per `src/` function under
test, run directly with `python <script>.py`, plain `assert` statements — no
pytest configured anywhere in this repo) plus calibration-study tooling/data
and one-off investigation folders (e.g. `martinique_diagnostics/`,
`norway_diagnostics/`).

## Further reading

- `docs/flood_depth_method.md` — the flood-solver methodology.
- `docs/python_vs_julia_qa.md` — validation of the Python port against the
  original Julia reference implementation.
- `docs/flood_extent_validation_plan.md` / `docs/flood_extent_validation_caveats.md`
  — the benchmark-validation methodology and known caveats.
- `docs/calibration_sweep_plan.md` — a planned (not yet implemented)
  parameter-sensitivity study.
- `snakemake_workflow/hpc.md` — HPC/SLURM setup and dispatch.
- `snakemake_workflow/memory.md` — running engineering-notes/design-log for
  the codebase (what each part does and why, not development history).
