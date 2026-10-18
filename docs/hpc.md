# Running on a SLURM cluster

The full grid runs as generated sbatch scripts: preprocessing batches, simulation waves, postprocessing and exposure analysis.
The solver is pure Python (Numba), so any number of tiles can run concurrently across nodes.
The generators can run on the local machine (writing scripts for the cluster) or on the cluster itself.

## One-time site setup

Copy the two example site files to their gitignored names and fill in your cluster's values; every HPC generator and shell driver stops with a message naming the missing file:

```
cp snakemake_workflow/config/config_hpc.example.yml snakemake_workflow/config/config_hpc.yml
cp sfincs_tiles/site_env.example.sh sfincs_tiles/site_env.sh
```

The examples hold the values of Hydrax (Deltares) as a worked example.

`config_hpc.yml` is layered over `config.yml` + `config_local.yml` and holds only overrides:

| Key | Value |
|---|---|
| `paths.root` | the cluster mount of `<data_root>` (same storage as the local `paths.root`) |
| `paths.code_root` | absolute path of the cluster checkout (`<code_root>`), no `~` |
| `hpc.sbatch.partition` / `account` / `mem` / `time` / `cpus_per_task` | default job resources; `""` omits the `#SBATCH` line |
| `hpc.sbatch.env_activate_cmd` | shell line activating the `gfm` env in every job |
| `hpc.wave_tiers` | optional per-hop node count and resources (semantics in `config.yml`) |
| `hpc.calibration_sbatch`, `hpc.flood_totals_sbatch` | resources of the SFINCS calibration and flood-totals jobs; empty falls back to `hpc.sbatch` |

```
paths:
  root: "<data_root>"
  code_root: "<code_root>"
hpc:
  sbatch:
    partition: "<partition>"
    mem: "7G"
    env_activate_cmd: 'module load miniconda && eval "$(conda shell.bash hook)" && conda activate gfm'
```

`sfincs_tiles/site_env.sh` holds the same roots (`GFM_DATA_ROOT`, `GFM_CODE_ROOT`, which must equal `paths.root` / `paths.code_root`), the python of each env, the SFINCS container image and runtime, and the resources of the calibration post job.
Keep every line in the `NAME="${NAME:-default}"` form: a set environment variable overrides the value.

`hpc.n_nodes` caps the scripts per wave, and `hpc.sbatch.cpus_per_task` is the real concurrency: tiles solved in parallel per simulation job (each solve is single-threaded) and `snakemake --cores` per preprocessing job.

## Hop ordering

A `hop_distance >= 1` tile is seeded from its lower-hop neighbours' output for the same scenario, so wave `N+1` may start only once every wave-`N` job has ended.
The generated `submit_waves.sh` submits each wave with `--dependency=afterany:<all prior wave job ids>`.
`afterany` (not `afterok`) lets a failed tile fail only its own downstream neighbours, which a `--resume` dispatch picks up.

## Generators

All write to `hpc.jobs_dir` (default `<data_root>/model_outputs/hpc_jobs`) and print the submit command; submit from a login node with `bash <script>`.

| Generator | Writes | Submit |
|---|---|---|
| `python snakemake_workflow/scripts/generate_hpc_preprocess_job.py` | `preprocess_batch_*.sbatch` + `*_targets.txt`, `generate_jobs_and_dispatch.sbatch` | `submit_preprocess_and_dispatch.sh` |
| `snakemake generate_eikonal_jobs --cores N` | `wave{H}_batch_{id}.sbatch` + `*_triples.csv`, `resolved_config.yml` (after verifying preprocessing) | `submit_waves.sh` |
| `python snakemake_workflow/scripts/generate_hpc_simulation_jobs.py [--resume]` | the same wave scripts without the DAG check, honouring `hpc.wave_tiers`; `--resume` batches only missing outputs | `submit_waves.sh` / `submit_resume_waves.sh` |
| `python snakemake_workflow/scripts/generate_hpc_postprocess_job.py` | merge, exposure-grid and flood-fraction batches, exposure analysis, flood totals + plots | `submit_postprocess_and_exposure.sh` |
| `python snakemake_workflow/scripts/generate_hpc_exposure_jobs.py` | exposure analysis only | `submit_exposure_analysis.sh` |
| `python snakemake_workflow/scripts/generate_hpc_flood_totals_and_plots_job.py` | one flood-totals + plots job | printed `sbatch` command |

The chain is self-continuing: `submit_preprocess_and_dispatch.sh` builds the shared inputs (geoid offset, station caches) on the submitting node, runs the preprocessing batches in parallel, then a job generates and submits the simulation waves.
After the last wave, `submit_waves.sh` runs one more job that generates and submits the postprocessing and exposure chain.
So, after `preparation/run_preparation.py`, a full run is:

```
python snakemake_workflow/scripts/generate_hpc_preprocess_job.py
bash <data_root>/model_outputs/hpc_jobs/submit_preprocess_and_dispatch.sh
```

Each Snakemake call inside a job is retried as a whole against transient read errors on the shared mount.
Generating the scripts again after changing anything under `hpc:` rewrites them; scripts of batches that no longer exist stay in `jobs_dir` but are not submitted.
Without Snakemake metadata for the old scripts (a fresh checkout), force regeneration with `--forcerun generate_eikonal_jobs`.

## Failures and retries

A failed `(tile, return_period, waterlevel_name)` does not abort its batch; it is appended to `logs/wave{H}_batch_{id}_failures.txt`.
A hop-1 job whose neighbour output is missing fails without writing output; a neighbour without flooding in the overlap gives a real all-zero result.
Collect everything into `failure_report.csv`:

```
python snakemake_workflow/scripts/collect_hpc_failures.py [--out <report.csv>]
```

`job_error` rows need a rerun; `no_boundary_stations`, `no_upstream_flooding` and `oom_too_large` rows are deliberate results.
Rerun the missing outputs with `generate_hpc_simulation_jobs.py --resume` (cancel still-running jobs of the original dispatch first), or one item by hand:

```
python <code_root>/snakemake_workflow/scripts/run_eikonal_cli.py \
    --config <data_root>/model_outputs/hpc_jobs/resolved_config.yml \
    --tile-id <id> --return-period <RP> --waterlevel-name <SLR>
```

Monitor with `squeue` / `sacct` and the progress monitors in [running.md](running.md#progress-monitors).

## Postprocessing by hand

For a small run, or after a resume you want to control, run postprocessing on any machine with the shared mount instead of the generated chain:

```
snakemake postprocess --cores N
```

The HPC results sit at the paths `run_eikonal` writes, so Snakemake treats the simulation as up to date.

## SFINCS calibration and solver studies

```
python sfincs_tiles/generate_calibration_jobs.py stage1          # then: bash <printed submit_chain.sh>
bash sfincs_tiles/rerun_calibration.sh --part eikonal|sfincs
python calibration_studies/generate_calibration_batch_jobs.py sweep_budget|obstacle_coupling
```

Calibration resources come from `hpc.calibration_sbatch`, roots, pythons and the container from `site_env.sh`; the solver studies default to `hpc.sbatch`.
