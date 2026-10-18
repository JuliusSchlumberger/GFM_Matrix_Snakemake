# Running on a SLURM cluster

The full grid runs as generated sbatch scripts: preprocessing batches, simulation waves, postprocessing and exposure analysis.
The solver is pure Python (Numba), so any number of tiles can run concurrently across nodes.
The generators can run on the local machine (writing scripts for the cluster) or on the cluster itself.

## One-time site setup

Create the `gfm` env on the cluster from the same `environment.yml` ([install.md](install.md)); the jobs activate it with `hpc.sbatch.env_activate_cmd`.
No rule uses a Snakemake `conda:` directive, so the env plays no part in Snakemake's rerun decisions.

Copy the example site files to their gitignored names and fill in your cluster's values:

```
cp snakemake_workflow/config/config_hpc.example.yml snakemake_workflow/config/config_hpc.yml
cp sfincs_tiles/site_env.example.sh sfincs_tiles/site_env.sh
```

Every HPC generator needs `config_hpc.yml`; only the `sfincs_tiles/` and `calibration_studies/` drivers need `site_env.sh`.
Each stops with a message naming the missing file.
The examples hold the values of Hydrax (Deltares) as a worked example.

`config_hpc.yml` holds only overrides:

| Key | Value |
|---|---|
| `paths.root` | the cluster mount of `<data_root>` |
| `paths.code_root` | absolute path of the cluster checkout (`<code_root>`), no `~` |
| `hpc.sbatch.partition` / `account` / `mem` / `time` / `cpus_per_task` | default job resources; `""` omits the `partition` / `account` line; `mem` (per job, `--mem`) and `time` are required |
| `hpc.sbatch.env_activate_cmd` | shell line activating the `gfm` env in every job |
| `hpc.n_nodes` | most batches per wave; a wave's tiles are split evenly over them |
| `hpc.jobs_dir` | where the generators write scripts, `resolved_config.yml`, submit scripts and `logs/` (default `<data_root>/model_outputs/hpc_jobs`) |
| `hpc.wave_tiers` | optional per-hop resources, read only by `generate_hpc_simulation_jobs.py`: a list of tiers, each with `hops`, `n_nodes`, `partition`, `cpus_per_task`, `mem`, `time`; the first tier whose `hops` contains the wave's `hop_distance` wins, `n_nodes: null` means one job per tile, and the last tier must be the catch-all `hops: null` |
| `hpc.calibration_sbatch`, `hpc.flood_totals_sbatch` | resources of the SFINCS calibration and flood-totals jobs; empty falls back to `hpc.sbatch` |

```
paths:
  root: "<data_root>"
  code_root: "<code_root>"
hpc:
  sbatch:
    partition: "<partition>"
    mem: "<mem>"
    env_activate_cmd: '<command activating the gfm env>'
```
```
paths:
  root: "/data/gfm"
  code_root: "/home/me/gfm"
hpc:
  sbatch:
    partition: "1vcpu"
    mem: "7G"
    env_activate_cmd: 'module load miniconda && eval "$(conda shell.bash hook)" && conda activate gfm'
```

`hpc.sbatch.cpus_per_task` is the real concurrency: tiles solved in parallel per simulation job (each solve is single-threaded) and `snakemake --cores` per preprocessing job.
One solve peaks at about `base_mb + bytes_per_pixel x DEM pixels` (`simulation.eikonal_mem_estimate`), so give a simulation job at least `cpus_per_task` times that for its largest tile.
The example's `7G` is the usable memory of a Hydrax 1-vCPU node, not a pipeline requirement.

`sfincs_tiles/site_env.sh` keeps every line in the `NAME="${NAME:-default}"` form, so a set environment variable overrides the value:

| Variable | Meaning |
|---|---|
| `GFM_DATA_ROOT`, `GFM_CODE_ROOT` | cluster data root and checkout; must equal `paths.root` / `paths.code_root` of `config_hpc.yml` (the calibration generators check) |
| `GFM_PYTHON` | python of the `gfm` env |
| `GFM_SFINCS_PYTHON` | python of the `hydromt-sfincs-dev` env |
| `SFINCS_IMAGE` | SFINCS container image, example `docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release` |
| `CONTAINER_CMD` | container runtime, run as `<cmd> exec -B <dir>:/mnt/data <image> sfincs`; example `apptainer` |
| `CAL_PARTITION`, `CAL_CPUS`, `CAL_MEM`, `CAL_TIME` | partition, CPUs (also the dedup's `--workers`), memory and wall-clock limit of `rerun_calibration.sh`'s post job |

## Config layering on a cluster

Each generator loads its `--config` (default `config.yml`) twice:

- with the sibling `config_local.yml`: the paths as the machine running the generator sees them; the scripts are written there, under `hpc.jobs_dir`;
- with `config_local.yml` and then the sibling `config_hpc.yml`: the paths as the compute nodes see them; these are embedded in the scripts and written, fully expanded, to `<jobs_dir>/resolved_config.yml`, which every job reads through `GFM_CONFIG_PATH`.

So any checkout that runs a generator needs both files in `snakemake_workflow/config/`; on the cluster checkout, `config_local.yml` holds the same paths as `config_hpc.yml`.
A materialized scenario config sits in the same directory, so `config_hpc.yml` layers over it too.
Generating on a workstation requires it to see the same storage as the cluster (it reads the tile grid and writes the scripts there); without shared storage, run the generators on the cluster.
Run `preparation/run_preparation.py` before the first dispatch, on any machine that sees `<data_root>`; its outputs (VRTs with relative sources, tile grid, water-level NetCDFs) are read from there.

## Hop ordering

A `hop_distance >= 1` tile is seeded from its lower-hop neighbours' output for the same scenario, so wave `N+1` may start only once every wave-`N` job has ended.
The generated `submit_waves.sh` submits each wave with `--dependency=afterany:<all prior wave job ids>`.
`afterany` (not `afterok`) lets a failed tile fail only its own downstream neighbours, which a `--resume` dispatch picks up.

## Generators

All write to `hpc.jobs_dir` and print the submit command; submit from a login node with `bash <script>`.

| Generator | Writes | Submit |
|---|---|---|
| `python snakemake_workflow/scripts/generate_hpc_preprocess_job.py [--config <config.yml>] [--calibration]` | `preprocess_batch_*.sbatch` + `*_targets.txt`, `generate_jobs_and_dispatch.sbatch`, `resolved_config.yml` | `submit_preprocess_and_dispatch.sh` |
| `snakemake generate_eikonal_jobs --cores N` | `wave{H}_batch_{id}.sbatch` + `*_triples.csv`, `resolved_config.yml` (after verifying preprocessing) | `submit_waves.sh` |
| `python snakemake_workflow/scripts/generate_hpc_simulation_jobs.py [--config <config.yml>] [--resume]` | the same wave scripts without the DAG check, honouring `hpc.wave_tiers`; `--resume` batches only missing outputs | `submit_waves.sh` / `submit_resume_waves.sh` |
| `python snakemake_workflow/scripts/generate_hpc_postprocess_job.py` | merge, exposure-grid and flood-fraction batches, exposure analysis, flood totals + plots | `submit_postprocess_and_exposure.sh` |
| `python snakemake_workflow/scripts/generate_hpc_exposure_jobs.py` | exposure analysis only | `submit_exposure_analysis.sh` |
| `python snakemake_workflow/scripts/generate_hpc_flood_totals_and_plots_job.py` | one flood-totals + plots job | printed `sbatch` command |

The chain is self-continuing: `submit_preprocess_and_dispatch.sh` activates the env, builds the shared inputs (geoid offset raster, one station cache per return period and SLR level) on the submitting node, runs the preprocessing batches in parallel, then a job generates and submits the simulation waves.
After the last wave, `submit_waves.sh` runs one more job that generates and submits the postprocessing and exposure chain.
So, after `preparation/run_preparation.py`, a full run is:

```
python snakemake_workflow/scripts/generate_hpc_preprocess_job.py
bash <data_root>/model_outputs/hpc_jobs/submit_preprocess_and_dispatch.sh
```

The wave-generation job of that chain runs `snakemake generate_eikonal_jobs`, which verifies every preprocessing output first and always reads `config.yml`, not a scenario config.
With `--calibration` it runs `generate_hpc_simulation_jobs.py` instead, which honours the `--config` scenario and `hpc.wave_tiers` and skips the check.
Run `generate_hpc_simulation_jobs.py` by hand only when preprocessing is known to be complete, or with `--resume`.

Each Snakemake call inside a job is retried as a whole against transient read errors on the shared mount.
Generating the scripts again after changing anything under `hpc:` rewrites them; scripts of batches that no longer exist stay in `jobs_dir` but are not submitted.
Without Snakemake metadata for the old scripts (a fresh checkout), force regeneration with `--forcerun generate_eikonal_jobs`.

## A small scenario on the cluster

Test a site setup with the Bangkok scenario ([running.md](running.md#4-bangkok--chao-phraya-case-study), steps 1-2 first).
Its scenario config writes the job scripts to `<data_root>/bangkok_chao_phraya/model_outputs/hpc_jobs` (`hpc.jobs_dir`), apart from production:

```
python snakemake_workflow/scripts/materialize_configs.py bangkok_chao_phraya
python snakemake_workflow/scripts/generate_hpc_preprocess_job.py \
    --config snakemake_workflow/config/bangkok_chao_phraya_materialized.yml --calibration
bash <data_root>/bangkok_chao_phraya/model_outputs/hpc_jobs/submit_preprocess_and_dispatch.sh
```

## Failures and retries

A failed `(tile, return_period, waterlevel_name)` does not abort its batch; it is appended to `<jobs_dir>/logs/wave{H}_batch_{id}_failures.txt`.
Valid `return_period` and `waterlevel_name` values: [running.md](running.md#scenario-names).
A hop-1 job whose neighbour output is missing fails without writing output; a neighbour without flooding in the overlap gives a real all-zero result.
Collect everything into `failure_report.csv`:

```
python snakemake_workflow/scripts/collect_hpc_failures.py [--out <report.csv>]
```

`job_error` rows need a rerun; `no_boundary_stations`, `no_upstream_flooding` and `oom_too_large` rows are deliberate results.
Rerun the missing outputs with `generate_hpc_simulation_jobs.py --resume`, after cancelling the original dispatch's remaining jobs (named after their batch, `wave{H}_batch_{id}`):

```
squeue -u $USER -h -o "%i %j" | awk '$2 ~ /^wave[0-9]+_batch_/ {print $1}' | xargs -r scancel
```

Or rerun one item by hand:

```
python <code_root>/snakemake_workflow/scripts/run_eikonal_cli.py \
    --config <jobs_dir>/resolved_config.yml \
    --tile-id <id> --return-period <RP> --waterlevel-name <SLR>
```
```
python /home/me/gfm/snakemake_workflow/scripts/run_eikonal_cli.py \
    --config /data/gfm/model_outputs/hpc_jobs/resolved_config.yml \
    --tile-id 3930 --return-period RP100 --waterlevel-name SLR_500
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

`stage1` writes the candidate shards and their screen, boundary-line and pre-flight jobs plus a bridge job that selects the tiles and runs `stage2 --submit`, which spreads the unfinished tiles over jobs.
Results land in `<data_root>/calibration_roughness_SFINCS/` (`--base-dir-name`).
`rerun_calibration.sh` reruns part of a finished calibration in place: `--part eikonal` only the eikonal sweep (with `POST=1`, the default, followed by the dedup and sweep analyses), `--part sfincs` the forcing, SFINCS model build and run (run the analyses by hand afterwards); `DRY_RUN=1` reports what it would reset.
Calibration resources come from `hpc.calibration_sbatch`, roots, pythons and the container from `site_env.sh`; the solver studies default to `hpc.sbatch`.
