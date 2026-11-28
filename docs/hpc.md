# Running on a SLURM cluster

The full grid runs as generated sbatch scripts: preprocessing batches, simulation waves, postprocessing and exposure analysis.
The solver is pure Python (Numba), so any number of tiles can run concurrently across nodes.
Every job chain is written and submitted through one front door, `hpc/gfm_hpc.py`, run on the cluster login node with the `gfm` env's python.

## Site setup

Create the `gfm` env (and, for the SFINCS calibration, `hydromt-sfincs-dev`) on the cluster from `environment.yml` / `environment_sfincs.yml` ([install.md](install.md)).
No rule uses a Snakemake `conda:` directive, so the envs play no part in Snakemake's rerun decisions.

Copy the one site file to its gitignored name and fill in your cluster's values; the example holds those of Hydrax (Deltares):

```
cp snakemake_workflow/config/config_hpc.example.yml snakemake_workflow/config/config_hpc.yml
```

`config_hpc.yml` holds only overrides of `config.yml`:

| Key | Value |
|---|---|
| `paths.root` | the cluster mount of `<data_root>` |
| `paths.code_root` | absolute path of the cluster checkout (`<code_root>`), no `~` |
| `hpc.python.gfm`, `hpc.python.sfincs` | absolute interpreters of the `gfm` and `hydromt-sfincs-dev` envs |
| `hpc.account` | `#SBATCH --account` of every job; `""` omits it |
| `hpc.container.cmd`, `hpc.container.sfincs_image` | SFINCS runs as `<cmd> exec -B <dir>:/mnt/data <image> sfincs` |
| `hpc.state_root` | harness plan state and the HPC-view configs `gfm_hpc.py config --hpc` writes |
| `hpc.job_env` | shell lines every job runs after the interpreter exports |
| `hpc.profiles.<name>` | job resources (`partition`, `time`, `mem`, `cpus_per_task`, `account`), see [Resource profiles](#resource-profiles) |
| `hpc.wave_tiers` | optional per-hop resources of the simulation waves |
| `hpc.profiles.simulate_packed` | resources of the simulation jobs of a scenario with `hpc.wave_packing` |

No job activates an env: each starts with

```
export GFM_PY="<hpc.python.gfm>"
export SFINCS_PY="<hpc.python.sfincs>"
export PATH="<bin dir of the job's main env>:$PATH"
<hpc.job_env lines>
```

and calls `"$GFM_PY" script.py`, `"$SFINCS_PY" script.py` and `"$GFM_PY" -m snakemake`; subprocesses use `sys.executable`.

### Config layering on a cluster

Each generator loads its config twice:

- with the sibling `config_local.yml`: the paths as the machine running the generator sees them; the scripts are written there, under `hpc.jobs_dir`;
- with `config_local.yml` and then the sibling `config_hpc.yml`: the paths as the compute nodes see them; these are embedded in the scripts and written, fully expanded, to `<jobs_dir>/resolved_config.yml`, which every job reads (staged to node-local disk first).

So the cluster checkout needs both files in `snakemake_workflow/config/`, with `config_local.yml` holding the same paths as `config_hpc.yml`.
A materialized scenario config sits in the same directory, so `config_hpc.yml` layers over it too.
Run `preparation/run_preparation.py` (or `gfm_hpc.py product`) before the first dispatch; its outputs (VRTs with relative sources, tile grid, water-level NetCDFs) are read from `<data_root>`.

## The front door

```
python hpc/gfm_hpc.py run preprocess --scenario NAME [--preprocess-only | --fuse] [--submit]
python hpc/gfm_hpc.py run simulate|postprocess|exposure --scenario NAME [--resume] [--submit]
python hpc/gfm_hpc.py run flood-totals --scenario NAME [--submit]
python hpc/gfm_hpc.py product [--submit]
python hpc/gfm_hpc.py studies sweep|obstacle --source landcover [uniform] [--extra-args '...'] [--submit]
python hpc/gfm_hpc.py studies wet-tiles|figures --source landcover [uniform] [--submit]
python hpc/gfm_hpc.py sfincs stage1|stage2|post|rerun [...] [--submit]
python hpc/gfm_hpc.py validate --scenario NAME [--only GROUP ...] [--submit]
python hpc/gfm_hpc.py progress preprocess|simulate|postprocess|sfincs --scenario NAME [--watch SECONDS]
python hpc/gfm_hpc.py failures --scenario NAME [--out <report.csv>]
python hpc/gfm_hpc.py config NAME [--hpc] [--key dotted.key]
```

`--scenario NAME` means `snakemake_workflow/config/<NAME>_materialized.yml` (`production`: `config.yml`), `--variant V` its flood-model variant `<NAME>__<V>_materialized.yml`; `--config PATH` takes any config file.
A materialized file older than its override, `config.yml` or `config_local.yml` gets a warning: rerun `materialize_configs.py`.
A generator writes its job scripts, a submit script and `<submit script>.jobs.json` (jobs, resources, dependencies) and prints the submit script; `--submit` also runs it.
`product`, `validate` and `studies figures` run in place, or with `--submit` as one job.
`config --hpc` writes the compute nodes' view to `<hpc.state_root>/configs/` and prints its path, for scripts called directly; `--key` prints one value.
The generators also run alone (`snakemake_workflow/scripts/generate_hpc_*.py`, `sfincs_tiles/generate_calibration_jobs.py`, `calibration_studies/generate_calibration_batch_jobs.py`), with the same options.

## The pipeline chain

```
python hpc/gfm_hpc.py run preprocess --scenario production --submit
```

runs the whole pipeline as one self-continuing chain:

1. The submit script builds the shared inputs on the login node (geoid offset raster, the GOCO06s absolute geoid when a local DEM source bridges through it, one station cache per return period and SLR level unless `boundary_conditions.stations_cache_prebuilt`), then queues the preprocessing batches: tiles split evenly, in tile-id order, over `hpc.n_nodes` (`hpc.single_preprocess_job`: one).
2. A dispatch job (afterany on every batch) checks that every batch target exists and stops the chain with the count and the first missing paths otherwise; it then writes and submits the simulation waves.
3. A tile with `hop_distance >= 1` is seeded from its lower-hop neighbours' output for the same scenario, so wave `N+1` waits (afterany) until every wave-`N` job has ended. A wave's tiles are split evenly, in tile-grid order, over `hpc.n_nodes`, the matching `hpc.wave_tiers` entry, or, for a scenario with `hpc.wave_packing`, as many jobs of its profile as the wave's solves need (see [Resource profiles](#resource-profiles)); a tile's return periods and SLR levels stay in one job.
4. After the last wave, one job writes and submits the postprocessing chain: `merge_chunk` -> `prepare_exposure_grid_chunk` -> `compute_flood_fraction_chunk` (whole chunks per job, longest-first by tile count, when there are at least as many chunks as nodes; otherwise three phases, each waiting for the previous), then the exposure analysis (pass 1 -> shares -> pass 2 -> write) and the flood-totals + plots job.

`afterany` (not `afterok`) lets a failed job fail only what reads its output, which a `--resume` dispatch picks up.
A scenario with `postprocessing.merge_only` (the local-input validation scenarios) gets the merges only: `merge_chunk`, or, with a `simulation.dem_source_manifest` (tiles in the local DEM's projected CRS, never reprojected), per return period and SLR level `postprocessing.native_merge_jobs` `preparation/merge_england_local_tiles.py` jobs over square `postprocessing.native_chunk_m` chunks in that CRS, then one job writing `<merged_outputs>/local_native/waterdepth_<RP>_<SLR>.vrt` over them, which the local scorers read.
`--preprocess-only` stops after step 1 (inputs for the SFINCS calibration); `--fuse` (every tile at hop 0) preprocesses and simulates each batch in one job.
Each Snakemake call inside a job is retried as a whole against transient read errors on the shared mount.

## Resource profiles

Every job gets its resources from one profile in `hpc.profiles`: `default`, overlaid with the profile's set values.
A job uses exactly its `cpus_per_task` cores: Snakemake runs with `--cores "$SLURM_CPUS_PER_TASK"`, every worker pool sizes itself with `process_pool.available_cpus()` (`SLURM_CPUS_PER_TASK` first), and SFINCS runs that many OpenMP threads.
Every job exports `OMP_NUM_THREADS=1` (and the OpenBLAS, MKL and numba equivalents), so N processes stay on N cores; SFINCS sets its own.
Size `mem` for the processes' resident memory with a margin; SLURM's figure also counts file cache, which the kernel frees under pressure.

| Profile | Jobs |
|---|---|
| `preprocess` | preprocessing batches |
| `simulate` | simulation waves (`cpus_per_task` = tiles solved at once), overridden per wave by `hpc.wave_tiers` |
| `simulate_packed` | simulation waves of a scenario with `hpc.wave_packing` |
| `postprocess`, `exposure`, `flood_totals` | postprocessing chain |
| `dispatch` | jobs that only write and submit the next chain |
| `solver_study`, `study_figures` | eikonal solver studies (`--time` computed from a worst-case estimate) and their figures |
| `sfincs_stage1`, `sfincs_bridge`, `sfincs_tile`, `sfincs_post` | SFINCS calibration (`--time` computed per `sfincs_tile` job) |
| `validation`, `product` | `validate --submit`, `product --submit` |

`--profile-set PROFILE.FIELD=VALUE` (repeatable) overrides one value for a run; it is written into the chain's `resolved_config.yml`, so jobs submitted later by the chain keep it.
One simulation solve peaks at about `base_mb + bytes_per_pixel x DEM pixels` (`simulation.eikonal_mem_estimate`), so give a simulation job at least `cpus_per_task` times that for its largest tile.
`hpc.wave_tiers` is a list of tiers with `hops`, `n_nodes`, `partition`, `cpus_per_task`, `mem`, `time`; the first tier whose `hops` contains the wave's `hop_distance` wins, `n_nodes: null` means one job per tile, and the last tier must be the catch-all `hops: null`.
A scenario's `hpc.wave_packing` (`solve_minutes`, `target_hours`, `profile`) replaces the tiers for that scenario: each wave gets ceil(solves x `solve_minutes` / (`cpus_per_task` x `target_hours` x 60)) jobs of `profile`, at most one per tile.

Each solve in a simulation job is stopped after `hpc.solve_timeout_min` (default 20) and logged as failed, so a hung shared-mount read cannot hold the job; raise it for scenarios whose largest tiles solve slower.

## Logs, failures and resume

A job named `<name>` logs to `<jobs_dir>/logs/<name>_%j.out` / `.err` (the exposure jobs under `<jobs_dir>/exposure/logs/`).
A failed `(tile, return_period, waterlevel_name)` does not abort its batch; it is appended to `<jobs_dir>/logs/<job>_failures.txt`.
Valid `return_period` and `waterlevel_name` values: [running.md](running.md#scenario-names).
A hop-1 job whose neighbour output is missing fails without writing output; a neighbour without flooding in the overlap gives a real all-zero result.

```
python hpc/gfm_hpc.py failures --scenario NAME        # <jobs_dir>/failure_report.csv
```

`job_error` rows need a rerun; `no_boundary_stations`, `no_upstream_flooding` and `oom_too_large` rows are deliberate results.
`run simulate --resume` batches only the missing outputs (`resume_wave*`); `run postprocess --resume` and `run exposure --resume` likewise per phase and per exposure batch.
Cancel the original dispatch's remaining jobs first (named after their batch, `wave{H}_batch_{id}`):

```
squeue -u $USER -h -o "%i %j" | awk '$2 ~ /^wave[0-9]+_batch_/ {print $1}' | xargs -r scancel
```

Or rerun one item by hand:

```
"$GFM_PY" <code_root>/snakemake_workflow/scripts/run_eikonal_cli.py \
    --config <jobs_dir>/resolved_config.yml \
    --tile-id <id> --return-period <RP> --waterlevel-name <SLR>
```

Monitor with `squeue` / `sacct` and `gfm_hpc.py progress` (the monitors of [running.md](running.md#progress-monitors)).
Generating again after changing anything under `hpc:` rewrites the scripts; scripts of batches that no longer exist stay in `jobs_dir` but are not submitted.

## A small scenario on the cluster

Test a site setup with the Bangkok scenario ([running.md](running.md#4-bangkok--chao-phraya-case-study), steps 1-2 first); its job scripts go to `<data_root>/bangkok_chao_phraya/model_outputs/hpc_jobs`, apart from production:

```
python snakemake_workflow/scripts/materialize_configs.py bangkok_chao_phraya
python hpc/gfm_hpc.py run preprocess --scenario bangkok_chao_phraya --submit
python hpc/gfm_hpc.py progress simulate --scenario bangkok_chao_phraya --watch 300
```

## Postprocessing by hand

For a small run, or after a resume you want to control, run postprocessing on any machine with the shared mount instead of the generated chain:

```
snakemake postprocess --cores N
```

The HPC results sit at the paths `run_eikonal` writes, so Snakemake treats the simulation as up to date.

## SFINCS calibration and solver studies

```
python hpc/gfm_hpc.py sfincs stage1 --n-tiles 1500 --submit
python hpc/gfm_hpc.py sfincs rerun --part models --phase sweep|duration [--reset-runs] [--dry-run] [--submit]
python hpc/gfm_hpc.py sfincs rerun --part sfincs [--dry-run] [--submit]
python hpc/gfm_hpc.py studies sweep --source landcover uniform --submit
```

`stage1` writes the candidate shards and their screen, boundary-line and pre-flight jobs plus a bridge job that selects the tiles and runs `stage2 --submit`, which spreads the unfinished tiles over jobs (longest first by estimated runtime) and adds a post job (variant selection, dedup, analyses).
Results land in `<data_root>/calibration_roughness_SFINCS/` (`--base-dir-name`).
`rerun` reruns part of a finished calibration in place: `--part models` one phase of the flood-model runs (`--reset-runs` deletes every model raster first; the post job, unless `--no-post`, writes `variant_selection.json` for the next phase, which a non-sweep phase requires), `--part sfincs` the forcing, SFINCS model build and run, then the sweep phase with its post job.
Both delete the tile markers that make `run_final_tile.sh` skip a tile and the stage-2 job files of the previous run, and rewrite the calibration's `resolved_config.yml` from the current config, so the rerun uses today's solver settings; `--dry-run` only reports what would be reset.
The raw analysis also writes `variant_runs_solver.csv` (per tile and run: inner sweep rounds and newly blocked cells per outer iteration, the last outer iteration that still blocked cells, solve seconds) and its per-variant summary `variant_runs_solver_summary.csv`.
The phases and the calibration/validation split are in [methods_04a](methods_04a_SFINCSvalidation.md#flood-model-variants).
`run_final_tile.sh` takes the roots, interpreters, container and `hpc.sfincs_timeout_s` from the exports of its job header.

The solver studies run on `calibration_500_tiles`: `studies sweep`, then `studies wet-tiles` (the tiles the obstacle study uses), `studies obstacle`, `studies figures`.

## Validation

A local-input scenario's `validation.runs` holds its validation commands in named groups; `{config}` stands for the scenario's HPC-view config, `{config:NAME}` for scenario NAME's.

`{variant}` is empty in a base config and `__<variant>` in a flood-model variant's (`materialize_configs.py --variants`), so each variant scores its own outputs.
`check-run` exits non-zero unless every (tile, RP, SLR) waterdepth and every merged output of a run exists and no tile ran out of memory (an out-of-memory solve fails and writes `<model_outputs>/oom_tiles/<tile>.txt`).

```
python hpc/gfm_hpc.py check-run --scenario wales_local_input [--variant eikonal_uniform]
python hpc/gfm_hpc.py validate --scenario wales_local_input [--variant eikonal_uniform] [--only scores] [--submit]
```

`hpc/plans/local_validation.yml` runs England, Wales and SF Bay with local and global input, base and three variants, through these checks to the scores and the local-vs-global exposure table.

## One core budget for several runs

`hpc/harness.py` runs a plan (`hpc/plans/*.yml`) of steps under one core budget (`max_cores`); run it with the `gfm` env's python, which the steps reuse as `$PY`.
A step is a launcher command run on the login node, or a job (`job:`), and waits for the steps in its `after:` list.
Inside a step and every job it starts, `hpc/bin/sbatch` stands first on PATH and queues each submission, answering with a placeholder job id; the controller submits a queued job once its dependencies have ended and its cores fit (a `<N>vcpu` partition counts N).
The front door, submit scripts and self-submitting jobs therefore run unchanged.
Steps marked `destructive` (deleting old outputs) run only with `--allow-destructive`; `requires:` lists site files checked before anything is queued.

```
tmux new -s harness
python hpc/harness.py run --plan hpc/plans/smoke.yml --poll-s 10   # once per cluster: a few minutes
python hpc/harness.py run --plan hpc/plans/<plan>.yml [--allow-destructive] [--only a,b]
python hpc/harness.py status --plan hpc/plans/<plan>.yml           # from another shell
python hpc/harness.py retry|mark-done --plan ... <step>             # after a failed or interrupted step
```

`hpc/plans/hydrax_rerun.yml` lists in its header the `--only` selection for running only the eikonal solver studies.

State (step states, queued jobs, logs) lives in the plan's `state_dir`, so a stopped controller resumes where it was; a launcher step whose command was still running when the controller stopped comes back as `interrupted` (retry or mark-done it); one whose command had ended keeps waiting on its jobs. `status` counts each step's jobs since the step last started, with earlier attempts as `(previous runs: ...)`.
A job asking for more cores than the budget is refused (`TOO_BIG`), and one with an `afterok` on a failed job is never submitted (`DEP_NEVER`).
