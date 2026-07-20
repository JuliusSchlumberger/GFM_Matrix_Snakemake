# Running the pipeline on an HPC

Aqueduct (`aqueduct.exe`) can only run one instance at a time **per
machine** — its Julia JIT crashes if two instances share one process space
(`rules/simulation.smk`'s `run_aqueduct` docstring). That's not a global
constraint, so separate HPC nodes can each run their own serial stream of
Aqueduct invocations concurrently. `generate_aqueduct_jobs` turns the
simulation stage into a set of per-node sbatch scripts instead of having
Snakemake invoke `aqueduct.exe` directly.

Three stages, run separately:

1. **Preprocessing + job generation** — Snakemake, on a many-core prep/login node.
2. **Simulation** — SLURM, submitted manually, across compute nodes.
3. **Postprocessing** — Snakemake, back on the machine with the shared filesystem.

## One-time setup

1. Copy the HPC path override template and fill in your cluster's mount/build paths:
   ```
   cp snakemake_workflow/config/config_hpc.yml.example snakemake_workflow/config/config_hpc.yml
   ```
   Set `paths.root` (Linux mount of the same storage as the local `paths.root`),
   `paths.code_root` (Linux checkout of this repo — used to locate
   `run_aqueduct_cli.py`), and `paths.aqueduct_root`/`simulation.aqueduct_executable`
   (the **Linux** Aqueduct build — a separate binary from `aqueduct.exe`).
   Git-ignored, like `config_local.yml`.

2. Fill in the placeholders under `hpc:` in `config.yml`:
   `n_nodes` (how many sbatch scripts to generate), and `sbatch.partition` /
   `account` / `time` / `mem` / `cpus_per_task` / `env_activate_cmd` (the
   module-load/conda-activate line each sbatch script runs before calling
   Python).

## 1. Preprocess + generate jobs

```
snakemake generate_aqueduct_jobs --cores N
```

Runs every preprocessing rule (DEM/mask/friction/boundaries/TOML, one set
per tile) in parallel across `--cores N`, then writes to
`{root}/model_outputs/hpc_jobs/`:

- `aqueduct_batch_000.sbatch` … `aqueduct_batch_{n_nodes-1}.sbatch` — one
  script per node, each looping sequentially over its tiles' full
  return_period × SLR set (a tile's jobs always stay on one node).
- `resolved_config.yml` — the fully Linux-path-expanded config each sbatch
  job's Python call reads.
- `submit_all.sh` — loops `sbatch` over every generated script.
- `logs/` — sbatch stdout/stderr and a per-batch failure log land here once jobs run.

Tiles/scenarios with no boundary stations are resolved immediately during
this step (nodata placeholder written locally, same as `run_aqueduct` does
today) and never show up in any sbatch script.

## 2. Submit to SLURM

```
bash {root}/model_outputs/hpc_jobs/submit_all.sh
```

or submit individual `aqueduct_batch_*.sbatch` files by hand. Each job calls
`run_aqueduct_cli.py` once per `(tile, return_period, waterlevel_name)`,
sequentially (not backgrounded — this is what keeps the per-machine JIT
constraint satisfied). A failed call doesn't abort the rest of that node's
batch; it's appended to `logs/batch_{id}_failures.txt` instead. Aqueduct's
own OOM handling (mark tile, skip remaining scenarios, write nodata
placeholder) works exactly as it does locally — it's just discovered live on
whichever node hits it.

Monitor with `squeue`/`sacct` as usual. To retry one failed job by hand:

```
python {code_root}/snakemake_workflow/scripts/run_aqueduct_cli.py \
    --config {root}/model_outputs/hpc_jobs/resolved_config.yml \
    --tile-id <id> --return-period <RP..> --waterlevel-name <SLR_..>
```

## 3. Postprocess

Once all batches finish, back on the machine with the shared mount:

```
snakemake postprocess --cores N
```

Results land at the same `model_outputs/{tile_id}/results/
waterdepth_{rp}_{slr}.tif` paths `run_aqueduct` would have produced, so
Snakemake sees them as already up to date and proceeds straight to merging/
plotting — no different from a fully-local run.

## Notes

- `run_aqueduct` itself is untouched — still the right choice for a small
  local/debug run, no SLURM involved.
- Re-running `generate_aqueduct_jobs` after changing `hpc.n_nodes` won't
  regenerate scripts on its own (Snakemake sees the old outputs as already
  present) — delete `model_outputs/hpc_jobs/` first, or use `--forcerun
  generate_aqueduct_jobs`.
- The per-job TOML files need no path rewriting for Linux — they only
  contain relative filenames (`input_dir = "."`), resolved against the
  TOML's own directory, so the same files preprocessing already wrote work
  unchanged from either OS.
