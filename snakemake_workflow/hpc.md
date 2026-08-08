# Running the pipeline on an HPC

The flood solver is pure Python (`src/flood_model.py` + `src/eikonal.py`, a
Numba JIT) - no compiled executable and no per-machine single-instance
constraint, so any number of tiles can run concurrently across nodes.

The one real ordering constraint is wave-based hinterland forcing
(`src/tile_chunking.compute_run_order`'s `hop_distance` column, see
`rules/simulation.smk`'s docstring): a `hop_distance >= 1` tile is seeded
from a strictly-lower-`hop_distance` neighbour's own output for the SAME
scenario (`src/boundaries.collect_neighbor_wave_seeds`). Wave `N+1` can
therefore only start once every wave-`N` job has reached a terminal state -
not "the neighbour happened to finish first". `generate_aqueduct_jobs`
groups tiles by wave, generates one set of sbatch scripts per wave, and
writes a `submit_waves.sh` driver that submits each wave to SLURM with a
`--dependency=afterany:<every prior-wave job id>` barrier.

Three stages:

1. **Preprocessing + job generation** — Snakemake, on a many-core prep/login node.
2. **Simulation** — SLURM, submitted via `submit_waves.sh`, across compute nodes.
3. **Postprocessing** — Snakemake, back on the machine with the shared filesystem.

Stage 1 can run either on your local preprocessing machine (as below) or as
one more Hydrax job that also submits stage 2 for you — see "Bundling
preprocessing + dispatch into one Hydrax job" below.

## One-time setup

1. Copy the HPC path override template and fill in your cluster's mount/checkout paths:
   ```
   cp snakemake_workflow/config/config_hpc.yml.example snakemake_workflow/config/config_hpc.yml
   ```
   Set `paths.root` (Linux mount of the same storage as the local `paths.root`)
   and `paths.code_root` (Linux checkout of this repo — used to locate
   `run_aqueduct_cli.py`). Git-ignored, like `config_local.yml`. Not needed
   at all if you're running stage 1 directly ON Hydrax (see below) — in
   that case `config_local.yml` alone already points at Hydrax's own paths.

2. Fill in the placeholders under `hpc:` in `config.yml`:
   `n_nodes` (max sbatch scripts to generate PER WAVE, PER SIZE CLASS — a
   group with fewer tiles than `n_nodes` gets one script per tile instead,
   never an empty one), `sbatch.partition` / `account` / `time` / `mem` /
   `cpus_per_task` / `env_activate_cmd` (the per-tile simulation jobs for
   most tiles), `sbatch_large.*` (same keys, a bigger-RAM partition for
   tiles at/above `large_tile_pixel_threshold` — see "Tile-size routing"
   below), and `preprocess_sbatch.*` (same keys again, for the optional
   bundled preprocessing job below — typically a bigger partition, since
   preprocessing parallelises via plain Snakemake `--cores` on one node).

### Tile-size routing

Hydrax's regular partitions all scale RAM at a fixed 8GB/vCPU (`1vcpu`=8GB,
`4vcpu`=32GB, `16vcpu`=128GB, `24vcpu`=192GB, `44vcpu`=352GB, `60vcpu`=480GB).
`hpc.sbatch` is meant for the common case (e.g. `1vcpu` — by far the
most-provisioned partition); `hpc.large_tile_pixel_threshold` (default
70,000,000) routes any tile at/above that estimated pixel count to
`hpc.sbatch_large` (e.g. `16vcpu`) instead, so the largest tiles in the
domain don't OOM on a small node's RAM. Pixel count is estimated from each
tile's own bounding box (`area_deg2 * 3600**2`, DeltaDTM's ~1 arcsec native
resolution — see `hpc_dispatch.smk`), not by reading the real DEM, so it's
computable before any preprocessing has run; treat it as an approximation
and keep the threshold's implied memory estimate a comfortable margin below
`sbatch`'s partition RAM (see the comment in `config.yml`). Expect very few
`*_large_*` batches — most tiles stay on `sbatch`.

## 1. Preprocess + generate jobs

```
snakemake generate_aqueduct_jobs --cores N
```

Runs every preprocessing rule (DEM/mask/friction/boundaries, one set per
tile) in parallel across `--cores N`, then writes to
`{root}/model_outputs/hpc_jobs/`:

- `wave{H}_{small|large}_batch_{id}.sbatch` — one script per (wave `H`, size
  class, node batch `id`), each looping sequentially over its tiles' full
  return_period × SLR set (a tile's jobs always stay on one node; all of a
  batch's tiles share the same `hop_distance` AND the same size class — see
  "Tile-size routing" above).
- `resolved_config.yml` — the fully Linux-path-expanded config each sbatch
  job's Python call reads.
- `submit_waves.sh` — submits wave 0 immediately, then each later wave with
  a SLURM `afterany` dependency on the full previous wave.
- `logs/` — sbatch stdout/stderr and a per-batch failure log land here once jobs run.

Wave-0 tiles/scenarios with no boundary stations are resolved immediately
during this step (nodata placeholder written locally, same as
`run_aqueduct` does today) and never show up in any sbatch script.
Hop_distance≥1 tiles are never pre-excluded this way — whether a hinterland
tile has any upstream flooding to seed from can only be known once its
neighbour has actually run, so every hop≥1 tile always gets a job; a tile
whose neighbour ends up dry is resolved live, at run time, into a real-zero
result (see step 2).

### Bundling preprocessing + dispatch into Hydrax jobs

Instead of running step 1 locally, generate sbatch scripts that run
preprocessing directly on Hydrax — spread across `hpc.n_nodes` PARALLEL
nodes, same as simulation batching, since preprocessing has no
wave/hop_distance ordering constraint (every tile's DEM/mask/friction/
boundaries are independent of every other tile) — then, once every node's
batch has finished, one more job generates the wave sbatch scripts and
submits `submit_waves.sh` itself:

```
python snakemake_workflow/scripts/generate_hpc_preprocess_job.py
bash {jobs_dir}/submit_preprocess_and_dispatch.sh
```

This writes, per node batch, `{jobs_dir}/preprocess_batch_{id}.sbatch`
(runs `snakemake --cores N <that batch's explicit target file list>`,
using `hpc.preprocess_sbatch.*` from `config.yml`) plus a
`preprocess_batch_{id}_targets.txt` (the tile's dem/mask/friction +
every `(return_period, waterlevel_name)` boundaries file for that batch's
tiles — read at runtime via `$(cat ...)` rather than inlined, since a
batch's target list can run into the thousands of paths). It also writes
`{jobs_dir}/generate_jobs_and_dispatch.sbatch` (runs `snakemake
generate_aqueduct_jobs` — fast, since every preprocessing output already
exists by then — then `submit_waves.sh`), and
`{jobs_dir}/submit_preprocess_and_dispatch.sh`, which submits every
preprocessing batch with NO dependency between them (fully parallel), then
submits `generate_jobs_and_dispatch.sbatch` with
`--dependency=afterany:<every batch job id>` — the same afterany-join
pattern `submit_waves.sh` already uses between simulation waves, just one
more phase in front of wave 0.

Unlike the per-tile `run_job()` wrapper, every script here uses `set -euo
pipefail` with no per-step failure catch — each is either one Snakemake
invocation across many tiles or one monolithic dispatch step, and if it
fails there's nothing sensible to chain into.

Can be generated from either machine: from the local Windows box (writing
scripts meant for Hydrax, same `config_hpc.yml`-based dual path resolution
as `generate_aqueduct_jobs.py`), or run natively ON Hydrax itself, in which
case there's no separate "local" machine to reconcile paths with —
`config_hpc.yml` is simply unnecessary and `config_local.yml` alone already
points at Hydrax's own paths.

## 2. Submit to SLURM

```
bash {root}/model_outputs/hpc_jobs/submit_waves.sh
```

Each job calls `run_aqueduct_cli.py` once per `(tile, return_period,
waterlevel_name)`, sequentially within its batch. A failed call doesn't
abort the rest of that batch's jobs; it's appended to
`logs/wave{H}_batch_{id}_failures.txt` instead. For a `hop_distance == 0`
tile, `run_aqueduct_cli.py` reads that tile's real boundary stations; for
`hop_distance >= 1`, it reads its lower-hop neighbours' already-written
`waterdepth` rasters directly off disk (guaranteed to exist by the wave
barrier) and seeds from their wet cells — if none of them show any
flooding in the overlap, it writes a real all-zero result itself, exactly
like a local run would. Aqueduct's own OOM handling (mark tile, skip
remaining scenarios, write nodata placeholder) works exactly as it does
locally too — it's just discovered live on whichever node hits it.

Monitor with `squeue`/`sacct` as usual. To retry one failed job by hand:

```
python {code_root}/snakemake_workflow/scripts/run_aqueduct_cli.py \
    --config {root}/model_outputs/hpc_jobs/resolved_config.yml \
    --tile-id <id> --return-period <RP..> --waterlevel-name <SLR_..>
```

Every read on the HPC side that touches the shared P:\ mount (dem/mask/
friction/boundaries, and a hop≥1 job's neighbour dem/mask/waterdepth) is
retried on transient I/O errors (`config_utils.retry_transient_io`), and
neighbour-availability checks use `config_utils.path_ready` rather than bare
`os.path.exists` — the latter swallows `OSError` and just returns `False`,
so a momentary network blip would otherwise look identical to "neighbour
genuinely hasn't run yet" and silently produce a wrongly-confident real-zero
result instead of retrying.

## 2b. Compile the failure report

Once a wave (or the whole run) finishes:

```
python snakemake_workflow/scripts/collect_hpc_failures.py
```

Writes `{jobs_dir}/failure_report.csv`, combining two genuinely different
categories:

- `job_error` — from every batch's `logs/wave*_batch_*_failures.txt`: a
  crash/timeout/bad input, a genuine unresolved unknown. Needs
  investigating and re-running (see the retry command above).
- `no_boundary_stations` / `no_upstream_flooding` / `oom_too_large` — from
  `model_outputs/skipped_tiles/*.txt`: confidently-resolved results written
  during the run itself (a real zero, or a real "too large"), not
  failures. Listed for visibility only.

## 3. Postprocess

Once all waves finish, back on the machine with the shared mount:

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
- `submit_waves.sh` uses `afterany`, not `afterok` — a wave starts once its
  predecessor has finished regardless of individual job failures, since a
  failed tile there just means its downstream neighbours fall back to a
  real-zero result for that scenario, not that the whole next wave should
  be blocked.
