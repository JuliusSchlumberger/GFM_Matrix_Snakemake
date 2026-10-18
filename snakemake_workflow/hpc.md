# Running the pipeline on an HPC

The flood solver is pure Python (`src/flood_model.py` + `src/eikonal.py`, a
Numba JIT) - no compiled executable and no per-machine single-instance
constraint, so any number of tiles can run concurrently across nodes.

The one real ordering constraint is wave-based hinterland forcing
(`hop_distance`, computed by `src/connectivity_tiling.compute_hop_distances`, see
`rules/simulation.smk`'s docstring): a `hop_distance >= 1` tile is seeded
from a strictly-lower-`hop_distance` neighbour's own output for the SAME
scenario (`src/boundaries.collect_neighbor_wave_seeds`). Wave `N+1` can
therefore only start once every wave-`N` job has reached a terminal state -
not "the neighbour happened to finish first". `generate_eikonal_jobs`
groups tiles by wave, generates one set of sbatch scripts per wave, and
writes a `submit_waves.sh` driver that submits each wave to SLURM with a
`--dependency=afterany:<every prior-wave job id>` barrier.

Four stages:

1. **Preprocessing + job generation** — Snakemake, on a many-core prep/login node.
2. **Simulation** — SLURM, submitted via `submit_waves.sh`, across compute nodes.
3. **Postprocessing** — Snakemake, back on the machine with the shared filesystem.
4. **Exposure analysis** — Snakemake/SLURM, on top of postprocessing's output.

Stage 1 can run either on your local preprocessing machine (as below) or as
one more Hydrax job that also submits stage 2 for you — see "Bundling
preprocessing + dispatch into one Hydrax job" below. Stages 3 and 4 can
likewise run manually (as documented below) or automatically once every
simulation wave finishes — `submit_waves.sh` ends with one more job,
gated `--dependency=afterany` on every wave's job ids, that generates and
submits the postprocessing+exposure dispatch itself (see "## 3. Postprocess"
below). So a single `bash submit_preprocess_and_dispatch.sh` (or a plain
`bash submit_waves.sh`, if job generation already happened) carries all
the way through to exposure analysis unattended, once
`preparation/run_preparation.py` has been run.

## One-time setup

1. `snakemake_workflow/config/config_hpc.yml` (tracked) holds the Hydrax
   values for `paths.root` (Linux mount of the same storage as the local
   `paths.root`) and `paths.code_root` (Linux checkout of this repo — used to
   locate `run_eikonal_cli.py`). For another cluster, start from the template:
   ```
   cp snakemake_workflow/config/config_hpc.yml.example snakemake_workflow/config/config_hpc.yml
   ```
   Not needed at all if you're running stage 1 directly ON Hydrax (see below) — in
   that case `config_local.yml` alone already points at Hydrax's own paths.

2. Check the values under `hpc:` in `config.yml` (set for Hydrax):
   `n_nodes` (max sbatch scripts to generate PER WAVE — a wave with fewer
   tiles than `n_nodes` gets one script per tile instead, never an empty
   one), and `sbatch.partition` / `account` / `time` / `mem` /
   `cpus_per_task` / `env_activate_cmd`. `sbatch` is shared by BOTH
   simulation batching AND preprocessing batching — there is no separate
   preprocessing-only sbatch config.

   Optionally, `hpc.wave_tiers` overrides `n_nodes`/`sbatch` per
   hop_distance wave (see config.yml's own comment on that key) — e.g. a
   big, heavily-parallel `24vcpu` allocation for hop_distance=0 (by far the
   most tiles) and a smaller one for hop=1/2, falling back to a plain
   uncapped `1vcpu`-per-tile allocation for every later wave. Read only by
   the standalone `generate_hpc_simulation_jobs.py` CLI, not by
   `hpc_dispatch.smk`'s own Snakemake-rule entry point.

`cpus_per_task` is the real degree of concurrency for a batch, not just a
SLURM allocation size: for preprocessing it's passed straight to
`snakemake --cores N`; for simulation, `run_simulation_batch.py`
runs that many tiles' solves concurrently via a thread pool (each eikonal
solve is single-threaded numba, so N concurrent solves on an N-cpu node is a
genuine use of the allocation, not oversubscription). A `cpus_per_task: 1`
partition means fully serial, one tile/job at a time; bump it up (directly,
or via a `wave_tiers` entry for simulation) if throughput matters more than
a minimal per-node footprint.

There is no size-class routing: `tile_generation.hard_ceiling_cells` caps
every tile's size, so all tiles of a wave share one partition.

## 1. Preprocess + generate jobs

```
snakemake generate_eikonal_jobs --cores N
```

Runs every preprocessing rule (DEM/mask/friction/boundaries, one set per
tile) in parallel across `--cores N`, then writes to
`{root}/model_outputs/hpc_jobs/`:

- `wave{H}_batch_{id}.sbatch` — one script per (wave `H`, node batch `id`),
  running its tiles' full return_period × SLR set CONCURRENTLY up to that
  batch's `cpus_per_task` via `run_simulation_batch.py` (a tile's jobs
  always stay on one node; all of a batch's tiles share the same
  `hop_distance`). A companion `wave{H}_batch_{id}_triples.csv` lists the
  (tile_id, return_period, waterlevel_name) rows that runner reads.
- `resolved_config.yml` — the fully Linux-path-expanded config each sbatch
  job's Python call reads.
- `submit_waves.sh` — submits wave 0 immediately, then each later wave with
  a SLURM `afterany` dependency on the full previous wave.
- `logs/` — sbatch stdout/stderr and a per-batch failure log land here once jobs run.

Every `(tile, return_period, waterlevel_name)` gets a job, with no
pre-exclusion: whether a hinterland tile has any upstream flooding to seed
from can only be known once its neighbour has actually run. A wave-0 tile
without boundary stations, or a hop≥1 tile whose neighbours end up dry, is
resolved live, at run time, into a real-zero result (see step 2).

### Bundling preprocessing + dispatch into Hydrax jobs

Instead of running step 1 locally, generate sbatch scripts that run
preprocessing directly on Hydrax — every tile's target files split evenly
across up to `hpc.n_nodes` PARALLEL nodes (no size-class split), since
preprocessing has no wave/hop_distance ordering constraint (every tile's
DEM/mask/friction/boundaries are independent of every other tile) — then,
once every node's batch has finished, one more job generates the wave
sbatch scripts and submits `submit_waves.sh` itself:

```
python snakemake_workflow/scripts/generate_hpc_preprocess_job.py
bash {jobs_dir}/submit_preprocess_and_dispatch.sh
```

This writes, per node batch, `{jobs_dir}/preprocess_batch_{id}.sbatch` (runs
`snakemake --cores N <that batch's explicit target file list>`, using
`hpc.sbatch` from `config.yml` — same partition simulation batching uses, no
separate preprocessing-only config) plus a
`preprocess_batch_{id}_targets.txt` (the tile's dem/mask/friction + every
`(return_period, waterlevel_name)` boundaries file for that batch's tiles —
read at runtime via `$(cat ...)` rather than inlined, since a batch's target
list can run into the thousands of paths). It also writes
`{jobs_dir}/generate_jobs_and_dispatch.sbatch` (uses `hpc.sbatch` —
lightweight, just runs `snakemake generate_eikonal_jobs` — fast, since
every preprocessing output already exists by then — then `submit_waves.sh`),
and `{jobs_dir}/submit_preprocess_and_dispatch.sh`, which first builds the
shared, tile-independent inputs (geoid-offset raster, cached station
GeoPackages) synchronously on the submitting node, then submits every
preprocessing batch with NO dependency between them (fully parallel), then
submits `generate_jobs_and_dispatch.sbatch` with
`--dependency=afterany:<every batch job id>` — the same afterany-join
pattern `submit_waves.sh` already uses between simulation waves, just one
more phase in front of wave 0.

Unlike `run_simulation_batch.py`'s own per-item failure handling, every
script here uses `set -euo pipefail` with no per-tile failure catch — each
is either one Snakemake invocation across many tiles or one monolithic
dispatch step, and if it
fails there's nothing sensible to chain into. Each Snakemake call is only
retried as a whole (`run_snakemake_with_retry`, up to 10 attempts) against
transient P:\ read errors.

Can be generated from either machine: from the local Windows box (writing
scripts meant for Hydrax, same `config_hpc.yml`-based dual path resolution
as `generate_eikonal_jobs.py`), or run natively ON Hydrax itself, in which
case there's no separate "local" machine to reconcile paths with —
`config_hpc.yml` is simply unnecessary and `config_local.yml` alone already
points at Hydrax's own paths.

## 2. Submit to SLURM

```
bash {root}/model_outputs/hpc_jobs/submit_waves.sh
```

Each job calls `run_eikonal_cli.py` once per `(tile, return_period,
waterlevel_name)`, up to `cpus_per_task` at once (`run_simulation_batch.py`). A failed call doesn't
abort the rest of that batch's jobs; it's appended to
`logs/wave{H}_batch_{id}_failures.txt` instead. For a `hop_distance == 0`
tile, `run_eikonal_cli.py` reads that tile's real boundary stations; for
`hop_distance >= 1`, it reads its lower-hop neighbours' already-written
`waterdepth` rasters directly off disk (guaranteed to exist by the wave
barrier) and seeds from their wet cells — if none of them show any
flooding in the overlap, it writes a real all-zero result itself, exactly
like a local run would. The eikonal runner's own OOM handling (mark tile, skip
remaining scenarios, write nodata placeholder) works exactly as it does
locally too — it's just discovered live on whichever node hits it.

Monitor with `squeue`/`sacct` as usual. To retry one failed job by hand:

```
python {code_root}/snakemake_workflow/scripts/run_eikonal_cli.py \
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

**Automatic (default once you've submitted via `submit_waves.sh`):** the
last phase `submit_waves.sh` submits — `gfm_generate_postprocess_and_dispatch`
— is gated `--dependency=afterany:<every wave's job ids>` (via an
`ALL_WAVE_IDS` accumulator kept alongside the wave-to-wave
`PREV_IDS`, since `PREV_IDS` alone only ever holds the *previous* wave's
ids). Once every wave has reached a terminal state, it runs
`generate_hpc_postprocess_job.py` (fresh, no `--resume`) and submits the
resulting `submit_postprocess_and_exposure.sh` itself — see that script's
own docstring (`snakemake_workflow/scripts/generate_hpc_postprocess_job.py`)
for the merge_chunk → prepare_exposure_grid_chunk →
compute_flood_fraction_chunk → exposure-analysis phase chain it generates,
plus a final flood-totals + plots job (`compute_flood_totals.py` +
`run_analysis.py --only-plots`; off with `--no-flood-totals-and-plots`).
No separate action needed; check `squeue`/`sacct` for
`gfm_generate_postprocess_and_dispatch` and the phases after it the same way
you'd watch any other stage.

**Manual (still available — a small/debug run, or after a `--resume`
situation you want to control by hand):** once all waves finish, back on the
machine with the shared mount:

```
snakemake postprocess --cores N
```

Results land at the same `model_outputs/{tile_id}/results/
waterdepth_{rp}_{slr}.tif` paths `run_eikonal` would have produced, so
Snakemake sees them as already up to date and proceeds straight to merging/
plotting — no different from a fully-local run.

## Notes

- The `run_eikonal` rule remains the right choice for a small
  local/debug run, no SLURM involved.
- Re-running `generate_eikonal_jobs` after changing `hpc.n_nodes` won't
  regenerate scripts on its own (Snakemake sees the old outputs as already
  present) — delete `model_outputs/hpc_jobs/` first, or use `--forcerun
  generate_eikonal_jobs`.
- `submit_waves.sh` uses `afterany`, not `afterok` — a wave starts once its
  predecessor has finished regardless of individual job failures, since a
  failed tile there just means its downstream neighbours fall back to a
  real-zero result for that scenario, not that the whole next wave should
  be blocked. The postprocessing bridge job at the end uses the same
  `afterany` philosophy for the same reason.
