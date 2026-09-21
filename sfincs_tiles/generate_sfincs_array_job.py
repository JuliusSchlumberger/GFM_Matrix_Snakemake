"""Generate ONE SLURM job-array sbatch script to run already-built (or
still-being-built) SFINCS tile models on Hydrax, instead of
generate_sfincs_hpc_jobs.py's fixed N-way batch split.

Why a job array here: the 258-tile local build (run_sfincs_tiles.py
--build-only, under hydromt-sfincs-dev on Windows) runs sequentially and
takes a while - by the time it's a third of the way through, waiting for
ALL 258 tiles' sfincs.inp to exist before generating/submitting ANY HPC
work (generate_sfincs_hpc_jobs.py's own assumption) wastes however long
the rest of the build takes. A job array sidesteps this entirely: submit
ALL 258 tasks right now, each one independently polls for its OWN tile's
sfincs.inp to appear (already there -> starts almost immediately; still
being built -> waits, cheaply, in a sleep loop) before staging/running/
copying back - so tiles that are already built start now, and tiles still
being generated get picked up the moment they're ready, all from one
submission. Concurrency is capped by SLURM's own array throttle
(--array=0-N%<max_concurrent>), not our own batch-count math - the same
"one core"->"4 cores" node shape as generate_sfincs_hpc_jobs.py
(CPUS_PER_TASK_DEFAULT cores, 4vcpu partition), just N independent
single-tile tasks instead of a handful of multi-tile sequential batches.

Known-permanently-failing tiles (antimeridian - see build_sfincs_tile.py's
own TopologyException handling) are excluded from the array up front via
--skip-tile-ids, so they don't occupy a wait-slot for the full timeout.

Run this under the MAIN gfm_python_preprocessing env (same reasoning as
generate_sfincs_hpc_jobs.py's own docstring - needs config_utils.py, not
hydromt_sfincs).

Usage:
    python generate_sfincs_array_job.py --skip-tile-ids 2029 2077
    sbatch <printed sbatch path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402

N_CONCURRENT_DEFAULT = 20
PARTITION_DEFAULT = "4vcpu"
TIME_DEFAULT = "08:00:00"
CPUS_PER_TASK_DEFAULT = 4
MEM_DEFAULT = "30G"
MAX_WAIT_ATTEMPTS_DEFAULT = 240  # * WAIT_INTERVAL_S below = max time a task waits for its own tile to be built
WAIT_INTERVAL_S = 30
SIF_PATH_DEFAULT = (
    "/p/11202255-sfincs/executables/SFINCS_2026/SFINCS_2026_01/"
    "v2.4.0_Galibier_Release_CPU_apptainer/sfincs-cpu_v2.4.0-Galibier-Release.sif"
)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids-file", default=None, help="text file, one tile ID per line (default: validation_sfincs/test_tile_selection_ids.txt)")
    parser.add_argument("--skip-tile-ids", type=str, nargs="*", default=[], help="tile IDs to exclude entirely (e.g. known antimeridian failures)")
    parser.add_argument("--n-concurrent", type=int, default=N_CONCURRENT_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--max-wait-attempts", type=int, default=MAX_WAIT_ATTEMPTS_DEFAULT, help=f"* {WAIT_INTERVAL_S}s poll interval = max time a task waits for its own tile's sfincs.inp before giving up")
    parser.add_argument("--account", default="")
    parser.add_argument("--sif-path", default=SIF_PATH_DEFAULT)
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else local_root / "validation_sfincs" / "test_tile_selection_ids.txt"
    all_tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    skip = set(args.skip_tile_ids)
    tile_ids = [t for t in all_tile_ids if t not in skip]
    if not tile_ids:
        raise ValueError(f"no tile IDs left in {tile_ids_file} after excluding {skip}")
    print(f"{len(tile_ids)} tile(s) from {tile_ids_file} ({len(skip)} excluded: {sorted(skip)})")

    local_jobs_dir = local_root / "validation_sfincs" / "hpc_jobs"
    linux_jobs_dir = f"{linux_root}/validation_sfincs/hpc_jobs"
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Task N (0-indexed) reads line N+1 of this file at RUN time, not generation
    # time - so tiles appearing later on disk are picked up correctly too, this
    # file just needs to exist and be stable for the whole array's lifetime.
    array_tile_ids_path = local_jobs_dir / "array_tile_ids.txt"
    with open(array_tile_ids_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tile_ids) + "\n")
    linux_tile_ids_file = f"{linux_jobs_dir}/array_tile_ids.txt"

    name = "sfincs_array"
    n_tasks = len(tile_ids)
    array_spec = f"0-{n_tasks - 1}%{args.n_concurrent}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={args.partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines += [
        f"#SBATCH --time={args.time}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%A_%a.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%A_%a.err",
        "",
        "set -uo pipefail",  # not -e: this task's own failure must not be treated as a script bug
        f'SFINCS_ROOT="{linux_root}"',
        f'SIF_PATH="{args.sif_path}"',
        f'TILE_IDS_FILE="{linux_tile_ids_file}"',
        f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
        "",
        # sed line numbers are 1-indexed, SLURM_ARRAY_TASK_ID is 0-indexed
        'TILE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TILE_IDS_FILE" | tr -d "\\r")',
        'if [ -z "$TILE_ID" ]; then',
        '  echo "task $SLURM_ARRAY_TASK_ID: no tile ID at that line in $TILE_IDS_FILE" >> "$FAIL_LOG"',
        "  exit 0",
        "fi",
        "",
        'REMOTE_DIR="$SFINCS_ROOT/validation_sfincs/${TILE_ID}/sfincs_model"',
        'LOCAL_DIR="${TMPDIR:-/tmp}/sfincs_${TILE_ID}_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"',
        "",
        "# Wait for THIS task's own tile to be built (still-running Windows-side",
        "# --build-only) - already built -> proceeds immediately.",
        "wait_attempt=0",
        f'while [ ! -f "$REMOTE_DIR/sfincs.inp" ] && [ "$wait_attempt" -lt {args.max_wait_attempts} ]; do',
        f'  sleep {WAIT_INTERVAL_S}',
        "  wait_attempt=$((wait_attempt + 1))",
        "done",
        'if [ ! -f "$REMOTE_DIR/sfincs.inp" ]; then',
        f'  echo "$TILE_ID  gave up waiting for sfincs.inp after {args.max_wait_attempts * WAIT_INTERVAL_S}s (build never finished/failed for this tile)" >> "$FAIL_LOG"',
        "  exit 0",
        "fi",
        "",
        'rm -rf "$LOCAL_DIR"; mkdir -p "$LOCAL_DIR"',
        "attempt=1",
        'while [ "$attempt" -le 5 ]; do',
        '  cp -r "$REMOTE_DIR"/. "$LOCAL_DIR"/ 2>/dev/null',
        '  [ -f "$LOCAL_DIR/sfincs.inp" ] && break',
        '  echo "  [stage retry $attempt/4] tile $TILE_ID input copy failed - retrying in 5s..." >&2',
        "  sleep 5",
        "  attempt=$((attempt + 1))",
        "done",
        'if [ ! -f "$LOCAL_DIR/sfincs.inp" ]; then',
        '  echo "$TILE_ID  failed to stage input to node-local scratch after 5 attempts" >> "$FAIL_LOG"',
        '  rm -rf "$LOCAL_DIR"',
        "  exit 0",
        "fi",
        "",
        f"export OMP_NUM_THREADS={args.cpus_per_task}",
        '( cd "$LOCAL_DIR" && apptainer exec -B "$LOCAL_DIR":/mnt/data "$SIF_PATH" sfincs ) '
        '> "$LOCAL_DIR/sfincs_hpc_run.log" 2>&1',
        "run_rc=$?",
        "",
        'if [ "$run_rc" -ne 0 ] || [ ! -f "$LOCAL_DIR/sfincs_map.nc" ]; then',
        '  echo "$TILE_ID  sfincs run failed (exit $run_rc) or produced no sfincs_map.nc" >> "$FAIL_LOG"',
        '  cp -f "$LOCAL_DIR/sfincs_hpc_run.log" "$REMOTE_DIR/" 2>/dev/null',
        '  rm -rf "$LOCAL_DIR"',
        "  exit 0",
        "fi",
        "",
        'cp -f "$LOCAL_DIR/sfincs_map.nc" "$REMOTE_DIR/"',
        'cp -f "$LOCAL_DIR/sfincs.log" "$REMOTE_DIR/" 2>/dev/null',
        'cp -f "$LOCAL_DIR/sfincs_hpc_run.log" "$REMOTE_DIR/"',
        'rm -rf "$LOCAL_DIR"',
        'echo "tile $TILE_ID done"',
        "",
    ]

    script_path = local_jobs_dir / f"{name}.sbatch"
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {array_tile_ids_path} ({n_tasks} tile ID(s))")
    print(f"Wrote {script_path} (array={array_spec}, {args.n_concurrent} concurrent, "
          f"each task waits up to {args.max_wait_attempts * WAIT_INTERVAL_S}s for its own tile to be built)")
    print(f"\nSubmit on Hydrax with: sbatch {linux_jobs_dir}/{name}.sbatch")


if __name__ == "__main__":
    main()
