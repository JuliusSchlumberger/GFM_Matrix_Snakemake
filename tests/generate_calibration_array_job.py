"""Generate ONE SLURM job-array sbatch script to run the 260-tile
sweep-budget or obstacle-coupling calibration study on Hydrax, one array
task per tile (2026-09-24 - "run the whole thing on the HPC in parallel for
the different tiles").

Unlike generate_sfincs_array_job.py, no local-scratch staging/apptainer is
needed here - both calibration scripts are pure Python/Numba
(src/eikonal.py, no compiled executable - see config_hpc.yml's own comment
on this), reading a handful of small files (dem.tif/mask.tif/friction.tif/
boundaries_*.gpkg, a few MB at most) and writing one small per-tile CSV
back - cheap enough to run directly against the shared P:/HPC mount with no
staging step at all.

1 vcpu / 7G per task (2026-09-24, user direction - Hydrax's 1vcpu partition
caps memory at 7.95GB per core, 7G leaves headroom): the eikonal kernel
(`_dense_sweep`/`_update`) is plain `@njit(cache=True)`, no `parallel=True`,
so extra cores would sit idle - matches this session's own worst-case
memory estimate (~5-7GB peak on the single largest ~207M-cell candidate
tile, several float32/int8/bool arrays of that size alive at once).

Note on Numba's on-disk JIT cache (`cache=True` in eikonal.py): many array
tasks starting near-simultaneously could all attempt to compile+write that
cache the first time this code runs on a shared filesystem - numba's cache
writes are designed to tolerate concurrent writers (atomic rename), so this
is expected to be safe, just flagged here as a known/watched risk rather
than something worked around.

Usage:
    python generate_calibration_array_job.py sweep_budget
    python generate_calibration_array_job.py obstacle_coupling --tile-ids-file <wet_tiles_selected.txt>
    sbatch <printed sbatch path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

N_CONCURRENT_DEFAULT = 50  # lightweight 1vcpu/7G tasks - can run far more concurrently than
# the SFINCS array job's own N_CONCURRENT_DEFAULT=20 (full SFINCS solves, 4vcpu/30G each)
PARTITION_DEFAULT = "1vcpu"
MEM_DEFAULT = "7G"
CPUS_PER_TASK_DEFAULT = 1
GFM_PY_LINUX = "/u/schlumbe/.conda/envs/gfm/bin/python"

SCRIPTS = {
    "sweep_budget": {
        "script_name": "test_sweep_budget_calibration.py",
        "out_subdir": "sweep_budget",
        "time_default": "04:00:00",  # worst case (no round-level early exit ever, 100 rounds
        # = 400 sweeps) on the largest ~207.6M-cell candidate: ~400 * 207.6M * ~0.05s/(M*sweep)
        # ~= 69 min - 4h leaves a comfortable margin (real runs converge far earlier on almost
        # every tile thanks to the round-level early exit - see that script's own docstring).
        "default_tile_ids_file": "candidate_tiles.txt",
    },
    "obstacle_coupling": {
        "script_name": "test_obstacle_coupling_calibration.py",
        "out_subdir": "obstacle_coupling",
        "time_default": "16:00:00",  # worst case (zero early exit across ALL (max_outer+1)=16
        # solves x inner_max_rounds=50 rounds = 3200 sweeps) on the largest candidate:
        # ~3200 * 207.6M * ~0.05s/(M*sweep) ~= 9.2h - see the conversation this was sized in for
        # why the 15/50 defaults (vs production's own 5/12) pushed this well past the sweep-
        # budget study's own margin; real runs should finish far sooner via the softened 0.1m
        # epsilon default's more frequent early exits, but this leaves genuine worst-case room.
        "default_tile_ids_file": "sweep_budget/wet_tiles_selected.txt",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=list(SCRIPTS), help="which calibration study to generate an array job for")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids-file", default=None, help="default: calibration_260_tiles/<study's own default>, see SCRIPTS above")
    parser.add_argument("--n-concurrent", type=int, default=N_CONCURRENT_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=None, help="default: per-study worst-case estimate, see SCRIPTS above")
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--extra-args", default="",
        help="extra CLI args forwarded verbatim to the per-tile script call, e.g. "
             "'--epsilon 0.03' or '--max-outer 5 --inner-max-rounds 12' to reproduce "
             "production's own defaults instead of this study's own wider ones",
    )
    args = parser.parse_args()

    spec = SCRIPTS[args.study]
    time_limit = args.time or spec["time_default"]

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]

    study_root_local = local_root / "calibration_260_tiles"
    study_root_linux = f"{linux_root}/calibration_260_tiles"

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else study_root_local / spec["default_tile_ids_file"]
    all_tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    if not all_tile_ids:
        raise ValueError(f"no tile IDs in {tile_ids_file}")
    print(f"{len(all_tile_ids)} tile(s) from {tile_ids_file}")

    local_jobs_dir = study_root_local / "hpc_jobs"
    linux_jobs_dir = f"{study_root_linux}/hpc_jobs"
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    array_tile_ids_path = local_jobs_dir / f"{args.study}_array_tile_ids.txt"
    with open(array_tile_ids_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(all_tile_ids) + "\n")
    linux_tile_ids_file = f"{linux_jobs_dir}/{args.study}_array_tile_ids.txt"

    out_dir_linux = f"{study_root_linux}/{spec['out_subdir']}"
    retry_transient_io((local_root / "calibration_260_tiles" / spec["out_subdir"]).mkdir, parents=True, exist_ok=True)

    # config_hpc.yml is a git-ignored, Windows-machine-only file (see its own
    # header) - it does NOT exist on the HPC side, so each array task can't
    # just point --config at it there. Same resolved_config.yml staging
    # pattern generate_v2_batch_jobs.py already established for the SFINCS
    # validation batches: write the ALREADY-Linux-resolved config once, here,
    # to shared storage, and have every task load that directly instead.
    resolved_config_path = study_root_local / "resolved_config.yml"
    retry_transient_io(study_root_local.mkdir, parents=True, exist_ok=True)
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    print(f"Wrote {resolved_config_path} (Linux path view, for --config on every array task)")
    linux_config_path = f"{study_root_linux}/resolved_config.yml"

    name = f"calib_{args.study}"
    n_tasks = len(all_tile_ids)
    array_spec = f"0-{n_tasks - 1}%{args.n_concurrent}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={args.partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines += [
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%A_%a.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%A_%a.err",
        "",
        "set -uo pipefail",  # not -e: this task's own failure must not be treated as a script bug
        f'GFM_PY="{GFM_PY_LINUX}"',
        f'CODE_ROOT="{linux_code_root}"',
        f'CONFIG="{linux_config_path}"',
        f'TILE_IDS_FILE="{linux_tile_ids_file}"',
        f'OUT_DIR="{out_dir_linux}"',
        f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
        "",
        'TILE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TILE_IDS_FILE" | tr -d "\\r")',
        'if [ -z "$TILE_ID" ]; then',
        '  echo "task $SLURM_ARRAY_TASK_ID: no tile ID at that line in $TILE_IDS_FILE" >> "$FAIL_LOG"',
        "  exit 0",
        "fi",
        "",
        f'cd "$CODE_ROOT/tests"',
        f'echo "=== tile $TILE_ID ({args.study}) ==="',
        f'"$GFM_PY" {spec["script_name"]} "$OUT_DIR" --config "$CONFIG" --tile-ids "$TILE_ID" {args.extra_args} '
        '|| echo "$TILE_ID  script exited non-zero" >> "$FAIL_LOG"',
        'echo "tile $TILE_ID done"',
        "",
    ]

    script_path = local_jobs_dir / f"{name}.sbatch"
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"Wrote {array_tile_ids_path} ({n_tasks} tile ID(s))")
    print(f"Wrote {script_path} (array={array_spec}, {args.n_concurrent} concurrent, "
          f"time={time_limit}, mem={args.mem}, partition={args.partition})")
    print(f"\nSubmit on Hydrax with: sbatch {linux_jobs_dir}/{name}.sbatch")


if __name__ == "__main__":
    main()
