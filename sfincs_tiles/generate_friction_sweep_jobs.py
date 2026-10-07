"""Generates N independent sbatch scripts that dispatch the
friction_scale_factor sensitivity sweep (select_friction_sweep_tiles.py +
run_eikonal_on_sfincs_subgrid.py's --friction-scale-factor), one job per
node, each running up to --cpus-per-task work items CONCURRENTLY via
run_friction_sweep_batch.py's thread pool.

Work items = the full (tile_id, friction_scale_factor) cross product,
flattened and split evenly across --n-nodes batches - fully independent,
no wave/dependency ordering (every tile's own geometry/boundaries/SFINCS
ground truth already exists from the real validation_sfincs_v5 batch;
friction_scale_factor only changes the eikonal solve's own friction
raster, re-decoded from manning_subgrid.tif fresh each run - see
run_eikonal_on_sfincs_subgrid.py's build_inputs_from_sfincs_subgrid).

Default partition/mem mirror generate_validation_batch_jobs.py's own
4vcpu/30G choice for this same validation_sfincs_v5 directory - but unlike
that script (which runs one internally-multithreaded SFINCS solve at a
time per node), this one runs --cpus-per-task solves CONCURRENTLY per
node: each eikonal solve is single-threaded numba (src/eikonal.py has no
parallel=True/prange), so N-way concurrency on an N-cpu node is a genuine
use of the allocation, not oversubscription.

Usage:
    python select_friction_sweep_tiles.py --base-dir-name validation_sfincs_v5
    python generate_friction_sweep_jobs.py --base-dir-name validation_sfincs_v5 --n-nodes 2
    bash <printed submit path>
    # once every batch has finished:
    python compute_friction_sweep_metrics.py --base-dir-name validation_sfincs_v5 \\
        --tile-ids-file <root>/validation_sfincs_v5/friction_sweep_tile_ids.txt \\
        --friction-scale-factors 3 6 9 12 15 18 21 24 27 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

N_NODES_DEFAULT = 2
PARTITION_DEFAULT = "4vcpu"
TIME_DEFAULT = "16:00:00"
CPUS_PER_TASK_DEFAULT = 4
MEM_DEFAULT = "30G"
FRICTION_SCALE_FACTORS_DEFAULT = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]  # 0.1x-1.0x of
# 30.0, the full grid-resolution-corrected baseline (production's value AT THE TIME this sweep
# was designed and run - see config.yml's own comment on simulation.flooding.friction_scale_factor
# for why ×30 is a required unit correction, not itself a calibration choice). This frozen list is
# what the already-collected sweep data on disk (sfincs_tiles/.../eikonal_on_subgrid_waterdepth_
# ..._fsf<v>.tif) is keyed on, so it stays as-is even though production's own friction_scale_factor
# has since moved to 9.0 (= 30 x 0.3, this exact sweep's own best-CSI finding against SFINCS,
# 2026-10-03) - see select_friction_sweep_tiles.py's own docstring for why "scale the current values by 0.1-1.0"
# means these absolute values, not friction_scale_factor=0.1..1.0 literally.


def generate_batches(
    pairs: list[tuple[str, float]], n_nodes: int, partition: str, time_limit: str,
    mem: str, cpus_per_task: int, account: str, env_activate_cmd: str, batch_runner_linux: str,
    linux_jobs_dir: str, local_jobs_dir: Path, submit_path: Path, base_dir_name: str, config_linux: str,
    max_outer_iterations: int | None = None,
) -> None:
    n_batches = min(n_nodes, len(pairs))
    k, m = divmod(len(pairs), n_batches)
    batches = []
    for i in range(n_batches):
        batch_pairs = pairs[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if batch_pairs:
            batches.append((f"{i:03d}", batch_pairs))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    script_paths = []
    for batch_id, batch_pairs in batches:
        name = f"friction_sweep_batch_{batch_id}"

        pairs_filename = f"{name}_pairs.csv"
        pairs_path_local = local_jobs_dir / pairs_filename
        with open(pairs_path_local, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for tile_id, fsf in batch_pairs:
                writer.writerow([tile_id, fsf])
        pairs_path_linux = f"{linux_jobs_dir}/{pairs_filename}"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={partition}",
        ]
        if account:
            lines.append(f"#SBATCH --account={account}")
        lines += [
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",  # not -e: one pair's failure must not abort the rest of this node's batch
            env_activate_cmd,
            f'mkdir -p "{linux_jobs_dir}/logs"',
            "",
            f'python "{batch_runner_linux}" --pairs-file "{pairs_path_linux}" '
            f'--base-dir-name "{base_dir_name}" --config "{config_linux}" '
            f'--workers {cpus_per_task}'
            + (f' --max-outer-iterations {max_outer_iterations}' if max_outer_iterations is not None else ''),
            "",
        ]

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_pairs)} pair(s)) + {pairs_path_local}")

    # Fully independent batches - no --dependency chain.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for script in script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
        ]
    with open(submit_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    print(f"\nDone. {len(batches)} batch(es), {len(pairs)} (tile_id, friction_scale_factor) pair(s) total.")
    print(f"Wrote {submit_path}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument(
        "--tile-ids-file", default=None,
        help="default: {base_dir_name}/friction_sweep_tile_ids.txt - select_friction_sweep_tiles.py's output",
    )
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT)
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--max-outer-iterations", type=int, default=None,
        help="forwarded verbatim to run_friction_sweep_batch.py's own --max-outer-iterations for "
             "every pair in this dispatch (default: run_eikonal_on_sfincs_subgrid.py's own default, "
             "4, matching production).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]

    base_dir_local = local_root / args.base_dir_name
    base_dir_linux = f"{linux_root}/{args.base_dir_name}"

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else base_dir_local / "friction_sweep_tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    if not tile_ids:
        raise ValueError(f"no tile IDs found in {tile_ids_file} - run select_friction_sweep_tiles.py first")
    print(f"{len(tile_ids)} tile(s) from {tile_ids_file}")
    print(f"{len(args.friction_scale_factors)} friction_scale_factor value(s): {args.friction_scale_factors}")

    pairs = [(tid, fsf) for fsf in args.friction_scale_factors for tid in tile_ids]
    print(f"{len(pairs)} (tile_id, friction_scale_factor) pair(s) total")

    local_jobs_dir = base_dir_local / "friction_sweep_hpc_jobs"
    linux_jobs_dir = f"{base_dir_linux}/friction_sweep_hpc_jobs"
    batch_runner_linux = f"{linux_code_root}/sfincs_tiles/run_friction_sweep_batch.py"
    submit_path = local_jobs_dir / "submit_friction_sweep_batches.sh"

    # The sbatch scripts need a Linux-path config to pass through to run_eikonal_on_sfincs_subgrid.py's
    # own --config - write the same resolved Linux-view config generate_validation_batch_jobs.py uses.
    resolved_config_path = base_dir_local / "resolved_config.yml"
    retry_transient_io(base_dir_local.mkdir, parents=True, exist_ok=True)
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    config_linux = f"{base_dir_linux}/resolved_config.yml"
    print(f"Wrote {resolved_config_path} (Linux path view, for every batch's --config)")

    # Same env_activate_cmd as every other partition - just module load + conda activate, not partition-specific.
    env_activate_cmd = local_config["hpc"]["sbatch"]["env_activate_cmd"]

    generate_batches(
        pairs=pairs, n_nodes=args.n_nodes, partition=args.partition, time_limit=args.time,
        mem=args.mem, cpus_per_task=args.cpus_per_task, account=args.account,
        env_activate_cmd=env_activate_cmd, batch_runner_linux=batch_runner_linux,
        linux_jobs_dir=linux_jobs_dir, local_jobs_dir=local_jobs_dir,
        submit_path=submit_path, base_dir_name=args.base_dir_name, config_linux=config_linux,
        max_outer_iterations=args.max_outer_iterations,
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/submit_friction_sweep_batches.sh")


if __name__ == "__main__":
    main()
