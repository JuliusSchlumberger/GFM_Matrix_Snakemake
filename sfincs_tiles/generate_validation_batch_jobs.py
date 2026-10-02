"""Generates N independent sbatch scripts that dispatch a validation batch
(see select_validation_tiles.py, which writes tile_ids.txt) end to end, one
job per script, each looping sequentially through its own tile-id slice and
calling run_one_tile.sh <tile_id> per tile.

Submitted as N independent sbatch scripts (not a SLURM --array job), so
SLURM schedules each onto a node as one frees up.

Resolves both the local (Windows, for writing files here) and Linux (HPC,
for the generated scripts' own paths) views of paths.root from the same
config.yml.

Usage:
    python generate_validation_batch_jobs.py --base-dir-name validation_sfincs_v5
    bash <printed submit_..._batches.sh path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

N_NODES_DEFAULT = 20
PARTITION_DEFAULT = "4vcpu"  # matches this pipeline's SFINCS solve step (OMP_NUM_THREADS=4,
# see run_one_tile.sh)
TIME_DEFAULT = "16:00:00"  # headroom for a batch of tiles/job each up to a 4h SFINCS timeout,
# plus the build/eikonal steps
CPUS_PER_TASK_DEFAULT = 4
MEM_DEFAULT = "30G"
RUNNER_SCRIPT_NAME_DEFAULT = "run_one_tile.sh"


def _batch_name_prefix(base_dir_name: str) -> str:
    """e.g. 'validation_sfincs_v5' -> 'v5_batch', so batch/submit script
    filenames reflect whichever base_dir_name they were generated for."""
    prefix = "validation_sfincs_"
    tag = base_dir_name[len(prefix):] if base_dir_name.startswith(prefix) else base_dir_name
    return f"{tag}_batch"


def generate_batches(
    tile_ids: list[str], n_nodes: int, partition: str, time_limit: str,
    mem: str, cpus_per_task: int, account: str, runner_script_linux: str,
    linux_jobs_dir: str, local_jobs_dir: Path, submit_path: Path, batch_name_prefix: str,
    base_dir_name: str, runner_extra_args: str = "",
) -> None:
    n_batches = min(n_nodes, len(tile_ids))
    k, m = divmod(len(tile_ids), n_batches)
    batches = []
    for i in range(n_batches):
        batch_tile_ids = tile_ids[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if batch_tile_ids:
            batches.append((f"{i:03d}", batch_tile_ids))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    script_paths = []
    for batch_id, batch_tile_ids in batches:
        name = f"{batch_name_prefix}_{batch_id}"
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
            "set -uo pipefail",  # not -e: one tile's failure must not abort the rest of this node's batch
            f'mkdir -p "{linux_jobs_dir}/logs"',
            "",
        ]
        for tile_id in batch_tile_ids:
            extra = f" {runner_extra_args}" if runner_extra_args else ""
            # BASE_DIR_NAME=... prefix: run_one_tile.sh requires this env var explicitly.
            lines.append(f'BASE_DIR_NAME="{base_dir_name}" bash "{runner_script_linux}" {tile_id}{extra}')
        lines.append("")

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tile_ids)} tile(s))")

    # Fully independent batches - every tile is self-contained end to end, so
    # every batch submits immediately, no --dependency chain.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for script in script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
        ]
    with open(submit_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    print(f"\nDone. {len(batches)} batch(es), {len(tile_ids)} tile(s) total.")
    print(f"Wrote {submit_path}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v5 - created by select_validation_tiles.py")
    parser.add_argument(
        "--runner-script-name", default=RUNNER_SCRIPT_NAME_DEFAULT,
        help="per-tile runner script under sfincs_tiles/",
    )
    parser.add_argument(
        "--tile-ids-file", default=None,
        help="default: {base_dir_name}/tile_ids.txt - current selection output",
    )
    parser.add_argument("--skip-tile-ids", type=str, nargs="*", default=[], help="tile IDs to exclude entirely")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--runner-extra-args", default="",
        help="extra args appended verbatim to every generated 'bash <runner> tile_id' line, "
             "e.g. '--models bathtub,sfincs' or '--models eikonal --max-rounds 40'",
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

    skip = set(args.skip_tile_ids)
    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else base_dir_local / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    tile_ids = [t for t in tile_ids if t not in skip]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_file}")

    if not tile_ids:
        raise ValueError("no tile IDs to dispatch after excluding skip list")
    print(f"{len(tile_ids)} tile(s) total ({len(skip)} excluded: {sorted(skip)})")

    # -- resolved_config.yml (Linux path view), staged where run_one_tile.sh expects it --
    resolved_config_path = base_dir_local / "resolved_config.yml"
    retry_transient_io(base_dir_local.mkdir, parents=True, exist_ok=True)
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    print(f"Wrote {resolved_config_path} (Linux path view, for --config on every generated tile's Python steps)")

    local_jobs_dir = base_dir_local / "hpc_jobs"
    linux_jobs_dir = f"{base_dir_linux}/hpc_jobs"
    runner_script_linux = f"{linux_code_root}/sfincs_tiles/{args.runner_script_name}"

    batch_name_prefix = _batch_name_prefix(args.base_dir_name)
    submit_filename = f"submit_{batch_name_prefix}es.sh"  # e.g. 'v5_batch' -> 'submit_v5_batches.sh'

    generate_batches(
        tile_ids=tile_ids, n_nodes=args.n_nodes, partition=args.partition, time_limit=args.time,
        mem=args.mem, cpus_per_task=args.cpus_per_task, account=args.account,
        runner_script_linux=runner_script_linux, linux_jobs_dir=linux_jobs_dir, local_jobs_dir=local_jobs_dir,
        submit_path=local_jobs_dir / submit_filename, batch_name_prefix=batch_name_prefix,
        base_dir_name=args.base_dir_name, runner_extra_args=args.runner_extra_args,
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/{submit_filename}")
    print(f"(make sure {args.runner_script_name} is executable / callable via `bash` - no chmod needed since it's invoked as `bash <path>`)")


if __name__ == "__main__":
    main()
