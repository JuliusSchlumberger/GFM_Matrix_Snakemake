"""Generate SLURM batch scripts to run run_eikonal_on_sfincs_subgrid.py
(bathtub + eikonal-on-SFINCS-subgrid, see that script's own module
docstring) for the 258-tile SFINCS validation batch, across n_nodes 1vcpu
nodes - the same pattern as generate_aqueduct_validation_sfincs_jobs.py,
targeting this different script.

Usage:
    python generate_eikonal_on_subgrid_jobs.py
    bash <printed submit_eikonal_on_subgrid.sh path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

N_NODES_DEFAULT = 20
PARTITION_DEFAULT = "1vcpu"
TIME_DEFAULT = "08:00:00"
MEM_DEFAULT = "7500M"


def generate_batches(
    tile_ids: list[str], n_nodes: int, partition: str, time_limit: str, mem: str, account: str,
    env_activate_cmd: str, runner_script_linux: str, linux_resolved_config: str,
    linux_jobs_dir: str, local_jobs_dir: Path, submit_path: Path,
) -> None:
    n_batches = min(n_nodes, len(tile_ids))
    k, m = divmod(len(tile_ids), n_batches)
    batches = []
    for i in range(n_batches):
        batch_tiles = tile_ids[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if batch_tiles:
            batches.append((f"{i:03d}", batch_tiles))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    script_paths = []
    for batch_id, batch_tiles in batches:
        name = f"eikonal_subgrid_batch_{batch_id}"
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
            "#SBATCH --cpus-per-task=1",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",
            env_activate_cmd,
            "",
        ]
        for tile_id in batch_tiles:
            lines.append(f'python "{runner_script_linux}" --config "{linux_resolved_config}" --tile-id {tile_id}')
        lines.append("")

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tiles)} tile(s))")

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
    parser.add_argument("--tile-ids-file", default=None, help="text file, one tile ID per line (default: validation_sfincs/test_tile_selection_ids.txt)")
    parser.add_argument("--skip-tile-ids", type=str, nargs="*", default=[], help="tile IDs to exclude entirely")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--account", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]
    env_activate_cmd = linux_config["hpc"]["sbatch"]["env_activate_cmd"]

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else local_root / "validation_sfincs" / "test_tile_selection_ids.txt"
    all_tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    skip = set(args.skip_tile_ids)
    tile_ids = [t for t in all_tile_ids if t not in skip]
    if not tile_ids:
        raise ValueError(f"no tile IDs left in {tile_ids_file} after excluding {skip}")
    print(f"{len(tile_ids)} tile(s) from {tile_ids_file} ({len(skip)} excluded: {sorted(skip)})")

    local_jobs_dir = local_root / "validation_sfincs" / "hpc_jobs_eikonal_subgrid"
    linux_jobs_dir = f"{linux_root}/validation_sfincs/hpc_jobs_eikonal_subgrid"
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")

    runner_script_linux = f"{linux_code_root}/sfincs_tiles/run_eikonal_on_sfincs_subgrid.py"

    generate_batches(
        tile_ids=tile_ids, n_nodes=args.n_nodes, partition=args.partition, time_limit=args.time,
        mem=args.mem, account=args.account, env_activate_cmd=env_activate_cmd,
        runner_script_linux=runner_script_linux, linux_resolved_config=f"{linux_jobs_dir}/resolved_config.yml",
        linux_jobs_dir=linux_jobs_dir, local_jobs_dir=local_jobs_dir,
        submit_path=local_jobs_dir / "submit_eikonal_on_subgrid.sh",
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/submit_eikonal_on_subgrid.sh")


if __name__ == "__main__":
    main()
