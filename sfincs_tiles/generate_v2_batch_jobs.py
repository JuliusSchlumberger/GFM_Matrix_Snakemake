"""Generate the N independent sbatch scripts that dispatch the v2 validation
batch (see select_validation_tiles.py - current selections write a single
tile_ids.txt, no more Set A/B, 2026-09-24 user direction; --set-a-file/
--set-b-file remain as an explicit opt-in for regenerating older v2/v3
batches that do still carry a set label) end to end, one job per script, each
looping sequentially through its own tile-id slice and calling
run_one_tile_v2.sh <tile_id> [<set>] per tile - the per-tile pipeline
covering input-copy -> SFINCS-input build (hydromt-sfincs-dev env) ->
bathtub+eikonal (gfm env) -> SFINCS run (apptainer) -> postprocess+summary
(hydromt-sfincs-dev env), all inside that one script.

Deliberately NOT a SLURM --array job (unlike generate_sfincs_array_job.py) -
per user direction, N independent sbatch scripts submitted individually, so
SLURM schedules each onto a node as one frees up rather than managing array
task indices.

Reuses this repo's established dual-path-view pattern
(config_utils.load_config(..., extra_override=config_hpc.yml), see
generate_sfincs_hpc_jobs.py/generate_eikonal_on_subgrid_jobs.py) to resolve
both the local (Windows, for writing files here) and Linux (HPC, for the
generated scripts' own paths) views of paths.root from the same config.yml.

Usage:
    python generate_v2_batch_jobs.py
    bash <printed submit_v2_batches.sh path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

N_NODES_DEFAULT = 20
PARTITION_DEFAULT = "4vcpu"  # matches generate_sfincs_hpc_jobs.py's own switch from 1vcpu -
# this pipeline runs the same SFINCS solve step (OMP_NUM_THREADS=4, see run_one_tile_v2.sh)
TIME_DEFAULT = "16:00:00"  # raised from 12h 2026-09-23 (user direction) - more headroom for a
# batch of 26 tiles/job each up to a 4h SFINCS timeout, plus the build/eikonal steps
CPUS_PER_TASK_DEFAULT = 4
MEM_DEFAULT = "30G"
BASE_DIR_NAME_DEFAULT = "validation_sfincs_v2"
RUNNER_SCRIPT_NAME_DEFAULT = "run_one_tile_v2.sh"


def _batch_name_prefix(base_dir_name: str) -> str:
    """e.g. 'validation_sfincs_v3' -> 'v3_batch', so batch/submit script
    filenames reflect whichever base_dir_name they were generated for
    instead of always saying 'v2' (real confusion, 2026-09-23: a v3 run's
    files still being named v2_batch_*.sbatch/submit_v2_batches.sh made a
    user try `bash .../submit_v3_batches.sh`, which never existed)."""
    prefix = "validation_sfincs_"
    tag = base_dir_name[len(prefix):] if base_dir_name.startswith(prefix) else base_dir_name
    return f"{tag}_batch"


def generate_batches(
    tile_set_pairs: list[tuple[str, str | None]], n_nodes: int, partition: str, time_limit: str,
    mem: str, cpus_per_task: int, account: str, runner_script_linux: str,
    linux_jobs_dir: str, local_jobs_dir: Path, submit_path: Path, batch_name_prefix: str,
    runner_extra_args: str = "",
) -> None:
    n_batches = min(n_nodes, len(tile_set_pairs))
    k, m = divmod(len(tile_set_pairs), n_batches)
    batches = []
    for i in range(n_batches):
        batch_pairs = tile_set_pairs[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if batch_pairs:
            batches.append((f"{i:03d}", batch_pairs))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    script_paths = []
    for batch_id, batch_pairs in batches:
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
        for tile_id, tile_set in batch_pairs:
            set_arg = f" {tile_set}" if tile_set else ""
            extra = f" {runner_extra_args}" if runner_extra_args else ""
            lines.append(f'bash "{runner_script_linux}" {tile_id}{set_arg}{extra}')
        lines.append("")

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_pairs)} tile(s))")

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

    print(f"\nDone. {len(batches)} batch(es), {len(tile_set_pairs)} tile(s) total.")
    print(f"Wrote {submit_path}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default=BASE_DIR_NAME_DEFAULT)
    parser.add_argument(
        "--runner-script-name", default=RUNNER_SCRIPT_NAME_DEFAULT,
        help="per-tile runner script under sfincs_tiles/ (e.g. run_one_tile_v3.sh for a fresh "
             "output tree that starts with no stale per-stage outputs to skip)",
    )
    parser.add_argument(
        "--tile-ids-file", default=None,
        help="default: {base_dir_name}/tile_ids.txt (no set label) - current selection output",
    )
    parser.add_argument(
        "--set-a-file", default=None,
        help="opt-in backward-compat path for an older two-set batch (validation_sfincs_v2/v3); "
             "if given (with --set-b-file), overrides --tile-ids-file entirely",
    )
    parser.add_argument("--set-b-file", default=None, help="see --set-a-file")
    parser.add_argument("--skip-tile-ids", type=str, nargs="*", default=[], help="tile IDs to exclude entirely")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--runner-extra-args", default="",
        help="extra args appended verbatim to every generated 'bash <runner> tile_id [set]' line "
             "(2026-09-24, user direction) - e.g. '--models bathtub,sfincs' to skip eikonal while "
             "its own sweep-count calibration (tests/test_sweep_budget_calibration.py) is still "
             "running, or '--models eikonal --max-rounds 25' once that study settles on a value",
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
    tile_set_pairs: list[tuple[str, str | None]] = []

    if args.set_a_file or args.set_b_file:
        set_a_file = Path(args.set_a_file) if args.set_a_file else base_dir_local / "set_a_tile_ids.txt"
        set_b_file = Path(args.set_b_file) if args.set_b_file else base_dir_local / "set_b_tile_ids.txt"
        for path, set_label in [(set_a_file, "A"), (set_b_file, "B")]:
            ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            ids = [t for t in ids if t not in skip]
            tile_set_pairs += [(t, set_label) for t in ids]
            print(f"Set {set_label}: {len(ids)} tile(s) from {path}")
    else:
        tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else base_dir_local / "tile_ids.txt"
        ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
        ids = [t for t in ids if t not in skip]
        tile_set_pairs += [(t, None) for t in ids]
        print(f"{len(ids)} tile(s) from {tile_ids_file}")

    if not tile_set_pairs:
        raise ValueError("no tile IDs to dispatch after excluding skip list")
    print(f"{len(tile_set_pairs)} tile(s) total ({len(skip)} excluded: {sorted(skip)})")

    # -- resolved_config.yml (Linux path view), staged where run_one_tile_v2.sh expects it --
    resolved_config_path = base_dir_local / "resolved_config.yml"
    retry_transient_io(base_dir_local.mkdir, parents=True, exist_ok=True)
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    print(f"Wrote {resolved_config_path} (Linux path view, for --config on every generated tile's Python steps)")

    local_jobs_dir = base_dir_local / "hpc_jobs"
    linux_jobs_dir = f"{base_dir_linux}/hpc_jobs"
    runner_script_linux = f"{linux_code_root}/sfincs_tiles/{args.runner_script_name}"

    batch_name_prefix = _batch_name_prefix(args.base_dir_name)
    submit_filename = f"submit_{batch_name_prefix}es.sh"  # e.g. 'v3_batch' -> 'submit_v3_batches.sh'

    generate_batches(
        tile_set_pairs=tile_set_pairs, n_nodes=args.n_nodes, partition=args.partition, time_limit=args.time,
        mem=args.mem, cpus_per_task=args.cpus_per_task, account=args.account,
        runner_script_linux=runner_script_linux, linux_jobs_dir=linux_jobs_dir, local_jobs_dir=local_jobs_dir,
        submit_path=local_jobs_dir / submit_filename, batch_name_prefix=batch_name_prefix,
        runner_extra_args=args.runner_extra_args,
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/{submit_filename}")
    print(f"(make sure {args.runner_script_name} is executable / callable via `bash` - no chmod needed since it's invoked as `bash <path>`)")


if __name__ == "__main__":
    main()
