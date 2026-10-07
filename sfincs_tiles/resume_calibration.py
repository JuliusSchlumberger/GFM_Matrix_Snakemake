"""THE single resume command for a friction-sweep calibration study (e.g.
sfincs_calibration): regenerates report_calibration_tile_status.py's two
targets fresh (never stale - reads real file presence on disk right now),
then dispatches whichever repair path(s) actually have work:

  - tile_ids_needs_rebuild.txt non-empty -> generates a full per-tile
    pipeline rerun batch (generate_validation_batch_jobs.py --defer-eikonal),
    split across --n-nodes. Tiles permanently unsolvable (no_station/
    no_boundary_cells/antimeridian - tile_status.py) are skipped INSTANTLY
    by run_one_tile.sh's own early short-circuit, so including them here is
    harmless, not wasteful.
  - missing_eikonal_pairs.csv non-empty -> generates sweep-only batches
    (run_friction_sweep_batch.py directly, bypassing the full per-tile
    pipeline since SFINCS is already built for these tiles), split across
    --n-nodes-sweep.

Prints the submit command(s) - run on Hydrax yourself, same convention
every other generate_*_job.py script in this repo uses (writes sbatch
scripts, never submits them itself).

Usage:
    python resume_calibration.py --base-dir-name sfincs_calibration --n-nodes 6 --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

FRICTION_SCALE_FACTORS_DEFAULT = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
GFM_PY_LINUX = "/u/schlumbe/.conda/envs/gfm/bin/python"  # matches generate_validation_batch_jobs.py's own constant -
# full binary path, not `conda activate` (a non-interactive sbatch subshell doesn't source ~/.bashrc).


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _write_sweep_batches(
    base_dir: Path, pairs: list[tuple[str, float]], n_nodes: int, partition: str, cpus_per_task: int,
    mem: str, time_limit: str, config_linux: str, jobs_dir_local: Path, jobs_dir_linux: str,
    code_root_linux: str, base_dir_name: str, max_outer_iterations: int,
) -> Path:
    """Splits `pairs` evenly across n_nodes sbatch scripts, each a direct
    run_friction_sweep_batch.py call (no per-tile pipeline rerun - these
    tiles already have SFINCS built, only the sweep itself is missing)."""
    n_batches = min(n_nodes, len(pairs))
    k, m = divmod(len(pairs), n_batches)
    jobs_dir_local.mkdir(parents=True, exist_ok=True)
    (jobs_dir_local / "logs").mkdir(parents=True, exist_ok=True)

    script_paths = []
    for i in range(n_batches):
        batch_pairs = pairs[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if not batch_pairs:
            continue
        name = f"resume_sweep_batch_{i:03d}"
        pairs_path = jobs_dir_local / f"{name}_pairs.csv"
        pairs_path.write_text("\n".join(f"{tid},{fsf:g}" for tid, fsf in batch_pairs) + "\n", encoding="utf-8", newline="\n")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --output={jobs_dir_linux}/logs/{name}_%j.out",
            f"#SBATCH --error={jobs_dir_linux}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",
            f'cd "{code_root_linux}/sfincs_tiles"',
            (
                f'"{GFM_PY_LINUX}" run_friction_sweep_batch.py --pairs-file "{jobs_dir_linux}/{name}_pairs.csv" '
                f'--base-dir-name "{base_dir_name}" --config "{config_linux}" '
                f'--workers {cpus_per_task} --python "{GFM_PY_LINUX}" --max-outer-iterations {max_outer_iterations}'
            ),
            "",
        ]
        script_path = jobs_dir_local / f"{name}.sbatch"
        script_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        script_paths.append(f"{jobs_dir_linux}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_pairs)} pair(s))")

    submit_path = jobs_dir_local / "submit_resume_sweep_batches.sh"
    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for script in script_paths:
        submit_lines += [f'JID=$(sbatch --parsable "{script}")', f'echo "submitted {script} -> job $JID"']
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8", newline="\n")
    return submit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument(
        "--preprocess-config", default=None,
        help="forwarded to generate_validation_batch_jobs.py for the rebuild path - default: "
             "snakemake_workflow/config/{base-dir-name}_preprocess_materialized.yml",
    )
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT)
    parser.add_argument("--max-outer-iterations", type=int, default=5)
    parser.add_argument("--n-nodes", type=int, default=6, help="rebuild path (generate_validation_batch_jobs.py)")
    parser.add_argument("--n-nodes-sweep", type=int, default=6, help="sweep-only path (run_friction_sweep_batch.py)")
    parser.add_argument("--partition", default="4vcpu")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="30G")
    parser.add_argument("--time", default="16:00:00")
    args = parser.parse_args()

    python = sys.executable

    # Step 1: fresh status - never stale, reads real file presence on disk.
    _run([
        python, str(_THIS_DIR / "report_calibration_tile_status.py"),
        "--config", args.config, "--base-dir-name", args.base_dir_name,
        "--friction-scale-factors", *[str(f) for f in args.friction_scale_factors],
        "--max-outer-iterations", str(args.max_outer_iterations),
    ])

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    # Step 2: rebuild path.
    rebuild_path = base_dir / "tile_ids_needs_rebuild.txt"
    needs_rebuild = [l.strip() for l in rebuild_path.read_text().splitlines() if l.strip()] if rebuild_path.exists() else []
    if needs_rebuild:
        preprocess_config = args.preprocess_config or str(
            _REPO_ROOT / "snakemake_workflow" / "config" / f"{args.base_dir_name}_preprocess_materialized.yml"
        )
        _run([
            python, str(_THIS_DIR / "generate_validation_batch_jobs.py"),
            "--config", args.config, "--base-dir-name", args.base_dir_name,
            "--tile-ids-file", str(rebuild_path), "--preprocess-config", preprocess_config,
            "--defer-eikonal", "--defer-eikonal-max-outer-iterations", str(args.max_outer_iterations),
            "--defer-eikonal-factors", *[str(f) for f in args.friction_scale_factors],
            "--n-nodes", str(args.n_nodes), "--partition", args.partition,
            "--cpus-per-task", str(args.cpus_per_task), "--mem", args.mem, "--time", args.time,
        ])
    else:
        print("\nNo tiles need a full rebuild - skipping that path.")

    # Step 3: sweep-only path.
    pairs_path = base_dir / "missing_eikonal_pairs.csv"
    pairs: list[tuple[str, float]] = []
    if pairs_path.exists():
        for line in pairs_path.read_text().splitlines():
            if not line.strip():
                continue
            tid, fsf = line.split(",")
            pairs.append((tid, float(fsf)))
    if pairs:
        # Linux-view config (same pattern generate_validation_batch_jobs.py itself uses) -
        # config_hpc.yml is a git-ignored, Windows-machine-only override.
        import yaml
        config_path = Path(args.config)

        def _load_yaml_merged(path: Path, override: Path | None) -> dict:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if override and override.exists():
                ov = yaml.safe_load(override.read_text(encoding="utf-8"))
                data = {**data, **ov}
            return data

        linux_raw = _load_yaml_merged(config_path, config_path.parent / "config_hpc.yml")
        linux_root = linux_raw["paths"]["root"]
        linux_code_root = linux_raw["paths"]["code_root"]
        linux_base_dir = f"{linux_root}/{args.base_dir_name}"
        linux_config_path = f"{linux_base_dir}/resolved_config.yml"

        jobs_dir_local = base_dir / "hpc_jobs"
        jobs_dir_linux = f"{linux_base_dir}/hpc_jobs"

        submit_path = _write_sweep_batches(
            base_dir, pairs, args.n_nodes_sweep, args.partition, args.cpus_per_task, args.mem, args.time,
            linux_config_path, jobs_dir_local, jobs_dir_linux, linux_code_root, args.base_dir_name,
            args.max_outer_iterations,
        )
        print(f"\nWrote {submit_path}")
        print(f"Submit on Hydrax with: bash {jobs_dir_linux}/{submit_path.name}")
    else:
        print("\nNo missing sweep pairs - skipping that path.")


if __name__ == "__main__":
    main()
