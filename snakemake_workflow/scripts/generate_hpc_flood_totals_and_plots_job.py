"""Generate one single-node sbatch script for compute_flood_totals.py + analysis plots.

Runs compute_flood_totals.py (--workers) then run_analysis.py --only-plots
--skip-flood-totals in one job. One node suffices: the work is I/O-bound reads
of existing per-chunk files, so a wide thread pool saturates useful concurrency.

Usage:
    python generate_hpc_flood_totals_and_plots_job.py [--config path/to/config.yml]
        [--partition P] [--cpus-per-task N] [--mem M] [--time 1-00:00:00]
    bash <printed submit command>
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import atomic_write, load_config, load_hpc_config, retry_transient_io  # noqa: E402
from hpc_common import _stage_configfile_lines, job_sbatch_cfg, sbatch_header  # noqa: E402


def write_flood_totals_and_plots_job(
    local_config: dict, linux_config: dict,
    partition: str | None = None, cpus: int | None = None, mem: str | None = None, time: str | None = None,
) -> tuple[Path, str]:
    """Write the flood_totals_and_plots.sbatch script, without submitting it.

    Resources default to hpc.flood_totals_sbatch (falling back to hpc.sbatch);
    `cpus` is also passed as compute_flood_totals.py's --workers. Reuses an
    existing resolved_config.yml in hpc.jobs_dir. Returns the local script path
    and its Linux path.
    """
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    sbatch_cfg = job_sbatch_cfg(linux_config["hpc"], "flood_totals_sbatch")
    for key, value in (("partition", partition), ("cpus_per_task", cpus), ("mem", mem), ("time", time)):
        if value:
            sbatch_cfg[key] = value

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Reuse generate_hpc_postprocess_job.py's resolved_config.yml if present.
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    if not resolved_config_path.exists():
        atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    name = "flood_totals_and_plots"
    lines = [
        *sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs"),
        *_stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== compute_flood_totals.py ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        "",
        # Own checked line: inside an argument, a failing $(...) would go unnoticed.
        (
            'OUTDIR=$(python -c \'import yaml,sys; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); '
            'print(c["postprocessing"]["merged_outputs"] + "/exposure")\' "$LOCAL_CONFIGFILE") || exit 1'
        ),
        (
            f'python analysis/compute_flood_totals.py --config "$LOCAL_CONFIGFILE" '
            f'--out-dir "$OUTDIR" --workers {sbatch_cfg["cpus_per_task"]}'
        ),
        "",
        'echo "=== run_analysis.py --only-plots (flood totals already done above) ==="',
        'python analysis/run_analysis.py --config "$LOCAL_CONFIGFILE" --only-plots --skip-flood-totals',
        "",
    ]
    script_path = local_jobs_dir / f"{name}.sbatch"
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return script_path, f"{linux_jobs_dir}/{name}.sbatch"


def main() -> None:
    """Write the flood-totals-and-plots sbatch script and print the submit command."""
    # allow_abbrev=False: a removed flag that is a prefix of its replacement must be an error.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
    )
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--partition", default=None, help="default: hpc.flood_totals_sbatch.partition")
    parser.add_argument("--cpus-per-task", type=int, default=None, help="default: hpc.flood_totals_sbatch.cpus_per_task")
    parser.add_argument("--mem", default=None, help="default: hpc.flood_totals_sbatch.mem")
    parser.add_argument("--time", default=None, help="default: hpc.sbatch.time from --config")
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)

    script_path, linux_script_path = write_flood_totals_and_plots_job(
        local_config, linux_config, args.partition, args.cpus_per_task, args.mem, args.time,
    )
    print(f"Wrote {script_path}")
    print(f"Submit on the cluster with: sbatch {linux_script_path}")


if __name__ == "__main__":
    main()
