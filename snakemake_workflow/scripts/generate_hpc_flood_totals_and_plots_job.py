"""Generate a single HPC sbatch script that runs compute_flood_totals.py
(parallelized via its own --workers, 2026-10-09) followed by the enabled
analysis.plot_* steps (run_analysis.py --only-plots --skip-flood-totals),
on one node.

Why a single node rather than the multi-node phase-barrier dispatch
generate_hpc_postprocess_job.py/generate_exposure_jobs.py use: both steps
here are reading many small per-chunk files that already exist (flood
fraction / population / geogunit chunks, already produced by the completed
postprocess + exposure-analysis dispatch) - the work is I/O-wait bound, not
compute-bound, so a wide thread pool on ONE node (compute_flood_totals.py's
own --workers, matching run_simulation_batch.py's established reasoning for
why N concurrent I/O-bound workers on an N-cpu node is a real use of the
allocation) already saturates the useful concurrency; splitting this across
multiple nodes would just add SLURM queueing/dependency overhead for no
real speedup, unlike simulation/postprocessing's own genuinely CPU-bound,
much larger workloads.

Both steps run in the SAME job (sequential, no SLURM dependency chain
needed) since compute_flood_totals.py must finish before run_analysis.py's
plot steps can read its output for validate_country.py integration (not
actually a dependency of the plots used here, but it's cheap and clean to
keep them in one job rather than two).

Usage:
    python generate_hpc_flood_totals_and_plots_job.py [--config path/to/config.yml]
        [--partition 24vcpu] [--cpus 24] [--mem 32G] [--time 1-00:00:00]
    bash <printed submit command>
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402
from generate_hpc_postprocess_job import _account_line, _stage_configfile_lines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--partition", default="24vcpu")
    parser.add_argument("--cpus", type=int, default=24)
    parser.add_argument("--mem", default="32G", help="a guess at this partition's safe ceiling - "
                         "confirm against `sinfo -N -o \"%%N %%P %%m %%c\"` before relying on it "
                         "(see hpc.sbatch.mem's own comment on 1vcpu's real-vs-nominal gap)")
    parser.add_argument("--time", default=None, help="default: hpc.sbatch.time from --config")
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    sbatch_cfg = dict(linux_config["hpc"]["sbatch"])
    sbatch_cfg["partition"] = args.partition
    sbatch_cfg["cpus_per_task"] = args.cpus
    sbatch_cfg["mem"] = args.mem
    if args.time:
        sbatch_cfg["time"] = args.time

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Reused if generate_hpc_postprocess_job.py already wrote one for this
    # same hpc.jobs_dir (it has, for a completed postprocess dispatch) -
    # same resolved config either way (merged_outputs etc. don't depend on
    # which script generated it), so no reason to duplicate it.
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    if not resolved_config_path.exists():
        atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    name = "flood_totals_and_plots"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={sbatch_cfg['partition']}",
        *_account_line(sbatch_cfg),
        f"#SBATCH --time={sbatch_cfg['time']}",
        f"#SBATCH --mem={sbatch_cfg['mem']}",
        f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
        f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
        "",
        "set -euo pipefail",
        sbatch_cfg["env_activate_cmd"],
        "",
        *_stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== compute_flood_totals.py ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        "",
        (
            f'python analysis/compute_flood_totals.py --config "$LOCAL_CONFIGFILE" '
            f'--outdir "$(python -c \'import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); '
            f'print(c["postprocessing"]["merged_outputs"] + "/exposure")\' "$LOCAL_CONFIGFILE")" '
            f'--workers {sbatch_cfg["cpus_per_task"]}'
        ),
        "",
        'echo "=== run_analysis.py --only-plots (flood totals already done above) ==="',
        'python analysis/run_analysis.py --config "$LOCAL_CONFIGFILE" --only-plots --skip-flood-totals',
        "",
    ]
    script_path = local_jobs_dir / f"{name}.sbatch"
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"Wrote {script_path}")
    print(f"Submit on Hydrax with: sbatch {linux_jobs_dir}/{name}.sbatch")


if __name__ == "__main__":
    main()
