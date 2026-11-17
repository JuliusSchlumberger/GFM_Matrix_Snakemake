"""Write one single-node job for compute_flood_totals.py + analysis plots; entry point `generate()`.

Runs compute_flood_totals.py (--workers = the flood_totals profile's cpus_per_task) then
run_analysis.py --only-plots --skip-flood-totals in one job. One node suffices: the work is
I/O-bound reads of existing per-chunk files, so a wide thread pool saturates useful concurrency.

Usage:
    python generate_hpc_flood_totals_and_plots_job.py [--config path/to/config.yml] [--profile-set P.F=V]
    bash <printed submit script>
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import atomic_write, load_config, load_hpc_config, retry_transient_io  # noqa: E402
from hpc_common import JobChain, add_profile_set_argument, apply_profile_sets, resolve_profile  # noqa: E402

NAME = "flood_totals_and_plots"


def add_flood_totals_job(chain: JobChain, linux_config: dict, after: list[str]) -> str:
    """Add the flood-totals + plots job to `chain`; return its name."""
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    workers = resolve_profile(chain.hpc_cfg, "flood_totals")["cpus_per_task"]
    chain.add(NAME, "flood_totals", [
        f'cd "{linux_config["paths"]["code_root"]}"',
        'echo "=== compute_flood_totals.py ==="',
        f'stage_configfile_locally "{linux_jobs_dir}/resolved_config.yml" LOCAL_CONFIGFILE || exit 1',
        "",
        # Own checked line: inside an argument, a failing $(...) would go unnoticed.
        (
            'OUTDIR=$("$GFM_PY" -c \'import yaml,sys; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); '
            'print(c["postprocessing"]["merged_outputs"] + "/exposure")\' "$LOCAL_CONFIGFILE") || exit 1'
        ),
        f'"$GFM_PY" analysis/compute_flood_totals.py --config "$LOCAL_CONFIGFILE" --out-dir "$OUTDIR" --workers {workers}',
        "",
        'echo "=== run_analysis.py --only-plots (flood totals already done above) ==="',
        '"$GFM_PY" analysis/run_analysis.py --config "$LOCAL_CONFIGFILE" --only-plots --skip-flood-totals',
    ], after=after, helpers=("stage_config",))
    return NAME


def generate(config_path: str | Path, *, profile_sets: list[str] | None = None) -> str:
    """Write the job, resolved_config.yml and a submit script; return the submit script's Linux path."""
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    hpc_cfg = apply_profile_sets(linux_config["hpc"], profile_sets)
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)
    atomic_write(local_jobs_dir / "resolved_config.yml", lambda f: yaml.safe_dump(linux_config, f),
                 encoding="utf-8", newline="")
    chain = JobChain(hpc_cfg, local_jobs_dir, hpc_cfg["jobs_dir"])
    add_flood_totals_job(chain, linux_config, after=[])
    return chain.write_submit(f"submit_{NAME}.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"))
    add_profile_set_argument(parser)
    args = parser.parse_args()
    submit = generate(args.config, profile_sets=args.profile_set)
    print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
