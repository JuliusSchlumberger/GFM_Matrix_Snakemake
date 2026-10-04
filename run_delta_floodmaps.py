"""Run the global-deltas flood hazard map deliverable locally, for one or
both scenarios (ssp126 = RCP2.6-equivalent, ssp245 = RCP4.5-equivalent),
at friction_scale_factor=9.0, RP100, all 5 production SLR magnitudes, over
the 400-tile delta subset (48 of 49 deltas - rebuilt 2026-10 against the
connectivity-first production grid; see
preparation/build_delta_tile_subset.py and
snakemake_workflow/config/deltas_ssp126.yml for the full rationale).

For each scenario, runs in order:
  1. preparation/run_preparation.py boundary_conditions - builds that
     scenario's own SLR fingerprint NetCDFs into its isolated
     waterlevel_nc_dir (never touches production's or the other
     scenario's boundary-condition data).
  2. snakemake preprocess  - DEM/mask/friction/boundary extraction for the
     400 delta tiles.
  3. snakemake simulate    - RP100 x 5 SLR magnitudes x 400 tiles.
  4. snakemake postprocess - merge/mosaic per-tile results.

Each Snakemake call is run with GFM_CONFIG_PATH pointed at that scenario's
materialized config (snakemake_workflow/config/deltas_{scenario}_
materialized.yml) - same mechanism the Wales/Scotland friction9 run used,
NOT --configfile (see Snakefile's own comment on why). Every stage call
also passes `--rerun-incomplete` - if a run is ever interrupted (crash,
Ctrl+C, a killed background process) mid-job, Snakemake's own metadata
correctly flags whatever was actively being written at that moment as
incomplete and refuses to trust it; this flag tells it to just redo THOSE
specific targets, not everything, rather than requiring a manual
`snakemake --rerun-incomplete` afterward.

Subprocess output streams straight to this console (not captured) so you
see exactly what a manual run would show. Stops immediately on the first
failing stage for a scenario (the remaining stages for that scenario
would read incomplete/missing inputs) - that scenario is marked FAILED in
the final summary, and any OTHER scenario already queued still runs.

Usage:
    python run_delta_floodmaps.py                          # both scenarios, all stages
    python run_delta_floodmaps.py --scenarios ssp126        # just one
    python run_delta_floodmaps.py --cores 4 --mem-mb 24000
    python run_delta_floodmaps.py --skip-boundary-conditions --stages preprocess simulate
    python run_delta_floodmaps.py --stages postprocess       # re-run just one stage
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "snakemake_workflow" / "config"

ALL_SCENARIOS = ["ssp126", "ssp245"]
ALL_STAGES = ["boundary_conditions", "preprocess", "simulate", "postprocess"]


def _run(cmd: list[str], env: dict | None = None) -> bool:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return result.returncode == 0


def run_scenario(scenario: str, cores: int, mem_mb: int | None, stages: list[str]) -> dict[str, bool]:
    import os

    materialized = CONFIG_DIR / f"deltas_{scenario}_materialized.yml"
    if not materialized.exists():
        raise FileNotFoundError(
            f"{materialized} does not exist - run "
            f"snakemake_workflow/scripts/build_delta_run_configs.py first."
        )

    env = os.environ.copy()
    env["GFM_CONFIG_PATH"] = str(materialized)

    resources = [f"--resources", f"mem_mb={mem_mb}"] if mem_mb else []
    results: dict[str, bool] = {}

    for stage in stages:
        print(f"\n{'=' * 60}\n  [{scenario}] {stage}\n{'=' * 60}")
        t0 = time.time()
        if stage == "boundary_conditions":
            ok = _run([sys.executable, "preparation/run_preparation.py", "boundary_conditions",
                       "--config", str(materialized)])
        else:
            ok = _run(["snakemake", stage, "--cores", str(cores), *resources, "-p", "--rerun-incomplete"], env=env)
        elapsed = time.time() - t0
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {scenario}/{stage} in {elapsed:.0f}s")
        results[stage] = ok
        if not ok:
            print(f"  Stopping {scenario} - later stages would read incomplete/missing inputs.")
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS, default=ALL_SCENARIOS)
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--mem-mb", type=int, default=None)
    parser.add_argument("--skip-boundary-conditions", action="store_true",
                         help="shorthand for --stages preprocess simulate postprocess")
    args = parser.parse_args()

    stages = args.stages
    if args.skip_boundary_conditions:
        stages = [s for s in stages if s != "boundary_conditions"]

    all_results: dict[str, dict[str, bool]] = {}
    t_start = time.time()
    for scenario in args.scenarios:
        all_results[scenario] = run_scenario(scenario, args.cores, args.mem_mb, stages)

    total = time.time() - t_start
    print(f"\n{'=' * 60}\n  Delta flood-hazard-map run complete ({total / 60:.1f} min)\n{'=' * 60}")
    any_failed = False
    for scenario, results in all_results.items():
        for stage, ok in results.items():
            icon = "[OK]" if ok else "[FAIL]"
            print(f"  {icon}  {scenario}/{stage}")
            any_failed = any_failed or not ok

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
