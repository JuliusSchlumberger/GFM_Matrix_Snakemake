"""Run the global-deltas flood hazard maps locally (ssp126 / ssp245, RP100, 400 tiles).

Stages per scenario: boundary_conditions, preprocess, simulate, postprocess, with
GFM_CONFIG_PATH = deltas_{scenario}_materialized.yml. simulate runs in ascending
hop_distance waves: a hop>=1 tile whose lower-hop neighbours are not yet done is
permanently zeroed as "no upstream flooding". A failing stage stops its scenario.

Usage:
    python run_delta_floodmaps.py                          # both scenarios, all stages
    python run_delta_floodmaps.py --scenarios ssp126        # just one
    python run_delta_floodmaps.py --cores 4 --mem-mb 24000
    python run_delta_floodmaps.py --skip-boundary-conditions --stages preprocess simulate
    python run_delta_floodmaps.py --stages postprocess       # re-run just one stage
    # design-event floods (SLR_0) of the FLOPROS-protected deltas, after analysis/prepare_delta_design_runs.py:
    python run_delta_floodmaps.py --scenarios design --stages boundary_conditions preprocess simulate
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "snakemake_workflow" / "config"

ALL_SCENARIOS = ["ssp126", "ssp245"]
# "design": SLR_0 design-event floods for FLOPROS-protected deltas; config written by
# analysis/prepare_delta_design_runs.py.
EXTRA_SCENARIOS = ["design"]
ALL_STAGES = ["boundary_conditions", "preprocess", "simulate", "postprocess"]

# Keeps each call under Windows' ~32,767-char CreateProcess limit (~130 chars/path).
_MAX_TARGETS_PER_SNAKEMAKE_CALL = 150


def _run(cmd: list[str], env: dict | None = None) -> bool:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from step_runner import run_step

    return run_step(cmd, label=" ".join(cmd[:2]), cwd=REPO_ROOT, env=env)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_simulate_stage(materialized: Path, cores: int, mem_mb: int | None, env: dict) -> bool:
    """Run `simulate` wave by wave in ascending hop_distance, blocking per wave.

    Targets match the Snakefile's _SIMULATION_OUTPUTS, split by hop_distance.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import geopandas as gpd
    from config_utils import load_config, merged_slr_scenarios

    cfg = load_config(materialized)
    tile_grid = gpd.read_file(cfg["tile_grid"]["path"])
    tile_grid["tile_id"] = tile_grid["tile_id"].astype(int)
    model_outputs = cfg["simulation"]["model_outputs"]
    return_periods = [f"RP{rp}" for rp in cfg["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(cfg["boundary_conditions"], cfg["adaptation"])

    resources = ["--resources", f"mem_mb={mem_mb}"] if mem_mb else []
    waves = sorted(tile_grid["hop_distance"].unique().tolist())
    for wave in waves:
        wave_tile_ids = tile_grid.loc[tile_grid["hop_distance"] == wave, "tile_id"].tolist()
        targets = [
            f"{model_outputs}/{tid}/results/waterdepth_{rp}_{wl}.tif"
            for tid in wave_tile_ids for rp in return_periods for wl in waterlevel_names
        ]
        print(f"\n  -- simulate wave hop_distance={wave}: {len(wave_tile_ids)} tile(s), "
              f"{len(targets)} target(s) --")
        for chunk in _chunked(targets, _MAX_TARGETS_PER_SNAKEMAKE_CALL):
            ok = _run(["snakemake", "--cores", str(cores), *resources, "-p",
                       "--rerun-incomplete", *chunk], env=env)
            if not ok:
                print(f"  Stopping simulate - wave hop_distance={wave} failed; "
                      f"later waves would read incomplete neighbour outputs.")
                return False
    return True


def run_scenario(scenario: str, cores: int, mem_mb: int | None, stages: list[str]) -> dict[str, bool]:
    import os

    materialized = CONFIG_DIR / f"deltas_{scenario}_materialized.yml"
    if not materialized.exists():
        raise FileNotFoundError(
            f"{materialized} does not exist - run "
            + ("analysis/prepare_delta_design_runs.py" if scenario in EXTRA_SCENARIOS
               else f"snakemake_workflow/scripts/materialize_configs.py deltas_{scenario}") + " first."
        )

    env = os.environ.copy()
    env["GFM_CONFIG_PATH"] = str(materialized)

    resources = ["--resources", f"mem_mb={mem_mb}"] if mem_mb else []
    results: dict[str, bool] = {}

    for stage in stages:
        print(f"\n{'=' * 60}\n  [{scenario}] {stage}\n{'=' * 60}")
        t0 = time.time()
        if stage == "boundary_conditions":
            ok = _run([sys.executable, "preparation/run_preparation.py", "boundary_conditions",
                       "--config", str(materialized)])
        elif stage == "simulate":
            ok = run_simulate_stage(materialized, cores, mem_mb, env)
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
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS + EXTRA_SCENARIOS, default=ALL_SCENARIOS,
                        help="'design' = design-event floods for the protected deltas (prepare_delta_design_runs.py)")
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
