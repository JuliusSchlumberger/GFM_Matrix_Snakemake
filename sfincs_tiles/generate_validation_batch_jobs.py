"""Generates N independent sbatch scripts that dispatch a validation batch
(see select_validation_tiles.py, which writes tile_ids.txt) end to end, one
job per script, each looping sequentially through its own tile-id slice and
calling run_one_tile.sh <tile_id> per tile.

Submitted as N independent sbatch scripts (not a SLURM --array job), so
SLURM schedules each onto a node as one frees up.

Resolves both the local (Windows, for writing files here) and Linux (HPC,
for the generated scripts' own paths) views of paths.root from the same
config.yml.

--preprocess-config PATH (2026-10, new): prepends a plain aqueduct
preprocessing snakemake call (dem/mask/friction/boundaries - the inputs
run_one_tile.sh itself then copies from model_outputs/) to the FRONT of
each batch's own sbatch script, for that SAME batch's own tile_ids, before
its run_one_tile.sh loop - one combined submit step instead of two
sequential ones (generate_hpc_preprocess_job.py --preprocess-only run
first, this script's own submit run second). Valid for any base_dir_name's
tile set - not sfincs_calibration-specific - as long as the given config's
tile_grid already covers (a superset of) this script's own tile_ids.txt;
reuses generate_hpc_preprocess_job.py's own target-path/retry/config-
staging helpers directly (one implementation, two entry points) rather
than reimplementing them. Writes the shared, tile-independent preprocessing
inputs (geoid-offset raster, per-scenario cached water-level stations)
SYNCHRONOUSLY in the submit script before any batch is submitted, same
--nolock-safety reasoning as that script's own module docstring - a batch
script itself passes --nolock, so a concurrent first-time build of either
shared file across 40+ simultaneously-starting batches would otherwise
race. Preprocessing writes into THAT config's own simulation.model_outputs
(typically the shared production tree, not this study's own base_dir_name)
- same tile-geometry-is-not-scenario-isolated reasoning documented in
snakemake_workflow/config/sfincs_calibration_preprocess.yml.

--defer-eikonal (2026-10, new): each tile's eikonal solve is single-
threaded numba (see run_friction_sweep_batch.py's own docstring), but
run_one_tile.sh's default per-tile loop runs it BEFORE that same tile's
SFINCS build+run, serially, one tile at a time - wasting (cpus_per_task-1)
of the node's cores for the whole batch's accumulated eikonal time, and
gaining nothing from it (eikonal doesn't block or feed into the SFINCS
build/run at all - see run_eikonal_on_sfincs_subgrid.py; it only needs the
SFINCS model's own subgrid tables, built in step 3, not the actual SFINCS
run's output). With --defer-eikonal: (1) the main per-tile loop runs
`--models bathtub,sfincs` only (skips eikonal, so each tile's own
postprocess_tile_summary.py call writes null eikonal fields for now); (2)
once every tile in the batch has gone through that, one
run_friction_sweep_batch.py invocation runs eikonal for the FULL (tile_id,
friction_scale_factor) cross product of this batch's own tiles x
--defer-eikonal-factors (default: the real 10-point sweep,
EIKONAL_DEFERRED_FACTORS_DEFAULT - same list generate_friction_sweep_jobs.py
defaults to, so the standalone sweep dispatch that script generates is now
redundant for a base_dir_name built with --defer-eikonal and doesn't need
to be run separately), CONCURRENTLY across cpus_per_task workers; (3) a
final pass re-invokes run_one_tile.sh per tile with `--models ""` (runs
nothing but postprocess_tile_summary.py, which always re-reads whatever is
on disk right now) so every tile's summary.json picks up the DEFAULT
factor's eikonal stats specifically (run_eikonal_on_sfincs_subgrid.py's own
_fsf_tag logic keeps exactly one factor - whichever equals
FRICTION_SCALE_FACTOR_DEFAULT - at the untagged filename
postprocess_tile_summary.py reads; every other factor's output is its own
separately-tagged file, read later by compute_friction_sweep_metrics.py,
not by postprocess_tile_summary.py). Net effect: the batch's entire eikonal
sweep (not just its default point) drops from fully serial to
~1/cpus_per_task of its own wall-clock, overlapping none of it with idle
cores during the SFINCS-dominated main loop, AND happens in the same
submission as the SFINCS runs instead of a separate later one.

Usage:
    python generate_validation_batch_jobs.py --base-dir-name validation_sfincs_v5
    python generate_validation_batch_jobs.py --base-dir-name sfincs_calibration \\
        --preprocess-config snakemake_workflow/config/sfincs_calibration_preprocess_materialized.yml
    bash <printed submit_..._batches.sh path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "snakemake_workflow" / "scripts"))
from generate_hpc_preprocess_job import (  # noqa: E402
    _retry_wrapper_lines, _stage_configfile_lines, _target_paths,
)

GFM_PY_LINUX = "/u/schlumbe/.conda/envs/gfm/bin/python"  # matches run_one_tile.sh's own hardcoded $GFM_PY -
# each generated batch script is its own bash process, separate from run_one_tile.sh's, so this can't be
# inherited and must be redefined here too (only used by --defer-eikonal's own lines).
EIKONAL_DEFERRED_FACTORS_DEFAULT = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]  # same list as
# generate_friction_sweep_jobs.py's own FRICTION_SCALE_FACTORS_DEFAULT (0.1x-1.0x of production's
# simulation.flooding.friction_scale_factor=30.0) - --defer-eikonal's default IS the real sweep, not
# just the single production value, now that the sweep is fused in here rather than dispatched
# separately (see --defer-eikonal's own module-docstring entry). 30.0 keeps its untagged, "real"
# filename (run_eikonal_on_sfincs_subgrid.py's own _fsf_tag logic) regardless of list order/content,
# so fusing more factors in never changes what postprocess_tile_summary.py's own eikonal stats mean.

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
    base_dir_name: str, runner_extra_args: str = "", preprocess: dict | None = None,
    defer_eikonal: bool = False, sfincs_resolved_config_linux: str | None = None,
    eikonal_deferred_factors: list[float] | None = None,
) -> None:
    """`preprocess`, if given (see --preprocess-config), is a dict with
    keys: return_periods, waterlevel_names, linux_model_outputs,
    linux_resolved_config, linux_shared_targets_file - prepends an aqueduct
    preprocessing snakemake call (this batch's own tile_ids, as explicit
    targets) to the front of every generated batch script, and a
    synchronous shared-inputs build to the front of the submit script - see
    module docstring.

    `defer_eikonal`/`sfincs_resolved_config_linux`/`eikonal_deferred_factors`
    - see --defer-eikonal's own module-docstring entry.
    """
    eikonal_deferred_factors = eikonal_deferred_factors if eikonal_deferred_factors is not None else EIKONAL_DEFERRED_FACTORS_DEFAULT
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
        if preprocess is not None:
            targets = [
                p for tile_id in batch_tile_ids
                for p in _target_paths(
                    f"{preprocess['linux_model_outputs']}/{tile_id}",
                    preprocess["return_periods"], preprocess["waterlevel_names"],
                )
            ]
            targets_path = local_jobs_dir / f"{name}_preprocess_targets.txt"
            with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(targets) + "\n")
            lines += [
                preprocess["env_activate_cmd"],
                "",
                *_retry_wrapper_lines(),
                *_stage_configfile_lines(),
                "",
                f'cd "{preprocess["linux_code_root"]}"',
                f'echo "=== Preprocessing batch {batch_id}: {len(batch_tile_ids)} tile(s) ==="',
                f'stage_configfile_locally "{preprocess["linux_resolved_config"]}" PREPROCESS_CONFIGFILE || exit 1',
                "",
                f'while IFS= read -r shared_target; do',
                f'    if [ ! -f "$shared_target" ]; then',
                f'        echo "ERROR: shared preprocessing input not found: $shared_target" >&2',
                f'        echo "This batch must not start before the shared-inputs build (in'
                f' {submit_path.name}) completes." >&2',
                "        exit 1",
                "    fi",
                f'done < "{preprocess["linux_shared_targets_file"]}"',
                "",
                (
                    f'GFM_CONFIG_PATH="$PREPROCESS_CONFIGFILE" run_snakemake_with_retry '
                    f'snakemake --cores {cpus_per_task} --nolock --rerun-triggers=mtime --rerun-incomplete '
                    f'$(cat "{linux_jobs_dir}/{name}_preprocess_targets.txt")'
                ),
                "",
                f'echo "=== Running SFINCS batch {batch_id}: {len(batch_tile_ids)} tile(s) ==="',
            ]
        main_extra = "--models bathtub,sfincs" if defer_eikonal else runner_extra_args
        for tile_id in batch_tile_ids:
            extra = f" {main_extra}" if main_extra else ""
            # BASE_DIR_NAME=... prefix: run_one_tile.sh requires this env var explicitly.
            lines.append(f'BASE_DIR_NAME="{base_dir_name}" bash "{runner_script_linux}" {tile_id}{extra}')
        lines.append("")

        if defer_eikonal:
            pairs_path = local_jobs_dir / f"{name}_eikonal_pairs.csv"
            with open(pairs_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(
                    f"{tile_id},{fsf}" for fsf in eikonal_deferred_factors for tile_id in batch_tile_ids
                ) + "\n")
            friction_sweep_batch_linux = f"{Path(runner_script_linux).parent.as_posix()}/run_friction_sweep_batch.py"
            n_pairs = len(batch_tile_ids) * len(eikonal_deferred_factors)
            lines += [
                f'GFM_PY="{GFM_PY_LINUX}"',
                f'echo "=== Eikonal sweep (concurrent, {cpus_per_task} worker(s)) for batch {batch_id}:'
                f' {len(batch_tile_ids)} tile(s) x {len(eikonal_deferred_factors)} factor(s) = {n_pairs} pair(s) ==="',
                (
                    f'"$GFM_PY" "{friction_sweep_batch_linux}" --pairs-file "{linux_jobs_dir}/{name}_eikonal_pairs.csv" '
                    f'--base-dir-name "{base_dir_name}" --config "{sfincs_resolved_config_linux}" '
                    f'--workers {cpus_per_task} --python "$GFM_PY"'
                ),
                "",
                f'echo "=== Re-running postprocess (eikonal stats) for batch {batch_id}: {len(batch_tile_ids)} tile(s) ==="',
            ]
            for tile_id in batch_tile_ids:
                lines.append(f'BASE_DIR_NAME="{base_dir_name}" bash "{runner_script_linux}" {tile_id} --models ""')
            lines.append("")

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tile_ids)} tile(s))")

    # Fully independent batches - every tile is self-contained end to end, so
    # every batch submits immediately, no --dependency chain.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    if preprocess is not None:
        submit_lines += [
            preprocess["env_activate_cmd"],
            "",
            *_retry_wrapper_lines(),
            *_stage_configfile_lines(),
            "",
            f'cd "{preprocess["linux_code_root"]}"',
            'echo "=== Building shared preprocessing inputs (synchronous, on this login node) ==="',
            f'stage_configfile_locally "{preprocess["linux_resolved_config"]}" PREPROCESS_CONFIGFILE || exit 1',
            (
                f'GFM_CONFIG_PATH="$PREPROCESS_CONFIGFILE" run_snakemake_with_retry '
                'snakemake --cores 1 --nolock --rerun-triggers=mtime '
                f'$(cat "{preprocess["linux_shared_targets_file"]}") '
                f'2>&1 | tee "{linux_jobs_dir}/logs/build_shared_inputs.log"'
            ),
            "",
        ]
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
    parser.add_argument(
        "--preprocess-config", default=None,
        help="prepend an aqueduct preprocessing snakemake call (dem/mask/friction/boundaries - "
             "the files run_one_tile.sh itself then copies in) to each batch, for that batch's own "
             "tiles, using THIS config's tile_grid/boundary_conditions/simulation.model_outputs "
             "(typically a different, isolated config from --config above, which stays the plain "
             "production one run_one_tile.sh's own per-tile Python steps read) - see module "
             "docstring. Skips this entirely (two separate submit steps, as before) if omitted.",
    )
    parser.add_argument(
        "--defer-eikonal", action="store_true",
        help="run each batch's eikonal solves (the full --defer-eikonal-factors sweep, by default) as "
             "one concurrent (cpus_per_task-worker) pass AFTER its main bathtub+sfincs loop, instead "
             "of serially before each tile's own SFINCS run - see module docstring. Mutually exclusive "
             "with putting --models in --runner-extra-args (this flag controls --models itself).",
    )
    parser.add_argument(
        "--defer-eikonal-factors", type=float, nargs="+", default=EIKONAL_DEFERRED_FACTORS_DEFAULT,
        help="friction_scale_factor value(s) to sweep in --defer-eikonal's own pass - default is the "
             "real 10-point sweep (same list generate_friction_sweep_jobs.py defaults to). Pass a "
             "single value (e.g. just 30.0) to only compute the production default, with no sweep.",
    )
    args = parser.parse_args()
    if args.defer_eikonal and "--models" in args.runner_extra_args:
        parser.error("--defer-eikonal already controls --models (bathtub,sfincs then \"\") - "
                     "don't also pass --models via --runner-extra-args")

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

    preprocess = None
    if args.preprocess_config:
        pre_path = Path(args.preprocess_config)
        pre_local = load_config(pre_path)
        pre_linux = load_config(pre_path, extra_override=pre_path.parent / "config_hpc.yml")

        retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
        pre_resolved_path = local_jobs_dir / "resolved_config_preprocess.yml"
        atomic_write(str(pre_resolved_path), lambda f: yaml.safe_dump(pre_linux, f), encoding="utf-8", newline="")
        print(f"Wrote {pre_resolved_path} (preprocessing's own config - distinct from {resolved_config_path.name})")

        return_periods = [f"RP{rp}" for rp in pre_linux["boundary_conditions"]["return_periods"]]
        waterlevel_names = merged_slr_scenarios(pre_linux["boundary_conditions"], pre_linux["adaptation"])

        # Same shared, tile-independent preprocessing targets generate_hpc_preprocess_job.py
        # itself builds (geoid-offset raster + per-scenario cached water-level stations) -
        # see this script's own module docstring for why they must be built synchronously,
        # once, before any --nolock batch starts.
        pre_shared_targets = [pre_linux["vertical_datum_correction"]["offset_raster_path"]]
        pre_stations_cache_dir = f"{pre_linux['paths']['processed_inputs_dir']}/WL_scenarios_cache"
        for rp in return_periods:
            for slr in waterlevel_names:
                pre_shared_targets.append(f"{pre_stations_cache_dir}/stations_{rp}_{slr}.gpkg")
        pre_shared_targets_path = local_jobs_dir / "shared_targets_preprocess.txt"
        with open(pre_shared_targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(pre_shared_targets) + "\n")

        preprocess = {
            "return_periods": return_periods,
            "waterlevel_names": waterlevel_names,
            "linux_model_outputs": pre_linux["simulation"]["model_outputs"],
            "linux_code_root": pre_linux["paths"]["code_root"],
            "linux_resolved_config": f"{linux_jobs_dir}/resolved_config_preprocess.yml",
            "linux_shared_targets_file": f"{linux_jobs_dir}/shared_targets_preprocess.txt",
            "env_activate_cmd": pre_linux["hpc"]["sbatch"]["env_activate_cmd"],
        }

    generate_batches(
        tile_ids=tile_ids, n_nodes=args.n_nodes, partition=args.partition, time_limit=args.time,
        mem=args.mem, cpus_per_task=args.cpus_per_task, account=args.account,
        runner_script_linux=runner_script_linux, linux_jobs_dir=linux_jobs_dir, local_jobs_dir=local_jobs_dir,
        submit_path=local_jobs_dir / submit_filename, batch_name_prefix=batch_name_prefix,
        base_dir_name=args.base_dir_name, runner_extra_args=args.runner_extra_args, preprocess=preprocess,
        defer_eikonal=args.defer_eikonal, sfincs_resolved_config_linux=f"{base_dir_linux}/resolved_config.yml",
        eikonal_deferred_factors=args.defer_eikonal_factors,
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/{submit_filename}")
    print(f"(make sure {args.runner_script_name} is executable / callable via `bash` - no chmod needed since it's invoked as `bash <path>`)")


if __name__ == "__main__":
    main()
