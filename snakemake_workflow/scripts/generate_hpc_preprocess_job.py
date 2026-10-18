"""Generate HPC sbatch scripts for preprocessing across hpc.n_nodes parallel nodes.

Tile targets are split into independent `snakemake --nolock` batches, followed by
one afterany job that generates and submits the simulation waves.
submit_preprocess_and_dispatch.sh first builds every shared, tile-independent
output (geoid-offset raster, cached station GeoPackages) synchronously on the
machine that wrote resolved_config.yml: --nolock is only safe once those exist.
Target paths mirror preprocessing.smk's templates.

--calibration --fuse: when EVERY tile is hop_distance==0, each batch preprocesses
then simulates its own tiles (no cross-node dependency); writes
submit_fused_batches.sh instead.

Usage:
    python generate_hpc_preprocess_job.py [--config path/to/config.yml]
    python generate_hpc_preprocess_job.py --config ... --calibration --fuse
    bash <printed submit_preprocess_and_dispatch.sh / submit_fused_batches.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, load_hpc_config, merged_waterlevel_names, retry_transient_io  # noqa: E402
from hpc_common import _retry_wrapper_lines, _stage_configfile_lines, read_targets_lines, sbatch_header  # noqa: E402


def _target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    paths = [f"{tile_dir}/inputs/dem.tif", f"{tile_dir}/inputs/mask.tif", f"{tile_dir}/inputs/friction.tif"]
    for rp in return_periods:
        for slr in waterlevel_names:
            paths.append(f"{tile_dir}/inputs/boundaries_{rp}_{slr}.gpkg")
    return paths


def _simulate_target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    return [f"{tile_dir}/results/waterdepth_{rp}_{slr}.tif" for rp in return_periods for slr in waterlevel_names]


def main() -> None:
    """Generate (and print how to submit) the HPC preprocessing dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--calibration", action="store_true",
        help="use generate_hpc_simulation_jobs.py directly for the post-preprocessing "
             "wave-dispatch phase instead of `snakemake generate_eikonal_jobs` - skips "
             "Snakemake's own full-DAG preprocessing-output verification (safe here: this "
             "phase only runs after submit_preprocess_and_dispatch.sh's own afterany chain "
             "already guarantees every preprocessing batch above finished), and - unlike "
             "`rule generate_eikonal_jobs`, whose own base_config_path is a hardcoded "
             "literal path to production config.yml regardless of --configfile (see "
             "hpc_dispatch.smk) - correctly honors a scenario --config end to end. "
             "Appropriate at calibration-subset scale, where the full-DAG build's "
             "multi-hour cost at production scale buys nothing anyway.",
    )
    parser.add_argument(
        "--fuse", action="store_true",
        help="requires --calibration. Collapses preprocessing and simulate into ONE "
             "sbatch script per batch (preprocess this batch's tiles, then immediately "
             "simulate the same tiles, sequentially on the same node) instead of the "
             "normal two-phase design (every preprocessing batch across every node, "
             "THEN one dispatch job that generates and submits simulate waves). Valid "
             "ONLY when every tile in the run is hop_distance==0 (asserted below) - "
             "the two-phase design exists purely to let hop>=1 simulate wait for its "
             "lower-hop neighbours' own simulate output, which can land on a DIFFERENT "
             "node's batch; a hop=0-only tile's simulate step has no such cross-tile "
             "dependency at all (only its own preprocessing), so there is nothing for "
             "the dispatch phase's cross-node synchronization to protect here - it was "
             "pure added wait, inherited from the general multi-wave machinery rather "
             "than being a real requirement for a single-wave study like this one.",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="skip phase 2 (generate+dispatch simulate waves) entirely - no "
             "`generate_eikonal_jobs`/`generate_hpc_simulation_jobs.py` call, no dispatch "
             "sbatch, no afterany chain. For call sites that only need preprocessing "
             "outputs (dem/mask/friction/boundaries) and never intend to run the eikonal "
             "flood solve at all - e.g. building shared model_outputs/ inputs for a "
             "downstream pipeline (SFINCS) that reads them directly and has its own, "
             "separate run/dispatch mechanism. Mutually exclusive with --fuse (nothing to "
             "fuse simulate into). Writes submit_preprocess_batches.sh instead of "
             "submit_preprocess_and_dispatch.sh.",
    )
    args = parser.parse_args()
    if args.fuse and not args.calibration:
        parser.error("--fuse requires --calibration")
    if args.fuse and args.preprocess_only:
        parser.error("--fuse and --preprocess-only are mutually exclusive")

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    linux_model_outputs = linux_config["simulation"]["model_outputs"]
    # Preprocessing targets (inputs/ only); see preprocessing.smk.
    linux_preprocessing_inputs_dir = linux_config["simulation"].get("preprocessing_inputs_dir") or linux_model_outputs
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Referenced via GFM_CONFIG_PATH so compute nodes see this --config, not the
    # default config.yml. atomic_write: another node reading over P: right after
    # a non-atomic write can see a partial file (UnicodeDecodeError).
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    # Local view: linux_config's "/p/..." path only works embedded in sbatch
    # scripts, not opened from Windows.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    return_periods = [f"RP{rp}" for rp in linux_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_waterlevel_names(linux_config["boundary_conditions"], linux_config["adaptation"])

    if args.fuse:
        non_wave0 = tile_gdf.loc[tile_gdf["hop_distance"] != 0, "tile_id"].tolist()
        if non_wave0:
            raise ValueError(
                f"--fuse requires every tile to be hop_distance==0 (a fused batch runs "
                f"simulate right after preprocessing, with no cross-batch ordering, so a "
                f"hop>=1 tile could run before its neighbour) - found {len(non_wave0)} "
                f"tile(s) with hop_distance!=0: {non_wave0[:20]}"
                f"{'...' if len(non_wave0) > 20 else ''}. Drop --fuse and use the normal "
                f"two-phase --calibration dispatch instead."
            )

    # Shared, non-tile outputs: must be built before any --nolock batch starts.
    linux_shared_targets = [linux_config["vertical_datum_correction"]["offset_raster_path"]]
    linux_stations_cache_dir = f"{linux_config['paths']['processed_inputs_dir']}/WL_scenarios_cache"
    for rp in return_periods:
        for slr in waterlevel_names:
            linux_shared_targets.append(f"{linux_stations_cache_dir}/stations_{rp}_{slr}.gpkg")

    shared_targets_path = local_jobs_dir / "shared_targets.txt"
    with open(shared_targets_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(linux_shared_targets) + "\n")
    linux_shared_targets_file = f"{linux_jobs_dir}/shared_targets.txt"

    all_tiles = sorted(tile_gdf["tile_id"].astype(int).astype(str).tolist(), key=int)

    # hpc.single_preprocess_job: one job for all tiles; at calibration scale
    # SLURM queue wait (hours) dominates minutes of compute.
    if hpc_cfg.get("single_preprocess_job", False):
        n_batches = 1
        batches = [("000", all_tiles)]
    else:
        n_batches = min(n_nodes, len(all_tiles))
        k, m = divmod(len(all_tiles), n_batches)
        batches = [  # [(batch_id, [tile_id, ...]), ...]
            (f"{i:03d}", all_tiles[i * k + min(i, m): (i + 1) * k + min(i + 1, m)])
            for i in range(n_batches)
        ]

    if args.fuse:
        submit_script_name = "submit_fused_batches.sh"
    elif args.preprocess_only:
        submit_script_name = "submit_preprocess_batches.sh"
    else:
        submit_script_name = "submit_preprocess_and_dispatch.sh"

    batch_script_paths = []
    for batch_id, batch_tiles in batches:
        sbatch_cfg = hpc_cfg["sbatch"]
        name = f"preprocess_batch_{batch_id}"
        targets = [
            p for tile_id in batch_tiles
            for p in _target_paths(f"{linux_preprocessing_inputs_dir}/{tile_id}", return_periods, waterlevel_names)
        ]

        # Targets in a file read via mapfile (read_targets_lines), keeping the script short.
        targets_path = local_jobs_dir / f"{name}_targets.txt"
        with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(targets) + "\n")

        # --fuse: this batch's simulate targets (all tiles hop_distance==0).
        if args.fuse:
            simulate_targets = [
                p for tile_id in batch_tiles
                for p in _simulate_target_paths(f"{linux_model_outputs}/{tile_id}", return_periods, waterlevel_names)
            ]
            simulate_targets_path = local_jobs_dir / f"{name}_simulate_targets.txt"
            with open(simulate_targets_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(simulate_targets) + "\n")

        lines = [
            *sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs"),
            *_retry_wrapper_lines(),
            *_stage_configfile_lines(),
            "",
            f'cd "{linux_code_root}"',
            f'echo "=== Preprocessing batch {batch_id}: {len(batch_tiles)} tiles ==="',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            "",
            # Pre-flight: --nolock is only safe if the shared build ran first; a
            # hand-submitted batch would otherwise race it (LockException or a
            # corrupted shared file).
            'while IFS= read -r shared_target; do',
            '    if [ ! -f "$shared_target" ]; then',
            '        echo "ERROR: shared preprocessing input not found: $shared_target" >&2',
            f'        echo "This batch must not start before the shared-inputs build (in'
            f' {submit_script_name}) completes." >&2',
            f'        echo "Submit via {submit_script_name} (which orders this'
            f' correctly) rather than running this .sbatch file directly/out of order." >&2',
            "        exit 1",
            "    fi",
            f'done < "{linux_shared_targets_file}"',
            "",
            *read_targets_lines("TARGETS", f"{linux_jobs_dir}/{name}_targets.txt"),
            (
                f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                '--rerun-triggers=mtime --rerun-incomplete '
                '"${TARGETS[@]}"'
            ),
            "",
        ]
        if args.fuse:
            lines += [
                f'echo "=== Simulating batch {batch_id}: {len(batch_tiles)} tiles ==="',
                *read_targets_lines("SIMULATE_TARGETS", f"{linux_jobs_dir}/{name}_simulate_targets.txt"),
                (
                    f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                    f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                    '--rerun-triggers=mtime --rerun-incomplete '
                    '"${SIMULATE_TARGETS[@]}"'
                ),
                "",
            ]
        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        batch_script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tiles)} tiles, {len(targets)} target files)")

    # Phase 2 (not under --fuse): generate and submit the wave scripts.
    # --calibration uses generate_hpc_simulation_jobs.py because
    # `snakemake generate_eikonal_jobs` builds resolved_config.yml from a
    # hardcoded production config.yml path, ignoring GFM_CONFIG_PATH.
    dispatch_script_path = None
    if not args.fuse and not args.preprocess_only:
        if args.calibration:
            generate_call = (
                'python snakemake_workflow/scripts/generate_hpc_simulation_jobs.py '
                '--config "$LOCAL_CONFIGFILE"'
            )
        else:
            generate_call = (
                'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                'snakemake generate_eikonal_jobs --cores 1 --nolock '
                '--rerun-triggers=mtime'
            )

        dispatch_cfg = hpc_cfg["sbatch"]
        dispatch_lines = [
            *sbatch_header(
                "gfm_generate_jobs_and_dispatch", dispatch_cfg, f"{linux_jobs_dir}/logs",
                cpus_per_task=1, log_name="generate_jobs_and_dispatch",
            ),
            *_retry_wrapper_lines(),  # only used by the non-calibration snakemake branch below; harmless if unused
            *_stage_configfile_lines(),
            "",
            f'cd "{linux_code_root}"',
            'echo "=== Generating wave sbatch scripts ==="',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            generate_call,
            "",
            'echo "=== Submitting simulation waves ==="',
            f'bash "{linux_jobs_dir}/submit_waves.sh"',
            "",
        ]
        dispatch_script_path = local_jobs_dir / "generate_jobs_and_dispatch.sbatch"
        with open(dispatch_script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(dispatch_lines))
        print(f"  wrote {dispatch_script_path}")

    # Build shared inputs synchronously here (same machine that wrote
    # resolved_config.yml), then submit batches, then the afterany dispatch job.
    shared_cfg = hpc_cfg["sbatch"]
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        shared_cfg["env_activate_cmd"],
        "",
        *_retry_wrapper_lines(),
        *_stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== Building shared preprocessing inputs (synchronous, on this login node) ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        *read_targets_lines("SHARED_TARGETS", linux_shared_targets_file),
        (
            f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
            'snakemake --cores 1 --nolock --rerun-triggers=mtime '
            f'"${{SHARED_TARGETS[@]}}" 2>&1 | tee "{linux_jobs_dir}/logs/build_shared_inputs.log"'
        ),
        "",
        'IDS=""',
    ]
    for script in batch_script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
            'IDS="${IDS:+$IDS:}$JID"',
        ]
    if args.fuse or args.preprocess_only:
        # No dispatch job; remove a stale one so it cannot be submitted by mistake.
        stale_dispatch = local_jobs_dir / "generate_jobs_and_dispatch.sbatch"
        if stale_dispatch.exists():
            stale_dispatch.unlink()
            print(f"  removed stale {stale_dispatch} (leftover from a prior two-phase generation)")
    else:
        submit_lines += [
            "",
            f'JID=$(sbatch --parsable --dependency=afterany:$IDS "{linux_jobs_dir}/generate_jobs_and_dispatch.sbatch")',
            (
                f'echo "submitted {linux_jobs_dir}/generate_jobs_and_dispatch.sbatch -> job $JID '
                f'(depends on all {len(batch_script_paths)} preprocessing batches)"'
            ),
        ]
    submit_script_path = local_jobs_dir / submit_script_name
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")
    # Only one kind of submit script should exist per jobs_dir.
    all_submit_names = {"submit_fused_batches.sh", "submit_preprocess_batches.sh", "submit_preprocess_and_dispatch.sh"}
    for other_name in all_submit_names - {submit_script_name}:
        stale_submit = local_jobs_dir / other_name
        if stale_submit.exists():
            stale_submit.unlink()
            print(f"  removed stale {stale_submit} (leftover from a prior generation of a different kind)")

    if hpc_cfg.get("single_preprocess_job", False):
        node_summary = "single 1vcpu job, no batching - hpc.single_preprocess_job"
    else:
        node_summary = f"{n_batches} nodes, {n_nodes} total budget"
    if args.fuse:
        print(
            f"\nDone. 1 synchronous shared-inputs build + {len(batch_script_paths)} fused "
            f"preprocess+simulate batch(es) ({node_summary}) written to {local_jobs_dir}"
        )
    elif args.preprocess_only:
        print(
            f"\nDone. 1 synchronous shared-inputs build + {len(batch_script_paths)} preprocessing-only "
            f"batch(es) ({node_summary}) written to {local_jobs_dir} - no simulate dispatch generated."
        )
    else:
        print(
            f"\nDone. 1 synchronous shared-inputs build + {len(batch_script_paths)} preprocessing batch(es) "
            f"({node_summary}) + 1 dispatch job written to {local_jobs_dir}"
        )
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/{submit_script_name}")


if __name__ == "__main__":
    main()
