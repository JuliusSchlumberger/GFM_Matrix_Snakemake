"""Write the HPC preprocessing jobs: node batches, then the simulation dispatch; entry point `generate()`.

Tile targets are split into independent `snakemake --nolock` batches, followed by one afterany
job that checks every batch target exists, then generates and submits the simulation waves.
The submit script first builds every shared, tile-independent output (geoid-offset raster,
GOCO06s absolute geoid when a local DEM bridges through it, cached station GeoPackages)
synchronously on the machine that runs it: --nolock is only safe once those exist.
Target paths mirror preprocessing.smk's templates.

fuse: when EVERY tile is hop_distance==0, each batch preprocesses then simulates its own tiles
(no cross-node dependency); writes submit_fused_batches.sh instead.
preprocess_only: no dispatch job; writes submit_preprocess_batches.sh.

Usage:
    python generate_hpc_preprocess_job.py [--config path] [--fuse | --preprocess-only] [--profile-set P.F=V]
    bash <printed submit script>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import (  # noqa: E402
    atomic_write, dem_source_by_tile, dem_source_meta, get_data_catalog, load_config, load_hpc_config,
    merged_waterlevel_names, retry_transient_io,
)
from hpc_common import (  # noqa: E402
    SNAKEMAKE, JobChain, add_profile_set_argument, apply_profile_sets, env_lines, even_split, read_targets_lines,
    resolve_profile, retry_wrapper_lines, stage_configfile_lines,
)
from vertical_datum import HYBRID_GRID_BRIDGES  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_NAMES = {"fuse": "submit_fused_batches.sh", "preprocess_only": "submit_preprocess_batches.sh",
                "dispatch": "submit_preprocess_and_dispatch.sh"}


def _target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    paths = [f"{tile_dir}/inputs/dem.tif", f"{tile_dir}/inputs/mask.tif", f"{tile_dir}/inputs/friction.tif"]
    for rp in return_periods:
        for slr in waterlevel_names:
            paths.append(f"{tile_dir}/inputs/boundaries_{rp}_{slr}.gpkg")
    return paths


def _needs_goco06s_absolute_raster(config: dict) -> bool:
    """Return whether any tile's dem_source bridges through GOCO06s's absolute geoid (as extract_dem)."""
    sources = set(dem_source_by_tile(config).values())
    if not sources:
        return False
    catalog_paths = config["paths"]["hydromt_data_catalog"]
    catalog_paths = [catalog_paths] if isinstance(catalog_paths, str) else catalog_paths
    catalog = get_data_catalog([_REPO_ROOT / p for p in catalog_paths], root=config["paths"]["root"])
    return any(
        dem_source_meta(catalog, source, "native_datum", "egm2008", config["paths"]["root"]) in HYBRID_GRID_BRIDGES
        for source in sources
    )


def _simulate_target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    return [f"{tile_dir}/results/waterdepth_{rp}_{slr}.tif" for rp in return_periods for slr in waterlevel_names]


def _write_lines(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def generate(config_path: str | Path, *, fuse: bool = False, preprocess_only: bool = False,
             profile_sets: list[str] | None = None) -> str:
    """Write the preprocessing batches (+ dispatch job) and the submit script; return its Linux path."""
    if fuse and preprocess_only:
        raise ValueError("fuse and preprocess_only are mutually exclusive")
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    hpc_cfg = apply_profile_sets(linux_config["hpc"], profile_sets)

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = hpc_cfg["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    linux_model_outputs = linux_config["simulation"]["model_outputs"]
    # Preprocessing targets (inputs/ only); see preprocessing.smk.
    linux_preprocessing_inputs_dir = linux_config["simulation"].get("preprocessing_inputs_dir") or linux_model_outputs

    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Referenced via GFM_CONFIG_PATH so compute nodes see this config, not the default
    # config.yml. atomic_write: another node reading over the shared mount right after
    # a non-atomic write can see a partial file (UnicodeDecodeError).
    atomic_write(local_jobs_dir / "resolved_config.yml", lambda f: yaml.safe_dump(linux_config, f),
                 encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    # Local view: linux_config's absolute Linux path only works embedded in job scripts.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    return_periods = [f"RP{rp}" for rp in linux_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_waterlevel_names(linux_config["boundary_conditions"], linux_config["adaptation"])

    if fuse:
        non_wave0 = tile_gdf.loc[tile_gdf["hop_distance"] != 0, "tile_id"].tolist()
        if non_wave0:
            raise ValueError(
                f"fuse requires every tile to be hop_distance==0 (a fused batch simulates right after "
                f"preprocessing, with no cross-batch ordering, so a hop>=1 tile could run before its "
                f"neighbour) - found {len(non_wave0)} tile(s) with hop_distance!=0: {non_wave0[:20]}"
                f"{'...' if len(non_wave0) > 20 else ''}. Use the two-phase dispatch instead."
            )

    # Shared, non-tile outputs: must be built before any --nolock batch starts.
    linux_shared_targets = [linux_config["vertical_datum_correction"]["offset_raster_path"]]
    if _needs_goco06s_absolute_raster(local_config):
        linux_shared_targets.append(linux_config["vertical_datum_correction"]["goco06s_absolute_raster_path"])
    linux_bc = linux_config["boundary_conditions"]
    # As preprocessing.smk: the scenario's own stations_cache_dir, nothing to build when prebuilt.
    linux_stations_cache_dir = (linux_bc.get("stations_cache_dir")
                                or f"{linux_config['paths']['processed_inputs_dir']}/WL_scenarios_cache")
    if not linux_bc["stations_cache_prebuilt"]:
        for rp in return_periods:
            for slr in waterlevel_names:
                linux_shared_targets.append(f"{linux_stations_cache_dir}/stations_{rp}_{slr}.gpkg")
    _write_lines(local_jobs_dir / "shared_targets.txt", linux_shared_targets)
    linux_shared_targets_file = f"{linux_jobs_dir}/shared_targets.txt"

    all_tiles = sorted(tile_gdf["tile_id"].astype(int).astype(str).tolist(), key=int)
    # hpc.single_preprocess_job: one job for all tiles; at calibration scale
    # SLURM queue wait (hours) dominates minutes of compute.
    n_batches = 1 if hpc_cfg.get("single_preprocess_job", False) else hpc_cfg["n_nodes"]
    batches = [(f"{i:03d}", tiles) for i, tiles in enumerate(even_split(all_tiles, n_batches))]

    kind = "fuse" if fuse else "preprocess_only" if preprocess_only else "dispatch"
    submit_script_name = SUBMIT_NAMES[kind]
    cores = resolve_profile(hpc_cfg, "preprocess")["cpus_per_task"]
    snakemake_batch = (f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry {SNAKEMAKE} --cores {cores} '
                       '--nolock --rerun-triggers=mtime --rerun-incomplete')

    chain = JobChain(hpc_cfg, local_jobs_dir, linux_jobs_dir)
    for batch_id, batch_tiles in batches:
        name = f"preprocess_batch_{batch_id}"
        targets = [
            p for tile_id in batch_tiles
            for p in _target_paths(f"{linux_preprocessing_inputs_dir}/{tile_id}", return_periods, waterlevel_names)
        ]
        # Targets in a file read via mapfile (read_targets_lines), keeping the script short.
        _write_lines(local_jobs_dir / f"{name}_targets.txt", targets)
        body = [
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
            f'{snakemake_batch} "${{TARGETS[@]}}"',
        ]
        if fuse:
            simulate_targets = [
                p for tile_id in batch_tiles
                for p in _simulate_target_paths(f"{linux_model_outputs}/{tile_id}", return_periods, waterlevel_names)
            ]
            _write_lines(local_jobs_dir / f"{name}_simulate_targets.txt", simulate_targets)
            body += [
                "",
                f'echo "=== Simulating batch {batch_id}: {len(batch_tiles)} tiles ==="',
                *read_targets_lines("SIMULATE_TARGETS", f"{linux_jobs_dir}/{name}_simulate_targets.txt"),
                f'{snakemake_batch} "${{SIMULATE_TARGETS[@]}}"',
            ]
        chain.add(name, "preprocess", body, helpers=("retry", "stage_config"))
        print(f"  wrote {name} ({len(batch_tiles)} tiles, {len(targets)} target files)")

    stale_dispatch = local_jobs_dir / "generate_jobs_and_dispatch.sbatch"
    if kind == "dispatch":
        # The preprocess batches end afterany: a failed batch must stop the chain here, not
        # surface as missing-input failures in every simulate job.
        target_files = " ".join(f'"{linux_jobs_dir}/{name}_targets.txt"' for name in chain.names())
        chain.add("gfm_generate_jobs_and_dispatch", "dispatch", [
            f'cd "{linux_code_root}"',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            'echo "=== Checking preprocessing outputs ==="',
            f'"$GFM_PY" hpc/gfm_hpc.py check-targets {target_files} || exit 1',
            'echo "=== Generating wave sbatch scripts ==="',
            '"$GFM_PY" hpc/gfm_hpc.py run simulate --config "$LOCAL_CONFIGFILE"',
            'echo "=== Submitting simulation waves ==="',
            f'bash "{linux_jobs_dir}/submit_waves.sh"',
        ], after=chain.names(), filename=stale_dispatch.name, helpers=("stage_config",))
    elif stale_dispatch.exists():
        # No dispatch job; remove a stale one so it cannot be submitted by mistake.
        stale_dispatch.unlink()
        print(f"  removed stale {stale_dispatch}")

    # Shared inputs are built synchronously by the submit script, before any batch is queued.
    prelude = [
        *env_lines(hpc_cfg),
        "",
        *retry_wrapper_lines(),
        *stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== Building shared preprocessing inputs (synchronous, on this login node) ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        *read_targets_lines("SHARED_TARGETS", linux_shared_targets_file),
        (
            f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry {SNAKEMAKE} --cores 1 --nolock '
            f'--rerun-triggers=mtime "${{SHARED_TARGETS[@]}}" 2>&1 | tee "{linux_jobs_dir}/logs/build_shared_inputs.log"'
        ),
        "",
    ]
    submit = chain.write_submit(submit_script_name, prelude)
    # Only one kind of submit script should exist per jobs_dir.
    for other_name in set(SUBMIT_NAMES.values()) - {submit_script_name}:
        stale_submit = local_jobs_dir / other_name
        if stale_submit.exists():
            stale_submit.unlink()
            print(f"  removed stale {stale_submit}")

    print(f"Done: shared-inputs build + {len(batches)} {kind} batch(es)"
          f"{' + dispatch job' if kind == 'dispatch' else ''} -> {local_jobs_dir}")
    return submit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fuse", action="store_true",
                      help="every tile hop_distance==0: each batch preprocesses then simulates its own tiles")
    mode.add_argument("--preprocess-only", action="store_true",
                      help="no simulation dispatch (e.g. inputs for the SFINCS calibration)")
    add_profile_set_argument(parser)
    args = parser.parse_args()
    submit = generate(args.config, fuse=args.fuse, preprocess_only=args.preprocess_only, profile_sets=args.profile_set)
    print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
