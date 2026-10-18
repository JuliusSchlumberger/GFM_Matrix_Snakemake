"""Generate the simulation-wave sbatch scripts directly from the tile grid, without a DAG check.

Computes the same (wave, batch_id, tile_ids) partition as hpc_dispatch.smk and calls
the same generate_wave_dispatch(), skipping rule generate_eikonal_jobs' slow
_PREPROCESS_OUTPUTS verification. Does NOT verify preprocessing is complete.
The tile grid is read via the local config; embedded paths use the Linux view.

--resume re-batches only missing (tile, rp, slr) items evenly across hpc.n_nodes
into resume_wave*.sbatch / submit_resume_waves.sh. Cancel the original dispatch's
still-running jobs first: _output_already_done does not cover in-flight work.

Usage:
    python generate_hpc_simulation_jobs.py [--config path/to/config.yml]
    python generate_hpc_simulation_jobs.py --resume [--config path/to/config.yml]
    bash <printed submit_waves.sh / submit_resume_waves.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import load_config, merged_waterlevel_names, retry_transient_io  # noqa: E402
from generate_eikonal_jobs import generate_resume_dispatch, generate_wave_dispatch  # noqa: E402


def _resolve_wave_tier(hop: int, wave_tiers: list[dict]) -> dict:
    """Return the first hpc.wave_tiers entry whose `hops` contains `hop`.

    `hops: null` is a catch-all and must be last.
    """
    for tier in wave_tiers:
        hops = tier.get("hops")
        if hops is None or hop in hops:
            return tier
    raise ValueError(f"no hpc.wave_tiers entry matches hop_distance={hop} - add a catch-all 'hops: null' tier")


def main() -> None:
    """Write the sbatch scripts for a fresh or --resume simulation wave dispatch."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--resume", action="store_true",
        help="rebalance only the still-missing (tile, rp, slr) work across the full node "
             "budget, instead of a fresh whole-tile dispatch - see module docstring. "
             "Cancel the original dispatch's still-running jobs FIRST.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]
    model_outputs = Path(local_config["simulation"]["model_outputs"])

    # Local view: linux_config's path is only for embedding in sbatch scripts.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    tile_ids = tile_gdf["tile_id"].astype(int).tolist()
    hop_by_tile = {str(tid): int(hop) for tid, hop in zip(tile_gdf["tile_id"], tile_gdf["hop_distance"])}
    return_periods = [f"RP{rp}" for rp in local_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_waterlevel_names(local_config["boundary_conditions"], local_config["adaptation"])

    hpc_waves: dict[int, list[str]] = {}
    for tid in tile_ids:
        hpc_waves.setdefault(hop_by_tile[str(tid)], []).append(str(tid))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    if not args.resume:
        wave_tiers = hpc_cfg.get("wave_tiers")
        wave_sbatch_cfg: dict[int, dict] = {}

        batches = []  # [(wave, batch_id, [tile_id, ...]), ...]
        for wave in sorted(hpc_waves):
            wave_tiles = hpc_waves[wave]
            if wave_tiers:
                # See config.yml's comment on hpc.wave_tiers.
                tier = _resolve_wave_tier(wave, wave_tiers)
                wave_sbatch_cfg[wave] = {
                    "partition": tier["partition"], "cpus_per_task": tier["cpus_per_task"],
                    "mem": tier["mem"], "time": tier["time"],
                    "account": hpc_cfg["sbatch"].get("account", ""),
                    "env_activate_cmd": hpc_cfg["sbatch"]["env_activate_cmd"],
                }
                n_batches = len(wave_tiles) if tier["n_nodes"] is None else min(tier["n_nodes"], len(wave_tiles))
            else:
                n_batches = min(n_nodes, len(wave_tiles))

            k, m = divmod(len(wave_tiles), n_batches)
            for i in range(n_batches):
                batch_tiles = wave_tiles[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
                batches.append((wave, f"{i:03d}", batch_tiles))

        script_paths = [
            str(local_jobs_dir / f"wave{wave}_batch_{batch_id}.sbatch")
            for wave, batch_id, _ in batches
        ]

        n_by_wave: dict[int, int] = {}
        for wave, _, _ in batches:
            n_by_wave[wave] = n_by_wave.get(wave, 0) + 1
        if wave_tiers:
            print(f"{len(tile_ids)} tiles, {len(hpc_waves)} wave(s), hpc.wave_tiers active:")
            for wave in sorted(n_by_wave):
                tier = _resolve_wave_tier(wave, wave_tiers)
                print(f"  wave {wave}: {n_by_wave[wave]} batch(es) (tier: {tier['partition']}, "
                      f"{tier['cpus_per_task']}-way concurrent, n_nodes={tier['n_nodes']})")
        else:
            print(f"{len(tile_ids)} tiles, {len(hpc_waves)} wave(s), n_nodes={n_nodes}:")
            for wave in sorted(n_by_wave):
                print(f"  wave {wave}: {n_by_wave[wave]} batch(es)")
        print()

        generate_wave_dispatch(
            batches=batches,
            return_periods=return_periods,
            waterlevel_names=waterlevel_names,
            hpc_cfg=hpc_cfg,
            base_config_path=config_path,
            script_paths=script_paths,
            resolved_config_path=str(local_jobs_dir / "resolved_config.yml"),
            submit_waves_path=str(local_jobs_dir / "submit_waves.sh"),
            wave_sbatch_cfg=wave_sbatch_cfg or None,
        )
        print(f"\nSubmit on Hydrax with: bash {linux_config['hpc']['jobs_dir']}/submit_waves.sh")
        return

    wave_tiers = hpc_cfg.get("wave_tiers")
    wave_sbatch_cfg: dict[int, dict] = {}

    print("Scanning existing results/ dirs for already-completed scenarios…")
    all_scenario_names = {f"{rp}_{slr}" for rp in return_periods for slr in waterlevel_names}
    remaining_by_wave: dict[int, list[tuple[str, str, str]]] = {}
    n_done_total = 0
    for tid in tile_ids:
        tid_s = str(tid)
        wave = hop_by_tile[tid_s]
        results_dir = model_outputs / tid_s / "results"
        done = set()
        if results_dir.is_dir():
            for p in results_dir.glob("waterdepth_*.tif"):
                done.add(p.stem[len("waterdepth_"):])
        n_done_total += len(done & all_scenario_names)
        missing = all_scenario_names - done
        if not missing:
            continue
        remaining_by_wave.setdefault(wave, [])
        for scenario_name in missing:
            rp, slr = scenario_name.split("_", 1)
            remaining_by_wave[wave].append((tid_s, rp, slr))

    n_total = len(tile_ids) * len(all_scenario_names)
    n_remaining = n_total - n_done_total
    print(f"{n_done_total}/{n_total} scenario-jobs already done, {n_remaining} remaining.\n")

    batches = []  # [(wave, batch_id, [(tile_id, rp, slr), ...]), ...]
    for wave in sorted(hpc_waves):
        items = remaining_by_wave.get(wave, [])
        if not items:
            continue
        if wave_tiers:
            tier = _resolve_wave_tier(wave, wave_tiers)
            wave_sbatch_cfg[wave] = {
                "partition": tier["partition"], "cpus_per_task": tier["cpus_per_task"],
                "mem": tier["mem"], "time": tier["time"],
                "account": hpc_cfg["sbatch"].get("account", ""),
                "env_activate_cmd": hpc_cfg["sbatch"]["env_activate_cmd"],
            }
            if tier["n_nodes"] is None:
                # Uncapped tier means one node per TILE: group the flat triples
                # by tile_id, else every missing triple becomes its own job.
                items_by_tile: dict[str, list[tuple[str, str, str]]] = {}
                for tile_id, rp, slr in items:
                    items_by_tile.setdefault(tile_id, []).append((tile_id, rp, slr))
                for i, tile_items in enumerate(items_by_tile.values()):
                    batches.append((wave, f"{i:03d}", tile_items))
                continue
            n_batches = min(tier["n_nodes"], len(items))
        else:
            n_batches = min(n_nodes, len(items))
        k, m = divmod(len(items), n_batches)
        for i in range(n_batches):
            batch_items = items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
            if batch_items:
                batches.append((wave, f"{i:03d}", batch_items))

    script_paths = [
        str(local_jobs_dir / f"resume_wave{wave}_batch_{batch_id}.sbatch")
        for wave, batch_id, _ in batches
    ]

    n_by_wave: dict[int, int] = {}
    for wave, _, items in batches:
        n_by_wave[wave] = n_by_wave.get(wave, 0) + len(items)
    print(f"n_nodes={n_nodes}, remaining work by wave:")
    for wave in sorted(n_by_wave):
        print(f"  wave {wave}: {n_by_wave[wave]} job(s) across "
              f"{sum(1 for w, _, _ in batches if w == wave)} batch(es)")
    print()

    generate_resume_dispatch(
        hpc_cfg=hpc_cfg,
        base_config_path=config_path,
        batches=batches,
        script_paths=script_paths,
        resolved_config_path=str(local_jobs_dir / "resolved_config.yml"),
        submit_waves_path=str(local_jobs_dir / "submit_resume_waves.sh"),
        wave_sbatch_cfg=wave_sbatch_cfg or None,
    )
    print("\nCancel the original dispatch's still-running jobs for these wave(s) FIRST (see module docstring), then:")
    print(f"Submit on Hydrax with: bash {linux_config['hpc']['jobs_dir']}/submit_resume_waves.sh")


if __name__ == "__main__":
    main()
