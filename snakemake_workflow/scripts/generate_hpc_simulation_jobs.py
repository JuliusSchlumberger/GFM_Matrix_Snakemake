"""Generate the simulation-wave sbatch scripts directly from the tile grid,
with NO Snakemake DAG/dependency check at all - the simulation-side
counterpart to generate_hpc_preprocess_job.py.

`rule generate_aqueduct_jobs` (hpc_dispatch.smk) is gated on `input:
_PREPROCESS_OUTPUTS` - every preprocessing output across the WHOLE domain
(~2578 tiles x ~45 scenarios each) - so Snakemake has to verify that entire
dependency list before this rule can even start, which is genuinely slow
over a network mount (confirmed live 2026-08-10: several hours for this
exact DAG build on a real run). That verification is real and correct the
first time - but once preprocessing is confirmed done (e.g. via
check_preprocess_progress.py), REGENERATING the wave scripts afterwards
(e.g. after an hpc.n_nodes change, like the 2026-08-10 fix reducing peak
concurrent nodes from 40 to 20) does not need to re-pay that cost every
time. This script computes the exact same (wave, size_class, batch_id,
tile_ids) partition hpc_dispatch.smk does, directly from the tile grid, and
calls the SAME generate_wave_dispatch() function
scripts/generate_aqueduct_jobs.py's Snakemake path uses - one implementation,
two entry points, so they can't drift apart.

This does NOT verify preprocessing is actually complete (unlike the
Snakemake path) - that's the caller's own responsibility. It also does NOT
run the wave-0-boundaries-empty pre-check any differently - that still
happens here too (see generate_wave_dispatch's own docstring), just without
Snakemake's own dependency-freshness check wrapped around the whole thing.

Uses the same local-view/Linux-view path resolution as
generate_hpc_preprocess_job.py (config_hpc.yml, if present): the tile grid
is read via the LOCAL config (this machine's own reachable mount), while
every path string embedded into generated sbatch scripts uses the Linux
view - see generate_hpc_preprocess_job.py's own note on why the tile-grid
read specifically must use the local view.

--resume (2026-08, new): instead of a fresh whole-tile dispatch, scans every
tile's results/ dir once to find which (rp, slr) scenarios are still
missing, then re-batches ONLY that remaining work - flat (tile, rp, slr)
items, not whole tiles - evenly across the full hpc.n_nodes budget again
(scripts/generate_aqueduct_jobs.py's generate_resume_dispatch). This is for
rebalancing a wave whose ORIGINAL batches finished their own (uneven) tile
lists at different times and went idle one by one, silently shrinking
effective parallelism (found live 2026-08-11: 5 of wave 0's 17 small
batches had already finished while the wave overall was still <40% done).
Writes resume_wave*.sbatch / submit_resume_waves.sh - separate filenames
from a fresh dispatch's wave*.sbatch / submit_waves.sh, so neither
overwrites the other.

**Before submitting a --resume dispatch, cancel whatever's still running
from the ORIGINAL dispatch for the same wave(s) first** (e.g. `scancel
<jobids>` from `squeue`). The remaining-work scan is a snapshot; if the old
batches keep running alongside newly-submitted resume batches, both can
pick up the same still-incomplete item and redo it concurrently for as long
as both stay alive - run_aqueduct_cli.py's own idempotency check
(_output_already_done) only prevents redoing work that has ALREADY finished
and been written, not work still in flight in another job.

Usage:
    python generate_hpc_simulation_jobs.py [--config path/to/config.yml]
    python generate_hpc_simulation_jobs.py --resume [--config path/to/config.yml]
    bash <printed submit_waves.sh / submit_resume_waves.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, merged_slr_scenarios, retry_transient_io, split_batches_proportionally  # noqa: E402
from generate_aqueduct_jobs import generate_resume_dispatch, generate_wave_dispatch  # noqa: E402


def main() -> None:
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
    large_pixel_threshold = hpc_cfg["large_tile_pixel_threshold"]
    model_outputs = Path(local_config["simulation"]["model_outputs"])

    # Local view (this machine's own reachable mount), not linux_config's -
    # see generate_hpc_preprocess_job.py's own note on this exact point.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    tile_ids = tile_gdf["tile_id"].astype(int).tolist()
    hop_by_tile = {str(tid): int(hop) for tid, hop in zip(tile_gdf["tile_id"], tile_gdf["hop_distance"])}
    return_periods = [f"RP{rp}" for rp in local_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(local_config["boundary_conditions"], local_config["adaptation"])

    # Same bbox-area pixel-count proxy hpc_dispatch.smk uses.
    bounds = tile_gdf.geometry.bounds
    approx_pixels_by_tile = {
        str(tid): float(maxx - minx) * float(maxy - miny) * 3600.0 * 3600.0
        for tid, minx, miny, maxx, maxy in zip(
            tile_gdf["tile_id"], bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"],
        )
    }

    hpc_waves: dict[int, list[str]] = {}
    for tid in tile_ids:
        hpc_waves.setdefault(hop_by_tile[str(tid)], []).append(str(tid))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    if not args.resume:
        # Same proportional node-budget split as hpc_dispatch.smk (2026-08 fix -
        # see split_batches_proportionally's own docstring for why this matters:
        # without it, a wave with both size classes present could claim up to
        # 2x n_nodes at once).
        batches = []  # [(wave, size_class, batch_id, [tile_id, ...]), ...]
        for wave in sorted(hpc_waves):
            wave_tiles = hpc_waves[wave]
            tiles_by_class = {"small": [], "large": []}
            for t in wave_tiles:
                is_large = approx_pixels_by_tile[t] >= large_pixel_threshold
                tiles_by_class["large" if is_large else "small"].append(t)
            class_n_nodes = split_batches_proportionally(
                {c: len(ts) for c, ts in tiles_by_class.items()}, n_nodes,
            )
            for size_class, class_tiles in tiles_by_class.items():
                if not class_tiles:
                    continue
                n_batches = class_n_nodes[size_class]
                k, m = divmod(len(class_tiles), n_batches)
                for i in range(n_batches):
                    batch_tiles = class_tiles[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
                    batches.append((wave, size_class, f"{i:03d}", batch_tiles))

        script_paths = [
            str(local_jobs_dir / f"wave{wave}_{size_class}_batch_{batch_id}.sbatch")
            for wave, size_class, batch_id, _ in batches
        ]

        n_by_wave: dict[int, int] = {}
        for wave, _, _, _ in batches:
            n_by_wave[wave] = n_by_wave.get(wave, 0) + 1
        print(f"{len(tile_ids)} tiles, {len(hpc_waves)} wave(s), n_nodes={n_nodes}:")
        for wave in sorted(n_by_wave):
            print(f"  wave {wave}: {n_by_wave[wave]} batch(es)")
        print()

        generate_wave_dispatch(
            model_outputs=str(model_outputs),
            tile_ids=tile_ids,
            hop_by_tile=hop_by_tile,
            batches=batches,
            return_periods=return_periods,
            waterlevel_names=waterlevel_names,
            raster_config=local_config["raster_format"],
            hpc_cfg=hpc_cfg,
            base_config_path=config_path,
            script_paths=script_paths,
            resolved_config_path=str(local_jobs_dir / "resolved_config.yml"),
            submit_waves_path=str(local_jobs_dir / "submit_waves.sh"),
        )
        print(f"\nSubmit on Hydrax with: bash {linux_config['hpc']['jobs_dir']}/submit_waves.sh")
        return

    # ── --resume: scan what's actually missing, rebalance across all nodes ──
    print("Scanning existing results/ dirs for already-completed scenarios…")
    all_scenario_names = {f"{rp}_{slr}" for rp in return_periods for slr in waterlevel_names}
    remaining_by_wave_class: dict[tuple[int, str], list[tuple[str, str, str]]] = {}
    n_done_total = 0
    for tid in tile_ids:
        tid_s = str(tid)
        wave = hop_by_tile[tid_s]
        size_class = "large" if approx_pixels_by_tile[tid_s] >= large_pixel_threshold else "small"
        results_dir = model_outputs / tid_s / "results"
        done = set()
        if results_dir.is_dir():
            for p in results_dir.glob("waterdepth_*.tif"):
                done.add(p.stem[len("waterdepth_"):])
        n_done_total += len(done & all_scenario_names)
        missing = all_scenario_names - done
        if not missing:
            continue
        key = (wave, size_class)
        remaining_by_wave_class.setdefault(key, [])
        for scenario_name in missing:
            rp, slr = scenario_name.split("_", 1)
            remaining_by_wave_class[key].append((tid_s, rp, slr))

    n_total = len(tile_ids) * len(all_scenario_names)
    n_remaining = n_total - n_done_total
    print(f"{n_done_total}/{n_total} scenario-jobs already done, {n_remaining} remaining.\n")

    batches = []  # [(wave, size_class, batch_id, [(tile_id, rp, slr), ...]), ...]
    for wave in sorted(hpc_waves):
        class_sizes = {
            c: len(remaining_by_wave_class.get((wave, c), []))
            for c in ("small", "large")
        }
        if not any(class_sizes.values()):
            continue
        class_n_nodes = split_batches_proportionally(class_sizes, n_nodes)
        for size_class in ("small", "large"):
            items = remaining_by_wave_class.get((wave, size_class), [])
            if not items:
                continue
            n_batches = class_n_nodes[size_class]
            k, m = divmod(len(items), n_batches)
            for i in range(n_batches):
                batch_items = items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
                if batch_items:
                    batches.append((wave, size_class, f"{i:03d}", batch_items))

    script_paths = [
        str(local_jobs_dir / f"resume_wave{wave}_{size_class}_batch_{batch_id}.sbatch")
        for wave, size_class, batch_id, _ in batches
    ]

    n_by_wave: dict[int, int] = {}
    for wave, _, _, items in batches:
        n_by_wave[wave] = n_by_wave.get(wave, 0) + len(items)
    print(f"n_nodes={n_nodes}, remaining work by wave:")
    for wave in sorted(n_by_wave):
        print(f"  wave {wave}: {n_by_wave[wave]} job(s) across "
              f"{sum(1 for w, _, _, _ in batches if w == wave)} batch(es)")
    print()

    generate_resume_dispatch(
        hpc_cfg=hpc_cfg,
        base_config_path=config_path,
        batches=batches,
        script_paths=script_paths,
        resolved_config_path=str(local_jobs_dir / "resolved_config.yml"),
        submit_waves_path=str(local_jobs_dir / "submit_resume_waves.sh"),
    )
    print(f"\nCancel the original dispatch's still-running jobs for these wave(s) FIRST (see module docstring), then:")
    print(f"Submit on Hydrax with: bash {linux_config['hpc']['jobs_dir']}/submit_resume_waves.sh")


if __name__ == "__main__":
    main()
