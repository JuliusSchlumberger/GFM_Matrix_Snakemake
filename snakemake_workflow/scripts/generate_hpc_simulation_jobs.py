"""Write the simulation-wave jobs from the tile grid; entry point `generate()`.

A hop_distance>=1 tile is seeded from a lower-hop neighbour's output for the same scenario, so
wave N+1 (afterany) waits until every wave-N job has ended. Within a wave, whole tiles are split
evenly in tile-grid order over hpc.n_nodes batches, or over the matching hpc.wave_tiers entry.
Each batch runs its (tile, RP, SLR) items via run_simulation_batch.py, cpus_per_task at a time.
After the last wave, one job generates and submits the postprocessing chain. Does not check that
preprocessing is complete: the preprocess chain's dispatch job does that before calling this.

--resume batches only the missing (tile, RP, SLR) items into resume_wave*.sbatch. Cancel the
original dispatch's still-running jobs first: in-flight work counts as missing.

Usage:
    python generate_hpc_simulation_jobs.py [--config path/to/config.yml] [--resume] [--profile-set P.F=V]
    bash <printed submit script>
"""

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, load_hpc_config, merged_waterlevel_names, retry_transient_io  # noqa: E402
from hpc_common import (  # noqa: E402
    JobChain, add_profile_set_argument, apply_profile_sets, even_split, resolve_profile, tile_grid_override_lines,
)


def _resolve_wave_tier(hop: int, wave_tiers: list[dict]) -> dict:
    """Return the first hpc.wave_tiers entry whose `hops` contains `hop`; `hops: null` matches all."""
    for tier in wave_tiers:
        hops = tier.get("hops")
        if hops is None or hop in hops:
            return tier
    raise ValueError(f"no hpc.wave_tiers entry matches hop_distance={hop} - add a catch-all 'hops: null' tier")


def _fresh_batches(hpc_waves: dict[int, list[str]], hpc_cfg: dict, return_periods: list[str],
                   waterlevel_names: list[str]) -> list[tuple[int, str, list[tuple[str, str, str]]]]:
    """Whole tiles per batch; a tile's full RP x SLR set stays on one node."""
    batches = []
    for wave in sorted(hpc_waves):
        tiles = hpc_waves[wave]
        n = hpc_cfg["n_nodes"]
        if hpc_cfg.get("wave_tiers"):
            tier = _resolve_wave_tier(wave, hpc_cfg["wave_tiers"])
            n = len(tiles) if tier["n_nodes"] is None else tier["n_nodes"]
        for i, batch_tiles in enumerate(even_split(tiles, n)):
            items = [(t, rp, slr) for t in batch_tiles for rp in return_periods for slr in waterlevel_names]
            batches.append((wave, f"{i:03d}", items))
    return batches


def _resume_batches(hpc_waves: dict[int, list[str]], hpc_cfg: dict, model_outputs: Path,
                    return_periods: list[str], waterlevel_names: list[str]) -> list:
    """Only the missing (tile, RP, SLR) items, re-split per wave over the full node budget."""
    scenarios = [(rp, slr) for rp in return_periods for slr in waterlevel_names]
    batches = []
    n_done = n_total = 0
    for wave in sorted(hpc_waves):
        items = []
        for tid in hpc_waves[wave]:
            results_dir = model_outputs / tid / "results"
            done = ({p.stem[len("waterdepth_"):] for p in results_dir.glob("waterdepth_*.tif")}
                    if results_dir.is_dir() else set())
            n_total += len(scenarios)
            n_done += sum(f"{rp}_{slr}" in done for rp, slr in scenarios)
            items += [(tid, rp, slr) for rp, slr in scenarios if f"{rp}_{slr}" not in done]
        if not items:
            continue
        n = hpc_cfg["n_nodes"]
        if hpc_cfg.get("wave_tiers"):
            tier = _resolve_wave_tier(wave, hpc_cfg["wave_tiers"])
            if tier["n_nodes"] is None:
                # One node per tile, not per missing item.
                by_tile: dict[str, list] = {}
                for item in items:
                    by_tile.setdefault(item[0], []).append(item)
                batches += [(wave, f"{i:03d}", v) for i, v in enumerate(by_tile.values())]
                continue
            n = tier["n_nodes"]
        batches += [(wave, f"{i:03d}", b) for i, b in enumerate(even_split(items, n))]
    print(f"{n_done}/{n_total} scenario-jobs already done, {n_total - n_done} remaining.")
    return batches


def generate(config_path: str | Path, *, resume: bool = False, profile_sets: list[str] | None = None) -> str:
    """Write the wave jobs, resolved_config.yml and the submit script; return its Linux path."""
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    hpc_cfg = apply_profile_sets(linux_config["hpc"], profile_sets)

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = hpc_cfg["jobs_dir"]
    code_root = linux_config["paths"]["code_root"]
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)
    # atomic_write: a compute node reading this over the shared mount right after a
    # non-atomic write can see a partial file (UnicodeDecodeError).
    atomic_write(local_jobs_dir / "resolved_config.yml", lambda f: yaml.safe_dump(linux_config, f),
                 encoding="utf-8", newline="")

    # Local view: linux_config's path is only for embedding in job scripts.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    hpc_waves: dict[int, list[str]] = {}
    for tid, hop in zip(tile_gdf["tile_id"], tile_gdf["hop_distance"]):
        hpc_waves.setdefault(int(hop), []).append(str(int(tid)))
    return_periods = [f"RP{rp}" for rp in local_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_waterlevel_names(local_config["boundary_conditions"], local_config["adaptation"])

    if resume:
        batches = _resume_batches(hpc_waves, hpc_cfg, Path(local_config["simulation"]["model_outputs"]),
                                  return_periods, waterlevel_names)
        prefix, submit_name = "resume_", "submit_resume_waves.sh"
    else:
        batches = _fresh_batches(hpc_waves, hpc_cfg, return_periods, waterlevel_names)
        prefix, submit_name = "", "submit_waves.sh"

    chain = JobChain(hpc_cfg, local_jobs_dir, linux_jobs_dir)
    prev_wave: list[str] = []
    this_wave: list[str] = []
    current = None
    for wave, batch_id, items in batches:
        if wave != current:
            prev_wave = this_wave or prev_wave
            this_wave, current = [], wave
        name = f"{prefix}wave{wave}_batch_{batch_id}"
        with open(local_jobs_dir / f"{name}_triples.csv", "w", encoding="utf-8", newline="\n") as f:
            csv.writer(f).writerows(items)
        resources = None
        if hpc_cfg.get("wave_tiers"):
            tier = _resolve_wave_tier(wave, hpc_cfg["wave_tiers"])
            resources = {k: tier[k] for k in ("partition", "cpus_per_task", "mem", "time")}
        workers = resolve_profile(hpc_cfg, "simulate", resources)["cpus_per_task"]
        body = [
            f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
            ': > "$FAIL_LOG"',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{linux_config["tile_grid"]["path"]}" LOCAL_TILE_GRID || exit 1',
            *tile_grid_override_lines(),
            "",
            f'"$GFM_PY" "{code_root}/snakemake_workflow/scripts/run_simulation_batch.py" \\',
            f'  --triples-file "{linux_jobs_dir}/{name}_triples.csv" \\',
            '  --config "$LOCAL_CONFIGFILE" \\',
            f"  --workers {workers} \\",
            '  --fail-log "$FAIL_LOG"',
        ]
        # Not -e: one failed (tile, rp, slr) must not abort the rest of this node's batch, so
        # each setup step exits explicitly on failure instead.
        chain.add(name, "simulate", body, after=prev_wave, resources=resources, errexit=False,
                  helpers=("stage_config", "stage_tile_grid"))
        this_wave.append(name)
        print(f"  {name}: {len(items)} item(s), {workers}-way concurrent")

    # The postprocessing chain is generated once every wave has ended; its generator must get this
    # dispatch's config, or a scenario run would postprocess production's outputs.
    bridge = "gfm_generate_postprocess_and_dispatch"
    chain.add(bridge, "dispatch", [
        f'cd "{code_root}"',
        # The staged config is what the postprocess generator embeds, so tile_grid.path is not staged.
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        'echo "=== Generating postprocessing + exposure dispatch ==="',
        '"$GFM_PY" hpc/gfm_hpc.py run postprocess --config "$LOCAL_CONFIGFILE"',
        'echo "=== Submitting postprocessing + exposure ==="',
        f'bash "{linux_jobs_dir}/submit_postprocess_and_exposure.sh"',
    ], after=chain.names(), helpers=("stage_config",))

    submit = chain.write_submit(submit_name)
    print(f"{len(batches)} wave job(s) + postprocess dispatch -> {local_jobs_dir}")
    if resume:
        print("Cancel the original dispatch's still-running wave jobs first (docs/hpc.md).")
    return submit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"))
    parser.add_argument("--resume", action="store_true", help="batch only the missing (tile, RP, SLR) items")
    add_profile_set_argument(parser)
    args = parser.parse_args()
    submit = generate(args.config, resume=args.resume, profile_sets=args.profile_set)
    print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
