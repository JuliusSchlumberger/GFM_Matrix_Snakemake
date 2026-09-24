"""Generate N independent sbatch scripts (one per node) to run the 260-tile
sweep-budget or obstacle-coupling calibration study on Hydrax, each node
processing its OWN slice of tiles sequentially in one continuous process
(2026-09-24 - replaces generate_calibration_array_job.py's one-task-per-tile
design: submitting/scheduling ~300 individual SLURM tasks pays real
per-task dispatch overhead - conda/env setup, Python interpreter start,
Numba JIT (re-)compile if the on-disk cache isn't warm yet - on every single
tile regardless of how small, which dominates for the many tiles that only
take a fraction of a second to actually compute. Batching ~10 tiles into one
continuous process per node, like generate_v2_batch_jobs.py already does
for the SFINCS validation batches, pays that overhead once per NODE instead
of once per TILE).

Tiles are distributed by LPT (Longest Processing Time first) greedy
bin-packing on ESTIMATED cost, not round-robin or a contiguous slice
(2026-09-24, user direction: "spread the tiles based on their size across
n batches"): each candidate's pixel count (`area_deg2 * 3600**2` from
domain_tiles_global.gpkg's own bbox, EPSG:4326 ~1-arcsecond-native DeltaDTM
tiles - the SAME proxy select_calibration_tiles.py already uses/trusts for
this) stands in for its compute cost, tiles are sorted largest-first, and
each one goes to whichever node currently has the LOWEST accumulated cost
so far - the standard, simple, effective heuristic for balanced multiway
partitioning without needing an exact solve. This directly targets the
straggler problem a naive split (contiguous OR round-robin) doesn't fully
avoid: candidate_tiles.txt is ordered "bin, then percentile-within-bin" (see
select_calibration_tiles.py), so nearby entries can be similar in size by
construction.

The same per-tile pixel-count estimate also drives a real --time estimate
(not a blanket guess): each node's own worst-case wall-clock time is its
assigned tiles' total pixel count, converted via this session's own
empirically-measured rate (~0.05s per (million cells x sweep), confirmed
live on 3 real tiles spanning 196K-18.9M cells within ~10% of each other),
times the study's own worst-case sweep count per tile (assumes zero early
exit ever - a deliberately pessimistic bound, since real tiles mostly
converge well before their round ceiling). Printed per node so the actual
--time (or the default) can be judged against real numbers instead of
guessed - see estimate_worst_case_hours() below.

Each node calls the target script ONCE with its full tile-id list (already
multi-tile-capable via --tile-ids id1 id2 ...) - `set -uo pipefail`, not
`-e`, matching this repo's established convention, so one tile's failure
doesn't abort the rest of that node's own slice.

wet_tiles_selected.txt is deliberately NEVER written by these node
processes (see test_sweep_budget_calibration.py's own
--write-wet-tiles-summary flag, off by default here) - build it with
aggregate_wet_tiles.py once every node's CSVs are on disk.

Usage:
    python generate_calibration_batch_jobs.py sweep_budget
    python generate_calibration_batch_jobs.py obstacle_coupling --tile-ids-file <wet_tiles_selected.txt>
    bash <printed submit script path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

N_NODES_DEFAULT = 30
PARTITION_DEFAULT = "1vcpu"
MEM_DEFAULT = "7G"
CPUS_PER_TASK_DEFAULT = 1
GFM_PY_LINUX = "/u/schlumbe/.conda/envs/gfm/bin/python"

# Empirically measured this session (tests/test_sweep_budget_calibration.py, 3 real tiles:
# 196K/2.49M/18.9M cells, 0.040-0.049 s per (million cells x 50 sweeps), and confirmed again
# on tile 948's full 400-sweep run at 369s for 18.9M cells) - used only for a worst-case
# --time ESTIMATE, not for the bin-packing itself (pixel count alone is enough for balance).
RATE_S_PER_MCELL_SWEEP = 0.05

SWEEPS_PER_ROUND = 4

SCRIPTS = {
    "sweep_budget": {
        "script_name": "test_sweep_budget_calibration.py",
        "out_subdir": "sweep_budget",
        "worst_case_sweeps_per_tile": 100 * SWEEPS_PER_ROUND,  # MAX_ROUNDS_CEILING - zero early exit ever
        "default_tile_ids_file": "candidate_tiles.txt",
    },
    "obstacle_coupling": {
        "script_name": "test_obstacle_coupling_calibration.py",
        "out_subdir": "obstacle_coupling",
        # (max_outer+1) solves x inner_max_rounds rounds x 4 sweeps, at this study's own
        # 15/50 defaults (DEFAULT_MAX_OUTER/DEFAULT_INNER_MAX_ROUNDS) - zero early exit ever.
        "worst_case_sweeps_per_tile": 16 * 50 * SWEEPS_PER_ROUND,
        "default_tile_ids_file": "sweep_budget/wet_tiles_selected.txt",
    },
}


def load_pixel_counts(tile_ids: list[str], domain_tiles_path: Path) -> dict[str, float]:
    """tile_id (str) -> approx_px, same proxy select_calibration_tiles.py's
    own candidate-selection table already prints and trusts for this."""
    gdf = gpd.read_file(domain_tiles_path)
    bounds = gdf.geometry.bounds
    area_deg2 = (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"])
    px_by_tile_id = {str(int(t)): float(a) * 3600.0 * 3600.0 for t, a in zip(gdf["tile_id"], area_deg2)}
    missing = [t for t in tile_ids if t not in px_by_tile_id]
    if missing:
        raise ValueError(f"{len(missing)} tile ID(s) not found in {domain_tiles_path}: {missing[:10]}...")
    return {t: px_by_tile_id[t] for t in tile_ids}


def lpt_bin_pack(tile_ids: list[str], px_by_tile_id: dict[str, float], n_nodes: int) -> list[list[str]]:
    """Longest-Processing-Time-first greedy bin packing: sort descending by
    estimated cost, assign each tile to whichever bucket has the lowest
    accumulated cost so far. See module docstring for why (vs round-robin
    or a contiguous slice)."""
    ordered = sorted(tile_ids, key=lambda t: px_by_tile_id[t], reverse=True)
    buckets: list[list[str]] = [[] for _ in range(n_nodes)]
    bucket_cost = [0.0] * n_nodes
    for tile_id in ordered:
        idx = min(range(n_nodes), key=lambda i: bucket_cost[i])
        buckets[idx].append(tile_id)
        bucket_cost[idx] += px_by_tile_id[tile_id]
    return buckets


def estimate_worst_case_hours(batch_tiles: list[str], px_by_tile_id: dict[str, float], worst_case_sweeps_per_tile: int) -> float:
    total_px = sum(px_by_tile_id[t] for t in batch_tiles)
    seconds = (total_px / 1e6) * worst_case_sweeps_per_tile * RATE_S_PER_MCELL_SWEEP
    return seconds / 3600.0


def hours_to_slurm_time(hours: float) -> str:
    """e.g. 2.3 -> '0-03:00:00' - rounds UP to the next whole hour (a --time
    limit shorter than the real worst case defeats the point of estimating
    it) - SLURM's own D-HH:MM:SS format."""
    import math
    total_hours = max(1, math.ceil(hours))
    days, rem_hours = divmod(total_hours, 24)
    return f"{days}-{rem_hours:02d}:00:00"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=list(SCRIPTS))
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids-file", default=None, help="default: calibration_260_tiles/<study's own default>, see SCRIPTS above")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=None, help="default: derived from the worst-loaded node's own estimated worst-case time, see estimate_worst_case_hours()")
    parser.add_argument("--time-margin", type=float, default=1.5, help="safety multiplier applied to the worst-loaded node's estimated worst-case hours before rounding up to --time (default: 1.5x)")
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--extra-args", default="",
        help="extra CLI args forwarded verbatim to each node's script call, e.g. '--epsilon 0.03' "
             "or '--max-outer 5 --inner-max-rounds 12' to reproduce production's own defaults",
    )
    args = parser.parse_args()

    spec = SCRIPTS[args.study]

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]

    study_root_local = local_root / "calibration_260_tiles"
    study_root_linux = f"{linux_root}/calibration_260_tiles"

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else study_root_local / spec["default_tile_ids_file"]
    all_tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    if not all_tile_ids:
        raise ValueError(f"no tile IDs in {tile_ids_file}")
    print(f"{len(all_tile_ids)} tile(s) from {tile_ids_file}")

    n_nodes = min(args.n_nodes, len(all_tile_ids))
    domain_tiles_path = local_root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg"
    px_by_tile_id = load_pixel_counts(all_tile_ids, domain_tiles_path)
    batches = lpt_bin_pack(all_tile_ids, px_by_tile_id, n_nodes)
    sizes = [len(b) for b in batches]
    hours_per_batch = [estimate_worst_case_hours(b, px_by_tile_id, spec["worst_case_sweeps_per_tile"]) for b in batches]
    max_hours = max(hours_per_batch) if hours_per_batch else 0.0
    print(f"{n_nodes} node(s), {min(sizes)}-{max(sizes)} tile(s) each (LPT bin-packed by estimated pixel count)")
    print(f"worst-case estimated time per node: {min(hours_per_batch):.2f}-{max_hours:.2f}h "
          f"(zero early-exit-ever bound, {spec['worst_case_sweeps_per_tile']} sweeps/tile)")

    time_limit = args.time or hours_to_slurm_time(max_hours * args.time_margin)
    print(f"--time={time_limit} ({'explicit' if args.time else f'derived: {max_hours:.2f}h worst case x {args.time_margin} margin'})")

    local_jobs_dir = study_root_local / "hpc_jobs"
    linux_jobs_dir = f"{study_root_linux}/hpc_jobs"
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_root / "calibration_260_tiles" / spec["out_subdir"]).mkdir, parents=True, exist_ok=True)

    # Same resolved_config.yml staging pattern as generate_calibration_array_job.py
    # (config_hpc.yml is a git-ignored, Windows-machine-only file - doesn't exist on
    # the HPC side) and generate_v2_batch_jobs.py's own SFINCS-batch precedent.
    resolved_config_path = study_root_local / "resolved_config.yml"
    atomic_write(str(resolved_config_path), lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    print(f"Wrote {resolved_config_path} (Linux path view, for --config on every node)")
    linux_config_path = f"{study_root_linux}/resolved_config.yml"

    out_dir_linux = f"{study_root_linux}/{spec['out_subdir']}"
    batch_name_prefix = f"calib_{args.study}_batch"

    script_paths = []
    for i, batch_tiles in enumerate(batches):
        if not batch_tiles:
            continue
        batch_id = f"{i:03d}"
        name = f"{batch_name_prefix}_{batch_id}"
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={args.partition}",
        ]
        if args.account:
            lines.append(f"#SBATCH --account={args.account}")
        lines += [
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH --cpus-per-task={args.cpus_per_task}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",  # not -e: one tile's failure must not abort the rest of this node's batch
            f'"{GFM_PY_LINUX}" "{linux_code_root}/tests/{spec["script_name"]}" "{out_dir_linux}" '
            f'--config "{linux_config_path}" --tile-ids {" ".join(batch_tiles)} {args.extra_args}',
            "",
        ]
        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tiles)} tile(s))")

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for script in script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
        ]
    submit_filename = f"submit_{batch_name_prefix}es.sh"
    submit_path = local_jobs_dir / submit_filename
    with open(submit_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    print(f"\nDone. {len(script_paths)} batch(es), {len(all_tile_ids)} tile(s) total.")
    print(f"Wrote {submit_path}")
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/{submit_filename}")


if __name__ == "__main__":
    main()
