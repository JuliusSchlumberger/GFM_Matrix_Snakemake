"""Generate per-node sbatch scripts for a sweep-budget/obstacle-coupling study.

One process per node runs its tile slice sequentially (amortizes env setup and
Numba JIT). Tiles are LPT bin-packed on pixel count (bbox area_deg2 * 3600**2,
EPSG:4326 ~1-arcsecond DeltaDTM); --time is a no-early-exit worst case.
Build wet_tiles_selected.txt afterwards with aggregate_wet_tiles.py.
Partition, mem, cpus and account default to hpc.sbatch (config_hpc.yml).

Usage:
    python generate_calibration_batch_jobs.py sweep_budget
    python generate_calibration_batch_jobs.py obstacle_coupling --tile-ids-file <wet_tiles_selected.txt>
    python generate_calibration_batch_jobs.py sweep_budget --study-dir-name calibration_500_tiles
    bash <printed submit script path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, load_hpc_config, retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sfincs_tiles"))
from gfm_config import check_site_roots, site_paths  # noqa: E402  (site python path: sfincs_tiles/site_env.sh)

_REPO_ROOT = Path(__file__).resolve().parent.parent

N_NODES_DEFAULT = 30

# Measured on tiles of 196K-18.9M cells (within ~10%); used only for the --time estimate.
RATE_S_PER_MCELL_SWEEP = 0.05

SWEEPS_PER_ROUND = 4

SCRIPTS = {
    "sweep_budget": {
        "script_name": "run_sweep_budget_study.py",
        "out_subdir": "sweep_budget",
        "worst_case_sweeps_per_tile": 100 * SWEEPS_PER_ROUND,  # MAX_ROUNDS_CEILING, no early exit
        "default_tile_ids_file": "candidate_tiles.txt",
    },
    "obstacle_coupling": {
        "script_name": "run_obstacle_coupling_study.py",
        "out_subdir": "obstacle_coupling",
        # (max_outer+1) x inner_max_rounds x 4 sweeps at the 15/40 defaults, no early exit.
        "worst_case_sweeps_per_tile": 16 * 40 * SWEEPS_PER_ROUND,
        "default_tile_ids_file": "sweep_budget/wet_tiles_selected.txt",
    },
}


def load_pixel_counts(tile_ids: list[str], domain_tiles_path: Path) -> dict[str, float]:
    """Map tile_id to approximate pixel count from its bbox (1 arcsecond cells)."""
    gdf = gpd.read_file(domain_tiles_path)
    bounds = gdf.geometry.bounds
    area_deg2 = (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"])
    px_by_tile_id = {str(int(t)): float(a) * 3600.0 * 3600.0 for t, a in zip(gdf["tile_id"], area_deg2)}
    missing = [t for t in tile_ids if t not in px_by_tile_id]
    if missing:
        raise ValueError(f"{len(missing)} tile ID(s) not found in {domain_tiles_path}: {missing[:10]}...")
    return {t: px_by_tile_id[t] for t in tile_ids}


def lpt_bin_pack(tile_ids: list[str], px_by_tile_id: dict[str, float], n_nodes: int) -> list[list[str]]:
    """Split tiles into n_nodes buckets by LPT greedy bin packing on pixel count."""
    ordered = sorted(tile_ids, key=lambda t: px_by_tile_id[t], reverse=True)
    buckets: list[list[str]] = [[] for _ in range(n_nodes)]
    bucket_cost = [0.0] * n_nodes
    for tile_id in ordered:
        idx = min(range(n_nodes), key=lambda i: bucket_cost[i])
        buckets[idx].append(tile_id)
        bucket_cost[idx] += px_by_tile_id[tile_id]
    return buckets


def estimate_worst_case_hours(batch_tiles: list[str], px_by_tile_id: dict[str, float], worst_case_sweeps_per_tile: int) -> float:
    """Estimate a batch's wall-clock hours assuming no early exit."""
    total_px = sum(px_by_tile_id[t] for t in batch_tiles)
    seconds = (total_px / 1e6) * worst_case_sweeps_per_tile * RATE_S_PER_MCELL_SWEEP
    return seconds / 3600.0


def hours_to_slurm_time(hours: float) -> str:
    """Format hours as SLURM D-HH:MM:SS, rounded up to a whole hour (2.3 -> 0-03:00:00)."""
    import math
    total_hours = max(1, math.ceil(hours))
    days, rem_hours = divmod(total_hours, 24)
    return f"{days}-{rem_hours:02d}:00:00"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=list(SCRIPTS))
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument(
        "--study-dir-name", default="calibration_500_tiles",
        help="name of the study's own output directory under paths.root - default is the "
             "current global-representativeness study; pass a different name for an "
             "independent, separately-sized study",
    )
    parser.add_argument("--tile-ids-file", default=None, help="default: <study-dir-name>/<study's own default file>, see SCRIPTS above")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=None, help="default: hpc.sbatch.partition")
    parser.add_argument("--time", default=None, help="default: derived from the worst-loaded node's own estimated worst-case time, see estimate_worst_case_hours()")
    parser.add_argument("--time-margin", type=float, default=1.5, help="safety multiplier applied to the worst-loaded node's estimated worst-case hours before rounding up to --time (default: 1.5x)")
    parser.add_argument("--mem", default=None, help="default: hpc.sbatch.mem")
    parser.add_argument("--cpus-per-task", type=int, default=None, help="default: hpc.sbatch.cpus_per_task")
    parser.add_argument("--account", default=None, help="default: hpc.sbatch.account")
    parser.add_argument(
        "--extra-args", default="",
        help="extra CLI args forwarded verbatim to each node's script call, e.g. '--epsilon 0.03' "
             "or '--max-outer-iterations 4 --max-rounds 40' to reproduce production's config.yml values",
    )
    args = parser.parse_args()

    spec = SCRIPTS[args.study]

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    site = site_paths()
    check_site_roots(site, linux_config["paths"])
    gfm_py_linux = site["GFM_PYTHON"]
    sbatch_cfg = linux_config["hpc"]["sbatch"]
    for key in ("partition", "mem", "cpus_per_task", "account"):
        if getattr(args, key) is None:
            setattr(args, key, sbatch_cfg.get(key, ""))

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]

    study_root_local = local_root / args.study_dir_name
    study_root_linux = f"{linux_root}/{args.study_dir_name}"

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
    retry_transient_io((study_root_local / spec["out_subdir"]).mkdir, parents=True, exist_ok=True)

    # Nodes get the pre-merged Linux-path config so they need no override files.
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
        ]
        if args.partition:
            lines.append(f"#SBATCH --partition={args.partition}")
        if args.account:
            lines.append(f"#SBATCH --account={args.account}")
        lines += [
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH --cpus-per-task={args.cpus_per_task}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            # -e would change nothing: the node runs one command, and the study script itself
            # carries on past a failing tile.
            "set -uo pipefail",
            "unset PROJ_LIB PROJ_DATA GDAL_DATA",  # as the sfincs_tiles jobs: no stale PROJ/GDAL data dirs
            f'"{gfm_py_linux}" "{linux_code_root}/calibration_studies/{spec["script_name"]}" --out-dir "{out_dir_linux}" '
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
    print(f"\nSubmit on the cluster with: bash {linux_jobs_dir}/{submit_filename}")


if __name__ == "__main__":
    main()
