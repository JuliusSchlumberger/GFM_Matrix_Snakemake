"""Write per-node jobs for a sweep-budget/obstacle-coupling study; entry point `generate()`.

One process per node runs its tile slice sequentially (amortizes env setup and
Numba JIT). Tiles are LPT bin-packed on pixel count (bbox area_deg2 * 3600**2,
EPSG:4326 ~1-arcsecond DeltaDTM); --time is a no-early-exit worst case.
Build wet_tiles_selected.txt afterwards with aggregate_wet_tiles.py.
Resources: the solver_study profile of config_hpc.yml (--profile-set to override). --friction-source
picks the study's friction (land cover at the config's friction_scale_factor, or a uniform
rate); each source writes to its own <out_subdir>_<source> folder.

Usage:
    python generate_calibration_batch_jobs.py sweep_budget
    python generate_calibration_batch_jobs.py obstacle_coupling --tile-ids-file <wet_tiles_selected.txt>
    python generate_calibration_batch_jobs.py sweep_budget --friction-source uniform --uniform-attenuation-m-per-km 0.2
    python generate_calibration_batch_jobs.py sweep_budget --study-dir-name calibration_500_tiles
    bash <printed submit script>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, load_hpc_config, retry_transient_io  # noqa: E402
from hpc_common import JobChain, add_profile_set_argument, apply_profile_sets, lpt_pack  # noqa: E402

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
        "default_tile_ids_file": "sweep_budget_{source}/wet_tiles_selected.txt",
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


def estimate_worst_case_hours(batch_tiles: list[str], px_by_tile_id: dict[str, float], worst_case_sweeps_per_tile: int) -> float:
    """Estimate a batch's wall-clock hours assuming no early exit."""
    total_px = sum(px_by_tile_id[t] for t in batch_tiles)
    seconds = (total_px / 1e6) * worst_case_sweeps_per_tile * RATE_S_PER_MCELL_SWEEP
    return seconds / 3600.0


def hours_to_slurm_time(hours: float) -> str:
    """Format hours as SLURM D-HH:MM:SS, rounded up to a whole hour (2.3 -> 0-03:00:00)."""
    total_hours = max(1, math.ceil(hours))
    days, rem_hours = divmod(total_hours, 24)
    return f"{days}-{rem_hours:02d}:00:00"


def generate(study: str, config_path: str | Path, *, friction_source: str = "landcover",
             study_dir_name: str = "calibration_500_tiles", tile_ids_file: str | None = None,
             n_nodes: int = N_NODES_DEFAULT, time: str | None = None, time_margin: float = 1.5,
             uniform_attenuation_m_per_km: float | None = None, extra_args: str = "",
             profile_sets: list[str] | None = None) -> str:
    """Write the study's batch jobs, resolved_config.yml and submit script; return its Linux path."""
    spec = SCRIPTS[study]
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    apply_profile_sets(linux_config["hpc"], profile_sets)

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]
    linux_code_root = linux_config["paths"]["code_root"]
    study_root_local = local_root / study_dir_name
    study_root_linux = f"{linux_root}/{study_dir_name}"

    out_subdir = f"{spec['out_subdir']}_{friction_source}"
    friction_args = f"--friction-source {friction_source}" + (
        f" --uniform-attenuation-m-per-km {uniform_attenuation_m_per_km:g}"
        if uniform_attenuation_m_per_km is not None else "")
    ids_path = (Path(tile_ids_file) if tile_ids_file
                else study_root_local / spec["default_tile_ids_file"].format(source=friction_source))
    all_tile_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if not all_tile_ids:
        raise ValueError(f"no tile IDs in {ids_path}")
    print(f"{len(all_tile_ids)} tile(s) from {ids_path}")

    n_nodes = min(n_nodes, len(all_tile_ids))
    domain_tiles_path = local_root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg"
    px_by_tile_id = load_pixel_counts(all_tile_ids, domain_tiles_path)
    batches = lpt_pack(px_by_tile_id, n_nodes)
    sizes = [len(b) for b in batches]
    # Correction B solves every tile twice (production flood_depth_dense), and so do the studies.
    passes = 2 if local_config["simulation"]["flooding"]["duration_correction"]["enabled"] else 1
    hours_per_batch = [passes * estimate_worst_case_hours(b, px_by_tile_id, spec["worst_case_sweeps_per_tile"])
                       for b in batches]
    max_hours = max(hours_per_batch) if hours_per_batch else 0.0
    print(f"{n_nodes} node(s), {min(sizes)}-{max(sizes)} tile(s) each (LPT bin-packed by estimated pixel count)")
    print(f"worst-case estimated time per node: {min(hours_per_batch):.2f}-{max_hours:.2f}h "
          f"(zero early-exit-ever bound, {spec['worst_case_sweeps_per_tile']} sweeps/tile)")
    time_limit = time or hours_to_slurm_time(max_hours * time_margin)
    print(f"--time={time_limit} ({'explicit' if time else f'derived: {max_hours:.2f}h worst case x {time_margin} margin'})")

    local_jobs_dir = study_root_local / "hpc_jobs"
    linux_jobs_dir = f"{study_root_linux}/hpc_jobs"
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)
    retry_transient_io((study_root_local / out_subdir).mkdir, parents=True, exist_ok=True)

    # Nodes get the pre-merged Linux-path config so they need no override files.
    atomic_write(str(study_root_local / "resolved_config.yml"), lambda f: yaml.safe_dump(linux_config, f),
                 encoding="utf-8", newline="")
    linux_config_path = f"{study_root_linux}/resolved_config.yml"

    out_dir_linux = f"{study_root_linux}/{out_subdir}"
    batch_name_prefix = f"calib_{study}_{friction_source}_batch"
    chain = JobChain(linux_config["hpc"], local_jobs_dir, linux_jobs_dir)
    for i, batch_tiles in enumerate(batches):
        if not batch_tiles:
            continue
        # -e would change nothing: the node runs one command, and the study script itself
        # carries on past a failing tile.
        chain.add(f"{batch_name_prefix}_{i:03d}", "solver_study", [
            f'"$GFM_PY" "{linux_code_root}/calibration_studies/{spec["script_name"]}" --out-dir "{out_dir_linux}" '
            f'--config "{linux_config_path}" {friction_args} --tile-ids {" ".join(batch_tiles)} {extra_args}',
        ], resources={"time": time_limit}, errexit=False)
        print(f"  {batch_name_prefix}_{i:03d}: {len(batch_tiles)} tile(s)")
    return chain.write_submit(f"submit_{batch_name_prefix}es.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=list(SCRIPTS))
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    add_study_arguments(parser)
    args = parser.parse_args()
    submit = generate(args.study, args.config, **study_kwargs(args))
    print(f"Submit on the cluster with: bash {submit}")


def add_study_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the study options shared by this CLI and hpc/gfm_hpc.py studies."""
    parser.add_argument(
        "--study-dir-name", default="calibration_500_tiles",
        help="the study's output directory under paths.root; another name gives an independent study",
    )
    parser.add_argument("--tile-ids-file", default=None, help="default: <study-dir-name>/<the study's default file>")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--time", default=None, help="default: the worst-loaded node's worst-case estimate x --time-margin")
    parser.add_argument("--time-margin", type=float, default=1.5)
    parser.add_argument("--friction-source", choices=["landcover", "uniform"], default="landcover")
    parser.add_argument("--uniform-attenuation-m-per-km", type=float, default=None,
                        help="with --friction-source uniform (default: the config's uniform_attenuation_m_per_km)")
    parser.add_argument(
        "--extra-args", default="",
        help="extra CLI args forwarded verbatim to each node's script call, e.g. '--epsilon 0.03' "
             "or '--max-outer-iterations 4' to reproduce production's config.yml value",
    )
    add_profile_set_argument(parser)


def study_kwargs(args: argparse.Namespace) -> dict:
    """Return generate()'s keyword arguments from parsed add_study_arguments options."""
    return {"friction_source": args.friction_source, "study_dir_name": args.study_dir_name,
            "tile_ids_file": args.tile_ids_file, "n_nodes": args.n_nodes, "time": args.time,
            "time_margin": args.time_margin, "uniform_attenuation_m_per_km": args.uniform_attenuation_m_per_km,
            "extra_args": args.extra_args, "profile_sets": args.profile_set}


if __name__ == "__main__":
    main()
