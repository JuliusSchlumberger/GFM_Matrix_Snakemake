"""Write the SFINCS calibration's HPC chain (calibration_roughness_SFINCS).

stage1 (local): candidate shards + per-shard screen/boundary-line/pre-flight jobs
and a bridge job (afterany) that selects tiles and runs `stage2 --submit`.
stage2: spreads unfinished tiles over jobs, longest-processing-time-first.

Usage:
    python generate_calibration_jobs.py stage1            # then on the cluster: bash .../hpc_jobs_stage1/submit_chain.sh
    python generate_calibration_jobs.py stage2 --submit   # what the bridge job runs

Job resources: hpc.calibration_sbatch (config_hpc.yml); roots and pythons: site_env.sh.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import check_site_roots, read_root, site_paths  # noqa: E402

# Site paths: single definition in site_env.sh (env-overridable), shared with the shell drivers.
_HX = site_paths()
LINUX_DATA_ROOT = _HX["GFM_DATA_ROOT"]
LINUX_CODE_ROOT = _HX["GFM_CODE_ROOT"]
GFM_PY = _HX["GFM_PYTHON"]
HYDROMT_PY = _HX["GFM_SFINCS_PYTHON"]
DONE_MARKER = "calibration_run.done"
FALLBACK_MCELLS = 2.0  # tile without a pre-flight record


def _hpc_view(args) -> tuple[dict, dict]:
    """Return the HPC view of --config and its hpc.calibration_sbatch with the CLI overrides.

    Also checks that site_env.sh and the config agree on the roots.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from config_utils import load_hpc_config  # noqa: E402
    from hpc_common import job_sbatch_cfg  # noqa: E402

    linux_config = load_hpc_config(args.config)
    check_site_roots(_HX, linux_config["paths"])
    sbatch = job_sbatch_cfg(linux_config["hpc"], "calibration_sbatch")
    if args.partition:
        sbatch["partition"] = args.partition
    if args.mem:
        sbatch["mem"] = args.mem
    return linux_config, sbatch


def _sbatch_header(name: str, sbatch: dict, hours: int, log_dir: str) -> list[str]:
    # No -e: a job's later steps still run after an earlier one partly failed (stage 1: the
    # pre-flight after tiles errored in the line build; stage 2: the rest of the tile loop).
    # Steps the rest cannot do without end in `|| exit 1`.
    partition = [f"#SBATCH --partition={sbatch['partition']}"] if sbatch.get("partition") else []
    return ["#!/bin/bash", f"#SBATCH --job-name={name}", *partition,
            f"#SBATCH --cpus-per-task={sbatch['cpus_per_task']}", f"#SBATCH --mem={sbatch['mem']}",
            f"#SBATCH --time={hours // 24}-{hours % 24:02d}:00:00",
            f"#SBATCH --output={log_dir}/{name}_%j.out", f"#SBATCH --error={log_dir}/{name}_%j.err", "",
            "set -uo pipefail", "unset PROJ_LIB PROJ_DATA GDAL_DATA", f'cd "{LINUX_CODE_ROOT}/sfincs_tiles" || exit 1', ""]


def cmd_stage1(args) -> None:
    """Write the stage-1 sbatch jobs, the bridge job and submit_chain.sh.

    Side effects: writes {base}/resolved_config.yml (Linux path view) and runs
    ``select_calibration_tiles.py pool``.
    """
    import yaml

    config_path = Path(args.config)
    linux_config, sbatch = _hpc_view(args)
    cpus = sbatch["cpus_per_task"]
    root = read_root(config_path)
    base = root / args.base_dir_name
    base.mkdir(parents=True, exist_ok=True)
    (base / "resolved_config.yml").write_text(yaml.safe_dump(linux_config), encoding="utf-8", newline="\n")

    subprocess.run([sys.executable, str(Path(__file__).parent / "select_calibration_tiles.py"), "pool",
                    "--config", str(config_path), "--base-dir-name", args.base_dir_name,
                    "--n-shards", str(args.n_jobs)], check=True)

    lb = f"{LINUX_DATA_ROOT}/{args.base_dir_name}"
    cfg = f"{lb}/resolved_config.yml"
    jobs_dir = base / "hpc_jobs_stage1"
    (jobs_dir / "logs").mkdir(parents=True, exist_ok=True)
    ljobs = f"{lb}/hpc_jobs_stage1"
    common = f"--config {cfg} --base-dir-name {args.base_dir_name}"
    names = []
    for k in range(args.n_jobs):
        name = f"stage1_{k:03d}"
        names.append(name)
        shard, screened = f"{lb}/candidates/shard_{k:03d}.txt", f"{lb}/candidates/screened_{k:03d}.txt"
        n = len((base / "candidates" / f"shard_{k:03d}.txt").read_text().split())
        lines = _sbatch_header(f"cal_{name}", sbatch, args.stage1_hours, f"{ljobs}/logs") + [
            f"# {n} candidate tile(s): screen -> boundary lines -> SFINCS mask pre-flight",
            f"{GFM_PY} select_calibration_tiles.py screen {common} --tile-ids-file {shard} --out-ids {screened} || exit 1",
            f"{GFM_PY} build_station_boundary_lines.py {common} --tiles-dir-name model_outputs "
            f"--tile-ids-file {screened} --workers {cpus} --skip-existing --plot all",
            f"{HYDROMT_PY} preflight_sfincs_mask.py {common} --tiles-dir-name model_outputs "
            f"--tile-ids-file {screened} --skip-existing",
            ""]
        (jobs_dir / f"{name}.sbatch").write_text("\n".join(lines), newline="\n")

    bridge = _sbatch_header("cal_bridge", sbatch, 2, f"{ljobs}/logs") + [
        "# after every stage-1 job: line summary -> tile selection + maps -> stage-2 jobs (submitted)",
        f"{GFM_PY} build_station_boundary_lines.py {common} --summarize",
        f"{GFM_PY} select_calibration_tiles.py select {common} --n-tiles {args.n_tiles} || exit 1",
        f"{GFM_PY} generate_calibration_jobs.py stage2 {common} --n-jobs {args.stage2_jobs} --submit",
        ""]
    (jobs_dir / "bridge.sbatch").write_text("\n".join(bridge), newline="\n")
    submit = ["#!/bin/bash", "set -euo pipefail", "ids=()"]
    submit += [f"ids+=($(sbatch --parsable {ljobs}/{n}.sbatch))" for n in names]
    submit += ['dep=$(IFS=:; echo "${ids[*]}")',
               f'bridge=$(sbatch --parsable --dependency=afterany:$dep {ljobs}/bridge.sbatch)',
               'echo "stage 1: ${ids[*]}"', 'echo "bridge (selection + stage-2 submission): $bridge"', ""]
    (jobs_dir / "submit_chain.sh").write_text("\n".join(submit), newline="\n")
    print(f"wrote {jobs_dir}: {len(names)} stage-1 job(s) + bridge\n"
          f"Submit on the cluster with: bash {ljobs}/submit_chain.sh")


def cmd_stage2(args) -> None:
    """Write the stage-2 sbatch jobs for unfinished tiles; submit them with --submit.

    Per-tile minutes = overhead_min + active Mcells (pre-flight) x
    (sfincs_min_per_mcell + eikonal_min_per_mcell).
    """
    _, sbatch = _hpc_view(args)
    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    ids = [t.strip() for t in (base / "tile_ids.txt").read_text().split() if t.strip()]
    tiles, n_done = [], 0
    for tid in ids:
        if (base / tid / "outputs" / DONE_MARKER).exists():
            n_done += 1
            continue
        pf = base / "boundary_lines" / "preflight" / f"{tid}.json"
        mcells = json.loads(pf.read_text()).get("n_active", 0) / 1e6 if pf.exists() else FALLBACK_MCELLS
        est = args.overhead_min + mcells * (args.sfincs_min_per_mcell + args.eikonal_min_per_mcell)
        tiles.append((tid, est))
    if not tiles:
        print(f"Nothing to do: {n_done} tile(s) already done.")
        return

    heap = [(0.0, j) for j in range(min(args.n_jobs, len(tiles)))]
    jobs = {j: [] for _, j in heap}
    load = {j: 0.0 for _, j in heap}
    for tid, est in sorted(tiles, key=lambda x: -x[1]):
        total, j = heapq.heappop(heap)
        jobs[j].append(tid)
        load[j] = total + est
        heapq.heappush(heap, (load[j], j))

    out_dir = base / "hpc_jobs_stage2"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    lout = f"{LINUX_DATA_ROOT}/{args.base_dir_name}/hpc_jobs_stage2"
    names = []
    for j, tids in sorted(jobs.items()):
        name = f"final_{j:03d}"
        names.append(name)
        (out_dir / f"tiles_{j:03d}.txt").write_text("\n".join(tids) + "\n", newline="\n")
        hours = max(1, math.ceil(load[j] * args.time_margin / 60.0))
        lines = _sbatch_header(f"cal_{name}", sbatch, hours, f"{lout}/logs") + [
            f"# {len(tids)} tile(s), estimated {load[j] / 60:.1f} h (generate_calibration_jobs.py stage2)",
            "while read -r TILE; do",
            '  [ -n "$TILE" ] || continue',
            f'  BASE_DIR_NAME={args.base_dir_name} SFINCS_THREADS={sbatch["cpus_per_task"]} '
            f'bash {LINUX_CODE_ROOT}/sfincs_tiles/run_final_tile.sh "$TILE" </dev/null',
            f"done < {lout}/tiles_{j:03d}.txt", ""]
        (out_dir / f"{name}.sbatch").write_text("\n".join(lines), newline="\n")
    (out_dir / "submit_all.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + "".join(f"sbatch {lout}/{n}.sbatch\n" for n in names), newline="\n")
    loads = sorted(load.values())
    print(f"{len(tiles)} tile(s) to run ({n_done} already done) over {len(jobs)} job(s) on {sbatch['partition']}")
    print(f"estimated hours per job: {loads[0] / 60:.1f}-{loads[-1] / 60:.1f} (x{args.time_margin:g} margin for --time; "
          f"total {sum(loads) / 60:.0f} h)")
    if args.submit:
        for n in names:
            subprocess.run(["sbatch", f"{lout}/{n}.sbatch"], check=True)
        print(f"submitted {len(names)} stage-2 job(s)")
    else:
        print(f"submit with: bash {lout}/submit_all.sh")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["stage1", "stage2"])
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--n-jobs", type=int, default=None, help="default: 10 for stage1, 20 for stage2")
    parser.add_argument("--stage2-jobs", type=int, default=20, help="stage1: forwarded to the bridge's stage2 call")
    parser.add_argument("--n-tiles", type=int, default=1000)
    parser.add_argument("--partition", default=None, help="default: hpc.calibration_sbatch.partition")
    parser.add_argument("--mem", default=None, help="default: hpc.calibration_sbatch.mem")
    parser.add_argument("--stage1-hours", type=int, default=24)
    parser.add_argument("--overhead-min", type=float, default=35.0)  # inputs, model build + subgrid, postprocessing
    parser.add_argument("--sfincs-min-per-mcell", type=float, default=12.5)  # p90 over 550 SFINCS runs, 4 threads
    parser.add_argument("--eikonal-min-per-mcell", type=float, default=10.0,
                        help="all 10 friction points together, per million active SFINCS cells (rough)")
    parser.add_argument("--time-margin", type=float, default=1.5)
    parser.add_argument("--submit", action="store_true", help="stage2: sbatch the jobs right away")
    args = parser.parse_args()
    if args.n_jobs is None:
        args.n_jobs = 10 if args.command == "stage1" else 20
    {"stage1": cmd_stage1, "stage2": cmd_stage2}[args.command](args)


if __name__ == "__main__":
    main()
