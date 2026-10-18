"""HPC chain for the final SFINCS calibration (calibration_roughness_SFINCS).

  stage1  (run locally) - writes {base}/resolved_config.yml (Linux path view),
          the candidate pool + shards (select_calibration_tiles.py pool) and
            hpc_jobs_stage1/stage1_NNN.sbatch  x --n-jobs (default 10, 4vcpu):
                screen -> boundary lines (4 workers) -> SFINCS mask pre-flight
                for that shard's candidates
            hpc_jobs_stage1/bridge.sbatch (afterany on all stage-1 jobs):
                line summary -> tile selection (+ maps) -> `stage2 --submit`
            hpc_jobs_stage1/submit_chain.sh - submits all of the above with the
                dependency wired; run it on Hydrax.
  stage2  (run by the bridge job, or by hand) - spreads {base}/tile_ids.txt
          (tiles without outputs/calibration_run.done) over --n-jobs (default 20,
          4vcpu) jobs, longest-processing-time-first on an estimate from the
          pre-flight's active cell count, and writes
            hpc_jobs_stage2/final_NNN.sbatch + tiles_NNN.txt + submit_all.sh;
          --submit also submits them.

Per-tile estimate (stage 2): --overhead-min (inputs, elevation/roughness, model
build incl. subgrid table, postprocessing) + active cells (M) x
(--sfincs-min-per-mcell + --eikonal-min-per-mcell). SFINCS: p90 of the bline
runs (550 tiles) was 12.5 min per million active cells at 4 threads.

Usage:
    python generate_calibration_jobs.py stage1            # then on Hydrax: bash .../hpc_jobs_stage1/submit_chain.sh
    python generate_calibration_jobs.py stage2 --submit   # what the bridge job runs
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
from gfm_config import hydrax_paths, read_root  # noqa: E402

# Hydrax paths: single definition in hydrax_env.sh (env-overridable), shared with the shell drivers.
_HX = hydrax_paths()
LINUX_DATA_ROOT = _HX["GFM_DATA_ROOT"]
LINUX_CODE_ROOT = _HX["GFM_CODE_ROOT"]
GFM_PY = _HX["GFM_PYTHON"]
HYDROMT_PY = _HX["GFM_SFINCS_PYTHON"]
DONE_MARKER = "calibration_run.done"
FALLBACK_MCELLS = 2.0  # tile without a pre-flight record


def _sbatch_header(name: str, partition: str, cpus: int, mem: str, hours: int, log_dir: str) -> list[str]:
    return ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus}", f"#SBATCH --mem={mem}",
            f"#SBATCH --time={hours // 24}-{hours % 24:02d}:00:00",
            f"#SBATCH --output={log_dir}/{name}_%j.out", f"#SBATCH --error={log_dir}/{name}_%j.err", "",
            "set -uo pipefail", "unset PROJ_LIB PROJ_DATA GDAL_DATA", f"cd {LINUX_CODE_ROOT}/sfincs_tiles || exit 1", ""]


def cmd_stage1(args) -> None:
    """Write the stage-1 sbatch jobs, the bridge job, and submit_chain.sh.

    Resolves the config to a Linux path view, generates the candidate pool and
    shards via select_calibration_tiles.py, and writes one sbatch script per shard
    (screen -> boundary lines -> SFINCS mask pre-flight) plus a bridge job that runs
    the line summary, tile selection and submits stage 2.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (config, base_dir_name, n_jobs, partition, mem,
        stage1_hours, n_tiles, stage2_jobs).
    """
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from config_utils import load_config  # noqa: E402 - local-only (hydromt chain), not needed on Hydrax

    config_path = Path(args.config)
    root = read_root(config_path)
    base = root / args.base_dir_name
    base.mkdir(parents=True, exist_ok=True)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")
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
        lines = _sbatch_header(f"cal_{name}", args.partition, 4, args.mem, args.stage1_hours, f"{ljobs}/logs") + [
            f"# {n} candidate tile(s): screen -> boundary lines -> SFINCS mask pre-flight",
            f"{GFM_PY} select_calibration_tiles.py screen {common} --tile-ids-file {shard} --out-ids {screened}",
            f"{GFM_PY} build_station_boundary_lines.py {common} --tiles-dir-name model_outputs "
            f"--tile-ids-file {screened} --workers 4 --skip-done --plot all",
            f"{HYDROMT_PY} preflight_sfincs_mask.py {common} --tiles-dir-name model_outputs "
            f"--tile-ids-file {screened} --skip-done",
            ""]
        (jobs_dir / f"{name}.sbatch").write_text("\n".join(lines), newline="\n")

    bridge = _sbatch_header("cal_bridge", args.partition, 4, args.mem, 2, f"{ljobs}/logs") + [
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
          f"submit on Hydrax with: bash {ljobs}/submit_chain.sh")


def cmd_stage2(args) -> None:
    """Write the stage-2 final-run sbatch jobs, and submit them if requested.

    Distributes tiles without a `calibration_run.done` marker over `--n-jobs` jobs
    using longest-processing-time-first on a per-tile runtime estimate derived from
    the pre-flight active-cell count, then writes one sbatch script and tile list
    per job plus submit_all.sh.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (config, base_dir_name, n_jobs, partition, mem,
        overhead_min, sfincs_min_per_mcell, eikonal_min_per_mcell, time_margin,
        submit).
    """
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
        lines = _sbatch_header(f"cal_{name}", args.partition, 4, args.mem, hours, f"{lout}/logs") + [
            f"# {len(tids)} tile(s), estimated {load[j] / 60:.1f} h (generate_calibration_jobs.py stage2)",
            "while read -r TILE; do",
            '  [ -n "$TILE" ] || continue',
            f'  BASE_DIR_NAME={args.base_dir_name} SFINCS_THREADS=4 bash {LINUX_CODE_ROOT}/sfincs_tiles/run_final_tile.sh "$TILE" </dev/null',
            f"done < {lout}/tiles_{j:03d}.txt", ""]
        (out_dir / f"{name}.sbatch").write_text("\n".join(lines), newline="\n")
    (out_dir / "submit_all.sh").write_text(
        "#!/bin/bash\n" + "".join(f"sbatch {lout}/{n}.sbatch\n" for n in names), newline="\n")
    loads = sorted(load.values())
    print(f"{len(tiles)} tile(s) to run ({n_done} already done) over {len(jobs)} job(s) on {args.partition}")
    print(f"estimated hours per job: {loads[0] / 60:.1f}-{loads[-1] / 60:.1f} (x{args.time_margin:g} margin for --time; "
          f"total {sum(loads) / 60:.0f} h)")
    if args.submit:
        for n in names:
            subprocess.run(["sbatch", f"{lout}/{n}.sbatch"], check=True)
        print(f"submitted {len(names)} stage-2 job(s)")
    else:
        print(f"submit with: bash {lout}/submit_all.sh")


def main() -> None:
    """Parse CLI arguments and dispatch to `cmd_stage1` or `cmd_stage2`."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["stage1", "stage2"])
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--n-jobs", type=int, default=None, help="default: 10 for stage1, 20 for stage2")
    parser.add_argument("--stage2-jobs", type=int, default=20, help="stage1: forwarded to the bridge's stage2 call")
    parser.add_argument("--n-tiles", type=int, default=1000)
    parser.add_argument("--partition", default="4vcpu")
    parser.add_argument("--mem", default="30G")
    parser.add_argument("--stage1-hours", type=int, default=24)
    parser.add_argument("--overhead-min", type=float, default=35.0)
    parser.add_argument("--sfincs-min-per-mcell", type=float, default=12.5)
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
