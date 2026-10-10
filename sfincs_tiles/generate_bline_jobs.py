"""Writes sbatch scripts that build + run every suitable calibration tile's
SFINCS model on its station boundary line into a separate base dir
(run_bline_tile.sh per tile - see that script for the steps), with the same
downstream summaries/metrics as the smoothed-forcing rerun.

Tiles: <src>/boundary_lines/suitable_tiles.txt (build_station_boundary_lines.py
--summarize), or --tile-ids-file; tiles whose outputs/bline_run.done exists
in the target base dir are left out, so re-running this generator after a
partial run only schedules what's left. Spread over --n-jobs jobs by
longest-processing-time-first on each tile's previous SFINCS wall time (source
tile's sfincs.log "Total time", run at 4 threads - an upper bound, the
boundary-line domain is smaller) / --speedup + a fixed per-tile overhead
(model build incl. subgrid table, staging, postprocessing, cache rebuild).
Each job's --time is its own estimate x --time-margin, rounded up to whole hours.

Writes <root>/<base-dir-name>/tile_ids.txt (every tile scheduled so far, for
run_calibration_sweep_analysis.py) and <root>/<base-dir-name>/hpc_jobs_bline/:
one bline_NNN.sbatch + tiles_NNN.txt per job, submit_all.sh, logs/.

Usage:
    python generate_bline_jobs.py --n-jobs 12
"""

from __future__ import annotations

import argparse
import heapq
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

LINUX_DATA_ROOT = "/p/11212688-004-global-floodmaps/modelling"
LINUX_CODE_ROOT = "/u/schlumbe/gfm_code"
DONE_MARKER = "bline_run.done"
FALLBACK_MINUTES_4T = 30.0  # previous runtime unknown


def previous_runtime_min(sfincs_dir: Path) -> float:
    for name in ("sfincs.log", "sfincs_origforcing.log"):
        p = sfincs_dir / name
        if p.exists():
            m = re.search(r"Total time\s*:\s*([\d.]+)", p.read_text(errors="ignore"))
            if m:
                return float(m.group(1)) / 60.0
    return FALLBACK_MINUTES_4T


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--src-base-dir-name", default="sfincs_calibration")
    parser.add_argument("--base-dir-name", default="sfincs_calibration_bline")
    parser.add_argument("--tile-ids-file", default=None, help="default: <src>/boundary_lines/suitable_tiles.txt")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--partition", default="4vcpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mem", default="30G")
    parser.add_argument("--speedup", type=float, default=1.0, help="assumed SFINCS speedup at --threads vs the original 4")
    parser.add_argument("--overhead-min", type=float, default=25.0,
                        help="per-tile model build (subgrid table) + staging + postprocessing + cache rebuild")
    parser.add_argument("--time-margin", type=float, default=1.5)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    src, base = root / args.src_base_dir_name, root / args.base_dir_name
    ids_file = Path(args.tile_ids_file) if args.tile_ids_file else src / "boundary_lines" / "suitable_tiles.txt"
    ids = [t.strip() for t in ids_file.read_text().split() if t.strip()]

    tiles, skipped_done = [], 0
    for tid in ids:
        if (base / tid / "outputs" / DONE_MARKER).exists():
            skipped_done += 1
            continue
        est = previous_runtime_min(src / tid / "sfincs_model") / args.speedup + args.overhead_min
        tiles.append((tid, est))
    base.mkdir(parents=True, exist_ok=True)
    existing = set((base / "tile_ids.txt").read_text().split()) if (base / "tile_ids.txt").exists() else set()
    (base / "tile_ids.txt").write_text("".join(f"{t}\n" for t in sorted(existing | set(ids), key=int)), newline="\n")
    if not tiles:
        print(f"Nothing to do: {skipped_done} tile(s) already done.")
        return

    # longest-processing-time-first onto the currently least-loaded job
    heap = [(0.0, j) for j in range(min(args.n_jobs, len(tiles)))]
    jobs: dict[int, list[str]] = {j: [] for _, j in heap}
    load = {j: 0.0 for _, j in heap}
    for tid, est in sorted(tiles, key=lambda x: -x[1]):
        total, j = heapq.heappop(heap)
        jobs[j].append(tid)
        load[j] = total + est
        heapq.heappush(heap, (load[j], j))

    out_dir = base / "hpc_jobs_bline"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    linux_out = f"{LINUX_DATA_ROOT}/{args.base_dir_name}/hpc_jobs_bline"
    names = []
    for j, tids in sorted(jobs.items()):
        name = f"bline_{j:03d}"
        names.append(name)
        (out_dir / f"tiles_{j:03d}.txt").write_text("\n".join(tids) + "\n", newline="\n")
        hours = max(1, math.ceil(load[j] * args.time_margin / 60.0))
        script = "\n".join([
            "#!/bin/bash",
            f"#SBATCH --job-name=sfincs_{name}",
            f"#SBATCH --partition={args.partition}",
            f"#SBATCH --cpus-per-task={args.threads}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH --time={hours // 24}-{hours % 24:02d}:00:00",
            f"#SBATCH --output={linux_out}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_out}/logs/{name}_%j.err",
            "",
            f"# {len(tids)} tile(s), estimated {load[j] / 60:.1f} h (boundary-line SFINCS build + run,",
            "# see sfincs_tiles/generate_bline_jobs.py)",
            "set -uo pipefail",
            "while read -r TILE; do",
            '  [ -n "$TILE" ] || continue',
            f"  BASE_DIR_NAME={args.base_dir_name} SRC_BASE_DIR_NAME={args.src_base_dir_name} "
            f'SFINCS_THREADS={args.threads} bash {LINUX_CODE_ROOT}/sfincs_tiles/run_bline_tile.sh "$TILE"',
            f"done < {linux_out}/tiles_{j:03d}.txt",
            "",
        ])
        (out_dir / f"{name}.sbatch").write_text(script, newline="\n")
    (out_dir / "submit_all.sh").write_text(
        "#!/bin/bash\n" + "".join(f"sbatch {linux_out}/{n}.sbatch\n" for n in names), newline="\n")

    loads = sorted(load.values())
    print(f"{len(tiles)} tile(s) to run ({skipped_done} already done) over {len(jobs)} job(s) on {args.partition}, "
          f"{args.threads} threads each")
    print(f"estimated hours per job: {loads[0] / 60:.1f}-{loads[-1] / 60:.1f} "
          f"(total {sum(loads) / 60:.0f} h, {args.speedup:g}x speedup + {args.overhead_min:g} min/tile)")
    print(f"wrote {out_dir} (submit with: bash {linux_out}/submit_all.sh)")


if __name__ == "__main__":
    main()
