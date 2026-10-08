"""Writes sbatch scripts that re-run every built calibration tile's SFINCS
model with the smoothed nearest-boundary forcing (rerun_smoothed_forcing_tile.sh
per tile - see boundary_forcing_smoothed.py for the method and why).

Every tile with a built model (sfincs_model/sfincs.inp) and no
outputs/rerun_smoothed_forcing.done marker is included, so re-running this
generator after a partial run only schedules what's left. Tiles are spread
over --n-jobs jobs by longest-processing-time-first on each tile's previous
SFINCS wall time (sfincs.log "Total time", run at 4 threads), scaled by
--speedup for --threads threads plus a fixed per-tile overhead (staging the
~1.4 GB subgrid table, postprocessing, cache rebuild). Each job's --time is its
own estimate x --time-margin, rounded up to whole hours.

Writes into <root>/<base-dir-name>/hpc_jobs_smoothed_forcing/: one
rerun_NNN.sbatch + tiles_NNN.txt per job, submit_all.sh, and logs/.

Usage:
    python generate_smoothed_forcing_rerun_jobs.py --base-dir-name sfincs_calibration --n-jobs 15
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
DONE_MARKER = "rerun_smoothed_forcing.done"
FALLBACK_MINUTES_4T = 30.0  # previous runtime unknown (no sfincs.log total time)


def previous_runtime_min(sfincs_dir: Path) -> float:
    for name in ("sfincs_origforcing.log", "sfincs.log"):
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
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--n-jobs", type=int, default=15, help="15 x 16 cores = Hydrax's ~240-core responsible-use ceiling")
    parser.add_argument("--partition", default="16vcpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mem", default="120G")
    parser.add_argument("--speedup", type=float, default=3.0, help="assumed SFINCS speedup at --threads vs the original 4")
    parser.add_argument("--overhead-min", type=float, default=6.0, help="per-tile staging + postprocessing + cache rebuild")
    parser.add_argument("--time-margin", type=float, default=1.5)
    args = parser.parse_args()

    base = read_root(Path(args.config)) / args.base_dir_name
    tiles, skipped_done = [], 0
    for td in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        if not (td / "sfincs_model" / "sfincs.inp").exists():
            continue
        if (td / "outputs" / DONE_MARKER).exists():
            skipped_done += 1
            continue
        est = previous_runtime_min(td / "sfincs_model") / args.speedup + args.overhead_min
        tiles.append((td.name, est))
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

    out_dir = base / "hpc_jobs_smoothed_forcing"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    linux_out = f"{LINUX_DATA_ROOT}/{args.base_dir_name}/hpc_jobs_smoothed_forcing"
    names = []
    for j, tids in sorted(jobs.items()):
        name = f"rerun_{j:03d}"
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
            f"# {len(tids)} tile(s), estimated {load[j] / 60:.1f} h (smoothed-forcing SFINCS rerun,",
            "# see sfincs_tiles/generate_smoothed_forcing_rerun_jobs.py)",
            "set -uo pipefail",
            f'while read -r TILE; do',
            f'  [ -n "$TILE" ] || continue',
            f'  BASE_DIR_NAME={args.base_dir_name} SFINCS_THREADS={args.threads} '
            f'bash {LINUX_CODE_ROOT}/sfincs_tiles/rerun_smoothed_forcing_tile.sh "$TILE"',
            f'done < {linux_out}/tiles_{j:03d}.txt',
            "",
        ])
        (out_dir / f"{name}.sbatch").write_text(script, newline="\n")
    (out_dir / "submit_all.sh").write_text(
        "#!/bin/bash\n" + "".join(f"sbatch {linux_out}/{n}.sbatch\n" for n in names), newline="\n")

    loads = sorted(load.values())
    print(f"{len(tiles)} tile(s) to rerun ({skipped_done} already done) over {len(jobs)} job(s) on {args.partition}, "
          f"{args.threads} threads each")
    print(f"estimated hours per job: {loads[0] / 60:.1f}-{loads[-1] / 60:.1f} "
          f"(total {sum(loads) / 60:.0f} h, assuming {args.speedup:g}x speedup + {args.overhead_min:g} min/tile)")
    print(f"wrote {out_dir} (submit with: bash {linux_out}/submit_all.sh)")


if __name__ == "__main__":
    main()
