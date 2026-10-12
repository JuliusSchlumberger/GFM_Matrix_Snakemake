"""Progress of the boundary-line SFINCS runs (generate_bline_jobs.py /
run_bline_tile.sh), read-only, from the per-tile files on disk:

  done        outputs/bline_run.done   (model built, SFINCS run, summaries + cache)
  failed      outputs/bline_run.failed (last line = reason)
  running     the tile dir has its copied boundary_line_<tile>.gpkg, no marker yet -
              with the step it got to: forcing / model built / SFINCS run / postprocessed
  pending     none of the above

Per job (hpc_jobs_bline/tiles_NNN.txt): done/failed/total, the tile currently
running (and its step), and a projected finish - estimated work done so far
(each tile weighted as the job generator does: the source tile's previous
SFINCS runtime + a fixed overhead) over the time since the job's first tile
started (the copied boundary line's mtime).

Usage:
    python check_bline_progress.py [--base-dir-name sfincs_calibration_bline] [--watch 600] [--list-failed]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_bline_jobs import previous_runtime_min  # noqa: E402
from gfm_config import read_root  # noqa: E402

OVERHEAD_MIN = 25.0  # same per-tile overhead generate_bline_jobs.py assumes


def tile_state(td: Path) -> tuple[str, str]:
    out, sm = td / "outputs", td / "sfincs_model"
    if (out / "bline_run.done").exists():
        return "done", ""
    if (out / "bline_run.failed").exists():
        lines = (out / "bline_run.failed").read_text(errors="ignore").strip().splitlines()
        return "failed", lines[-1] if lines else ""
    if (td / f"boundary_line_{td.name}.gpkg").exists():
        if (out / "summary_sfincs.json").exists():
            step = "postprocessed"
        elif (sm / "sfincs_map.nc").exists():
            step = "SFINCS run"
        elif (sm / "sfincs.inp").exists():
            step = "model built"
        else:
            step = "forcing"
        return "running", step
    return "pending", ""


def report(src: Path, base: Path, list_failed: bool) -> None:
    jobs_dir = base / "hpc_jobs_bline"
    job_files = sorted(jobs_dir.glob("tiles_*.txt"))
    if not job_files:
        print(f"no tiles_*.txt in {jobs_dir}")
        return
    now = time.time()
    totals = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    failed = []
    print(f"{datetime.now():%Y-%m-%d %H:%M}  {base}")
    print(f"{'job':>4} {'done':>5} {'fail':>5} {'total':>5}  {'running (step)':>24}  {'work done':>9}  projected finish")
    for jf in job_files:
        tids = [t.strip() for t in jf.read_text().splitlines() if t.strip()]
        n = {"done": 0, "failed": 0, "running": 0, "pending": 0}
        work_total = work_done = 0.0
        running, first_start = "-", None
        for t in tids:
            td = base / t
            est = previous_runtime_min(src / t / "sfincs_model") + OVERHEAD_MIN
            work_total += est
            state, msg = tile_state(td)
            n[state] += 1
            if state in ("done", "failed"):
                work_done += est
            if state == "running":
                running = f"{t} ({msg})"
            if state == "failed":
                failed.append((t, msg))
            line = td / f"boundary_line_{t}.gpkg"  # copied at the start of every tile
            if state != "pending" and line.exists():
                started = line.stat().st_mtime
                first_start = min(first_start or started, started)
        for k in totals:
            totals[k] += n[k]
        frac = work_done / work_total if work_total else 0.0
        if n["done"] + n["failed"] == len(tids):
            eta = "finished"
        elif first_start and work_done > 0:
            rate = work_done / (now - first_start)  # estimated minutes of work per second
            eta = f"{datetime.fromtimestamp(now + (work_total - work_done) / rate):%a %d %b %H:%M}"
        else:
            eta = "not started" if not first_start else "too early to tell"
        print(f"{jf.stem[-3:]:>4} {n['done']:5d} {n['failed']:5d} {len(tids):5d}  {running:>24}  {100 * frac:8.0f}%  {eta}")
    n_all = sum(totals.values())
    print(f"\ntotal: {totals['done']}/{n_all} done, {totals['failed']} failed, {totals['running']} running, "
          f"{totals['pending']} pending")
    if failed:
        print(f"failed tiles ({len(failed)}):" + ("" if list_failed else " (--list-failed for reasons)"))
        if list_failed:
            for t, msg in failed:
                print(f"  {t}: {msg}")
        else:
            print("  " + ", ".join(t for t, _ in failed[:40]) + (" ..." if len(failed) > 40 else ""))


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--src-base-dir-name", default="sfincs_calibration")
    parser.add_argument("--base-dir-name", default="sfincs_calibration_bline")
    parser.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    parser.add_argument("--list-failed", action="store_true")
    args = parser.parse_args()
    root = read_root(Path(args.config))
    while True:
        report(root / args.src_base_dir_name, root / args.base_dir_name, args.list_failed)
        if not args.watch:
            break
        print(f"\n(next update in {args.watch} s)\n")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
