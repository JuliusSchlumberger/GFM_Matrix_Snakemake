"""Progress of the smoothed-forcing SFINCS rerun (generate_smoothed_forcing_rerun_jobs.py
/ rerun_smoothed_forcing_tile.sh), read-only, from the per-tile markers on disk:

  done        outputs/rerun_smoothed_forcing.done  (SFINCS + postprocessing + cache rebuilt)
  failed      outputs/rerun_smoothed_forcing.failed (last line = reason)
  running     backup sfincs_model/sfincs_origforcing.bzs exists, no done marker
  pending     none of the above

Per job (hpc_jobs_smoothed_forcing/tiles_NNN.txt): done/failed/total, the tile
currently running, and a projected finish - estimated work done so far (each
tile weighted by its previous SFINCS runtime, as the job generator does) over
the time since the job's first tile started. Also checks that every done
tile's sweep_comparison_cache.json was rebuilt after its new sfincs_map.nc.

Usage:
    python check_smoothed_forcing_rerun_progress.py [--base-dir-name sfincs_calibration] [--watch 600] [--list-failed]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_smoothed_forcing_rerun_jobs import previous_runtime_min  # noqa: E402
from gfm_config import read_root  # noqa: E402

OVERHEAD_MIN = 6.0  # same per-tile overhead the job generator assumes


def tile_state(td: Path) -> tuple[str, str]:
    out, sm = td / "outputs", td / "sfincs_model"
    if (out / "rerun_smoothed_forcing.done").exists():
        return "done", ""
    if (out / "rerun_smoothed_forcing.failed").exists():
        lines = (out / "rerun_smoothed_forcing.failed").read_text(errors="ignore").strip().splitlines()
        return "failed", lines[-1] if lines else ""
    if (sm / "sfincs_origforcing.bzs").exists():
        return "running", ""
    return "pending", ""


def report(base: Path, list_failed: bool) -> None:
    jobs_dir = base / "hpc_jobs_smoothed_forcing"
    job_files = sorted(jobs_dir.glob("tiles_*.txt"))
    if not job_files:
        print(f"no tiles_*.txt in {jobs_dir}")
        return
    now = time.time()
    totals = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    failed, cache_stale = [], []
    print(f"{datetime.now():%Y-%m-%d %H:%M}  {base}")
    print(f"{'job':>4} {'done':>5} {'fail':>5} {'total':>5}  {'running':>8}  {'work done':>9}  projected finish")
    for jf in job_files:
        tids = [t.strip() for t in jf.read_text().splitlines() if t.strip()]
        n = {"done": 0, "failed": 0, "running": 0, "pending": 0}
        work_total = work_done = 0.0
        running, first_start = "-", None
        for t in tids:
            td = base / t
            est = previous_runtime_min(td / "sfincs_model") + OVERHEAD_MIN
            work_total += est
            state, msg = tile_state(td)
            n[state] += 1
            if state in ("done", "failed"):
                work_done += est
            if state == "running":
                running = t
            if state == "failed":
                failed.append((t, msg))
            if state == "done":
                cache, smap = td / "outputs" / "sweep_comparison_cache.json", td / "sfincs_model" / "sfincs_map.nc"
                if not cache.exists() or (smap.exists() and cache.stat().st_mtime < smap.stat().st_mtime):
                    cache_stale.append(t)
            # Start time = the new sfincs.bnd's mtime (rewritten at the start of every rerun
            # tile) - NOT the backup's: `cp -p` keeps the ORIGINAL file's (weeks-old) mtime.
            if state != "pending" and (td / "sfincs_model" / "sfincs.bnd").exists():
                started = (td / "sfincs_model" / "sfincs.bnd").stat().st_mtime
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
        print(f"{jf.stem[-3:]:>4} {n['done']:5d} {n['failed']:5d} {len(tids):5d}  {running:>8}  {100 * frac:8.0f}%  {eta}")
    n_all = sum(totals.values())
    print(f"\ntotal: {totals['done']}/{n_all} done, {totals['failed']} failed, {totals['running']} running, "
          f"{totals['pending']} pending")
    print(f"caches: {totals['done'] - len(cache_stale)}/{totals['done']} done tiles have a cache rebuilt after their "
          f"new sfincs_map.nc" + (f" - NOT for: {', '.join(cache_stale[:20])}" if cache_stale else ""))
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
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    parser.add_argument("--list-failed", action="store_true")
    args = parser.parse_args()
    base = read_root(Path(args.config)) / args.base_dir_name
    while True:
        report(base, args.list_failed)
        if not args.watch:
            break
        print(f"\n(next update in {args.watch} s)\n")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
