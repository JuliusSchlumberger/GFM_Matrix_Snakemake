"""Report progress of the SFINCS calibration chain (generate_calibration_jobs.py).

Usage:
    python check_calibration_progress.py [--base-dir-name calibration_roughness_SFINCS] [--list-failed] [--watch 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402


def _ids(p: Path) -> list[str]:
    return [t.strip() for t in p.read_text().split() if t.strip()] if p.exists() else []


def _json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def stage1(base: Path) -> None:
    """Print stage-1 counts per candidate shard."""
    shards = sorted((base / "candidates").glob("shard_*.txt"))
    if not shards:
        print("stage 1: no candidate shards yet")
        return
    tot = [0] * 7
    print("stage 1   candidates  screened  passed  lines  suitable  preflight  ok")
    for s in shards:
        ids = _ids(s)
        scr = [_json(base / "candidates" / "screen" / f"{t}.json") for t in ids]
        passed = [t for t, r in zip(ids, scr) if r and r.get("ok")]
        st = [_json(base / "boundary_lines" / "status" / f"{t}.json") for t in passed]
        pf = [_json(base / "boundary_lines" / "preflight" / f"{t}.json") for t in passed]
        row = [len(ids), sum(r is not None for r in scr), len(passed), sum(r is not None for r in st),
               sum(bool(r and r.get("suitable")) for r in st), sum(r is not None for r in pf)]
        row.append(sum(bool(r and r.get("ok")) for r in pf))
        tot = [a + b for a, b in zip(tot, row)]
        print(f"{s.stem:9s} {row[0]:10d} {row[1]:9d} {row[2]:7d} {row[3]:6d} {row[4]:9d} {row[5]:10d} {row[6]:3d}")
    print(f"{'total':9s} {tot[0]:10d} {tot[1]:9d} {tot[2]:7d} {tot[3]:6d} {tot[4]:9d} {tot[5]:10d} {tot[6]:3d}")


def stage2(base: Path, list_failed: bool) -> None:
    """Print stage-2 done/failed/running/pending counts per job file."""
    jobs = sorted((base / "hpc_jobs_stage2").glob("tiles_*.txt"))
    print(f"\nbridge: tile_ids.txt {'present (' + str(len(_ids(base / 'tile_ids.txt'))) + ' tiles)' if (base / 'tile_ids.txt').exists() else 'not yet'}, "
          f"stage-2 jobs {'written (' + str(len(jobs)) + ')' if jobs else 'not yet'}")
    if not jobs:
        return
    totals = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    failed = []
    print("stage 2    done  failed  running  pending  total   current")
    for j in jobs:
        n = {"done": 0, "failed": 0, "running": 0, "pending": 0}
        current = ""
        for t in _ids(j):
            out = base / t / "outputs"
            if (out / "calibration_run.done").exists():
                n["done"] += 1
            elif (out / "calibration_run.step").exists():
                n["running"] += 1
                current = f"{t}: {(out / 'calibration_run.step').read_text().strip()}"
            elif (out / "calibration_run.failed").exists():
                n["failed"] += 1
                last = (out / "calibration_run.failed").read_text(errors="ignore").strip().splitlines()
                failed.append((t, last[-1] if last else ""))
            else:
                n["pending"] += 1
        for k in totals:
            totals[k] += n[k]
        print(f"{j.stem.replace('tiles', 'final'):9s} {n['done']:5d} {n['failed']:7d} {n['running']:8d} "
              f"{n['pending']:8d} {sum(n.values()):6d}   {current}")
    print(f"{'total':9s} {totals['done']:5d} {totals['failed']:7d} {totals['running']:8d} {totals['pending']:8d} "
          f"{sum(totals.values()):6d}")
    if list_failed and failed:
        print("\nfailed:")
        for t, msg in failed:
            print(f"  {t}: {msg}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--list-failed", action="store_true")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                        help="re-check and reprint every SECONDS (0 or absent: one report)")
    args = parser.parse_args()
    base = read_root(Path(args.config)) / args.base_dir_name
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        stage1(base)
        stage2(base, args.list_failed)
        if not args.watch:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
