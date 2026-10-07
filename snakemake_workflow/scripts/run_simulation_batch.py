"""Runs one production simulation HPC batch: a flat list of (tile_id,
return_period, waterlevel_name) triples, up to --workers of them
CONCURRENTLY via a thread pool of subprocess.run calls - one subprocess per
triple, each a full `run_aqueduct_cli.py` invocation (keeps every skip/
idempotency check - already-done output, no-stations zero-write, OOM marking
- in that one script, rather than duplicating it here).

Why this exists (2026-10-08): `generate_aqueduct_jobs.py`'s own generated
sbatch scripts used to run every (tile, rp, slr) triple in a batch through a
single bash `run_job` call, SEQUENTIALLY, one after another - `cpus_per_task`
was requested from SLURM but never actually used for simulation throughput
(see hpc.md's own comment: "each node runs one tile at a time regardless of
cpus_per_task"). That's fine for a `1vcpu` batch, but makes a bigger
allocation (e.g. `24vcpu`, requested specifically to run many tiles
CONCURRENTLY on one node for a hop_distance wave with heavy node contention)
pure waste - 23 of 24 cores sit idle the whole time. This script is what
`generate_wave_dispatch` now calls instead, genuinely using the node's
`cpus_per_task` the same way `sfincs_tiles/run_friction_sweep_batch.py`
already does for the SFINCS friction sweep: each eikonal solve is
single-threaded numba (src/eikonal.py has no parallel=True/prange anywhere),
so N concurrent solves on an N-cpu node is a real use of the allocation, not
oversubscription - one implementation idea, two separate scripts, since they
wrap different underlying runners (`run_aqueduct_cli.py` vs
`run_eikonal_on_sfincs_subgrid.py`) with different per-item argument shapes.

One triple's failure (non-zero exit) is logged to FAIL_LOG and does NOT
abort the rest of the batch - same "set -uo pipefail, not -e" philosophy
the old sequential `run_job` loop already used for one node's own batch of
otherwise-independent work items.

Usage:
    python run_simulation_batch.py --triples-file batch_000_triples.csv \\
        --config resolved_config.yml --workers 24 --fail-log failures.txt
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "run_aqueduct_cli.py"


def _run_one(python_exe: str, config_path: str, tile_id: str, rp: str, slr: str):
    cmd = [
        python_exe, str(RUNNER),
        "--config", config_path,
        "--tile-id", tile_id, "--return-period", rp, "--waterlevel-name", slr,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return tile_id, rp, slr, result.returncode, elapsed, result.stdout.strip(), result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples-file", required=True, help="CSV, no header: tile_id,return_period,waterlevel_name per line")
    parser.add_argument("--config", required=True, help="path to a fully-resolved config.yml (e.g. resolved_config.yml)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fail-log", default=None, help="path to append 'tile_id rp slr' lines for any failed triple")
    args = parser.parse_args()

    triples: list[tuple[str, str, str]] = []
    with open(args.triples_file, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            triples.append((row[0], row[1], row[2]))
    print(f"{len(triples)} (tile_id, return_period, waterlevel_name) triple(s) in this batch, {args.workers} worker(s)", flush=True)

    n_ok = n_fail = 0
    t_batch0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, args.python, args.config, tile_id, rp, slr)
            for tile_id, rp, slr in triples
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            tile_id, rp, slr, code, elapsed, out, err = fut.result()
            status = "OK" if code == 0 else "FAIL"
            if code == 0:
                n_ok += 1
            else:
                n_fail += 1
                if args.fail_log:
                    with open(args.fail_log, "a", encoding="utf-8") as f:
                        f.write(f"{tile_id} {rp} {slr}\n")
            last_line = out.splitlines()[-1] if out else ""
            print(f"[{i}/{len(triples)}] [{status}] tile={tile_id} rp={rp} slr={slr} ({elapsed:.0f}s): {last_line}", flush=True)
            if code != 0:
                err_line = err.splitlines()[-1] if err else "(empty stderr)"
                print(f"  stderr: {err_line}", flush=True)

    print(f"\nBatch done in {time.time() - t_batch0:.0f}s: {n_ok} ok, {n_fail} failed, {len(triples)} total", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
