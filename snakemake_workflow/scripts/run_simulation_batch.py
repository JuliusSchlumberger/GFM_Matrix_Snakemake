"""Runs one production simulation HPC batch: a flat list of (tile_id,
return_period, waterlevel_name) triples, up to --workers of them
CONCURRENTLY via a thread pool of subprocess.run calls - one subprocess per
triple, each a full `run_eikonal_cli.py` invocation (keeps every skip/
idempotency check - already-done output, no-stations zero-write, OOM marking
- in that one script, rather than duplicating it here).

Why this exists (2026-10-08): `generate_eikonal_jobs.py`'s own generated
sbatch scripts used to run every (tile, rp, slr) triple in a batch through a
single bash `run_job` call, SEQUENTIALLY, one after another - `cpus_per_task`
was requested from SLURM but never actually used for simulation throughput
(see hpc.md's own comment: "each node runs one tile at a time regardless of
cpus_per_task"). That's fine for a `1vcpu` batch, but makes a bigger
allocation (e.g. `24vcpu`, requested specifically to run many tiles
CONCURRENTLY on one node for a hop_distance wave with heavy node contention)
pure waste - 23 of 24 cores sit idle the whole time. This script is what
`generate_wave_dispatch` now calls instead, genuinely using the node's
`cpus_per_task`: each eikonal solve is single-threaded numba (src/eikonal.py
has no parallel=True/prange anywhere), so N concurrent solves on an N-cpu
node is a real use of the allocation, not oversubscription.

One triple's failure (non-zero exit) is logged to FAIL_LOG and does NOT
abort the rest of the batch - same "set -uo pipefail, not -e" philosophy
the old sequential `run_job` loop already used for one node's own batch of
otherwise-independent work items.

Hard per-triple timeout (2026-10-13, --timeout-min, default 20): confirmed
live that a P:\\ mount hang on a compute node (not reproducible from the
login node or from other machines - looks specific to the 24vcpu partition,
reproduced on two separate, otherwise-healthy job submissions) can block
run_eikonal_cli.py's own very first, unprotected `open(args.config)` call
indefinitely - every worker thread hangs on the exact same read
simultaneously, nothing ever raises, so retry_transient_io (which only
retries real exceptions) never gets a chance to help, and the whole batch
goes silent for hours with zero diagnostic signal. subprocess.run(timeout=)
turns that silent hang into a clean, logged failure instead - a timed-out
triple still gets picked up correctly on the NEXT resume dispatch (its
output file never got written, so run_eikonal_cli.py's own idempotency
check won't skip it), same as any other genuine failure.

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

RUNNER = Path(__file__).resolve().parent / "run_eikonal_cli.py"


def _run_one(python_exe: str, config_path: str, tile_id: str, rp: str, slr: str, timeout_s: float):
    cmd = [
        python_exe, str(RUNNER),
        "--config", config_path,
        "--tile-id", tile_id, "--return-period", rp, "--waterlevel-name", slr,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        return tile_id, rp, slr, result.returncode, elapsed, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        out = (exc.stdout or b"").decode("utf-8", "replace").strip() if isinstance(exc.stdout, bytes) else (exc.stdout or "").strip()
        return (tile_id, rp, slr, -1, elapsed, out,
                f"TIMEOUT after {timeout_s:.0f}s (likely a hung P:\\ read on this node - see module docstring)")


def main() -> None:
    """Parse CLI arguments and run this batch's simulation triples concurrently.

    Reads the triples CSV and fans the batch out across a thread pool of
    ``run_eikonal_cli.py`` subprocess calls (one per triple), honoring
    ``--workers`` and a hard per-triple ``--timeout-min``. A failed or
    timed-out triple is logged (stdout) and, if ``--fail-log`` is given,
    appended to that file; it does not abort the rest of the batch. Exits
    with status 1 if any triple failed.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples-file", required=True, help="CSV, no header: tile_id,return_period,waterlevel_name per line")
    parser.add_argument("--config", required=True, help="path to a fully-resolved config.yml (e.g. resolved_config.yml)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fail-log", default=None, help="path to append 'tile_id rp slr' lines for any failed triple")
    parser.add_argument(
        "--timeout-min", type=float, default=20.0,
        help="hard per-triple wall-clock limit (minutes) - a triple that exceeds this is killed and logged as "
             "failed rather than silently hanging the whole batch (see module docstring). Set well above the "
             "slowest real solve this batch's tiles need, not just the median case.",
    )
    args = parser.parse_args()

    triples: list[tuple[str, str, str]] = []
    with open(args.triples_file, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            triples.append((row[0], row[1], row[2]))
    print(f"{len(triples)} (tile_id, return_period, waterlevel_name) triple(s) in this batch, {args.workers} worker(s)", flush=True)

    n_ok = n_fail = n_timeout = 0
    timeout_s = args.timeout_min * 60.0
    t_batch0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, args.python, args.config, tile_id, rp, slr, timeout_s)
            for tile_id, rp, slr in triples
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            tile_id, rp, slr, code, elapsed, out, err = fut.result()
            status = "OK" if code == 0 else ("TIMEOUT" if code == -1 else "FAIL")
            if code == 0:
                n_ok += 1
            else:
                n_fail += 1
                if code == -1:
                    n_timeout += 1
                if args.fail_log:
                    with open(args.fail_log, "a", encoding="utf-8") as f:
                        f.write(f"{tile_id} {rp} {slr}\n")
            last_line = out.splitlines()[-1] if out else ""
            print(f"[{i}/{len(triples)}] [{status}] tile={tile_id} rp={rp} slr={slr} ({elapsed:.0f}s): {last_line}", flush=True)
            if code != 0:
                err_line = err.splitlines()[-1] if err else "(empty stderr)"
                print(f"  stderr: {err_line}", flush=True)

    print(f"\nBatch done in {time.time() - t_batch0:.0f}s: {n_ok} ok, {n_fail} failed "
          f"({n_timeout} of those were timeouts), {len(triples)} total", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
