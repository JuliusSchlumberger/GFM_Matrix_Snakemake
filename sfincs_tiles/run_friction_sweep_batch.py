"""Runs one friction-sweep HPC batch: a flat list of (tile_id,
friction_scale_factor) pairs (generate_friction_sweep_jobs.py), up to
--workers of them CONCURRENTLY via a thread pool of subprocess.run calls -
one subprocess per pair, each a full `run_eikonal_on_sfincs_subgrid.py
--friction-scale-factor ...` invocation (keeps every skip/idempotency check
- already-done output, missing/empty boundaries - in that one script,
rather than duplicating it here).

Genuine use of the node's cpus_per_task, not oversubscription: each
eikonal solve is single-threaded numba (src/eikonal.py has no
parallel=True/prange anywhere), so N concurrent solves on an N-cpu node
doesn't contend for the same cores the way N copies of an
internally-multithreaded solver would.

One pair's failure (non-zero exit) is logged and does NOT abort the rest
of the batch - same "set -uo pipefail, not -e" philosophy
generate_validation_batch_jobs.py's sbatch scripts already use for a batch
of otherwise-independent work items.

Usage:
    python run_friction_sweep_batch.py --pairs-file batch_000_pairs.csv \\
        --base-dir-name validation_sfincs_v5 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_friction_sweep_metrics import MAX_OUTER_ITERATIONS_DEFAULT  # noqa: E402
from gfm_config import read_root  # noqa: E402
from tile_sweep_cache import build_tile_sweep_cache, write_cache  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "run_eikonal_on_sfincs_subgrid.py"


def _run_one(
    python_exe: str, config_path: str | None, base_dir_name: str, tile_id: str, fsf: float,
    max_outer_iterations: int | None = None,
):
    cmd = [
        python_exe, str(RUNNER),
        "--tile-id", str(tile_id), "--base-dir-name", base_dir_name,
        "--models", "eikonal", "--friction-scale-factor", str(fsf),
    ]
    if config_path:
        cmd += ["--config", config_path]
    if max_outer_iterations is not None:
        cmd += ["--max-outer-iterations", str(max_outer_iterations)]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return tile_id, fsf, result.returncode, elapsed, result.stdout.strip(), result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-file", required=True, help="CSV, no header: tile_id,friction_scale_factor per line")
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--max-outer-iterations", type=int, default=None,
        help="forwarded verbatim to every run_eikonal_on_sfincs_subgrid.py call in this batch "
             "(default: that script's own default, 4, matching production) - see its own "
             "--max-outer-iterations help text for the output-filename tagging this implies.",
    )
    args = parser.parse_args()

    pairs: list[tuple[str, float]] = []
    with open(args.pairs_file, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            pairs.append((row[0], float(row[1])))
    print(f"{len(pairs)} (tile_id, friction_scale_factor) pair(s) in this batch, {args.workers} worker(s)", flush=True)

    n_ok = n_fail = 0
    t_batch0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, args.python, args.config, args.base_dir_name, tile_id, fsf, args.max_outer_iterations)
            for tile_id, fsf in pairs
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            tile_id, fsf, code, elapsed, out, err = fut.result()
            status = "OK" if code == 0 else "FAIL"
            if code == 0:
                n_ok += 1
            else:
                n_fail += 1
            last_line = out.splitlines()[-1] if out else ""
            print(f"[{i}/{len(pairs)}] [{status}] tile={tile_id} fsf={fsf:g} ({elapsed:.0f}s): {last_line}", flush=True)
            if code != 0:
                err_line = err.splitlines()[-1] if err else "(empty stderr)"
                print(f"  stderr: {err_line}", flush=True)

    print(f"\nBatch done in {time.time() - t_batch0:.0f}s: {n_ok} ok, {n_fail} failed, {len(pairs)} total", flush=True)

    # -- build each distinct tile's sweep-vs-SFINCS(-and-bathtub) comparison cache,
    # right here, right after ALL of this batch's own sweep points are done - "the
    # required postprocess ... done as part of the main pipeline too, so that the
    # postprocessing analysis actually is very quick" (2026-10-07). Opens SFINCS
    # once per tile (tile_sweep_cache.load_tile_sfincs), not once per (tile, fsf)
    # pair - see that module's own docstring. --
    if args.config is None:
        print("\nSkipping sweep-comparison cache build - no --config given, can't resolve paths.root", flush=True)
        return
    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tiles_by_fsf: dict[str, list[float]] = {}
    for tile_id, fsf in pairs:
        tiles_by_fsf.setdefault(tile_id, []).append(fsf)
    distinct_tile_ids = sorted(tiles_by_fsf, key=lambda t: int(t) if t.isdigit() else t)

    # None here means "run_eikonal_on_sfincs_subgrid.py's own default" (see its --max-outer-iterations
    # forwarding above) - resolve it to that same concrete value so _eikonal_path's own outer-tag
    # matches the files actually on disk instead of comparing None against an int.
    max_outer_iterations = args.max_outer_iterations if args.max_outer_iterations is not None else MAX_OUTER_ITERATIONS_DEFAULT

    print(f"\nBuilding sweep-comparison cache for {len(distinct_tile_ids)} distinct tile(s) in this batch...", flush=True)
    n_cached = 0
    for i, tile_id in enumerate(distinct_tile_ids, start=1):
        cache = build_tile_sweep_cache(base_dir / tile_id, sorted(set(tiles_by_fsf[tile_id])), max_outer_iterations)
        if cache is not None:
            write_cache(base_dir / tile_id, cache)
            n_cached += 1
        if i % 20 == 0 or i == len(distinct_tile_ids):
            print(f"  [{i}/{len(distinct_tile_ids)}] {n_cached} cache(s) written so far", flush=True)
    print(f"Wrote {n_cached}/{len(distinct_tile_ids)} sweep-comparison cache(s) "
          f"({len(distinct_tile_ids) - n_cached} skipped - SFINCS not built yet for that tile)", flush=True)


if __name__ == "__main__":
    main()
