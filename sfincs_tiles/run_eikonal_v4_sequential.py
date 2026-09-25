"""Run the eikonal-only solve (run_eikonal_on_sfincs_subgrid.py --models eikonal)
across every tile in validation_sfincs_v4's tile_ids.txt, one at a time on this
machine - no HPC, no parallelism.

The v4 HPC batch was generated with --runner-extra-args "--models bathtub,sfincs"
(run_one_tile_v3.sh), so this fills in the missing eikonal leg locally instead of
resubmitting a fresh HPC batch for just one model.

Idempotent: run_eikonal_on_sfincs_subgrid.py itself skips a tile whose eikonal
output already exists, so interrupting (Ctrl+C) and re-running this driver just
resumes where it left off. A tile whose sfincs_model/ isn't built yet is skipped
up front rather than left to fail inside the subprocess.

Usage:
    python run_eikonal_v4_sequential.py
    python run_eikonal_v4_sequential.py --max-rounds 40 --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"P:\11212688-004-global-floodmaps\modelling")
BASE_DIR_NAME = "validation_sfincs_v4"
DEFAULT_PYTHON = r"C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rounds", type=int, default=40,
                         help="forwarded to run_eikonal_on_sfincs_subgrid.py's own --max-rounds "
                              "(default: 40, matching production's simulation.flooding.max_rounds)")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                         help="python interpreter to invoke the eikonal script with")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tiles (smoke test)")
    args = parser.parse_args()

    tile_ids_path = DATA_ROOT / BASE_DIR_NAME / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]
    if args.limit:
        tile_ids = tile_ids[: args.limit]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_path}")

    script = REPO_ROOT / "sfincs_tiles" / "run_eikonal_on_sfincs_subgrid.py"
    fail_log = DATA_ROOT / BASE_DIR_NAME / "run_eikonal_local_failures.txt"

    n_ok = n_skip_not_built = n_skip_done = n_fail = 0
    run_times: list[float] = []  # actual per-tile solve times only, for the ETA estimate
    t_start = time.time()

    for i, tile_id in enumerate(tile_ids, 1):
        tile_dir = DATA_ROOT / BASE_DIR_NAME / tile_id
        sfincs_dir = tile_dir / "sfincs_model"
        if not (sfincs_dir / "elevation_combined_subgrid_src.tif").exists():
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: SKIP - sfincs_model not built yet", flush=True)
            n_skip_not_built += 1
            continue

        eikonal_out = tile_dir / "outputs" / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"
        if eikonal_out.exists():
            n_skip_done += 1
            continue

        cmd = [args.python, str(script), "--tile-id", tile_id, "--base-dir-name", BASE_DIR_NAME, "--models", "eikonal"]
        if args.max_rounds is not None:
            cmd += ["--max-rounds", str(args.max_rounds)]

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT / "sfincs_tiles"))
        dt = time.time() - t0
        run_times.append(dt)
        avg_s = sum(run_times) / len(run_times)
        if result.returncode == 0:
            n_ok += 1
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: done ({dt:.1f}s, avg {avg_s:.1f}s/tile so far)", flush=True)
        else:
            n_fail += 1
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: FAILED (exit {result.returncode})", flush=True)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{tile_id}\n")

    dt_total = time.time() - t_start
    print(
        f"\nDone in {dt_total / 60:.1f} min: {n_ok} run, {n_skip_done} already done, "
        f"{n_skip_not_built} skipped (sfincs_model not built), {n_fail} failed"
        + (f" (see {fail_log})" if n_fail else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
