"""Runs postprocess_tile_summary.py for every tile in a validation batch's
tile_ids.txt, one at a time on this machine - (re-)writes each tile's
outputs/summary.json from whichever bathtub/eikonal/SFINCS rasters
currently exist on disk for it, combining outputs that may have been
produced by separate runs (e.g. bathtub+SFINCS on the HPC, eikonal locally
via run_models_sequential.py) into one merged summary per tile.

postprocess_tile_summary.py degrades any missing model's own stats to null
per its own per-model `path.exists()` checks, so it is safe to run against
every tile regardless of how complete that tile's own bathtub/eikonal/sfincs
set is - this driver only skips a tile that has no inputs/mask.tif at all
(never reached step 1 of the per-tile pipeline). Always re-writes (no
skip-if-exists check) - summary.json is cheap to rebuild and is meant to
always reflect the current rasters, not a stale prior run.

Usage:
    python postprocess_sequential.py --base-dir-name validation_sfincs_v4
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"P:\11212688-004-global-floodmaps\modelling")
DEFAULT_PYTHON = r"C:\Users\schlumbe\AppData\Local\miniforge3\envs\hydromt-sfincs-dev\python.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v4 - created by select_validation_tiles.py")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tiles (smoke test)")
    args = parser.parse_args()

    base_dir = DATA_ROOT / args.base_dir_name
    tile_ids_path = base_dir / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]
    if args.limit:
        tile_ids = tile_ids[: args.limit]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_path}")

    script = REPO_ROOT / "sfincs_tiles" / "postprocess_tile_summary.py"
    fail_log = base_dir / "postprocess_local_failures.txt"

    n_ok = n_skip_no_inputs = n_fail = 0
    t_start = time.time()

    for i, tile_id in enumerate(tile_ids, 1):
        tile_dir = base_dir / tile_id
        if not (tile_dir / "inputs" / "mask.tif").exists():
            n_skip_no_inputs += 1
            continue

        t0 = time.time()
        result = subprocess.run(
            [args.python, str(script), "--tile-id", tile_id, "--base-dir-name", args.base_dir_name],
            cwd=str(REPO_ROOT / "sfincs_tiles"),
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        if result.returncode == 0:
            n_ok += 1
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: done ({dt:.1f}s)", flush=True)
        else:
            n_fail += 1
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: FAILED (exit {result.returncode})", flush=True)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{tile_id}\n{result.stderr[-2000:]}\n---\n")

    dt_total = time.time() - t_start
    print(
        f"\nDone in {dt_total / 60:.1f} min: {n_ok} run, {n_skip_no_inputs} skipped (no inputs/mask.tif), "
        f"{n_fail} failed" + (f" (see {fail_log})" if n_fail else "")
    )


if __name__ == "__main__":
    main()
