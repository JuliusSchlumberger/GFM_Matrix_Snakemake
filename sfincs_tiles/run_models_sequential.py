"""Runs run_eikonal_on_sfincs_subgrid.py (bathtub and/or eikonal) for every
tile in a validation batch's tile_ids.txt, one at a time on this machine -
no HPC, no parallelism. Local counterpart to the bathtub+eikonal leg of
run_one_tile.sh's own per-tile pipeline.

Each requested model writes its own independent output file
(outputs/bathtub_waterdepth_{RP}_{SLR}.tif, outputs/
eikonal_on_subgrid_waterdepth_{RP}_{SLR}.tif), so --models bathtub and
--models eikonal can be run as separate invocations.
postprocess_tile_summary.py later combines whichever outputs exist into one
summary.json per tile.

Idempotent: skips a tile whose requested model output(s) already exist, so
interrupting (Ctrl+C) and re-running resumes where it left off. Tiles whose
sfincs_model/ isn't built yet are skipped up front.

Usage:
    python run_models_sequential.py --base-dir-name validation_sfincs_v4 --models eikonal
    python run_models_sequential.py --base-dir-name validation_sfincs_v5 --models bathtub,eikonal --limit 5
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"P:\11212688-004-global-floodmaps\modelling")
DEFAULT_PYTHON = r"C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v4 - created by select_validation_tiles.py")
    parser.add_argument("--models", default="eikonal", help="comma-separated subset of bathtub,eikonal to run "
                         "(default: eikonal - bathtub is cheap enough it's normally run on the HPC batch instead)")
    parser.add_argument("--max-rounds", type=int, default=40,
                         help="forwarded to run_eikonal_on_sfincs_subgrid.py's own --max-rounds, only meaningful "
                              "when eikonal is in --models (default: 40, matching production's "
                              "simulation.flooding.max_rounds)")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                         help="python interpreter to invoke the eikonal script with")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tiles (smoke test)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ("bathtub", "eikonal"):
            parser.error(f"--models: unknown model {m!r} (choose from bathtub, eikonal)")

    base_dir = DATA_ROOT / args.base_dir_name
    tile_ids_path = base_dir / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]
    if args.limit:
        tile_ids = tile_ids[: args.limit]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_path}, models={models}")

    script = REPO_ROOT / "sfincs_tiles" / "run_eikonal_on_sfincs_subgrid.py"
    fail_log = base_dir / "run_models_local_failures.txt"

    n_ok = n_skip_not_built = n_skip_done = n_fail = 0
    run_times: list[float] = []  # actual per-tile solve times only, for the average-pace readout
    t_start = time.time()

    for i, tile_id in enumerate(tile_ids, 1):
        tile_dir = base_dir / tile_id
        sfincs_dir = tile_dir / "sfincs_model"
        if not (sfincs_dir / "elevation_combined_subgrid_src.tif").exists():
            print(f"[{i}/{len(tile_ids)}] tile {tile_id}: SKIP - sfincs_model not built yet", flush=True)
            n_skip_not_built += 1
            continue

        outputs_by_model = {
            "bathtub": tile_dir / "outputs" / "bathtub_waterdepth_RP100_SLR_0.tif",
            "eikonal": tile_dir / "outputs" / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif",
        }
        if all(outputs_by_model[m].exists() for m in models):
            n_skip_done += 1
            continue

        cmd = [args.python, str(script), "--tile-id", tile_id, "--base-dir-name", args.base_dir_name,
               "--models", *models]
        if "eikonal" in models and args.max_rounds is not None:
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
