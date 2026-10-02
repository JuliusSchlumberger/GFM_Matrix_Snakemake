"""Build and run SFINCS for a list of GFM tiles, end to end.

Runs the full sfincs_tiles/ pipeline (build_elevation.py ->
build_roughness.py -> build_boundary_forcing.py -> build_sfincs_tile.py ->
run_sfincs_tile.py) for each tile ID given, entirely under one
environment/interpreter (whichever `python` launches this orchestrator,
e.g. hydromt-sfincs-dev).

Continues to the next tile if one fails, then prints a final per-tile
PASS/FAIL summary rather than stopping the whole batch on the first
failure. Python equivalent of run_sfincs_tiles.ps1.

Requires setup_batch_inputs.py to have been run first for this batch, to
populate each tile's {base_dir_name}/{tile_id}/inputs/. By default also
regenerates dem.tif/mask.tif per tile with the current
extract_dem/extract_dem_mask logic (regenerate_dem_mask.py) as a
non-fatal first step, under --gfm-python since that step needs the
gfm_python_preprocessing env; skip it with --skip-dem-regen to use the
plain copied dem.tif/mask.tif instead.

Usage:
    python run_sfincs_tiles.py --tile-ids 1573 929 851 --base-dir-name validation_sfincs_v5
    python run_sfincs_tiles.py --tile-ids 1573 --base-dir-name validation_sfincs_v5 --sfincs-exe "C:\\path\\to\\sfincs.exe"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

_DEFAULT_SFINCS_EXE = (
    r"C:\Users\schlumbe\hydromt_sfincs\delta_model\software"
    r"\SFINCS_v2.3.0_mt_Faber_release_exe\sfincs.exe"
)
_DEFAULT_CONFIG = str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
_DEFAULT_GFM_PYTHON = r"C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe"


def _steps(sfincs_exe: str, timeout_s: float, build_only: bool, gfm_python: str, skip_dem_regen: bool) -> list[dict]:
    """One (label, script, extra_args, interpreter, non_fatal) step per
    pipeline stage, in the order every tile runs them. Every step runs under
    sys.executable except "regenerate dem/mask", which uses its own explicit
    interpreter (gfm_python) since it imports src/config_utils.py.

    build_only=True drops the "run SFINCS + postprocess" step, for building
    a batch of tiles' sfincs_model/ dirs locally without running the
    simulation here."""
    steps = []
    if not skip_dem_regen:
        steps.append({
            "label": "regenerate dem/mask", "script": "regenerate_dem_mask.py",
            "interpreter": gfm_python, "non_fatal": True,
        })
    steps += [
        {"label": "prep: elevation", "script": "build_elevation.py"},
        {"label": "prep: roughness", "script": "build_roughness.py"},
        {"label": "prep: boundary forcing", "script": "build_boundary_forcing.py"},
        {"label": "build SFINCS model", "script": "build_sfincs_tile.py"},
    ]
    if not build_only:
        steps.append({
            "label": "run SFINCS + postprocess", "script": "run_sfincs_tile.py",
            "extra_args": ["--sfincs-exe", sfincs_exe, "--timeout-s", str(timeout_s)],
        })
    return steps


def run_tile(tile_id: int, config_path: str, base_dir_name: str, steps: list[dict]) -> tuple[bool, str | None]:
    """Runs every step for one tile in order; stops at the first failing
    step unless it's marked non_fatal.

    Returns (passed, failed_step_label_or_None); failed_step_label is set
    even for a non-fatal failure, purely informational.
    """
    for step in steps:
        interpreter = step.get("interpreter", sys.executable)
        args = [interpreter, str(_SCRIPT_DIR / step["script"]), "--tile-id", str(tile_id),
                "--config", config_path, "--base-dir-name", base_dir_name]
        args += step.get("extra_args", [])

        print(f"\n--- [{step['label']}] tile {tile_id} ---")
        t0 = time.monotonic()
        result = subprocess.run(args)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            if step.get("non_fatal"):
                print(f"non-fatal FAILURE: [{step['label']}] tile {tile_id} exited with code {result.returncode} "
                      f"after {elapsed:.1f}s - continuing (falling back to whatever dem.tif/mask.tif already exist)")
                continue
            print(f"FAILED: [{step['label']}] tile {tile_id} exited with code {result.returncode} after {elapsed:.1f}s")
            return False, step["label"]
        print(f"OK: [{step['label']}] tile {tile_id} ({elapsed:.1f}s)")

    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-ids", type=int, nargs="+", default=None, help="one or more tile IDs, e.g. --tile-ids 1573 929 851")
    parser.add_argument("--tile-ids-file", default=None, help="text file, one tile ID per line (alternative to --tile-ids, for a large batch)")
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v5 - every step writes/reads {base-dir-name}/{tile_id}/... . "
                         "Run setup_batch_inputs.py --base-dir-name <name> first to populate each tile's inputs/.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--sfincs-exe", default=_DEFAULT_SFINCS_EXE)
    parser.add_argument("--timeout-s", type=float, default=14400.0,
                         help="per-tile SFINCS wall-clock timeout - matches run_one_tile.sh's own "
                              "SFINCS_TIMEOUT_S (14400s/4h) so large tiles aren't spuriously killed early")
    parser.add_argument("--gfm-python", default=_DEFAULT_GFM_PYTHON,
                         help="interpreter for the one step (regenerate dem/mask) that needs the "
                              "gfm_python_preprocessing env instead of whichever env launched this script")
    parser.add_argument("--skip-dem-regen", action="store_true",
                         help="skip the 'regenerate dem/mask' step entirely and use the plain "
                              "model_outputs/-copied dem.tif/mask.tif as-is")
    parser.add_argument(
        "--build-only", action="store_true",
        help="only run the prep+build steps (no local SFINCS simulation/postprocess) - "
             "for building a batch of tiles' sfincs_model/ dirs ahead of running them "
             "elsewhere, e.g. run_one_tile.sh's own HPC batch dispatch.",
    )
    args = parser.parse_args()
    if not args.tile_ids and not args.tile_ids_file:
        parser.error("one of --tile-ids or --tile-ids-file is required")
    tile_ids = args.tile_ids or [
        int(line.strip()) for line in Path(args.tile_ids_file).read_text().splitlines() if line.strip()
    ]

    steps = _steps(args.sfincs_exe, args.timeout_s, args.build_only, args.gfm_python, args.skip_dem_regen)

    results: dict[int, tuple[bool, str | None]] = {}
    for tile_id in tile_ids:
        print(f"\n{'=' * 50}\n=== Tile {tile_id} ===\n{'=' * 50}")
        results[tile_id] = run_tile(tile_id, args.config, args.base_dir_name, steps)

    print(f"\n{'=' * 50}\n=== Summary ===\n{'=' * 50}")
    any_failed = False
    for tile_id, (passed, failed_step) in results.items():
        status = "PASSED" if passed else f"FAILED at '{failed_step}'"
        any_failed = any_failed or not passed
        print(f"tile {tile_id:<8} {status}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
