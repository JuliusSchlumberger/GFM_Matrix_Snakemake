"""Build and run SFINCS for a list of GFM tiles, end to end.

Runs the full sfincs_tiles/ pipeline (build_elevation.py ->
build_roughness.py -> build_boundary_forcing.py -> build_sfincs_tile.py ->
run_sfincs_tile.py) for each tile ID given - entirely under ONE
environment/interpreter (whichever `python` you launch this orchestrator
with, e.g. hydromt-sfincs-dev). No `conda run`, no second environment: none
of these scripts import src/config_utils.py (see gfm_config.py's own module
docstring for why - it needs a much older hydromt than hydromt_sfincs
does), so there's nothing left that requires the main pipeline's own
gfm_python_preprocessing env.

Continues to the next tile if one fails (e.g. "no COAST-HG station within
range" is a real, expected drop for some tiles per the plan doc, not
necessarily a bug) - prints a final per-tile PASS/FAIL summary rather than
stopping the whole batch on the first failure. Python equivalent of
run_sfincs_tiles.ps1 - use whichever you prefer running from your own
terminal.

Usage:
    python run_sfincs_tiles.py --tile-ids 1573 929 851
    python run_sfincs_tiles.py --tile-ids 1573 --sfincs-exe "C:\\path\\to\\sfincs.exe"
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


def _steps(sfincs_exe: str, timeout_s: float, build_only: bool) -> list[dict]:
    """One (label, script, extra_args) step per pipeline stage, in the
    exact order every tile must run them - every step now runs under the
    SAME interpreter (sys.executable), so there's no per-step environment
    to choose.

    build_only=True drops the "run SFINCS + postprocess" step entirely -
    for building a large batch of tiles' sfincs_model/ dirs locally so
    they're ready for generate_sfincs_hpc_jobs.py's own HPC batch dispatch
    to actually run them, without also running (and therefore waiting on)
    the simulation itself here."""
    steps = [
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


def run_tile(tile_id: int, config_path: str, steps: list[dict]) -> tuple[bool, str | None]:
    """Runs every step for one tile in order; stops at the first failing
    step (real cross-step dependency - build_sfincs_tile.py needs the prep
    stage's own output files, run_sfincs_tile.py needs the built model).

    Returns (passed, failed_step_label_or_None).
    """
    for step in steps:
        args = [sys.executable, str(_SCRIPT_DIR / step["script"]), "--tile-id", str(tile_id), "--config", config_path]
        args += step.get("extra_args", [])

        print(f"\n--- [{step['label']}] tile {tile_id} ---")
        t0 = time.monotonic()
        result = subprocess.run(args)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            print(f"FAILED: [{step['label']}] tile {tile_id} exited with code {result.returncode} after {elapsed:.1f}s")
            return False, step["label"]
        print(f"OK: [{step['label']}] tile {tile_id} ({elapsed:.1f}s)")

    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-ids", type=int, nargs="+", default=None, help="one or more tile IDs, e.g. --tile-ids 1573 929 851")
    parser.add_argument("--tile-ids-file", default=None, help="text file, one tile ID per line (alternative to --tile-ids, for a large batch)")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--sfincs-exe", default=_DEFAULT_SFINCS_EXE)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--build-only", action="store_true",
        help="only run the prep+build steps (no local SFINCS simulation/postprocess) - "
             "for building a batch of tiles' sfincs_model/ dirs ahead of running them "
             "elsewhere, e.g. generate_sfincs_hpc_jobs.py's own HPC batch dispatch.",
    )
    args = parser.parse_args()
    if not args.tile_ids and not args.tile_ids_file:
        parser.error("one of --tile-ids or --tile-ids-file is required")
    tile_ids = args.tile_ids or [
        int(line.strip()) for line in Path(args.tile_ids_file).read_text().splitlines() if line.strip()
    ]

    steps = _steps(args.sfincs_exe, args.timeout_s, args.build_only)

    results: dict[int, tuple[bool, str | None]] = {}
    for tile_id in tile_ids:
        print(f"\n{'=' * 50}\n=== Tile {tile_id} ===\n{'=' * 50}")
        results[tile_id] = run_tile(tile_id, args.config, steps)

    print(f"\n{'=' * 50}\n=== Summary ===\n{'=' * 50}")
    any_failed = False
    for tile_id, (passed, failed_step) in results.items():
        status = "PASSED" if passed else f"FAILED at '{failed_step}'"
        any_failed = any_failed or not passed
        print(f"tile {tile_id:<8} {status}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
