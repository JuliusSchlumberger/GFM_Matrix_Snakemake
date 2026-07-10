"""Single entry point for the pre-processing preparation pipeline.

Runs the one-off steps that must complete before the Snakemake DAG's
`preprocess` target can run: downloading DeltaDTM DEM/mask tiles, building the
DeltaDTM+COAST-RP-derived tile grid (three sequential steps — each consumes
the previous step's output file), and generating the COAST-RP + SLR
fingerprint boundary-condition NetCDFs. Config is loaded once and passed to
every step's `run()` function in-process; a failure in one step is caught
and reported, and the rest continue unless --fail-fast is set.

Steps (in order) — also the names used to select them on the command line
and the keys read from config.yml's preparation.* switches:
  sync_deltadtm       — download DeltaDTM DEM/mask tiles into the
                         data catalog's deltadtm/deltadtm_mask dirs
  tile_mask_creation  — build the 5deg -> 3.75deg overlapping tile grid
  select_tiles        — filter to tiles with DeltaDTM mask coverage
  merge_tiles         — merge/drop undersized tiles -> tile_grid.path
  boundary_conditions — COAST-RP + SLR fingerprint scenario NetCDFs

The individual step modules (sync_deltadtm.py, tile_mask_creation.py,
select_tiles.py, merge_tiles.py, prepare_boundary_conditions.py) are no
longer standalone entry points — each exposes a `run(config, ...)` function
and is only ever invoked from here, not via `python <script>.py` directly.

Usage:
    python snakemake_workflow/preparation/run_preparation.py \\
        [STEP ...] [--config  snakemake_workflow/config/config.yml] \\
        [--force] [--fail-fast]

    With no STEP given, runs whichever steps are enabled in config.yml's
    preparation.* block (the default: all of them). Name one or more STEPs
    to run exactly those instead, ignoring preparation.* entirely:

        python run_preparation.py boundary_conditions
        python run_preparation.py tile_mask_creation select_tiles merge_tiles
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import merge_tiles  # noqa: E402
import prepare_boundary_conditions  # noqa: E402
import select_tiles  # noqa: E402
import sync_deltadtm  # noqa: E402
import tile_mask_creation  # noqa: E402

ALL_STEPS = [
    "sync_deltadtm",
    "tile_mask_creation",
    "select_tiles",
    "merge_tiles",
    "boundary_conditions",
]


def _run_step(fn, label: str, fail_fast: bool, **kwargs) -> bool:
    """Call `fn(**kwargs)`; return True on success, catching/reporting exceptions.

    Prints a banner, timing, and [OK]/[FAIL] status, aborting on exception if
    `fail_fast` is set. Steps run in-process, so failures are caught as
    Python exceptions rather than via a subprocess return code.
    """
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        fn(**kwargs)
    except Exception:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] FAILED after {elapsed:.0f}s - {label}")
        traceback.print_exc()
        if fail_fast:
            print("  Aborting (--fail-fast).")
            sys.exit(1)
        return False
    print(f"\n  [OK] Done in {time.time() - t0:.0f}s - {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(SCRIPTS_DIR.parent / "config" / "config.yml")
    parser.add_argument(
        "steps", nargs="*", metavar="STEP",
        help=f"Step(s) to run, from: {', '.join(ALL_STEPS)}. "
             "Omit to use preparation.* in config.yml instead.",
    )
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--force", action="store_true",
                        help="forwarded to the boundary_conditions step's cache "
                             "(recompute/overwrite cached intermediate files); "
                             "the tile-grid steps have no equivalent cache to bypass")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort on first failed step")
    args = parser.parse_args()

    # Validated manually rather than via argparse's `choices=` on this
    # positional: `choices` combined with `nargs="*"` incorrectly validates
    # the empty-list default against `choices` when zero STEP args are
    # given (a long-standing argparse quirk), raising a spurious
    # "invalid choice: []" error on the otherwise-valid no-args case.
    invalid = [s for s in args.steps if s not in ALL_STEPS]
    if invalid:
        parser.error(
            f"invalid STEP(s): {', '.join(invalid)} (choose from: {', '.join(ALL_STEPS)})"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if args.steps:
        selected = set(args.steps)
    else:
        sw = cfg.get("preparation", {})
        selected = {name for name in ALL_STEPS if sw.get(name, True)}

    results: dict[str, bool] = {}
    t_start = time.time()

    # ── Step 1: Download DeltaDTM DEM/mask tiles ───────────────────────────────
    if "sync_deltadtm" in selected:
        results["sync_deltadtm"] = _run_step(
            sync_deltadtm.run, "Download DeltaDTM DEM/mask tiles",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Download DeltaDTM tiles")

    # ── Step 2: Build the 5deg -> 3.75deg overlapping tile grid ────────────────
    if "tile_mask_creation" in selected:
        results["tile_mask_creation"] = _run_step(
            tile_mask_creation.run, "Build overlapping tile grid (DeltaDTM + COAST-RP coverage)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Tile mask creation")

    # ── Step 3: Filter to tiles with DeltaDTM mask coverage ────────────────────
    if "select_tiles" in selected:
        results["select_tiles"] = _run_step(
            select_tiles.run, "Filter tiles to DeltaDTM mask coverage",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Select tiles")

    # ── Step 4: Merge/drop undersized tiles ─────────────────────────────────────
    if "merge_tiles" in selected:
        results["merge_tiles"] = _run_step(
            merge_tiles.run, "Merge/drop undersized tiles",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Merge tiles")

    # ── Step 5: Boundary condition NetCDFs ──────────────────────────────────────
    if "boundary_conditions" in selected:
        results["boundary_conditions"] = _run_step(
            prepare_boundary_conditions.run, "Boundary conditions (COAST-RP + SLR fingerprints)",
            args.fail_fast, config=cfg, force=args.force,
        )
    else:
        print("\n  [ SKIP ] Boundary conditions")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Preparation pipeline complete  ({total / 60:.1f} min)")
    print(f"{'=' * 60}")
    for step, ok in results.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon}  {step}")
    if results and not all(results.values()):
        print("\n  Some steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
