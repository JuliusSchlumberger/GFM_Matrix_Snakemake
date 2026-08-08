"""Single entry point for the pre-processing preparation pipeline.

Runs the one-off steps that must complete before the Snakemake DAG's
`preprocess` target can run: downloading DeltaDTM DEM/mask tiles, building the
fixed-DeltaDTM-tile chunk manifest (2026-08 - see src/tile_chunking.py), and
generating the COAST-RP + SLR fingerprint boundary-condition NetCDFs. Config
is loaded once and passed to every step's `run()` function in-process; a
failure in one step is caught and reported, and the rest continue unless
--fail-fast is set.

tile_generation (this file's step, below) depends only on the
mask/DEM/elev_threshold_m, never on scenario, so it's computed once and
frozen; boundary_conditions is independent of it in the other direction
(no scenario ever feeds tile_generation).

Steps (in order) — also the names used to select them on the command line
and the keys read from config.yml's preparation.* switches:
  sync_deltadtm    — download DeltaDTM DEM/mask tiles into the
                     data catalog's deltadtm/deltadtm_mask dirs
  tile_generation  — build the DeltaDTM-tile-based chunk manifest ->
                     tile_grid.path (preparation/build_tile_manifest.py;
                     REPLACES the pre-2026-08 tile_mask_creation/select_
                     tiles/merge_tiles chain AND the adaptive parent/child
                     pipeline that itself replaced it)
  boundary_conditions — COAST-RP + SLR fingerprint scenario NetCDFs
                     (prepare_boundary_conditions.py)

The individual step modules (sync_deltadtm.py, build_tile_manifest.py,
prepare_boundary_conditions.py) are no longer standalone entry points —
each exposes a `run(config, ...)` function and is only ever invoked from
here, not via `python <script>.py` directly.

RETIRED (2026-08): connectivity_map / src/connectivity_forcing.py (the
straight-line-IDW along-water boundary forcing feature it built an index
for) - never validated beyond a regional subset, superseded by the
frozen-geometry chunk-generation pipeline's own hop-distance/neighbour-
forcing direction (src/tile_chunking.py's compute_run_order) as the
intended way to give hinterland chunks non-ocean boundary forcing. A
chunk that can't find a real COAST-RP station now gets an explicit empty
placeholder (see extract_boundaries.py) rather than being dropped from
tile_grid.path.

Usage:
    python snakemake_workflow/preparation/run_preparation.py \\
        [STEP ...] [--config  snakemake_workflow/config/config.yml] \\
        [--force] [--fail-fast]

    With no STEP given, runs whichever steps are enabled in config.yml's
    preparation.* block (the default: all of them). Name one or more STEPs
    to run exactly those instead, ignoring preparation.* entirely:

        python run_preparation.py boundary_conditions
        python run_preparation.py tile_generation
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

import build_tile_manifest  # noqa: E402
import prepare_boundary_conditions  # noqa: E402
import sync_deltadtm  # noqa: E402

ALL_STEPS = [
    "sync_deltadtm",
    "tile_generation",
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
    parser.add_argument(
        "--start-from", default=None, metavar="CHECKPOINT",
        help="resume tile_generation from a checkpoint file instead of running from scratch - "
             f"one of: {', '.join(build_tile_manifest.CHECKPOINTS)}. Only valid when tile_generation "
             "is the sole selected step (requires tile_generation.write_debug_gpkg=true, since the "
             "checkpoint files ARE the debug GeoPackages from a previous run).",
    )
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

    if args.start_from is not None:
        if args.steps != ["tile_generation"]:
            parser.error(
                "--start-from is only valid when tile_generation is the SOLE selected step "
                f"(got steps={args.steps!r}) - resuming a specific sub-stage of a different "
                "step doesn't make sense."
            )
        if args.start_from not in build_tile_manifest.CHECKPOINTS:
            parser.error(
                f"invalid --start-from={args.start_from!r} "
                f"(choose from: {', '.join(build_tile_manifest.CHECKPOINTS)})"
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

    # ── Step 2: Build the DeltaDTM-tile-based chunk manifest (2026-08) ─────────
    if "tile_generation" in selected:
        results["tile_generation"] = _run_step(
            build_tile_manifest.run, "Build DeltaDTM-tile chunk manifest -> tile_grid.path",
            args.fail_fast, config=cfg, start_from=args.start_from,
        )
    else:
        print("\n  [ SKIP ] Tile generation")

    # ── Step 3: Boundary condition NetCDFs ──────────────────────────────────────
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
