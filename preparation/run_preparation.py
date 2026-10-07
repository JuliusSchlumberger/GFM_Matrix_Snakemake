"""Single entry point for the pre-processing preparation pipeline.

Runs the one-off steps that must complete before the Snakemake DAG's
`preprocess` target can run: downloading DeltaDTM DEM/mask tiles, building
their VRT mosaics, building the connectivity-first domain manifest (2026-10
- see src/connectivity_tiling.py and docs/methods_01_tile_processing_and_
waterlevels.md section 3), and generating the COAST-RP + SLR fingerprint
boundary-condition NetCDFs. Config is loaded once and passed to every
step's `run()` function in-process; a failure in one step is caught and
reported, and the rest continue unless --fail-fast is set.

tile_generation (this file's step, below) depends only on the
mask/DEM/elev_threshold_m, never on scenario, so it's computed once and
frozen; boundary_conditions is independent of it in the other direction
(no scenario ever feeds tile_generation).

Steps (in order) — also the names used to select them on the command line
and the keys read from config.yml's preparation.* switches:
  sync_deltadtm      — download DeltaDTM DEM/mask tiles into the
                     data catalog's deltadtm/deltadtm_mask dirs
  fix_ocean_mask     — correct ocean_code mask misclassification against
                     real OSM land polygons, IN PLACE on the per-tile mask
                     .tif files (preparation/fix_ocean_mask_with_osm_land.py;
                     runs before build_deltadtm_vrt so the mosaic is built
                     from already-corrected tiles - see that script's
                     docstring for why DeltaDTM's mask needs this)
  build_deltadtm_vrt — build the deltadtm.vrt / deltadtm_mask.vrt mosaics
                     over those tiles, with portable RELATIVE source paths
                     (preparation/build_deltadtm_vrt.py; separate from
                     sync_deltadtm so the mosaics can be rebuilt on their
                     own, cheaply, without re-downloading tiles - see that
                     script's docstring for the cross-platform-path bug
                     this split fixed)
  tile_generation    — build the connectivity-first domain manifest ->
                     tile_grid.path (preparation/build_tile_manifest.py;
                     REPLACES the pre-2026-08 tile_mask_creation/select_
                     tiles/merge_tiles chain, the adaptive parent/child
                     pipeline that replaced it, and the 13-stage greedy-
                     covering/shave pipeline that replaced THAT in turn -
                     see src/connectivity_tiling.py)
  boundary_conditions — COAST-RP + SLR fingerprint scenario NetCDFs
                     (prepare_boundary_conditions.py)

The individual step modules (sync_deltadtm.py, fix_ocean_mask_with_osm_land.py,
build_deltadtm_vrt.py, build_tile_manifest.py, prepare_boundary_conditions.py)
are no longer standalone entry points — each exposes a `run(config, ...)`
function and is only ever invoked from here, not via `python <script>.py`
directly (fix_ocean_mask_with_osm_land.py is the one exception, with its own
`--dry-run`/`--tiles` ad hoc testing mode - see its own `__main__` block).

RETIRED (2026-08): connectivity_map / src/connectivity_forcing.py (the
straight-line-IDW along-water boundary forcing feature it built an index
for) - never validated beyond a regional subset, superseded by the
frozen-geometry tile-generation pipeline's own hop-distance/neighbour-
forcing direction (now src/connectivity_tiling.py::compute_hop_distances)
as the intended way to give hinterland chunks non-ocean boundary forcing. A
chunk that can't find a real COAST-RP station now gets an explicit empty
placeholder (see extract_boundaries.py) rather than being dropped from
tile_grid.path.

Usage:
    python snakemake_workflow/preparation/run_preparation.py \\
        [STEP ...] [--config  snakemake_workflow/config/config.yml] \\
        [--no-force] [--fail-fast]

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

import build_deltadtm_vrt  # noqa: E402
import build_tile_manifest  # noqa: E402
import fix_ocean_mask_with_osm_land  # noqa: E402
import prepare_boundary_conditions  # noqa: E402
import sync_deltadtm  # noqa: E402

ALL_STEPS = [
    "sync_deltadtm",
    "fix_ocean_mask",
    "build_deltadtm_vrt",
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
    _default_cfg = str(SCRIPTS_DIR.parent / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument(
        "steps", nargs="*", metavar="STEP",
        help=f"Step(s) to run, from: {', '.join(ALL_STEPS)}. "
             "Omit to use preparation.* in config.yml instead.",
    )
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--no-force", dest="force", action="store_false", default=True,
                        help="reuse the boundary_conditions step's cached intermediate/scenario "
                             "files instead of recomputing them (default: always recompute - "
                             "see prepare_boundary_conditions.run's own docstring for why a silent "
                             "cache-hit here is dangerous); the tile-grid steps have no equivalent "
                             "cache to bypass, so this flag only affects boundary_conditions")
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

    # ── Step 1a: Fix ocean_code mask misclassification against OSM land ────────
    if "fix_ocean_mask" in selected:
        results["fix_ocean_mask"] = _run_step(
            fix_ocean_mask_with_osm_land.run, "Fix ocean_code mask misclassification (OSM land polygons)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Fix ocean_code mask misclassification")

    # ── Step 1b: Build the deltadtm/deltadtm_mask VRT mosaics ──────────────────
    if "build_deltadtm_vrt" in selected:
        results["build_deltadtm_vrt"] = _run_step(
            build_deltadtm_vrt.run, "Build DeltaDTM VRT mosaics (relative source paths)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Build DeltaDTM VRT mosaics")

    # ── Step 2: Build the connectivity-first domain manifest (2026-10) ─────────
    if "tile_generation" in selected:
        results["tile_generation"] = _run_step(
            build_tile_manifest.run, "Build connectivity-first domain manifest -> tile_grid.path",
            args.fail_fast, config=cfg,
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
