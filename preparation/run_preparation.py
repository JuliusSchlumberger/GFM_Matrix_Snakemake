"""Run the one-off preparation steps needed before the Snakemake `preprocess` target.

Config is loaded once and passed to each step's run(); a failed step is
reported and the rest continue unless --fail-fast. Steps, in order (also the
CLI names and config.yml preparation.* switches):
  sync_deltadtm       - download DeltaDTM DEM/mask tiles
  fix_ocean_mask      - reclassify ocean_code cells inside OSM land, in place
                        (before build_deltadtm_vrt)
  build_deltadtm_vrt  - build DEM/mask VRT mosaics with relative source paths
  tile_generation     - connectivity-first domain manifest -> tile_grid.path
                        (scenario-independent)
  boundary_conditions - COAST-RP + SLR fingerprint scenario NetCDFs
  sync_gtsm           - download raw hourly GTSM reanalysis water levels (opt-in
                        only - name it explicitly; not part of the default step set)
  ea_coastal_dtm      - download the EA LIDAR Composite DTM 1m (2022) for coastal
                        England (opt-in only, same reason as sync_gtsm)
  england_dtm_goco06s - convert the EA coastal DTM (ODN via OSGM15) to GOCO06s,
                        per tile (opt-in only, same reason as sync_gtsm; needs
                        ea_coastal_dtm's output already on disk)
  sf_bay_usgs10m_dem  - reproject the (hand-downloaded) USGS 3DEP 10m DEM for SF Bay
                        to the repo's working grid (opt-in only, same reason as
                        sync_gtsm)
  scotland_lidar_dem  - download Scottish public-sector LIDAR DTM tiles for the
                        Firth of Forth region (opt-in only, same reason as sync_gtsm)

Usage:
    python preparation/run_preparation.py \
        [STEP ...] [--config  snakemake_workflow/config/config.yml] \
        [--skip-existing] [--fail-fast]

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
import convert_england_dtm_to_goco06s  # noqa: E402
import ea_coastal_dtm  # noqa: E402
import fix_ocean_mask_with_osm_land  # noqa: E402
import prepare_boundary_conditions  # noqa: E402
import scotland_lidar_dem  # noqa: E402
import sf_bay_usgs10m_dem  # noqa: E402
import sync_deltadtm  # noqa: E402
import sync_gtsm_water_levels  # noqa: E402

ALL_STEPS = [
    "sync_deltadtm",
    "fix_ocean_mask",
    "build_deltadtm_vrt",
    "tile_generation",
    "boundary_conditions",
    "sync_gtsm",
    "ea_coastal_dtm",
    "england_dtm_goco06s",
    "sf_bay_usgs10m_dem",
    "scotland_lidar_dem",
]

# Opt-in only: excluded from the default step set even though no config.yml
# preparation.* switch says so (see main()'s `selected` below) - a 75-year,
# two-variable, credentialed CDS download (sync_gtsm), two many-tile coastal
# LIDAR downloads (ea_coastal_dtm, scotland_lidar_dem), a many-tile vertical-datum
# conversion (england_dtm_goco06s) and a large single-raster reproject
# (sf_bay_usgs10m_dem) should never start just because someone ran
# `run_preparation.py` with no arguments.
_OPT_IN_STEPS = {
    "sync_gtsm", "ea_coastal_dtm", "england_dtm_goco06s", "sf_bay_usgs10m_dem", "scotland_lidar_dem",
}


def _run_step(fn, label: str, fail_fast: bool, **kwargs) -> bool:
    """Run fn(**kwargs) with a banner; return success, or exit on failure if fail_fast."""
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
    """Run the selected preparation steps in order; exit 1 if any failed."""
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
    parser.add_argument("--skip-existing", dest="force", action="store_false", default=True,
                        help="reuse the boundary_conditions step's cached intermediate/scenario "
                             "files instead of recomputing them (default: always recompute - "
                             "see prepare_boundary_conditions.run's own docstring for why a silent "
                             "cache-hit here is dangerous); the tile-grid steps have no equivalent "
                             "cache to bypass, so this flag only affects boundary_conditions")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort on first failed step")
    args = parser.parse_args()

    # Not argparse choices=: with nargs="*" it rejects the empty default ("invalid choice: []").
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
        selected = {name for name in ALL_STEPS if sw.get(name, name not in _OPT_IN_STEPS)}

    results: dict[str, bool] = {}
    t_start = time.time()

    if "sync_deltadtm" in selected:
        results["sync_deltadtm"] = _run_step(
            sync_deltadtm.run, "Download DeltaDTM DEM/mask tiles",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Download DeltaDTM tiles")

    if "fix_ocean_mask" in selected:
        results["fix_ocean_mask"] = _run_step(
            fix_ocean_mask_with_osm_land.run, "Fix ocean_code mask misclassification (OSM land polygons)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Fix ocean_code mask misclassification")

    if "build_deltadtm_vrt" in selected:
        results["build_deltadtm_vrt"] = _run_step(
            build_deltadtm_vrt.run, "Build DeltaDTM VRT mosaics (relative source paths)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Build DeltaDTM VRT mosaics")

    if "tile_generation" in selected:
        results["tile_generation"] = _run_step(
            build_tile_manifest.run, "Build connectivity-first domain manifest -> tile_grid.path",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Tile generation")

    if "boundary_conditions" in selected:
        results["boundary_conditions"] = _run_step(
            prepare_boundary_conditions.run, "Boundary conditions (COAST-RP + SLR fingerprints)",
            args.fail_fast, config=cfg, force=args.force,
        )
    else:
        print("\n  [ SKIP ] Boundary conditions")

    if "sync_gtsm" in selected:
        results["sync_gtsm"] = _run_step(
            sync_gtsm_water_levels.run, "Download raw hourly GTSM reanalysis water levels",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Download GTSM hourly water levels (opt-in - name it explicitly: "
              "run_preparation.py sync_gtsm)")

    if "ea_coastal_dtm" in selected:
        results["ea_coastal_dtm"] = _run_step(
            ea_coastal_dtm.run, "Download EA LIDAR Composite DTM 1m (2022), coastal England",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Download EA coastal LIDAR DTM (opt-in - name it explicitly: "
              "run_preparation.py ea_coastal_dtm)")

    if "england_dtm_goco06s" in selected:
        results["england_dtm_goco06s"] = _run_step(
            convert_england_dtm_to_goco06s.run, "Convert England coastal DTM (ODN/OSGM15) to GOCO06s",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Convert England coastal DTM to GOCO06s (opt-in - name it explicitly: "
              "run_preparation.py england_dtm_goco06s)")

    if "sf_bay_usgs10m_dem" in selected:
        results["sf_bay_usgs10m_dem"] = _run_step(
            sf_bay_usgs10m_dem.run, "Reproject USGS 3DEP 10m DEM for SF Bay",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Reproject USGS 3DEP 10m SF Bay DEM (opt-in - name it explicitly: "
              "run_preparation.py sf_bay_usgs10m_dem)")

    if "scotland_lidar_dem" in selected:
        results["scotland_lidar_dem"] = _run_step(
            scotland_lidar_dem.run, "Download Scottish LIDAR DTM (Firth of Forth)",
            args.fail_fast, config=cfg,
        )
    else:
        print("\n  [ SKIP ] Download Scottish LIDAR DTM (opt-in - name it explicitly: "
              "run_preparation.py scotland_lidar_dem)")

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
