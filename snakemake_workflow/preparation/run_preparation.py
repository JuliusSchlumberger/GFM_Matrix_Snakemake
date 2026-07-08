"""Single entry point for the pre-processing preparation pipeline.

Runs the one-off scripts that must complete before the Snakemake DAG's
`preprocess` target can run: syncing/verifying DeltaDTM tiles, building the
DeltaDTM+COAST-RP-derived tile grid (three sequential steps — each consumes
the previous step's output file), and generating the COAST-RP + SLR
fingerprint boundary-condition NetCDFs. Each step runs in its own Python
subprocess so modules are properly isolated and crashes in one step do not
abort subsequent steps (unless --fail-fast is set).

Preparation switches in config.yml:
  preparation.sync_deltadtm       — sync/verify DeltaDTM tiles from the local source dir
  preparation.tile_mask_creation  — build the 5deg -> 3.75deg overlapping tile grid
  preparation.select_tiles        — filter to tiles with DeltaDTM mask coverage
  preparation.merge_tiles         — merge/drop undersized tiles -> tile_grid.path
  preparation.boundary_conditions — COAST-RP + SLR fingerprint scenario NetCDFs

Usage:
    python snakemake_workflow/preparation/run_preparation.py \\
        [--config  snakemake_workflow/config/config.yml] \\
        [--force] \\
        [--fail-fast]

Override switches without editing config.yml by passing --skip-* / --only-*:
    --only-tile-grid            run only sync_deltadtm + tile_mask_creation +
                                 select_tiles + merge_tiles
    --only-boundary-conditions  run only prepare_boundary_conditions.py
    --skip-sync-deltadtm        skip syncing/downloading DeltaDTM tiles
                                 (e.g. if already synced — this step can be slow)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(
    script: Path,
    extra_args: list[str],
    label: str,
    fail_fast: bool,
) -> bool:
    """Run `script` in a subprocess; return True on success."""
    cmd = [PYTHON, str(script)] + extra_args
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'═' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s — {label}")
        if fail_fast:
            print("  Aborting (--fail-fast).")
            sys.exit(result.returncode)
        return False
    print(f"\n  ✓ Done in {elapsed:.0f}s — {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(SCRIPTS_DIR.parent / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--force", action="store_true",
                        help="forwarded to prepare_boundary_conditions.py --force "
                             "(recompute/overwrite cached intermediate files); "
                             "the tile-grid steps have no equivalent cache to bypass")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort on first failed step")
    # Convenience overrides
    parser.add_argument("--only-tile-grid",           action="store_true")
    parser.add_argument("--only-boundary-conditions", action="store_true")
    parser.add_argument("--skip-sync-deltadtm",       action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Forwarded to every step below so a custom --config (e.g. a machine-local
    # override) is actually honored everywhere, not just by
    # prepare_boundary_conditions.py.
    config_args = ["--config", str(config_path)]

    # ── Resolve which steps to run from config + CLI overrides ────────────────
    sw = cfg.get("preparation", {})

    do_sync       = sw.get("sync_deltadtm",      True)
    do_grid       = sw.get("tile_mask_creation",  True)
    do_select     = sw.get("select_tiles",        True)
    do_merge      = sw.get("merge_tiles",         True)
    do_boundaries = sw.get("boundary_conditions", True)

    if args.only_tile_grid:
        do_boundaries = False
    if args.only_boundary_conditions:
        do_sync = do_grid = do_select = do_merge = False
    if args.skip_sync_deltadtm:
        do_sync = False

    results: dict[str, bool] = {}
    t_start = time.time()

    # ── Step 1: Sync/verify DeltaDTM tiles ─────────────────────────────────────
    if do_sync:
        success = _run(
            SCRIPTS_DIR / "sync_deltadtm.py",
            config_args,
            "Sync/verify DeltaDTM tiles",
            args.fail_fast,
        )
        results["sync_deltadtm"] = success
    else:
        print("\n  [ SKIP ] Sync DeltaDTM tiles (preparation.sync_deltadtm = false)")

    # ── Step 2: Build the 5deg -> 3.75deg overlapping tile grid ────────────────
    if do_grid:
        success = _run(
            SCRIPTS_DIR / "tile_mask_creation.py",
            config_args,
            "Build overlapping tile grid (DeltaDTM + COAST-RP coverage)",
            args.fail_fast,
        )
        results["tile_mask_creation"] = success
    else:
        print("\n  [ SKIP ] Tile mask creation (preparation.tile_mask_creation = false)")

    # ── Step 3: Filter to tiles with DeltaDTM mask coverage ────────────────────
    if do_select:
        success = _run(
            SCRIPTS_DIR / "select_tiles.py",
            config_args,
            "Filter tiles to DeltaDTM mask coverage",
            args.fail_fast,
        )
        results["select_tiles"] = success
    else:
        print("\n  [ SKIP ] Select tiles (preparation.select_tiles = false)")

    # ── Step 4: Merge/drop undersized tiles ─────────────────────────────────────
    if do_merge:
        success = _run(
            SCRIPTS_DIR / "merge_tiles.py",
            config_args,
            "Merge/drop undersized tiles",
            args.fail_fast,
        )
        results["merge_tiles"] = success
    else:
        print("\n  [ SKIP ] Merge tiles (preparation.merge_tiles = false)")

    # ── Step 5: Boundary condition NetCDFs ──────────────────────────────────────
    if do_boundaries:
        bc_args = config_args + (["--force"] if args.force else [])
        success = _run(
            SCRIPTS_DIR / "prepare_boundary_conditions.py",
            bc_args,
            "Boundary conditions (COAST-RP + SLR fingerprints)",
            args.fail_fast,
        )
        results["boundary_conditions"] = success
    else:
        print("\n  [ SKIP ] Boundary conditions (preparation.boundary_conditions = false)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  Preparation pipeline complete  ({total / 60:.1f} min)")
    print(f"{'═' * 60}")
    for step, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {step}")
    if not all(results.values()):
        print("\n  Some steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
