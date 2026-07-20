"""Run the Snakemake DAG, automatically splitting any tile that runs out of
memory in Aqueduct into two smaller, overlapping sub-tiles and retrying.

Aqueduct's OOM handling (see src/aqueduct_runner.py) never fails the
Snakemake rule itself - a tile that runs out of memory gets marked
(model_outputs/oom_tiles/{tile_id}.txt) and every scenario for it gets an
all-nodata placeholder waterdepth raster instead, so `snakemake` always
exits 0 for this reason. This script is the retry loop around that: run
snakemake to completion, scan model_outputs/oom_tiles/ for markers
belonging to tiles still present in tile_grid.path, split each one
(src/tile_split.split_tile - picks whichever axis best balances land
coverage between the two halves; it also deletes the OOM'd tile's stale
model_outputs directory, removes its OOM/skip markers, and updates
tile_grid.path with the two new child tile_ids), then re-run snakemake so
it regenerates the full input chain for the new tile_ids from scratch
(chunk membership and every per-tile preprocessing rule are recomputed
fresh from tile_grid.path on every invocation - no other cache needs
updating). Repeats up to tile_split.max_retries times. A tile is only
split up to tile_split.max_depth times; beyond that it's left with the
existing give-up/nodata-placeholder behaviour and a warning is printed.

This is deliberately simple: each snakemake invocation runs to normal
completion before the next OOM scan, rather than watching/interrupting a
live run. That means a tile which OOMs early in a long run sits with its
nodata placeholder until that invocation finishes - acceptable for the
same reason a fresh invocation is cheap (Snakemake skips everything already
built) and avoids the complexity of managing a live subprocess.

This does not replace `snakemake` for other targets/flags - it always runs
the `all` target (or --target) with `--resources mem_mb=<mem_mb>`, matching
the documented manual invocation.

Usage:
    python snakemake_workflow/run_pipeline.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--cores 1] \\
        [--mem-mb 8000] \\
        [--target simulate]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config  # noqa: E402
from tiles import load_tile_grid  # noqa: E402
from tile_split import split_depth, split_tile  # noqa: E402


def _discover_oom_tiles(model_outputs_dir: Path, current_tile_ids: set[int]) -> list[int]:
    """OOM marker tile_ids that are still live in the current tile_grid.

    A marker can be stale if a previous retry-loop iteration already split
    that tile_id out of tile_grid.path (the marker file itself is deleted by
    split_tile, but this guards against any left over from outside this
    script, e.g. a manual snakemake run).
    """
    oom_dir = model_outputs_dir / "oom_tiles"
    if not oom_dir.exists():
        return []
    found = []
    for f in oom_dir.glob("*.txt"):
        try:
            tid = int(f.stem)
        except ValueError:
            continue
        if tid in current_tile_ids:
            found.append(tid)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_cfg = str(Path(__file__).resolve().parent / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--mem-mb", type=int, default=8000,
                        help="--resources mem_mb budget passed to snakemake (default: 8000); "
                             "leave headroom below total system RAM for the OS and other processes")
    parser.add_argument("--target", default="simulate")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    ts_cfg = cfg.get("tile_split", {})
    fraction = float(ts_cfg.get("fraction", 2 / 3))
    max_depth = int(ts_cfg.get("max_depth", 2))
    max_retries = int(ts_cfg.get("max_retries", 5))

    tile_grid_path = Path(cfg["tile_grid"]["path"])
    model_outputs_dir = Path(cfg["simulation"]["model_outputs"])

    for attempt in range(1, max_retries + 1):
        cmd = [
            "snakemake", args.target,
            "--cores", str(args.cores),
            "--resources", f"mem_mb={args.mem_mb}",
            "--configfile", str(config_path),
            # Restrict rerun-triggering to file-timestamp staleness only
            # (Snakemake's classic Make-like behaviour), NOT rule code/params/
            # input-signature changes too (Snakemake's default since 7.8).
            # Without this, any edit to a rule's params:/input:/script body -
            # even one that's a no-op for tiles where the new behaviour is
            # config-disabled, e.g. vertical_datum_correction.enabled=false -
            # marks every tile that ever used that rule as stale, cascading
            # into re-running already-completed (expensive) Aqueduct
            # simulations for every downstream job. Trade-off: a genuine
            # rule-logic change now needs its outputs manually deleted/
            # touched to force a rerun - it's no longer auto-detected.
            "--rerun-triggers", "mtime",
        ]
        print(f"\n{'=' * 60}")
        print(f"  Running Snakemake (attempt {attempt}/{max_retries})")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd)

        current_ids = set(load_tile_grid(tile_grid_path)["tile_id"].astype(int).tolist())
        oom_ids = _discover_oom_tiles(model_outputs_dir, current_ids)

        if not oom_ids:
            if result.returncode != 0:
                print(
                    "\n  Snakemake failed for a reason other than tile-size OOM "
                    "(no OOM markers found for any live tile) — nothing to split. Aborting."
                )
                sys.exit(result.returncode)
            print("\nNo OOM'd tiles remaining — pipeline complete.")
            return
        if result.returncode != 0:
            print(
                "\n  NOTE: snakemake also reported a nonzero exit alongside the OOM tiles "
                "found below — there may be an UNRELATED failure needing separate attention."
            )

        splittable = [t for t in oom_ids if split_depth(t) < max_depth]
        exhausted = [t for t in oom_ids if t not in splittable]
        if exhausted:
            print(
                f"\n  WARNING: {len(exhausted)} tile(s) still OOM at max split depth "
                f"({max_depth}); giving up on them (nodata placeholder retained): {exhausted}"
            )
        if not splittable:
            print("\nNo further tiles can be split — stopping.")
            return

        print(f"\n  Splitting {len(splittable)} OOM'd tile(s) (attempt {attempt}/{max_retries}): {splittable}")
        for tid in splittable:
            child_a, child_b = split_tile(tid, tile_grid_path, model_outputs_dir, fraction)
            print(f"    tile {tid} -> {child_a}, {child_b}")

    print(
        f"\n  WARNING: reached max_retries={max_retries} snakemake re-invocations; "
        "some OOM tiles may remain unresolved. Re-run this script to continue."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
