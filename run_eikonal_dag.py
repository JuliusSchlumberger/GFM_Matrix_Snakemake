"""Run Snakemake, splitting out-of-memory tiles into overlapping halves and retrying.

OOM tiles get a marker and nodata placeholders (snakemake still exits 0). After each
run, marked live tiles are split (up to tile_split.max_depth) and snakemake is rerun,
up to tile_split.max_retries times.

Usage:
    python run_eikonal_dag.py \\
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
    _default_cfg = str(Path(__file__).resolve().parent / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--mem-mb", type=int, default=8000,
                        help="--resources mem_mb budget passed to snakemake (default: 8000); "
                             "leave headroom below total system RAM for the OS and other processes")
    parser.add_argument("--target", default="simulate")
    parser.add_argument("--batch", default=None,
                        help="forwarded as --batch <target>=<this value> to every snakemake "
                             "invocation below, e.g. --batch 3/20 (see snakemake --help for the "
                             "underlying <i>/<n> format) - splits the DAG into a smaller slice per "
                             "invocation the same way as the standalone preprocess batch runner, "
                             "for the same reason (faster DAG builds on P:\\). Split-child tile_ids "
                             "(src/tile_split.py: always one more than the current global max "
                             "tile_id) are always numerically larger than any original tile_id, so "
                             "they sort to the end of the (tile_id-ordered) job list and land in the "
                             "last batch or close to it, regardless of which batch their parent was "
                             "in - a single forward sweep through batches 1..N still converges, it "
                             "just means the LAST batch(es) may pick up extra work created by "
                             "earlier ones.")
    parser.add_argument("--latency-wait", type=int, default=60,
                        help="forwarded to snakemake's own --latency-wait (default: 60s, vs "
                             "Snakemake's own default of 5s). On P:\\ a job can finish writing its "
                             "output and that file still not be visible to Snakemake's process "
                             "within 5s, raising a MissingOutputException for an output that IS "
                             "there and correct - not a real failure, just needs more patience.")
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
            # No --configfile: a CLI configfile is layered after the Snakefile's
            # config_local.yml and would revert its overrides (e.g. paths.root).
            # mtime-only rerun triggers: code/params edits must not re-run
            # finished eikonal solves; real logic changes need outputs touched.
            "--rerun-triggers", "mtime",
            "--latency-wait", str(args.latency_wait),
        ]
        if args.batch:
            cmd += ["--batch", f"{args.target}={args.batch}"]
        print(f"\n{'=' * 60}")
        print(f"  Running Snakemake (attempt {attempt}/{max_retries})")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd)

        tile_grid = load_tile_grid(tile_grid_path)
        current_ids = set(tile_grid["tile_id"].astype(int).tolist())
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

        splittable = [t for t in oom_ids if split_depth(tile_grid, t) < max_depth]
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
