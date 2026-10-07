"""Writes `{base_dir_name}/tile_ids_remaining.txt`: the subset of
`tile_ids.txt` whose roughness/friction sweep isn't fully done yet - i.e.
missing at least one of the expected
`eikonal_on_subgrid_waterdepth_RP100_SLR_0{_fsf<v>}{_outer<o>}.tif` files.

Reuses check_sfincs_calibration_progress.py's own per-tile completeness
logic directly (`_eikonal_filename`/`_entries`) rather than re-deriving the
filename-tagging convention - that script is read-only reporting; this one
is its natural companion for building a resubmission target list. A tile
missing an EARLIER stage (inputs/build/run/bathtub) is also "remaining" by
construction (it can't have the sweep files without those), so no separate
check is needed - the sweep-file check alone is the complete definition of
"done".

Feeds directly into generate_validation_batch_jobs.py's own --tile-ids-file,
for a resubmission scoped to only what's left, on however many nodes you
choose (the idempotent skip checks throughout run_one_tile.sh/
run_friction_sweep_batch.py mean this is safe to point at a MIX of
partially-done and untouched tiles - no separate "already started" case to
handle).

Usage:
    python find_incomplete_calibration_tiles.py
    python find_incomplete_calibration_tiles.py --base-dir-name sfincs_calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sfincs_calibration_progress import (  # noqa: E402
    FRICTION_SCALE_FACTORS_DEFAULT, MAX_OUTER_ITERATIONS_SWEEP_DEFAULT, _entries, _eikonal_filename,
)
from gfm_config import read_root  # noqa: E402


def find_incomplete_tiles(
    base_dir: Path, tile_ids: list[str], friction_scale_factors: list[float], max_outer_iterations: int,
) -> list[str]:
    sweep_filenames = [_eikonal_filename(fsf, max_outer_iterations) for fsf in friction_scale_factors]
    remaining = []
    for tid in tile_ids:
        output_entries = _entries(base_dir / tid / "outputs")
        if not all(fname in output_entries for fname in sweep_filenames):
            remaining.append(tid)
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    _default_cfg = str(_repo_root / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT)
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_SWEEP_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {base_dir_name}/tile_ids_remaining.txt")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids_path = base_dir / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_path}")

    remaining = find_incomplete_tiles(base_dir, tile_ids, args.friction_scale_factors, args.max_outer_iterations)
    print(f"{len(remaining)}/{len(tile_ids)} tile(s) missing at least one sweep point "
          f"({len(tile_ids) - len(remaining)} fully done)")

    out_path = Path(args.out) if args.out else base_dir / "tile_ids_remaining.txt"
    out_path.write_text("\n".join(remaining) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
