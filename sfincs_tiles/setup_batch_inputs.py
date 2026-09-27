"""Local batch equivalent of run_one_tile.sh's own step 1: for every tile in
a batch's tile_ids.txt, copies the fixed set of eikonal/SFINCS input files
from the shared, once-per-tile-ID model_outputs/{tile_id}/inputs/ into that
batch's own working copy at {base_dir_name}/{tile_id}/inputs/ - every
downstream script (regenerate_dem_mask.py, build_elevation.py, ...) reads
from the batch's own copy, never from model_outputs/ directly (see
build_elevation.py's own comment on this).

Plain file copies, no hydromt/heavy imports - safe to run under any of this
project's conda envs. Skip-if-already-there (idempotent), never overwrites
an existing file in the batch's own inputs/ - if you deliberately want a
tile's inputs/ regenerated from scratch, delete that tile's inputs/ dir (or
just the specific file) first, same convention as every other script in
this pipeline.

Usage:
    python setup_batch_inputs.py --base-dir-name validation_sfincs_v5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

INPUT_FILES = (
    "tile_geometry.gpkg", "model_bbox.json", "dem.tif", "mask.tif",
    "friction.tif", "boundaries_RP100_SLR_0.gpkg",
)


def ensure_tile_inputs(tile_id: str, root: Path, base_dir_name: str) -> tuple[bool, str | None]:
    """Returns (ok, missing_file_or_None). ok=False means model_outputs/
    itself is missing a required file for this tile - same "skip tile"
    condition run_one_tile.sh's own step 1 treats as unrunnable, not a bug
    to crash on (a tile can legitimately have incomplete upstream prep)."""
    inputs_dir = root / base_dir_name / tile_id / "inputs"
    model_outputs_dir = root / "model_outputs" / tile_id / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        dst = inputs_dir / f
        if dst.exists():
            continue
        src = model_outputs_dir / f
        if not src.exists():
            return False, str(src)
        shutil.copyfile(src, dst)
    return True, None


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    tile_ids_path = root / args.base_dir_name / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) from {tile_ids_path}")

    n_ok = n_missing = n_already = 0
    for tile_id in tile_ids:
        inputs_dir = root / args.base_dir_name / tile_id / "inputs"
        already_complete = all((inputs_dir / f).exists() for f in INPUT_FILES)
        ok, missing = ensure_tile_inputs(tile_id, root, args.base_dir_name)
        if not ok:
            print(f"tile {tile_id}: MISSING {missing} - cannot set up, skipping")
            n_missing += 1
        elif already_complete:
            n_already += 1
        else:
            n_ok += 1

    print(f"\nDone: {n_ok} set up, {n_already} already complete, {n_missing} missing upstream (see above)")


if __name__ == "__main__":
    main()
