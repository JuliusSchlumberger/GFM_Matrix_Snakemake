"""Copies the fixed set of eikonal/SFINCS input files for every tile in a
batch's tile_ids.txt from the shared model_outputs/{tile_id}/inputs/ into
that batch's own working copy at {base_dir_name}/{tile_id}/inputs/. Every
downstream script reads from the batch's own copy, never from
model_outputs/ directly.

Plain file copies, no heavy imports - safe under any conda env. Idempotent:
skips files already present, never overwrites. To force a tile's inputs/ to
be regenerated, delete that tile's inputs/ dir (or the specific file) first.

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
    """Returns (ok, missing_file_or_None). ok=False means model_outputs/ is
    missing a required file for this tile (incomplete upstream prep, not an
    error)."""
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
