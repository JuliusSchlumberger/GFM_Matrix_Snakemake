"""Write calibration_result.json per tile, each counting only its Voronoi core.

Sweep tiles overlap (boundary-condition buffer), so pooling raw counts counts
overlap zones once per tile (summed footprints overcount the union by 1.237x).
Output mirrors the original layout in a separate base dir. Needs hydromt-sfincs-dev.

Usage:
    python compute_calibration_result_dedup.py --config <config.yml> \\
        --base-dir-name calibration_roughness_SFINCS \\
        --out-dir-name calibration_roughness_SFINCS_dedup \\
        [--tile-id 1122] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_calibration_result import compute_result  # noqa: E402
from sfincs_grid import build_tile_tree, core_mask  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

def compute_result_dedup(tile_dir: Path, flooding: dict, tree: cKDTree, tile_ids: np.ndarray) -> dict:
    """Build one tile's calibration_result dict, restricted to its Voronoi core.

    `compute_calibration_result.compute_result` with the core mask (on the
    hmax_subgrid.tif grid) as its ``core_mask``.
    """
    with retry_transient_io(rasterio.open, tile_dir / "sfincs_model" / "hmax_subgrid.tif") as src:
        transform, crs, shape = src.transform, src.crs, src.shape
    core = core_mask(shape, transform, crs, tile_dir.name, tree, tile_ids)
    return compute_result(tile_dir, flooding, core_mask=core)


def _process_one(args_tuple) -> tuple[str, str | None]:
    tile_dir, out_tile_dir, flooding, tree, tile_ids = args_tuple
    tile_id = tile_dir.name
    try:
        result = compute_result_dedup(tile_dir, flooding, tree, tile_ids)
    except Exception as exc:
        return tile_id, f"{type(exc).__name__}: {exc}"
    out_dir = out_tile_dir / "outputs"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((out_dir / "calibration_result.json").write_text, json.dumps(result, indent=1), encoding="utf-8")
    return tile_id, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--out-dir-name", default="calibration_roughness_SFINCS_dedup")
    parser.add_argument("--tile-id", default=None, help="single tile only (default: every tile in tile_ids.txt)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    out_base_dir = root / args.out_dir_name
    retry_transient_io(out_base_dir.mkdir, parents=True, exist_ok=True)

    print("Building Voronoi tile-center tree from tile_selection.gpkg...", flush=True)
    tree, tile_ids = build_tile_tree(base_dir / "tile_selection.gpkg")
    print(f"  {len(tile_ids)} selected tile(s) in the partition.", flush=True)

    if args.tile_id:
        tile_list = [args.tile_id]
    else:
        tile_list = (base_dir / "tile_ids.txt").read_text().split()
    print(f"{len(tile_list)} tile(s) to process -> {out_base_dir}", flush=True)

    flooding = config["simulation"]["flooding"]

    jobs = [
        (base_dir / tid, out_base_dir / tid, flooding, tree, tile_ids)
        for tid in tile_list
        if (base_dir / tid / "sfincs_model" / "hmax_subgrid.tif").exists()
    ]
    print(f"{len(jobs)} tile(s) have the required source files.", flush=True)

    n_ok = 0
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            tile_id, err = fut.result()
            if err:
                failed.append((tile_id, err))
                print(f"  [FAILED] tile {tile_id}: {err}", flush=True)
            else:
                n_ok += 1
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta_min = (len(jobs) - i) / rate / 60 if rate else float("nan")
                print(f"  {i}/{len(jobs)} done ({n_ok} ok, {len(failed)} failed) - "
                      f"{elapsed:.0f}s elapsed, ETA {eta_min:.1f} min", flush=True)

    print(f"\nDone: {n_ok}/{len(jobs)} tile(s) written to {out_base_dir}, {len(failed)} failed.")
    if failed:
        print("Failed tiles:", ", ".join(t for t, _ in failed[:40]) + (" ..." if len(failed) > 40 else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
