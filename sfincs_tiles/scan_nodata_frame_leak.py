"""Scan the calibration tiles for eikonal flooding fed through the no-data
frame of the rotated SFINCS subgrid (fixed in run_calibration_models.py's
solver_inputs - see there; tile 631 was the case that exposed it).

Works on the EXISTING eikonal outputs, no re-solve: flooded non-ocean cells
are labelled into 8-connected components WITHOUT the no-data frame, and any
flooded land not in a component touching the coastline seeds
(run_calibration_models.subgrid_coastline_mask) could only have been reached
through the frame. That leaked area is what the fix removes outright; the
frame can also have raised levels inside coast-connected areas (a shortcut
along the border), which this scan does not measure - every tile with
n_frame_cells_wet > 0 is worth a rerun.

Writes {base}/nodata_frame_leak_scan_fsf<X>.csv, one row per tile, sorted by
leaked area.

Usage:
    python scan_nodata_frame_leak.py --base-dir-name calibration_roughness_SFINCS --friction-scale-factor 12 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import eikonal_path  # noqa: E402
from gfm_config import read_root  # noqa: E402
from run_calibration_models import LAND_CODE, NODATA_CODE, OCEAN_CODE, subgrid_coastline_mask  # noqa: E402

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def scan_tile(tile_dir: Path, fsf: float, river_code: int) -> dict:
    row = {"tile_id": tile_dir.name}
    with rasterio.open(eikonal_path(tile_dir / "outputs", fsf)) as src:
        wet = src.read(1)
        wet = (wet != src.nodata) & (wet > 0) if src.nodata is not None else wet > 0
        transform, crs, shape = src.transform, src.crs, src.shape
    with rasterio.open(tile_dir / "inputs" / "mask.tif") as src:
        mask_f = np.empty(shape, dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=mask_f, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
    mask = np.where(np.isin(mask_f, [0, 1, 2, 3]), mask_f, NODATA_CODE).astype(np.int8)
    px_km2 = abs(transform.a * transform.e) / 1e6

    coast = subgrid_coastline_mask(mask, OCEAN_CODE, river_code)
    region = wet & (mask != OCEAN_CODE) & (mask != NODATA_CODE)
    labels, _ = ndimage.label(region, structure=_STRUCTURE_8)
    touching = np.unique(labels[ndimage.binary_dilation(coast, structure=_STRUCTURE_8)])
    connected = np.isin(labels, touching[touching != 0])
    land = mask == LAND_CODE
    row.update(
        eikonal_land_km2=float((wet & land).sum() * px_km2),
        leaked_land_km2=float((region & ~connected & land).sum() * px_km2),
        n_frame_cells_wet=int((wet & (mask == NODATA_CODE)).sum()),
    )
    return row


def _scan_safe(args) -> dict:
    tile_dir, fsf, river_code = args
    try:
        return scan_tile(tile_dir, fsf, river_code)
    except Exception as e:  # missing output etc. - report, keep scanning
        return {"tile_id": tile_dir.name, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--friction-scale-factor", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-ids", nargs="*", default=None, help="only these tiles (default: every tile with the output)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        river_code = int(yaml.safe_load(f)["tile_generation"]["river_code"])
    base = read_root(Path(args.config)) / args.base_dir_name
    fsf = args.friction_scale_factor
    tiles = [base / t for t in args.tile_ids] if args.tile_ids else sorted(
        (d for d in base.iterdir() if d.is_dir() and eikonal_path(d / "outputs", fsf).exists()), key=lambda d: d.name)
    print(f"scanning {len(tiles)} tile(s) at fsf {fsf:g} with {args.workers} worker(s)", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(_scan_safe, [(t, fsf, river_code) for t in tiles]), 1):
            rows.append(row)
            if k % 50 == 0 or k == len(tiles):
                print(f"  [{k}/{len(tiles)}]", flush=True)

    df = pd.DataFrame(rows)
    if "leaked_land_km2" in df:
        df = df.sort_values("leaked_land_km2", ascending=False)
    out = base / f"nodata_frame_leak_scan_fsf{fsf:g}.csv"
    df.to_csv(out, index=False)
    ok = df[df.get("error").isna()] if "error" in df else df
    print(f"\nwrote {out}")
    print(f"tiles with leaked land > 0: {(ok['leaked_land_km2'] > 0).sum()}, > 1 km2: {(ok['leaked_land_km2'] > 1).sum()}, "
          f"total leaked {ok['leaked_land_km2'].sum():.1f} km2 | tiles with wet frame cells: {(ok['n_frame_cells_wet'] > 0).sum()}")
    if "error" in df:
        print(f"errors: {df['error'].notna().sum()}")
    with pd.option_context("display.width", 160):
        print(ok.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
