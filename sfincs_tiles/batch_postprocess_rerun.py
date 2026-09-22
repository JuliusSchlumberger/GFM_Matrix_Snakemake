"""Regenerate outputs/hmax.tif from the corrected (zsini-fixed) sfincs_map.nc
for every tile whose SFINCS model has been rerun since the batch zsini fix,
so the triplet comparison and boundary-distance correlation can be redone
on clean data instead of the stale, zsini-buggy hmax.tif files.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sfincs_tile import compute_max_inundation, reproject_to_4326  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
BASE = ROOT / "validation_sfincs"
CUTOFF = datetime(2026, 9, 22, 15, 0)

df = pd.read_csv(BASE / "triplet_comparison_RP100_SLR_0.csv")
df["tile_id"] = df["tile_id"].astype(str)
tile_ids = df["tile_id"].tolist()

n_done = 0
n_skipped_stale = 0
n_failed = 0
for i, tile_id in enumerate(tile_ids):
    sfincs_dir = BASE / tile_id / "sfincs_model"
    map_path = sfincs_dir / "sfincs_map.nc"
    if not map_path.exists():
        continue
    mtime = datetime.fromtimestamp(map_path.stat().st_mtime)
    if mtime < CUTOFF:
        n_skipped_stale += 1
        continue

    land_mask_path = BASE / tile_id / "inputs" / "mask.tif"
    out_path = BASE / tile_id / "outputs" / "hmax.tif"
    try:
        hmax, grid_info = compute_max_inundation(sfincs_dir, land_mask_path)
        reproject_to_4326(hmax, grid_info["transform"], grid_info["crs"], out_path)
        n_done += 1
    except Exception as e:
        n_failed += 1
        print(f"tile {tile_id}: FAILED - {type(e).__name__}: {e}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(tile_ids)}  done={n_done} failed={n_failed} skipped_stale={n_skipped_stale}", flush=True)

print(f"\nFINAL: done={n_done} failed={n_failed} skipped_stale={n_skipped_stale} total={len(tile_ids)}")
