"""One-off: scan every exposure_population_grid_*.tif / exposure_geogunit_grid_*.tif
for corruption (unreadable as a valid raster) - leftover from the
non-atomic-write race in prepare_exposure_grid_chunk (fixed 2026-10-15, but
not retroactively - any file already corrupted before the fix needs to be
deleted so Snakemake's own file-exists check stops treating it as done).
"""
import os
import sys
from pathlib import Path

import rasterio

# Windows path (P:\...) when run on the preprocessing machine, Linux path
# (/p/...) when run on Hydrax - same network share, different mount
# convention. Pick whichever actually exists rather than hardcoding one and
# silently matching zero files on the other OS (confirmed live: the
# Windows-only version ran "successfully" on Hydrax with 0 files checked).
_WIN_PATH = r"P:\11212688-004-global-floodmaps\modelling\merged_results\chunks"
_LINUX_PATH = "/p/11212688-004-global-floodmaps/modelling/merged_results/chunks"
CHUNKS_DIR = Path(_WIN_PATH if os.name == "nt" else _LINUX_PATH)
if not CHUNKS_DIR.is_dir():
    sys.exit(f"ERROR: {CHUNKS_DIR} does not exist or is not a directory - check the mount path for this OS.")

bad = []
paths = list(CHUNKS_DIR.glob("exposure_population_grid_*.tif")) + list(CHUNKS_DIR.glob("exposure_geogunit_grid_*.tif"))
print(f"checking {len(paths)} file(s)...")
for p in paths:
    try:
        with rasterio.open(p) as src:
            src.read(1, window=((0, 1), (0, 1)))
    except Exception as e:
        bad.append(p)
        print(f"CORRUPT: {p.name} ({e})")

print(f"\n{len(bad)} corrupt file(s) found.")
if "--delete" in sys.argv:
    for p in bad:
        p.unlink()
    print(f"deleted {len(bad)} file(s).")
else:
    print("re-run with --delete to remove them.")
