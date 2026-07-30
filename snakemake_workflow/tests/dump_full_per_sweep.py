"""Dump our own FULL per-sweep (o1, o2, o3) t-array snapshots for tile 2660,
for a direct full-tile diff against the second machine's live Julia trace
(see dump_window_per_sweep.py for the earlier, window-only version - this
extends that to the whole tile now that the user is transferring full
solver.t dumps rather than a single window).

Saved as .npy (binary, compact) rather than CSV - a full (14045, 9272)
float32 array is ~521MB as raw floats; CSV text would be several times
larger and much slower to write/read.
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import _dense_sweep  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask  # noqa: E402

tile_dir = Path("D:/GFM/model_outputs/2660")
scenario = "RP100_SLR_0"
inputs = tile_dir / "inputs"
with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
    toml_cfg = tomllib.load(f)
resolution = toml_cfg["flooding"]["resolution"]
knn = toml_cfg["waterlevels"]["knn"]
variable = toml_cfg["waterlevels"]["name"]

with rasterio.open(inputs / "dem.tif") as src:
    dem = src.read(1)
    transform = src.transform
with rasterio.open(inputs / "mask.tif") as src:
    mask = src.read(1).astype(np.int8)
with rasterio.open(inputs / "friction.tif") as src:
    friction = src.read(1)

boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")
dem = effective_dem(dem, mask)
friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
coastline = coastline_mask(mask, ocean_code=1)
coastline_rows, coastline_cols = np.nonzero(coastline)

stations_lonlat = np.column_stack(
    [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
)
station_values = boundaries[variable].to_numpy()
initial = _idw_seed_values(
    coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
    min(knn, len(station_values)),
)

m, n = friction.shape
t = np.zeros((m + 1, n + 1), dtype=friction.dtype)
t[coastline_rows, coastline_cols] = -initial

dtype = friction.dtype
neg_two = dtype.type(-2.0)
eight = dtype.type(8.0)
four = dtype.type(4.0)

out_dir = Path("C:/Users/Schlu005/AppData/Local/Temp/claude/c--Users-Schlu005-GFM/a8a8b928-166a-4b97-9af7-2bd3f822ce01/scratchpad")
print(f"t shape: {t.shape}")
for sweepnum, orthant in enumerate((1, 2, 3), start=1):
    start = time.perf_counter()
    _dense_sweep(t, friction, orthant, neg_two, eight, four)
    elapsed = time.perf_counter() - start
    np.save(out_dir / f"python_full_sweep{sweepnum}.npy", t.copy())
    print(f"sweep {sweepnum} (orthant {orthant}) done in {elapsed:.1f}s, "
          f"min={t.min():.6f} max={t.max():.6f}")

print("saved python_full_sweep{1,2,3}.npy")
