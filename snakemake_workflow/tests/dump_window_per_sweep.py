"""Dump our own per-sweep (o1, o2, o3) t-array snapshots for the same
diagnostic window chosen in tile 2660 (Julia 1-indexed vertex rows
2694:2846, cols 13305:13457), ready to diff directly against the second
machine's live Julia trace on the identical real tile.
"""
import sys
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

# python 0-indexed vertex window matching the Julia 1-indexed [2694:2846, 13305:13457]
r0, r1 = 2693, 2846
c0, c1 = 13304, 13457

out_dir = Path("C:/Users/Schlu005/AppData/Local/Temp/claude/c--Users-Schlu005-GFM/394f620d-c47d-4845-aa31-31c47492cead/scratchpad")
for sweepnum, orthant in enumerate((1, 2, 3), start=1):
    _dense_sweep(t, friction, orthant, neg_two, eight, four)
    window = t[r0:r1, c0:c1]
    np.savetxt(out_dir / f"python_window_sweep{sweepnum}.csv", window, delimiter=",", fmt="%.9e")
    print(f"sweep {sweepnum} done, window shape {window.shape}, "
          f"min={window.min():.6f} max={window.max():.6f}")

print("saved to python_window_sweep{1,2,3}.csv")
