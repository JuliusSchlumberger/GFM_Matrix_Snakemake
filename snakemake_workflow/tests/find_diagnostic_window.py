"""Find a compact, informative bounding box in tile 2660 for the
cross-machine intermediate-state comparison: a window small enough to
report as literal text/small files, containing a real mix of
both-flooded-agree, extra (python-only), and ideally missing (julia-only)
cells, with enough surrounding context to be useful.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import solve_eikonal_dense  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402

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
epsilon = float(friction.min()) / (resolution * 10.0)
t = solve_eikonal_dense(friction, coastline_rows, coastline_cols, -initial, epsilon, sweep_budget=3)
waterlevel = -t[1:, 1:]
flood = (waterlevel > dem) & (mask != 1)
flood = prune_to_coast_connected(flood, coastline)

julia_path = tile_dir / "results" / f"waterdepth_{scenario}.tif"
with rasterio.open(julia_path) as src:
    julia_waterdepth = src.read(1).astype(np.float64)
julia_valid = julia_waterdepth < np.finfo(np.float32).max
julia_flood = (julia_waterdepth > 0) & julia_valid

extra = flood & ~julia_flood & julia_valid
missing = julia_flood & ~flood

print(f"extra cells: {int(extra.sum()):,}   missing cells: {int(missing.sum()):,}")

# find a window containing both extra AND missing cells if possible, compact
WIN = 150
labels_extra, n_extra = ndimage.label(extra, structure=np.ones((3, 3)))
sizes = ndimage.sum(extra, labels_extra, index=np.arange(1, n_extra + 1))
order = np.argsort(sizes)[::-1]

best = None
for rank in order[:200]:
    comp_id = rank + 1
    rr, cc = np.nonzero(labels_extra == comp_id)
    r0, r1 = rr.min(), rr.max()
    c0, c1 = cc.min(), cc.max()
    cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
    wr0, wr1 = max(0, cy - WIN // 2), min(dem.shape[0], cy + WIN // 2)
    wc0, wc1 = max(0, cx - WIN // 2), min(dem.shape[1], cx + WIN // 2)
    n_missing_in_win = int(missing[wr0:wr1, wc0:wc1].sum())
    n_extra_in_win = int(extra[wr0:wr1, wc0:wc1].sum())
    n_coast_in_win = int(coastline[wr0:wr1, wc0:wc1].sum())
    score = n_missing_in_win * 10 + n_extra_in_win
    if best is None or score > best[0]:
        best = (score, wr0, wr1, wc0, wc1, n_extra_in_win, n_missing_in_win, n_coast_in_win)

score, wr0, wr1, wc0, wc1, n_extra_in_win, n_missing_in_win, n_coast_in_win = best
print(f"\nchosen window: rows [{wr0}:{wr1}]  cols [{wc0}:{wc1}]  "
      f"({wr1-wr0}x{wc1-wc0})")
print(f"  extra cells in window: {n_extra_in_win}")
print(f"  missing cells in window: {n_missing_in_win}")
print(f"  coastline cells in window: {n_coast_in_win}")
print(f"  both-flooded-agree in window: {int((flood & julia_flood)[wr0:wr1, wc0:wc1].sum())}")
