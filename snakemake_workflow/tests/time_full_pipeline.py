"""Full-pipeline, stage-by-stage timing for the Python dense+3-sweep port -
the true counterpart to the second machine's stage breakdown of Julia's
`flood_depth` (read -> mask/coastline setup -> IDW -> sweep! ->
connected-component pruning), so the two can be compared like-for-like.

Every earlier Python timing in this investigation (test_dense_three_sweep.py,
run_20tile_report.py) wrapped ONLY `solve_eikonal_dense` - excluding file
I/O, coastline/mask setup, IDW interpolation, and the connected-component
pruning step entirely. Julia's measured "~40s mean" / "6.0s sweep!, 11.1s
component-pruning, 9.6s I/O" (tile 2660) is the FULL pipeline minus only the
final raster write. Comparing those two numbers directly was comparing a
partial Python measurement against a full Julia one - this script fixes
that by timing the equivalent full Python pipeline, stage-by-stage, on the
same tile (2660) the second machine profiled.

Usage:
    python time_full_pipeline.py <tile_dir> <return_period> <waterlevel_name>
"""
import sys
import time
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

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def main() -> None:
    tile_dir = Path(sys.argv[1])
    return_period = sys.argv[2]
    waterlevel_name = sys.argv[3]
    scenario = f"{return_period}_{waterlevel_name}"
    inputs = tile_dir / "inputs"

    t0 = time.perf_counter()

    # --- stage 1: read dem/mask/friction/boundaries ---
    t_read0 = time.perf_counter()
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
    resolution = toml_cfg["flooding"]["resolution"]
    knn = toml_cfg["waterlevels"]["knn"]
    variable = toml_cfg["waterlevels"]["name"]

    with rasterio.open(inputs / "dem.tif") as src:
        dem = src.read(1)
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        mask = src.read(1).astype(np.int64)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = src.read(1)
    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")
    t_read = time.perf_counter() - t_read0

    # --- stage 2: effective_dem + friction floor ---
    t_dem0 = time.perf_counter()
    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    t_dem = time.perf_counter() - t_dem0

    # --- stage 3: coastline mask (mask coalesce + dilate equivalent) ---
    t_coast0 = time.perf_counter()
    coastline = coastline_mask(mask, ocean_code=1)
    coastline_rows, coastline_cols = np.nonzero(coastline)
    t_coast = time.perf_counter() - t_coast0

    # --- stage 4: boundary KNN + IDW interpolation ---
    t_idw0 = time.perf_counter()
    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    station_values = boundaries[variable].to_numpy()
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)),
    )
    t_idw = time.perf_counter() - t_idw0

    # --- stage 5: the eikonal sweep itself (dense, exactly 3 sweeps) ---
    epsilon = float(friction.min()) / (resolution * 10.0)
    t_sweep0 = time.perf_counter()
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon, sweep_budget=3,
    )
    t_sweep = time.perf_counter() - t_sweep0

    # --- stage 6: flood classification + connected-component pruning ---
    t_prune0 = time.perf_counter()
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)
    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]
    t_prune = time.perf_counter() - t_prune0

    t_total = time.perf_counter() - t0

    print(f"tile={tile_dir.name} scenario={scenario}  grid={dem.shape}")
    print(f"  1. read dem/mask/friction/boundaries:  {t_read:7.3f}s")
    print(f"  2. effective_dem + friction floor:      {t_dem:7.3f}s")
    print(f"  3. coastline mask:                      {t_coast:7.3f}s")
    print(f"  4. KNN + IDW interpolation:              {t_idw:7.3f}s")
    print(f"  5. sweep (solve_eikonal_dense, 3-sweep): {t_sweep:7.3f}s")
    print(f"  6. flood classification + pruning:       {t_prune:7.3f}s")
    print(f"  TOTAL (excl. output write):              {t_total:7.3f}s")
    print()
    print(f"  as % of total: read={100*t_read/t_total:.1f}%  dem={100*t_dem/t_total:.1f}%  "
          f"coast={100*t_coast/t_total:.1f}%  idw={100*t_idw/t_total:.1f}%  "
          f"sweep={100*t_sweep/t_total:.1f}%  prune={100*t_prune/t_total:.1f}%")


if __name__ == "__main__":
    main()
