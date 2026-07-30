"""Watch per-round convergence rate on a tile, without waiting for full
convergence - diagnostic for tile 2660/26729, where full convergence was
observed to take >150x longer than the 3-sweep pass with no end in sight
after 30 minutes (unlike well-behaved tiles, where full convergence is only
2-4x the 3-sweep runtime).

Usage:
    python watch_convergence.py <tile_dir> <return_period> <waterlevel_name> [max_rounds]
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

from eikonal import solve_eikonal_dense  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask  # noqa: E402


def main() -> None:
    tile_dir = Path(sys.argv[1])
    return_period = sys.argv[2]
    waterlevel_name = sys.argv[3]
    max_rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    scenario = f"{return_period}_{waterlevel_name}"

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
    print(f"tile: {tile_dir.name}  grid: {dem.shape}  epsilon={epsilon:.6g}  "
          f"max_rounds={max_rounds}")
    solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon,
        max_rounds=max_rounds, verbose=True,
    )


if __name__ == "__main__":
    main()
