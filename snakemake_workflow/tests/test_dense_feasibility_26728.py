"""Does a fully dense (no compaction) solve fit in memory on tile 26728 -
the original OOM case (358M px) that motivated the whole compacted-domain
architecture?

Context: Julia's documented OOM culprit was `component_indices` in the
post-solve connectivity filter (per-label index arrays, including the huge
background label) - NOT the `FastSweeping` solve array itself. That specific
issue is already independently fixed here via `np.isin`-based labeling
(`prune_to_coast_connected`). Separately, dense-domain solving was just
measured to be substantially MORE ACCURATE than the compacted domain (tile
1962: 98.98% vs 97.70% Jaccard, and zero missed cells vs 103) - because
compaction's land exclusion severs low-friction "conduit" paths Julia's
real dense solve uses. And `solve_eikonal_dense` needs no per-cell
neighbor-index bookkeeping at all (unlike the compacted path's 4 int64 +
4 float32 arrays per valid VERTEX - ~11GB alone for a domain this size).
So it's worth directly checking whether ditching compaction entirely is
actually affordable now, given the original OOM culprit is already fixed.

This script only exercises the eikonal solve + peak memory/time - it does
NOT compare against Julia's real waterdepth output for this tile (that
output was itself produced with a resolution/config that may reflect a
successful crop_flood_extent-assisted run, not a guaranteed apples-to-apples
uncropped baseline - accuracy comparison is a separate, secondary question
from "does this fit in memory and finish in reasonable time").

Usage:
    python test_dense_feasibility_26728.py <tile_dir> <return_period> <waterlevel_name> [sweep_budget]
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import psutil
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eikonal import solve_eikonal_dense  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402

_proc = psutil.Process()


def _rss_gb() -> float:
    return _proc.memory_info().rss / 1e9


def main() -> None:
    tile_dir = Path(sys.argv[1])
    return_period = sys.argv[2]
    waterlevel_name = sys.argv[3]
    sweep_budget = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    scenario = f"{return_period}_{waterlevel_name}"

    inputs = tile_dir / "inputs"
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
    resolution = toml_cfg["flooding"]["resolution"]
    knn = toml_cfg["waterlevels"]["knn"]
    variable = toml_cfg["waterlevels"]["name"]

    t0 = time.perf_counter()
    print(f"rss before load: {_rss_gb():.2f} GB")

    with rasterio.open(inputs / "dem.tif") as src:
        dem = src.read(1)
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        mask = src.read(1).astype(np.int8)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = src.read(1)

    print(f"grid: {dem.shape} = {dem.size:,} cells   dtypes: dem={dem.dtype} "
          f"mask={mask.dtype} friction={friction.dtype}")
    print(f"rss after load: {_rss_gb():.2f} GB   ({time.perf_counter()-t0:.1f}s)")

    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")
    print(f"stations: {len(boundaries)}  resolution: {resolution}  knn: {knn}  variable: {variable!r}")

    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))

    coastline = coastline_mask(mask, ocean_code=1)
    coastline_rows, coastline_cols = np.nonzero(coastline)
    print(f"coastline cells: {len(coastline_rows):,}")

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    station_values = boundaries[variable].to_numpy()
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)),
    )

    epsilon = float(friction.min()) / (resolution * 10.0)
    print(f"rss before solve: {_rss_gb():.2f} GB")

    t_solve_start = time.perf_counter()
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon,
        sweep_budget=sweep_budget,
    )
    solve_elapsed = time.perf_counter() - t_solve_start
    peak_rss = _rss_gb()
    print(f"SOLVE DONE: {solve_elapsed:.1f}s   rss after solve: {peak_rss:.2f} GB")

    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)
    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]

    print(f"flooded px: {int(flood.sum()):,}")
    print(f"rss final: {_rss_gb():.2f} GB")
    print(f"TOTAL TIME: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
