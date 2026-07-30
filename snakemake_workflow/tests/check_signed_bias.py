"""Is there a systematic (signed) bias between Python's and Julia's
propagated water levels, not just symmetric noise? The 2660/26729
extra/missing counts are lopsided 4-12x toward Python over-flooding, which
plain random tie-breaking oscillation wouldn't produce on its own - that
needs a genuine, small, consistently-directional difference somewhere,
which the oscillation on complex tiles would then amplify into large
visible extent differences.

Prints mean SIGNED diff (python - julia) on cells both mark as flooded, and
the fraction of those cells where python > julia vs python < julia, for
each tile - across both "stable" and "oscillating" tiles for comparison.
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


def run(tile_dir: Path, return_period: str, waterlevel_name: str) -> None:
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
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon, sweep_budget=3,
    )
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)
    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]

    julia_path = tile_dir / "results" / f"waterdepth_{scenario}.tif"
    with rasterio.open(julia_path) as src:
        julia_waterdepth = src.read(1).astype(np.float64)
    julia_valid = julia_waterdepth < np.finfo(np.float32).max
    julia_flood = (julia_waterdepth > 0) & julia_valid

    both = flood & julia_flood & julia_valid
    py_extra_flood = int((flood & ~julia_flood & julia_valid).sum())
    jl_extra_flood = int((julia_flood & ~flood).sum())

    diff = waterdepth[both] - julia_waterdepth[both]
    n = len(diff)
    print(f"tile={tile_dir.name} scenario={scenario}  both-flooded n={n:,}")
    print(f"  extra(py>jl extent)={py_extra_flood:,}  missing(jl>py extent)={jl_extra_flood:,}  "
          f"ratio={py_extra_flood/max(jl_extra_flood,1):.2f}")
    if n:
        print(f"  signed mean diff (py-julia): {diff.mean():+.6f} m   "
              f"median: {np.median(diff):+.6f} m")
        print(f"  fraction py>julia: {100*(diff>0).mean():.2f}%   "
              f"fraction py<julia: {100*(diff<0).mean():.2f}%   "
              f"fraction equal: {100*(diff==0).mean():.2f}%")

    # localize: does the bias GROW with distance from coast (accumulating
    # per-sweep propagation difference) or stay roughly flat (fixed
    # seed-level offset)? coastline cells themselves are ocean (mask==1) and
    # excluded from `flood` by definition, so can't be checked directly -
    # this is the next best localization signal.
    dist_to_coast = ndimage.distance_transform_edt(~coastline)
    d_both = dist_to_coast[both]
    diff_both = waterdepth[both] - julia_waterdepth[both]
    for lo, hi in [(0, 5), (5, 20), (20, 50), (50, 200), (200, 1e9)]:
        sel = (d_both >= lo) & (d_both < hi)
        n = int(sel.sum())
        if n:
            dd = diff_both[sel]
            rng = f"{lo:.0f}-{hi:.0f}px" if hi < 1e9 else f">{lo:.0f}px"
            print(f"  dist {rng}: n={n:,}  mean diff={dd.mean():+.6f} m  "
                  f"frac py>jl={100*(dd>0).mean():.1f}%")
    print()


def main() -> None:
    specs = [
        ("1963", "RP100", "SLR_0"),
        ("1981", "RP100", "SLR_0"),
        ("1990", "RP100", "SLR_0"),
        ("2660", "RP100", "SLR_0"),
        ("26729", "RP100", "SLR_0"),
    ]
    for tile, rp, sl in specs:
        run(Path(f"D:/GFM/model_outputs/{tile}"), rp, sl)


if __name__ == "__main__":
    main()
