"""Why are tiles 2660/26729 much worse (95-97% Jaccard) than the other 8
tiles in the 10-tile dense+3-sweep batch (all 97.6-100%)? Both have far more
"extra" (python-only) false positives than "missing" (julia-only) - the
opposite ratio from the earlier 5-tile investigation, which was
under-flooding dominated. This script characterizes WHERE and WHAT the
extra/missing cells are: mask type breakdown, connected-component structure
(few huge blobs vs. many scattered cells), and distance-to-coast - the same
categorical/spatial breakdown that found the ocean-domain-exclusion bug
earlier in this project.

Usage:
    python diagnose_bad_tile.py <tile_dir> <return_period> <waterlevel_name>
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

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
_MASK_NAMES = {0: "land", 1: "ocean", 2: "lake", 3: "river"}


def main() -> None:
    tile_dir = Path(sys.argv[1])
    return_period = sys.argv[2]
    waterlevel_name = sys.argv[3]
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

    print(f"tile: {tile_dir.name}  grid: {dem.shape} = {dem.size:,} cells")
    for code, name in _MASK_NAMES.items():
        n = int((mask == code).sum())
        print(f"  mask={name}: {n:,} ({100*n/mask.size:.2f}%)")

    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")
    print(f"  stations: {len(boundaries)}")

    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    coastline = coastline_mask(mask, ocean_code=1)
    coastline_rows, coastline_cols = np.nonzero(coastline)
    print(f"  coastline cells: {len(coastline_rows):,}")

    # how many separate coastline segments (8-connected components)?
    coast_labels, n_coast_components = ndimage.label(coastline, structure=_STRUCTURE_8)
    print(f"  coastline connected components: {n_coast_components:,}")

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

    julia_path = tile_dir / "results" / f"waterdepth_{scenario}.tif"
    with rasterio.open(julia_path) as src:
        julia_waterdepth = src.read(1).astype(np.float64)
    julia_valid = julia_waterdepth < np.finfo(np.float32).max
    julia_flood = (julia_waterdepth > 0) & julia_valid

    extra = flood & ~julia_flood & julia_valid   # python says flooded, julia doesn't
    missing = julia_flood & ~flood                # julia says flooded, python doesn't

    print(f"\n  python flooded: {int(flood.sum()):,}   julia flooded: {int(julia_flood.sum()):,}")
    print(f"  extra (python-only): {int(extra.sum()):,}   missing (julia-only): {int(missing.sum()):,}")

    def _breakdown(region: np.ndarray, label: str) -> None:
        print(f"\n  --- {label} cells: mask type breakdown ---")
        total = int(region.sum())
        if total == 0:
            print("    (none)")
            return
        for code, name in _MASK_NAMES.items():
            n = int((region & (mask == code)).sum())
            if n:
                print(f"    {name}: {n:,} ({100*n/total:.2f}%)")

        labels, n_components = ndimage.label(region, structure=_STRUCTURE_8)
        sizes = ndimage.sum(region, labels, index=np.arange(1, n_components + 1))
        sizes = np.sort(sizes)[::-1]
        print(f"    connected components: {n_components:,}")
        print(f"    largest 10 component sizes: {sizes[:10].astype(int).tolist()}")
        print(f"    top-1 component: {100*sizes[0]/total:.2f}% of all {label} cells"
              if len(sizes) else "")

        # distance to nearest coastline cell, binned
        dist_to_coast = ndimage.distance_transform_edt(~coastline)
        d = dist_to_coast[region]
        for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 50), (50, 1e9)]:
            n = int(((d >= lo) & (d < hi)).sum())
            if n:
                label_range = f"{lo:.0f}-{hi:.0f}px" if hi < 1e9 else f">{lo:.0f}px"
                print(f"    distance-to-coast {label_range}: {n:,} ({100*n/total:.2f}%)")

        depth_vals = waterlevel[region] - dem[region] if label == "extra" else \
            julia_waterdepth[region]
        print(f"    depth stats: min={depth_vals.min():.4f} max={depth_vals.max():.4f} "
              f"mean={depth_vals.mean():.4f} median={np.median(depth_vals):.4f}")

    _breakdown(extra, "extra")
    _breakdown(missing, "missing")


if __name__ == "__main__":
    main()
