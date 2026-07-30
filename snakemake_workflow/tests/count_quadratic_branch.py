"""How often does the eikonal update's 2D quadratic branch actually trigger,
vs. falling back to the simple 1D form? Given real friction magnitudes
(~1e-4) are many orders of magnitude smaller than typical water-level
differences (~1-10m), the quadratic branch's validity condition
(|t_a-t_b| <= v*sqrt(2)) should only ever be satisfied in an extremely
narrow near-tie band - meaning it should be RARE, and its evaluation
(disc = b^2-4ac, subtracting two O(t^2) terms to recover an O(v^2) signal)
should be a genuine catastrophic-cancellation hot spot exactly where it
does trigger. This instruments one real sweep pass to count how often it
actually happens, comparing a stable tile (1981) against an oscillating one
(2660) - if the unstable tile has proportionally far more quadratic-branch
triggers, that's direct empirical confirmation of the mechanism.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from numba import njit
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask  # noqa: E402


@njit(cache=True)
def _update_counted(t_a, t_b, v):
    a = 2.0
    b = -2.0 * (t_a + t_b)
    c = t_a * t_a + t_b * t_b - v * v
    disc = b * b - 4.0 * a * c
    best = np.inf
    if disc >= 0.0:
        cand = (-b + np.sqrt(disc)) / (2.0 * a)
        if cand > max(t_a, t_b):
            best = cand
    fallback = min(t_a + v, t_b + v)
    used_quad = 1 if best < fallback else 0
    return min(best, fallback), used_quad


@njit(cache=True)
def _sweep_orthant1_counted(t, friction):
    m, n = friction.shape
    n_visited = 0
    n_accepted = 0
    n_quad = 0
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            n_visited += 1
            cand, used_quad = _update_counted(t[i - 1, j], t[i, j - 1], friction[i - 1, j - 1])
            if cand < t[i, j]:
                n_accepted += 1
                if used_quad:
                    n_quad += 1
                t[i, j] = cand
    return n_visited, n_accepted, n_quad


def run(tile_dir: Path, return_period: str, waterlevel_name: str, n_rounds: int) -> None:
    scenario = f"{return_period}_{waterlevel_name}"
    inputs = tile_dir / "inputs"
    with open(inputs / f"aqueduct_{scenario}.toml", "rb") as f:
        toml_cfg = tomllib.load(f)
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

    print(f"tile: {tile_dir.name}  grid: {dem.shape}")
    for r in range(n_rounds):
        n_visited, n_accepted, n_quad = _sweep_orthant1_counted(t, friction)
        frac_quad = 100 * n_quad / max(n_accepted, 1)
        print(f"  round {r+1} orthant1: visited={n_visited:,} accepted={n_accepted:,} "
              f"quadratic-branch-used={n_quad:,} ({frac_quad:.4f}% of accepted)")


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("=== stable tile 1981 ===")
    run(Path("D:/GFM/model_outputs/1981"), "RP100", "SLR_0", n_rounds)
    print("\n=== unstable tile 2660 ===")
    run(Path("D:/GFM/model_outputs/2660"), "RP100", "SLR_0", n_rounds)


if __name__ == "__main__":
    main()
