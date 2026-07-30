"""Diagnostic: does a FULLY DENSE domain (zero restriction - every cell,
exactly like Julia) with EXACTLY 3 single-pass sweeps (no iteration) match
real `aqueduct.exe` output better than the current production approach
(compacted domain, full 4-orthant convergence)?

Motivation: a second machine confirmed via live instrumentation, on real
34M/51M-cell production tiles (not just a small synthetic case), that
Eikonal.jl's `sweep!` deterministically runs exactly 3 of 4 orthants and
never more - 0 positive relative-change updates across ~175M real cell
updates. Yet re-implementing that literal 3-sweep truncation on OUR
compacted domain (candidate | coastline | ocean) measured WORSE on all 5
real tiles than running our solver to full convergence. The reconciling
theory: Fast Sweeping's "3 sweeps suffices" property relies on a single
directional Gauss-Seidel pass being able to propagate information across the
ENTIRE grid in one go - true on Julia's fully dense grid (every cell
participates, even high-friction ones, just at high cost) but NOT true on a
domain with genuine topological gaps (our compacted domain excludes
non-candidate land entirely) - going around a real hole needs multiple
passes/rounds, which a fixed 3-sweep budget doesn't give it. This script
tests that theory directly: use a fully dense valid mask (matching Julia's
literal domain) combined with the literal 3-sweep budget (matching Julia's
literal algorithm), and see if that combination - which has no gaps left to
break the single-pass assumption - reproduces Julia's real output much more
closely than anything tried so far.

Usage:
    python test_dense_three_sweep.py <tile_dir> <return_period> <waterlevel_name>
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eikonal import solve_eikonal_dense  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402


def run(tile_dir: Path, return_period: str, waterlevel_name: str, sweep_budget) -> None:
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
        mask = src.read(1).astype(np.int64)
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

    # THE key difference from production flood_depth: EVERY cell
    # participates, no candidate/elevation restriction at all - matching
    # Julia's literal domain (the whole dense tile, unconditionally). Uses
    # `solve_eikonal_dense` (plain array indexing, no compacted neighbor-index
    # bookkeeping) rather than `build_vertex_domain`+`solve_eikonal` with an
    # all-True mask - the latter was validated to produce IDENTICAL output,
    # but carries the compacted path's O(N) int64 neighbor-index arrays
    # regardless of whether the domain actually has holes, which is wasted
    # memory for a domain that never has any - confirmed to OOM on tile 2660
    # this way (`_ArrayMemoryError` building the lexsort order arrays).
    epsilon = float(friction.min()) / (resolution * 10.0)

    import time
    start = time.perf_counter()
    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon,
        sweep_budget=sweep_budget,
    )
    elapsed = time.perf_counter() - start

    waterlevel = -t[1:, 1:]

    flood = (waterlevel > dem) & (mask != 1)
    flood = prune_to_coast_connected(flood, coastline)

    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]

    julia_path = tile_dir / "results" / f"waterdepth_{scenario}.tif"
    with rasterio.open(julia_path) as src:
        julia_waterdepth = src.read(1).astype(np.float64)

    julia_valid = julia_waterdepth < np.finfo(np.float32).max
    py = waterdepth[julia_valid]
    jl = julia_waterdepth[julia_valid]
    diff = py - jl
    py_flood = py > 0
    jl_flood = jl > 0
    both = py_flood & jl_flood
    either = py_flood | jl_flood
    only_py = py_flood & ~jl_flood
    only_jl = jl_flood & ~py_flood

    label = "full-conv" if sweep_budget is None else f"{sweep_budget}-sweep"
    print(f"tile={tile_dir.name} scenario={scenario} mode={label} "
          f"time={elapsed:.1f}s")
    print(f"  python flooded px: {py_flood.sum():,}   julia flooded px: {jl_flood.sum():,}")
    print(f"  flood/no-flood agreement: {100*(py_flood==jl_flood).mean():.4f}%")
    print(f"  max abs diff: {np.abs(diff).max():.4f} m   mean abs diff (all valid): {np.abs(diff).mean():.6f} m")
    if either.any():
        print(f"  Jaccard: {100*both.sum()/either.sum():.3f}%")
    print(f"  only python: {only_py.sum():,}   only julia: {only_jl.sum():,}")
    if both.any():
        print(f"  mean abs depth diff (both flooded): {np.abs(diff[both]).mean():.6f} m")
    print(f"  RMSE (all valid cells): {np.sqrt((diff**2).mean()):.6f} m")


def main() -> None:
    tile_dir = Path(sys.argv[1])
    return_period = sys.argv[2]
    waterlevel_name = sys.argv[3]
    only_3sw = len(sys.argv) > 4 and sys.argv[4] == "--3sw-only"

    print("=== dense domain, exactly 3 sweeps (matches Julia's literal algorithm) ===")
    run(tile_dir, return_period, waterlevel_name, sweep_budget=3)
    if not only_3sw:
        print()
        print("=== dense domain, full convergence (for comparison) ===")
        run(tile_dir, return_period, waterlevel_name, sweep_budget=None)


if __name__ == "__main__":
    main()
