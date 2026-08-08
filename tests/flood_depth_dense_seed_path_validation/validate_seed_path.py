"""Validate `flood_model.flood_depth_dense`'s 2026-08 explicit-seed path
(`seed_rows`/`seed_cols`/`seed_values`, the hop>=1 hinterland forcing
mechanism - an alternative to the existing `boundaries`/IDW path) and its
mutual-exclusivity contract with `boundaries`.

Synthetic domain unit tests - no raster I/O, no dependency on real DEM/mask
data.

Usage:
    python validate_seed_path.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import flood_model as fm  # noqa: E402


def test_seed_propagation_and_pruning() -> None:
    print("=== explicit seed path: propagation from a seed cell + prune_to_coast_connected drops a disconnected flooded region ===")
    # 5 rows x 15 cols: a low near-basin (cols 0-4) containing the seed, a
    # high-dem barrier (cols 5-6) no real water level can cross, and a low
    # far-basin (cols 7-11) that is NOT reachable from the seed through the
    # boolean flood mask (even though its own raw solved water level ends up
    # above its own dem, since friction-cost propagation doesn't "see" dem at
    # all) - this is exactly what prune_to_coast_connected must remove.
    shape = (5, 15)
    dem = np.zeros(shape, dtype=np.float64)
    dem[:, 5:7] = 100.0  # barrier: far above any reachable propagated water level
    mask = np.zeros(shape, dtype=np.int64)  # all "land" - ocean_code (ndef 1) never appears
    friction = np.full(shape, 0.001, dtype=np.float64)
    transform = Affine.identity()

    seed_rows = np.array([2])
    seed_cols = np.array([0])
    seed_values = np.array([5.0])

    waterdepth, diagnostics = fm.flood_depth_dense(
        dem, mask, friction, transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        resolution=30.0,
    )
    assert diagnostics == {"obstacle_coupling": False}, diagnostics

    near_basin = waterdepth[:, 0:5]
    barrier = waterdepth[:, 5:7]
    far_basin = waterdepth[:, 7:12]

    assert (near_basin > 0).any(), "expected flooding to propagate into the near basin from the seed"
    print("PASS: near basin (seed-connected) flooded as expected")

    assert not barrier.any(), "barrier (dem=100m, unreachable) must never flood"
    print("PASS: high-dem barrier correctly never floods")

    assert not far_basin.any(), (
        "far basin must be pruned by prune_to_coast_connected - its raw solved water level "
        "exceeds its own (low) dem, but it is not 8-connected back to the seed through the "
        "boolean flood mask because of the barrier"
    )
    print("PASS: far basin (flood-mask-disconnected from the seed) correctly pruned away")
    print()


def _dummy_domain() -> tuple[np.ndarray, np.ndarray, np.ndarray, Affine]:
    shape = (3, 3)
    return (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.int64),
        np.full(shape, 0.001, dtype=np.float64),
        Affine.identity(),
    )


def test_mutual_exclusivity_validation() -> None:
    print("=== boundaries / seed-triple mutual-exclusivity validation ===")
    dem, mask, friction, transform = _dummy_domain()
    dummy_boundaries = gpd.GeoDataFrame(
        {"waterlevel": [1.0]}, geometry=gpd.points_from_xy([0], [0]), crs="EPSG:4326",
    )
    seed_rows, seed_cols, seed_values = np.array([1]), np.array([1]), np.array([2.0])

    # neither given
    try:
        fm.flood_depth_dense(dem, mask, friction, transform)
        raise AssertionError("expected ValueError when neither boundaries nor seeds are given")
    except ValueError:
        print("PASS: neither boundaries nor seeds given -> ValueError")

    # both given
    try:
        fm.flood_depth_dense(
            dem, mask, friction, transform,
            boundaries=dummy_boundaries,
            seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        )
        raise AssertionError("expected ValueError when both boundaries and seeds are given")
    except ValueError:
        print("PASS: both boundaries and seeds given -> ValueError")

    # partial seed triple
    try:
        fm.flood_depth_dense(dem, mask, friction, transform, seed_rows=seed_rows, seed_cols=seed_cols)
        raise AssertionError("expected ValueError for a partial seed triple")
    except ValueError:
        print("PASS: partial seed triple (seed_values missing) -> ValueError")

    # seeds + obstacle_coupling: no longer rejected (2026-08) - see
    # test_seed_path_obstacle_coupling below for the real behavioural check.
    waterdepth, diagnostics = fm.flood_depth_dense(
        dem, mask, friction, transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        obstacle_coupling=True,
    )
    assert diagnostics["obstacle_coupling"] is True
    print("PASS: seeds + obstacle_coupling=True runs without error")
    print()


def test_seed_path_obstacle_coupling() -> None:
    print("=== explicit seed path: obstacle_coupling=True matches obstacle_coupling=False ===")
    # Same near-basin/barrier/far-basin domain as test_seed_propagation_and_pruning.
    shape = (5, 15)
    dem = np.zeros(shape, dtype=np.float64)
    dem[:, 5:7] = 100.0
    mask = np.zeros(shape, dtype=np.int64)
    friction = np.full(shape, 0.001, dtype=np.float64)
    transform = Affine.identity()

    seed_rows = np.array([2])
    seed_cols = np.array([0])
    seed_values = np.array([5.0])

    wd_off, diag_off = fm.flood_depth_dense(
        dem, mask, friction, transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        resolution=30.0, obstacle_coupling=False,
    )
    wd_on, diag_on = fm.flood_depth_dense(
        dem, mask, friction, transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        resolution=30.0, obstacle_coupling=True,
    )
    assert diag_on["obstacle_coupling"] is True
    assert diag_on["outer_converged"] is True, diag_on
    assert np.allclose(wd_off, wd_on, atol=1e-3), (
        f"coupled/uncoupled results diverged: max diff {np.abs(wd_off - wd_on).max()}"
    )
    print("PASS: obstacle_coupling=True on the seed path matches the uncoupled solve "
          f"(converged in {diag_on['outer_iterations_used']} outer iteration(s))")
    print()


def main() -> None:
    test_seed_propagation_and_pruning()
    test_mutual_exclusivity_validation()
    test_seed_path_obstacle_coupling()
    print("All flood_depth_dense explicit-seed-path validation checks passed.")


if __name__ == "__main__":
    main()
