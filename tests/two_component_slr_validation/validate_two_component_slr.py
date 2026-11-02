"""Test two-component SLR propagation (static level unattenuated, storm tide attenuated).

Synthetic 24 km transect: ocean in the first two columns, forcing seeded there.
Friction 0.005 m per 30 m step (~0.17 m/km, inside the 0.08-0.48 m/km range).
"""

import sys
from pathlib import Path

import numpy as np
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flood_model import flood_depth_dense, static_inundation, static_level_grid  # noqa: E402

N_ROWS, N_COLS = 40, 800
FRICTION = 0.005
TRANSFORM = Affine(0.00027, 0.0, 100.0, 0.0, -0.00027, 10.0)


def _transect(dem_land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((N_ROWS, N_COLS), dtype=np.int8)
    mask[:, :2] = 1
    dem = np.broadcast_to(dem_land, (N_ROWS, N_COLS)).astype(np.float32).copy()
    return dem, mask


def _solve(dem, mask, total_level: float, static: float | None, seed_cols=(0, 1)):
    rows, cols = np.meshgrid(np.arange(N_ROWS), np.array(seed_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    friction = np.full(dem.shape, FRICTION, dtype=np.float32)
    static_level = None if static is None else np.full(dem.shape, static, dtype=np.float64)
    depth, diag = flood_depth_dense(
        dem, mask, friction, TRANSFORM,
        seed_rows=rows, seed_cols=cols, seed_values=np.full(rows.size, total_level),
        obstacle_coupling=True, max_outer_iterations=4, static_level=static_level,
    )
    return depth, diag


def test_plain_below_new_sea_level_floods_beyond_attenuation_reach() -> None:
    dem, mask = _transect(np.full(N_COLS, 0.3))
    attenuated, _ = _solve(dem, mask, total_level=2.0, static=None)
    two_comp, diag = _solve(dem, mask, total_level=2.0, static=1.0)
    reach_attenuated = int((attenuated[N_ROWS // 2] > 0).sum())
    assert reach_attenuated < 400, reach_attenuated          # ~10 km of 24 km
    assert (two_comp[:, 2:] > 0).all()                       # whole plain flooded
    assert diag["n_static_wet_cells"] == N_ROWS * (N_COLS - 2)
    assert np.allclose(two_comp[:, 2:], 1.7, atol=0.01)      # surge carried over the new sea


def test_identical_when_no_land_is_below_static_level() -> None:
    dem, mask = _transect(np.linspace(1.2, 3.0, N_COLS))
    attenuated, _ = _solve(dem, mask, total_level=2.0, static=None)
    two_comp, diag = _solve(dem, mask, total_level=2.0, static=1.0)
    assert diag["n_static_wet_cells"] == 0
    assert np.array_equal(attenuated, two_comp)


def test_enclosed_depression_behind_ridge_is_not_statically_inundated() -> None:
    land = np.full(N_COLS, 0.5)
    land[100:110] = 3.0                                       # ridge above the total level
    dem, mask = _transect(land)
    coast = np.zeros(dem.shape, dtype=bool)
    coast[:, :2] = True
    wet = static_inundation(np.where(mask == 0, dem, 0.0), mask, np.full(dem.shape, 1.0), coast)
    assert wet[:, 2:100].all()
    assert not wet[:, 110:].any()
    two_comp, _ = _solve(dem, mask, total_level=2.0, static=1.0)
    assert not (two_comp[:, 110:] > 0).any()


def test_flooded_area_never_decreases_with_slr() -> None:
    dem, mask = _transect(np.linspace(0.0, 2.5, N_COLS))
    areas = []
    for slr in (0.0, 0.5, 1.0, 1.5, 2.0):
        depth, _ = _solve(dem, mask, total_level=1.0 + slr, static=slr)
        areas.append(int((depth > 0).sum()))
    assert areas == sorted(areas), areas


def test_split_tile_seeded_from_neighbour_matches_single_solve() -> None:
    dem, mask = _transect(np.full(N_COLS, 0.3))
    full, _ = _solve(dem, mask, total_level=2.0, static=1.0)
    # Right half seeded from the left half's wet cells in a 10-column overlap, as hop>=1 tiles.
    left = slice(0, 410)
    left_depth, _ = _solve(dem[:, left], mask[:, left], total_level=2.0, static=1.0)
    level_left = left_depth + dem[:, left]
    r0 = 400
    rows, cols = np.nonzero(left_depth[:, r0:410] > 0)
    friction = np.full((N_ROWS, N_COLS - r0), FRICTION, dtype=np.float32)
    right, _ = flood_depth_dense(
        dem[:, r0:], np.zeros((N_ROWS, N_COLS - r0), dtype=np.int8), friction, TRANSFORM,
        seed_rows=rows, seed_cols=cols, seed_values=level_left[rows, cols + r0].astype(np.float64),
        obstacle_coupling=True, max_outer_iterations=4,
        static_level=np.full((N_ROWS, N_COLS - r0), 1.0),
    )
    assert np.allclose(right, full[:, r0:], atol=0.01)


def test_storm_tide_stays_local_across_the_new_sea() -> None:
    # Two coasts (2.0 m and 4.0 m total level) joined by a plain below the 1.0 m static level:
    # each side keeps its own coast's level; the high coast's level attenuates from the
    # midpoint with land friction instead of covering the whole new sea.
    dem, mask = _transect(np.full(N_COLS, 0.3))
    mask[:, -2:] = 1
    rows = np.repeat(np.arange(N_ROWS), 4)
    cols = np.tile(np.array([0, 1, N_COLS - 2, N_COLS - 1]), N_ROWS)
    values = np.where(cols < 2, 2.0, 4.0)
    depth, _ = flood_depth_dense(
        dem, mask, np.full(dem.shape, FRICTION, dtype=np.float32), TRANSFORM,
        seed_rows=rows, seed_cols=cols, seed_values=values,
        obstacle_coupling=True, max_outer_iterations=4, static_level=np.full(dem.shape, 1.0),
    )
    level = depth + dem
    assert np.allclose(level[:, 2:20], 2.0, atol=0.1), level[N_ROWS // 2, 2:20].max()
    assert level[:, 2:200].max() < 3.0
    assert np.allclose(level[:, -50:-2], 4.0, atol=0.05)


def test_static_level_grid_reproduces_station_field() -> None:
    lonlat = np.array([[100.0, 10.0], [100.2, 10.0], [100.0, 9.9], [100.2, 9.9]])
    values = np.array([1.0, 1.0, 1.0, 1.0])
    grid = static_level_grid((N_ROWS, N_COLS), TRANSFORM, lonlat, values)
    assert grid.shape == (N_ROWS, N_COLS)
    assert np.allclose(grid, 1.0)
