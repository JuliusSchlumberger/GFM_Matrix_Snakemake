"""Coast-mode SFINCS boundary: which cells are forced, and forcing points SFINCS interpolates right."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_coast_forcing import (  # noqa: E402
    INTERP_TOL_M, MSK_ACTIVE, MSK_BOUNDARY, MSK_INACTIVE, choose_points, coarsen_mask, coast_msk,
    sfincs_interpolation,
)

LAND, OCEAN, RIVER = 0, 1, 3


def _mask() -> np.ndarray:
    m = np.full((20, 30), LAND, dtype=np.int8)
    m[:, :10] = OCEAN                 # open sea on the west edge
    m[4, 10:18] = OCEAN               # 1-cell channel from the sea
    m[8:11, 20:23] = OCEAN            # closed lagoon, not reaching the edge
    m[15, 10:20] = RIVER              # river entering the sea
    return m


def test_coastline_cells_are_forced_and_open_sea_beyond_them_is_inactive() -> None:
    msk = coast_msk(_mask(), river_code=RIVER)
    assert (msk[:, 9] == MSK_BOUNDARY).all()
    assert (msk[:, :8] == MSK_INACTIVE).all()
    assert (msk[_mask() == LAND] != MSK_BOUNDARY).all()


def test_one_cell_channel_is_forced_along_its_length_and_closed_lagoon_stays_active_unforced() -> None:
    msk = coast_msk(_mask(), river_code=RIVER)
    assert (msk[4, 10:18] == MSK_BOUNDARY).all()
    assert (msk[8:11, 20:23] == MSK_ACTIVE).all()
    assert (msk[15, 10:20] == MSK_ACTIVE).all()  # the river is carried by SFINCS, not forced


def test_seaward_offset_moves_the_boundary_offshore_and_closes_the_sea_band() -> None:
    msk = coast_msk(_mask(), river_code=RIVER, offset_cells=2)
    assert (msk[1:-1, 7] == MSK_BOUNDARY).all()
    assert (msk[1:-1, 8:10] == MSK_ACTIVE).all()
    assert (msk[:, :7] == MSK_INACTIVE).all()
    assert (msk[4, 10:18] == MSK_ACTIVE).all()  # the channel is fed through the band, not forced
    assert (msk[[0, -1], 7:10] == MSK_BOUNDARY).all()  # band meets the domain edge
    active = msk == MSK_ACTIVE
    inactive = msk == MSK_INACTIVE
    assert not (active[:, 1:] & inactive[:, :-1]).any() and not (active[1:, :] & inactive[:-1, :]).any()


def test_coarsened_cell_is_ocean_only_from_half_ocean() -> None:
    fine = np.zeros((4, 8), dtype=np.int8)
    fine[:, :2] = OCEAN          # block 1: half ocean
    fine[:1, 4:6] = OCEAN        # block 2: a quarter ocean
    assert coarsen_mask(fine, 4).tolist() == [[OCEAN, LAND]]


def _coast(y_m: float, n: int, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n) * 120.0
    return np.column_stack([x, np.full(n, y_m)]), np.full(n, row), np.arange(n)


def test_straight_coast_keeps_a_thinned_point_set() -> None:
    xy, rows, cols = _coast(0.0, 200, 0)
    t = np.linspace(0, 2 * np.pi, 50)
    series = 1.0 + 0.002 * cols[:, None] + np.sin(t)[None, :]  # smooth along the coast
    idx, _, err = choose_points(xy, rows, cols, series)
    assert err <= INTERP_TOL_M
    assert len(idx) < len(xy) / 3


def test_points_are_added_where_a_spit_would_borrow_the_other_side_s_level() -> None:
    xa, ra, ca = _coast(0.0, 100, 0)       # seaward side
    xb, rb, cb = _coast(240.0, 100, 2)     # lagoon side 240 m away, 1 m lower
    xy, rows, cols = np.vstack([xa, xb]), np.concatenate([ra, rb]), np.concatenate([ca, cb])
    t = np.linspace(0, 2 * np.pi, 50)
    series = np.vstack([np.tile(2.0 + np.sin(t), (100, 1)), np.tile(1.0 + np.sin(t), (100, 1))])
    idx, rounds, err = choose_points(xy, rows, cols, series)
    assert rounds > 1 and err <= INTERP_TOL_M
    interp = sfincs_interpolation(xy, xy[idx], series[idx])
    assert np.abs(interp - series).max() <= INTERP_TOL_M
