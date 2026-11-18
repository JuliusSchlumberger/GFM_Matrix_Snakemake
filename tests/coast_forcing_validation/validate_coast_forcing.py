"""Coastline boundary cells and hydrograph jumps of the coast-boundary test (build_coast_forcing)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
from build_coast_forcing import (  # noqa: E402
    MSK_ACTIVE, MSK_BOUNDARY, MSK_INACTIVE, adjacent_pairs, cell_forcing, coarsen_mask, coast_boundary_msk, level_jumps,
)

LAND, OCEAN, RIVER = 0, 1, 3


def _synthetic_mask() -> np.ndarray:
    m = np.full((20, 20), LAND, dtype=np.int16)
    m[:, :8] = OCEAN                 # open sea along the west edge
    m[3, 8:12] = OCEAN               # 1-cell inlet ...
    m[2:5, 12:16] = OCEAN            # ... into a lagoon
    m[10:13, 14:17] = OCEAN          # closed lagoon, inlet not resolved
    m[16, 8:20] = RIVER              # river reaching the sea
    return m


def test_open_coast_is_boundary_and_sea_behind_it_inactive() -> None:
    msk = coast_boundary_msk(_synthetic_mask(), RIVER)
    assert (msk[:, 7] == MSK_BOUNDARY).all()
    assert (msk[:, :6] == MSK_INACTIVE).all()


def test_one_cell_inlet_is_all_boundary_cells() -> None:
    msk = coast_boundary_msk(_synthetic_mask(), RIVER)
    assert (msk[3, 8:12] == MSK_BOUNDARY).all()


def test_closed_lagoon_and_river_stay_active() -> None:
    msk = coast_boundary_msk(_synthetic_mask(), RIVER)
    assert (msk[10:13, 14:17] == MSK_ACTIVE).all()
    assert (msk[16, 8:] == MSK_ACTIVE).all()


def test_no_active_cell_borders_inactive_sea() -> None:
    msk = coast_boundary_msk(_synthetic_mask(), RIVER)
    active = msk == MSK_ACTIVE
    inactive = np.pad(msk == MSK_INACTIVE, 1)
    four_nb = inactive[:-2, 1:-1] | inactive[2:, 1:-1] | inactive[1:-1, :-2] | inactive[1:-1, 2:]
    assert not (active & four_nb).any()


def test_coarsen_mask_takes_ocean_from_half_the_block() -> None:
    m = np.full((4, 8), LAND, dtype=np.int16)
    m[:2, :4] = OCEAN                # 8 of 16: ocean
    m[:2, 4:7] = OCEAN               # 6 of 16: land
    m[2:, 4:] = RIVER                # river outnumbers land in the second block
    out = coarsen_mask(m, 4)
    assert out.tolist() == [[OCEAN, RIVER]]


def test_adjacent_pairs_are_eight_connected_and_unique() -> None:
    a, b = adjacent_pairs(np.array([0, 0, 1, 5]), np.array([0, 1, 1, 5]))
    assert sorted(zip(a.tolist(), b.tolist())) == [(0, 1), (0, 2), (1, 2)]


def _tides() -> np.ndarray:
    t = np.arange(0, 70, 0.5)
    tide = np.cos(2 * np.pi * (t - 34.5) / 12.42)
    return np.stack([tide, 0.5 * tide])  # same phase, half the amplitude


def test_forced_series_peak_at_the_seed_in_both_modes() -> None:
    idx = np.array([[0, 1], [1, 0]])
    dist = np.array([[1.0, 2.0], [1.0, 3.0]])
    seed = np.array([2.0, 2.5])
    for mode in ("nearest", "blend"):
        z = cell_forcing(_tides(), idx, dist, seed, mode, 3.0)
        assert np.allclose(z.max(axis=1), seed)


def test_level_jump_is_the_unshifted_difference_over_time() -> None:
    z = cell_forcing(_tides(), np.array([[0, 0], [1, 1]]), np.ones((2, 2)), np.array([2.0, 2.0]), "nearest", 3.0)
    jump, at = level_jumps(z, np.array([0]), np.array([1]))
    assert np.isclose(jump[0], 1.0, atol=1e-3)  # at low water (sampled at 0.5 h): 0 m vs 1 m below the peak
    assert np.isclose(np.abs(z[0] - z[1])[at[0]], jump[0])


def test_blend_lies_between_the_stations() -> None:
    idx, dist, seed = np.array([[0, 1]]), np.array([[1.0, 1.0]]), np.array([1.0])
    near, blend = (cell_forcing(_tides(), idx, dist, seed, m, 3.0)[0] for m in ("nearest", "blend"))
    other = cell_forcing(_tides(), idx[:, ::-1], dist, seed, "nearest", 3.0)[0]
    assert np.all(blend >= np.minimum(near, other) - 1e-9) and np.all(blend <= np.maximum(near, other) + 1e-9)
