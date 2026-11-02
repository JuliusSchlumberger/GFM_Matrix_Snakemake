"""Test the event-volume corrections on a synthetic coast-sill-basin transect."""

import sys
from pathlib import Path

import numpy as np
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from event_volume import (  # noqa: E402
    DurationCorrection,
    basin_volume_cap,
    duration_friction_multiplier,
    level_pool_level,
    spill_level,
    time_within_peak,
    weir_inflow_volume,
)
from flood_model import flood_depth_dense  # noqa: E402

DT = 600.0
TRANSFORM = Affine(0.00027, 0.0, 100.0, 0.0, -0.00027, 10.0)


def _transect() -> tuple[np.ndarray, np.ndarray]:
    # Columns: 0 sea, 1-9 rising coast, 10 sill at 1.5 m, 11-40 basin floor at 0.2 m, 41+ high ground.
    dem = np.full((5, 50), 5.0)
    dem[:, 0] = 0.0
    dem[:, 1:10] = np.linspace(0.3, 1.2, 9)
    dem[:, 10] = 1.5
    dem[:, 11:41] = 0.2
    sources = np.zeros(dem.shape, dtype=bool)
    sources[:, 0] = True
    return dem, sources


def test_spill_level_is_ground_on_rising_coast_and_sill_in_basin() -> None:
    dem, sources = _transect()
    fill = spill_level(dem, sources)
    assert np.allclose(fill[:, 1:11], dem[:, 1:11])
    assert np.allclose(fill[:, 11:41], 1.5)


def test_time_within_peak_and_weir_volume() -> None:
    h = np.concatenate([np.zeros(10), np.full(6, 2.0), np.zeros(10)])
    assert time_within_peak(h, DT, np.array([0.5]))[0] == 6 * DT
    assert np.isclose(weir_inflow_volume(h, 1.0, 10.0, DT), 1.7 * 10.0 * 1.0 * 6 * DT)


def test_level_pool_level_conserves_volume() -> None:
    z = np.array([0.0, 0.5, 1.0, 1.0])
    level = level_pool_level(z, 100.0, 120.0)
    assert np.isclose(np.clip(level - z, 0, None).sum() * 100.0, 120.0)


def test_basin_cap_limits_fill_to_sill_inflow() -> None:
    dem, sources = _transect()
    fill = spill_level(dem, sources)
    depth = np.zeros_like(dem)
    depth[:, 1:41] = np.clip(1.8 - dem[:, 1:41], 0, None)   # bathtub-like fill at 1.8 m
    # Peak stays above the 1.5 m sill for 2 steps only.
    h = np.concatenate([np.zeros(20), [1.0, 1.8, 1.8, 1.0], np.zeros(20)])
    capped, records = basin_volume_cap(depth, dem, fill, lambda r, c: h, DT, 900.0, 30.0)
    rec = records[0]
    assert rec["capped"]
    assert np.isclose(capped[:, 11:41].sum() * 900.0, rec["inflow_m3"], rtol=1e-6)
    assert np.array_equal(capped[:, 1:11], depth[:, 1:11])  # open coast untouched


def test_duration_multiplier_penalises_short_head_over_sill_only() -> None:
    dem, sources = _transect()
    fill = spill_level(dem, sources)
    depth = np.zeros_like(dem)
    depth[:, 1:41] = np.clip(1.8 - dem[:, 1:41], 0, None)
    h = np.concatenate([np.zeros(20), np.linspace(0, 1.8, 30), np.linspace(1.8, 0, 30), np.zeros(20)])
    mult = duration_friction_multiplier(
        depth, dem, fill, lambda r, c: ([h], np.zeros(r.size, dtype=int)), DT, t_ref_s=12 * 3600, alpha=1.0,
    )
    # Inside the basin the head over the 1.5 m sill is 0.3 m, shorter than the head over
    # the local ground near the coast (up to 1.5 m), so the basin gets the larger multiplier.
    assert mult[:, 20].mean() > mult[:, 1].mean()
    assert (mult >= 1.0).all()


def _correction(h: np.ndarray, t_ref_h: float = 12.0) -> DurationCorrection:
    return DurationCorrection(np.array([[100.0, 10.0]]), h[None, :], DT, t_ref_s=t_ref_h * 3600.0, alpha=1.0)


def _solve(dem, correction=None):
    mask = np.zeros(dem.shape, dtype=np.int8)
    mask[:, 0] = 1
    rows = np.arange(dem.shape[0])
    return flood_depth_dense(
        dem, mask, np.full(dem.shape, 0.002, dtype=np.float32), TRANSFORM,
        seed_rows=rows, seed_cols=np.zeros_like(rows), seed_values=np.full(rows.size, 1.8),
        obstacle_coupling=True, max_outer_iterations=4, duration_correction=correction,
    )


def test_solver_duration_correction_dries_basin_behind_sill_under_short_peak() -> None:
    dem, _ = _transect()
    plain, _ = _solve(dem)
    short = np.concatenate([np.zeros(20), [1.0, 1.8, 1.8, 1.0], np.zeros(20)])
    corrected, diag = _solve(dem, _correction(short))
    assert (plain[:, 11:41] > 0).all()
    assert corrected[:, 11:41].sum() < plain[:, 11:41].sum()
    assert diag["n_duration_corrected_cells"] > 0


def test_solver_duration_correction_is_identity_for_long_events() -> None:
    dem, _ = _transect()
    plain, _ = _solve(dem)
    # Water stands above every head for >= t_ref: multiplier 1 everywhere.
    long = np.concatenate([np.zeros(5), np.full(200, 1.8), np.zeros(5)])
    corrected, diag = _solve(dem, _correction(long, t_ref_h=6.0))
    assert np.array_equal(plain, corrected)
    assert diag["n_duration_corrected_cells"] == 0
