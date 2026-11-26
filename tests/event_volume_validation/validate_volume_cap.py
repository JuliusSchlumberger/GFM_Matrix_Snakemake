"""Test correction C: inflow budget over the coastal rings and the access-order fill."""

import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from event_volume import VolumeCap, inflow_volume  # noqa: E402
from flood_model import apply_volume_cap, flood_depth_dense  # noqa: E402

DT = 600.0
CELL = 30.0
TRANSFORM = Affine(CELL, 0.0, 500000.0, 0.0, -CELL, 5000000.0)
UTM = "EPSG:32631"


def _hydrograph(peak_hours: float) -> np.ndarray:
    # 2 m above its minimum for `peak_hours`, triangular flanks of 6 h either side.
    t = np.arange(0, 48 * 3600, DT) / 3600.0
    plateau = np.abs(t - 24.0) <= peak_hours / 2
    flank = np.clip(1.0 - (np.abs(t - 24.0) - peak_hours / 2) / 6.0, 0.0, 1.0)
    return np.where(plateau, 2.0, 2.0 * flank)


def _cap(series: np.ndarray, n_rings: int = 6, velocity_m_s: float = 1.0) -> VolumeCap:
    return VolumeCap(np.array([[3.0, 45.0]]), np.atleast_2d(series), DT, velocity_m_s=velocity_m_s, n_rings=n_rings,
                     crs=UTM)


def _sea_left(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    sources = np.zeros(shape, dtype=bool)
    sources[:, 0] = True
    return sources, np.where(sources, 1.5, -np.inf)


def test_inflow_is_velocity_times_head_times_width_and_zero_above_the_peak() -> None:
    shape_rel = (_hydrograph(2.0) - 2.0)[None, :]
    v = inflow_volume(np.array([1.5]), np.array([0.5]), shape_rel, DT, CELL, 1.0)
    assert np.isclose(v[0], CELL * np.clip(1.0 + shape_rel[0], 0, None).sum() * DT)
    assert inflow_volume(np.array([1.5]), np.array([1.6]), shape_rel, DT, CELL, 1.0)[0] == 0.0


def test_budget_is_set_by_the_narrowest_ring() -> None:
    # Wet land behind the sea column; a dry wall at column 3 with a one-cell gap.
    depth = np.full((21, 30), 0.5)
    depth[:, 3] = 0.0
    depth[10, 3] = 0.5
    sources, seed_level = _sea_left(depth.shape)
    depth[sources] = 0.0
    ground = np.zeros_like(depth)
    patch = (depth > 0) & ~sources
    labels, n = ndimage.label(patch, structure=np.ones((3, 3), bool))
    assert n == 1
    budget, limiting = _cap(_hydrograph(2.0)).budgets(patch, labels, n, ground, sources, seed_level, TRANSFORM, CELL, CELL)
    assert limiting[1] == 3  # the gap is three cells inland
    assert budget[1] < 21 * 1.0 * CELL * (1.5 + _hydrograph(2.0) - 2.0).clip(0).sum() * DT / 10


def test_level_order_keeps_the_least_attenuated_cells_within_the_budget() -> None:
    sources, seed_level = _sea_left((5, 60))
    depth = np.tile(np.linspace(1.5, 0.2, 60), (5, 1))
    depth[sources] = 0.0
    ground = np.zeros_like(depth)
    out, rec = _cap(_hydrograph(0.5), n_rings=1, velocity_m_s=0.05).apply(depth, ground + depth, ground, sources, seed_level,
                                                      TRANSFORM, CELL, CELL)
    (patch,) = rec
    kept = out > 0
    assert patch["capped"] and patch["kept_m3"] <= patch["inflow_m3"]
    assert kept[:, 1].all() and not kept[:, -1].any()      # dries from the far, low end
    assert np.array_equal(out[kept], depth[kept])          # kept cells keep their depth


def test_storm_volume_and_inflow_width_follow_the_cell_width_and_height() -> None:
    sources, seed_level = _sea_left((5, 60))
    depth = np.tile(np.linspace(1.5, 0.2, 60), (5, 1))
    depth[sources] = 0.0
    ground = np.zeros_like(depth)
    cap = _cap(_hydrograph(0.5), n_rings=1, velocity_m_s=0.05)
    _, (square,) = cap.apply(depth, ground + depth, ground, sources, seed_level, TRANSFORM, CELL, CELL)
    _, (narrow,) = cap.apply(depth, ground + depth, ground, sources, seed_level, TRANSFORM, CELL / 2, CELL)
    assert np.isclose(narrow["model_m3"], square["model_m3"] / 2)
    assert np.isclose(narrow["inflow_m3"], square["inflow_m3"] / np.sqrt(2))


def test_distance_order_fills_a_flat_level_from_the_coast_and_drops_disconnected_cells() -> None:
    sources, seed_level = _sea_left((5, 60))
    depth = np.full((5, 60), 1.0)
    depth[sources] = 0.0
    depth[:, 30] = 0.0                                    # a dry ridge splits off a pocket...
    depth[2, 30] = 1.0                                    # ...joined by a one-cell gap
    ground = 1.5 - np.where(sources, 1.5, depth)
    out, (patch,) = _cap(_hydrograph(0.5), n_rings=1, velocity_m_s=0.05).apply(depth, ground + depth, ground, sources, seed_level,
                                                           TRANSFORM, CELL, CELL, fill_order="distance")
    kept = out > 0
    assert patch["capped"]
    assert kept[:, 1].all() and not kept[:, 31:].any()
    labels, _ = ndimage.label(kept, structure=np.ones((3, 3), bool))
    assert set(np.unique(labels[kept])) == set(np.unique(labels[:, 1][kept[:, 1]]))  # all kept cells reach the sea


def _inlet_tile() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    # Sea column 0, a 1-cell inlet at row 20, flat hinterland at 0 m behind a 3 m wall.
    mask = np.zeros((41, 80), dtype=np.int8)
    mask[:, 0] = 1
    dem = np.full(mask.shape, 0.0, dtype=np.float32)
    dem[:, 1] = 3.0
    dem[20, 1] = 0.5
    friction = np.full(mask.shape, 0.0003, dtype=np.float32)
    common = dict(seed_rows=np.arange(41), seed_cols=np.zeros(41, dtype=int), seed_values=np.full(41, 1.5),
                  cell_dx_m=CELL, cell_dy_m=CELL, ocean_code=1, idw_power=2.0, k=5)
    return dem, mask, friction, common


def test_short_storm_caps_a_large_flat_hinterland_and_long_storm_does_not() -> None:
    dem, mask, friction, common = _inlet_tile()
    base, _ = flood_depth_dense(dem, mask, friction, TRANSFORM, **common)
    short, d_short = flood_depth_dense(dem, mask, friction, TRANSFORM, volume_cap=_cap(_hydrograph(0.5)), **common)
    long_, d_long = flood_depth_dense(dem, mask, friction, TRANSFORM, volume_cap=_cap(np.full(2000, 2.0)), **common)
    assert d_short["n_volume_capped_patches"] == 1 and short.sum() < base.sum()
    assert (short <= base + 1e-6).all()
    assert d_long["n_volume_capped_patches"] == 0 and np.array_equal(long_, base)


def test_an_inlet_river_cell_counts_with_its_real_ground_not_the_zeroed_one() -> None:
    dem, mask, friction, common = _inlet_tile()
    mask[20, 1] = 3
    low, high = dem.copy(), dem.copy()
    low[20, 1], high[20, 1] = 0.0, 1.4
    cap = _cap(_hydrograph(6.0))
    d_low, _ = flood_depth_dense(low, mask, friction, TRANSFORM, volume_cap=cap, **common)
    d_high, diag = flood_depth_dense(high, mask, friction, TRANSFORM, volume_cap=cap, **common)
    assert diag["n_volume_capped_patches"] == 1 and d_high[:, 2:].sum() < d_low[:, 2:].sum()


def test_connected_bathtub_is_capped_and_the_unconnected_bathtub_refused() -> None:
    dem, mask, friction, common = _inlet_tile()
    base, _ = flood_depth_dense(dem, mask, friction, TRANSFORM, model="bathtub_connected", **common)
    capped, diag = flood_depth_dense(dem, mask, friction, TRANSFORM, model="bathtub_connected",
                                     volume_cap=_cap(_hydrograph(0.5)), **common)
    assert diag["n_volume_capped_patches"] == 1 and 0 < capped.sum() < base.sum()
    with pytest.raises(ValueError, match="bathtub"):
        flood_depth_dense(dem, mask, friction, TRANSFORM, model="bathtub", volume_cap=_cap(_hydrograph(0.5)), **common)


@pytest.mark.parametrize("model", ["eikonal", "straight_line", "bathtub_connected"])
def test_capping_a_finished_run_equals_capping_inside_the_solve(model: str) -> None:
    dem, mask, friction, common = _inlet_tile()
    cap = _cap(_hydrograph(0.5))
    kw = dict(model=model, attenuation_m_per_m=2e-4, **common)
    inside, d_inside = flood_depth_dense(dem, mask, friction, TRANSFORM, volume_cap=cap, **kw)
    base, _ = flood_depth_dense(dem, mask, friction, TRANSFORM, **kw)
    after, d_after = apply_volume_cap(base, dem, mask, common["seed_rows"], common["seed_cols"], common["seed_values"],
                                      cap, TRANSFORM, CELL, CELL, model)
    assert d_inside["n_volume_capped_patches"] == 1 and np.array_equal(inside, after)
    assert d_after["volume_cap_removed_m3"] == d_inside["volume_cap_removed_m3"]
