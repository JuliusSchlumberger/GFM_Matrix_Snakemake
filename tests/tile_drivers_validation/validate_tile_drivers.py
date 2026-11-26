"""Test the per-tile disagreement drivers and the plausibility figures on synthetic tiles."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))
from compute_tile_drivers import tile_drivers  # noqa: E402
from plot_tile_plausibility import (  # noqa: E402
    BATHTUB_DRIVERS, EIKONAL_DRIVERS, driver_correlations, plot_driver_scatter, plot_metric_distributions,
)


def _ramp_tile(slope_m_per_cell: float) -> dict:
    # Sea in the first 10 columns, land rising eastward; a 2 m surge.
    mask = np.zeros((50, 100), dtype=np.int8)
    mask[:, :10] = 1
    dem = np.tile(np.arange(100, dtype=np.float32) - 9.0, (50, 1)) * slope_m_per_cell
    dem[:, :10] = -1.0
    t = np.linspace(0, 1, 49)
    hydro = np.column_stack([0.2 + 1.8 * np.exp(-((t - 0.5) / 0.1) ** 2)] * 3)
    return {"dem_m": dem, "mask": mask, "friction_per_m": np.full(mask.shape, 0.0004, np.float32),
            "peak_levels_m": np.array([1.9, 2.0, 2.4]), "hydrographs_m": hydro, "hydrograph_dt_s": 600.0, "cell_dx_m": 30.0, "cell_dy_m": 30.0}


def test_flatter_coast_has_longer_bathtub_reach() -> None:
    steep = tile_drivers(**_ramp_tile(0.5))
    flat = tile_drivers(**_ramp_tile(0.05))
    assert flat["reach_km"] > 5 * steep["reach_km"] > 0
    assert np.isclose(flat["coast_km"], 50 * 30 / 1000, rtol=0.05)
    assert np.isclose(flat["friction_m_per_km"], 0.4)
    assert np.isclose(flat["surge_range_m"], 1.8, atol=0.01)
    assert np.isclose(flat["peak_level_spread_m"], 0.5)


def test_land_behind_a_ridge_is_not_in_the_bathtub_zone() -> None:
    tile = _ramp_tile(0.0)
    tile["dem_m"][:, 10:] = 1.0
    open_zone = tile_drivers(**tile)["bathtub_zone_km2"]
    tile["dem_m"][:, 12] = 5.0
    assert tile_drivers(**tile)["bathtub_zone_km2"] < 0.05 * open_zone


def test_coast_above_the_peak_level_does_not_overtop_and_halves_the_inflow() -> None:
    tile = _ramp_tile(0.05)
    full = tile_drivers(**tile)
    tile["dem_m"][:25, 10] = 5.0
    half = tile_drivers(**tile)
    assert np.isclose(full["overtop_coast_pct"], 100.0)
    assert np.isclose(half["overtop_coast_pct"], 50.0, atol=2.0)
    assert half["storage_to_inflow_ratio"] > 1.5 * full["storage_to_inflow_ratio"]


def test_longer_storm_lowers_storage_to_inflow_ratio() -> None:
    tile = _ramp_tile(0.05)
    short = tile_drivers(**tile)["storage_to_inflow_ratio"]
    tile["hydrograph_dt_s"] = 1200.0
    assert np.isclose(tile_drivers(**tile)["storage_to_inflow_ratio"], short / 2)


def test_a_surge_plateau_holds_the_peak_longer_than_repeated_high_tides() -> None:
    tile = _ramp_tile(0.05)
    t_h = np.arange(0, 72, 600 / 3600)
    tides = 1.0 + np.cos(2 * np.pi * t_h / 12.42)                          # five high tides near 2 m
    plateau = 0.5 + 1.5 * np.exp(-(((t_h - 36) / 14) ** 4))               # one long surge at 2 m
    tile["hydrographs_m"] = np.column_stack([tides] * 3)
    tidal = tile_drivers(**tile)["peak_hold_h"]
    tile["hydrographs_m"] = np.column_stack([plateau] * 3)
    assert tile_drivers(**tile)["peak_hold_h"] > 3 * tidal > 0


def test_lake_share_counts_lake_cells() -> None:
    tile = _ramp_tile(0.05)
    tile["mask"][:5, 50:60] = 2
    assert np.isclose(tile_drivers(**tile)["lake_pct"], 100 * 50 / 5000)


def _synthetic_runs(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    drivers = pd.DataFrame({"tile_id": np.arange(n), "river_pct": rng.uniform(0.01, 5, n),
                            "reach_km": rng.uniform(0.01, 3, n), "friction_m_per_km": rng.uniform(0.1, 2, n),
                            "surge_range_m": rng.uniform(0.5, 4, n), "peak_level_spread_m": rng.uniform(0, 2, n),
                            "lake_pct": rng.uniform(0, 3, n), "overtop_coast_pct": rng.uniform(0, 100, n),
                            "storage_to_inflow_ratio": rng.uniform(0.01, 10, n), "abs_lat_deg": rng.uniform(0, 70, n),
                            "peak_hold_h": rng.uniform(1, 20, n)})
    rows = []
    for key, over in (("bathtub_connected", 3.0), ("eikonal_landcover_fsf0.3", 1.0)):
        sfincs = rng.uniform(5, 50, n)
        model = sfincs * over * (1 + drivers["river_pct"].to_numpy())
        matched = np.minimum(sfincs, model) * 0.9
        rows.append(pd.DataFrame({"tile_id": np.arange(n), "run_key": key, "matched_km2": matched,
                                  "model_only_km2": model - matched, "sfincs_only_km2": sfincs - matched}))
    per_tile = pd.concat(rows)
    per_tile["CSI"] = per_tile.matched_km2 / (per_tile.matched_km2 + per_tile.model_only_km2 + per_tile.sfincs_only_km2)
    per_tile["HT"] = per_tile.matched_km2 / (per_tile.matched_km2 + per_tile.sfincs_only_km2)
    per_tile["FAR"] = per_tile.model_only_km2 / (per_tile.matched_km2 + per_tile.model_only_km2)
    return per_tile, drivers


def test_plausibility_figures_render_and_rank_the_planted_driver_first(tmp_path: Path) -> None:
    per_tile, drivers = _synthetic_runs()
    runs = {"bathtub_connected": "bathtub_connected", "eikonal_landcover": "eikonal_landcover_fsf0.3"}
    labels = {"bathtub_connected": "Connected bathtub", "eikonal_landcover": "Eikonal (fsf 0.3)"}
    plot_metric_distributions(per_tile, runs, labels, 1.0, tmp_path / "dist.png")
    data, table = driver_correlations(per_tile, drivers, runs, 1.0, BATHTUB_DRIVERS)
    plot_driver_scatter(data, table, labels, tmp_path / "drivers.png", BATHTUB_DRIVERS, "Message.")
    one = {"eikonal_landcover": runs["eikonal_landcover"]}
    data, eik = driver_correlations(per_tile, drivers, one, 1.0, EIKONAL_DRIVERS)
    plot_driver_scatter(data, eik, labels, tmp_path / "drivers_eikonal.png", EIKONAL_DRIVERS, "Message.")
    assert (tmp_path / "dist.png").exists() and (tmp_path / "drivers.txt").exists()
    assert (tmp_path / "drivers_eikonal.png").exists()
    assert len(table) == len(runs) * len(BATHTUB_DRIVERS) and len(eik) == len(EIKONAL_DRIVERS)
    for t in (table, eik):
        top = t.loc[t.groupby("variant").partial_spearman_rho.idxmax(), "driver"]
        assert set(top) == {"river_pct"}
