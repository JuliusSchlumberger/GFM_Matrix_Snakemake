"""Test the per-tile disagreement drivers and the plausibility figures on synthetic tiles."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))
from calibration_paths import shard, threaded_map  # noqa: E402
from compute_tile_drivers import DRIVER_COLUMNS, read_drivers, tile_drivers  # noqa: E402
from plot_tile_plausibility import (  # noqa: E402
    BATHTUB_DRIVERS, EIKONAL_DRIVERS, contrast_table, driver_correlations, plot_contrast_drivers, optimum_summary, per_tile_optimum, plot_driver_scatter,
    plot_metric_distributions, plot_optimum_drivers, plot_optimum_gain,
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


def test_coast_above_the_peak_level_does_not_overtop() -> None:
    tile = _ramp_tile(0.05)
    full = tile_drivers(**tile)
    tile["dem_m"][:25, 10] = 5.0
    half = tile_drivers(**tile)
    assert np.isclose(full["overtop_coast_pct"], 100.0)
    assert np.isclose(half["overtop_coast_pct"], 50.0, atol=2.0)


def test_low_land_behind_an_overtopped_sill_counts_as_enclosed() -> None:
    tile = _ramp_tile(0.0)
    tile["dem_m"][:, 10:] = 0.5
    assert tile_drivers(**tile)["enclosed_pct"] == 0.0
    tile["dem_m"][:, 15] = 1.5  # sill below the 2 m peak: the land behind it is still in the bathtub zone
    enclosed = tile_drivers(**tile)["enclosed_pct"]
    assert np.isclose(enclosed, 100 * 84 / 90, atol=1.0)


def test_a_surge_plateau_holds_the_peak_longer_than_repeated_high_tides() -> None:
    tile = _ramp_tile(0.05)
    t_h = np.arange(0, 72, 600 / 3600)
    tides = 1.0 + np.cos(2 * np.pi * t_h / 12.42)                          # five high tides near 2 m
    plateau = 0.5 + 1.5 * np.exp(-(((t_h - 36) / 14) ** 4))               # one long surge at 2 m
    tile["hydrographs_m"] = np.column_stack([tides] * 3)
    tidal = tile_drivers(**tile)["peak_hold_h"]
    tile["hydrographs_m"] = np.column_stack([plateau] * 3)
    assert tile_drivers(**tile)["peak_hold_h"] > 3 * tidal > 0


def test_owned_area_counts_only_owned_cells_but_keeps_connectivity() -> None:
    tile = _ramp_tile(0.05)
    tile["mask"][:, 60:62] = 3  # a river across the north and south halves
    whole = tile_drivers(**tile)
    owned = np.zeros(tile["mask"].shape, dtype=bool)
    owned[:25] = True  # the tile owns the northern half only
    half = tile_drivers(**tile, owned=owned)
    assert np.isclose(half["coast_km"], whole["coast_km"] / 2)
    assert np.isclose(half["bathtub_zone_km2"], whole["bathtub_zone_km2"] / 2)
    assert np.isclose(half["river_pct"], whole["river_pct"]) and whole["river_pct"] > 0
    assert half["surge_range_m"] == whole["surge_range_m"]
    tile["dem_m"][25:, 10:] = 0.5
    tile["dem_m"][25:, 15] = 1.5  # a sill in the unowned half does not change the owned half
    assert tile_drivers(**tile, owned=owned)["enclosed_pct"] == half["enclosed_pct"] == 0.0


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
                            "enclosed_pct": rng.uniform(0, 60, n), "peak_hold_h": rng.uniform(1, 20, n)})
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


def _sweep(n: int = 40) -> pd.DataFrame:
    # Tile i is best at grid value i % 3 (0.05, 0.1, 0.2); the global value 0.1 is best overall.
    rows = []
    for i in range(n):
        best = [0.05, 0.1, 0.2][i % 3]
        for p in (0.05, 0.1, 0.2):
            csi = 0.8 - 0.3 * abs(np.log2(p / best))
            matched = 100 * csi
            rows.append({"tile_id": i, "group": "calibration" if i % 2 else "validation",
                         "variant": "eikonal_uniform_cap", "parameter": p, "matched_km2": matched,
                         "model_only_km2": (100 - matched) / 2, "sfincs_only_km2": (100 - matched) / 2, "CSI": csi})
    return pd.DataFrame(rows)


def test_per_tile_optimum_finds_each_tiles_best_value_and_its_gain(tmp_path: Path) -> None:
    sweep = _sweep()
    opt = per_tile_optimum(sweep, "eikonal_uniform_cap", 0.1, 1.0)
    expect = np.array([[0.05, 0.1, 0.2][i % 3] for i in opt["tile_id"]])
    assert np.array_equal(opt["best_value"].to_numpy(), expect)
    assert (opt.loc[opt["best_value"] == 0.1, "cls"] == "same").all()
    assert set(opt.loc[opt["best_value"] == 0.05, "cls"]) == {"lower"}
    assert np.allclose(opt.loc[opt["best_value"] != 0.1, "gain"], 0.3)
    summary = optimum_summary(sweep, opt)
    assert (summary["CSI_per_tile_best"] > summary["CSI_global"]).all()
    plot_optimum_gain(opt, summary, sweep, tmp_path / "gain.png")
    drivers = pd.DataFrame({"tile_id": opt["tile_id"], **{d[0]: np.linspace(0.1, 10, len(opt)) for d in EIKONAL_DRIVERS}})
    rho = plot_optimum_drivers(opt, drivers, tmp_path / "opt_drivers.png")
    assert (tmp_path / "gain.png").exists() and (tmp_path / "opt_drivers.txt").exists()
    assert len(rho) == len(EIKONAL_DRIVERS)


def test_shards_split_the_tiles_without_overlap_or_gap() -> None:
    tiles = list(range(10))
    parts = [shard(tiles, f"{k}/4") for k in range(4)]
    assert sorted(sum(parts, [])) == tiles and parts[1] == [1, 5, 9]
    assert shard(tiles, None) == tiles


def test_read_drivers_joins_shard_parts_and_ignores_outdated_files(tmp_path: Path) -> None:
    row = {c: 1.0 for c in DRIVER_COLUMNS}
    pd.DataFrame([{"tile_id": 1, **row}]).to_csv(tmp_path / "tile_drivers.csv", index=False)
    pd.DataFrame([{"tile_id": 2, **row}]).to_csv(tmp_path / "tile_drivers.part0of2.csv", index=False)
    pd.DataFrame([{"tile_id": 3, "coast_km": 1.0}]).to_csv(tmp_path / "tile_drivers.part1of2.csv", index=False)
    assert list(read_drivers(tmp_path)["tile_id"]) == [1, 2]
    assert read_drivers(tmp_path / "missing").empty


def test_threaded_map_keeps_input_order_with_a_small_window() -> None:
    import time
    items = list(range(50))
    out = list(threaded_map(lambda i: time.sleep(0.001 * (i % 7)) or i * i, items, workers=4, window=3))
    assert out == [i * i for i in items]


def test_contrast_classes_follow_the_csi_change(tmp_path: Path) -> None:
    rows = []
    for i in range(30):
        base = 0.6
        gain = (0.1, 0.0, -0.1)[i % 3]  # compared better, as good, worse
        for variant, key, csi in (("eikonal_uniform", "u", base), ("eikonal_uniform_cap", "uc", base + gain),
                                  ("eikonal_landcover", "l", base), ("eikonal_landcover_cap", "lc", base + gain)):
            rows.append({"tile_id": i, "variant": variant, "run_key": key, "CSI": csi,
                         "matched_km2": 10.0, "model_only_km2": 1.0, "sfincs_only_km2": 1.0})
    per_tile = pd.DataFrame(rows)
    runs = {"eikonal_uniform": "u", "eikonal_uniform_cap": "uc", "eikonal_landcover": "l", "eikonal_landcover_cap": "lc"}
    c = contrast_table(per_tile, runs, 1.0)
    cap = c[c["contrast"] == "Cap, uniform rate"].sort_values("tile_id")
    assert list(cap["cls"][:3]) == ["better", "same", "worse"]
    assert (c.loc[c["contrast"] == "Land cover vs uniform", "cls"] == "same").all()
    drivers = pd.DataFrame({"tile_id": range(30), **{d[0]: np.linspace(1, 10, 30) for d in EIKONAL_DRIVERS}})
    rho = plot_contrast_drivers(c, drivers, tmp_path / "contrast.png")
    assert (tmp_path / "contrast.txt").exists() and len(rho) == 3 * len(EIKONAL_DRIVERS)
