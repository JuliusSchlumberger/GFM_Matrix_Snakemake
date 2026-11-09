"""The SFINCS calibration's split, phase plan, run tables and variant selection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from calibration_paths import DURATION_ALPHAS, FRICTION_SCALE_FACTORS, UNIFORM_RATES_M_PER_KM  # noqa: E402
from calibration_results import run_tables, select_runs  # noqa: E402
from calibration_split import CALIBRATION_SHARE, balance_table, split_group, tile_groups  # noqa: E402
from run_calibration_models import parse_run, phase_runs  # noqa: E402


def test_split_is_stable_and_near_the_share() -> None:
    ids = range(5000)
    groups = tile_groups(ids)
    share = (groups == "calibration").mean()
    assert abs(share - CALIBRATION_SHARE) < 0.03
    assert all(split_group(t) == g for t, g in groups.iloc[:50].items())
    assert tile_groups(range(6000)).iloc[:5000].equals(groups)  # adding tiles moves none


def test_balance_table_reports_both_groups() -> None:
    rng = np.random.default_rng(0)
    tiles = pd.DataFrame({"tile_id": range(400), "lat": rng.uniform(-60, 70, 400),
                          "n_cells": rng.integers(1e6, 4e7, 400), "ocean_frac": rng.uniform(0, 1, 400),
                          "river_frac": 0.0, "sfincs_flooded_km2": np.nan})
    tiles["group"] = tile_groups(tiles.tile_id).to_numpy()
    b = balance_table(tiles)
    assert {"calibration", "validation"} <= set(b.columns)
    assert b[b.what == "tiles"].iloc[0][["calibration", "validation"]].sum() == 400


def test_phase_runs() -> None:
    sweep = phase_runs("sweep", None)
    assert len(sweep) == 2 + 2 * len(UNIFORM_RATES_M_PER_KM) + len(FRICTION_SCALE_FACTORS)
    selection = {"variants": {"eikonal_uniform": {"parameter": 0.1, "alpha": None},
                              "eikonal_landcover": {"parameter": 0.3, "alpha": None},
                              "bathtub": {"parameter": None, "alpha": None}}}
    duration = phase_runs("duration", selection)
    assert ("eikonal_landcover_duration", 0.3, DURATION_ALPHAS[0]) in duration
    assert len(duration) == 2 + 2 * len(DURATION_ALPHAS)
    assert parse_run("eikonal_landcover_duration:0.25:0.5") == ("eikonal_landcover_duration", 0.25, 0.5)
    assert parse_run("bathtub") == ("bathtub", None, None)


def _result(tid: int, group: str, runs: dict) -> dict:
    return {"tile_id": str(tid), "group": group, "runs": {
        k: {"variant": v, "parameter": p, "alpha": a,
            "vs_sfincs": {"matched_km2": m, "model_only_km2": o, "sfincs_only_km2": s, "HT": 0, "FAR": 0,
                          "CSI": m / (m + o + s), "bias": (m + o) / (m + s)}}
        for k, (v, p, a, m, o, s) in runs.items()}}


def test_run_tables_pool_per_group_and_select_the_best_run() -> None:
    results = {}
    for tid, group in ((1, "calibration"), (2, "calibration"), (3, "validation")):
        results[tid] = _result(tid, group, {
            "eikonal_landcover_fsf0.2": ("eikonal_landcover", 0.2, None, 8, 4, 1),
            "eikonal_landcover_fsf0.3": ("eikonal_landcover", 0.3, None, 8, 1, 2),
            "bathtub_connected": ("bathtub_connected", None, None, 6, 6, 1),
        })
    per_tile, pooled = run_tables(results)
    assert len(per_tile) == 9
    cal = pooled[(pooled.group == "calibration") & (pooled.run_key == "eikonal_landcover_fsf0.3")].iloc[0]
    assert cal.n_tiles_scored == 2 and np.isclose(cal.CSI, 16 / 22)
    sel = select_runs(pooled)
    lc = sel["variants"]["eikonal_landcover"]
    assert lc["run_key"] == "eikonal_landcover_fsf0.3" and lc["parameter"] == 0.3 and lc["alpha"] is None
    assert lc["validation"]["n_tiles_scored"] == 1
    assert sel["best_variant"] == "eikonal_landcover"
    assert select_runs(pooled, "bias")["variants"]["eikonal_landcover"]["run_key"] == "eikonal_landcover_fsf0.3"


def test_study_friction_settings() -> None:
    sys.path.insert(0, str(REPO_ROOT / "calibration_studies"))
    from run_sweep_budget_study import friction_settings

    flooding = {"friction_source": "landcover", "friction_scale_factor": 0.25, "uniform_attenuation_m_per_km": 0.2,
                "resolution": 30}
    assert friction_settings(flooding, None, None) == {"friction_scale_factor": 7.5, "uniform_friction": None}
    uniform = friction_settings(flooding, "uniform", None)
    assert uniform["friction_scale_factor"] == 1.0 and np.isclose(uniform["uniform_friction"], 0.006)
    assert np.isclose(friction_settings(flooding, "uniform", 0.1)["uniform_friction"], 0.003)


def test_depth_accuracy_pools_exact_differences(tmp_path) -> None:
    from calibration_paths import DEPTH_ACCURACY_NAME
    from calibration_results import depth_accuracy_table
    from flood_agreement import WET_THRESHOLD_M, depth_diff_hists

    rng = np.random.default_rng(1)
    results, truth = {}, []
    for tid in (1, 2):
        sfincs = np.where(rng.random(4000) < 0.7, rng.uniform(0, 3.5, 4000), np.nan)
        model = np.clip(np.nan_to_num(sfincs) + rng.normal(0, 0.3, 4000), 0, None)
        s_wet, m_wet = np.nan_to_num(sfincs) > WET_THRESHOLD_M, model > WET_THRESHOLD_M
        h = depth_diff_hists(model, sfincs, m_wet, s_wet, np.ones(4000))
        out = tmp_path / str(tid) / "outputs"
        out.mkdir(parents=True)
        np.savez_compressed(out / DEPTH_ACCURACY_NAME, **{"r__area": h})
        results[tid] = {"group": "calibration",
                        "runs": {"r": {"variant": "eikonal_landcover", "parameter": 0.25, "alpha": None}}}
        truth.append(np.abs(model - np.nan_to_num(sfincs))[s_wet])
    table = depth_accuracy_table(tmp_path, results)
    row = table[(table.group == "all") & (table.cells == "reference_wet") & (table.sfincs_class_m == "all")].iloc[0]
    err = np.concatenate(truth)
    assert row.n == len(err) and row.n_tiles == 2
    for tau in (0.1, 0.2, 0.5):
        # 1 cm bins: a cell within tau of a bin edge may land either side
        assert abs(row[f"within_{tau:g}m"] - (err <= tau).mean()) < 0.01
    either = table[(table.group == "all") & (table.cells == "either_wet")].iloc[0]
    assert either.n >= row.n
