"""The calibration's per-run solver table (collect_solver_diagnostics.py)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
from collect_solver_diagnostics import sfincs_timing, solver_rows, summarize  # noqa: E402

EIKONAL = {"variant": "eikonal_landcover", "parameter": 0.35, "alpha": None, "solve_s": 12.3, "runtime_s": 13.0,
           "solver": {"obstacle_coupling": True, "outer_iterations_used": 3, "outer_converged": True,
                      "inner_rounds_per_outer": [9, 4, 2], "inner_converged_per_outer": [True, True, True],
                      "newly_blocked_per_outer": [500, 12, 0], "last_blocking_outer": 2}}
BATHTUB = {"variant": "bathtub", "parameter": None, "alpha": None, "solve_s": 0.4, "runtime_s": 0.6,
           "solver": {"model": "bathtub"}}


def test_eikonal_run_reports_rounds_blocking_and_solve_time() -> None:
    rows = {r["run_key"]: r for r in solver_rows("17", {"eik": EIKONAL, "bt": BATHTUB})}
    eik = rows["eik"]
    assert (eik["inner_rounds_per_outer"], eik["max_inner_rounds"], eik["all_inner_converged"]) == ("9 4 2", 9, True)
    assert (eik["last_blocking_outer"], eik["solve_s"]) == (2, 12.3)
    assert rows["bt"]["max_inner_rounds"] is None and rows["bt"]["solve_s"] == 0.4


def test_sfincs_timing_reads_the_log_summary(tmp_path) -> None:
    log = tmp_path / "sfincs.log"
    log.write_text(" Total time             :    423.643\n Total simulation time  :    422.886\n"
                   " Time in momentum       :    220.864 ( 52.2%)\n Average time step (s)  :      5.920\n")
    assert sfincs_timing(log) == {"sfincs_s": 423.643, "sfincs_mean_dt_s": 5.92}
    assert sfincs_timing(tmp_path / "missing.log") == {"sfincs_s": None, "sfincs_mean_dt_s": None}


def test_summary_has_one_row_per_variant_and_counts_unconverged_solves() -> None:
    slow = {**EIKONAL, "solver": {**EIKONAL["solver"], "outer_converged": False,
                                  "inner_converged_per_outer": [True, False, True]}}
    table = pd.DataFrame([{**r, "sfincs_s": 400.0} for r in solver_rows("1", {"a": EIKONAL, "b": BATHTUB})]
                         + [{**r, "sfincs_s": None} for r in solver_rows("2", {"a": slow})])
    summary = summarize(table).set_index("variant")
    assert sorted(summary.index) == ["bathtub", "eikonal_landcover", "sfincs"]
    eik = summary.loc["eikonal_landcover"]
    assert (eik["n_runs"], eik["n_inner_not_converged"], eik["n_outer_not_converged"]) == (2, 1, 1)
    assert summary.loc["bathtub", "n_runs"] == 1
    assert (summary.loc["sfincs", "n_runs"], summary.loc["sfincs", "solve_s_median"]) == (1, 400.0)  # tile 2: no log
