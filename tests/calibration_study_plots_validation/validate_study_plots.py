"""The solver-study figures read either solve pass of a two-pass (correction B) trace."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIES = REPO_ROOT / "calibration_studies"
sys.path.insert(0, str(STUDIES))

from plot_sweep_budget_convergence import collect, read_trace_csv  # noqa: E402

EPSILON = 0.03


def _sweep_rows(tile: int, solve_pass: str, n_rounds: int) -> list[dict]:
    # Converges in the last round: its max change drops to 0.01 m.
    rows = []
    for s in range(1, 4 * n_rounds + 1):
        rows.append({"tile": tile, "solve_pass": solve_pass, "n_cells": 1000, "sweep_count": s, "cum_time_s": 0.1 * s,
                     "sweep_max_change_raw": 0.01 if s > 4 * (n_rounds - 1) else 1.0, "n_inundated": 10 * s,
                     "max_depth_change_abs": 0.5, "n_newly_flooded": 3, "status": "ok"})
    return rows


def _obstacle_rows(tile: int, n_first: int, n_duration: int) -> list[dict]:
    rows = [{"tile": tile, "solve_pass": "baseline", "n_outer": 0, "n_inundated": 50, "pct_newly_blocked": 0.0,
             "status": "ok"}]
    for solve_pass, n in (("first", n_first), ("duration", n_duration)):
        rows += [{"tile": tile, "solve_pass": solve_pass, "n_outer": k, "n_inundated": 40,
                  "pct_newly_blocked": 1.0 / k, "status": "ok"} for k in range(1, n + 1)]
    return rows


def _write_traces(tmp_path: Path) -> tuple[Path, Path]:
    sweep_dir, obstacle_dir = tmp_path / "sweep_budget_uniform", tmp_path / "obstacle_coupling_uniform"
    sweep_dir.mkdir()
    obstacle_dir.mkdir()
    for tile in (1, 2, 3):
        pd.DataFrame(_sweep_rows(tile, "first", 3) + _sweep_rows(tile, "duration", 7)).to_csv(
            sweep_dir / f"{tile}.csv", index=False)
        pd.DataFrame(_obstacle_rows(tile, 2, 4)).to_csv(obstacle_dir / f"{tile}.csv", index=False)
    return sweep_dir, obstacle_dir


def test_each_solve_pass_is_read_on_its_own(tmp_path) -> None:
    sweep_dir, _ = _write_traces(tmp_path)
    assert set(read_trace_csv(sweep_dir / "1.csv", solve_pass="first")["solve_pass"]) == {"first"}
    assert set(read_trace_csv(sweep_dir / "1.csv")["solve_pass"]) == {"duration"}
    assert set(collect(sweep_dir, EPSILON, "first")["n_rounds_used"]) == {3}
    assert set(collect(sweep_dir, EPSILON, "duration")["n_rounds_used"]) == {7}


def test_figures_are_written_per_solve_pass(tmp_path) -> None:
    sweep_dir, obstacle_dir = _write_traces(tmp_path)
    fig_dir = tmp_path / "figures"
    for solve_pass in ("first", "duration"):
        for script, data_dir in (("plot_sweep_calibration_bands.py", sweep_dir),
                                 ("plot_sweep_time_vs_size.py", sweep_dir),
                                 ("plot_obstacle_coupling_calibration_bands.py", obstacle_dir)):
            subprocess.run([sys.executable, str(STUDIES / script), str(data_dir), "--fig-dir", str(fig_dir),
                            "--solve-pass", solve_pass], check=True, capture_output=True)
    assert sorted(p.name for p in fig_dir.glob("*.png")) == sorted(
        f"{stem}_{p}.png" for p in ("first", "duration")
        for stem in ("sweep_calibration_bands", "sweep_time_vs_size", "obstacle_coupling_calibration_bands"))
