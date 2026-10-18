"""Detect per-tile round convergence from sweep-budget per-sweep traces.

A round converges when the max of its 4 sweep_max_change_raw values <= epsilon,
as in solve_eikonal_dense; no convergence within N_COMPLETE_ROUNDS is right-censored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SWEEPS_PER_ROUND = 4
N_COMPLETE_ROUNDS = 100  # = run_sweep_budget_study.MAX_ROUNDS_CEILING


def _tile_convergence(csv_path: Path, epsilon: float) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None  # still being written
    if df.empty or "sweep_max_change_raw" not in df.columns:
        return None
    if len(df) < SWEEPS_PER_ROUND:
        return None  # dry tile (1 row)

    n_cells = int(df["n_cells"].iloc[0])
    tile_id = int(df["tile"].iloc[0])

    max_change = df["sweep_max_change_raw"].to_numpy()
    cum_time = df["cum_time_s"].to_numpy()
    n_rounds_available = min(N_COMPLETE_ROUNDS, len(max_change) // SWEEPS_PER_ROUND)

    n_rounds_used = None
    time_to_converge_s = None
    for r in range(1, n_rounds_available + 1):
        round_max_change = max_change[(r - 1) * SWEEPS_PER_ROUND: r * SWEEPS_PER_ROUND].max()
        if round_max_change <= epsilon:
            n_rounds_used = r
            time_to_converge_s = float(cum_time[r * SWEEPS_PER_ROUND - 1])
            break

    return {
        "tile": tile_id,
        "n_cells": n_cells,
        "converged": n_rounds_used is not None,
        "n_rounds_used": n_rounds_used,
        "time_to_converge_s": time_to_converge_s,
        "n_rounds_available": n_rounds_available,
    }


def collect(sweep_budget_dir: Path, epsilon: float) -> pd.DataFrame:
    """Return one convergence record per readable wet tile CSV."""
    records = []
    csvs = sorted(sweep_budget_dir.glob("*.csv"))
    for p in csvs:
        rec = _tile_convergence(p, epsilon)
        if rec is not None:
            records.append(rec)
    return pd.DataFrame.from_records(records)


