"""Detect per-tile round convergence from sweep-budget per-sweep traces.

A round converges when the max of its 4 sweep_max_change_raw values <= epsilon,
as in solve_eikonal_dense; no convergence within N_COMPLETE_ROUNDS is right-censored.
Also holds the trace reader and band helpers shared by the calibration band plots
(plot_sweep_calibration_bands.py, plot_obstacle_coupling_calibration_bands.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SWEEPS_PER_ROUND = 4  # one full cycle of eikonal._ORTHANT_ORDER
N_COMPLETE_ROUNDS = 100  # = run_sweep_budget_study.MAX_ROUNDS_CEILING


def read_trace_csv(csv_path: Path, required_col: str | None = None) -> pd.DataFrame | None:
    """Read one tile's study trace CSV; None while it is still being written (empty).

    Also None when it has no rows or lacks `required_col` (e.g. an error-only CSV).
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None  # still being written
    if df.empty or (required_col is not None and required_col not in df.columns):
        return None
    return df


def band_stats(long_df: pd.DataFrame, value_col: str, by: str) -> pd.DataFrame:
    """Per-`by` (round / outer iteration) median, 2nd/98th percentile, min, max and count of `value_col`."""
    g = long_df.groupby(by)[value_col]
    return pd.DataFrame({
        "median": g.median(), "p02": g.quantile(0.02), "p98": g.quantile(0.98),
        "min": g.min(), "max": g.max(), "n": g.count(),
    })


def plot_band(ax, s: pd.DataFrame, color: str, label: str, floor: float | None = None,
              show_p02: bool = False, range_label: str | None = None) -> None:
    """Draw a min-max band, the median and the 98th (optionally 2nd) percentile of `band_stats` output.

    `floor` clips the values for a log axis; `range_label` labels the band in the legend.
    """
    lo, hi, med, p98 = s["min"], s["max"], s["median"], s["p98"]
    if floor is not None:
        lo, hi, med, p98 = (x.clip(lower=floor) for x in (lo, hi, med, p98))
    band_kw = {} if range_label is None else {"label": range_label}
    ax.fill_between(s.index, lo, hi, alpha=0.15, color=color, **band_kw)
    ax.plot(s.index, med, color=color, linewidth=2, label=f"{label} - median")
    ax.plot(s.index, p98, color=color, linewidth=1.3, linestyle="--", label=f"{label} - 98th percentile")
    if show_p02:
        p02 = s["p02"].clip(lower=floor) if floor is not None else s["p02"]
        ax.plot(s.index, p02, color=color, linewidth=1.3, linestyle="-.", label=f"{label} - 2nd percentile")


def _tile_convergence(csv_path: Path, epsilon: float) -> dict | None:
    df = read_trace_csv(csv_path, "sweep_max_change_raw")
    if df is None or len(df) < SWEEPS_PER_ROUND:
        return None  # unreadable, or a dry tile (1 row)

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


