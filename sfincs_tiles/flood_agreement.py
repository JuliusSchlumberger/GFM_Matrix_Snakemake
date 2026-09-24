"""Shared flood-extent agreement primitives - used by postprocess_tile_summary.py
(per-tile, on the HPC node that already has the rasters loaded) to compute
raw contingency counts, and by compute_calibration_metrics.py (a cheap
JSON-only aggregation pass) to turn summed counts into HT/FAR/CSI/bias.

Moved out of compute_calibration_metrics.py (2026-09-24, user direction):
computing per-tile counts there meant re-opening/re-reprojecting every
tile's rasters a second time, sequentially, from one machine over the
network mount - wasteful when postprocess_tile_summary.py already has those
exact same arrays in memory, per-tile, in parallel, on the HPC node that
just ran that tile. See module docstrings of both callers.
"""

from __future__ import annotations

import numpy as np

WET_THRESHOLD_M = 0.05


def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted (matched, model_only, benchmark_only) km2 - matched = both
    wet (agreement), model_only = model wet AND NOT benchmark wet
    (over-prediction), benchmark_only = NOT model wet AND benchmark wet
    (under-prediction). Same tp/fp/fn convention as src/validation.py's own
    confusion_counts, renamed here for direct readability in the per-tile
    JSON/CSV output. tn (neither wet) is never needed - HT/FAR/CSI/bias
    don't use it."""
    d = domain_mask
    matched = float(weight[d & model_wet & benchmark_wet].sum())
    model_only = float(weight[d & model_wet & ~benchmark_wet].sum())
    benchmark_only = float(weight[d & ~model_wet & benchmark_wet].sum())
    return matched, model_only, benchmark_only


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(matched: float, model_only: float, benchmark_only: float) -> dict[str, float]:
    """HT, FAR, CSI, bias - same formulas as src/validation.py's
    metrics_from_counts (HT here == that module's "HR", same hit-rate
    definition, renamed to match this comparison's own terminology). Only
    ever meaningful on counts already summed across many tiles - a ratio
    computed from one tile's own small counts and averaged across tiles
    would let small, lightly-flooded tiles dominate the average as much as
    large, heavily-flooded ones."""
    tp, fp, fn = matched, model_only, benchmark_only
    return {
        "HT": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
