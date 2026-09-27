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

# Depth-agreement bin edges (metres), shared between postprocess_tile_summary.py
# (which builds the per-tile joint histograms) and plot_validation_results.py
# (which pools and plots them) - kept here so both always agree on what a
# "bin" means without passing edges around at runtime.
#
# FINE: for the pooled depth-correlation density plot - 0.1m steps, 0 to 3m,
# with implicit <0 (never occurs - both inputs are already wet-thresholded)
# and >3m catch-all bins added by depth_joint_hist.
DEPTH_CORR_FINE_EDGES = np.arange(0.0, 3.01, 0.1)

# CATEGORY: the coarser 0.1-1.5m/0.2m-wide scheme used for the depth-bin
# alignment heatmap - matches the range already described in the manuscript
# text this supports. Implicit <0.1m and >1.5m catch-all bins are added by
# depth_joint_hist, same as above.
DEPTH_CATEGORY_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def depth_joint_hist(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """2D histogram of (x, y) using the same `edges` on both axes, with an
    implicit catch-all bin below the first edge and above the last - so
    every finite (x, y) pair lands somewhere, never silently dropped for
    falling outside the named range. Shape: (len(edges)+1, len(edges)+1).
    """
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    hist, _, _ = np.histogram2d(x, y, bins=[full_edges, full_edges])
    return hist


def depth_corr_sufficient_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """n and the five sums needed to compute a pooled Pearson r later
    (pearson_r_from_sums) from many tiles' worth of (x, y) pairs without
    ever storing the raw pairs themselves - same "pool first, ratio last"
    reasoning as metrics_from_counts above, applied to a correlation
    instead of a contingency-table ratio."""
    return {
        "n": int(x.size),
        "sum_x": float(x.sum()), "sum_y": float(y.sum()),
        "sum_x2": float(np.square(x).sum()), "sum_y2": float(np.square(y).sum()),
        "sum_xy": float((x * y).sum()),
    }


def pearson_r_from_sums(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    """Pearson correlation from pooled sufficient statistics (see
    depth_corr_sufficient_stats) - algebraically identical to computing it
    from the raw pairs directly, but computable after summing many tiles'
    stats together."""
    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return float(num / den) if den > 0 else float("nan")


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
