"""Shared flood-extent agreement primitives, used by postprocess_tile_summary.py
to compute raw per-tile contingency counts and depth-agreement stats, and by
compute_calibration_metrics.py/compute_metrics_overview_table.py to pool them
into HT/FAR/CSI/bias and depth-error metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

WET_THRESHOLD_M = 0.10  # postprocessing/reporting's "flooded" cutoff, independent
# from src/flood_model.py's MIN_FLOOD_DEPTH_M (the solver's own internal
# connectivity/flood-path threshold).

# Depth-agreement bin edges (metres), shared between postprocess_tile_summary.py
# (builds the per-tile joint histograms) and plot_validation_results.py (pools
# and plots them).
#
# FINE: pooled depth-correlation density plot - 0.1m steps, 0 to 3m, plus
# implicit <0/>3m catch-all bins added by depth_joint_hist.
DEPTH_CORR_FINE_EDGES = np.arange(0.0, 3.01, 0.1)

# CATEGORY: coarser 0.1-1.5m/0.2m-wide scheme for the depth-bin alignment
# heatmap, plus implicit <0.1m/>1.5m catch-all bins.
DEPTH_CATEGORY_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def depth_joint_hist(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """2D histogram of (x, y) using the same `edges` on both axes, with an
    implicit catch-all bin below the first edge and above the last.
    Shape: (len(edges)+1, len(edges)+1).
    """
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    hist, _, _ = np.histogram2d(x, y, bins=[full_edges, full_edges])
    return hist


def depth_corr_sufficient_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """n and the five sums needed to compute a pooled Pearson r later
    (pearson_r_from_sums) from many tiles' (x, y) pairs without storing the
    raw pairs."""
    return {
        "n": int(x.size),
        "sum_x": float(x.sum()), "sum_y": float(y.sum()),
        "sum_x2": float(np.square(x).sum()), "sum_y2": float(np.square(y).sum()),
        "sum_xy": float((x * y).sum()),
    }


def pearson_r_from_sums(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    """Pearson correlation from pooled sufficient statistics (see
    depth_corr_sufficient_stats)."""
    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return float(num / den) if den > 0 else float("nan")


def pool_depth_joint(base_dir: Path, model: str, exclude_tile_ids: set[int] | None = None) -> dict:
    """Pools every tile's {model}_depth_joint (postprocess_tile_summary.py:
    cell-level SFINCS-vs-model depth sufficient stats + 2D histograms, at
    every mutually-wet cell) into one dict: summed n/sums (for
    pearson_r_from_sums / depth_error_metrics_from_pooled) and
    elementwise-summed hist_fine/hist_category matrices.
    """
    n = sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    hist_fine = hist_category = None
    exclude_tile_ids = exclude_tile_ids or set()
    for p in sorted(base_dir.glob(f"*/outputs/summary_{model}.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if int(d["tile_id"]) in exclude_tile_ids:
            continue
        joint = d.get(f"{model}_depth_joint")
        if not joint:
            continue
        n += joint["n"]
        sum_x += joint["sum_x"]
        sum_y += joint["sum_y"]
        sum_x2 += joint["sum_x2"]
        sum_y2 += joint["sum_y2"]
        sum_xy += joint["sum_xy"]
        hf = np.asarray(joint["hist_fine"])
        hc = np.asarray(joint["hist_category"])
        hist_fine = hf if hist_fine is None else hist_fine + hf
        hist_category = hc if hist_category is None else hist_category + hc
    return {
        "n": n, "sum_x": sum_x, "sum_y": sum_y, "sum_x2": sum_x2, "sum_y2": sum_y2, "sum_xy": sum_xy,
        "hist_fine": hist_fine, "hist_category": hist_category,
    }


def depth_error_metrics_from_pooled(pooled: dict, within_band_m: float = 0.2) -> dict[str, float]:
    """Depth-agreement summary stats (model - SFINCS, pooled across tiles)
    from pool_depth_joint's output:
      - r: Pearson correlation (pattern/shape agreement).
      - bias_m, rmse_m: mean error and RMSE, exact from the pooled
        sufficient stats.
      - median_error_m, pct_within_{band}m: from the pooled hist_fine 2D
        histogram (DEPTH_CORR_FINE_EDGES, 0.1m bins), robust to the long
        right tail of a few very deep cells.
    Every field is NaN if n == 0.
    """
    band_key = f"pct_within_{within_band_m:g}m"
    n = pooled["n"]
    if not n:
        return {"r": float("nan"), "bias_m": float("nan"), "rmse_m": float("nan"),
                "median_error_m": float("nan"), band_key: float("nan")}

    r = pearson_r_from_sums(n, pooled["sum_x"], pooled["sum_y"], pooled["sum_x2"], pooled["sum_y2"], pooled["sum_xy"])
    bias_m = (pooled["sum_y"] - pooled["sum_x"]) / n
    mse = (pooled["sum_y2"] - 2 * pooled["sum_xy"] + pooled["sum_x2"]) / n
    rmse_m = float(np.sqrt(max(mse, 0.0)))

    hist_fine = pooled["hist_fine"]
    median_error_m = pct_within = float("nan")
    if hist_fine is not None:
        core = np.asarray(hist_fine)[1:-1, 1:-1]  # drop <0m/>3m catch-alls
        centers = (DEPTH_CORR_FINE_EDGES[:-1] + DEPTH_CORR_FINE_EDGES[1:]) / 2
        err = centers[np.newaxis, :] - centers[:, np.newaxis]  # [sfincs_bin, model_bin] -> model - sfincs
        weights = core.ravel()
        errs = err.ravel()
        total = weights.sum()
        if total > 0:
            order = np.argsort(errs)
            cum = np.cumsum(weights[order])
            median_error_m = float(errs[order][np.searchsorted(cum, total / 2.0)])
            pct_within = float(100 * weights[np.abs(errs) <= within_band_m].sum() / total)

    return {"r": float(r), "bias_m": float(bias_m), "rmse_m": rmse_m,
            "median_error_m": median_error_m, band_key: pct_within}


def prune_to_ocean_connected(
    flooded: np.ndarray, mask: np.ndarray,
    ocean_code: int = 1, land_code: int = 0, river_code: int | None = None,
) -> np.ndarray:
    """Keeps only the 8-connected components of `flooded` that touch a
    coastline cell (an ocean cell within 1px of land or river) - discards
    naive-bathtub flooding in interior basins with no hydraulic path to the
    sea. Same "label components, keep those touching a dilated coastline"
    logic as src/flood_model.py's prune_to_coast_connected, but without that
    module's edge-connectivity requirement - matches
    run_eikonal_on_sfincs_subgrid.py's sfincs_domain_coastline_mask instead,
    since SFINCS's rectangular UTM subgrid has no real wet edge.
    """
    landlike = mask == land_code
    if river_code is not None:
        landlike = landlike | (mask == river_code)
    coastline = ndimage.binary_dilation(landlike, structure=_STRUCTURE_8) & (mask == ocean_code)
    if not coastline.any() or not flooded.any():
        return np.zeros_like(flooded)
    dilated_coast = ndimage.binary_dilation(coastline, structure=_STRUCTURE_8)
    labels, _ = ndimage.label(flooded, structure=_STRUCTURE_8)
    touching = set(np.unique(labels[dilated_coast])) - {0}
    if not touching:
        return np.zeros_like(flooded)
    return np.isin(labels, list(touching))


def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted (matched, model_only, benchmark_only) km2: matched = both
    wet, model_only = model wet and not benchmark (over-prediction),
    benchmark_only = benchmark wet and not model (under-prediction)."""
    d = domain_mask
    matched = float(weight[d & model_wet & benchmark_wet].sum())
    model_only = float(weight[d & model_wet & ~benchmark_wet].sum())
    benchmark_only = float(weight[d & ~model_wet & benchmark_wet].sum())
    return matched, model_only, benchmark_only


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(matched: float, model_only: float, benchmark_only: float) -> dict[str, float]:
    """HT (hit rate), FAR, CSI, bias from contingency counts already summed
    across tiles."""
    tp, fp, fn = matched, model_only, benchmark_only
    return {
        "HT": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
