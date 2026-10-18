"""Shared flood-extent agreement primitives: compute_calibration_result.py uses
them for the per-tile contingency counts and depth-agreement stats,
calibration_results.py to pool those into HT/FAR/CSI/bias and depth-error metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

WET_THRESHOLD_M = 0.10  # postprocessing/reporting's "flooded" cutoff, independent
# from src/flood_model.py's MIN_FLOOD_DEPTH_M (the solver's own internal
# connectivity/flood-path threshold).

# Depth-agreement bin edges (metres), shared between compute_calibration_result.py
# (builds the per-tile joint histograms) and plot_validation_results.py (plots
# them, pooled by calibration_results.py).
#
# FINE: pooled depth-correlation density plot - 0.1m steps, 0 to 3m, plus
# implicit <0/>3m catch-all bins added by depth_joint_hist.
DEPTH_CORR_FINE_EDGES = np.arange(0.0, 3.01, 0.1)

# CATEGORY: coarser 0.1-1.5m/0.2m-wide scheme for the depth-bin alignment
# heatmap, plus implicit <0.1m/>1.5m catch-all bins.
DEPTH_CATEGORY_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def depth_joint_hist(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Build a 2D histogram of (x, y) using the same bin `edges` on both axes.

    Adds an implicit catch-all bin below the first edge and above the last, so the
    result always covers the full range of x and y.

    Parameters
    ----------
    x : numpy.ndarray
        Values for the first axis (e.g. SFINCS depths, metres).
    y : numpy.ndarray
        Values for the second axis (e.g. model depths, metres).
    edges : numpy.ndarray
        Shared bin edges for both axes (e.g. DEPTH_CORR_FINE_EDGES).

    Returns
    -------
    numpy.ndarray
        2D histogram counts, shape (len(edges)+1, len(edges)+1).
    """
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    hist, _, _ = np.histogram2d(x, y, bins=[full_edges, full_edges])
    return hist


def depth_corr_sufficient_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute n and the five sums needed to pool a Pearson r across tiles later.

    Lets many tiles' (x, y) depth pairs be summed and fed to `pearson_r_from_sums`
    without ever storing the raw per-cell pairs.

    Parameters
    ----------
    x : numpy.ndarray
        Values for the first variable (e.g. SFINCS depths, metres).
    y : numpy.ndarray
        Values for the second variable (e.g. model depths, metres).

    Returns
    -------
    dict[str, float]
        ``n``, ``sum_x``, ``sum_y``, ``sum_x2``, ``sum_y2``, ``sum_xy``.
    """
    return {
        "n": int(x.size),
        "sum_x": float(x.sum()), "sum_y": float(y.sum()),
        "sum_x2": float(np.square(x).sum()), "sum_y2": float(np.square(y).sum()),
        "sum_xy": float((x * y).sum()),
    }


def pearson_r_from_sums(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    """Compute the Pearson correlation from pooled sufficient statistics.

    Parameters
    ----------
    n : float
        Number of (x, y) pairs pooled.
    sum_x : float
        Sum of x.
    sum_y : float
        Sum of y.
    sum_x2 : float
        Sum of x squared.
    sum_y2 : float
        Sum of y squared.
    sum_xy : float
        Sum of x times y.

    Returns
    -------
    float
        Pearson r, or NaN if the denominator is zero (e.g. n == 0 or no variance).
    """
    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return float(num / den) if den > 0 else float("nan")


def depth_error_metrics_from_pooled(pooled: dict, within_band_m: float = 0.2) -> dict[str, float]:
    """Summarise depth agreement (model - SFINCS, pooled across tiles).

    `r`, `bias_m`, `rmse_m` come exactly from the pooled sufficient stats;
    `median_error_m` and `pct_within_{band}m` come from the pooled `hist_fine` 2D
    histogram (DEPTH_CORR_FINE_EDGES, 0.1m bins) and are robust to the long right
    tail of a few very deep cells.

    Parameters
    ----------
    pooled : dict
        Pooled depth-joint stats, as built by calibration_results.pooled_depth_joint
        (``n``, ``sum_x``/``sum_y``/``sum_x2``/``sum_y2``/``sum_xy``, ``hist_fine``).
    within_band_m : float, optional
        Error band (metres) for the ``pct_within_{band}m`` key, by default 0.2.

    Returns
    -------
    dict[str, float]
        ``r``, ``bias_m``, ``rmse_m``, ``median_error_m``, ``pct_within_{band}m``.
        Every field is NaN if ``pooled["n"]`` is 0.
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
    """Discard flooded cells with no hydraulic path to the sea.

    Keeps only the 8-connected components of `flooded` that touch a coastline cell
    (an ocean cell within 1px of land or river), dropping naive-bathtub flooding in
    interior basins. Same "label components, keep those touching a dilated
    coastline" logic as src/flood_model.py's prune_to_coast_connected, but without
    that module's edge-connectivity requirement, since SFINCS's rectangular UTM
    subgrid has no real wet edge.

    Parameters
    ----------
    flooded : numpy.ndarray
        Boolean array of candidate wet cells to prune.
    mask : numpy.ndarray
        Integer land/ocean/river classification, same shape as `flooded`.
    ocean_code : int, optional
        Mask value for ocean cells, by default 1.
    land_code : int, optional
        Mask value for land cells, by default 0.
    river_code : int or None, optional
        Mask value for river cells, also treated as land-like for the coastline;
        by default None (no river class).

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `flooded`, with interior-only components
        removed.
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
    """Compute weighted contingency areas between model and benchmark wet masks.

    Parameters
    ----------
    model_wet : numpy.ndarray
        Boolean array, cells the model calls wet.
    benchmark_wet : numpy.ndarray
        Boolean array, cells the benchmark calls wet.
    domain_mask : numpy.ndarray
        Boolean array restricting the comparison to valid cells.
    weight : numpy.ndarray
        Per-cell weight (e.g. area km2) to sum.

    Returns
    -------
    tuple[float, float, float]
        ``(matched, model_only, benchmark_only)``: matched = both wet, model_only =
        model wet and not benchmark (over-prediction), benchmark_only = benchmark
        wet and not model (under-prediction).
    """
    d = domain_mask
    matched = float(weight[d & model_wet & benchmark_wet].sum())
    model_only = float(weight[d & model_wet & ~benchmark_wet].sum())
    benchmark_only = float(weight[d & ~model_wet & benchmark_wet].sum())
    return matched, model_only, benchmark_only


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(matched: float, model_only: float, benchmark_only: float) -> dict[str, float]:
    """Compute HT (hit rate), FAR, CSI and bias from contingency counts.

    Parameters
    ----------
    matched : float
        Area (or count) where model and benchmark agree wet, e.g. from
        `confusion_counts`, already summed across tiles.
    model_only : float
        Area where only the model is wet (over-prediction).
    benchmark_only : float
        Area where only the benchmark is wet (under-prediction).

    Returns
    -------
    dict[str, float]
        ``HT``, ``FAR``, ``CSI``, ``bias``. Each is NaN if its denominator is zero.
    """
    tp, fp, fn = matched, model_only, benchmark_only
    return {
        "HT": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
