"""Flood-extent and depth agreement primitives, poolable across tiles by summing."""

from __future__ import annotations

import numpy as np

WET_THRESHOLD_M = 0.10  # reporting "flooded" cutoff; the model itself floods any depth > 0

# Depth bin edges (m); depth_joint_hist adds <first/>last catch-all bins.
DEPTH_CORR_FINE_EDGES = np.arange(0.0, 3.01, 0.1)
DEPTH_CATEGORY_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def depth_joint_hist(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Build a 2D histogram of (x, y) on shared `edges`, shape (len(edges)+1,) * 2.

    Adds catch-all bins below the first and above the last edge.
    """
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    hist, _, _ = np.histogram2d(x, y, bins=[full_edges, full_edges])
    return hist


def depth_corr_sufficient_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute n and the five sums that let a Pearson r be pooled across tiles."""
    return {
        "n": int(x.size),
        "sum_x": float(x.sum()), "sum_y": float(y.sum()),
        "sum_x2": float(np.square(x).sum()), "sum_y2": float(np.square(y).sum()),
        "sum_xy": float((x * y).sum()),
    }


def pearson_r_from_sums(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    """Compute Pearson r from pooled sufficient statistics; NaN if there is no variance."""
    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return float(num / den) if den > 0 else float("nan")


def depth_error_metrics_from_pooled(pooled: dict, within_band_m: float = 0.2) -> dict[str, float]:
    """Summarise pooled depth agreement (error = model - SFINCS, metres).

    r/bias_m/rmse_m are exact from the sufficient stats; median_error_m and
    pct_within_{band}m come from the 0.1 m `hist_fine` bins, robust to the long
    right tail of a few very deep cells. All NaN if ``pooled["n"]`` is 0.
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


# Twin of src/validation.py's confusion_counts / metrics_from_counts: hydromt-sfincs-dev can't import src/
# (src/config_utils needs hydromt 0.9). Pinned by tests/metric_parity_validation/validate_sfincs_src_parity.py.
def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float]:
    """Return weighted (matched, model_only, benchmark_only) areas within `domain_mask`."""
    d = domain_mask
    matched = float(weight[d & model_wet & benchmark_wet].sum())
    model_only = float(weight[d & model_wet & ~benchmark_wet].sum())
    benchmark_only = float(weight[d & ~model_wet & benchmark_wet].sum())
    return matched, model_only, benchmark_only


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(matched: float, model_only: float, benchmark_only: float) -> dict[str, float]:
    """Compute HT (hit rate), FAR, CSI and bias; each NaN if its denominator is zero."""
    tp, fp, fn = matched, model_only, benchmark_only
    return {
        "HT": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }


# Exact depth differences (model - SFINCS) as weighted 1 cm histograms, poolable by summing.
DIFF_EDGES = np.round(np.arange(-3.0, 3.001, 0.01), 2)  # m; catch-all bins beyond -3 / +3 m
SFINCS_CLASS_EDGES = (0.3, 0.5, 1.0, 1.5, 2.0, 3.0)  # m, from WET_THRESHOLD_M; the last class > 3 m
SFINCS_CLASS_LABELS = ("0.1-0.3", "0.3-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3")
CELL_KINDS = ("both_wet", "sfincs_only", "model_only")
TOLERANCES_M = (0.1, 0.2, 0.3, 0.5)


def depth_diff_hists(model: np.ndarray, sfincs: np.ndarray, model_wet: np.ndarray, sfincs_wet: np.ndarray,
                     weight: np.ndarray) -> np.ndarray:
    """Weighted histograms of model - SFINCS depth: (CELL_KINDS, SFINCS classes, DIFF_EDGES bins + 1).

    A one-sided cell is compared at the dry side's own sub-threshold depth (NaN counts as 0), so
    shares over all SFINCS-wet or all wet cells are exact. Class by SFINCS depth
    (SFINCS_CLASS_EDGES); model-only cells all fall in the first class.
    """
    m, s = np.nan_to_num(model, nan=0.0), np.nan_to_num(sfincs, nan=0.0)
    n_class, n_bin = len(SFINCS_CLASS_EDGES) + 1, len(DIFF_EDGES) + 1
    out = np.zeros((len(CELL_KINDS), n_class, n_bin))
    for k, sel in enumerate((model_wet & sfincs_wet, sfincs_wet & ~model_wet, model_wet & ~sfincs_wet)):
        idx = (np.digitize(s[sel], SFINCS_CLASS_EDGES) * n_bin
               + np.searchsorted(DIFF_EDGES, (m - s)[sel], side="right"))
        out[k] = np.bincount(idx, weights=weight[sel], minlength=n_class * n_bin).reshape(n_class, n_bin)
    return out


def diff_stats(hist: np.ndarray, edges: np.ndarray = DIFF_EDGES) -> dict:
    """Return within-tau shares, bias, MAE, RMSE and |error| quantiles from a 1 cm histogram.

    `hist` has catch-all bins at both ends: they count in the shares' denominator (as not
    within) and in share_beyond_3m, not in bias/MAE/RMSE.
    """
    h = np.asarray(hist, dtype=np.float64)[1:-1]
    total = float(np.asarray(hist).sum())
    if total <= 0:
        return {"n": 0.0}
    centres = (edges[:-1] + edges[1:]) / 2
    a = np.abs(centres)
    out = {"n": total, "share_beyond_3m": 1 - h.sum() / total,
           "bias_m": float((h * centres).sum() / h.sum()), "mae_m": float((h * a).sum() / h.sum()),
           "rmse_m": float(np.sqrt((h * centres ** 2).sum() / h.sum()))}
    inside = np.maximum(np.abs(edges[:-1]), np.abs(edges[1:]))  # a bin is within tau if both edges are
    for tau in TOLERANCES_M:
        out[f"within_{tau:g}m"] = float(h[inside <= tau + 1e-9].sum() / total)
    order = np.argsort(a)
    cum = np.cumsum(h[order]) / total
    for q, name in ((0.5, "p50"), (0.8, "tau80"), (0.9, "p90")):
        k = np.searchsorted(cum, q)
        out[f"{name}_abs_m"] = float(a[order][k]) if k < len(a) else np.nan
    return out
