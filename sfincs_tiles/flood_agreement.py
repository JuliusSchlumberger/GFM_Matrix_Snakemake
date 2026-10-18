"""Flood-extent and depth agreement primitives, poolable across tiles by summing."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

WET_THRESHOLD_M = 0.10  # reporting "flooded" cutoff; independent of flood_model.MIN_FLOOD_DEPTH_M

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


def prune_to_ocean_connected(
    flooded: np.ndarray, mask: np.ndarray,
    ocean_code: int = 1, land_code: int = 0, river_code: int | None = None,
) -> np.ndarray:
    """Discard flooded cells with no hydraulic path to the sea.

    Keeps 8-connected components touching a coastline cell (ocean within 1 px of
    land or river). As flood_model.prune_to_coast_connected but without its
    edge-connectivity requirement: SFINCS's rectangular UTM subgrid has no real wet edge.
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
