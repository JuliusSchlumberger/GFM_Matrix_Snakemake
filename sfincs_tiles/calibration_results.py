"""Reads the final SFINCS calibration's per-tile results
({base}/{tile_id}/outputs/calibration_result.json, compute_calibration_result.py)
into the tables the analysis scripts plot from. Everything here is pure JSON
aggregation - no raster I/O.

  load_results(base_dir, tile_ids=None)  -> {tile_id: result dict}
  sweep_tables(results)                  -> (per_tile, pooled) friction-sweep tables
  tile_table(results, fsf, meta)         -> one row per tile at one friction point
  pooled_depth_joint(results, model, fsf)-> summed depth-joint stats/histograms
  metrics_overview(results, fsf)         -> pooled HC-bathtub vs EA-bathtub table

Counts are on the 30 m subgrid over SFINCS-active land only (see
compute_calibration_result.py); every metric is pooled by summing counts
across tiles first (flood_agreement.metrics_from_counts).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flood_agreement import depth_error_metrics_from_pooled, metrics_from_counts

RESULT_NAME = "calibration_result.json"
DISPLAY_LABEL = {"bathtub": "HC-bathtub", "eikonal": "EA-bathtub"}


def fsf_key(fsf: float) -> str:
    """Build the JSON key for one eikonal friction-sweep point.

    Parameters
    ----------
    fsf : float
        Friction scale factor.

    Returns
    -------
    str
        Key of the form ``fsf_{fsf:g}``, as used under ``result["eikonal"]``.
    """
    return f"fsf_{fsf:g}"


def load_results(base_dir: Path, tile_ids: list[str] | None = None) -> dict[int, dict]:
    """Load every tile's calibration_result.json into a dict keyed by tile id.

    Skips tiles whose result file is missing or unreadable rather than raising.

    Parameters
    ----------
    base_dir : Path
        Tile-results root containing ``{tile_id}/outputs/calibration_result.json``.
    tile_ids : list of str, optional
        Tile ids to load; if None (default), all tiles found under `base_dir` are loaded.

    Returns
    -------
    dict[int, dict]
        Mapping from tile id to its parsed calibration_result.json contents.
    """
    if tile_ids is None:
        paths = sorted(base_dir.glob(f"*/outputs/{RESULT_NAME}"))
    else:
        paths = [base_dir / str(t) / "outputs" / RESULT_NAME for t in tile_ids]
    out = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out[int(d["tile_id"])] = d
    return out


def _model_entry(result: dict, model: str, fsf: float | None) -> dict | None:
    if model == "bathtub":
        return result.get("bathtub")
    return (result.get("eikonal") or {}).get(fsf_key(fsf))


def sweep_tables(results: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build eikonal-vs-SFINCS extent-agreement tables across the friction sweep.

    Parameters
    ----------
    results : dict[int, dict]
        Per-tile results as returned by `load_results`.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(per_tile, pooled)``: one row per (tile, friction_scale_factor) with
        matched/only areas (km2) and HT/FAR/CSI/bias, and the same metrics pooled
        by summing counts across tiles per friction_scale_factor.
    """
    rows = []
    for tid, r in results.items():
        for key, e in (r.get("eikonal") or {}).items():
            v = e["vs_sfincs"]
            rows.append({"tile_id": tid, "friction_scale_factor": float(key[4:]),
                         "matched_km2": v["matched_km2"], "eikonal_only_km2": v["model_only_km2"],
                         "sfincs_only_km2": v["sfincs_only_km2"],
                         "HT": v["HT"], "FAR": v["FAR"], "CSI": v["CSI"], "bias": v["bias"]})
    per_tile = pd.DataFrame(rows).sort_values(["friction_scale_factor", "tile_id"]).reset_index(drop=True)
    pooled = []
    for fsf, g in per_tile.groupby("friction_scale_factor"):
        s = g[["matched_km2", "eikonal_only_km2", "sfincs_only_km2"]].sum()
        pooled.append({"friction_scale_factor": fsf, "n_tiles_scored": len(g), **s.to_dict(),
                       **metrics_from_counts(s["matched_km2"], s["eikonal_only_km2"], s["sfincs_only_km2"])})
    return per_tile, pd.DataFrame(pooled)


def tile_table(results: dict[int, dict], fsf: float, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per tile at a single friction point, in the column layout
    the figure functions of plot_validation_results.py use.

    Parameters
    ----------
    results : dict[int, dict]
        Per-tile results as returned by `load_results`.
    fsf : float
        Friction scale factor to select the eikonal entry for.
    meta : pandas.DataFrame, optional
        Tile metadata (tile_selection_metadata.csv) with ``tile_id``, ``lon`` and
        ``lat`` columns; when given, lon/lat are merged into the output.

    Returns
    -------
    pandas.DataFrame
        One row per tile with land/active-land areas (km2), SFINCS/bathtub/eikonal
        flooded areas and median depths (m), and each model's matched/only/
        sfincs_only areas (km2) and CSI vs SFINCS.
    """
    rows = []
    for tid, r in results.items():
        row = {"tile_id": tid, "land_km2": r["domain"]["land_km2"],
               "sfincs_active_land_km2": r["domain"]["sfincs_active_land_km2"],
               "sfincs_inactive_land_km2": r["domain"]["sfincs_inactive_land_km2"],
               "sfincs_km2": r["sfincs"]["flooded_km2"], "sfincs_depth_median_m": r["sfincs"]["depth"]["median_m"]}
        for model in ("bathtub", "eikonal"):
            e = _model_entry(r, model, fsf)
            v = (e or {}).get("vs_sfincs") or {}
            row.update({f"{model}_km2": (e or {}).get("flooded_km2"),
                        f"{model}_depth_median_m": ((e or {}).get("depth") or {}).get("median_m"),
                        f"{model}_matched_km2": v.get("matched_km2"), f"{model}_only_km2": v.get("model_only_km2"),
                        f"{model}_sfincs_only_km2": v.get("sfincs_only_km2"), f"{model}_CSI": v.get("CSI")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if meta is not None and len(df):
        df = df.merge(meta[["tile_id", "lon", "lat"]], on="tile_id", how="left")
    return df


def pooled_depth_joint(results: dict[int, dict], model: str, fsf: float | None) -> dict:
    """Sum every tile's depth-joint sufficient statistics and histograms for one
    model vs SFINCS.

    Parameters
    ----------
    results : dict[int, dict]
        Per-tile results as returned by `load_results`.
    model : str
        Model to pool, either "bathtub" or "eikonal".
    fsf : float or None
        Friction scale factor selecting the eikonal entry; ignored for "bathtub".

    Returns
    -------
    dict
        Summed ``n``, ``sum_x``, ``sum_y``, ``sum_x2``, ``sum_y2``, ``sum_xy`` plus
        summed ``hist_fine``/``hist_category`` arrays (None if no tile had a
        depth_joint entry) - the shape
        `flood_agreement.depth_error_metrics_from_pooled` reads.
    """
    keys = ("n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy")
    out = {k: 0.0 for k in keys}
    out["hist_fine"] = out["hist_category"] = None
    for r in results.values():
        e = _model_entry(r, model, fsf)
        joint = ((e or {}).get("vs_sfincs") or {}).get("depth_joint")
        if not joint:
            continue
        for k in keys:
            out[k] += joint[k]
        for h in ("hist_fine", "hist_category"):
            a = np.asarray(joint[h])
            out[h] = a if out[h] is None else out[h] + a
    return out


def metrics_overview(results: dict[int, dict], fsf: float) -> pd.DataFrame:
    """Build the pooled HC-bathtub vs EA-bathtub agreement table.

    Parameters
    ----------
    results : dict[int, dict]
        Per-tile results as returned by `load_results`.
    fsf : float
        Friction scale factor to select the eikonal (EA-bathtub) entry for.

    Returns
    -------
    pandas.DataFrame
        One row per model with pooled extent agreement (HT/FAR/CSI/bias) and
        depth agreement (r/bias/RMSE/...) against SFINCS.
    """
    df = tile_table(results, fsf)
    rows = []
    for model in ("bathtub", "eikonal"):
        sub = df.dropna(subset=[f"{model}_matched_km2"])
        m, o, s = (sub[f"{model}_{c}"].sum() for c in ("matched_km2", "only_km2", "sfincs_only_km2"))
        rows.append({"model": DISPLAY_LABEL[model], "friction_scale_factor": fsf if model == "eikonal" else None,
                     "n_tiles_pooled": len(sub), "total_km2": float(sub[f"{model}_km2"].sum()),
                     "sfincs_km2": float(sub["sfincs_km2"].sum()),
                     "matched_km2": float(m), "model_only_km2": float(o), "sfincs_only_km2": float(s),
                     **metrics_from_counts(m, o, s),
                     **depth_error_metrics_from_pooled(pooled_depth_joint(results, model, fsf))})
    return pd.DataFrame(rows)
