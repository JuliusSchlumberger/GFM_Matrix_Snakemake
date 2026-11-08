"""Aggregate the per-tile calibration_result.json files into run tables and the selection.

Counts are on the 30 m subgrid over SFINCS-active land only; every metric is
pooled by summing counts across tiles first (flood_agreement.metrics_from_counts).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flood_agreement import depth_error_metrics_from_pooled, metrics_from_counts
from retry_io import retry_transient_io

RESULT_NAME = "calibration_result.json"
DISPLAY_LABEL = {
    "bathtub": "Bathtub",
    "bathtub_connected": "Connected bathtub",
    "straight_line": "Straight-line attenuation",
    "eikonal_uniform": "Eikonal, uniform rate",
    "eikonal_landcover": "Eikonal, land-cover rate",
    "eikonal_uniform_duration": "Eikonal, uniform rate + duration",
    "eikonal_landcover_duration": "Eikonal, land-cover rate + duration",
}
COUNTS = ("matched_km2", "model_only_km2", "sfincs_only_km2")
GROUPS = ("calibration", "validation")


def load_results(base_dir: Path, tile_ids: list[str] | None = None) -> dict[int, dict]:
    """Load each tile's calibration_result.json, keyed by tile id.

    Reads are retried on transient share errors. Missing or unreadable files are
    skipped, and their count and tile ids are printed.
    """
    if tile_ids is None:
        paths = sorted(base_dir.glob(f"*/outputs/{RESULT_NAME}"))
    else:
        paths = [base_dir / str(t) / "outputs" / RESULT_NAME for t in tile_ids]
    out = {}
    skipped = []
    for p in paths:
        try:
            # A missing file is not transient (tile not finished): skip it without retrying.
            if not p.exists():
                raise FileNotFoundError(p)
            d = json.loads(retry_transient_io(p.read_text))
        except (OSError, ValueError):
            skipped.append(p.parent.parent.name)
            continue
        out[int(d["tile_id"])] = d
    if skipped:
        print(f"load_results: {len(skipped)} tile(s) without a readable {RESULT_NAME} skipped: "
              f"{', '.join(skipped)}", flush=True)
    return out


def run_label(key: str, results: dict[int, dict]) -> str:
    """Return a display label for run `key`: the variant's name and its parameter/alpha."""
    for r in results.values():
        e = (r.get("runs") or {}).get(key)
        if e:
            label = DISPLAY_LABEL[e["variant"]]
            extra = [f"{'fsf' if 'landcover' in e['variant'] else 'rate'} {e['parameter']:g}"] if e["parameter"] is not None else []
            extra += [f"alpha {e['alpha']:g}"] if e["alpha"] is not None else []
            return f"{label} ({', '.join(extra)})" if extra else label
    return key


def run_tables(results: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_tile, pooled) run-vs-SFINCS extent tables.

    Areas in km2. Pooled rows sum counts across tiles per (group, run_key); group "all"
    pools both groups.
    """
    rows = []
    for tid, r in results.items():
        for key, e in (r.get("runs") or {}).items():
            v = e["vs_sfincs"]
            rows.append({"tile_id": tid, "group": r["group"], "run_key": key, "variant": e["variant"],
                         "parameter": e["parameter"], "alpha": e["alpha"], **{c: v[c] for c in COUNTS},
                         "HT": v["HT"], "FAR": v["FAR"], "CSI": v["CSI"], "bias": v["bias"]})
    per_tile = pd.DataFrame(rows)
    if per_tile.empty:
        return per_tile, per_tile
    per_tile = per_tile.sort_values(["variant", "run_key", "tile_id"]).reset_index(drop=True)
    pooled = []
    for group, sub in [*per_tile.groupby("group"), ("all", per_tile)]:
        for key, g in sub.groupby("run_key"):
            s = g[list(COUNTS)].sum()
            first = g.iloc[0]
            pooled.append({"group": group, "run_key": key, "variant": first.variant, "parameter": first.parameter,
                           "alpha": first.alpha, "n_tiles_scored": len(g), **s.to_dict(),
                           **metrics_from_counts(s["matched_km2"], s["model_only_km2"], s["sfincs_only_km2"])})
    return per_tile, pd.DataFrame(pooled)


def select_runs(pooled: pd.DataFrame, select_by: str = "CSI") -> dict:
    """Pick each variant's run on the calibration group; score it on both groups.

    `select_by` "CSI": highest pooled CSI; "bias": |bias - 1| smallest. The run sharing the
    calibration tiles of every variant ranks the variants for "best_variant". Runs must cover
    the same calibration tiles to compare (n_tiles_scored is reported).
    """
    cal = pooled[pooled.group == "calibration"]
    metrics = ("n_tiles_scored", "CSI", "bias", "HT", "FAR")
    out = {"select_by": select_by, "variants": {}}
    for variant, g in cal.groupby("variant"):
        best = g.loc[g.CSI.idxmax()] if select_by == "CSI" else g.loc[(g.bias - 1.0).abs().idxmin()]
        entry = {"run_key": best.run_key,
                 "parameter": None if pd.isna(best.parameter) else float(best.parameter),
                 "alpha": None if pd.isna(best.alpha) else float(best.alpha)}
        for group in GROUPS:
            row = pooled[(pooled.group == group) & (pooled.run_key == best.run_key)]
            entry[group] = ({m: (int(row.iloc[0][m]) if m == "n_tiles_scored" else float(row.iloc[0][m]))
                             for m in metrics} if len(row) else None)
        out["variants"][variant] = entry
    if out["variants"]:
        if select_by == "CSI":
            score = {v: e["calibration"]["CSI"] for v, e in out["variants"].items()}
            out["best_variant"] = max(score, key=score.get)
        else:
            score = {v: abs(e["calibration"]["bias"] - 1.0) for v, e in out["variants"].items()}
            out["best_variant"] = min(score, key=score.get)
    return out


def tile_table(results: dict[int, dict], runs: dict[str, str], meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per tile with the columns of each run in `runs` (prefix -> run key).

    Areas in km2, depths in m. `meta` (tile_selection_metadata.csv) adds lon/lat.
    """
    rows = []
    for tid, r in results.items():
        row = {"tile_id": tid, "group": r.get("group"), "land_km2": r["domain"]["land_km2"],
               "sfincs_active_land_km2": r["domain"]["sfincs_active_land_km2"],
               "sfincs_inactive_land_km2": r["domain"]["sfincs_inactive_land_km2"],
               "sfincs_km2": r["sfincs"]["flooded_km2"], "sfincs_depth_median_m": r["sfincs"]["depth"]["median_m"]}
        for prefix, key in runs.items():
            e = (r.get("runs") or {}).get(key)
            v = (e or {}).get("vs_sfincs") or {}
            row.update({f"{prefix}_km2": (e or {}).get("flooded_km2"),
                        f"{prefix}_depth_median_m": ((e or {}).get("depth") or {}).get("median_m"),
                        f"{prefix}_matched_km2": v.get("matched_km2"), f"{prefix}_only_km2": v.get("model_only_km2"),
                        f"{prefix}_sfincs_only_km2": v.get("sfincs_only_km2"), f"{prefix}_CSI": v.get("CSI")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if meta is not None and len(df):
        df = df.merge(meta[["tile_id", "lon", "lat"]], on="tile_id", how="left")
    return df


def pooled_depth_joint(results: dict[int, dict], key: str) -> dict:
    """Sum all tiles' depth-joint statistics and histograms for run `key` vs SFINCS.

    Histograms are None if no tile had a depth_joint.
    """
    keys = ("n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy")
    out = {k: 0.0 for k in keys}
    out["hist_fine"] = out["hist_category"] = None
    for r in results.values():
        e = (r.get("runs") or {}).get(key)
        joint = ((e or {}).get("vs_sfincs") or {}).get("depth_joint")
        if not joint:
            continue
        for k in keys:
            out[k] += joint[k]
        for h in ("hist_fine", "hist_category"):
            a = np.asarray(joint[h])
            out[h] = a if out[h] is None else out[h] + a
    return out


def metrics_overview(results: dict[int, dict], keys: list[str]) -> pd.DataFrame:
    """Build the pooled agreement table vs SFINCS for the runs `keys`."""
    df = tile_table(results, {f"r{i}": k for i, k in enumerate(keys)})
    rows = []
    for i, key in enumerate(keys):
        p = f"r{i}"
        sub = df.dropna(subset=[f"{p}_matched_km2"])
        m, o, s = (sub[f"{p}_{c}"].sum() for c in ("matched_km2", "only_km2", "sfincs_only_km2"))
        rows.append({"run": run_label(key, results), "run_key": key,
                     "n_tiles_pooled": len(sub), "total_km2": float(sub[f"{p}_km2"].sum()),
                     "sfincs_km2": float(sub["sfincs_km2"].sum()),
                     "matched_km2": float(m), "model_only_km2": float(o), "sfincs_only_km2": float(s),
                     **metrics_from_counts(m, o, s),
                     **depth_error_metrics_from_pooled(pooled_depth_joint(results, key))})
    return pd.DataFrame(rows)
