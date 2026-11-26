"""Aggregate the per-tile calibration_result.json files into run tables and the selection.

Counts are on the 30 m subgrid over SFINCS-active land only; every metric is
pooled by summing counts across tiles first (flood_agreement.metrics_from_counts).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_paths import DEPTH_ACCURACY_NAME, threaded_map
from flood_agreement import SFINCS_CLASS_LABELS, depth_error_metrics_from_pooled, diff_stats, metrics_from_counts
from retry_io import retry_transient_io

RESULT_NAME = "calibration_result.json"
DISPLAY_LABEL = {
    "bathtub": "Bathtub",
    "bathtub_connected": "Connected bathtub",
    "bathtub_connected_cap": "Connected bathtub + volume cap",
    "straight_line": "Straight-line attenuation",
    "straight_line_cap": "Straight-line attenuation + volume cap",
    "eikonal_uniform": "Eikonal, uniform rate",
    "eikonal_landcover": "Eikonal, land-cover rate",
    "eikonal_uniform_cap": "Eikonal, uniform rate + volume cap",
    "eikonal_landcover_cap": "Eikonal, land-cover rate + volume cap",
    "eikonal_uniform_duration": "Eikonal, uniform rate + duration",
    "eikonal_landcover_duration": "Eikonal, land-cover rate + duration",
}
COUNTS = ("matched_km2", "model_only_km2", "sfincs_only_km2")
COUNTS_POP = ("matched_pop", "model_only_pop", "sfincs_only_pop")  # people; absent in results without population
GROUPS = ("calibration", "validation")


def _read_result(path: Path) -> tuple[Path, dict | None]:
    try:
        # A missing file is not transient (tile not finished): skip it without retrying.
        if not path.exists():
            raise FileNotFoundError(path)
        return path, json.loads(retry_transient_io(path.read_text))
    except (OSError, ValueError):
        return path, None


def read_hists(path: Path, keep: Callable[[str], bool] = lambda name: True) -> dict[str, np.ndarray] | None:
    """Read the arrays of the .npz at `path` whose name passes `keep`, in file order; None if missing."""
    if not path.exists():
        return None
    with np.load(path) as hists:
        return {n: hists[n] for n in hists.files if keep(n)}


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
    for p, d in threaded_map(_read_result, paths):
        if d is None:
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
                         **{c: v.get(c, np.nan) for c in COUNTS_POP},
                         "HT": v["HT"], "FAR": v["FAR"], "CSI": v["CSI"], "bias": v["bias"]})
    per_tile = pd.DataFrame(rows)
    if per_tile.empty:
        return per_tile, per_tile
    per_tile = per_tile.sort_values(["variant", "run_key", "tile_id"]).reset_index(drop=True)
    pooled = []
    for group, sub in [*per_tile.groupby("group"), ("all", per_tile)]:
        for key, g in sub.groupby("run_key"):
            s = g[list(COUNTS)].sum()
            p = g[list(COUNTS_POP)].sum(min_count=1)
            first = g.iloc[0]
            pooled.append({"group": group, "run_key": key, "variant": first.variant, "parameter": first.parameter,
                           "alpha": first.alpha, "n_tiles_scored": len(g), **s.to_dict(),
                           **metrics_from_counts(s["matched_km2"], s["model_only_km2"], s["sfincs_only_km2"]),
                           "n_tiles_pop": int(g[COUNTS_POP[0]].notna().sum()), **p.to_dict(),
                           **{f"{m}_pop": val for m, val in metrics_from_counts(*p.to_numpy()).items()}})
    return per_tile, pd.DataFrame(pooled)


def _sum_depth_hists(job: tuple[Path, list[tuple[int, str, dict]]]) -> tuple[dict, dict, dict]:
    # Reading a histogram file is CPU-bound (one header parse per array), so blocks of tiles go to
    # separate processes; each sums its block in tile order.
    base_dir, tiles = job
    acc, n_tiles, meta = {}, {}, {}
    for tid, group, runs in tiles:
        hists = read_hists(base_dir / str(tid) / "outputs" / DEPTH_ACCURACY_NAME,
                           lambda name: name.rsplit("__", 1)[0] in runs)
        if hists is None:
            continue
        for name, h in hists.items():
            key, weight = name.rsplit("__", 1)
            meta[key] = runs[key]
            for g in (group, "all"):
                acc[g, key, weight] = acc.get((g, key, weight), 0) + h
                if weight == "area":
                    n_tiles[g, key] = n_tiles.get((g, key), 0) + 1
    return acc, n_tiles, meta


def depth_accuracy_table(base_dir: Path, results: dict[int, dict], workers: int = 1) -> pd.DataFrame:
    """Pool every run's depth-difference histograms per group; one row per denominator and class.

    Denominators: both_wet (cells both flood), reference_wet (every SFINCS-wet cell, a model-dry
    cell at its own sub-threshold depth), either_wet (every cell either floods; class "all"
    only). Weights: area (km2) and pop (people). Tiles without the histogram file are skipped.
    `workers` > 1 sums contiguous blocks of tiles in parallel and adds the blocks in order.
    """
    tiles = [(tid, r["group"], {k: (e["variant"], e["parameter"], e["alpha"]) for k, e in r["runs"].items()})
             for tid, r in results.items()]
    n_blocks = max(1, min(workers, len(tiles)))
    jobs = [(base_dir, [tiles[i] for i in idx]) for idx in np.array_split(np.arange(len(tiles)), n_blocks)]
    if n_blocks == 1:
        parts = [_sum_depth_hists(jobs[0])]
    else:
        with ProcessPoolExecutor(n_blocks) as ex:
            parts = list(ex.map(_sum_depth_hists, jobs))
    acc, n_tiles, meta = {}, {}, {}
    for part_acc, part_n, part_meta in parts:
        meta.update(part_meta)
        for k, h in part_acc.items():
            acc[k] = acc.get(k, 0) + h
        for k, n in part_n.items():
            n_tiles[k] = n_tiles.get(k, 0) + n
    rows = []
    for (group, key, weight), h in sorted(acc.items()):
        both, sfincs_only, model_only = h
        variant, parameter, alpha = meta[key]
        head = {"group": group, "run_key": key, "variant": variant, "parameter": parameter, "alpha": alpha,
                "n_tiles": n_tiles.get((group, key), 0), "weight": weight}
        for cells, by_class in (("both_wet", both), ("reference_wet", both + sfincs_only)):
            for label, hc in zip(SFINCS_CLASS_LABELS, by_class):
                rows.append({**head, "cells": cells, "sfincs_class_m": label, **diff_stats(hc)})
            rows.append({**head, "cells": cells, "sfincs_class_m": "all", **diff_stats(by_class.sum(axis=0))})
        rows.append({**head, "cells": "either_wet", "sfincs_class_m": "all", **diff_stats(h.sum(axis=(0, 1)))})
    return pd.DataFrame(rows)


SELECT_BY = ("CSI_pop", "CSI", "bias")


def _selection_score(df: pd.DataFrame, select_by: str) -> pd.Series:
    # Higher is better.
    if select_by == "bias":
        return -(df["bias"] - 1.0).abs()
    return df[select_by]


def select_runs(pooled: pd.DataFrame, select_by: str = "CSI_pop", select_group: str = "all") -> dict:
    """Pick each variant's run on `select_group`; score it on every group.

    `select_by` "CSI_pop": highest pooled population-weighted CSI; "CSI": highest pooled
    (area) CSI; "bias": |bias - 1| smallest. The same score on `select_group` ranks the
    variants for "best_variant". Runs must cover the same tiles to compare (n_tiles_scored
    is reported).
    """
    if select_by not in SELECT_BY:
        raise ValueError(f"select_by must be one of {SELECT_BY}, got {select_by!r}")
    cal = pooled[pooled.group == select_group]
    metrics = ("n_tiles_scored", "CSI", "bias", "HT", "FAR", "CSI_pop", "bias_pop")
    metrics = tuple(m for m in metrics if m in pooled.columns)
    out = {"select_by": select_by, "select_group": select_group, "variants": {}}
    for variant, g in cal.groupby("variant"):
        score = _selection_score(g, select_by)
        if score.isna().all():
            raise ValueError(f"no {select_by} for variant {variant}: cannot select its run")
        best = g.loc[score.idxmax()]
        entry = {"run_key": best.run_key,
                 "parameter": None if pd.isna(best.parameter) else float(best.parameter),
                 "alpha": None if pd.isna(best.alpha) else float(best.alpha)}
        for group in (*GROUPS, "all"):
            row = pooled[(pooled.group == group) & (pooled.run_key == best.run_key)]
            entry[group] = ({m: (int(row.iloc[0][m]) if m == "n_tiles_scored" else float(row.iloc[0][m]))
                             for m in metrics} if len(row) else None)
        out["variants"][variant] = entry
    if out["variants"]:
        cal_rows = pd.DataFrame({v: e[select_group] for v, e in out["variants"].items()}).T
        out["best_variant"] = _selection_score(cal_rows.astype(float), select_by).idxmax()
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
