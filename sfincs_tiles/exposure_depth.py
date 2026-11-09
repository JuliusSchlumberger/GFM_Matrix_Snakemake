"""Exposure and depth reliability of the eikonal against SFINCS (handoff_exposure_depth_assessment).

Stage 0 works on the calibration_result.json files alone: depth agreement within 0.1/0.2/0.3/0.5 m
per SFINCS depth class, same-bin hit rates, outlier tiles. Reads both the current (`runs`) and the
earlier (`eikonal` / `fsf_*`, effective friction factors) result layouts. Writes tables and
figures to --out-dir; reads the calibration directories only.

Stage-1 report: pools the exposure_depth_pass.py tiles in --out-dir/tiles.

Usage:
    python exposure_depth.py stage0 --config <config.yml> --out-dir <dir> [--fsf 12]
    python exposure_depth.py report --config <config.yml> --out-dir <stage-1 dir>
    python exposure_depth.py cdf --config <config.yml> --out-dir <stage-1 dir> [--sf-bay-cdf <csv>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_results import load_results  # noqa: E402
from flood_agreement import DEPTH_CORR_FINE_EDGES, DIFF_EDGES, SFINCS_CLASS_LABELS, diff_stats  # noqa: E402
from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import draw_panel_letter  # noqa: E402

BIN_M = 0.1  # hist_fine bin width: DEPTH_CORR_FINE_EDGES step
TOLERANCES_M = (0.1, 0.2, 0.3, 0.5)
# SFINCS depth classes (m) the reliability is reported for; the last is open-ended (> 3 m catch-all).
DEPTH_CLASSES_M = ((0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))
SAME_BIN_WIDTHS_M = (0.1, 0.3, 0.5)
CURVE_TAUS_M = np.round(np.arange(0.0, 1.01, 0.05), 2)


def legacy_entries(result: dict) -> dict[float, dict]:
    """Return friction factor -> run entry of a result in either layout (effective factors, old layout)."""
    if result.get("eikonal"):
        return {float(k[4:]): v for k, v in result["eikonal"].items()}
    return {e["parameter"]: e for e in (result.get("runs") or {}).values()
            if e["variant"] == "eikonal_landcover" and e["parameter"] is not None}


def _triangular_cdf(w: np.ndarray) -> np.ndarray:
    # CDF of u - v for u, v ~ U(0, 1): triangular on [-1, 1]
    w = np.clip(w, -1.0, 1.0)
    return np.where(w < 0, 0.5 * (1 + w) ** 2, 1 - 0.5 * (1 - w) ** 2)


def within_tolerance_share(k: np.ndarray, tau_m: float) -> np.ndarray:
    """P(|Delta d| <= tau) for two depths k bins apart, depths uniform within their 0.1 m bins."""
    t = tau_m / BIN_M
    return _triangular_cdf(t - k) - _triangular_cdf(-t - k)


def _core(hist) -> np.ndarray:
    # rows: SFINCS bin, columns: model bin; drop the < 0 m row/column (empty: both wet > 0.1 m)
    return np.asarray(hist, dtype=np.float64)[1:, 1:]


def tolerance_table(hist, taus=TOLERANCES_M) -> dict:
    """Share of both-wet cells within each tau: estimate, lower and upper bound (>3 m pairs excluded)."""
    h = _core(hist)[:-1, :-1]  # drop the > 3 m catch-all: its distance is unknown
    n = h.sum()
    i, j = np.indices(h.shape)
    k = np.abs(i - j).astype(float)
    out = {"n_cells": n}
    for tau in taus:
        steps = round(tau / BIN_M)
        out[f"within_{tau:g}m"] = float((h * within_tolerance_share(k, tau)).sum() / n) if n else np.nan
        out[f"within_{tau:g}m_low"] = float(h[k + 1 <= steps].sum() / n) if n else np.nan
        out[f"within_{tau:g}m_high"] = float(h[k - 1 < steps].sum() / n) if n else np.nan
    return out


def class_table(hist, taus=TOLERANCES_M) -> pd.DataFrame:
    """Per SFINCS depth class: cells, bias and MAE (bin centres), within-tau shares."""
    h = _core(hist)
    centres = np.append((DEPTH_CORR_FINE_EDGES[:-1] + DEPTH_CORR_FINE_EDGES[1:]) / 2, np.nan)
    rows = []
    for lo, hi in (*DEPTH_CLASSES_M, (3.0, np.inf)):
        sel = slice(round(lo / BIN_M), round(hi / BIN_M) if np.isfinite(hi) else None)
        sub = np.zeros_like(h)
        sub[sel, :] = h[sel, :]
        n = sub.sum()
        row = {"sfincs_class_m": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}", "n_cells": n,
               "share_of_cells": n / h.sum()}
        if np.isfinite(hi):
            core = sub[:-1, :-1]
            err = centres[None, :-1] - centres[:-1, None]
            m = core.sum()
            row["bias_m"] = float((core * err).sum() / m) if m else np.nan
            row["mae_m"] = float((core * np.abs(err)).sum() / m) if m else np.nan
            row["model_deeper_than_3m"] = float(sub[:, -1].sum() / n) if n else np.nan
            row.update({k: v for k, v in tolerance_table(np.pad(sub, ((1, 0), (1, 0))), taus).items() if k != "n_cells"})
        rows.append(row)
    return pd.DataFrame(rows)


def same_bin_shares(joint: dict) -> dict:
    """Diagonal share of the 0.2 m class matrix and of the 0.1 m histogram re-binned to 0.1/0.3/0.5 m."""
    out = {}
    cat = np.asarray(joint["hist_category"], dtype=np.float64)[1:, 1:]
    out["same_class_0.2m"] = float(np.trace(cat) / cat.sum()) if cat.sum() else np.nan
    h = _core(joint["hist_fine"])[:-1, :-1]
    for width in SAME_BIN_WIDTHS_M:
        f = round(width / BIN_M)
        idx = np.arange(h.shape[0]) // f
        m = np.zeros((idx.max() + 1,) * 2)
        np.add.at(m, (idx[:, None].repeat(h.shape[1], 1), idx[None, :].repeat(h.shape[0], 0)), h)
        out[f"same_bin_{width:g}m"] = float(np.trace(m) / m.sum()) if m.sum() else np.nan
    return out


def exact_errors(joint: dict) -> dict:
    """Bias and RMSE (m, model - SFINCS) from the sufficient statistics."""
    n = joint["n"]
    if not n:
        return {"bias_m": np.nan, "rmse_m": np.nan}
    mse = (joint["sum_y2"] - 2 * joint["sum_xy"] + joint["sum_x2"]) / n
    return {"bias_m": (joint["sum_y"] - joint["sum_x"]) / n, "rmse_m": float(np.sqrt(max(mse, 0.0)))}


def pooled_joint(results: dict, fsf: float) -> dict:
    keys = ("n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy")
    out = {k: 0.0 for k in keys}
    out["hist_fine"] = out["hist_category"] = 0.0
    for r in results.values():
        joint = ((legacy_entries(r).get(fsf) or {}).get("vs_sfincs") or {}).get("depth_joint")
        if joint:
            for k in keys:
                out[k] += joint[k]
            out["hist_fine"] = out["hist_fine"] + np.asarray(joint["hist_fine"])
            out["hist_category"] = out["hist_category"] + np.asarray(joint["hist_category"])
    return out


def tile_table(results: dict, fsf: float) -> pd.DataFrame:
    rows = []
    for tid, r in results.items():
        e = legacy_entries(r).get(fsf)
        if not e:
            continue
        v = e["vs_sfincs"]
        row = {"tile_id": tid, "sfincs_km2": r["sfincs"]["flooded_km2"], "model_km2": e["flooded_km2"],
               "CSI": v["CSI"], "bias": v["bias"], "sfincs_depth_median_m": r["sfincs"]["depth"]["median_m"],
               "boundary_dist_median_km": r["domain"].get("boundary_dist_median_km")}
        joint = v.get("depth_joint")
        if joint and joint["n"]:
            row.update(exact_errors(joint))
            row.update({k: val for k, val in tolerance_table(joint["hist_fine"]).items()
                        if not k.endswith(("_low", "_high"))})
        rows.append(row)
    return pd.DataFrame(rows)


def curve(hist) -> np.ndarray:
    h = _core(hist)[:-1, :-1]
    i, j = np.indices(h.shape)
    k = np.abs(i - j).astype(float)
    n = h.sum()
    return np.array([(h * within_tolerance_share(k, t)).sum() / n if n else np.nan for t in CURVE_TAUS_M])


def plot_stage0(results: dict, fsf: float, classes: pd.DataFrame, tiles: pd.DataFrame, out: Path) -> None:
    pooled = pooled_joint(results, fsf)
    per_tile = np.array([curve(e["vs_sfincs"]["depth_joint"]["hist_fine"]) for r in results.values()
                         for e in [legacy_entries(r).get(fsf)]
                         if e and (e["vs_sfincs"].get("depth_joint") or {}).get("n")])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(CURVE_TAUS_M, np.nanpercentile(per_tile, 10, axis=0), np.nanpercentile(per_tile, 90, axis=0),
                    color="#9ecae1", alpha=0.6, label="tiles, p10-p90")
    ax.plot(CURVE_TAUS_M, np.nanmedian(per_tile, axis=0), color="#3182bd", lw=1.2, ls="--", label="tiles, median")
    ax.plot(CURVE_TAUS_M, curve(pooled["hist_fine"]), color="#08306b", lw=2, label="pooled (cells)")
    for t in (0.1, 0.2, 0.5):
        ax.axvline(t, color="#bdbdbd", lw=0.8, zorder=0)
    ax.set(xlabel="tolerance |eikonal - SFINCS depth| (m)", ylabel="share of both-wet cells within",
           xlim=(0, 1), ylim=(0, 1), title=f"Depth agreement, RP100, friction factor {fsf:g} ({len(per_tile)} tiles)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "X1_depth_tolerance_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    c = classes[classes.sfincs_class_m != ">3"]
    x = np.arange(len(c))
    for off, tau, color in ((-0.25, 0.1, "#c6dbef"), (0.0, 0.2, "#6baed6"), (0.25, 0.5, "#08519c")):
        est, lo, hi = c[f"within_{tau:g}m"], c[f"within_{tau:g}m_low"], c[f"within_{tau:g}m_high"]
        ax.bar(x + off, est, width=0.24, color=color, label=f"within {tau * 100:.0f} cm")
        ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor="#525252", lw=0.8, capsize=2)
    ax.set_xticks(x, [f"{s} m\n({n / 1e6:.1f} M)" for s, n in zip(c.sfincs_class_m, c.n_cells)], fontsize=8)
    ax.set(ylabel="share of both-wet cells within tolerance", ylim=(0, 1),
           xlabel="SFINCS depth class (cells, millions)", title="Depth reliability by SFINCS depth class")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "X2_tolerance_by_depth_class.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, label, letter in ((axes[0], "sfincs_km2", "SFINCS flooded area (km2)", "a"),
                                   (axes[1], "boundary_dist_median_km", "median boundary distance (km)", "b")):
        t = tiles.dropna(subset=["CSI", col])
        ax.scatter(t[col], t.CSI, s=8, color="#3182bd", alpha=0.6, edgecolors="none")
        worst = t.nsmallest(8, "CSI")
        for _, w in worst.iterrows():
            ax.annotate(str(int(w.tile_id)), (w[col], w.CSI), fontsize=6, color="#a50f15")
        if col == "sfincs_km2":
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="CSI (tile)", ylim=(0, 1))
        draw_panel_letter(ax, letter)
    fig.suptitle(f"Tile CSI against tile properties, friction factor {fsf:g}; red: the 8 lowest")
    fig.tight_layout()
    fig.savefig(out / "X3_outlier_tiles.png", dpi=200)
    plt.close(fig)


def stage0(args) -> None:
    root = read_root(Path(args.config))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(root / args.dedup_dir_name)
    fsfs = sorted({f for r in results.values() for f in legacy_entries(r)})
    pooled_rows = []
    for fsf in fsfs:
        joint = pooled_joint(results, fsf)
        if not joint["n"]:
            continue
        pooled_rows.append({"friction_factor": fsf, **exact_errors(joint), **tolerance_table(joint["hist_fine"]),
                            **same_bin_shares(joint)})
    pooled = pd.DataFrame(pooled_rows)
    classes = class_table(pooled_joint(results, args.fsf)["hist_fine"])
    tiles = tile_table(results, args.fsf)
    # The dedup files carry no boundary distance; it is a property of the whole tile.
    raw = load_results(root / args.base_dir_name, [str(t) for t in tiles.tile_id])
    tiles["boundary_dist_median_km"] = tiles.tile_id.map(
        {t: r["domain"].get("boundary_dist_median_km") for t, r in raw.items()}).astype(float)
    pooled.to_csv(out / "T1_pooled_depth_agreement_by_fsf.csv", index=False)
    classes.to_csv(out / f"T2_depth_classes_fsf{args.fsf:g}.csv", index=False)
    tiles.to_csv(out / f"tiles_fsf{args.fsf:g}.csv", index=False)
    plot_stage0(results, args.fsf, classes, tiles, out)
    summary = {"tiles": len(results), "fsf": args.fsf, "fsfs": fsfs,
               "best_fsf_within_0.2m": float(pooled.loc[pooled["within_0.2m"].idxmax(), "friction_factor"]),
               "best_fsf_within_0.5m": float(pooled.loc[pooled["within_0.5m"].idxmax(), "friction_factor"])}
    (out / "stage0_summary.json").write_text(json.dumps(summary, indent=1))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(pooled.to_string(index=False))
        print(classes.to_string(index=False))
    print(json.dumps(summary))


FRINGE_M = 0.3  # an own-only cell shallower than this counts as fringe disagreement


def _metrics(m: float, mo: float, so: float) -> dict:
    return {"HR": m / (m + so) if m + so else np.nan, "FAR": mo / (m + mo) if m + mo else np.nan,
            "CSI": m / (m + mo + so) if m + mo + so else np.nan,
            "exposure_bias": (m + mo) / (m + so) if m + so else np.nan}


def report(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exposure_depth_pass import DENSITY_LABELS

    tiles_dir = Path(args.out_dir) / "tiles"
    ids = pass_tile_ids(Path(args.out_dir), args.all_tiles)
    out = _report_dir(args)
    meta, arrays = [], {}
    for tid in ids:
        info = json.loads((tiles_dir / f"{tid}.json").read_text())
        meta.append(info)
        arrays[info["tile_id"]] = dict(np.load(tiles_dir / f"{tid}.npz"))
    meta = pd.DataFrame(meta)
    density = pd.read_csv(Path(args.out_dir) / "tile_density.csv")
    meta = meta.merge(density[["tile_id", "density", "density_class"]].astype({"tile_id": str}), on="tile_id")

    rows = []
    for w in ("area", "pop"):
        counts = sum(a[f"counts_{w}"] for a in arrays.values())
        for c, label in enumerate(DENSITY_LABELS):
            rows.append({"weight": w, "density_class": label, "matched": counts[c, 0], "model_only": counts[c, 1],
                         "sfincs_only": counts[c, 2], **_metrics(*counts[c])})
        tot = counts.sum(axis=0)
        rows.append({"weight": w, "density_class": "all", "matched": tot[0], "model_only": tot[1],
                     "sfincs_only": tot[2], **_metrics(*tot)})
    by_density = pd.DataFrame(rows)

    per_tile = []
    for tid, a in arrays.items():
        m, mo, so = a["counts_pop"].sum(axis=0)
        per_tile.append({"tile_id": tid, "pop_sfincs": m + so, "pop_model": m + mo, **{
            f"{k}_pop": v for k, v in _metrics(m, mo, so).items()}})
    per_tile = meta.merge(pd.DataFrame(per_tile), on="tile_id")
    exposed = per_tile[(per_tile.pop_sfincs > 0) | (per_tile.pop_model > 0)]
    ratio = exposed.pop_model / exposed.pop_sfincs
    within2 = float(((ratio >= 0.5) & (ratio <= 2)).mean())

    depth_rows = []
    for w in ("area", "pop"):
        diff = sum(a[f"diff_{w}"] for a in arrays.values())
        for c, label in enumerate(SFINCS_CLASS_LABELS):
            depth_rows.append({"weight": w, "sfincs_class_m": label, **diff_stats(diff[c])})
        depth_rows.append({"weight": w, "sfincs_class_m": "all", **diff_stats(diff.sum(axis=0))})
    depth = pd.DataFrame(depth_rows)

    fringe = {}
    k = int(round(FRINGE_M / 0.1)) + 1  # bins below FRINGE_M after the < 0 catch-all
    for w in ("area", "pop"):
        for side in ("model_only", "sfincs_only"):
            h = sum(a[f"{side}_depth_{w}"] for a in arrays.values())
            fringe[f"{side}_{w}_share_below_{FRINGE_M:g}m"] = float(h[:k].sum() / h.sum()) if h.sum() else np.nan

    by_density.to_csv(out / "T3_extent_by_density_class.csv", index=False)
    per_tile.to_csv(out / "tiles_exposed_population.csv", index=False)
    depth.to_csv(out / "T4_depth_reliability_exact.csv", index=False)
    summary = {"tiles": len(arrays), "tiles_with_exposure": len(exposed),
               "share_tiles_population_within_factor_2": within2, **fringe}
    (out / "stage1_summary.json").write_text(json.dumps(summary, indent=1))
    plot_stage1(by_density, exposed, depth, out)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(by_density.to_string(index=False))
        print(depth.to_string(index=False))
    print(json.dumps(summary, indent=1))


def plot_stage1(by_density: pd.DataFrame, exposed: pd.DataFrame, depth: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    d = by_density[by_density.density_class != "all"]
    for ax, metric, letter in zip(axes, ("CSI", "HR", "exposure_bias"), "abc"):
        x = np.arange(d.density_class.nunique())
        for off, w, color, label in ((-0.18, "area", "#9ecae1", "area-weighted"),
                                     (0.18, "pop", "#08519c", "population-weighted")):
            sub = d[d.weight == w]
            ax.bar(x + off, sub[metric], width=0.34, color=color, label=label)
        ax.set_xticks(x, [f"{c}" for c in d[d.weight == "pop"].density_class], fontsize=8)
        ax.set(xlabel="population density of the cell (people/km2)", ylabel=metric)
        if metric == "exposure_bias":
            ax.axhline(1, color="#525252", lw=0.8)
        else:
            ax.set_ylim(0, 1)
        draw_panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "X4_metrics_by_density_class.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    e = exposed[(exposed.pop_sfincs > 0) & (exposed.pop_model > 0)]
    ax.scatter(e.pop_sfincs, e.pop_model, s=12, color="#3182bd", alpha=0.7, edgecolors="none")
    lo, hi = max(1.0, min(e.pop_sfincs.min(), e.pop_model.min())), max(e.pop_sfincs.max(), e.pop_model.max())
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="#252525", lw=1)
    ax.fill_between(xs, xs / 2, xs * 2, color="#bdbdbd", alpha=0.35, label="factor 2")
    ax.set(xscale="log", yscale="log", xlabel="exposed population, SFINCS", ylabel="exposed population, eikonal",
           title=f"RP100 exposed population per tile ({len(e)} tiles)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "X5_exposed_population_per_tile.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dd = depth[(depth.sfincs_class_m != "all")]
    classes = dd[dd.weight == "area"].sfincs_class_m.tolist()
    x = np.arange(len(classes))
    for off, tau, color in ((-0.27, 0.1, "#c6dbef"), (-0.09, 0.2, "#6baed6"), (0.09, 0.3, "#2171b5"),
                            (0.27, 0.5, "#08306b")):
        ax.bar(x + off, dd[dd.weight == "area"][f"within_{tau:g}m"], width=0.17, color=color,
               label=f"within {tau * 100:.0f} cm")
        ax.scatter(x + off, dd[dd.weight == "pop"][f"within_{tau:g}m"], marker="_", s=120, color="#a50f15",
                   zorder=3, label="population-weighted" if tau == 0.1 else None)
    ax.set_xticks(x, [f"{c} m" for c in classes], fontsize=8)
    ax.set(xlabel="SFINCS depth class", ylabel="share of both-wet cells within tolerance", ylim=(0, 1),
           title="Exact depth reliability by SFINCS depth class (bars: area, red: population)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "X2b_exact_tolerance_by_depth_class.png", dpi=200)
    plt.close(fig)


CDF_TAUS_M = np.round(np.arange(0.0, 1.501, 0.01), 2)
DENOMINATORS = {"both_wet": ("both flood", "#9ecae1"), "reference_wet": ("reference floods", "#08519c"),
                "either_wet": ("either floods", "#a50f15")}


def pass_tile_ids(stage1_dir: Path, all_tiles: bool) -> list[str]:
    """Tiles of the stage-1 pass to pool: the stratified subset, or every passed tile."""
    done = sorted(p.stem for p in (stage1_dir / "tiles").glob("*.npz"))
    if all_tiles:
        return done
    subset = set(pd.read_csv(stage1_dir / "subset_tiles.csv").tile_id.astype(str))
    return [t for t in done if t in subset]


def _report_dir(args) -> Path:
    out = Path(args.out_dir) / ("full" if args.all_tiles else "")
    out.mkdir(parents=True, exist_ok=True)
    return out


def sfincs_cdfs(tiles_dir: Path, ids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cumulative share within tau per denominator and weight: (lower, upper) bound arrays.

    Both-wet cells from the 1 cm difference histograms (bin fully within tau). A one-sided cell
    is known only by its own depth (0.1 m bins) against the other side's sub-threshold depth
    (0-0.10 m): within tau at least if its depth bin lies below tau, at most if below tau + 0.10.
    """
    from exposure_depth_pass import DEPTH_EDGES

    acc = {}
    for tid in ids:
        for k, v in np.load(tiles_dir / f"{tid}.npz").items():
            acc[k] = acc.get(k, 0) + v
    inside = np.maximum(np.abs(DIFF_EDGES[:-1]), np.abs(DIFF_EDGES[1:]))
    upper = np.concatenate((DEPTH_EDGES, [np.inf]))
    out = {}
    for w in ("area", "pop"):
        diff = acc[f"diff_{w}"].sum(axis=0)
        both = np.array([diff[1:-1][inside <= t + 1e-9].sum() for t in CDF_TAUS_M])
        s_only, m_only = acc[f"sfincs_only_depth_{w}"], acc[f"model_only_depth_{w}"]

        def one_sided(h, extra):
            return np.array([h[upper <= t + extra + 1e-9].sum() for t in CDF_TAUS_M])
        n_both = diff.sum()
        out[f"both_wet_{w}"] = (both / n_both, both / n_both)
        n_ref = n_both + s_only.sum()
        out[f"reference_wet_{w}"] = ((both + one_sided(s_only, 0.0)) / n_ref, (both + one_sided(s_only, 0.1)) / n_ref)
        n_eit = n_ref + m_only.sum()
        out[f"either_wet_{w}"] = ((both + one_sided(s_only, 0.0) + one_sided(m_only, 0.0)) / n_eit,
                                  (both + one_sided(s_only, 0.1) + one_sided(m_only, 0.1)) / n_eit)
    return out


def cdf(args) -> None:
    """Plot the cumulative depth agreement against SFINCS and, if given, SF Bay CoSMoS."""
    ids = pass_tile_ids(Path(args.out_dir), args.all_tiles)
    out = _report_dir(args)
    sf = sfincs_cdfs(Path(args.out_dir) / "tiles", ids)
    bay = pd.read_csv(args.sf_bay_cdf) if args.sf_bay_cdf else None
    rows = {"tau_m": CDF_TAUS_M}
    for k, (lo, hi) in sf.items():
        rows[f"{k}_low"], rows[f"{k}_high"] = lo, hi
    pd.DataFrame(rows).to_csv(out / "sfincs_depth_cdf.csv", index=False)

    panels = [("a", f"eikonal vs SFINCS ({len(ids)} tiles, same DEM and forcing)", sf)]
    if bay is not None:
        panels.append(("b", "eikonal vs CoSMoS, SF Bay (includes DEM and forcing differences)",
                       {c: (bay[c].to_numpy(), bay[c].to_numpy()) for c in bay.columns if c != "tau_m"}))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)
    for ax, (letter, title, curves) in zip(axes[0], panels):
        taus = CDF_TAUS_M if curves is sf else bay.tau_m.to_numpy()
        for den, (label, color) in DENOMINATORS.items():
            for w, ls in (("area", "-"), ("pop", "--")):
                lo, hi = curves[f"{den}_{w}"]
                ax.plot(taus, (lo + hi) / 2, color=color, ls=ls, lw=1.6,
                        label=f"{label}, {'area' if w == 'area' else 'population'}")
                if not np.allclose(lo, hi):
                    ax.fill_between(taus, lo, hi, color=color, alpha=0.15, lw=0)
        for t in (0.1, 0.2, 0.3, 0.5):
            ax.axvline(t, color="#d9d9d9", lw=0.8, zorder=0)
        ax.set(xlim=(0, 1.5), ylim=(0, 1), xlabel="|eikonal depth - reference depth| (m)",
               ylabel="share of cells within", title=title)
        ax.title.set_fontsize(9)
        draw_panel_letter(ax, letter)
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "X1c_depth_cdf_by_denominator.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out / 'X1c_depth_cdf_by_denominator.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["stage0", "report", "cdf"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--dedup-dir-name", default="calibration_roughness_SFINCS_dedup")
    parser.add_argument("--fsf", type=float, default=12.0, help="friction factor of the result files (old layout: effective)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--all-tiles", action="store_true",
                        help="report/cdf: every passed tile (into <out-dir>/full) instead of the subset")
    parser.add_argument("--sf-bay-cdf", default=None, help="cdf: sf_bay_depth_cdf.csv of sf_bay_depth_reliability.py")
    args = parser.parse_args()
    {"stage0": stage0, "report": report, "cdf": cdf}[args.stage](args)


if __name__ == "__main__":
    main()
