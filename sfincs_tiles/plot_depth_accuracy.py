"""Plot depth accuracy vs SFINCS: per swept run, and for the selected runs as tolerances by depth class.

depth_accuracy_by_run[_<group>].png (variant_runs_depth_accuracy.csv, variant_runs_pooled.csv): per
variant with more than one run, against its parameter, the share of SFINCS-wet cells within 10/20/50 cm,
the bias per SFINCS depth class and CSI. depth_tolerance.png (the tiles' depth-difference histograms):
for the selected runs, which depth tolerance the model meets, by SFINCS depth class and per tile.

Usage:
    python plot_depth_accuracy.py --base-dir-name calibration_roughness_SFINCS [--group calibration]
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
from calibration_paths import DEPTH_ACCURACY_NAME, SELECTION_NAME, threaded_map  # noqa: E402
from calibration_results import DISPLAY_LABEL, GROUPS, RESULT_NAME, read_hists  # noqa: E402
from flood_agreement import DIFF_EDGES, SFINCS_CLASS_LABELS, diff_stats  # noqa: E402
from gfm_config import read_figures_root, read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM, MM_PER_INCH, MUTED, apply_style, panel_letters, save_figure, write_caption,
)

TAU_COLORS = {0.1: "#9ecae1", 0.2: "#4292c6", 0.5: "#08306b"}
BIAS_CLASSES = {"0.1-0.3": "#fdae6b", "0.5-1": "#e6550d", "2-3": "#7f2704"}
WEIGHT_STYLE = {"area": "-", "pop": "--"}
SHORT_LABEL = {"straight_line": "Straight line", "straight_line_cap": "Straight line + cap",
               "eikonal_uniform": "Eik. uniform", "eikonal_landcover": "Eik. land-cover",
               "eikonal_uniform_cap": "Eik. uniform + cap", "eikonal_landcover_cap": "Eik. land-cover + cap",
               "eikonal_uniform_duration": "Eik. uniform + dur.", "eikonal_landcover_duration": "Eik. land-cover + dur.",
               "bathtub_connected": "Connected bathtub"}
TOLERANCE_VARIANTS = ("eikonal_uniform_cap", "eikonal_landcover_cap", "bathtub_connected")
TOLERANCE_STYLE = {"eikonal_uniform_cap": ("#0072B2", "-"), "eikonal_landcover_cap": ("#CC79A7", "--"),
                   "bathtub_connected": ("#E69F00", ":")}
TOLERANCE_GRID_M = np.round(np.arange(0.0, 1.0001, 0.01), 2)
REFERENCE_TOLERANCES_M = (0.1, 0.2, 0.5)
MIN_SFINCS_WET_KM2 = 1.0  # tiles in the per-tile panel
ALIGNMENT_BANDS_M = (0.05, 0.1, 0.25, 0.5)  # half-widths of the bands around the 1:1 line
TILE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CLASS_X_M = (0.2, 0.4, 0.75, 1.25, 1.75, 2.5, 3.25)  # SFINCS_CLASS_LABELS on a depth axis; > 3 m at 3.25
MIN_CLASS_KM2 = 0.1  # a tile counts in a depth class's spread with at least this much SFINCS-flooded area there


def swept_axis(variant: str) -> tuple[str, str]:
    if variant.endswith("_duration"):
        return "alpha", "Duration exponent"
    if "landcover" in variant:
        return "parameter", "fsf"
    return "parameter", "Rate (m/km)"


def plot(depth: pd.DataFrame, pooled: pd.DataFrame, selection: dict, group: str, out: Path) -> None:
    variants = [v for v in DISPLAY_LABEL if pooled[pooled.variant == v].run_key.nunique() > 1]
    if not variants:
        print("no variant with more than one run: nothing to plot")
        return
    fig, axes = plt.subplots(3, len(variants), figsize=(DOUBLE_COLUMN_MM / MM_PER_INCH, 6.8), squeeze=False,
                             sharey="row", layout="constrained")
    for j, variant in enumerate(variants):
        col, xlabel = swept_axis(variant)
        d = depth[depth.variant == variant].sort_values(col)
        p = pooled[pooled.variant == variant].sort_values(col)
        ref = d[(d.cells == "reference_wet") & (d.sfincs_class_m == "all")]
        for weight, ls in WEIGHT_STYLE.items():
            w = ref[ref.weight == weight]
            for tau, color in TAU_COLORS.items():
                axes[0, j].plot(w[col], w[f"within_{tau:g}m"], color=color, ls=ls, lw=1.2, marker="o", ms=2.5,
                                label=f"within {tau * 100:.0f} cm, {'area' if weight == 'area' else 'population'}")
        both = d[(d.cells == "both_wet") & (d.weight == "area")]
        for cls, color in BIAS_CLASSES.items():
            b = both[both.sfincs_class_m == cls]
            axes[1, j].plot(b[col], b.bias_m, color=color, lw=1.2, marker="o", ms=2.5, label=f"SFINCS {cls} m")
        axes[1, j].axhline(0, color="#525252", lw=0.8)
        axes[2, j].plot(p[col], p.CSI, color="#08519c", lw=1.2, marker="o", ms=2.5, label="area")
        if "CSI_pop" in p:
            axes[2, j].plot(p[col], p.CSI_pop, color="#08519c", ls="--", lw=1.2, marker="o", ms=2.5, label="population")
        chosen = (selection.get("variants") or {}).get(variant)
        if chosen and chosen.get(col) is not None:
            for ax in axes[:, j]:
                ax.axvline(chosen[col], color="#bdbdbd", lw=1.0, zorder=0)
        for row, ax in enumerate(axes[:, j]):
            letter = "abcdefghijklmnopqrstuvwxyz"[row * len(variants) + j]
            panel_letters(ax, letter, fontsize=6.5, label=SHORT_LABEL.get(variant, variant) if row == 0 else None)
        axes[2, j].set_xlabel(xlabel, fontsize=7)
    axes[0, 0].set(ylabel="share of SFINCS-wet cells within", ylim=(0, 1))
    axes[1, 0].set_ylabel("bias, eikonal - SFINCS (m)\nboth-wet cells, by SFINCS depth")
    axes[2, 0].set(ylabel="CSI (extent)", ylim=(0, 1))
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    for ax in (axes[0, 0], axes[1, 0], axes[2, 0]):
        ax.yaxis.label.set_fontsize(7)
    for ax in axes[:, -1]:
        ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_figure(fig, out)
    plt.close(fig)
    write_caption(out, [
        f"Depth and extent agreement vs SFINCS per run, {group} group - each column is one swept "
        "variant, run against its own friction parameter (or duration exponent alpha).",
        "Row 1: share of SFINCS-wet cells within 10/20/50 cm (colour), solid = area-weighted, "
        "dashed = population-weighted. Row 2: bias (eikonal - SFINCS, m) on both-wet cells, by "
        "SFINCS depth class (colour). Row 3: CSI (extent), solid = area, dashed = population.",
        "Grey vertical line: the run selected for this variant (chosen by CSI; depth is reported "
        "only, not used to select).",
    ])
    print(f"wrote {out}")


def tolerance_label(variant: str, entry: dict) -> str:
    """Return the legend label of a selected run, e.g. "Eik. uniform + cap (rate 0.1)"."""
    if entry.get("parameter") is None:
        return SHORT_LABEL[variant]
    return f"{SHORT_LABEL[variant]} ({'fsf' if 'landcover' in variant else 'rate'} {entry['parameter']:g})"


def load_tolerance_hists(base_dir: Path, run_keys: dict[str, str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Pool the selected runs' depth-difference histograms over tiles; also each tile's own stats.

    Returns ({variant: {weight: (CELL_KINDS, classes, bins) histogram}}, per-tile frame with the
    SFINCS-wet area and the 80 % tolerance of every run over the tile's SFINCS-wet cells, and a
    per-tile, per-SFINCS-depth-class frame with the area share inside each ALIGNMENT_BANDS_M band).
    """
    pooled: dict = {v: {} for v in run_keys}
    rows, class_rows = [], []
    bands = np.array(ALIGNMENT_BANDS_M)
    # histograms left by an unfinished rerun have no result file beside them
    paths = [p for p in sorted(base_dir.glob(f"*/outputs/{DEPTH_ACCURACY_NAME}")) if (p.parent / RESULT_NAME).exists()]
    wanted = {f"{key}__{weight}" for key in run_keys.values() for weight in ("area", "pop")}
    for path, hists in zip(paths, threaded_map(lambda p: read_hists(p, wanted.__contains__), paths)):
        tile = path.parent.parent.name
        if hists is None:
            continue
        for variant, key in run_keys.items():
            for weight in ("area", "pop"):
                name = f"{key}__{weight}"
                if name not in hists:
                    continue
                h = hists[name]
                pooled[variant][weight] = pooled[variant].get(weight, 0) + h
                if weight == "area":
                    reference = (h[0] + h[1]).sum(axis=0)
                    rows.append({"tile_id": int(tile), "variant": variant, "sfincs_wet_km2": float(reference.sum()),
                                 **{k: v for k, v in diff_stats(reference).items() if k != "n"}})
                    for label, hc in zip(SFINCS_CLASS_LABELS, h[0] + h[1]):
                        if hc.sum() > 0:
                            class_rows.append({"tile_id": int(tile), "variant": variant, "sfincs_class_m": label,
                                               "sfincs_wet_km2": float(hc.sum()),
                                               **{f"within_{b:g}m": s for b, s in zip(bands, _share_within(hc, bands))}})
    return pooled, pd.DataFrame(rows), pd.DataFrame(class_rows)


def _hist_quantiles(hist: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    # 1 cm bins of DIFF_EDGES plus catch-alls, placed just beyond -3 / +3 m.
    centres = np.concatenate([[DIFF_EDGES[0] - 0.005], (DIFF_EDGES[:-1] + DIFF_EDGES[1:]) / 2, [DIFF_EDGES[-1] + 0.005]])
    cum = np.cumsum(hist) / hist.sum()
    return np.array([centres[min(np.searchsorted(cum, q), len(centres) - 1)] for q in qs])


def _share_within(hist: np.ndarray, taus: np.ndarray) -> np.ndarray:
    inner = hist[1:-1]
    reach = np.maximum(np.abs(DIFF_EDGES[:-1]), np.abs(DIFF_EDGES[1:]))
    return np.array([inner[reach <= tau + 1e-9].sum() for tau in taus]) / hist.sum()


def tolerance_table(pooled: dict) -> pd.DataFrame:
    """Return diff_stats per selected run, weight and SFINCS depth class over all SFINCS-wet cells."""
    rows = []
    for variant, by_weight in pooled.items():
        for weight, h in by_weight.items():
            reference = h[0] + h[1]
            for label, hc in [*zip(SFINCS_CLASS_LABELS, reference), ("all", reference.sum(axis=0))]:
                rows.append({"variant": variant, "weight": weight, "sfincs_class_m": label, **diff_stats(hc)})
    return pd.DataFrame(rows)


def plot_depth_tolerance(pooled: dict, per_tile: pd.DataFrame, labels: dict[str, str], out: Path) -> None:
    """Plot which depth tolerance the selected runs meet against SFINCS, by depth class and per tile."""
    variants = [v for v in TOLERANCE_VARIANTS if pooled.get(v)]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_MM / MM_PER_INCH, 5.2), layout="constrained")
    ax = axes[0, 0]
    width = 0.8 / len(variants)
    lo, hi = 0.0, 0.0
    for k, v in enumerate(variants):
        color, _ = TOLERANCE_STYLE[v]
        reference = pooled[v]["area"][0] + pooled[v]["area"][1]
        for c, hc in enumerate(reference):
            if hc.sum() <= 0:
                continue
            q5, q25, q50, q75, q95 = _hist_quantiles(hc, (0.05, 0.25, 0.5, 0.75, 0.95))
            lo, hi = min(lo, q5), max(hi, q95)
            x = c - 0.4 + width * (k + 0.5)
            ax.plot([x, x], [q5, q95], color=color, linewidth=0.7)
            ax.add_patch(plt.Rectangle((x - width * 0.4, q25), width * 0.8, q75 - q25, facecolor=color, alpha=0.6,
                                       edgecolor=color, linewidth=0.5))
            ax.plot([x - width * 0.4, x + width * 0.4], [q50, q50], color="black", linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_xticks(range(len(SFINCS_CLASS_LABELS)), SFINCS_CLASS_LABELS, fontsize=7)
    ax.set_xlim(-0.6, len(SFINCS_CLASS_LABELS) - 0.4)
    ax.set_ylim(max(lo - 0.1, -3.1), min(hi + 0.1, 3.1))
    ax.set(xlabel="SFINCS depth (m)", ylabel="Depth error, model - SFINCS (m)")
    panel_letters(ax, "a", fontsize=8)
    for ax, weight, letter, ylabel in ((axes[0, 1], "area", "b", "Share of SFINCS-flooded area"),
                                       (axes[1, 0], "pop", "c", "Share of people in SFINCS-flooded cells")):
        for v in variants:
            if weight not in pooled[v]:
                continue
            color, ls = TOLERANCE_STYLE[v]
            h = (pooled[v][weight][0] + pooled[v][weight][1]).sum(axis=0)
            ax.plot(TOLERANCE_GRID_M, _share_within(h, TOLERANCE_GRID_M), color=color, linestyle=ls, linewidth=1.2,
                    label=labels[v])
        for tau in REFERENCE_TOLERANCES_M:
            ax.axvline(tau, color=MUTED, linewidth=0.5, linestyle=":")
        ax.axhline(0.8, color=MUTED, linewidth=0.5, linestyle=":")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Depth tolerance (m)", ylabel=ylabel)
        panel_letters(ax, letter, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[1, 1]
    for v in variants:
        color, ls = TOLERANCE_STYLE[v]
        tau80 = np.sort(per_tile.loc[(per_tile["variant"] == v) & (per_tile["sfincs_wet_km2"] >= MIN_SFINCS_WET_KM2),
                                     "tau80_abs_m"].dropna().to_numpy())
        ax.plot(tau80, np.arange(1, tau80.size + 1) / max(tau80.size, 1), color=color, linestyle=ls, linewidth=1.2)
    for tau in REFERENCE_TOLERANCES_M:
        ax.axvline(tau, color=MUTED, linewidth=0.5, linestyle=":")
    ax.set(xlim=(0, 1.5), ylim=(0, 1), xlabel="Tolerance met by 80 % of the tile's flooded area (m)",
           ylabel="Share of tiles")
    panel_letters(ax, "d", fontsize=8)
    save_figure(fig, out)
    plt.close(fig)
    n_tiles = per_tile.loc[per_tile["sfincs_wet_km2"] >= MIN_SFINCS_WET_KM2, "tile_id"].nunique()
    write_caption(out, [
        "Depth agreement with SFINCS sets the depth classes a depth-damage curve can use: read off the tolerance "
        "that 80 % of the flooded area (or people) meets.",
        "All cells SFINCS floods (> 0.10 m) on SFINCS-active land, pooled over tiles; a cell the model leaves dry "
        "counts with its own sub-threshold depth, so missed flooding is an error of about minus the SFINCS depth. "
        "Runs: " + "; ".join(labels[v] for v in variants) + " (colour and line style as in the legend of b).",
        "(a) Depth error by SFINCS depth class, area-weighted: box = interquartile range, line = median, "
        "whiskers = 5-95 %. (b, c) Share of SFINCS-flooded area (b) and of the people living in it (c, WorldPop) "
        "whose depth error is at most the tolerance on the x axis; dotted: 10, 20 and 50 cm and 80 %. "
        f"(d) Per tile (n = {n_tiles} with at least {MIN_SFINCS_WET_KM2:g} km2 SFINCS-flooded), the tolerance "
        "met by 80 % of its flooded area: the spread between places.",
        "Agreement is relative to SFINCS on the same DEM and forcing (RP100); DEM error against reality adds to "
        "it. Per-class numbers: depth_tolerance.csv.",
    ])
    print(f"wrote {out}")


def alignment_table(pooled: dict, per_class: pd.DataFrame) -> pd.DataFrame:
    """Return the area share inside each band per run and SFINCS depth class: pooled, and across tiles."""
    rows = []
    bands = np.array(ALIGNMENT_BANDS_M)
    for variant, by_weight in pooled.items():
        if "area" not in by_weight:
            continue
        reference = by_weight["area"][0] + by_weight["area"][1]
        for label, hc in zip(SFINCS_CLASS_LABELS, reference):
            tiles = per_class[(per_class["variant"] == variant) & (per_class["sfincs_class_m"] == label)
                              & (per_class["sfincs_wet_km2"] >= MIN_CLASS_KM2)]
            for band, share in zip(bands, _share_within(hc, bands) if hc.sum() > 0 else [np.nan] * len(bands)):
                col = f"within_{band:g}m"
                q = tiles[col].quantile(TILE_QUANTILES) if len(tiles) else pd.Series([np.nan] * len(TILE_QUANTILES))
                rows.append({"variant": variant, "sfincs_class_m": label, "band_m": band,
                             "sfincs_wet_km2": float(hc.sum()), "pooled_share": share, "n_tiles": len(tiles),
                             **{f"tile_p{round(qq * 100)}": val for qq, val in zip(TILE_QUANTILES, q)}})
    return pd.DataFrame(rows)


def plot_depth_alignment(table: pd.DataFrame, labels: dict[str, str], out: Path) -> None:
    """Plot, per depth band around SFINCS, the share of each tile's flooded area the model meets."""
    variants = [v for v in TOLERANCE_VARIANTS if v in set(table["variant"])]
    fig, axes = plt.subplots(1, len(ALIGNMENT_BANDS_M), figsize=(DOUBLE_COLUMN_MM / MM_PER_INCH, 2.7), sharey=True,
                             squeeze=False, layout="constrained")
    x = np.array(CLASS_X_M)
    for k, band in enumerate(ALIGNMENT_BANDS_M):
        ax = axes[0, k]
        for m, v in enumerate(variants):
            color, ls = TOLERANCE_STYLE[v]
            b = table[(table["variant"] == v) & (table["band_m"] == band)].set_index("sfincs_class_m")
            b = b.reindex(SFINCS_CLASS_LABELS)
            ax.vlines(x + (m - (len(variants) - 1) / 2) * 0.07, b["tile_p5"], b["tile_p95"], color=color,
                      linewidth=1.4, alpha=0.45)
            ax.plot(x, b["tile_p50"], color=color, linestyle=ls, marker="o", markersize=2.5, linewidth=1.1,
                    label=labels[v])
        ax.set(xlim=(0, 3.5), ylim=(0, 1), xticks=[0, 1, 2, 3], xlabel="SFINCS depth (m)",
               ylabel="Share of a tile's flooded area\ninside the band" if k == 0 else "")
        panel_letters(ax, "abcd"[k], fontsize=7, label=f"±{band * 100:.0f} cm")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(variants),
               frameon=False, fontsize=7)
    save_figure(fig, out)
    plt.close(fig)
    n_tiles = table.groupby("variant")["n_tiles"].max()
    write_caption(out, [
        "How closely each model's flood depth follows SFINCS in a typical tile: the share of the flooded area "
        "whose model depth lies within a given band of SFINCS, by how deep SFINCS floods.",
        "Per tile and SFINCS depth class (0.1-0.3, 0.3-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3 and > 3 m, plotted at "
        + ", ".join(f"{c:g}" for c in CLASS_X_M) + " m), the share of the area SFINCS floods whose model "
        "depth is within ± the band of the SFINCS depth. A cell the model leaves dry counts with its own "
        "sub-threshold depth, so a miss falls outside the narrow bands.",
        f"Line: median across tiles; bar: 5-95 % range across tiles. A tile counts in a class with at least "
        f"{MIN_CLASS_KM2:g} km2 of SFINCS-flooded area there (up to {int(n_tiles.max())} tiles per class). Colour "
        "and line style: model (legend).",
        "Per-class numbers, including the value pooled over tiles: depth_alignment.csv.",
    ])
    print(f"wrote {out}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="calibration")
    args = parser.parse_args()
    apply_style()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    depth = pd.read_csv(base_dir / "variant_runs_depth_accuracy.csv")
    pooled = pd.read_csv(base_dir / "variant_runs_pooled.csv")
    selection_path = base_dir / SELECTION_NAME
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    fig_dir = read_figures_root(Path(args.config)) / "sfincs_runs" / args.base_dir_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.group == "calibration" else f"_{args.group}"
    plot(depth[depth.group == args.group], pooled[pooled.group == args.group], selection, args.group,
         fig_dir / f"depth_accuracy_by_run{sfx}.png")

    entries = selection.get("variants") or {}
    run_keys = {v: entries[v]["run_key"] for v in TOLERANCE_VARIANTS if v in entries}
    if not run_keys:
        print("no selected run of the tolerance variants: no depth_tolerance figure")
        return
    hists, per_tile, per_class = load_tolerance_hists(base_dir, run_keys)
    labels = {v: tolerance_label(v, entries[v]) for v in run_keys}
    table = tolerance_table(hists)
    table.to_csv(fig_dir / "depth_tolerance.csv", index=False)
    per_tile.to_csv(fig_dir / "depth_tolerance_per_tile.csv", index=False)
    plot_depth_tolerance(hists, per_tile, labels, fig_dir / "depth_tolerance.png")

    alignment = alignment_table(hists, per_class)
    alignment.to_csv(fig_dir / "depth_alignment.csv", index=False)
    plot_depth_alignment(alignment, labels, fig_dir / "depth_alignment.png")


if __name__ == "__main__":
    main()
