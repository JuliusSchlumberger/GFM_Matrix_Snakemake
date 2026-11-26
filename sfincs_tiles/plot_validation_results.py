"""Plot one flood-model run against the selected connected bathtub, both vs SFINCS.

Writes {base}/figures/*_<run key>[_<group>] and {base}/metrics_overview_table_<run key>[_<group>].csv
(every selected run in the table). --group restricts to the calibration or validation tiles.

Usage:
    python plot_validation_results.py --base-dir-name calibration_roughness_SFINCS \
        --run-key eikonal_landcover_duration_fsf0.25_a1 --group validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import GROUPS, load_results, metrics_overview, pooled_depth_joint, run_label, tile_table  # noqa: E402
from flood_agreement import (  # noqa: E402
    DEPTH_CATEGORY_EDGES, metrics_from_counts,
)
from gfm_config import read_figures_root, read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM, LAND_COLOR, MM_PER_INCH, SEQUENTIAL_CMAP, apply_style, panel_letters, save_figure,
    write_caption,
)

FIG_WIDTH_IN = DOUBLE_COLUMN_MM / MM_PER_INCH

COAST_COLOR = "#b8b8b3"
HC_BATHTUB_COLOR = "#8a8a86"
EA_BATHTUB_COLOR = "#2a78d6"
LABEL = {"ref": "reference", "model": "model"}  # set in main from the two runs


def _category_bin_labels(edges: np.ndarray) -> list[str]:
    labels = [f"<{edges[0]:.1f}"]
    labels += [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:.1f}")
    return labels


def plot_depth_category_alignment(pooled: dict, out_path: Path) -> None:
    """Plot the row-normalised (per SFINCS bin) depth-bin confusion matrix, model run vs SFINCS.

    Bins 0.1-1.5 m in 0.2 m steps plus >1.5 m; both axes are wet-thresholded at
    WET_THRESHOLD_M = 0.10 m, so the always-empty <0.1 m bin is dropped.
    """
    labels = _category_bin_labels(DEPTH_CATEGORY_EDGES)[1:]
    model = "model"
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    fig.subplots_adjust(left=0.15, right=0.80, top=0.95, bottom=0.22)
    joint = pooled[model]
    hist = joint["hist_category"]
    if hist is None or joint["n"] == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, out_path)
        plt.close(fig)
        write_caption(out_path, ["No data for this run."])
        print(f"Wrote {out_path}")
        return
    hist = hist[1:, 1:]
    row_sums = hist.sum(axis=1, keepdims=True)
    pct = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0) * 100
    im = ax.imshow(pct, origin="upper", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"{LABEL[model]} depth bin (m)")
    ax.set_ylabel("SFINCS depth bin (m)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if pct[i, j] >= 1:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if pct[i, j] > 50 else "black")

    cax = fig.add_axes((0.83, 0.22, 0.03, 0.73))
    fig.colorbar(im, cax=cax, label="% of SFINCS-bin cells")
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        f"Depth-bin confusion matrix, {LABEL[model]} vs SFINCS, pooled across all tiles (both-wet "
        "cells, 0.2 m bins above the 0.10 m wet threshold).",
        "Each row (SFINCS depth bin) is normalised to 100%; colour and the printed number are the "
        f"% of that SFINCS bin's cells landing in each {LABEL[model]} depth bin. A well-calibrated "
        "run concentrates mass on the diagonal.",
    ])
    print(f"Wrote {out_path}")


def _csi(matched, model_only, sfincs_only):
    # CSI = Jaccard index, same as flood_agreement.metrics_from_counts["CSI"]
    denom = matched + model_only + sfincs_only
    return np.where(denom > 0, matched / denom, np.nan)


def plot_extent_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile flooded extent of each model vs SFINCS (log-log)."""
    sfincs = df["sfincs_km2"].to_numpy(dtype=float)
    ref = df["ref_km2"].to_numpy(dtype=float)
    model = df["model_km2"].to_numpy(dtype=float)
    valid_ref = np.isfinite(sfincs) & np.isfinite(ref) & (sfincs > 0) & (ref > 0)
    valid_model = np.isfinite(sfincs) & np.isfinite(model) & (sfincs > 0) & (model > 0)
    all_vals = np.concatenate([sfincs[valid_ref | valid_model], ref[valid_ref], model[valid_model]])
    lim = [all_vals.min(), all_vals.max()] if len(all_vals) else [0.1, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / 2 + 0.3), layout="constrained")
    panels = [("a", axes[0], ref, valid_ref, HC_BATHTUB_COLOR, LABEL["ref"]),
              ("b", axes[1], model, valid_model, EA_BATHTUB_COLOR, LABEL["model"])]
    for letter, ax, y, valid, color, label in panels:
        ax.scatter(sfincs[valid], y[valid], s=10, alpha=0.5, color=color, edgecolors="none")
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_box_aspect(1)
        ax.set_xlabel("SFINCS flooded area (km2)", fontsize=7)
        ax.set_ylabel(f"{label} flooded area (km2)", fontsize=7)
        ax.tick_params(labelsize=7)
        panel_letters(ax, letter, fontsize=8, label=f"{label} vs SFINCS, n={int(valid.sum())}")

    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        "Flooded extent agreement against SFINCS, per tile (log-log); dotted line is 1:1; both "
        "panels share the same axis limits.",
        f"(a) {LABEL['ref']} (reference run) vs SFINCS. (b) {LABEL['model']} (model run) vs SFINCS.",
    ])
    print(f"Wrote {out_path}")


def plot_agreement_hist(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile CSI histograms; print per-tile and area-weighted (pooled) CSI."""
    ref_csi = _csi(df["ref_matched_km2"], df["ref_only_km2"], df["ref_sfincs_only_km2"])
    model_csi = _csi(df["model_matched_km2"], df["model_only_km2"], df["model_sfincs_only_km2"])

    bt_mask = np.isfinite(ref_csi)
    ek_mask = np.isfinite(model_csi)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.2), layout="constrained")
    bins = np.linspace(0, 1, 26)
    bt_valid = ref_csi[bt_mask]
    ek_valid = model_csi[ek_mask]
    ax.hist(bt_valid, bins=bins, alpha=0.6, color=HC_BATHTUB_COLOR,
            label=f"{LABEL['ref']} (median={np.median(bt_valid):.2f}, n={len(bt_valid)})")
    ax.hist(ek_valid, bins=bins, alpha=0.6, color=EA_BATHTUB_COLOR,
            label=f"{LABEL['model']} (median={np.median(ek_valid):.2f}, n={len(ek_valid)})")
    ax.set_xlabel("extent agreement with SFINCS (CSI)", fontsize=7)
    ax.set_ylabel("number of tiles", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        "Per-tile extent agreement with SFINCS (CSI = matched / (matched + model-only + "
        "SFINCS-only)), reference run vs model run.",
    ])
    print(f"Wrote {out_path}")

    for prefix, label, mask, valid in (
        ("ref", LABEL["ref"], bt_mask, bt_valid),
        ("model", LABEL["model"], ek_mask, ek_valid),
    ):
        pooled = metrics_from_counts(
            df[f"{prefix}_matched_km2"][mask].sum(),
            df[f"{prefix}_only_km2"][mask].sum(),
            df[f"{prefix}_sfincs_only_km2"][mask].sum(),
        )
        print(f"  {label} CSI: per-tile median={np.median(valid):.3f}, per-tile mean={np.mean(valid):.3f} "
              f"(n={len(valid)}, unweighted) | area-weighted (pooled) mean={pooled['CSI']:.3f}")


def plot_depth_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile median flood depth of each model vs SFINCS."""
    sfincs = df["sfincs_depth_median_m"].to_numpy(dtype=float)
    ref = df["ref_depth_median_m"].to_numpy(dtype=float)
    model = df["model_depth_median_m"].to_numpy(dtype=float)
    valid_ref = np.isfinite(sfincs) & np.isfinite(ref)
    valid_model = np.isfinite(sfincs) & np.isfinite(model)
    all_vals = np.concatenate([sfincs[valid_ref | valid_model], ref[valid_ref], model[valid_model]])
    lim = [0, all_vals.max()] if len(all_vals) else [0, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / 2 + 0.3), layout="constrained")
    panels = [("a", axes[0], ref, valid_ref, HC_BATHTUB_COLOR, LABEL["ref"]),
              ("b", axes[1], model, valid_model, EA_BATHTUB_COLOR, LABEL["model"])]
    for letter, ax, y, valid, color, label in panels:
        ax.scatter(sfincs[valid], y[valid], s=10, alpha=0.5, color=color, edgecolors="none")
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_box_aspect(1)
        ax.set_xlabel("SFINCS median flood depth (m)", fontsize=7)
        ax.set_ylabel(f"{label} median flood depth (m)", fontsize=7)
        ax.tick_params(labelsize=7)
        panel_letters(ax, letter, fontsize=8, label=f"{label} vs SFINCS, n={int(valid.sum())}")

    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        "Median flood depth agreement against SFINCS, per tile; dotted line is 1:1; both panels "
        "share the same axis limits.",
        f"(a) {LABEL['ref']} (reference run) vs SFINCS. (b) {LABEL['model']} (model run) vs SFINCS.",
    ])
    print(f"Wrote {out_path}")


def _agreement_df(df: pd.DataFrame) -> pd.DataFrame:
    csi = _csi(df["model_matched_km2"], df["model_only_km2"], df["model_sfincs_only_km2"])
    plot_df = df.assign(csi=csi, misalignment=1 - csi)
    return plot_df[np.isfinite(plot_df["csi"]) & plot_df["lon"].notna()]


def plot_tile_agreement_map(df: pd.DataFrame, out_path: Path) -> None:
    """Map per-tile model-run vs SFINCS extent agreement (CSI) globally.

    Tiles flooded in neither model (CSI NaN) are left out.
    """
    csi = _csi(df["model_matched_km2"], df["model_only_km2"], df["model_sfincs_only_km2"])
    plot_df = df.assign(csi=csi)
    plot_df = plot_df[plot_df["lon"].notna() & np.isfinite(plot_df["csi"])]

    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.56), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    # CSI is a magnitude (0-1), not a diverging quantity - single-hue sequential, never red/green.
    sc = ax.scatter(
        plot_df["lon"], plot_df["lat"], transform=ccrs.PlateCarree(),
        c=plot_df["csi"], cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, s=10, alpha=0.9,
        linewidths=0.2, edgecolors="white", zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"{LABEL['model']}-SFINCS extent agreement (CSI)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        f"Per-tile extent agreement (CSI) between {LABEL['model']} and SFINCS, every scored tile "
        "globally. Equal Earth projection; land shown in light grey for orientation.",
        f"Colour: CSI = matched / (matched + {LABEL['model']}-only + SFINCS-only), 0 (no overlap) "
        "to 1 (perfect match).",
    ])
    print(f"Wrote {out_path}")


def plot_agreement_vs_tile_size(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile misalignment (1 - CSI) against tile land area (log scale).

    Reports Pearson r on log10(area) and Spearman rho when >= 3 tiles are valid.
    """
    plot_df = _agreement_df(df)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.72), layout="constrained")
    size = plot_df["land_km2"].to_numpy(dtype=float)
    mis = plot_df["misalignment"].to_numpy(dtype=float)
    valid = np.isfinite(size) & (size > 0) & np.isfinite(mis)
    ax.scatter(size[valid], mis[valid], s=12, alpha=0.5, color=EA_BATHTUB_COLOR, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("tile land area (km2, log)", fontsize=7)
    ax.set_ylabel("misalignment (1 - CSI)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    stats_line = f"n={int(valid.sum())}"
    if valid.sum() >= 3:
        r_log = np.corrcoef(np.log10(size[valid]), mis[valid])[0, 1]
        rho, _ = spearmanr(size[valid], mis[valid])
        stats_line += f", pearson r={r_log:.3f} (log-area), spearman rho={rho:.3f}"
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        f"Per-tile misalignment (1 - CSI) between {LABEL['model']} and SFINCS against tile land "
        "area (log scale) - does agreement degrade for small or large tiles?",
        stats_line,
    ])
    print(f"Wrote {out_path}")


def main() -> None:
    """Write all validation figures and tables for one run against the connected bathtub."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--run-key", required=True, help="the run to plot (variant_runs_per_tile.csv)")
    parser.add_argument("--reference-run-key", default=None,
                        help="default: the connected bathtub's run in variant_selection.json")
    parser.add_argument("--group", choices=GROUPS, default=None, help="only these tiles (default: all)")
    parser.add_argument("--tile-ids-file", default=None, help="restrict to these tiles (default: every tile with a result)")
    args = parser.parse_args()
    apply_style()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    fig_dir = read_figures_root(Path(args.config)) / "sfincs_runs" / args.base_dir_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((base_dir / SELECTION_NAME).read_text())
    key = args.run_key
    ref_key = args.reference_run_key or selection["variants"]["bathtub_connected"]["run_key"]
    sfx = f"_{key}" + (f"_{args.group}" if args.group else "")

    ids = Path(args.tile_ids_file).read_text().split() if args.tile_ids_file else None
    results = load_results(base_dir, ids)
    if args.group:
        results = {t: r for t, r in results.items() if r.get("group") == args.group}
    LABEL.update(ref=run_label(ref_key, results), model=run_label(key, results))
    meta = pd.read_csv(base_dir / "tile_selection_metadata.csv")
    df = tile_table(results, {"ref": ref_key, "model": key}, meta)
    print(f"{len(df)} tile result(s) loaded: {LABEL['model']} against {LABEL['ref']}")
    df.to_csv(fig_dir / f"validation_summary{sfx}.csv", index=False)
    print(f"Wrote {fig_dir / f'validation_summary{sfx}.csv'}")

    plot_extent_scatter(df, fig_dir / f"extent_scatter{sfx}.png")
    plot_agreement_hist(df, fig_dir / f"extent_agreement_hist{sfx}.png")
    plot_depth_scatter(df, fig_dir / f"depth_scatter{sfx}.png")
    plot_tile_agreement_map(df, fig_dir / f"agreement_map{sfx}.png")
    plot_agreement_vs_tile_size(df, fig_dir / f"agreement_vs_tile_size{sfx}.png")

    pooled_depth = {"ref": pooled_depth_joint(results, ref_key), "model": pooled_depth_joint(results, key)}
    plot_depth_category_alignment(pooled_depth, fig_dir / f"depth_category_alignment{sfx}.png")

    keys = [e["run_key"] for e in selection["variants"].values()]
    table = metrics_overview(results, keys if key in keys else [*keys, key])
    table_path = base_dir / f"metrics_overview_table{sfx}.csv"
    table.to_csv(table_path, index=False)
    print(f"\nWrote {table_path}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
