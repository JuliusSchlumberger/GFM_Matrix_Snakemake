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
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, depth_error_metrics_from_pooled, metrics_from_counts,
)
from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import LAND_COLOR  # noqa: E402

COAST_COLOR = "#b8b8b3"
HC_BATHTUB_COLOR = "#8a8a86"
EA_BATHTUB_COLOR = "#2a78d6"
LABEL = {"ref": "reference", "model": "model"}  # set in main from the two runs


def _category_bin_labels(edges: np.ndarray) -> list[str]:
    labels = [f"<{edges[0]:.1f}"]
    labels += [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:.1f}")
    return labels


def plot_depth_correlation(pooled: dict, out_path: Path) -> None:
    """Plot pooled cell-level depth density heatmaps (0.1 m bins) for each model vs SFINCS."""
    lo, hi = float(DEPTH_CORR_FINE_EDGES[0]), float(DEPTH_CORR_FINE_EDGES[-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, model, label in [(axes[0], "ref", LABEL["ref"]), (axes[1], "model", LABEL["model"])]:
        joint = pooled[model]
        if joint["hist_fine"] is None or joint["n"] == 0:
            ax.set_title(f"{label}: no data")
            ax.axis("off")
            continue
        m = depth_error_metrics_from_pooled(joint)
        core = joint["hist_fine"][1:-1, 1:-1]  # drop the <0m/>3m catch-all rows/cols
        im = ax.imshow(np.log1p(core.T), origin="lower", extent=[lo, hi, lo, hi],
                        aspect="auto", cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="white", linestyle=":", linewidth=1)
        ax.set_xlabel("SFINCS depth (m)")
        ax.set_ylabel(f"{label} depth (m)")
        ax.set_title(f"{label} vs SFINCS (n={int(joint['n']):,})\n"
                      f"r={m['r']:.3f}  bias={m['bias_m']:+.2f}m  RMSE={m['rmse_m']:.2f}m", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log(1 + cell count)")
        band_key = [k for k in m if k.startswith("pct_within_")][0]
        print(f"  {label} depth agreement: n={int(joint['n']):,} r={m['r']:.3f} bias={m['bias_m']:+.3f}m "
              f"RMSE={m['rmse_m']:.3f}m median_error={m['median_error_m']:+.3f}m {band_key}={m[band_key]:.1f}%")

    fig.suptitle("Cell-level depth agreement against SFINCS, pooled across all tiles\n"
                  "(every mutually-flooded cell, dotted = 1:1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


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
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
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
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _csi(matched, model_only, sfincs_only):
    # CSI = Jaccard index, same as flood_agreement.metrics_from_counts["CSI"]
    denom = matched + model_only + sfincs_only
    return np.where(denom > 0, matched / denom, np.nan)


def plot_extent_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile flooded extent of each model vs SFINCS (log-log)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["ref_km2"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.scatter(x[valid], y[valid], s=14, alpha=0.5, color=HC_BATHTUB_COLOR, edgecolors="none")
    if valid.any():
        lim = [min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())]
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFINCS flooded area (km2)")
    ax.set_ylabel(f"{LABEL['ref']} flooded area (km2)")
    ax.set_title(f"{LABEL['ref']} vs SFINCS (n={int(valid.sum())})")

    ax = axes[1]
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["model_km2"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.scatter(x[valid], y[valid], s=14, alpha=0.5, color=EA_BATHTUB_COLOR, edgecolors="none")
    if valid.any():
        lim = [min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())]
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFINCS flooded area (km2)")
    ax.set_ylabel(f"{LABEL['model']} flooded area (km2)")
    ax.set_title(f"{LABEL['model']} vs SFINCS (n={int(valid.sum())})")

    fig.suptitle("Flooded extent agreement against SFINCS (log-log, dotted = 1:1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_agreement_hist(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile CSI histograms; print per-tile and area-weighted (pooled) CSI."""
    ref_csi = _csi(df["ref_matched_km2"], df["ref_only_km2"], df["ref_sfincs_only_km2"])
    model_csi = _csi(df["model_matched_km2"], df["model_only_km2"], df["model_sfincs_only_km2"])

    bt_mask = np.isfinite(ref_csi)
    ek_mask = np.isfinite(model_csi)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 26)
    bt_valid = ref_csi[bt_mask]
    ek_valid = model_csi[ek_mask]
    ax.hist(bt_valid, bins=bins, alpha=0.6, color=HC_BATHTUB_COLOR,
            label=f"{LABEL['ref']} (median={np.median(bt_valid):.2f}, n={len(bt_valid)})")
    ax.hist(ek_valid, bins=bins, alpha=0.6, color=EA_BATHTUB_COLOR,
            label=f"{LABEL['model']} (median={np.median(ek_valid):.2f}, n={len(ek_valid)})")
    ax.set_xlabel("extent agreement with SFINCS (CSI: matched / (matched + model-only + SFINCS-only))")
    ax.set_ylabel("number of tiles")
    ax.legend(fontsize=9)
    ax.set_title("Extent agreement with SFINCS per tile")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, col, color, label in [
        (axes[0], "ref_depth_median_m", HC_BATHTUB_COLOR, LABEL["ref"]),
        (axes[1], "model_depth_median_m", EA_BATHTUB_COLOR, LABEL["model"]),
    ]:
        x = df["sfincs_depth_median_m"].to_numpy(dtype=float)
        y = df[col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=14, alpha=0.5, color=color, edgecolors="none")
        if valid.any():
            lim = [0, max(x[valid].max(), y[valid].max())]
            ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("SFINCS median flood depth (m)")
        ax.set_ylabel(f"{label} median flood depth (m)")
        ax.set_title(f"{label} vs SFINCS (n={int(valid.sum())})")

    fig.suptitle("Median flood depth agreement against SFINCS (dotted = 1:1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
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
    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    sc = ax.scatter(
        plot_df["lon"], plot_df["lat"], transform=ccrs.PlateCarree(),
        c=plot_df["csi"], cmap="RdYlGn", vmin=0, vmax=1, s=22, alpha=0.9,
        linewidths=0.3, edgecolors="white", zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"{LABEL['model']}-SFINCS extent agreement (CSI)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_agreement_vs_tile_size(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-tile misalignment (1 - CSI) against tile land area (log scale).

    Reports Pearson r on log10(area) and Spearman rho when >= 3 tiles are valid.
    """
    plot_df = _agreement_df(df)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    size = plot_df["land_km2"].to_numpy(dtype=float)
    mis = plot_df["misalignment"].to_numpy(dtype=float)
    valid = np.isfinite(size) & (size > 0) & np.isfinite(mis)
    ax.scatter(size[valid], mis[valid], s=18, alpha=0.5, color=EA_BATHTUB_COLOR, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("tile land area (km2, log)")
    ax.set_ylabel("misalignment (1 - CSI)")
    ax.grid(True, alpha=0.3)
    title = f"Misalignment vs. tile size (n={int(valid.sum())})"
    if valid.sum() >= 3:
        r_log = np.corrcoef(np.log10(size[valid]), mis[valid])[0, 1]
        rho, _ = spearmanr(size[valid], mis[valid])
        title += f"\npearson r={r_log:.3f} (log-area), spearman rho={rho:.3f}"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
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

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    fig_dir = base_dir / "figures"
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
    plot_depth_correlation(pooled_depth, fig_dir / f"depth_correlation{sfx}.png")
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
