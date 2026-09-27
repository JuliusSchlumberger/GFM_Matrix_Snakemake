"""Result visualizations for a SFINCS-vs-bathtub-vs-eikonal validation batch.
Reads every tile's outputs/summary.json (postprocess_tile_summary.py /
postprocess_sequential.py) plus tile_selection_metadata.csv for lon/lat.

--stale-cutoff optionally flags any tile whose eikonal raster predates a
given timestamp as "stale" (e.g. carried over from an earlier batch with
different settings via a recovery copy - see conversation 2026-09-25 for
why validation_sfincs_v4 needed this). Stale tiles are shown distinctly,
never silently pooled with the rest in summary statistics. Omit
--stale-cutoff for a freshly-run batch with no such history.

Usage:
    python plot_validation_results.py --base-dir-name validation_sfincs_v4
    python plot_validation_results.py --base-dir-name validation_sfincs_v4 --stale-cutoff "2026-09-24 12:00:00"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flood_agreement import DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, pearson_r_from_sums

DATA_ROOT = Path(r"P:\11212688-004-global-floodmaps\modelling")

LAND_COLOR = "#d8d8d4"
OCEAN_COLOR = "#fcfcfb"
COAST_COLOR = "#b8b8b3"
BATHTUB_COLOR = "#8a8a86"
EIKONAL_COLOR = "#2a78d6"
STALE_COLOR = "#d6572a"


def collect_summaries(base_dir: Path, stale_cutoff: pd.Timestamp | None) -> pd.DataFrame:
    """Merges each tile's separate summary_{bathtub,eikonal,sfincs}.json
    (postprocess_tile_summary.py - one independent file per model, possibly
    written by different runs at different times) into one row per tile.
    A tile missing one or more of these files simply contributes no columns
    for that model - not an error, since a model may not have been run
    (yet) for that tile."""
    by_tile: dict[str, dict] = {}
    for model in ("bathtub", "eikonal", "sfincs"):
        for p in sorted(base_dir.glob(f"*/outputs/summary_{model}.json")):
            try:
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # {model}_depth_joint (postprocess_tile_summary.py) is a nested
            # dict of sufficient stats + 2D histograms, not a scalar - kept
            # out of this flat, CSV-bound DataFrame; collect_depth_joint()
            # pools it separately for the depth-correlation/category plots.
            d = {k: v for k, v in d.items() if not k.endswith("_depth_joint")}
            row = by_tile.setdefault(d["tile_id"], {})
            row.update(d)
            if model == "eikonal":
                eik_path = p.parent / "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"
                row["has_eikonal"] = eik_path.exists()
                row["eikonal_stale"] = bool(
                    stale_cutoff is not None and eik_path.exists()
                    and pd.Timestamp(eik_path.stat().st_mtime, unit="s") < stale_cutoff
                )

    df = pd.DataFrame(list(by_tile.values()))
    df["tile_id"] = df["tile_id"].astype(int)
    if "has_eikonal" not in df.columns:
        df["has_eikonal"] = False
    if "eikonal_stale" not in df.columns:
        df["eikonal_stale"] = False
    # Every column the plotting functions below expect, even if no tile's
    # summary_{model}.json has been written yet for that model - keeps a
    # partially-complete batch plottable instead of raising KeyError.
    for name in ("bathtub", "eikonal"):
        for col in (f"{name}_km2", f"{name}_depth_median_m", f"{name}_matched_km2",
                    f"{name}_only_km2", f"{name}_sfincs_only_km2"):
            if col not in df.columns:
                df[col] = None
    for col in ("sfincs_km2", "sfincs_depth_median_m"):
        if col not in df.columns:
            df[col] = None

    meta = pd.read_csv(base_dir / "tile_selection_metadata.csv")[["tile_id", "lon", "lat"]]
    return df.merge(meta, on="tile_id", how="left")


def collect_depth_joint(base_dir: Path, stale_eikonal_tile_ids: set[int]) -> dict[str, dict]:
    """Pools every tile's {model}_depth_joint (postprocess_tile_summary.py -
    cell-level SFINCS-vs-model depth sufficient stats + 2D histograms, at
    every cell both models call wet) into one dict per model: summed n/sums
    (for a single pooled Pearson r, see flood_agreement.pearson_r_from_sums)
    and elementwise-summed hist_fine/hist_category matrices. Never averages
    a per-tile ratio - same "pool counts first" reasoning as collect_summaries'
    Jaccard/HT/FAR/CSI figures. Stale eikonal tiles are excluded from the
    eikonal pool, matching every other eikonal figure in this script."""
    pooled: dict[str, dict] = {}
    for model in ("bathtub", "eikonal"):
        n = sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
        hist_fine = hist_category = None
        for p in sorted(base_dir.glob(f"*/outputs/summary_{model}.json")):
            try:
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if model == "eikonal" and int(d["tile_id"]) in stale_eikonal_tile_ids:
                continue
            joint = d.get(f"{model}_depth_joint")
            if not joint:
                continue
            n += joint["n"]
            sum_x += joint["sum_x"]
            sum_y += joint["sum_y"]
            sum_x2 += joint["sum_x2"]
            sum_y2 += joint["sum_y2"]
            sum_xy += joint["sum_xy"]
            hf = np.asarray(joint["hist_fine"])
            hc = np.asarray(joint["hist_category"])
            hist_fine = hf if hist_fine is None else hist_fine + hf
            hist_category = hc if hist_category is None else hist_category + hc
        pooled[model] = {
            "n": n, "sum_x": sum_x, "sum_y": sum_y, "sum_x2": sum_x2, "sum_y2": sum_y2, "sum_xy": sum_xy,
            "hist_fine": hist_fine, "hist_category": hist_category,
        }
    return pooled


def _category_bin_labels(edges: np.ndarray) -> list[str]:
    labels = [f"<{edges[0]:.1f}"]
    labels += [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:.1f}")
    return labels


def plot_depth_correlation(pooled: dict, out_path: Path) -> None:
    """Pooled cell-level depth density (fine 0.1m bins) for bathtub/eikonal
    vs SFINCS, annotated with the pooled Pearson r - computed once over
    every mutually-wet cell across the whole batch (n in the thousands to
    millions), not averaged from per-tile medians like plot_depth_scatter."""
    lo, hi = float(DEPTH_CORR_FINE_EDGES[0]), float(DEPTH_CORR_FINE_EDGES[-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, model, label in [(axes[0], "bathtub", "bathtub"), (axes[1], "eikonal", "eikonal")]:
        joint = pooled[model]
        if joint["hist_fine"] is None or joint["n"] == 0:
            ax.set_title(f"{label}: no data")
            ax.axis("off")
            continue
        r = pearson_r_from_sums(joint["n"], joint["sum_x"], joint["sum_y"], joint["sum_x2"], joint["sum_y2"], joint["sum_xy"])
        # hist_fine's outer catch-all rows/cols (<0m, >3m) hold ~0 counts here
        # (both inputs are already wet-thresholded, rarely exceed 3m) - drop
        # them so the plotted grid matches DEPTH_CORR_FINE_EDGES' own extent.
        core = joint["hist_fine"][1:-1, 1:-1]
        im = ax.imshow(np.log1p(core.T), origin="lower", extent=[lo, hi, lo, hi],
                        aspect="auto", cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="white", linestyle=":", linewidth=1)
        ax.set_xlabel("SFINCS depth (m)")
        ax.set_ylabel(f"{label} depth (m)")
        ax.set_title(f"{label} vs SFINCS (n={int(joint['n']):,}, r={r:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log(1 + cell count)")

    fig.suptitle("Cell-level depth agreement against SFINCS, pooled across all tiles\n"
                  "(every mutually-flooded cell, dotted = 1:1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_depth_category_alignment(pooled: dict, out_path: Path) -> None:
    """Depth-bin confusion-matrix heatmap (0.1-1.5m/0.2m-wide bins, plus
    <0.1m/>1.5m catch-alls): for cells in a given SFINCS depth bin (rows),
    what fraction land in each model depth bin (columns) - row-normalized so
    each row sums to 100%, answering "when SFINCS says this depth range,
    what does the model say" directly."""
    labels = _category_bin_labels(DEPTH_CATEGORY_EDGES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, model, label in [(axes[0], "bathtub", "bathtub"), (axes[1], "eikonal", "eikonal")]:
        joint = pooled[model]
        hist = joint["hist_category"]
        if hist is None or joint["n"] == 0:
            ax.set_title(f"{label}: no data")
            ax.axis("off")
            continue
        row_sums = hist.sum(axis=1, keepdims=True)
        pct = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0) * 100
        im = ax.imshow(pct, origin="upper", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"{label} depth bin (m)")
        ax.set_ylabel("SFINCS depth bin (m)")
        ax.set_title(f"{label} vs SFINCS (n={int(joint['n']):,})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if pct[i, j] >= 1:
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if pct[i, j] > 50 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% of SFINCS-bin cells")

    fig.suptitle("Depth-bin alignment against SFINCS, pooled across all tiles\n"
                  "(row-normalized: each SFINCS depth-bin row sums to 100%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _jaccard(matched, model_only, sfincs_only):
    denom = matched + model_only + sfincs_only
    return np.where(denom > 0, matched / denom, np.nan)


def plot_extent_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["bathtub_km2"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.scatter(x[valid], y[valid], s=14, alpha=0.5, color=BATHTUB_COLOR, edgecolors="none")
    if valid.any():
        lim = [min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())]
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFINCS flooded area (km2)")
    ax.set_ylabel("bathtub flooded area (km2)")
    ax.set_title(f"Bathtub vs SFINCS (n={int(valid.sum())})")

    ax = axes[1]
    fresh = ~df["eikonal_stale"].fillna(False)
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["eikonal_km2"].to_numpy(dtype=float)
    valid_fresh = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & fresh.to_numpy()
    valid_stale = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & ~fresh.to_numpy()
    ax.scatter(x[valid_fresh], y[valid_fresh], s=14, alpha=0.5, color=EIKONAL_COLOR,
               edgecolors="none", label=f"current settings (n={int(valid_fresh.sum())})")
    if valid_stale.any():
        ax.scatter(x[valid_stale], y[valid_stale], s=14, alpha=0.6, color=STALE_COLOR,
                   edgecolors="none", label=f"stale (n={int(valid_stale.sum())})")
    all_valid = valid_fresh | valid_stale
    if all_valid.any():
        lim = [min(x[all_valid].min(), y[all_valid].min()), max(x[all_valid].max(), y[all_valid].max())]
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFINCS flooded area (km2)")
    ax.set_ylabel("eikonal flooded area (km2)")
    ax.set_title("Eikonal vs SFINCS")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Flooded extent agreement against SFINCS (log-log, dotted = 1:1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_agreement_hist(df: pd.DataFrame, out_path: Path) -> None:
    bathtub_jac = _jaccard(df["bathtub_matched_km2"], df["bathtub_only_km2"], df["bathtub_sfincs_only_km2"])
    eikonal_jac = _jaccard(df["eikonal_matched_km2"], df["eikonal_only_km2"], df["eikonal_sfincs_only_km2"])
    fresh = ~df["eikonal_stale"].fillna(False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 26)
    bt_valid = bathtub_jac[np.isfinite(bathtub_jac)]
    ek_valid = eikonal_jac[np.isfinite(eikonal_jac) & fresh.to_numpy()]
    ax.hist(bt_valid, bins=bins, alpha=0.6, color=BATHTUB_COLOR,
            label=f"bathtub (median={np.median(bt_valid):.2f}, n={len(bt_valid)})")
    ax.hist(ek_valid, bins=bins, alpha=0.6, color=EIKONAL_COLOR,
            label=f"eikonal, current settings (median={np.median(ek_valid):.2f}, n={len(ek_valid)})")
    ax.set_xlabel("extent agreement with SFINCS (Jaccard: matched / (matched + model-only + SFINCS-only))")
    ax.set_ylabel("number of tiles")
    ax.legend(fontsize=9)
    ax.set_title("Extent agreement with SFINCS - bathtub vs eikonal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"  bathtub Jaccard: median={np.median(bt_valid):.3f}, mean={np.mean(bt_valid):.3f} (n={len(bt_valid)})")
    print(f"  eikonal Jaccard (current settings): median={np.median(ek_valid):.3f}, mean={np.mean(ek_valid):.3f} (n={len(ek_valid)})")


def plot_depth_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fresh = ~df["eikonal_stale"].fillna(False)

    for ax, col, color, label in [
        (axes[0], "bathtub_depth_median_m", BATHTUB_COLOR, "bathtub"),
        (axes[1], "eikonal_depth_median_m", EIKONAL_COLOR, "eikonal"),
    ]:
        x = df["sfincs_depth_median_m"].to_numpy(dtype=float)
        y = df[col].to_numpy(dtype=float)
        mask = fresh.to_numpy() if col.startswith("eikonal") else np.ones(len(df), dtype=bool)
        valid = np.isfinite(x) & np.isfinite(y) & mask
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


def plot_agreement_map(df: pd.DataFrame, out_path: Path) -> None:
    fresh = ~df["eikonal_stale"].fillna(False)
    jac = _jaccard(df["eikonal_matched_km2"], df["eikonal_only_km2"], df["eikonal_sfincs_only_km2"])
    plot_df = df.assign(jac=jac)
    plot_df = plot_df[np.isfinite(plot_df["jac"]) & fresh & plot_df["lon"].notna()]

    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor=OCEAN_COLOR)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    sc = ax.scatter(
        plot_df["lon"], plot_df["lat"], transform=ccrs.PlateCarree(),
        c=plot_df["jac"], cmap="RdYlGn", vmin=0, vmax=1, s=22, alpha=0.9,
        linewidths=0.3, edgecolors="white", zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("eikonal-SFINCS extent agreement (Jaccard)")
    ax.set_title(f"Eikonal-SFINCS agreement by tile (n={len(plot_df)}, current-settings tiles only)", fontsize=12)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=OCEAN_COLOR)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v4 - created by select_validation_tiles.py")
    parser.add_argument("--stale-cutoff", default=None, help="optional timestamp (e.g. '2026-09-24 12:00:00'); "
                         "any tile whose eikonal raster predates it is flagged 'stale' and excluded from "
                         "eikonal summary statistics/maps, shown separately in the extent scatter instead")
    args = parser.parse_args()
    stale_cutoff = pd.Timestamp(args.stale_cutoff) if args.stale_cutoff else None

    base_dir = DATA_ROOT / args.base_dir_name
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = collect_summaries(base_dir, stale_cutoff)
    n_stale = int(df["eikonal_stale"].sum())
    n_has_eikonal = int(df["has_eikonal"].sum())
    print(f"{len(df)} tile summaries loaded ({n_has_eikonal} with eikonal output, {n_stale} stale)")
    df.to_csv(fig_dir / "validation_summary.csv", index=False)
    print(f"Wrote {fig_dir / 'validation_summary.csv'}")

    plot_extent_scatter(df, fig_dir / "extent_scatter.png")
    plot_agreement_hist(df, fig_dir / "extent_agreement_hist.png")
    plot_depth_scatter(df, fig_dir / "depth_scatter.png")
    plot_agreement_map(df, fig_dir / "agreement_map.png")

    stale_tile_ids = set(df.loc[df["eikonal_stale"], "tile_id"].tolist())
    pooled_depth = collect_depth_joint(base_dir, stale_tile_ids)
    plot_depth_correlation(pooled_depth, fig_dir / "depth_correlation.png")
    plot_depth_category_alignment(pooled_depth, fig_dir / "depth_category_alignment.png")


if __name__ == "__main__":
    main()
