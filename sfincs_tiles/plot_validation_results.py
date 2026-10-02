"""Result visualizations for a SFINCS-vs-HC-bathtub-vs-EA-bathtub validation
batch. HC-bathtub = hydraulically-connected bathtub (elevation threshold +
ocean-connectivity pruning, see flood_agreement.prune_to_ocean_connected).
EA-bathtub = Eikonal-attenuated bathtub (the friction/propagation-aware
eikonal solve). Both read from summary_bathtub.json/summary_eikonal.json -
DISPLAY_LABEL below sets the display names only.
Reads every tile's outputs/summary.json (postprocess_tile_summary.py /
postprocess_sequential.py) plus tile_selection_metadata.csv for lon/lat.

--stale-cutoff flags any tile whose EA-bathtub raster predates a given
timestamp as "stale" (e.g. carried over from an earlier batch via a
recovery copy). Stale tiles are shown distinctly, never silently pooled
with the rest. Omit for a freshly-run batch.

Usage:
    python plot_validation_results.py --base-dir-name validation_sfincs_v4
    python plot_validation_results.py --base-dir-name validation_sfincs_v4 --stale-cutoff "2026-09-24 12:00:00"
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

from compute_metrics_overview_table import build_and_write_table
from flood_agreement import (
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, depth_error_metrics_from_pooled, metrics_from_counts,
    pool_depth_joint,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import LAND_COLOR, WATER_COLOR as OCEAN_COLOR, draw_caption_box  # noqa: E402

DATA_ROOT = Path(r"P:\11212688-004-global-floodmaps\modelling")

COAST_COLOR = "#b8b8b3"
HC_BATHTUB_COLOR = "#8a8a86"
EA_BATHTUB_COLOR = "#2a78d6"
STALE_COLOR = "#d6572a"

# Display names only - JSON field prefixes stay "bathtub"/"eikonal".
DISPLAY_LABEL = {"bathtub": "HC-bathtub", "eikonal": "EA-bathtub"}


def collect_summaries(base_dir: Path, stale_cutoff: pd.Timestamp | None) -> pd.DataFrame:
    """Merges each tile's separate summary_{bathtub,eikonal,sfincs}.json
    (postprocess_tile_summary.py) into one row per tile. A tile missing one
    or more of these files contributes no columns for that model."""
    by_tile: dict[str, dict] = {}
    for model in ("bathtub", "eikonal", "sfincs"):
        for p in sorted(base_dir.glob(f"*/outputs/summary_{model}.json")):
            try:
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # {model}_depth_joint is a nested dict, not a scalar - pooled
            # separately by collect_depth_joint() for the depth plots.
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
    # Ensures every column the plotting functions below expect exists, even
    # for a partially-complete batch.
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
    """{model: pool_depth_joint(...)} for bathtub/eikonal, excluding stale
    eikonal tiles from the eikonal pool."""
    return {
        model: pool_depth_joint(base_dir, model, stale_eikonal_tile_ids if model == "eikonal" else None)
        for model in ("bathtub", "eikonal")
    }


def _category_bin_labels(edges: np.ndarray) -> list[str]:
    labels = [f"<{edges[0]:.1f}"]
    labels += [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:.1f}")
    return labels


def plot_depth_correlation(pooled: dict, out_path: Path) -> None:
    """Pooled cell-level depth density (fine 0.1m bins) for bathtub/eikonal
    vs SFINCS, annotated with pooled r/bias/RMSE (see
    flood_agreement.depth_error_metrics_from_pooled). Median error/% within
    band are printed to console and written to metrics_overview_table.csv
    rather than crowding the title."""
    lo, hi = float(DEPTH_CORR_FINE_EDGES[0]), float(DEPTH_CORR_FINE_EDGES[-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, model, label in [(axes[0], "bathtub", DISPLAY_LABEL["bathtub"]), (axes[1], "eikonal", DISPLAY_LABEL["eikonal"])]:
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
    """Depth-bin confusion-matrix heatmap (0.1-1.5m/0.2m-wide bins, plus a
    >1.5m catch-all): for cells in a given SFINCS depth bin (rows), what
    fraction land in each model depth bin (columns), row-normalized to
    100%. No <0.1m catch-all bin - both axes are wet-thresholded at
    WET_THRESHOLD_M=0.10m, so that bin is always empty."""
    labels = _category_bin_labels(DEPTH_CATEGORY_EDGES)[1:]  # drop the <0.1m catch-all label
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    fig.subplots_adjust(left=0.08, right=0.85, top=0.90, bottom=0.22, wspace=0.12)
    im = None
    for ax, model, tag in [(axes[0], "eikonal", "a"), (axes[1], "bathtub", "b")]:
        joint = pooled[model]
        hist = joint["hist_category"]
        ax.text(0.0, 1.03, f"{tag})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        if hist is None or joint["n"] == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        hist = hist[1:, 1:]  # drop the <0.1m catch-all row/col
        row_sums = hist.sum(axis=1, keepdims=True)
        pct = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0) * 100
        im = ax.imshow(pct, origin="upper", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"{DISPLAY_LABEL[model]} depth bin (m)")
        ax.set_ylabel("SFINCS depth bin (m)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if pct[i, j] >= 1:
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if pct[i, j] > 50 else "black")

    cax = fig.add_axes((0.87, 0.22, 0.02, 0.68))
    fig.colorbar(im, cax=cax, label="% of SFINCS-bin cells")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _csi(matched, model_only, sfincs_only):
    """Critical Success Index / Threat Score - matched / (matched +
    model_only + sfincs_only). Equivalent to the Jaccard index (intersection
    over union); matches flood_agreement.metrics_from_counts's "CSI" key."""
    denom = matched + model_only + sfincs_only
    return np.where(denom > 0, matched / denom, np.nan)


def plot_extent_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["bathtub_km2"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.scatter(x[valid], y[valid], s=14, alpha=0.5, color=HC_BATHTUB_COLOR, edgecolors="none")
    if valid.any():
        lim = [min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())]
        ax.plot(lim, lim, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFINCS flooded area (km2)")
    ax.set_ylabel(f"{DISPLAY_LABEL['bathtub']} flooded area (km2)")
    ax.set_title(f"{DISPLAY_LABEL['bathtub']} vs SFINCS (n={int(valid.sum())})")

    ax = axes[1]
    fresh = ~df["eikonal_stale"].fillna(False)
    x = df["sfincs_km2"].to_numpy(dtype=float)
    y = df["eikonal_km2"].to_numpy(dtype=float)
    valid_fresh = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & fresh.to_numpy()
    valid_stale = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & ~fresh.to_numpy()
    ax.scatter(x[valid_fresh], y[valid_fresh], s=14, alpha=0.5, color=EA_BATHTUB_COLOR,
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
    ax.set_ylabel(f"{DISPLAY_LABEL['eikonal']} flooded area (km2)")
    ax.set_title(f"{DISPLAY_LABEL['eikonal']} vs SFINCS")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Flooded extent agreement against SFINCS (log-log, dotted = 1:1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_agreement_hist(df: pd.DataFrame, out_path: Path) -> None:
    bathtub_csi = _csi(df["bathtub_matched_km2"], df["bathtub_only_km2"], df["bathtub_sfincs_only_km2"])
    eikonal_csi = _csi(df["eikonal_matched_km2"], df["eikonal_only_km2"], df["eikonal_sfincs_only_km2"])
    fresh = ~df["eikonal_stale"].fillna(False)

    bt_mask = np.isfinite(bathtub_csi)
    ek_mask = np.isfinite(eikonal_csi) & fresh.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 26)
    bt_valid = bathtub_csi[bt_mask]
    ek_valid = eikonal_csi[ek_mask]
    ax.hist(bt_valid, bins=bins, alpha=0.6, color=HC_BATHTUB_COLOR,
            label=f"{DISPLAY_LABEL['bathtub']} (median={np.median(bt_valid):.2f}, n={len(bt_valid)})")
    ax.hist(ek_valid, bins=bins, alpha=0.6, color=EA_BATHTUB_COLOR,
            label=f"{DISPLAY_LABEL['eikonal']}, current settings (median={np.median(ek_valid):.2f}, n={len(ek_valid)})")
    ax.set_xlabel("extent agreement with SFINCS (CSI: matched / (matched + model-only + SFINCS-only))")
    ax.set_ylabel("number of tiles")
    ax.legend(fontsize=9)
    ax.set_title(f"Extent agreement with SFINCS - {DISPLAY_LABEL['bathtub']} vs {DISPLAY_LABEL['eikonal']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Per-tile mean/median treat every tile equally regardless of its own
    # flooded area ("typical tile" stat). The pooled CSI below (counts
    # summed across tiles first) is the area-weighted counterpart, reported
    # alongside rather than replacing the per-tile view.
    for prefix, label, mask, valid in (
        ("bathtub", DISPLAY_LABEL["bathtub"], bt_mask, bt_valid),
        ("eikonal", f"{DISPLAY_LABEL['eikonal']} (current settings)", ek_mask, ek_valid),
    ):
        pooled = metrics_from_counts(
            df[f"{prefix}_matched_km2"][mask].sum(),
            df[f"{prefix}_only_km2"][mask].sum(),
            df[f"{prefix}_sfincs_only_km2"][mask].sum(),
        )
        print(f"  {label} CSI: per-tile median={np.median(valid):.3f}, per-tile mean={np.mean(valid):.3f} "
              f"(n={len(valid)}, unweighted) | area-weighted (pooled) mean={pooled['CSI']:.3f}")


def plot_depth_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fresh = ~df["eikonal_stale"].fillna(False)

    for ax, col, color, label in [
        (axes[0], "bathtub_depth_median_m", HC_BATHTUB_COLOR, DISPLAY_LABEL["bathtub"]),
        (axes[1], "eikonal_depth_median_m", EA_BATHTUB_COLOR, DISPLAY_LABEL["eikonal"]),
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


def _agreement_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shared prep for the two agreement-map figures below: per-tile CSI +
    misalignment (1 - CSI), restricted to fresh, geolocated, finite-CSI
    tiles."""
    fresh = ~df["eikonal_stale"].fillna(False)
    csi = _csi(df["eikonal_matched_km2"], df["eikonal_only_km2"], df["eikonal_sfincs_only_km2"])
    plot_df = df.assign(csi=csi, misalignment=1 - csi)
    return plot_df[np.isfinite(plot_df["csi"]) & fresh & plot_df["lon"].notna()]


MIN_FLOODED_KM2_FOR_COLOR = 1.0  # below this union area, a tile's CSI is
# too noise-dominated to color - shown as a grey marker instead.


def plot_tile_agreement_map(df: pd.DataFrame, out_path: Path) -> None:
    """Global map of per-tile EA-bathtub-SFINCS extent agreement (CSI).
    Tiles with at least MIN_FLOODED_KM2_FOR_COLOR km2 of union area
    (matched + eikonal-only + SFINCS-only, CSI's own denominator) get the
    CSI color scale; every other fresh, geolocated tile is plotted as a
    flat grey marker.
    """
    fresh = ~df["eikonal_stale"].fillna(False)
    csi = _csi(df["eikonal_matched_km2"], df["eikonal_only_km2"], df["eikonal_sfincs_only_km2"])
    union_km2 = df["eikonal_matched_km2"] + df["eikonal_only_km2"] + df["eikonal_sfincs_only_km2"]
    plot_df = df.assign(csi=csi, union_km2=union_km2)
    plot_df = plot_df[fresh & plot_df["lon"].notna()]

    has_real_flooding = plot_df["union_km2"].fillna(0) >= MIN_FLOODED_KM2_FOR_COLOR
    colored_df = plot_df[has_real_flooding & np.isfinite(plot_df["csi"])]
    grey_df = plot_df[~(has_real_flooding & np.isfinite(plot_df["csi"]))]

    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor=OCEAN_COLOR)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    ax.scatter(
        grey_df["lon"], grey_df["lat"], transform=ccrs.PlateCarree(),
        c="#999990", s=18, alpha=0.7, linewidths=0.3, edgecolors="white", zorder=2,
        label=f"< {MIN_FLOODED_KM2_FOR_COLOR:g} km2 union area (n={len(grey_df)})",
    )
    sc = ax.scatter(
        colored_df["lon"], colored_df["lat"], transform=ccrs.PlateCarree(),
        c=colored_df["csi"], cmap="RdYlGn", vmin=0, vmax=1, s=22, alpha=0.9,
        linewidths=0.3, edgecolors="white", zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"{DISPLAY_LABEL['eikonal']}-SFINCS extent agreement (CSI)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    draw_caption_box(ax, [
        f"n={len(colored_df)} colored by CSI",
        f"n={len(grey_df)} below {MIN_FLOODED_KM2_FOR_COLOR:g} km2 shown grey",
        "current-settings tiles only",
    ])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=OCEAN_COLOR)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_agreement_vs_tile_size(df: pd.DataFrame, out_path: Path) -> None:
    """Per-tile misalignment (1 - CSI) against domain area (log scale).
    Pearson r on log10(area); Spearman rho alongside as a rank-based check."""
    plot_df = _agreement_df(df)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    size = plot_df["domain_area_km2"].to_numpy(dtype=float)
    mis = plot_df["misalignment"].to_numpy(dtype=float)
    valid = np.isfinite(size) & (size > 0) & np.isfinite(mis)
    ax.scatter(size[valid], mis[valid], s=18, alpha=0.5, color=EA_BATHTUB_COLOR, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("tile domain area (km2, log)")
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir-name", required=True, help="output root directory name under paths.root, "
                         "e.g. validation_sfincs_v4 - created by select_validation_tiles.py")
    parser.add_argument("--stale-cutoff", default=None, help="optional timestamp (e.g. '2026-09-24 12:00:00'); "
                         "any tile whose EA-bathtub raster predates it is flagged 'stale' and excluded from "
                         "EA-bathtub summary statistics/maps")
    args = parser.parse_args()
    stale_cutoff = pd.Timestamp(args.stale_cutoff) if args.stale_cutoff else None

    base_dir = DATA_ROOT / args.base_dir_name
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = collect_summaries(base_dir, stale_cutoff)
    n_stale = int(df["eikonal_stale"].sum())
    n_has_eikonal = int(df["has_eikonal"].sum())
    print(f"{len(df)} tile summaries loaded ({n_has_eikonal} with {DISPLAY_LABEL['eikonal']} output, {n_stale} stale)")
    df.to_csv(fig_dir / "validation_summary.csv", index=False)
    print(f"Wrote {fig_dir / 'validation_summary.csv'}")

    plot_extent_scatter(df, fig_dir / "extent_scatter.png")
    plot_agreement_hist(df, fig_dir / "extent_agreement_hist.png")
    plot_depth_scatter(df, fig_dir / "depth_scatter.png")
    plot_tile_agreement_map(df, fig_dir / "agreement_map.png")
    plot_agreement_vs_tile_size(df, fig_dir / "agreement_vs_tile_size.png")

    stale_tile_ids = set(df.loc[df["eikonal_stale"], "tile_id"].tolist())
    pooled_depth = collect_depth_joint(base_dir, stale_tile_ids)
    plot_depth_correlation(pooled_depth, fig_dir / "depth_correlation.png")
    plot_depth_category_alignment(pooled_depth, fig_dir / "depth_category_alignment.png")

    print()
    build_and_write_table(base_dir)


if __name__ == "__main__":
    main()
