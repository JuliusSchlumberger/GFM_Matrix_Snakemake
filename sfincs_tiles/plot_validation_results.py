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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import GROUPS, load_results, metrics_overview, run_label, tile_table  # noqa: E402
from flood_agreement import metrics_from_counts  # noqa: E402
from gfm_config import read_figures_root, read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM, LAND_COLOR, MM_PER_INCH, SEQUENTIAL_CMAP, apply_style, save_figure,
    write_caption,
)

FIG_WIDTH_IN = DOUBLE_COLUMN_MM / MM_PER_INCH

COAST_COLOR = "#b8b8b3"
HC_BATHTUB_COLOR = "#8a8a86"
EA_BATHTUB_COLOR = "#2a78d6"
LABEL = {"ref": "reference", "model": "model"}  # set in main from the two runs


def _csi(matched, model_only, sfincs_only):
    # CSI = Jaccard index, same as flood_agreement.metrics_from_counts["CSI"]
    denom = matched + model_only + sfincs_only
    return np.where(denom > 0, matched / denom, np.nan)


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

    plot_agreement_hist(df, fig_dir / f"extent_agreement_hist{sfx}.png")
    plot_tile_agreement_map(df, fig_dir / f"agreement_map{sfx}.png")

    keys = [e["run_key"] for e in selection["variants"].values()]
    table = metrics_overview(results, keys if key in keys else [*keys, key])
    table_path = base_dir / f"metrics_overview_table{sfx}.csv"
    table.to_csv(table_path, index=False)
    print(f"\nWrote {table_path}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
