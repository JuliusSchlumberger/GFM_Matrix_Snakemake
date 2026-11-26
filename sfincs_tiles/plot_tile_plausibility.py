"""Plot per-tile CSI/HR/FAR distributions and the per-tile drivers of disagreement with SFINCS.

Needs variant_runs_per_tile.csv and variant_selection.json (compute_variant_metrics.py) and
tile_drivers.csv (compute_tile_drivers.py, in --raster-base-dir-name when given).

Usage:
    python plot_tile_plausibility.py --base-dir-name calibration_roughness_SFINCS
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import DISPLAY_LABEL  # noqa: E402
from compute_tile_drivers import DRIVERS_NAME  # noqa: E402
from flood_agreement import metrics_from_counts  # noqa: E402
from gfm_config import read_figures_root, read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _figure_style import DOUBLE_COLUMN_MM, MM_PER_INCH, MUTED, apply_style, panel_letters, save_figure, write_caption  # noqa: E402

FIG_WIDTH_IN = DOUBLE_COLUMN_MM / MM_PER_INCH
BASE_VARIANTS = ("bathtub", "bathtub_connected", "straight_line", "eikonal_uniform", "eikonal_landcover")
REFERENCE_VARIANT = "bathtub_connected"
AREA_OFFSET_KM2 = 0.1  # keeps log2(model / SFINCS area) finite for a tile one side leaves dry
# (column, axis label, log x axis, lower clip for the log axis)
BATHTUB_DRIVERS = (
    ("river_pct", "River and channel share of tile (%)", True, 0.01),
    ("reach_km", "Connected-bathtub reach (km of land per km coast)", True, 0.001),
    ("friction_m_per_km", "Land-cover attenuation in bathtub zone (m/km)", False, None),
    ("surge_range_m", "Forcing hydrograph range (m)", False, None),
    ("peak_level_spread_m", "Peak-level spread along tile (m)", False, None),
)
EIKONAL_DRIVERS = (
    ("overtop_coast_pct", "Coastline below peak level (%)", False, None),
    ("river_pct", "River and channel share of tile (%)", True, 0.01),
    ("lake_pct", "Lake share of tile (%)", True, 0.01),
    ("friction_m_per_km", "Land-cover attenuation in bathtub zone (m/km)", False, None),
    ("storage_to_inflow_ratio", "Bathtub storage / storm inflow", True, 1e-4),
    ("abs_lat_deg", "Absolute latitude (°)", False, None),
)
_DRIVER_DEFINITIONS = {
    "river_pct": "river and channel share = tile cells masked as river",
    "reach_km": "reach = land below the tile's median boundary peak level connected to the open sea (bathtub "
                "zone), per km of coastline",
    "friction_m_per_km": "attenuation = median unscaled land-cover friction over the bathtub zone",
    "surge_range_m": "hydrograph range = median over boundary points of the SFINCS forcing's max - min",
    "peak_level_spread_m": "peak-level spread = max - min boundary peak level along the tile",
    "overtop_coast_pct": "coastline below peak level = share of coastline cells lower than the median boundary "
                         "peak level",
    "lake_pct": "lake share = tile cells masked as lake",
    "storage_to_inflow_ratio": "storage / inflow = water volume of the bathtub zone at the peak level over the "
                               "volume a broad-crested weir (q = 1.7 h^1.5) along the overtoppable coastline "
                               "passes during the median SFINCS forcing hydrograph; the inflow is an upper bound, "
                               "so the ratio ranks tiles rather than predicting whether the storm fills them",
    "abs_lat_deg": "absolute latitude of the tile centre",
}
METRICS = (("CSI", "CSI"), ("HT", "Hit rate"), ("FAR", "False alarm ratio"))


def _run_label(entry: dict, variant: str) -> str:
    extra = []
    if entry.get("parameter") is not None:
        extra.append(f"{'fsf' if 'landcover' in variant else 'rate'} {entry['parameter']:g}")
    if entry.get("alpha") is not None:
        extra.append(f"alpha {entry['alpha']:g}")
    return DISPLAY_LABEL[variant] + (f" ({', '.join(extra)})" if extra else "")


def _short_label(variant: str) -> str:
    return (DISPLAY_LABEL[variant].replace(", uniform rate", " uniform").replace(", land-cover rate", " land-cover")
            .replace(" + volume cap", " + cap").replace("Straight-line attenuation", "Straight line"))


def _scored(per_tile: pd.DataFrame, key: str, min_union_km2: float) -> pd.DataFrame:
    g = per_tile[per_tile["run_key"] == key]
    union = g["matched_km2"] + g["model_only_km2"] + g["sfincs_only_km2"]
    return g[union >= min_union_km2]


def plot_metric_distributions(per_tile: pd.DataFrame, runs: dict[str, str], labels: dict[str, str],
                              min_union_km2: float, out_path: Path) -> None:
    """Plot per-tile CSI, hit rate and FAR per run as box plots, with the pooled value."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, max(2.2, 0.35 * len(runs) + 1.0)), sharey=True, layout="constrained")
    order = list(runs)[::-1]  # first run on top
    for letter, ax, (metric, label) in zip("abc", axes, METRICS):
        for y, variant in enumerate(order):
            g = _scored(per_tile, runs[variant], min_union_km2)
            vals = g[metric].dropna().to_numpy()
            if len(vals):
                ax.boxplot(vals, positions=[y], vert=False, widths=0.55, whis=(5, 95), showfliers=False,
                           medianprops={"color": "black"}, boxprops={"linewidth": 0.6},
                           whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
            pooled = metrics_from_counts(g["matched_km2"].sum(), g["model_only_km2"].sum(), g["sfincs_only_km2"].sum())
            ax.plot(pooled[metric], y, marker="D", color=MUTED, markersize=4, linestyle="none")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel(label)
        panel_letters(ax, letter, fontsize=8)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([labels[v] for v in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    save_figure(fig, out_path)
    plt.close(fig)
    n = len(_scored(per_tile, runs[order[0]], min_union_km2))
    write_caption(out_path, [
        "Per-tile agreement with SFINCS: attenuating flooding lowers the false alarm ratio of every tile, at a "
        "small cost in hit rate.",
        f"Box: interquartile range, line: median, whiskers: 5-95 % across tiles where model or SFINCS floods at "
        f"least {min_union_km2:g} km2 (n = {n} for the top row). Grey diamond: value pooled over the same tiles "
        "(area counts summed before scoring).",
        "(a) CSI = matched / (matched + model only + SFINCS only); (b) hit rate = matched / SFINCS flooded; "
        "(c) false alarm ratio = model only / model flooded. Each run is its variant's best by pooled CSI on "
        "the calibration tiles.",
    ])
    print(f"Wrote {out_path}")


def _log_bias(g: pd.DataFrame) -> pd.Series:
    model = g["matched_km2"] + g["model_only_km2"]
    sfincs = g["matched_km2"] + g["sfincs_only_km2"]
    return np.log2((model + AREA_OFFSET_KM2) / (sfincs + AREA_OFFSET_KM2))


def _rank_partial(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    r = df[[x, y, *controls]].rank()
    z = np.column_stack([np.ones(len(r)), r[controls].to_numpy()])
    rx = r[x] - z @ np.linalg.lstsq(z, r[x], rcond=None)[0]
    ry = r[y] - z @ np.linalg.lstsq(z, r[y], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def driver_correlations(per_tile: pd.DataFrame, drivers: pd.DataFrame, runs: dict[str, str],
                        min_union_km2: float, driver_set: tuple = BATHTUB_DRIVERS,
                        ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return (per run: tiles with drivers and log2 area bias, Spearman and partial Spearman table).

    The partial rho of a driver controls for the other drivers of `driver_set`.
    """
    cols = [d[0] for d in driver_set]
    data, rows = {}, []
    for variant, key in runs.items():
        g = _scored(per_tile, key, min_union_km2)
        df = g.assign(log2_bias=_log_bias(g)).merge(drivers, on="tile_id").dropna(subset=[*cols, "log2_bias"])
        data[variant] = df
        for c in cols:
            rows.append({"variant": variant, "run_key": key, "driver": c, "n_tiles": len(df),
                         "spearman_rho": spearmanr(df[c], df["log2_bias"])[0],
                         "partial_spearman_rho": _rank_partial(df, c, "log2_bias", [o for o in cols if o != c])})
    return data, pd.DataFrame(rows)


def plot_driver_scatter(data: dict[str, pd.DataFrame], table: pd.DataFrame, labels: dict[str, str],
                        out_path: Path, driver_set: tuple = BATHTUB_DRIVERS, message: str = "") -> None:
    """Plot log2(model / SFINCS flooded area) against each driver.

    Several runs: one row per driver, one column per run. One run: the drivers in two columns.
    """
    variants = list(data)
    single = len(variants) == 1
    if single:
        panels = [(d, variants[0]) for d in driver_set]
        n_rows, n_cols = -(-len(panels) // 2), 2
    else:
        panels = [(d, v) for d in driver_set for v in variants]
        n_rows, n_cols = len(driver_set), len(variants)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN, 1.6 * n_rows + 0.3),
                             sharey=True, squeeze=False, layout="constrained")
    all_y = pd.concat([d["log2_bias"] for d in data.values()])
    ylim = (min(-1.0, all_y.quantile(0.01)), all_y.quantile(0.99))
    for k, ((col, xlabel, logx, clip), variant) in enumerate(panels):
        i, j = divmod(k, n_cols)
        df = data[variant]
        ax = axes[i, j]
        x = df[col].clip(lower=clip) if logx else df[col]
        x_all = pd.concat([d[col] for d in data.values()])
        ax.scatter(x, df["log2_bias"], s=3, alpha=0.25, color="black", edgecolors="none", rasterized=True)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(edges) > 2:
            b = pd.cut(x, edges, include_lowest=True)
            q = df.groupby(b, observed=True)["log2_bias"].quantile([0.25, 0.5, 0.75]).unstack()
            mid = x.groupby(b, observed=True).median()
            ax.fill_between(mid, q[0.25], q[0.75], color="#2a78d6", alpha=0.25, linewidth=0)
            ax.plot(mid, q[0.5], color="#2a78d6", linewidth=1.2)
        ax.axhline(0, color=MUTED, linewidth=0.6, linestyle=":")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(clip, x_all.max() * 1.2)
        else:
            ax.set_xlim(x_all.min(), x_all.quantile(0.99))
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.tick_params(labelsize=7)
        rho = table[(table.variant == variant) & (table.driver == col)].iloc[0]
        stats = f"ρ {rho.spearman_rho:+.2f}, partial {rho.partial_spearman_rho:+.2f}"
        header = f"{_short_label(variant)}: {stats}" if (i == 0 and not single) else stats
        panel_letters(ax, "abcdefghijklmnopqrstuvwxyz"[k], fontsize=7, label=header)
        if j == 0:
            ax.set_ylabel("log2(model / SFINCS area)", fontsize=7)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    save_figure(fig, out_path)
    plt.close(fig)
    n = {v: len(d) for v, d in data.items()}
    runs_text = "; ".join(f"{labels[v]} (n = {n[v]})" for v in data)
    write_caption(out_path, [
        message,
        "Each dot is one tile: log2 of the model's flooded area over SFINCS's (0 = equal area, +1 = twice SFINCS; "
        f"{AREA_OFFSET_KM2:g} km2 added to both); line and band: median and interquartile range in ten "
        "equal-count bins of the driver. " + (f"Run: {runs_text}." if single else f"Columns: {runs_text}."),
        "Panel header: Spearman rho with the log2 area ratio, and the partial rho controlling for the other "
        f"{len(driver_set) - 1} drivers (rank regression). Linear x axes end at the 99th percentile; log axes "
        "clip zeros to their lower limit.",
        "Drivers: " + "; ".join(_DRIVER_DEFINITIONS[d[0]] for d in driver_set) + ".",
    ])
    print(f"Wrote {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--raster-base-dir-name", default=None, help=f"base dir holding {DRIVERS_NAME}")
    parser.add_argument("--min-union-km2", type=float, default=1.0,
                        help="score only tiles where the run or SFINCS floods at least this area")
    args = parser.parse_args()
    apply_style()

    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    fig_dir = read_figures_root(Path(args.config)) / "sfincs_runs" / args.base_dir_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((base / SELECTION_NAME).read_text())
    entries = selection["variants"]
    shown = [v for v in BASE_VARIANTS if v in entries]
    if selection.get("best_variant") and selection["best_variant"] not in shown:
        shown.append(selection["best_variant"])
    runs = {v: entries[v]["run_key"] for v in shown}
    labels = {v: _run_label(entries[v], v) for v in shown}
    per_tile = pd.read_csv(base / "variant_runs_per_tile.csv")

    plot_metric_distributions(per_tile, runs, labels, args.min_union_km2, fig_dir / "metric_distributions.png")

    drivers = pd.read_csv(root / (args.raster_base_dir_name or args.base_dir_name) / DRIVERS_NAME)
    drivers["river_pct"] = drivers["river_frac"] * 100.0
    best = selection["best_variant"]
    pair = {v: runs[v] for v in (REFERENCE_VARIANT, best) if v in runs}
    data, table = driver_correlations(per_tile, drivers, pair, args.min_union_km2, BATHTUB_DRIVERS)
    plot_driver_scatter(data, table, labels, fig_dir / "disagreement_drivers.png", BATHTUB_DRIVERS,
                        "Where SFINCS floods less than the static models: tiles with many river channels and a "
                        "long connected bathtub reach inland.")
    best_data, best_table = driver_correlations(per_tile, drivers, {best: runs[best]}, args.min_union_km2,
                                                EIKONAL_DRIVERS)
    plot_driver_scatter(best_data, best_table, labels, fig_dir / "disagreement_drivers_eikonal.png",
                        EIKONAL_DRIVERS,
                        "What is left for the calibrated eikonal: it floods more than SFINCS where water enters "
                        "through few gaps in the coast, along channels and over rough land, and less at high "
                        "latitudes.")
    table = pd.concat([table.assign(driver_set="bathtub"), best_table.assign(driver_set="eikonal")],
                      ignore_index=True)
    table_path = fig_dir / "disagreement_drivers.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")
    with pd.option_context("display.float_format", "{:+.2f}".format, "display.width", 160):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
