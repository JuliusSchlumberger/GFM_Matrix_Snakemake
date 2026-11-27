"""Plot the plausibility of the SFINCS calibration per tile: metrics, drivers, per-tile optimum.

Writes metric_distributions, sweep_{CSI,HT,FAR,bias}, disagreement_drivers (bathtub set), disagreement_drivers_eikonal,
variant_contrasts, per_tile_optimum_gain and per_tile_optimum_drivers (.png + caption)
and their tables. Needs
variant_runs_per_tile.csv and variant_selection.json (compute_variant_metrics.py) and
tile_drivers[_owned].csv (compute_tile_drivers.py, in --raster-base-dir-name when given).

Usage:
    python plot_tile_plausibility.py --base-dir-name calibration_roughness_SFINCS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import DISPLAY_LABEL  # noqa: E402
from compute_tile_drivers import DRIVERS_NAME, DRIVERS_OWNED_NAME, read_drivers  # noqa: E402
from flood_agreement import metrics_from_counts  # noqa: E402
from gfm_config import read_figures_root, read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM, GRID, INK, MM_PER_INCH, MUTED, apply_style, panel_letters, save_figure, write_caption,
)

FIG_WIDTH_IN = DOUBLE_COLUMN_MM / MM_PER_INCH
BASE_VARIANTS = ("bathtub", "bathtub_connected", "straight_line", "eikonal_uniform", "eikonal_landcover")
REFERENCE_VARIANT = "bathtub_connected"
AREA_OFFSET_KM2 = 0.1  # keeps log2(model / SFINCS area) finite for a tile one side leaves dry
# (column, axis label, log x axis, lower clip for the log axis)
BATHTUB_DRIVERS = (
    ("river_pct", "River and channel share of tile (%)", True, 0.01),
    ("reach_km", "Connected-bathtub reach (km of land per km coast)", True, 0.001),
    ("friction_m_per_km", "Land-cover attenuation in bathtub zone (m/km)", False, None),
    ("surge_range_m", "Forcing range (m)", False, None),
    ("peak_level_spread_m", "Peak-level spread along tile (m)", False, None),
)
EIKONAL_DRIVERS = (
    ("overtop_coast_pct", "Coastline below peak level (%)", False, None),
    ("river_pct", "River and channel share of tile (%)", True, 0.01),
    ("lake_pct", "Lake share of tile (%)", True, 0.01),
    ("friction_m_per_km", "Land-cover attenuation in bathtub zone (m/km)", False, None),
    ("surge_range_m", "Forcing range (m)", False, None),
    ("enclosed_pct", "Bathtub zone behind a sill (%)", False, None),
    ("peak_hold_h", "Unbroken time in top 20 % of storm-tide range (h)", True, 1.0),
)
_DRIVER_DEFINITIONS = {
    "river_pct": "river and channel share = tile cells masked as river",
    "reach_km": "reach = land below the tile's median boundary peak level connected to the open sea (bathtub "
                "zone), per km of coastline",
    "friction_m_per_km": "attenuation = median unscaled land-cover friction over the bathtub zone",
    "surge_range_m": "forcing range = median over boundary points of the SFINCS forcing hydrograph's max - min "
                     "(tide plus surge)",
    "peak_level_spread_m": "peak-level spread = max - min boundary peak level along the tile",
    "overtop_coast_pct": "coastline below peak level = share of coastline cells lower than the median boundary "
                         "peak level",
    "lake_pct": "lake share = tile cells masked as lake",
    "enclosed_pct": "behind a sill = share of the bathtub zone whose spill level (lowest level connecting it to "
                    "the open sea, priority flood) is at least 0.2 m above its ground: land behind a dike, sea "
                    "bank or ridge",
    "peak_hold_h": "unbroken time in top 20 % of storm-tide range = median over boundary points of the longest "
                   "stretch the SFINCS forcing hydrograph stays above min + 0.8 (max - min) (separate high tides "
                   "do not add up)",
}
METRICS = (("CSI", "CSI"), ("HT", "Hit rate"), ("FAR", "False alarm ratio"))
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
# (variant without cap, panel header, x label); the "_cap" twin shares the panel
SWEEP_COLUMNS = (
    ("straight_line", "Straight line", "Attenuation rate (m/km)"),
    ("eikonal_uniform", "Eikonal, uniform rate", "Attenuation rate (m/km)"),
    ("eikonal_landcover", "Eikonal, land-cover rate", "Friction scale factor"),
)
SWEEP_METRICS = ("CSI", "HT", "FAR", "bias")
SWEEP_METRIC_LABELS = {"CSI": "CSI", "HT": "Hit rate", "FAR": "False alarm ratio", "bias": "Bias (log2 axis)"}
SWEEP_METRIC_DEFINITIONS = {
    "CSI": "CSI = matched / (matched + model only + SFINCS only).",
    "HT": "Hit rate = matched / SFINCS flooded.",
    "FAR": "False alarm ratio = model only / model flooded.",
    "bias": "Bias = model flooded / SFINCS flooded; dotted line: no bias.",
}
SWEEP_STYLE = {False: ("#E69F00", False), True: ("#0072B2", True)}  # cap -> (colour, filled box)
MIN_UNION_PEOPLE = 100.0  # per-tile exposure metrics swing between 0 and 1 on a handful of people
OPTIMUM_VARIANTS = ("eikonal_uniform_cap", "eikonal_landcover_cap")
OPTIMUM_COLORS = {"eikonal_uniform_cap": "#0072B2", "eikonal_landcover_cap": "#CC79A7"}
OPTIMUM_LINESTYLES = {"eikonal_uniform_cap": "-", "eikonal_landcover_cap": "--"}
TIE_CSI = 0.005  # parameters within this of a tile's best CSI count as equally good
SAME_CSI = 0.01  # a tile prefers another value only if it gains at least this much CSI
CLASS_COLORS = {"lower": "#56B4E9", "same": MUTED, "higher": "#D55E00"}
CLASS_LABELS = {"lower": "Lower value better", "same": "Global value as good", "higher": "Higher value better"}
# (label, variant compared, baseline variant), each at its selected run
CONTRASTS = (
    ("Land-cover vs uniform rate, both capped", "eikonal_landcover_cap", "eikonal_uniform_cap"),
    ("Volume cap vs none, uniform rate", "eikonal_uniform_cap", "eikonal_uniform"),
    ("Volume cap vs none, land-cover rate", "eikonal_landcover_cap", "eikonal_landcover"),
)
CONTRASTS_SHOWN = ("Land-cover vs uniform rate, both capped", "Volume cap vs none, uniform rate")  # the cap behaves the same at the land-cover rate
CONTRAST_ORDER = ("worse", "same", "better")
COUNT_COLUMNS = ("matched_km2", "model_only_km2", "sfincs_only_km2")
CONTRAST_COLORS = {"better": "#0072B2", "same": MUTED, "worse": "#D55E00"}
CONTRAST_LABELS = {"better": "Gains at least 0.01 CSI", "same": "Within 0.01 CSI", "worse": "Loses at least 0.01 CSI"}
# three columns leave no room for the long driver labels
SHORT_DRIVER_LABELS = {
    "overtop_coast_pct": "Coast below peak (%)", "river_pct": "River share (%)", "lake_pct": "Lake share (%)",
    "friction_m_per_km": "Land-cover attenuation (m/km)", "surge_range_m": "Forcing range (m)",
    "enclosed_pct": "Zone behind a sill (%)", "peak_hold_h": "Peak hold (h)",
}


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
        if metric == "HT":  # hit rates sit near 1; the 5 % whiskers stay above 0.5
            ax.set_xlim(0.5, 1)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        else:
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
        "(c) false alarm ratio = model only / model flooded. Each run is its variant's selected run "
        "(variant_selection.json), chosen by population-weighted CSI over all tiles.",
    ])
    print(f"Wrote {out_path}")


def _metrics_from_sums(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"CSI": tp / (tp + fp + fn), "HT": tp / (tp + fn), "FAR": fp / (tp + fp), "bias": (tp + fp) / (tp + fn)}


def sweep_table(per_tile: pd.DataFrame, min_union_km2: float, n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Return per swept run, weight and metric: per-tile quantiles and the pooled value with a 90 % interval.

    Per-tile quantiles over tiles where the run or SFINCS floods at least `min_union_km2`
    (area) or MIN_UNION_PEOPLE people; per-tile bias only where SFINCS floods. Pooled over all
    tiles, as the run selection; p5/p95 over `n_boot` tile bootstrap resamples, the same
    resamples for every run.
    """
    swept = per_tile[per_tile["variant"].isin([v for col in SWEEP_COLUMNS for v in (col[0], f"{col[0]}_cap")])]
    tiles = np.sort(per_tile["tile_id"].unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    freq = rng.multinomial(len(tiles), np.full(len(tiles), 1 / len(tiles)), size=n_boot).astype(float)
    rows = []
    for (variant, parameter, key), g in swept.groupby(["variant", "parameter", "run_key"]):
        g = g.set_index("tile_id").reindex(tiles, fill_value=0.0)
        for weight, unit, threshold in (("area", "km2", min_union_km2), ("pop", "pop", MIN_UNION_PEOPLE)):
            counts = g[[f"matched_{unit}", f"model_only_{unit}", f"sfincs_only_{unit}"]].to_numpy(dtype=float)
            per = _metrics_from_sums(*counts.T)
            pooled = _metrics_from_sums(*counts.sum(axis=0))
            boot = _metrics_from_sums(*(freq @ counts).T)
            keep = counts.sum(axis=1) >= threshold
            for metric in SWEEP_METRICS:
                vals = per[metric][keep]
                vals = vals[np.isfinite(vals)]
                q = np.percentile(vals, [5, 25, 50, 75, 95]) if len(vals) else np.full(5, np.nan)
                rows.append({"variant": variant, "parameter": parameter, "run_key": key, "weight": weight,
                             "metric": metric, "n_tiles": len(vals),
                             **{f"tile_p{p}": v for p, v in zip((5, 25, 50, 75, 95), q)},
                             "pooled": float(pooled[metric]), "pooled_p5": np.nanpercentile(boot[metric], 5),
                             "pooled_p95": np.nanpercentile(boot[metric], 95)})
    return pd.DataFrame(rows)


def plot_sweep_metric(table: pd.DataFrame, metric: str, selected: dict[str, float], min_union_km2: float,
                      out_path: Path) -> None:
    """Plot one metric over each swept model's parameter values, flood extent and exposure."""
    fig, axes = plt.subplots(2, len(SWEEP_COLUMNS), figsize=(FIG_WIDTH_IN, 4.4), sharey=True, squeeze=False,
                             layout="constrained")
    dodge = 0.2
    for j, (base, title, xlabel) in enumerate(SWEEP_COLUMNS):
        params = sorted(table.loc[table["variant"] == base, "parameter"].unique())
        pos = {p: i for i, p in enumerate(params)}
        for i, (weight, ylabel) in enumerate((("area", "flooded area"), ("pop", "people"))):
            ax = axes[i, j]
            if base in selected and selected[base] in pos:
                ax.axvspan(pos[selected[base]] - 0.45, pos[selected[base]] + 0.45, color=GRID, linewidth=0, zorder=0)
            for variant, shift in ((base, -dodge), (f"{base}_cap", dodge)):
                color, filled = SWEEP_STYLE[variant.endswith("_cap")]
                t = table[(table["variant"] == variant) & (table["weight"] == weight) & (table["metric"] == metric)]
                for r in t.itertuples():
                    x = pos[r.parameter] + shift
                    ax.bxp([{"med": r.tile_p50, "q1": r.tile_p25, "q3": r.tile_p75, "whislo": r.tile_p5,
                             "whishi": r.tile_p95, "fliers": []}], positions=[x], widths=0.32, patch_artist=True,
                           boxprops={"facecolor": color if filled else "white", "edgecolor": color, "linewidth": 0.6,
                                     "alpha": 0.9},
                           medianprops={"color": "black" if filled else color, "linewidth": 0.9},
                           whiskerprops={"color": color, "linewidth": 0.6}, capprops={"color": color, "linewidth": 0.6})
                    ax.errorbar(x, r.pooled, yerr=[[r.pooled - r.pooled_p5], [r.pooled_p95 - r.pooled]], fmt="D",
                                color=INK, markersize=2.6, elinewidth=0.8, capsize=0, zorder=4)
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels([f"{p:g}" if k % 2 == 0 else "" for k, p in enumerate(params)])
            ax.set_xlim(-0.6, len(params) - 0.4)
            if metric == "bias":
                ax.set_yscale("log", base=2)
                ax.set_ylim(0.25, 8)
                ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.axhline(1, color=MUTED, linewidth=0.6, linestyle=":", zorder=1)
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            if j == 0:
                ax.set_ylabel(f"{SWEEP_METRIC_LABELS[metric]}, {ylabel}")
            if i == 1:
                ax.set_xlabel(xlabel)
            panel_letters(ax, "abcdef"[i * len(SWEEP_COLUMNS) + j], fontsize=8, label=title if i == 0 else None)
    handles = [matplotlib.patches.Patch(facecolor=SWEEP_STYLE[cap][0] if SWEEP_STYLE[cap][1] else "white",
                                        edgecolor=SWEEP_STYLE[cap][0], label=label)
               for cap, label in ((False, "No volume cap"), (True, "Volume cap"))]
    handles.append(plt.Line2D([], [], color=INK, marker="D", markersize=2.6, linewidth=0.8,
                              label="Pooled over all tiles, 90 % interval"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    save_figure(fig, out_path)
    plt.close(fig)
    t = table[table["metric"] == metric]
    n = {w: (int(t.loc[t["weight"] == w, "n_tiles"].min()), int(t.loc[t["weight"] == w, "n_tiles"].max()))
         for w in ("area", "pop")}
    sel = ", ".join(f"{title.lower()} {selected[base]:g}" for base, title, _ in SWEEP_COLUMNS if base in selected)
    write_caption(out_path, [
        f"{SWEEP_METRIC_LABELS[metric]} of every swept run against SFINCS, per tile and pooled, for flooded area "
        "and for the people in it.",
        "Box: interquartile range, line: median, whiskers: 5-95 % across tiles; open boxes without, filled boxes "
        f"with the volume cap. Area row: tiles where the run or SFINCS floods at least {min_union_km2:g} "
        f"km2 (n = {n['area'][0]}-{n['area'][1]} per run); people row: at least {MIN_UNION_PEOPLE:g} people "
        f"(n = {n['pop'][0]}-{n['pop'][1]})" + ("; bias only where SFINCS floods" if metric == "bias" else "") + ".",
        "Black diamond: value pooled over all tiles (counts summed before scoring, as for the run selection); "
        f"bar: 90 % interval over {N_BOOTSTRAP} tile bootstrap resamples. Grey band: the selected value "
        f"({sel}), chosen by population-weighted CSI over all tiles.",
        SWEEP_METRIC_DEFINITIONS[metric] + " People: WorldPop counts in the cells.",
    ])
    print(f"Wrote {out_path}")


def _log_bias(g: pd.DataFrame) -> pd.Series:
    model = g["matched_km2"] + g["model_only_km2"]
    sfincs = g["matched_km2"] + g["sfincs_only_km2"]
    return np.log2((model + AREA_OFFSET_KM2) / (sfincs + AREA_OFFSET_KM2))


def _rank_rhos(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # ranks: (n, k + 1), the k drivers then the response. Returns (Spearman, partial Spearman) per driver.
    k = ranks.shape[1] - 1
    y = ranks[:, -1]
    rho, partial = np.empty(k), np.empty(k)
    for i in range(k):
        x = ranks[:, i]
        rho[i] = np.corrcoef(x, y)[0, 1]
        z = np.column_stack([np.ones(len(ranks)), np.delete(ranks[:, :k], i, axis=1)])
        rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        partial[i] = np.corrcoef(rx, ry)[0, 1]
    return rho, partial


def driver_correlations(per_tile: pd.DataFrame, drivers: pd.DataFrame, runs: dict[str, str],
                        min_union_km2: float, driver_set: tuple = BATHTUB_DRIVERS,
                        n_boot: int = N_BOOTSTRAP) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return (per run: tiles with drivers and log2 area bias, Spearman and partial Spearman table).

    The partial rho of a driver controls for the other drivers of `driver_set` (rank
    regression). p5/p95: 90 % interval over `n_boot` tile bootstrap resamples (fixed seed).
    """
    cols = [d[0] for d in driver_set]
    data, rows = {}, []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for variant, key in runs.items():
        g = _scored(per_tile, key, min_union_km2)
        df = g.assign(log2_bias=_log_bias(g)).merge(drivers, on="tile_id").dropna(subset=[*cols, "log2_bias"])
        data[variant] = df
        values = df[[*cols, "log2_bias"]].to_numpy(dtype=float)
        rho, partial = _rank_rhos(pd.DataFrame(values).rank().to_numpy())
        boot = [_rank_rhos(pd.DataFrame(values[rng.integers(0, len(values), len(values))]).rank().to_numpy())
                for _ in range(n_boot)]
        boot_rho = np.array([b[0] for b in boot])
        boot_partial = np.array([b[1] for b in boot])
        for i, c in enumerate(cols):
            rows.append({"variant": variant, "run_key": key, "driver": c, "n_tiles": len(df),
                         "spearman_rho": rho[i], "spearman_rho_p5": np.percentile(boot_rho[:, i], 5),
                         "spearman_rho_p95": np.percentile(boot_rho[:, i], 95),
                         "partial_spearman_rho": partial[i],
                         "partial_spearman_rho_p5": np.percentile(boot_partial[:, i], 5),
                         "partial_spearman_rho_p95": np.percentile(boot_partial[:, i], 95)})
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
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
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
        f"{len(driver_set) - 1} drivers (rank regression); 90 % tile-bootstrap intervals in "
        "disagreement_drivers.csv. Linear x axes end at the 99th percentile; log axes clip zeros to their lower "
        "limit.",
        "Drivers: " + "; ".join(_DRIVER_DEFINITIONS[d[0]] for d in driver_set) + ".",
    ])
    print(f"Wrote {out_path}")


def per_tile_optimum(per_tile: pd.DataFrame, variant: str, global_value: float, min_union_km2: float) -> pd.DataFrame:
    """Return each tile's best parameter of `variant` against `global_value`, with its CSI gain.

    Best: highest per-tile CSI; values within TIE_CSI of it are ties, settled towards the global
    value. cls: "lower"/"higher" when the best value gains at least SAME_CSI, else "same".
    """
    t = _scored_all_runs(per_tile, variant, min_union_km2)
    grid = np.sort(t["parameter"].unique())
    wide = t.pivot(index="tile_id", columns="parameter", values="CSI")
    best_csi = wide.max(axis=1)
    near = wide.ge(best_csi - TIE_CSI, axis=0).to_numpy()
    dist = np.abs(np.log(grid / global_value))
    best = grid[np.argmin(np.where(near, dist[None, :], np.inf), axis=1)]
    gain = (best_csi - wide[global_value]).to_numpy()
    cls = np.where(gain < SAME_CSI, "same", np.where(best < global_value, "lower", "higher"))
    groups = t.drop_duplicates("tile_id").set_index("tile_id")["group"]
    return pd.DataFrame({"tile_id": wide.index, "group": groups.reindex(wide.index).to_numpy(), "variant": variant,
                         "global_value": global_value, "best_value": best, "csi_global": wide[global_value].to_numpy(),
                         "csi_best": best_csi.to_numpy(), "gain": gain, "cls": cls,
                         "at_grid_edge": np.isin(best, [grid[0], grid[-1]])})


def _scored_all_runs(per_tile: pd.DataFrame, variant: str, min_union_km2: float) -> pd.DataFrame:
    # Tiles where any run of the variant or SFINCS floods min_union_km2, with every run of the variant.
    t = per_tile[per_tile["variant"] == variant]
    union = (t["matched_km2"] + t["model_only_km2"] + t["sfincs_only_km2"]).groupby(t["tile_id"]).transform("max")
    return t[union >= min_union_km2]


def optimum_summary(per_tile: pd.DataFrame, optimum: pd.DataFrame) -> pd.DataFrame:
    """Return pooled CSI per variant and group at the global value and at every tile's own best value."""
    rows = []
    for variant, o in optimum.groupby("variant", sort=False):
        t = per_tile[per_tile["variant"] == variant]
        for group, og in o.groupby("group"):
            at_global = t.merge(og[["tile_id", "global_value"]], left_on=["tile_id", "parameter"],
                                right_on=["tile_id", "global_value"])
            at_best = t.merge(og[["tile_id", "best_value"]], left_on=["tile_id", "parameter"],
                              right_on=["tile_id", "best_value"])
            m_g = metrics_from_counts(*at_global[["matched_km2", "model_only_km2", "sfincs_only_km2"]].sum())
            m_b = metrics_from_counts(*at_best[["matched_km2", "model_only_km2", "sfincs_only_km2"]].sum())
            rows.append({"variant": variant, "group": group, "global_value": og["global_value"].iloc[0],
                         "n_tiles": len(og), "CSI_global": m_g["CSI"], "CSI_per_tile_best": m_b["CSI"],
                         "bias_global": m_g["bias"], "bias_per_tile_best": m_b["bias"]})
    return pd.DataFrame(rows)


def plot_optimum_gain(optimum: pd.DataFrame, summary: pd.DataFrame, per_tile: pd.DataFrame, out_path: Path) -> None:
    """Plot how much CSI each tile gains at its own best value, and where those values lie."""
    variants = list(dict.fromkeys(optimum["variant"]))
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    for v in variants:
        gain = np.sort(optimum.loc[optimum["variant"] == v, "gain"].to_numpy())
        ax.plot(gain, np.arange(1, gain.size + 1) / gain.size, color=OPTIMUM_COLORS[v], linestyle=OPTIMUM_LINESTYLES[v],
                linewidth=1.2, label=_short_label(v))
    ax.axvline(SAME_CSI, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set(xlim=(0, 0.3), ylim=(0, 1), xlabel="CSI gain of the tile's own best value", ylabel="Share of tiles")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    panel_letters(ax, "a", fontsize=8)
    for k, v in enumerate(variants[:2]):
        o = optimum[optimum["variant"] == v]
        g = o["global_value"].iloc[0]
        grid = np.sort(per_tile.loc[per_tile["variant"] == v, "parameter"].unique())
        share = o["best_value"].value_counts(normalize=True).reindex(grid, fill_value=0.0)
        ax = fig.add_subplot(gs[0, 1] if k == 0 else gs[1, 0])
        x = np.arange(grid.size)
        ax.bar(x, share.to_numpy(), width=0.7, color=[OPTIMUM_COLORS[v] if p == g else MUTED for p in grid])
        ax.set_xticks(x, [f"{p:g}" for p in grid], fontsize=7)
        ax.set(xlabel=f"Tile's best {'fsf' if 'landcover' in v else 'rate (m/km)'}", ylabel="Share of tiles")
        panel_letters(ax, "bc"[k], fontsize=8, label=_short_label(v))
    ax = fig.add_subplot(gs[1, 1])
    labels, y = [], 0
    for v in variants:
        for group in ("calibration", "validation"):
            s = summary[(summary["variant"] == v) & (summary["group"] == group)]
            if s.empty:
                continue
            s = s.iloc[0]
            ax.plot([s.CSI_global, s.CSI_per_tile_best], [y, y], color=OPTIMUM_COLORS[v], linewidth=1.0)
            ax.plot(s.CSI_global, y, marker="o", color=OPTIMUM_COLORS[v], markersize=4, linestyle="none")
            ax.plot(s.CSI_per_tile_best, y, marker="D", color=OPTIMUM_COLORS[v], markerfacecolor="white",
                    markersize=4, linestyle="none")
            labels.append(f"{_short_label(v)}, {group}")
            y += 1
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled CSI")
    panel_letters(ax, "d", fontsize=8)
    save_figure(fig, out_path)
    plt.close(fig)
    globals_text = ", ".join(f"{_short_label(v)} {optimum.loc[optimum['variant'] == v, 'global_value'].iloc[0]:g}"
                             for v in variants)
    write_caption(out_path, [
        "How much a per-tile attenuation value could gain over one global value: the upper bound for any "
        "rule that sets the value from tile characteristics.",
        f"Global value: the selected run of each variant ({globals_text}; variant_selection.json, chosen by "
        f"population-weighted CSI over all tiles). A tile's best value: its highest CSI; values within {TIE_CSI} CSI of that maximum count as ties "
        "and the one closest to the global value is taken. Tiles where any run or SFINCS floods at least 1 km2.",
        f"(a) Cumulative share of tiles by the CSI gain of their own best value (dotted line: {SAME_CSI}). "
        "(b, c) Share of tiles by their best value; coloured bar: the global value; the end bars include tiles "
        "whose optimum lies beyond the tested range. (d) Pooled CSI at the global value (filled circle) and with "
        "every tile at its own best value (open diamond), per tile group.",
    ])
    print(f"Wrote {out_path}")


def plot_optimum_drivers(optimum: pd.DataFrame, drivers: pd.DataFrame, out_path: Path,
                         driver_set: tuple = EIKONAL_DRIVERS) -> pd.DataFrame:
    """Plot the drivers of tiles whose best value is lower, the same or higher; return the rho table."""
    variants = list(dict.fromkeys(optimum["variant"]))
    fig, axes = plt.subplots(len(driver_set), len(variants), figsize=(FIG_WIDTH_IN, 1.05 * len(driver_set) + 0.5),
                             sharey=True, squeeze=False, layout="constrained")
    order = ("higher", "same", "lower")
    rows = []
    for j, v in enumerate(variants):
        dd = optimum[optimum["variant"] == v].merge(drivers, on="tile_id")
        dd = dd.assign(log2_ratio=np.log2(dd["best_value"] / dd["global_value"]))
        for i, (col, xlabel, logx, clip) in enumerate(driver_set):
            ax = axes[i, j]
            for y, c in enumerate(order):
                vals = dd.loc[dd["cls"] == c, col].replace([np.inf, -np.inf], np.nan).dropna()
                if logx:
                    vals = vals.clip(lower=clip)
                if len(vals):
                    ax.boxplot(vals, positions=[y], vert=False, widths=0.6, whis=(5, 95), showfliers=False,
                               patch_artist=True, medianprops={"color": "black", "linewidth": 0.8},
                               boxprops={"facecolor": CLASS_COLORS[c], "linewidth": 0.5},
                               whiskerprops={"linewidth": 0.5}, capprops={"linewidth": 0.5})
            if logx:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ok = dd[[col, "log2_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = spearmanr(ok[col], ok["log2_ratio"])[0] if len(ok) > 2 else np.nan
            rows.append({"variant": v, "driver": col, "n_tiles": len(ok), "spearman_rho_vs_log2_best_over_global": rho})
            ax.set_xlabel(xlabel, fontsize=7)
            ax.tick_params(labelsize=7)
            panel_letters(ax, "abcdefghijklmnopqrstuvwxyz"[i * len(variants) + j], fontsize=7,
                          label=f"{_short_label(v)}: ρ {rho:+.2f}" if i == 0 else f"ρ {rho:+.2f}")
        axes[0, j].set_yticks(range(len(order)), [CLASS_LABELS[c] for c in order], fontsize=7)
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        "Which tiles want a different attenuation value than the global one, and in which direction.",
        f"Tiles grouped by their own best value relative to the global one: lower or higher when that value gains "
        f"at least {SAME_CSI} CSI, otherwise 'global value as good'. Box: interquartile range, line: median, "
        "whiskers: 5-95 %. Panel header: Spearman rho between the driver and log2(best value / global value) "
        "over all tiles of the column (negative: a lower value is better as the driver grows).",
        "Drivers: " + "; ".join(_DRIVER_DEFINITIONS[d[0]] for d in driver_set) + ". Log axes clip zeros to "
        "their lower limit.",
    ])
    print(f"Wrote {out_path}")
    return pd.DataFrame(rows)


def contrast_table(per_tile: pd.DataFrame, runs: dict[str, str], min_union_km2: float) -> pd.DataFrame:
    """Per tile, the CSI change of each CONTRASTS pair (compared minus baseline) and its class.

    Only tiles where both runs flood at least `min_union_km2` together with SFINCS; a change
    smaller than SAME_CSI counts as "same". Adds the SFINCS flooded area.
    """
    parts = []
    for label, compared, baseline in CONTRASTS:
        if compared not in runs or baseline not in runs:
            continue
        a = _scored(per_tile, runs[compared], min_union_km2)
        b = _scored(per_tile, runs[baseline], min_union_km2)
        b = b.assign(sfincs_km2=b["matched_km2"] + b["sfincs_only_km2"])
        m = a[["tile_id", "CSI"]].merge(b[["tile_id", "CSI", "sfincs_km2"]], on="tile_id",
                                        suffixes=("_compared", "_baseline")).dropna()
        d = m["CSI_compared"] - m["CSI_baseline"]
        cls = np.where(d >= SAME_CSI, "better", np.where(d <= -SAME_CSI, "worse", "same"))
        parts.append(m.assign(contrast=label, compared=runs[compared], baseline=runs[baseline], d_csi=d, cls=cls))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def contrast_driver_rhos(contrasts: pd.DataFrame, drivers: pd.DataFrame, driver_set: tuple = EIKONAL_DRIVERS,
                         n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Spearman rho of each driver with each contrast's CSI change; p5/p95 over tile bootstrap resamples."""
    cols = [d[0] for d in driver_set]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for label, g in contrasts.groupby("contrast", sort=False):
        df = g.merge(drivers, on="tile_id").dropna(subset=[*cols, "d_csi"])
        values = df[[*cols, "d_csi"]].to_numpy(dtype=float)
        ranks = pd.DataFrame(values).rank().to_numpy()
        rho = np.array([np.corrcoef(ranks[:, i], ranks[:, -1])[0, 1] for i in range(len(cols))])
        boot = []
        for _ in range(n_boot):
            r = pd.DataFrame(values[rng.integers(0, len(values), len(values))]).rank().to_numpy()
            boot.append([np.corrcoef(r[:, i], r[:, -1])[0, 1] for i in range(len(cols))])
        boot = np.array(boot)
        for i, c in enumerate(cols):
            rows.append({"contrast": label, "driver": c, "n_tiles": len(df), "spearman_rho_vs_d_csi": rho[i],
                         "p5": np.nanpercentile(boot[:, i], 5), "p95": np.nanpercentile(boot[:, i], 95)})
    return pd.DataFrame(rows)


def _pooled_csi(per_tile: pd.DataFrame, key: str, tiles) -> float:
    s = per_tile[(per_tile["run_key"] == key) & per_tile["tile_id"].isin(tiles)][list(COUNT_COLUMNS)].sum()
    return float(s["matched_km2"] / s.sum()) if s.sum() > 0 else np.nan


def plot_variant_contrasts(contrasts: pd.DataFrame, per_tile: pd.DataFrame, rhos: pd.DataFrame, out_path: Path,
                           shown: tuple[str, ...] = CONTRASTS_SHOWN) -> None:
    """Plot where land cover beats a uniform rate and where the volume cap helps or hurts."""
    labels = [c for c in shown if c in set(contrasts["contrast"])]
    fig, axes = plt.subplots(2, len(labels), figsize=(FIG_WIDTH_IN, 5.0), squeeze=False, layout="constrained",
                             gridspec_kw={"height_ratios": [1.2, 0.9]})
    size_edges = (0, 5, 20, 100, 500, np.inf)
    size_labels = ("< 5", "5-20", "20-100", "100-500", "> 500")
    lines = []
    for j, label in enumerate(labels):
        g = contrasts[contrasts["contrast"] == label]
        compared, baseline = g["compared"].iloc[0], g["baseline"].iloc[0]

        ax = axes[1, j]  # where: share of tiles by SFINCS flooded area, pooled change per class
        size = pd.cut(g["sfincs_km2"], size_edges, labels=size_labels, right=False)
        for y, sl in enumerate(size_labels):
            h = g[size == sl]
            if h.empty:
                continue
            left = 0.0
            for c in CONTRAST_ORDER:
                w = float(np.mean(h["cls"] == c)) * 100
                ax.barh(y, w, left=left, color=CONTRAST_COLORS[c], height=0.7)
                left += w
            pooled = _pooled_csi(per_tile, compared, h["tile_id"]) - _pooled_csi(per_tile, baseline, h["tile_id"])
            ax.text(102, y, f"{pooled:+.3f}", va="center", fontsize=6.5)
        ax.text(102, len(size_labels) - 0.45, "Pooled", fontsize=6.5, va="bottom")
        ax.set_yticks(range(len(size_labels)), size_labels)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of tiles (%)")
        ax.set_ylabel("SFINCS flooded area (km2)" if j == 0 else "")

        ax = axes[0, j]  # which tile properties go with a gain
        r = rhos[rhos["contrast"] == label].reset_index(drop=True)
        ys = np.arange(len(r))[::-1]
        est = r["spearman_rho_vs_d_csi"]
        ax.errorbar(est, ys, xerr=[est - r["p5"], r["p95"] - est], fmt="o", color="black", markersize=3,
                    elinewidth=0.8, capsize=0)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_yticks(ys, [SHORT_DRIVER_LABELS.get(d, d) for d in r["driver"]] if j == 0 else [])
        ax.set_xlabel("Spearman rho with CSI change")

        share = g["cls"].value_counts(normalize=True).reindex(CONTRAST_ORDER, fill_value=0.0) * 100
        area = g.groupby("cls")["sfincs_km2"].sum().reindex(CONTRAST_ORDER, fill_value=0.0) / g["sfincs_km2"].sum() * 100
        pooled_all = _pooled_csi(per_tile, compared, g["tile_id"]) - _pooled_csi(per_tile, baseline, g["tile_id"])
        lines.append(f"{label} ({compared} vs {baseline}, n = {len(g)} tiles): better on {share['better']:.0f} % of "
                     f"tiles ({area['better']:.0f} % of SFINCS flooded area), worse on {share['worse']:.0f} % "
                     f"({area['worse']:.0f} %); pooled CSI change {pooled_all:+.3f}")
        for i in range(2):
            panel_letters(axes[i, j], "abcd"[i * len(labels) + j], fontsize=8, label=label if i == 0 else None)
    handles = [matplotlib.patches.Patch(color=CONTRAST_COLORS[c], label=CONTRAST_LABELS[c])
               for c in reversed(CONTRAST_ORDER)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    save_figure(fig, out_path)
    plt.close(fig)
    extra = []
    for label in [c for c in dict.fromkeys(contrasts["contrast"]) if c not in labels]:
        share = (contrasts.loc[contrasts["contrast"] == label, "cls"].value_counts(normalize=True)
                 .reindex(CONTRAST_ORDER, fill_value=0.0) * 100)
        extra.append(f"{label} (not drawn): better on {share['better']:.0f} %, worse on {share['worse']:.0f} % of tiles")
    write_caption(out_path, [
        "The volume cap leaves most of the flooded area unchanged: it corrects the few tiles that flood far too much "
        "and costs a little on many small tiles that were already unbiased. Land cover instead of a uniform rate "
        "changes most tiles, with gains and losses that nearly cancel.",
        f"Per tile, CSI of the first run in the panel header minus CSI of the second (the baseline), each at its "
        f"selected run. Colour: the first run gains or loses at least {SAME_CSI} CSI (counted 'better' / 'worse' "
        "below), or stays within it. Tiles where both runs score.",
        "(a, b) Spearman rho between each tile property and the CSI change; bar: 90 % interval over "
        f"{N_BOOTSTRAP} tile bootstrap resamples. The properties are correlated with each other, so a rho marks "
        "which tiles gain, not why. (c, d) Share of tiles per class by SFINCS flooded area; number: CSI change "
        "pooled over the tiles of the size class (area counts summed before scoring).",
        "; ".join(lines + extra) + ".",
        "Tile properties: " + "; ".join(_DRIVER_DEFINITIONS[d[0]] for d in EIKONAL_DRIVERS) + ".",
    ])
    print(f"Wrote {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--raster-base-dir-name", default=None, help="base dir holding the driver files")
    parser.add_argument("--drivers-name", default=DRIVERS_NAME, choices=[DRIVERS_NAME, DRIVERS_OWNED_NAME],
                        help=f"{DRIVERS_OWNED_NAME} for results scored on the owned area (dedup)")
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
    sweep = sweep_table(per_tile, args.min_union_km2)
    if not sweep.empty:
        sweep.to_csv(fig_dir / "sweep_metrics.csv", index=False)
        selected = {base: float(entries[base]["parameter"]) for base, _, _ in SWEEP_COLUMNS
                    if entries.get(base, {}).get("parameter") is not None}
        for metric in SWEEP_METRICS:
            plot_sweep_metric(sweep, metric, selected, args.min_union_km2, fig_dir / f"sweep_{metric}.png")

    drivers = read_drivers(root / (args.raster_base_dir_name or args.base_dir_name), args.drivers_name)
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
                        "What is left for the calibrated eikonal: it floods more than SFINCS where the storm tide "
                        "rises and falls quickly (large forcing range), where water enters through few gaps in the "
                        "coast or fills land behind sills, and less where the storm tide holds its peak for long.")
    table = pd.concat([table.assign(driver_set="bathtub"), best_table.assign(driver_set="eikonal")],
                      ignore_index=True)
    table_path = fig_dir / "disagreement_drivers.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")
    with pd.option_context("display.float_format", "{:+.2f}".format, "display.width", 160):
        print(table.to_string(index=False))
    cols = [d[0] for d in EIKONAL_DRIVERS]
    matrix = best_data[best].reindex(columns=cols).corr(method="spearman")
    matrix.to_csv(fig_dir / "driver_correlation_matrix.csv")
    print(f"Wrote {fig_dir / 'driver_correlation_matrix.csv'}")

    contrasts = contrast_table(per_tile, {v: e["run_key"] for v, e in entries.items()}, args.min_union_km2)
    if not contrasts.empty:
        contrasts.to_csv(fig_dir / "variant_contrasts.csv", index=False)
        rho = contrast_driver_rhos(contrasts, drivers)
        rho.to_csv(fig_dir / "variant_contrast_drivers.csv", index=False)
        plot_variant_contrasts(contrasts, per_tile, rho, fig_dir / "variant_contrasts.png")
        print(contrasts.groupby("contrast")["cls"].value_counts(normalize=True).unstack().round(3).to_string())

    optimum = pd.concat([per_tile_optimum(per_tile, v, float(entries[v]["parameter"]), args.min_union_km2)
                         for v in OPTIMUM_VARIANTS
                         if v in entries and per_tile.loc[per_tile["variant"] == v, "parameter"].nunique() > 1],
                        ignore_index=True)
    if optimum.empty:
        print("no swept capped eikonal variant: no per-tile optimum figures")
        return
    summary = optimum_summary(per_tile, optimum)
    optimum.to_csv(fig_dir / "per_tile_optimum.csv", index=False)
    summary.to_csv(fig_dir / "per_tile_optimum_summary.csv", index=False)
    plot_optimum_gain(optimum, summary, per_tile, fig_dir / "per_tile_optimum_gain.png")
    rho = plot_optimum_drivers(optimum, drivers, fig_dir / "per_tile_optimum_drivers.png")
    rho.to_csv(fig_dir / "per_tile_optimum_drivers.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
