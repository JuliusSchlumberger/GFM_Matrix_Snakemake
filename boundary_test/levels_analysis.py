"""Step 1b/1d: compare the tidal levels of GESLA, GTSM and FES2022 and attribute the differences.

Comparisons (`tide_levels.parquet`, variant `no_seasonal` so GESLA's Sa/Ssa does not enter):
  gtsm_raw / gtsm_harm vs gesla_harm   GTSM against the gauge
  fes_gauge vs gesla_harm              FES2022 against the gauge, at the gauge position
  fes_station vs gtsm_harm             the two models at the same point (model difference only)
  fes_gauge vs fes_station             FES2022 between gauge and GTSM station (location only)
Per comparison: median difference and ratio, MAE, share within 0.1 m, per GESLA spring-range
class, per region and per gauge-to-station distance class. Constituent ratios (M2, S2, K1, O1,
M4) show which part of the tide carries a difference. Quality filter: >= 3 fitted years, fit
explains >= 50% of the variance, station within MAX_DIST_KM.

Writes <out-dir>/analysis/levels_*.csv and levels_*.png.

Usage:
    python boundary_test/levels_analysis.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEVELS = ("mhws_m", "mhw_m", "mlw_m", "mlws_m", "hat_m", "lat_m", "spring_range_m")
COMPARISONS = (("gtsm_raw", "gesla_harm"), ("gtsm_harm", "gesla_harm"), ("fes_gauge", "gesla_harm"),
               ("fes_station", "gtsm_harm"), ("fes_gauge", "fes_station"))
CONSTITUENTS = ("m2", "s2", "n2", "k1", "o1", "m4", "ms4")
RANGE_BINS = (0, 1, 2, 4, 8, 20)
DIST_BINS = (0, 1, 2, 5, 10, 100)
MAX_DIST_KM = 10.0
MIN_FIT_YEARS = 3.0
MIN_EXPLAINED = 0.5
REGIONS = (  # (name, lon_min, lon_max, lat_min, lat_max); first match wins
    ("Baltic", 9, 31, 53, 66), ("Mediterranean", -6, 37, 30, 46), ("NW Europe", -12, 31, 43, 72),
    ("N America Atlantic+Gulf", -100, -50, 15, 70), ("N America Pacific", -170, -100, 15, 72),
    ("Japan+E Asia", 100, 160, 0, 60), ("Australia+NZ", 110, 180, -50, -8),
    ("S America", -90, -30, -60, 15), ("Africa+Indian Ocean", -20, 100, -40, 30))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def region_of(lon: float, lat: float) -> str:
    for name, x0, x1, y0, y1 in REGIONS:
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return name
    return "other"


def wide_table(out: Path) -> pd.DataFrame:
    tl = pd.read_parquet(out / "tide_levels.parquet")
    tl = tl[(tl.variant == "no_seasonal") | (tl.source == "gtsm_raw")]
    wide = tl.pivot_table(index="file_name", columns="source", values=list(LEVELS))
    wide.columns = [f"{s}__{v}" for v, s in wide.columns]
    fit = tl[tl.source == "gesla_harm"].set_index("file_name")[["fit_years", "fit_explained", "fit_rms_m"]]
    pairs = pd.read_parquet(out / "pairs.parquet").set_index("file_name")
    t = pairs[["site_name", "country", "lon", "lat", "gtsm_dist_km"]].join(fit).join(wide)
    t["region"] = [region_of(x, y) for x, y in zip(t.lon, t.lat)]
    t["range_class"] = pd.cut(t["gesla_harm__spring_range_m"], RANGE_BINS)
    t["dist_class"] = pd.cut(t.gtsm_dist_km, DIST_BINS)
    t["quality"] = (t.fit_years >= MIN_FIT_YEARS) & (t.fit_explained >= MIN_EXPLAINED) & (t.gtsm_dist_km <= MAX_DIST_KM)
    return t


def constituent_table(out: Path, sites: pd.Index) -> pd.DataFrame:
    hc = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    names = list(hc["constituents"])
    rows = pd.DataFrame(index=pd.Index(hc["file_name"], name="file_name"))
    for source in ("gesla_harm", "gtsm_harm", "fes_gauge", "fes_station"):
        for c in CONSTITUENTS:
            rows[f"{source}__{c}"] = np.abs(hc[source][:, names.index(c)])
    return rows.loc[rows.index.intersection(sites)]


def summary(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    d, ratio = (a - b)[ok], (a / b)[ok & (b.abs() > 0.05)]
    return {"n": int(ok.sum()), "median_diff_m": d.median(), "p10_diff_m": d.quantile(0.1),
            "p90_diff_m": d.quantile(0.9), "mae_m": d.abs().mean(), "within_0.1m": (d.abs() <= 0.1).mean(),
            "median_ratio": ratio.median()}


def comparison_tables(t: pd.DataFrame, dest: Path) -> pd.DataFrame:
    q = t[t.quality]
    rows = []
    for a, b in COMPARISONS:
        for level in LEVELS:
            base = {"comparison": f"{a} vs {b}", "level": level}
            rows.append({**base, "group": "all", **summary(q[f"{a}__{level}"], q[f"{b}__{level}"])})
            for key in ("range_class", "region", "dist_class"):
                src = t[(t.fit_years >= MIN_FIT_YEARS) & (t.fit_explained >= MIN_EXPLAINED)] if key == "dist_class" else q
                for g, sub in src.groupby(key, observed=True):
                    rows.append({**base, "group": f"{key}={g}", **summary(sub[f"{a}__{level}"], sub[f"{b}__{level}"])})
    table = pd.DataFrame(rows)
    table.to_csv(dest / "levels_comparisons.csv", index=False)
    return table


def _save(fig, path: Path) -> None:
    # Windows intermittently refuses to truncate a PNG another process (viewer, indexer) holds.
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=fig.dpi)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


SOURCE_LABELS = {"gesla_harm": "GESLA (gauge)", "gtsm_harm": "GTSM", "fes_gauge": "FES2022 at the gauge",
                 "fes_station": "FES2022 at the GTSM station"}
# (y, x) pairs: each model against the gauge, then the two models at the same GTSM station.
MHWS_PANELS = (("gtsm_harm", "gesla_harm"), ("fes_gauge", "gesla_harm"), ("gtsm_harm", "fes_station"))


def plot_scatter(t: pd.DataFrame, dest: Path) -> None:
    q = t[t.quality]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    regions = sorted(q.region.unique())
    level = "mhws_m"
    for ax, (a, b) in zip(axes, MHWS_PANELS):
        for k, r in enumerate(regions):
            g = q[q.region == r]
            # More regions than palette slots: the second pass reuses the hues with a cross marker.
            ax.scatter(g[f"{b}__{level}"], g[f"{a}__{level}"], s=12, color=COLORS[k % len(COLORS)], alpha=0.7,
                       marker="o" if k < len(COLORS) else "x", linewidths=0 if k < len(COLORS) else 0.9, label=r)
        hi = float(np.nanmax(q[[f"{a}__{level}", f"{b}__{level}"]].to_numpy())) * 1.05
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1)
        s = summary(q[f"{a}__{level}"], q[f"{b}__{level}"])
        ax.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"{SOURCE_LABELS[b]} MHWS (m)", ylabel=f"{SOURCE_LABELS[a]} MHWS (m)",
               title=f"{SOURCE_LABELS[a]} vs {SOURCE_LABELS[b]}\nmedian ratio {s['median_ratio']:.2f}, "
                     f"MAE {s['mae_m']:.2f} m (n={s['n']})")
        ax.title.set_fontsize(9)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("MHWS above local MSL at GESLA sites, same harmonic method for every source "
                 "(quality sites: station <= 10 km, fit >= 3 y)", color=INK)
    fig.tight_layout()
    fig.set_dpi(150)
    _save(fig, dest / "levels_scatter.png")


def plot_maps(t: pd.DataFrame, dest: Path) -> None:
    q = t[t.quality]
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    for ax, (a, b) in zip(axes, MHWS_PANELS):
        ratio = q[f"{a}__mhws_m"] / q[f"{b}__mhws_m"]
        sc = ax.scatter(q.lon, q.lat, c=ratio, s=12, cmap="RdBu_r", vmin=0.6, vmax=1.4, linewidths=0)
        fig.colorbar(sc, ax=ax, label="MHWS ratio", shrink=0.8)
        ax.set(xlim=(-180, 180), ylim=(-60, 75),
               title=f"MHWS ratio {SOURCE_LABELS[a]} / {SOURCE_LABELS[b]} (median {ratio.median():.2f})")
        _style(ax)
    fig.tight_layout()
    fig.set_dpi(150)
    _save(fig, dest / "levels_ratio_maps.png")


def plot_constituents(c: pd.DataFrame, t: pd.DataFrame, dest: Path) -> pd.DataFrame:
    q = c.loc[c.index.intersection(t[t.quality].index)]
    rows = []
    for a, b in (("gtsm_harm", "gesla_harm"), ("fes_gauge", "gesla_harm"), ("fes_station", "gtsm_harm")):
        for k in CONSTITUENTS:
            ok = q[f"{b}__{k}"] > 0.02
            r = (q[f"{a}__{k}"] / q[f"{b}__{k}"])[ok]
            rows.append({"comparison": f"{a} vs {b}", "constituent": k, "n": int(ok.sum()), "median_ratio": r.median(),
                         "p25": r.quantile(0.25), "p75": r.quantile(0.75)})
    table = pd.DataFrame(rows)
    table.to_csv(dest / "levels_constituent_ratios.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comps = table.comparison.unique()
    x = np.arange(len(CONSTITUENTS))
    for k, comp in enumerate(comps):
        g = table[table.comparison == comp].set_index("constituent").loc[list(CONSTITUENTS)]
        ax.errorbar(x + (k - 1) * 0.22, g.median_ratio, yerr=[g.median_ratio - g.p25, g.p75 - g.median_ratio], fmt="o",
                    color=COLORS[k], capsize=3, label=comp)
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set(xticks=x, xticklabels=[c.upper() for c in CONSTITUENTS], ylabel="amplitude ratio (median, IQR)",
           title="Constituent amplitude ratios at GESLA sites (reference amplitude > 2 cm)")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.set_dpi(150)
    _save(fig, dest / "levels_constituents.png")
    return table


def extremes_table(out: Path, t: pd.DataFrame, dest: Path) -> pd.DataFrame | None:
    """COAST-RP vs GEV fits to GESLA and GTSM annual maxima (`levels_extremes.py`), quality sites."""
    if not (out / "extremes.parquet").exists():
        return None
    ex = pd.read_parquet(out / "extremes.parquet")
    g = ex[ex.source == "gesla"].set_index("file_name")
    m = ex[ex.source == "gtsm"].set_index("file_name")
    q = t[t.quality].index.intersection(g.index).intersection(m.index)
    g, m = g.loc[q], m.loc[q]
    tide_excess = (t.loc[q, "gtsm_raw__mhws_m"] - t.loc[q, "gesla_harm__mhws_m"])
    rows = []
    for rp in (10, 100):
        crp = g[f"coast_rp_rp{rp}_m"]
        pairs = {"COAST-RP vs GESLA GEV": (crp, g[f"rp{rp}_m"]), "GTSM GEV vs GESLA GEV": (m[f"rp{rp}_m"], g[f"rp{rp}_m"]),
                 "COAST-RP vs GTSM GEV": (crp, m[f"rp{rp}_m"])}
        for name, (a, b) in pairs.items():
            ok = a.notna() & b.notna()
            d = (a - b)[ok]
            row = {"rp": rp, "comparison": name, "n": int(ok.sum()), "median_diff_m": d.median(),
                   "mae_m": d.abs().mean(), "p10_diff_m": d.quantile(0.1), "p90_diff_m": d.quantile(0.9),
                   "corr_with_gtsm_tide_excess": d.corr(tide_excess[ok])}
            if name == "COAST-RP vs GESLA GEV":
                inside = (crp >= g[f"rp{rp}_p05_m"]) & (crp <= g[f"rp{rp}_p95_m"])
                row["share_inside_gesla_90ci"] = inside[ok].mean()
                row["median_diff_m_minus_tide_excess"] = (d - tide_excess[ok]).median()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(dest / "levels_extremes.csv", index=False)
    by_region = pd.DataFrame({"region": t.loc[q, "region"], "d100": g["coast_rp_rp100_m"] - g["rp100_m"]})
    by_region.groupby("region").d100.agg(["count", "median"]).to_csv(dest / "levels_extremes_by_region.csv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    t = wide_table(out)
    t.to_csv(dest / "levels_sites.csv")
    print(f"{len(t)} sites, {int(t.quality.sum())} pass the quality filter")
    table = comparison_tables(t, dest)
    with pd.option_context("display.width", 200):
        for level in ("mhws_m", "spring_range_m", "hat_m"):
            print(f"\n{level}")
            print(table[(table.level == level) & (table.group == "all")].drop(columns=["level", "group"]).round(3).to_string(index=False))
        print("\nMHWS by GESLA spring range")
        print(table[(table.level == "mhws_m") & table.group.str.startswith("range")].pivot(
            index="group", columns="comparison", values="median_ratio").round(3).to_string())
        print("\nMHWS by region")
        print(table[(table.level == "mhws_m") & table.group.str.startswith("region")].pivot(
            index="group", columns="comparison", values="median_ratio").round(3).to_string())
        print("\nMHWS by gauge-station distance")
        print(table[(table.level == "mhws_m") & table.group.str.startswith("dist")].pivot(
            index="group", columns="comparison", values="median_ratio").round(3).to_string())
    print("\nconstituent ratios")
    print(plot_constituents(constituent_table(out, t.index), t, dest).pivot(
        index="constituent", columns="comparison", values="median_ratio").round(3).to_string())
    plot_scatter(t, dest)
    plot_maps(t, dest)
    ex = extremes_table(out, t, dest)
    if ex is not None:
        print("\nextremes (m)\n" + ex.round(3).to_string(index=False))
        print(pd.read_csv(dest / "levels_extremes_by_region.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
