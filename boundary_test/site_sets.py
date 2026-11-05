"""Test 1: re-run the extremes and shape statistics on a record-based site set, next to the tide set.

The tide set (`levels_analysis` quality flag: harmonic fit >= 3 y explaining >= 50% of the
variance, GTSM station <= 10 km) is right for tidal levels but drops surge-dominated microtidal
coasts. The extremes set keeps every site with >= MIN_GOOD_YEARS complete years since 1979
(`gesla_records.csv`) and a GTSM station <= 10 km, whatever its tide. Both sets are scored the same:
  extremes   COAST-RP - GESLA GEV and GTSM GEV - GESLA GEV, RP10 / RP100, per region and class
  durations  all annual maxima (GESLA, GTSM): contiguous time above MSL, share > 12 h, T(0.25)
  top events the 3 largest events per site (`sources_compare.top_event_table`)

Writes <out-dir>/analysis/{site_sets.csv, sets_extremes.csv, sets_durations.csv,
sets_top_events.csv} and sets_map.png, sets_extremes.png, sets_durations.png.

Usage:
    python boundary_test/site_sets.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources_compare as sc  # noqa: E402

MIN_GOOD_YEARS = 20
MAX_DIST_KM = 10.0
RANGE_BINS = (0, 1, 2, 4, 20)
T_REF_HR = 12.0
SETS = ("tide", "extremes")
SET_LABELS = {"tide": "tide set (harmonic quality)", "extremes": f"extremes set (>= {MIN_GOOD_YEARS} complete years)"}
SET_COLORS = {"tide": "#2a78d6", "extremes": "#eb6834"}
MEMBERSHIP_COLORS = {"both": "#898781", "tide only": "#2a78d6", "extremes only": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _save(fig, path: Path) -> None:
    # Windows intermittently refuses to truncate a PNG another process (viewer, indexer) holds.
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=140)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)


def site_table(out: Path) -> pd.DataFrame:
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    records = pd.read_csv(out / "gesla_records.csv", index_col=0)
    sites["good_years"] = records.good_years.reindex(sites.index)
    sites["tide"] = sites.quality.astype(bool)
    sites["extremes"] = (sites.good_years >= MIN_GOOD_YEARS) & (sites.gtsm_dist_km <= MAX_DIST_KM)
    sites["membership"] = np.select([sites.tide & sites.extremes, sites.tide, sites.extremes],
                                    ["both", "tide only", "extremes only"], "neither")
    sites["range_class"] = pd.cut(sites.gesla_harm__spring_range_m, RANGE_BINS).astype(str)
    return sites


def _spread(d: pd.Series) -> dict:
    d = d.dropna()
    return {"n": len(d), "median_m": d.median(), "p25_m": d.quantile(0.25), "p75_m": d.quantile(0.75),
            "p10_m": d.quantile(0.1), "p90_m": d.quantile(0.9), "mae_m": d.abs().mean()}


def extremes_table(ex: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    g = ex[ex.source == "gesla"].set_index("file_name")
    t = ex[ex.source == "gtsm"].set_index("file_name")
    classes = sc.site_class(g)
    rows = []
    for set_name in SETS:
        keep = sites.index[sites[set_name]].intersection(g.index).intersection(t.index)
        for rp in (10, 100):
            diffs = {"COAST-RP - GESLA": g.loc[keep, f"coast_rp_rp{rp}_m"] - g.loc[keep, f"rp{rp}_m"],
                     "GTSM - GESLA": t.loc[keep, f"rp{rp}_m"] - g.loc[keep, f"rp{rp}_m"]}
            for comparison, d in diffs.items():
                groupings = {"all": pd.Series("all", index=keep), "region": sites.region.reindex(keep),
                             "station_class": classes.reindex(keep)}
                for grouping, labels in groupings.items():
                    for label, idx in labels.groupby(labels).groups.items():
                        rows.append({"set": set_name, "rp": rp, "comparison": comparison, "grouping": grouping,
                                     "group": label, **_spread(d.loc[idx])})
    return pd.DataFrame(rows)


def duration_table(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for set_name in SETS:
        e = ev[ev.source.isin(["gesla", "gtsm"]) & ev.file_name.isin(sites.index[sites[set_name]])].copy()
        e["range_class"] = e.file_name.map(sites.range_class)
        e["region"] = e.file_name.map(sites.region)
        for grouping in ("range_class", "region"):
            for (source, label), g in e.groupby(["source", grouping]):
                rows.append({"set": set_name, "source": source, "grouping": grouping, "group": label,
                             "n_events": len(g), "n_sites": g.file_name.nunique(),
                             "d_above_msl_median_hr": g.d_above_msl.median(),
                             "d_above_msl_p25_hr": g.d_above_msl.quantile(0.25),
                             "d_above_msl_p75_hr": g.d_above_msl.quantile(0.75),
                             "share_above_msl_gt12h": (g.d_above_msl > T_REF_HR).mean(),
                             "t025_median_hr": g["t_within_0.25"].median(),
                             "t025_p25_hr": g["t_within_0.25"].quantile(0.25),
                             "t025_p75_hr": g["t_within_0.25"].quantile(0.75)})
    return pd.DataFrame(rows)


def top_events_table(ev: pd.DataFrame, profiles: np.ndarray, offset: np.ndarray, sites: pd.DataFrame,
                     classes: pd.Series) -> pd.DataFrame:
    rows = []
    for set_name in SETS:
        e = ev[ev.file_name.isin(sites.index[sites[set_name]])]
        top = sc.top_event_table(e, profiles, offset, sites, classes)
        top["set"] = set_name
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def plot_map(sites: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, color in MEMBERSHIP_COLORS.items():
        s = sites[sites.membership == label]
        ax.scatter(s.lon, s.lat, s=14 if label == "both" else 22, color=color, linewidths=0,
                   alpha=0.6 if label == "both" else 0.95, label=f"{label} (n={len(s)})")
    rest = sites[sites.membership == "neither"]
    ax.scatter(rest.lon, rest.lat, s=8, facecolors="none", edgecolors=MUTED, linewidths=0.6,
               label=f"neither (n={len(rest)})")
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="longitude", ylabel="latitude",
           title="GESLA sites by set: tide set (harmonic fit) vs extremes set (record length)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    _save(fig, dest / "sets_map.png")


def plot_extremes(table: pd.DataFrame, dest: Path) -> None:
    t = table[(table.comparison == "COAST-RP - GESLA") & (table.grouping.isin(["all", "region"]))]
    groups = (t[(t.set == "extremes") & (t.rp == 100)].sort_values("median_m").group.tolist())
    groups = ["all"] + [g for g in groups if g != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(groups) + 1.8), sharey=True)
    y = np.arange(len(groups))
    for ax, rp in zip(axes, (10, 100)):
        for k, set_name in enumerate(SETS):
            s = t[(t.set == set_name) & (t.rp == rp)].set_index("group").reindex(groups)
            yy = y + (k - 0.5) * 0.32
            ax.hlines(yy, s.p25_m, s.p75_m, color=SET_COLORS[set_name], linewidth=2.2, alpha=0.6)
            ax.scatter(s.median_m, yy, s=36, color=SET_COLORS[set_name], zorder=3, label=SET_LABELS[set_name])
            for yi, (m, n) in zip(yy, zip(s.median_m, s.n)):
                if np.isfinite(m):
                    ax.annotate(f"{int(n)}", (m, yi), xytext=(4, -3), textcoords="offset points", fontsize=7,
                                color=MUTED)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set(yticks=y, yticklabels=groups, xlabel=f"COAST-RP - GESLA GEV, RP{rp} (m)",
               title=f"RP{rp}: median (dot) and interquartile range (bar); label = sites")
        ax.title.set_fontsize(10)
        ax.invert_yaxis()
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, dest / "sets_extremes.png")


def plot_durations(table: pd.DataFrame, dest: Path) -> None:
    t = table[(table.grouping == "region") & (table.source == "gesla")]
    groups = t[t.set == "extremes"].sort_values("d_above_msl_median_hr", ascending=False).group.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(groups) + 1.8), sharey=True)
    y = np.arange(len(groups))
    panels = (("d_above_msl", "contiguous time above MSL around the peak (h)"),
              ("t025", "T(0.25): hours within 25% of the peak, all tidal cycles (h)"))
    for ax, (m, xlabel) in zip(axes, panels):
        for k, set_name in enumerate(SETS):
            s = t[t.set == set_name].set_index("group").reindex(groups)
            yy = y + (k - 0.5) * 0.32
            ax.hlines(yy, s[f"{m}_p25_hr"], s[f"{m}_p75_hr"], color=SET_COLORS[set_name], linewidth=2.2, alpha=0.6)
            ax.scatter(s[f"{m}_median_hr"], yy, s=36, color=SET_COLORS[set_name], zorder=3,
                       label=SET_LABELS[set_name])
        ax.axvline(T_REF_HR, color=MUTED, linestyle="--", linewidth=1)
        ax.set(yticks=y, yticklabels=groups, xscale="log", xlim=(1, 160), xlabel=xlabel,
               title="GESLA annual maxima: median (dot), interquartile range (bar); dashed: 12 h")
        ax.title.set_fontsize(10)
        ax.invert_yaxis()
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, dest / "sets_durations.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    sites = site_table(out)
    sites.to_csv(dest / "site_sets.csv")
    print(sites.membership.value_counts().to_string())
    ex = pd.read_parquet(out / "extremes.parquet")
    ev = pd.read_parquet(out / "events.parquet")
    prof = np.load(out / "event_profiles.npz")
    classes = sc.site_class(ex[ex.source == "gesla"].set_index("file_name"))

    extremes = extremes_table(ex, sites)
    extremes.to_csv(dest / "sets_extremes.csv", index=False)
    durations = duration_table(ev, sites)
    durations.to_csv(dest / "sets_durations.csv", index=False)
    top = top_events_table(ev, prof["profiles"], prof["offset_hr"], sites, classes)
    top.to_csv(dest / "sets_top_events.csv", index=False)

    cols = ["n", "median_m", "p10_m", "p90_m", "mae_m"]
    print("\nCOAST-RP - GESLA GEV\n" + extremes[(extremes.comparison == "COAST-RP - GESLA")
                                            & extremes.grouping.isin(["all", "station_class"])]
          .set_index(["set", "rp", "group"])[cols].round(3).to_string())
    print("\nby region, RP100\n" + extremes[(extremes.comparison == "COAST-RP - GESLA") & (extremes.rp == 100)
                                         & (extremes.grouping == "region")]
          .pivot(index="group", columns="set", values=["n", "median_m", "mae_m"]).round(3).to_string())
    print("\nGESLA durations by range class\n" + durations[(durations.source == "gesla")
                                                          & (durations.grouping == "range_class")]
          .set_index(["group", "set"])[["n_sites", "d_above_msl_median_hr", "share_above_msl_gt12h", "t025_median_hr"]]
          .round(2).to_string())
    print("\nGESLA durations by region\n" + durations[(durations.source == "gesla") & (durations.grouping == "region")]
          .pivot(index="group", columns="set", values=["n_sites", "d_above_msl_median_hr", "share_above_msl_gt12h"])
          .round(2).to_string())
    print("\ntop events by range class (medians)\n" + top.groupby(["set", "range_class", "source"])
          [["nb24", "t_0.25_hr", "rmse_vs_gesla"]].median().round(2).to_string())
    plot_map(sites, dest)
    plot_extremes(extremes, dest)
    plot_durations(durations, dest)


if __name__ == "__main__":
    main()
