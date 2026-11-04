"""Step 2a-2c: compare event shapes of GESLA, GTSM and COAST-HG; duration vs event size and regime.

Inputs: `events.parquet` / `event_profiles.npz` (`shapes_events.py`), `levels_sites.csv`
(`levels_analysis.py`, for the GESLA spring range and the quality flag). Only quality sites.
Tables:
  shapes_durations.csv  per source, tidal-range class and empirical-RP class: medians and the
                        share of events above 12 h, for each duration measure
  shapes_pairs.csv      per site, GESLA vs GTSM medians of the measures, and their correlation
  shapes_rp_trend.csv   per source and range class: Spearman correlation of each measure with the
                        event's empirical return period, pooled over sites (rank within site)
Figures: shapes_profiles.png (median normalised profile per range class and source),
shapes_profiles_by_rp.png (the same per empirical-RP class),
shapes_durations.png (distribution of time above MSL and of B's T against 12 h).

Usage:
    python boundary_test/shapes_analysis.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

MEASURES = ("d_above_msl", "d_above_mhw", "d_above_mhws", "cum_above_msl", "t_within_0.1", "t_within_0.25",
            "t_within_0.5", "t_within_0.75", "surge_dur50")
RANGE_BINS = (0, 1, 2, 4, 20)
RP_BINS = (0, 2, 5, 10, 1000)
T_REF_HR = 12.0
SOURCES = ("gesla", "gtsm", "coast_hg_average", "coast_hg_spring")
COLORS = dict(zip(SOURCES, ("#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a")))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def load(out: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ev = pd.read_parquet(out / "events.parquet")
    prof = np.load(out / "event_profiles.npz")
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    ev["range_class"] = pd.cut(ev.file_name.map(sites["gesla_harm__spring_range_m"]), RANGE_BINS)
    ev["rp_class"] = pd.cut(ev.rp_empirical_yr, RP_BINS)
    keep = ev.file_name.map(sites.quality).fillna(False).astype(bool).to_numpy()
    return ev[keep].reset_index(drop=True), prof["profiles"][keep], prof["offset_hr"]


def duration_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in ev.groupby(["source", "range_class", "rp_class"], observed=True, dropna=False):
        row = {"source": keys[0], "range_class": str(keys[1]), "rp_class": str(keys[2]), "n_events": len(g),
               "n_sites": g.file_name.nunique()}
        for m in MEASURES:
            if m in g and g[m].notna().any():
                row[f"{m}_median_hr"] = g[m].median()
                row[f"{m}_share_gt12h"] = (g[m] > T_REF_HR).mean()
        row["censored_msl_share"] = g.censored_msl.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def pair_table(ev: pd.DataFrame) -> pd.DataFrame:
    med = ev[ev.source.isin(["gesla", "gtsm"])].groupby(["file_name", "source"])[list(MEASURES)].median().unstack("source")
    med.columns = [f"{m}__{s}" for m, s in med.columns]
    return med


def rp_trend(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, rc), g in ev[ev.source.isin(["gesla", "gtsm"])].groupby(["source", "range_class"], observed=True):
        rank = g.groupby("file_name").rp_empirical_yr.rank(pct=True)
        for m in MEASURES:
            ok = g[m].notna()
            if ok.sum() > 30:
                rho, p = spearmanr(rank[ok], g[m][ok])
                top = g[ok & (g.rp_empirical_yr >= 10)][m].median()
                low = g[ok & (g.rp_empirical_yr < 2)][m].median()
                rows.append({"source": source, "range_class": str(rc), "measure": m, "spearman_rho": rho, "p_value": p,
                             "median_rp_lt2_hr": low, "median_rp_ge10_hr": top, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def plot_profiles(ev: pd.DataFrame, prof: np.ndarray, offset: np.ndarray, dest: Path) -> None:
    classes = [c for c in ev.range_class.cat.categories]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 3.8), sharey=True)
    for ax, rc in zip(axes, classes):
        for s in SOURCES:
            sel = ((ev.source == s) & (ev.range_class == rc)).to_numpy()
            if s in ("gesla", "gtsm"):
                sel &= (ev.rp_empirical_yr >= 5).to_numpy()
            if sel.sum() < 5:
                continue
            p = np.nanmedian(prof[sel], axis=0)
            lo, hi = np.nanquantile(prof[sel], [0.25, 0.75], axis=0)
            ax.plot(offset, p, color=COLORS[s], linewidth=1.8, label=f"{s} (n={int(sel.sum())})")
            ax.fill_between(offset, lo, hi, color=COLORS[s], alpha=0.12, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set(xlim=(-48, 48), ylim=(-1.2, 1.05), title=f"GESLA spring range {rc} m", xlabel="hours from peak")
        ax.title.set_fontsize(10)
        _style(ax)
        ax.legend(frameon=False, fontsize=7, loc="lower center")
    axes[0].set_ylabel("level / peak (local MSL = 0)")
    fig.suptitle("Median event profile (IQR shaded): GESLA/GTSM annual maxima with empirical RP >= 5 y, COAST-HG RP100",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest / "shapes_profiles.png", dpi=140)
    plt.close(fig)


def plot_profiles_by_rp(ev: pd.DataFrame, prof: np.ndarray, offset: np.ndarray, dest: Path) -> None:
    classes = list(ev.range_class.cat.categories)
    rp_classes = list(ev.rp_class.cat.categories)
    shades = ("#a9c8f0", "#6a9fe4", "#2a78d6", "#123f78")
    fig, axes = plt.subplots(2, len(classes), figsize=(4.2 * len(classes), 7.2), sharex=True, sharey=True)
    for i, s in enumerate(("gesla", "gtsm")):
        for ax, rc in zip(axes[i], classes):
            for rp, color in zip(rp_classes, shades):
                sel = ((ev.source == s) & (ev.range_class == rc) & (ev.rp_class == rp)).to_numpy()
                if sel.sum() >= 5:
                    ax.plot(offset, np.nanmedian(prof[sel], axis=0), color=color, linewidth=1.8,
                            label=f"RP {rp} y (n={int(sel.sum())})")
            hg = ((ev.source == "coast_hg_spring") & (ev.range_class == rc)).to_numpy()
            if hg.sum() >= 5:
                ax.plot(offset, np.nanmedian(prof[hg], axis=0), color=COLORS["coast_hg_spring"], linewidth=1.4,
                        linestyle="--", label=f"COAST-HG spring RP100 (n={int(hg.sum())})")
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set(xlim=(-48, 48), ylim=(-1.2, 1.05), title=f"{s.upper()}, GESLA spring range {rc} m",
                   xlabel="hours from peak" if i == 1 else None)
            ax.title.set_fontsize(10)
            _style(ax)
            ax.legend(frameon=False, fontsize=7, loc="lower center")
        axes[i, 0].set_ylabel("level / peak (local MSL = 0)")
    fig.suptitle("Median annual-maximum profile by empirical return period (rank within site)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest / "shapes_profiles_by_rp.png", dpi=140)
    plt.close(fig)


def plot_durations(ev: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m, title in zip(axes, ("d_above_msl", "t_within_0.25", "t_within_0.5"),
                            ("contiguous time above MSL around the peak",
                             "B's T for head = 25% of the storm tide", "B's T for head = 50% of the storm tide")):
        for s in SOURCES:
            x = np.sort(ev.loc[ev.source == s, m].dropna().to_numpy())
            if len(x):
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLORS[s], linewidth=1.8, label=s)
        ax.axvline(T_REF_HR, color=MUTED, linestyle="--", linewidth=1)
        ax.set(xscale="log", xlim=(1, 160), xlabel="hours", ylabel="cumulative share of events", title=title)
        ax.title.set_fontsize(10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Durations per event (dashed: t_ref = 12 h of correction B)", color=INK)
    fig.tight_layout()
    fig.savefig(dest / "shapes_durations.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    ev, prof, offset = load(out)
    print(f"{len(ev)} events at {ev.file_name.nunique()} quality sites")
    dur = duration_table(ev)
    dur.to_csv(dest / "shapes_durations.csv", index=False)
    overall = ev.groupby("source")[list(MEASURES)].median().round(1)
    share = ev.groupby("source")[list(MEASURES)].agg(lambda x: (x > T_REF_HR).mean()).round(2)
    print("\nmedian duration (h) per source\n" + overall.to_string())
    print("\nshare of events > 12 h per source\n" + share.to_string())
    print("\ncontiguous time above MSL, median (h), by GESLA spring range and source\n"
          + ev.pivot_table(index="range_class", columns="source", values="d_above_msl", aggfunc="median",
                           observed=True).round(1).to_string())
    print("\nshare censored at the +-72 h window (time above MSL)\n" + ev.groupby("source").censored_msl.mean().round(3).to_string())
    pairs = pair_table(ev)
    pairs.to_csv(dest / "shapes_pairs.csv")
    print("\nGESLA vs GTSM per-site medians: Spearman rho / median ratio")
    for m in ("d_above_msl", "t_within_0.25", "t_within_0.5", "surge_dur50"):
        a, b = pairs[f"{m}__gtsm"], pairs[f"{m}__gesla"]
        ok = a.notna() & b.notna() & (b > 0)
        print(f"  {m}: rho {spearmanr(a[ok], b[ok])[0]:.2f}, median ratio {(a[ok] / b[ok]).median():.2f}, n {ok.sum()}")
    trend = rp_trend(ev)
    trend.to_csv(dest / "shapes_rp_trend.csv", index=False)
    print("\nduration vs event size (empirical RP rank within site)\n"
          + trend[trend.measure.isin(["d_above_msl", "t_within_0.25", "t_within_0.5", "surge_dur50"])]
          .round(3).to_string(index=False))
    plot_profiles(ev, prof, offset, dest)
    plot_profiles_by_rp(ev, prof, offset, dest)
    plot_durations(ev, dest)


if __name__ == "__main__":
    main()
