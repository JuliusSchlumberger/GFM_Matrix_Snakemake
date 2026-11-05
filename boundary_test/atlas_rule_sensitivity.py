"""Test 6: sensitivity of the FES version rule to its two thresholds.

Rule (`atlas_check.version_flag`): FES2022 is replaced by FES2014 where the amplitude proxy
A = |M2|+|S2|+|K1|+|O1| satisfies A_FES2022 > R x A_FES2014 and A_FES2022 - A_FES2014 > D.
Scanned over RATIOS x DIFFS_M against three kinds of evidence:

- Gauges (all sites with GESLA, FES2022 and FES2014 MHWS; `atlas_gauges.csv`): the rule is
  evaluated on the proxy at the gauge and scored on MHWS. A firing is a fix when FES2014 is closer
  to the gauge by more than MARGIN_M, harm when it is further by more than MARGIN_M.
- Known artefacts (`atlas_nodes.parquet`): nodes in the four basin boxes whose FES2022 MHWS
  exceeds IMPLAUSIBLE_FACTOR x the largest observed MHWS there (half the record or upper spring
  range, INPUT_ANALYSIS §3). This set is defined without FES2014, so detection is not circular.
- Replaced nodes: how many of the 89,628 boundary nodes change, and in how many 2-deg cells.

Writes <out-dir>/analysis/{atlas_rule_grid.csv, atlas_rule_bands.csv, atlas_rule_sensitivity.png}.

Usage:
    python boundary_test/atlas_rule_sensitivity.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_check import BASINS, PROXY, VERSION_DIFF_M, VERSION_RATIO, version_flag  # noqa: E402
from site_sets import INK, MUTED, _save, _style  # noqa: E402

RATIOS = (1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0)
DIFFS_M = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MARGIN_M = 0.05
DIFF_BANDS_M = (-np.inf, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, np.inf)
# Largest observed MHWS above MSL per basin: half the record / upper spring range (Minas 16.3 m,
# Desplanque and Mossman 2001; Rio Gallegos ~11 m; Colorado mouth 10 m, Archer 2013) and the
# Skagway gauge (2.78 m) for Lynn Canal.
OBSERVED_MHWS_M = {"Minas Basin / Cobequid Bay": 8.15, "Strait of Magellan / Bahia Lomas": 5.5,
                   "Upper Gulf of California": 5.0, "Glacier Bay / Lynn Canal": 2.8}
IMPLAUSIBLE_FACTOR = 1.25


def gauge_frame(dest: Path) -> pd.DataFrame:
    g = pd.read_csv(dest / "atlas_gauges.csv", index_col=0)
    g["p22"] = sum(g[f"fes__{c}_m"] for c in PROXY)
    g["p14"] = sum(g[f"fes2014__{c}_m"] for c in PROXY)
    g = g.dropna(subset=["p22", "p14", "gesla_harm__mhws_m", "fes_gauge__mhws_m", "fes2014__mhws_m"]).copy()
    g["e22"] = (g.fes_gauge__mhws_m - g.gesla_harm__mhws_m).abs()
    g["e14"] = (g.fes2014__mhws_m - g.gesla_harm__mhws_m).abs()
    return g


def known_artefacts(nodes: pd.DataFrame) -> pd.Series:
    known = pd.Series(False, index=nodes.index)
    for name, (x0, x1, y0, y1) in BASINS.items():
        box = nodes.node_lon.between(x0, x1) & nodes.node_lat.between(y0, y1)
        known |= box & (nodes.fes_mhws_m > IMPLAUSIBLE_FACTOR * OBSERVED_MHWS_M[name])
    return known


def grid_table(g: pd.DataFrame, nodes: pd.DataFrame, known: pd.Series) -> pd.DataFrame:
    rows = []
    cells = (nodes.node_lon // 2).astype(int).astype(str) + "/" + (nodes.node_lat // 2).astype(int).astype(str)
    for r in RATIOS:
        for d in DIFFS_M:
            fire = version_flag(g.p22, g.p14, r, d)
            hybrid = np.where(fire, g.e14, g.e22)
            node_fire = version_flag(nodes.fes_proxy_m, nodes.fes2014_proxy_m, r, d)
            rows.append({"ratio": r, "diff_m": d, "gauges_fired": int(fire.sum()),
                         "gauges_fixed": int((fire & (g.e14 < g.e22 - MARGIN_M)).sum()),
                         "gauges_harmed": int((fire & (g.e14 > g.e22 + MARGIN_M)).sum()),
                         "gauge_net_gain_m": float((g.e22 - g.e14)[fire].sum()),
                         "hybrid_mae_m": float(hybrid.mean()), "hybrid_n_over_0p5m": int((hybrid > 0.5).sum()),
                         "nodes_replaced": int(node_fire.sum()), "cells_replaced": int(cells[node_fire].nunique()),
                         "known_caught": int((node_fire & known).sum()), "known_total": int(known.sum()),
                         "replaced_outside_known": int((node_fire & ~known).sum())})
    return pd.DataFrame(rows)


def band_table(g: pd.DataFrame) -> pd.DataFrame:
    band = pd.cut(g.p22 - g.p14, DIFF_BANDS_M)
    t = g.assign(gain=g.e22 - g.e14, win=g.e14 < g.e22).groupby(band, observed=True).agg(
        n=("gain", "size"), fes2014_closer=("win", "mean"), median_gain_m=("gain", "median"),
        mean_gain_m=("gain", "mean"))
    t.index = [f"{iv.left:+.2f} .. {iv.right:+.2f}" for iv in t.index]
    return t


def _heatmap(ax, grid: pd.DataFrame, col: str, title: str, fmt, cmap: str, centred: bool = False) -> None:
    m = grid.pivot(index="ratio", columns="diff_m", values=col).to_numpy(dtype=float)
    lim = float(np.abs(m).max()) or 1.0
    norm = TwoSlopeNorm(0.0, -lim, lim) if centred else Normalize(float(m.min()), float(m.max()))
    ax.imshow(m, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ratios, diffs = sorted(grid.ratio.unique()), sorted(grid.diff_m.unique())
    for i, r in enumerate(ratios):
        for j, d in enumerate(diffs):
            level = float(norm(m[i, j]))
            dark = abs(level - 0.5) > 0.3 if centred else level > 0.6
            ax.text(j, i, fmt(grid[(grid.ratio == r) & (grid.diff_m == d)].iloc[0]), ha="center", va="center",
                    fontsize=6.5, color="white" if dark else INK)
    i0, j0 = ratios.index(VERSION_RATIO), diffs.index(VERSION_DIFF_M)
    ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, edgecolor="#e34948", linewidth=2.5))
    ax.set_xticks(range(len(diffs)), [f"{d:g}" for d in diffs], fontsize=8)
    ax.set_yticks(range(len(ratios)), [f"{r:g}" for r in ratios], fontsize=8)
    ax.set(xlabel="D: FES2022 - FES2014 proxy floor (m)", ylabel="R: FES2022 / FES2014 proxy ratio", title=title)
    ax.title.set_fontsize(9)


def plot(grid: pd.DataFrame, bands: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    y = np.arange(len(bands))
    ax.barh(y, bands.mean_gain_m, color=np.where(bands.mean_gain_m > 0, "#1f9e89", "#2a78d6"), height=0.6)
    for yi, (_, b) in zip(y, bands.iterrows()):
        ax.annotate(f"n={int(b.n)}, FES2014 closer {b.fes2014_closer:.0%}, median {b.median_gain_m:+.2f} m",
                    (max(b.mean_gain_m, 0.0), yi), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(min(-0.9, float(bands.mean_gain_m.min()) - 0.1), float(bands.mean_gain_m.max()) + 1.6)
    ax.set_yticks(y, bands.index, fontsize=8)
    ax.set(xlabel="mean gain of switching to FES2014: |FES2022 - gauge| - |FES2014 - gauge| (m)",
           ylabel="FES2022 - FES2014 proxy at the gauge (m)",
           title="Gauges: switching pays only where FES2022 exceeds FES2014 by more than ~1 m")
    ax.title.set_fontsize(9)
    _style(ax)
    _heatmap(axes[0, 1], grid, "gauge_net_gain_m",
             f"Gauges (n=905): rule fires / fixes / harms (margin {MARGIN_M} m); colour and 2nd line: net gain (m)",
             lambda r: f"{int(r.gauges_fired)}/{int(r.gauges_fixed)}/{int(r.gauges_harmed)}\n{r.gauge_net_gain_m:+.1f}",
             "RdBu", centred=True)
    _heatmap(axes[1, 0], grid, "known_caught",
             f"Share of known artefacts replaced (FES2022 MHWS > {IMPLAUSIBLE_FACTOR} x observed, 4 basins, "
             f"n={int(grid.known_total.iloc[0])})",
             lambda r: f"{r.known_caught / r.known_total:.0%}", "Blues")
    _heatmap(axes[1, 1], grid, "nodes_replaced", "Boundary nodes replaced (2-deg cells) of 89,628",
             lambda r: f"{int(r.nodes_replaced)}\n({int(r.cells_replaced)})", "Oranges")
    fig.suptitle(f"FES version rule: sensitivity to R and D (box: adopted R = {VERSION_RATIO}, D = {VERSION_DIFF_M} m)",
                 color=INK)
    fig.tight_layout()
    _save(fig, dest / "atlas_rule_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "analysis"
    g = gauge_frame(dest)
    nodes = pd.read_parquet(dest / "atlas_nodes.parquet")
    known = known_artefacts(nodes)
    grid = grid_table(g, nodes, known)
    grid.round(4).to_csv(dest / "atlas_rule_grid.csv", index=False)
    bands = band_table(g)
    bands.round(3).to_csv(dest / "atlas_rule_bands.csv")
    print(f"gauges n={len(g)}, known artefacts n={int(known.sum())}")
    print(bands.round(3).to_string())
    print(grid.round(3).to_string(index=False))
    plot(grid, bands, dest)


if __name__ == "__main__":
    main()
