"""Test 3: size of the interannual-MSL term in gauge extremes (linear trend vs annual means removed).

Inputs: `extremes.parquet` (GESLA annual means removed) and `extremes_linear.parquet` (linear
trend removed; `levels_extremes.py --msl linear`), `analysis/site_sets.csv` (`site_sets.py`).
Per site of the extremes set: delta = GEV level (linear) - GEV level (annual means), RP10 / RP100.
Delta is the part of the gauge extremes that comes from interannual MSL anomalies (ENSO, NAO,
steric) and that COAST-RP, being barotropic and relative to a fixed MSL, cannot contain.

Writes <out-dir>/analysis/msl_sensitivity.csv (per site), msl_sensitivity_regions.csv and
msl_sensitivity.png (region medians with interquartile range; map of the RP100 delta).

Usage:
    python boundary_test/msl_sensitivity.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_sets import MUTED, _save, _style  # noqa: E402

RP_COLORS = {10: "#2a78d6", 100: "#eb6834"}


def site_deltas(out: Path) -> pd.DataFrame:
    annual = pd.read_parquet(out / "extremes.parquet")
    annual = annual[annual.source == "gesla"].set_index("file_name")
    linear = pd.read_parquet(out / "extremes_linear.parquet").set_index("file_name")
    sites = pd.read_csv(out / "analysis" / "site_sets.csv", index_col=0)
    keep = sites.index[sites.extremes].intersection(annual.index).intersection(linear.index)
    t = sites.loc[keep, ["site_name", "country", "region", "lon", "lat", "range_class"]].copy()
    for rp in (10, 100):
        t[f"gev_annual_rp{rp}_m"] = annual.loc[keep, f"rp{rp}_m"]
        t[f"gev_linear_rp{rp}_m"] = linear.loc[keep, f"rp{rp}_m"]
        t[f"delta_rp{rp}_m"] = t[f"gev_linear_rp{rp}_m"] - t[f"gev_annual_rp{rp}_m"]
        t[f"coast_rp_minus_linear_rp{rp}_m"] = annual.loc[keep, f"coast_rp_rp{rp}_m"] - t[f"gev_linear_rp{rp}_m"]
    return t.dropna(subset=["delta_rp10_m", "delta_rp100_m"])


def region_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in [("all", t)] + list(t.groupby("region")):
        row = {"region": label, "n": len(g)}
        for rp in (10, 100):
            d = g[f"delta_rp{rp}_m"]
            row.update({f"delta_rp{rp}_median_m": d.median(), f"delta_rp{rp}_p25_m": d.quantile(0.25),
                        f"delta_rp{rp}_p75_m": d.quantile(0.75),
                        f"coast_rp_minus_linear_rp{rp}_median_m": g[f"coast_rp_minus_linear_rp{rp}_m"].median()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, regions: pd.DataFrame, dest: Path) -> None:
    order = ["all"] + regions[regions.region != "all"].sort_values("delta_rp100_median_m", ascending=False).region.tolist()
    r = regions.set_index("region").reindex(order)
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_axes([0.16, 0.1, 0.3, 0.82])
    y = np.arange(len(order))
    for k, rp in enumerate((10, 100)):
        yy = y + (k - 0.5) * 0.3
        ax.hlines(yy, r[f"delta_rp{rp}_p25_m"], r[f"delta_rp{rp}_p75_m"], color=RP_COLORS[rp], linewidth=2.2, alpha=0.6)
        ax.scatter(r[f"delta_rp{rp}_median_m"], yy, s=36, color=RP_COLORS[rp], zorder=3, label=f"RP{rp}")
    for yi, n in zip(y, r.n):
        ax.annotate(f"n={int(n)}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(4, -3),
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set(yticks=y, yticklabels=order, xlabel="GEV (linear trend removed) - GEV (annual means removed) (m)",
           title="Interannual MSL in gauge extremes: median, IQR")
    ax.title.set_fontsize(10)
    ax.invert_yaxis()
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax2 = fig.add_axes([0.53, 0.1, 0.45, 0.82])
    sc = ax2.scatter(t.lon, t.lat, c=t.delta_rp100_m, s=16, cmap="RdBu_r", vmin=-0.2, vmax=0.2, linewidths=0)
    fig.colorbar(sc, ax=ax2, shrink=0.7, label="RP100 delta (m)")
    ax2.set(xlim=(-180, 180), ylim=(-60, 80), title="RP100 delta per site (extremes set)")
    ax2.title.set_fontsize(10)
    _style(ax2)
    _save(fig, dest / "msl_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    t = site_deltas(out)
    t.round(4).to_csv(dest / "msl_sensitivity.csv")
    regions = region_table(t)
    regions.to_csv(dest / "msl_sensitivity_regions.csv", index=False)
    print(regions.round(3).to_string(index=False))
    plot(t, regions, dest)


if __name__ == "__main__":
    main()
