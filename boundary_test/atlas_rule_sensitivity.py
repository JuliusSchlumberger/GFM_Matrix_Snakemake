"""Test 6: the FES version rule and the sensitivity of its two thresholds.

Rule (`version_flag`): at a boundary node FES2022 is replaced by FES2014 where
MHWS_FES2022 > R x MHWS_FES2014 and MHWS_FES2022 - MHWS_FES2014 > D (MHWS above local MSL, same
prediction and statistics as at the gauges). Scanned over RATIOS x DIFFS_M against three kinds
of evidence:

- Gauges (all sites with GESLA, FES2022 and FES2014 MHWS; `atlas_gauges.csv`): the rule is
  evaluated on the two atlases' MHWS at the gauge. A firing is a fix when FES2014 is closer to
  the gauge MHWS by more than MARGIN_M, harm when it is further by more than MARGIN_M.
- Known artefacts (`atlas_node_mhws.parquet`): nodes in the four basin boxes whose FES2022 MHWS
  exceeds IMPLAUSIBLE_FACTOR x the largest observed MHWS there (half the record or upper spring
  range, INPUT_ANALYSIS §3). The set is defined without FES2014, so detection is not circular.
- Replaced nodes: how many boundary nodes change.

Writes <out-dir>/analysis/{atlas_rule_grid.csv, atlas_rule_bands.csv}.

Usage:
    python boundary_test/atlas_rule_sensitivity.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_check import BASINS  # noqa: E402

VERSION_RATIO = 1.5
VERSION_DIFF_M = 1.25
RATIOS = (1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0)
DIFFS_M = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5)
MARGIN_M = 0.05
DIFF_BANDS_M = (-np.inf, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, np.inf)
# Largest observed MHWS above MSL per basin: half the record / upper spring range (Minas 16.3 m,
# Desplanque and Mossman 2001; Rio Gallegos ~11 m; Colorado mouth 10 m, Archer 2013) and the
# Skagway gauge (2.78 m) for Lynn Canal.
OBSERVED_MHWS_M = {"Minas Basin / Cobequid Bay": 8.15, "Strait of Magellan / Bahia Lomas": 5.5,
                   "Upper Gulf of California": 5.0, "Glacier Bay / Lynn Canal": 2.8}
IMPLAUSIBLE_FACTOR = 1.25


def version_flag(fes2022_m, fes2014_m, ratio: float = VERSION_RATIO, diff_m: float = VERSION_DIFF_M):
    """Flag where FES2022 is replaced by FES2014; NaN in either value never flags."""
    return (fes2022_m > ratio * fes2014_m) & (fes2022_m - fes2014_m > diff_m)


def gauge_frame(dest: Path) -> pd.DataFrame:
    g = pd.read_csv(dest / "atlas_gauges.csv", index_col=0)
    g = g.dropna(subset=["gesla_harm__mhws_m", "fes_gauge__mhws_m", "fes2014__mhws_m"]).copy()
    g["e22"] = (g.fes_gauge__mhws_m - g.gesla_harm__mhws_m).abs()
    g["e14"] = (g.fes2014__mhws_m - g.gesla_harm__mhws_m).abs()
    return g


def node_frame(dest: Path) -> pd.DataFrame:
    return pd.read_parquet(dest / "atlas_node_mhws.parquet").dropna(subset=["fes2022_mhws_m", "fes2014_mhws_m"])


def known_artefacts(nodes: pd.DataFrame) -> pd.Series:
    known = pd.Series(False, index=nodes.index)
    for name, (x0, x1, y0, y1) in BASINS.items():
        box = nodes.node_lon.between(x0, x1) & nodes.node_lat.between(y0, y1)
        known |= box & (nodes.fes2022_mhws_m > IMPLAUSIBLE_FACTOR * OBSERVED_MHWS_M[name])
    return known


def grid_table(g: pd.DataFrame, nodes: pd.DataFrame, known: pd.Series) -> pd.DataFrame:
    rows = []
    for r in RATIOS:
        for d in DIFFS_M:
            fire = version_flag(g.fes_gauge__mhws_m, g.fes2014__mhws_m, r, d)
            hybrid = np.where(fire, g.e14, g.e22)
            node_fire = version_flag(nodes.fes2022_mhws_m, nodes.fes2014_mhws_m, r, d)
            rows.append({"ratio": r, "diff_m": d, "gauges_fired": int(fire.sum()),
                         "gauges_fixed": int((fire & (g.e14 < g.e22 - MARGIN_M)).sum()),
                         "gauges_harmed": int((fire & (g.e14 > g.e22 + MARGIN_M)).sum()),
                         "hybrid_mae_m": float(hybrid.mean()), "nodes_replaced": int(node_fire.sum()),
                         "known_caught": int((node_fire & known).sum()), "known_total": int(known.sum())})
    return pd.DataFrame(rows)


def band_table(g: pd.DataFrame) -> pd.DataFrame:
    band = pd.cut(g.fes_gauge__mhws_m - g.fes2014__mhws_m, DIFF_BANDS_M)
    t = g.assign(gain=g.e22 - g.e14, win=g.e14 < g.e22).groupby(band, observed=True).agg(
        n=("gain", "size"), fes2014_closer=("win", "mean"), median_gain_m=("gain", "median"),
        mean_gain_m=("gain", "mean"))
    t.index = [f"{iv.left:+.2f} .. {iv.right:+.2f}" for iv in t.index]
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "analysis"
    g = gauge_frame(dest)
    nodes = node_frame(dest)
    known = known_artefacts(nodes)
    grid = grid_table(g, nodes, known)
    grid.round(4).to_csv(dest / "atlas_rule_grid.csv", index=False)
    bands = band_table(g)
    bands.round(3).to_csv(dest / "atlas_rule_bands.csv")
    print(f"gauges n={len(g)}, nodes n={len(nodes)}, known artefacts n={int(known.sum())}")
    print(bands.round(3).to_string())
    print(grid.round(3).to_string(index=False))
    fired = version_flag(g.fes_gauge__mhws_m, g.fes2014__mhws_m)
    print(g.loc[fired, ["site_name", "gesla_harm__mhws_m", "fes_gauge__mhws_m", "fes2014__mhws_m"]].round(2).to_string())


if __name__ == "__main__":
    main()
