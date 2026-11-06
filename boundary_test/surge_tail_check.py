"""Test 7b: does the GTSM surge residual carry the gauges' surge tail, or is it lost in the generator?

Per quality site, the GESLA and GTSM annual-maximum events of the same years (`events.parquet`,
`event_profiles.npz`) are compared on the surge residual (gauge: observed minus harmonic tide;
GTSM: CDS surge series), normalised as the generator does (`median_surge_shape`: centred on the
largest surge within +-12 h of the water-level peak, divided by it), raw and after the 13-h
running mean. Tail value at lag L = mean of the median shape at -L and +L. The levels between
high waters are compared on the peak-normalised total level: the lowest value of the median
profile 6-30 h from the peak, for the gauge, GTSM's own events and the generator
(`generator_profiles.npz`, generator_smooth).

Writes <out-dir>/analysis/surge_tail_sites.parquet and surge_tail_check.csv (medians per tidal
class and event set).

Usage:
    python boundary_test/surge_tail_check.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from storm_tide_hydrograph import median_surge_shape  # noqa: E402

LAGS_HR = (12, 24, 36, 48, 60)
SMOOTH_HR = 13
TROUGH_HR = (6, 30)
RANGE_BINS = (0, 1, 2, 4, 20)
MIN_EVENTS = 5


def _running_mean(windows: np.ndarray, width_hr: int) -> np.ndarray:
    kernel = np.ones(width_hr) / width_hr
    return np.vstack([np.convolve(np.nan_to_num(w), kernel, mode="same") for w in windows])


def tails(windows: np.ndarray) -> dict:
    shape = median_surge_shape(windows)[::6]  # 10-min back to hourly
    c = len(shape) // 2
    return {f"tail_{lag}": 0.5 * (shape[c - lag] + shape[c + lag]) for lag in LAGS_HR}


def trough(profile: np.ndarray) -> float:
    c = len(profile) // 2
    lo, hi = TROUGH_HR
    return float(np.nanmin(np.r_[profile[c - hi:c - lo + 1], profile[c + lo:c + hi + 1]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    sites = pd.read_csv(dest / "levels_sites.csv", index_col=0)
    sites = sites[sites.quality]
    ev = pd.read_parquet(out / "events.parquet").reset_index(drop=True)
    npz = np.load(out / "event_profiles.npz")
    prof, surge = npz["profiles"], npz["surge_m"]
    gp = np.load(dest / "generator_profiles.npz")
    gen = pd.DataFrame({"file_name": gp["file_name"], "rp": gp["rp"], "variant": gp["variant"]})

    rows = []
    for f in sites.index:
        g = ev[(ev.file_name == f) & (ev.source == "gesla")]
        t = ev[(ev.file_name == f) & (ev.source == "gtsm")]
        years = np.intersect1d(g.year.dropna(), t.year.dropna())
        for subset, rp_min in (("all", 0.0), ("rp>=10", 10.0)):
            row = {"file_name": f, "events": subset}
            for name, e in (("gauge", g), ("gtsm", t)):
                e = e[e.year.isin(years) & (e.rp_empirical_yr >= rp_min)]
                if len(e) < (MIN_EVENTS if rp_min == 0 else 2):
                    break
                w = surge[e.index]
                try:
                    row.update({f"{name}_raw_{k}": v for k, v in tails(w).items()})
                    row.update({f"{name}_smooth_{k}": v for k, v in tails(_running_mean(w, SMOOTH_HR)).items()})
                except ValueError:
                    break
                row[f"{name}_trough"] = trough(np.nanmedian(prof[e.index], axis=0))
                row[f"{name}_n"] = len(e)
                edge = np.r_[w[:, :13].ravel(), w[:, -13:].ravel()]
                row[f"{name}_edge_surge_m"] = float(np.nanmedian(edge))
            else:
                rp_gen = 2 if rp_min == 0 else 10
                k = gen.index[(gen.file_name == f) & (gen.rp == rp_gen) & (gen.variant == "generator_smooth")]
                if len(k):
                    row["generator_trough"] = trough(gp["profile"][k[0]].astype(float))
                rows.append(row)
    s = pd.DataFrame(rows)
    s["range_class"] = pd.cut(s.file_name.map(sites.gesla_harm__spring_range_m), RANGE_BINS).astype(str)
    s.to_parquet(dest / "surge_tail_sites.parquet")
    cols = [c for c in s.columns if c not in ("file_name", "events", "range_class")]
    table = s.groupby(["events", "range_class"])[cols].median()
    table.insert(0, "n_sites", s.groupby(["events", "range_class"]).size())
    table.to_csv(dest / "surge_tail_check.csv")
    pd.set_option("display.width", 250)
    for kind in ("raw", "smooth"):
        show = ["n_sites"] + [f"{src}_{kind}_tail_{lag}" for lag in LAGS_HR for src in ("gauge", "gtsm")]
        print(f"\nsurge shape tail ({kind}), fraction of the surge peak")
        print(table[show].round(2).to_string())
    print("\nlowest peak-normalised level 6-30 h from the peak, and median edge surge (m)")
    print(table[["n_sites", "gauge_trough", "gtsm_trough", "generator_trough", "gauge_edge_surge_m",
                 "gtsm_edge_surge_m"]].round(2).to_string())


if __name__ == "__main__":
    main()
