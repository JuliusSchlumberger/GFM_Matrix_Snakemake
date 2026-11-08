"""Test 11 analysis: flood extent and depth errors from hydrograph deviations (`sill_experiment.py`).

Part A: per controlled deviation, the flood measures relative to the undeviated hydrograph on the
same strip (ratio - 1), against the deviation of the hydrograph measures. Part B: per site, RP and
strip, the generator's flood measures against the spread over the gauge's own events (scaled to
the same peak): ratio to their median and whether it lies within their range and interquartile
range. Both parts: how well each hydrograph measure predicts the flood volume (R^2 of
log ratios), including the overflow volume = time integral of (level - sill crest) above the crest.

Writes <out-dir>/sill_experiment/{part_a_errors.csv, part_a_summary.csv, part_b_scores.csv,
part_b_summary.csv, predictors.csv}.

Usage:
    python boundary_test/sill_analysis.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLOOD = ("extent_m", "volume_m2", "max_depth_m")
HYDRO = ("t025_hr", "t05_hr", "nb24", "overflow_m_hr")
DT_HR = 1 / 6
MIN_VOLUME_M2 = 1.0  # runs that barely overtop give meaningless ratios


def overflow(runs: pd.DataFrame, h: dict) -> np.ndarray:
    return np.array([np.clip(h[name] - crest, 0.0, None).sum() * DT_HR
                     for name, crest in zip(runs.hydrograph, runs.crest_m)])


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2) if ok.sum() > 3 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "sill_experiment"
    meta = pd.read_parquet(dest / "hydrographs_both.parquet").set_index("hydrograph")
    h = dict(np.load(dest / "hydrographs_both.npz"))
    runs = pd.read_parquet(dest / "runs_both.parquet")
    runs = runs.join(meta[["part", "range_class", "rp", "kind", "value", "site", "t025_hr", "t05_hr", "nb24"]],
                     on="hydrograph")
    runs["overflow_m_hr"] = overflow(runs, h)
    domain = ["head_fraction", "terrain", "manning"]
    pd.set_option("display.width", 220)

    a = runs[runs.part == "A"].copy()
    base = a[(a.kind == "stretch") & (a.value == 1.0)].set_index(["range_class", "rp", *domain])
    keys = list(zip(a.range_class, a.rp, *(a[c] for c in domain)))
    for c in (*FLOOD, *HYDRO):
        ref = base[c].reindex(keys).to_numpy()
        a[f"{c}_ratio"] = a[c].to_numpy() / ref
    a["base_volume_m2"] = base.volume_m2.reindex(keys).to_numpy()
    a = a[((a.kind != "stretch") | (a.value != 1.0)) & (a.base_volume_m2 >= MIN_VOLUME_M2)]
    a.to_csv(dest / "part_a_errors.csv", index=False)
    summary_a = a.groupby(["kind", "value", "head_fraction"])[[f"{c}_ratio" for c in (*FLOOD, *HYDRO)]].median() - 1
    summary_a.to_csv(dest / "part_a_summary.csv")
    print("Part A: median (ratio - 1) to the undeviated hydrograph, by deviation and head over the sill")
    print((100 * summary_a).round(0).to_string())

    b = runs[runs.part == "B"]
    gauge = b[b.kind == "gauge"].groupby(["site", "rp", *domain])
    gen = b[b.kind == "generator"].set_index(["site", "rp", *domain])
    rows = []
    for key, g in gauge:
        if key not in gen.index or g.volume_m2.median() < MIN_VOLUME_M2:
            continue
        r = gen.loc[key]
        row = dict(zip(["site", "rp", *domain], key), range_class=r.range_class, n_events=len(g))
        for c in (*FLOOD, *HYDRO):
            v = g[c].to_numpy()
            row[f"{c}_ratio"] = r[c] / np.median(v)
            row[f"{c}_in_range"] = bool(v.min() <= r[c] <= v.max())
            row[f"{c}_in_iqr"] = bool(np.percentile(v, 25) <= r[c] <= np.percentile(v, 75))
            row[f"{c}_event_spread"] = (np.percentile(v, 75) - np.percentile(v, 25)) / np.median(v)
        rows.append(row)
    sb = pd.DataFrame(rows)
    sb.to_csv(dest / "part_b_scores.csv", index=False)
    cols = [f"{c}_{k}" for c in FLOOD for k in ("ratio", "in_range", "in_iqr", "event_spread")]
    summary_b = sb.groupby(["range_class", "rp"])[cols].median().join(sb.groupby(["range_class", "rp"]).size()
                                                                      .rename("n"))
    for c in FLOOD:
        for k in ("in_range", "in_iqr"):
            summary_b[f"{c}_{k}"] = sb.groupby(["range_class", "rp"])[f"{c}_{k}"].mean()
    summary_b.to_csv(dest / "part_b_summary.csv")
    print("\nPart B: generator vs the gauge's own events (ratio to their median; share within range / IQR;"
          " event IQR / median)")
    print(summary_b.round(2).to_string())

    pred = []
    for part, d in (("A", a), ("B", sb)):
        for c in HYDRO:
            for f in FLOOD:
                pred.append({"part": part, "predictor": c, "flood": f,
                             "r2_log": _r2(np.log(d[f"{c}_ratio"].to_numpy()), np.log(d[f"{f}_ratio"].to_numpy()))})
    pred = pd.DataFrame(pred).pivot_table(index=["part", "predictor"], columns="flood", values="r2_log")
    pred.to_csv(dest / "predictors.csv")
    print("\nR^2 of log(flood ratio) on log(hydrograph-measure ratio)")
    print(pred.round(2).to_string())


if __name__ == "__main__":
    main()
