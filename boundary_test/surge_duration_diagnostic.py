"""F6a: surge duration vs surge peak per station, and where per-RP event selection runs out.

Per test-tile station: GTSM surge events (>= 72 h apart), their dur50 (hours above half the
peak) against the peak, a log-linear fit, and the RP surge targets (COAST-RP level - MHWS; MHWS
from --mhws-col of stations.parquet). Reports the first RP whose target lies above every
recorded peak (HGRAPHER-RP's "above_record"). Also checks whether the residual carries annual
MSL variation (decision 4): spread of the annual means. Writes F6a figures and
surge_duration_summary.csv under <out-dir>.

Usage:
    python boundary_test/surge_duration_diagnostic.py --out-dir .cleanup_baseline/boundary_test
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from storm_tide_hydrograph import (  # noqa: E402
    fill_short_gaps,
    fit_duration_scale,
    remove_running_mean,
    storm_events,
    to_10min,
)

RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
HALF_WINDOW_HR = 60.0


def station_events(series: pd.Series) -> dict | None:
    filled = fill_short_gaps(series)
    if filled is None:
        return None
    return storm_events(to_10min(remove_running_mean(filled)), HALF_WINDOW_HR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mhws-col", default="spring_hw_proxy_m")
    args = parser.parse_args()
    out = Path(args.out_dir)
    surge_st, time, surge = load_series(out, "surge")
    stations = pd.read_parquet(out / "stations.parquet")
    strata = pd.read_csv(out / "test_tiles_provisional.csv", dtype={"tile_id": str}).set_index("tile_id").stratum
    tiles = pd.read_csv(out / "test_station_tiles.csv", dtype={"tile_id": str})
    tiles = tiles[tiles.tile_id.isin(strata.index)]
    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")

    rows, panels = [], []
    for j in np.flatnonzero(surge_st.gtsm_index.isin(tiles.gtsm_index)):
        lon, lat = float(surge_st.lon.iloc[j]), float(surge_st.lat.iloc[j])
        st = stations.iloc[tree.query(np.radians([[lat, lon]]), k=1)[1][0, 0]]
        series = pd.Series(station_series(surge, j), index=time)
        annual = series.groupby(series.index.year).mean()
        ev = station_events(series)
        tile = tiles[tiles.gtsm_index == int(surge_st.gtsm_index.iloc[j])].tile_id.iloc[0]
        rec = {"tile_id": tile, "stratum": strata.get(tile), "gtsm_station": int(surge_st.gtsm_station.iloc[j]),
               "lon": lon, "lat": lat, "tc_dominated": bool(st.tc_dominated),
               "annual_mean_std_m": float(annual.std()),
               "annual_mean_trend_mm_per_yr": float(np.polyfit(annual.index, annual.values, 1)[0] * 1000)}
        if ev is None or len(ev["surge_peak_m"]) < 20:
            rows.append({**rec, "note": "gaps > 3 h or < 20 events"})
            continue
        targets = np.array([st[f"rp{rp}_m"] - st[args.mhws_col] for rp in RPS])
        above = [rp for rp, s in zip(RPS, targets) if s > ev["surge_peak_m"].max()]
        a, b = fit_duration_scale(ev["surge_peak_m"], ev["dur50_hr"])
        rho = spearmanr(ev["surge_peak_m"], ev["dur50_hr"]).statistic
        top = np.argsort(ev["surge_peak_m"])[-10:]
        rec.update(n_events=len(ev["surge_peak_m"]), max_peak_m=float(ev["surge_peak_m"].max()),
                   surge_target_rp100_m=float(targets[RPS.index(100)]),
                   above_record_from_rp=above[0] if above else None,
                   fit_a=a, fit_b_per_m=b, spearman_rho=float(rho),
                   dur50_median_hr=float(np.median(ev["dur50_hr"])),
                   dur50_top10_median_hr=float(np.median(ev["dur50_hr"][top])),
                   tau_rp100_hr=float(np.exp(a + b * targets[RPS.index(100)])))
        rows.append(rec)
        panels.append((rec, ev, targets))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "surge_duration_summary.csv", index=False)

    # One panel per tile: the tile's station with the most events.
    best = {}
    for rec, ev, targets in panels:
        if rec["tile_id"] not in best or rec["n_events"] > best[rec["tile_id"]][0]["n_events"]:
            best[rec["tile_id"]] = (rec, ev, targets)
    n = len(best)
    ncol = 5
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(4 * ncol, 3.4 * np.ceil(n / ncol)),
                             squeeze=False)
    for ax, (rec, ev, targets) in zip(axes.ravel(), sorted(best.values(), key=lambda x: str(x[0]["stratum"]))):
        ax.scatter(ev["surge_peak_m"], ev["dur50_hr"], s=4, alpha=0.4, color="0.3")
        x = np.linspace(0, max(ev["surge_peak_m"].max(), np.nanmax(targets)) * 1.05, 50)
        ax.plot(x, np.exp(rec["fit_a"] + rec["fit_b_per_m"] * x), color="tab:red", lw=1)
        for rp, s in zip(RPS, targets):
            if rp in (10, 100, 1000):
                ax.axvline(s, color="tab:blue", lw=0.8, ls=":")
                ax.text(s, ax.get_ylim()[1] * 0.95, f"RP{rp}", fontsize=6, color="tab:blue", rotation=90, va="top")
        ax.set_yscale("log")
        ax.set_title(f"tile {rec['tile_id']} ({rec['stratum']}{', TC' if rec['tc_dominated'] else ''})\n"
                     f"rho {rec['spearman_rho']:.2f}, above record from RP{rec['above_record_from_rp']}",
                     fontsize=8)
        ax.set_xlabel("surge peak (m)", fontsize=7)
        ax.set_ylabel("dur50 (h)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("F6a: GTSM surge event duration (hours above half peak) vs peak; red = log-linear fit; "
                 "blue = RP surge targets (COAST-RP - MHWS proxy)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F6a_surge_duration_vs_peak.png", dpi=130)
    print(summary.drop(columns=["lon", "lat"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
