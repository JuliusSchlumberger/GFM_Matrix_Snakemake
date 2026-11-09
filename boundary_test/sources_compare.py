"""Step 2e-2h: GTSM vs COAST-RP / COAST-HG peaks and shapes, T by RP, GESLA shapes at TC stations.

Inputs (all from earlier steps, quality sites only): `extremes.parquet` (`levels_extremes.py`),
`events.parquet` / `event_profiles.npz` (`shapes_events.py`), `analysis/levels_sites.csv`;
COAST-RP / _ETC / _TC on P: (read-only) for the station classes.
  2e peaks      GTSM GEV - COAST-RP per RP, station class (TC- / ETC-dominated / no TC value)
                and region: median, 10-90% range, MAE
  2f top events per site: median normalised profile of the 3 largest annual maxima of GESLA and
                of GTSM, and the COAST-HG RP100 profiles. Metrics on the profiles: neighbouring
                high water (max of level / peak at 9-16 h and 21-28 h from the peak, both sides
                averaged), T(f) (hours with level / peak >= 1 - f), contiguous hours above MSL,
                and the profile RMSE against GESLA over +-48 h
  2g T by RP    T(0.1 / 0.25 / 0.5) per empirical-RP class, relative to the RP >= 10 y class
  2h TC         TC-dominated COAST-RP stations (TC RP100 > ETC RP100): distance to the nearest
                quality gauge; at TC-dominated gauges the surge duration of the largest events and
                the factor that scales their surge to COAST-RP RP100

Writes <out-dir>/analysis/compare_*.csv.

Usage:
    python boundary_test/sources_compare.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sfincs_tiles"))
from gfm_config import read_root  # noqa: E402

DATA_ROOT = read_root(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")

COAST_RP_DIR = DATA_ROOT / "inputs" / "COAST-RP_dataset"
EARTH_RADIUS_KM = 6371.0
SAME_STATION_KM = 0.1
PEAK_RPS = (2, 10, 50, 100)
RP_BINS = (0, 2, 5, 10, 1000)
RANGE_BINS = (0, 1, 2, 4, 20)
TOP_N = 3
FRACTIONS = (0.1, 0.25, 0.5)
NEIGHBOUR_WINDOWS_HR = {"nb12": (9, 16), "nb24": (21, 28)}
RMSE_HALF_WINDOW_HR = 48
DIST_CLASSES_KM = (25, 50, 100, 250)


def station_classes() -> pd.DataFrame:
    """Class of every COAST-RP station from its ETC and TC RP100 (TC file matched by coordinates)."""
    with netCDF4.Dataset(COAST_RP_DIR / "COAST-RP.nc") as ds:
        ll = np.column_stack([ds["station_y_coordinate"][:], ds["station_x_coordinate"][:]]).astype(float)
    with netCDF4.Dataset(COAST_RP_DIR / "COAST-RP_ETC.nc") as ds:
        etc = np.ma.filled(ds["storm_tide_rp_0100"][:], np.nan).astype(float)
    with netCDF4.Dataset(COAST_RP_DIR / "COAST-RP_TC.nc") as ds:
        tc_ll = np.column_stack([ds["station_y_coordinate"][:], ds["station_x_coordinate"][:]]).astype(float)
        tc_all = np.ma.filled(ds["storm_tide_rp_0100"][:], np.nan).astype(float)
    d, i = BallTree(np.radians(tc_ll), metric="haversine").query(np.radians(ll), k=1)
    tc = np.where(d[:, 0] * EARTH_RADIUS_KM <= SAME_STATION_KM, tc_all[i[:, 0]], np.nan)
    cls = np.where(np.isnan(tc), "no_tc", np.where(tc > etc, "tc", "etc"))
    return pd.DataFrame({"lat": ll[:, 0], "lon": ll[:, 1], "etc_rp100_m": etc, "tc_rp100_m": tc, "station_class": cls})


def site_class(gesla: pd.DataFrame) -> pd.Series:
    tc, etc = gesla.coast_rp_tc_rp100_m, gesla.coast_rp_etc_rp100_m
    return pd.Series(np.where(tc.isna(), "no_tc", np.where(tc > etc, "tc", "etc")), index=gesla.index)


def _spread(d: pd.Series) -> dict:
    d = d.dropna()
    return {"n": len(d), "median_m": d.median(), "p10_m": d.quantile(0.1), "p90_m": d.quantile(0.9),
            "mae_m": d.abs().mean()}


def peaks_table(ex: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    g = ex[ex.source == "gesla"].set_index("file_name")
    t = ex[ex.source == "gtsm"].set_index("file_name")
    keep = g.index.intersection(sites.index[sites.quality]).intersection(t.index)
    g, t = g.loc[keep], t.loc[keep]
    cls, region = site_class(g), sites.region.reindex(keep)
    rows = []
    for rp in PEAK_RPS:
        d = t[f"rp{rp}_m"] - g[f"coast_rp_rp{rp}_m"]
        for key, groups in (("all", pd.Series("all", index=keep)), ("station_class", cls), ("region", region)):
            for name, idx in groups.groupby(groups).groups.items():
                rows.append({"rp": rp, "grouping": key, "group": name, **_spread(d.loc[idx])})
    return pd.DataFrame(rows)


def profile_metrics(p: np.ndarray, offset: np.ndarray) -> dict:
    out = {}
    for name, (lo, hi) in NEIGHBOUR_WINDOWS_HR.items():
        left = np.nanmax(p[(offset >= -hi) & (offset <= -lo)])
        right = np.nanmax(p[(offset >= lo) & (offset <= hi)])
        out[name] = (left + right) / 2
    for f in FRACTIONS:
        out[f"t_{f:g}_hr"] = float(np.sum(p >= 1 - f))
    above = p > 0
    c = int(np.flatnonzero(offset == 0)[0])
    lo = c
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = c
    while hi < len(p) - 1 and above[hi + 1]:
        hi += 1
    out["d_above_msl_hr"] = float(hi - lo + 1)
    return out


def top_event_table(ev: pd.DataFrame, prof: np.ndarray, offset: np.ndarray, sites: pd.DataFrame,
                    classes: pd.Series) -> pd.DataFrame:
    window = np.abs(offset) <= RMSE_HALF_WINDOW_HR
    rows = []
    for site, g in ev.groupby("file_name"):
        profiles = {}
        for s in ("gesla", "gtsm"):
            top = g[g.source == s].nlargest(TOP_N, "peak_m")
            if len(top) == TOP_N:
                profiles[s] = np.nanmedian(prof[top.index], axis=0)
        for s in ("coast_hg_spring", "coast_hg_average"):
            hg = g[g.source == s]
            if len(hg):
                profiles[s] = prof[hg.index[0]]
        if "gesla" not in profiles:
            continue
        for s, p in profiles.items():
            rmse = np.sqrt(np.nanmean((p[window] - profiles["gesla"][window]) ** 2))
            rows.append({"file_name": site, "source": s, **profile_metrics(p, offset),
                         "rmse_vs_gesla": rmse if s != "gesla" else 0.0})
    out = pd.DataFrame(rows)
    out["range_class"] = pd.cut(out.file_name.map(sites.gesla_harm__spring_range_m), RANGE_BINS).astype(str)
    out["station_class"] = out.file_name.map(classes)
    return out


def t_by_rp_table(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    e = ev[ev.source.isin(["gesla", "gtsm"])].copy()
    e["rp_class"] = pd.cut(e.rp_empirical_yr, RP_BINS).astype(str)
    e["range_class"] = pd.cut(e.file_name.map(sites.gesla_harm__spring_range_m), RANGE_BINS).astype(str)
    cols = [f"t_within_{f:g}" for f in FRACTIONS]
    med = e.groupby(["source", "range_class", "rp_class"])[cols].median()
    ref = med.xs("(10, 1000]", level="rp_class")
    ratio = med.div(ref.reindex(med.index.droplevel("rp_class")).to_numpy())
    ratio.columns = [f"{c}_ratio_to_rp10plus" for c in cols]
    n = e.groupby(["source", "range_class", "rp_class"]).size().rename("n_events")
    return pd.concat([n, med, ratio], axis=1).reset_index()


def tc_table(stations: pd.DataFrame, ev: pd.DataFrame, ex: pd.DataFrame, sites: pd.DataFrame,
             classes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = sites[sites.quality]
    tree = BallTree(np.radians(q[["lat", "lon"]].to_numpy(dtype=float)), metric="haversine")
    d, _ = tree.query(np.radians(stations[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    stations = stations.assign(nearest_gauge_km=d[:, 0] * EARTH_RADIUS_KM)
    cover = [{"station_class": c, "n_stations": len(g),
              **{f"share_gauge_within_{k}km": (g.nearest_gauge_km <= k).mean() for k in DIST_CLASSES_KM}}
             for c, g in stations.groupby("station_class")]
    g = ex[ex.source == "gesla"].set_index("file_name")
    rows = []
    for site, e in ev[ev.source == "gesla"].groupby("file_name"):
        if site not in classes.index or site not in g.index:
            continue
        top = e.nlargest(TOP_N, "peak_m")
        rp100 = g.loc[site, "coast_rp_rp100_m"]
        rows.append({"file_name": site, "station_class": classes[site], "region": sites.region.get(site),
                     "surge_dur50_hr": top.surge_dur50.median(), "surge_share": (top.surge_at_peak_m / top.peak_m).median(),
                     "peak_max_m": top.peak_m.max(), "coast_rp_rp100_m": rp100,
                     "peak_scale": rp100 / top.peak_m.max(),
                     "surge_scale": ((rp100 - top.tide_at_peak_m) / top.surge_at_peak_m).median()})
    return pd.DataFrame(cover), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    sites = pd.read_csv(dest / "levels_sites.csv", index_col=0)
    ex = pd.read_parquet(out / "extremes.parquet")
    ev = pd.read_parquet(out / "events.parquet")
    prof = np.load(out / "event_profiles.npz")
    profiles, offset = prof["profiles"], prof["offset_hr"]
    quality = ev.file_name.map(sites.quality).fillna(False).astype(bool)
    ev = ev[quality]
    g = ex[ex.source == "gesla"].set_index("file_name")
    classes = site_class(g)

    peaks = peaks_table(ex, sites)
    peaks.to_csv(dest / "compare_peaks.csv", index=False)
    print("GTSM GEV - COAST-RP (m), by station class\n"
          + peaks[peaks.grouping.isin(["all", "station_class"])].round(3).to_string(index=False))

    top = top_event_table(ev, profiles, offset, sites, classes)
    top.to_csv(dest / "compare_top_events.csv", index=False)
    metrics = ["nb12", "nb24", "t_0.25_hr", "t_0.5_hr", "d_above_msl_hr", "rmse_vs_gesla"]
    print("\nlargest events (median over sites), by tidal range class\n"
          + top.groupby(["range_class", "source"])[metrics].median().round(2).to_string())
    print("\nlargest events, by station class\n"
          + top.groupby(["station_class", "source"])[metrics].median().round(2).to_string())

    tby = t_by_rp_table(ev, sites)
    tby.to_csv(dest / "compare_t_by_rp.csv", index=False)
    print("\nT by empirical RP class\n" + tby.round(2).to_string(index=False))

    cover, tc = tc_table(station_classes(), ev, ex, sites, classes)
    cover.to_csv(dest / "compare_tc_coverage.csv", index=False)
    tc.to_csv(dest / "compare_tc_events.csv", index=False)
    print("\nnearest quality gauge per COAST-RP station class\n" + cover.round(3).to_string(index=False))
    print("\nlargest GESLA events by station class (medians)\n"
          + tc.groupby("station_class")[["surge_dur50_hr", "surge_share", "peak_scale", "surge_scale"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
