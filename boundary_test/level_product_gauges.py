"""Storm-tide product at the ratio gauges, with r left one out, against the gauge and COAST-RP (§8.4).

Each gauge with a ratio takes its nearest product station within GAUGE_MAX_KM; that station is
rebuilt with r computed without the gauge, so no gauge calibrates itself. The reference is the
gauge's own skew-surge convolution of test 9 (`surge_correction.py`: gauge surge on the gauge
tide). Reads the product directory of `run_preparation.py storm_tide_product`.

Writes <product-dir>/gauge_check.parquet and gauge_check_summary.csv.

Usage:
    python boundary_test/level_product_gauges.py --product-dir <storm_tide_product_dir> \
        [--sources-dir .cleanup_baseline/level_sources] [--config snakemake_workflow/config/config.yml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
import storm_tide_product as stp  # noqa: E402
import tide_prediction as tp  # noqa: E402

GAUGE_MAX_KM = 25.0
GAUGE_RPS = (2, 10, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None, help="YAML merged over the config (e.g. a local ibtracs_path)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out, sources = Path(args.product_dir), Path(args.sources_dir)
    cfg = load_config(args.config, args.override)
    catalog = get_data_catalog(REPO / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    coast_rp = Path(catalog.get_source("coast_rp").path)
    stations = pd.read_csv(out / "stations.csv", index_col="station")
    lev = pd.read_parquet(out / "levels.parquet")
    gauges = pd.read_parquet(out / "gauges" / "gauges.parquet")
    reference = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
    lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
    c = np.load(out / "cache" / "fes_constants.npz")
    models = {"fes2022": tp.fes2022_model(out / "cache" / "fes2022"),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    hc = {a: tp.fill_masked(c[a], lonlat) for a in ("fes2022", "fes2014")}
    for a in hc:
        hc[a][:, tp.seasonal_columns(models[a])] = 0.0
    mh = np.load(out / "cache" / "fes_mhws.npz")
    tc_levels = stp.coast_rp_levels(coast_rp.parent / "COAST-RP_TC.nc")
    tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
    coast = stp.coast_rp_levels(coast_rp).to_numpy()
    donors = pd.read_parquet(out / "gauges" / "tc_donors.parquet")
    shapes = np.load(out / "gauges" / "tc_donor_shapes.npy")

    ratio_gauges = gauges[np.isfinite(gauges.ratio_fes)].index.intersection(reference.index)
    d, i = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine").query(
        np.radians(gauges.loc[ratio_gauges, ["lat", "lon"]].to_numpy()), k=1)
    todo, meta = [], []
    near = d[:, 0] * stp.EARTH_RADIUS_KM <= GAUGE_MAX_KM
    picked = stations.iloc[i[near, 0]]
    tc_hours = stp.tc_hours(picked[picked.station_id.isin(tc_levels.index)], len(tp.hourly_times()),
                            Path(cfg["storm_tide_product"]["ibtracs_path"]), tp.HOURLY_START)
    tc_shapes = stp.station_tc_shapes(picked, donors, shapes)
    for name, k, dist in zip(ratio_gauges[near], i[near, 0], d[near, 0] * stp.EARTH_RADIUS_KM):
        st = stations.index[k]
        r = float(stp.station_ratios(stations.iloc[[k]], gauges.drop(index=name)).r.iloc[0])
        a = lev.atlas.loc[st]
        sid = stations.station_id.loc[st]
        todo.append({"station": len(todo), "gtsm_index": int(stations.gtsm_index.loc[st]), "atlas": a,
                     "coast_rp": coast[k], "tc_hours": tc_hours.get(st), "hc": hc[a][k], "mhws_m": float(mh[a][k]),
                     "r": r, "tc": tc_levels.loc[sid].to_numpy() if sid in tc_levels.index else None,
                     "tc_shape": tc_shapes.get(st)})
        meta.append({"file_name": name, "station": st, "station_km": dist, "r_loo": r})
    print(f"{len(todo)} of {len(ratio_gauges)} ratio gauges within {GAUGE_MAX_KM:.0f} km of a station", flush=True)
    hyd = np.full((len(todo), len(stp.RPS), stp.N_STEPS), np.nan, dtype=np.float32)
    paths = {a: (out / "cache" / f"{a}_hourly_re.npy", out / "cache" / f"{a}_hourly_im.npy",
                 out / "cache" / f"{a}_nodal_re.npy", out / "cache" / f"{a}_nodal_im.npy") for a in hc}
    rows, _ = stp.build_stations(todo, out / "cache" / "gtsm_surge_stations", paths, hyd, args.workers)
    loo = pd.DataFrame(rows).set_index("station").sort_index()
    res = pd.DataFrame(meta)
    res = res.join(loo.reset_index(drop=True)[[f"rp{rp}_m" for rp in GAUGE_RPS] + [f"etc_raw_rp{rp}_m" for rp in GAUGE_RPS]]
                   .add_prefix("new_loo_").rename(columns=lambda s: s.replace("new_loo_etc_raw_", "new_loo_r1_")))
    res = res.set_index("file_name").join(reference[["region", "station_class"]])
    at = lev.loc[res.station]
    res["has_tc"] = np.isfinite(at["coast_rp_tc_rp100_m"]).to_numpy()
    for rp in GAUGE_RPS:
        res[f"gauge_rp{rp}_m"] = reference.loc[res.index, f"gauge_gauge_rp{rp}_m"]
        res[f"gev_rp{rp}_m"] = reference.loc[res.index, f"gev_gauge_rp{rp}_m"]
        res[f"coast_rp_rp{rp}_m"] = at[f"coast_rp_rp{rp}_m"].to_numpy()
        res[f"new_rp{rp}_m"] = at[f"rp{rp}_m"].to_numpy()
    res.to_parquet(out / "gauge_check.parquet")
    summary = []
    for label, sel in (("all", res.index == res.index), ("non-TC-dominated gauges", res.station_class != "tc_dominated"),
                       ("TC-dominated gauges", res.station_class == "tc_dominated"),
                       ("stations with a TC part", res.has_tc)):
        for rp in GAUGE_RPS:
            for name, col in (("COAST-RP", "coast_rp"), ("new, r left out", "new_loo"),
                              ("new, r = 1, no TC", "new_loo_r1"), ("new, r in sample", "new")):
                e = (res[f"{col}_rp{rp}_m"] - res[f"gauge_rp{rp}_m"])[sel].dropna()
                summary.append({"gauges": label, "rp": rp, "level": name, "n": len(e), "median": e.median(),
                                "mae": e.abs().mean(), "p10": e.quantile(0.1), "p90": e.quantile(0.9)})
    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(out / "gauge_check_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
