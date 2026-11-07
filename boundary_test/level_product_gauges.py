"""Storm-tide product at the ratio gauges, with r left one out, against the gauge and COAST-RP (§8.4).

Each gauge with a ratio takes its nearest product station within GAUGE_MAX_KM; that station is
rebuilt with r computed without the gauge, so no gauge calibrates itself. The reference is the
gauge's own skew-surge convolution of test 9 (`surge_correction.py`: gauge surge on the gauge
tide). Reads the product directory of `run_preparation.py storm_tide_product` and its cache.
`open_guesses.py` reuses `GaugeCheck` for variants of the product's rules.

Writes <product-dir>/gauge_check.parquet and gauge_check_summary.csv.

Usage:
    python boundary_test/level_product_gauges.py --product-dir <storm_tide_product_dir> \
        [--sources-dir .cleanup_baseline/level_sources] [--config ...] [--override local.yml]
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
GROUPS = ("all", "non-TC-dominated gauges", "TC-dominated gauges", "stations with a TC part")


class GaugeCheck:
    """The product's station inputs at the ratio gauges (r left one out), rebuilt on demand."""

    def __init__(self, out: Path, sources: Path, cfg: dict):
        self.out, self.cfg = out, cfg
        self.cache = Path(cfg["storm_tide_product"]["cache_dir"])
        catalog = get_data_catalog(REPO / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
        coast_rp = Path(catalog.get_source("coast_rp").path)
        self.stations = stations = pd.read_csv(out / "stations.csv", index_col="station")
        self.lev = pd.read_parquet(out / "levels.parquet")
        gauges = pd.read_parquet(out / "gauges" / "gauges.parquet")
        self.reference = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
        lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
        c = np.load(self.cache / "fes_constants.npz")
        models = {"fes2022": tp.fes2022_model(self.cache / "fes2022"),
                  "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
        self.hc = hc = {a: tp.fill_masked(c[a], lonlat) for a in ("fes2022", "fes2014")}
        for a in hc:
            hc[a][:, tp.seasonal_columns(models[a])] = 0.0
        self.mh = mh = np.load(self.cache / "fes_mhws.npz")
        tc_levels = stp.coast_rp_levels(coast_rp.parent / "COAST-RP_TC.nc")
        self.tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
        self.coast = coast = stp.coast_rp_levels(coast_rp).to_numpy()
        self.donors = donors = pd.read_parquet(out / "gauges" / "tc_donors.parquet")
        self.shapes = shapes = np.load(out / "gauges" / "tc_donor_shapes.npy")
        ratio_gauges = gauges[np.isfinite(gauges.ratio_fes)].index.intersection(self.reference.index)
        d, i = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine").query(
            np.radians(gauges.loc[ratio_gauges, ["lat", "lon"]].to_numpy()), k=1)
        near = d[:, 0] * stp.EARTH_RADIUS_KM <= GAUGE_MAX_KM
        self.picked = stations.iloc[i[near, 0]]
        tc_shapes = stp.station_tc_shapes(self.picked, donors, shapes)
        self.todo, meta = [], []
        for name, k, dist in zip(ratio_gauges[near], i[near, 0], d[near, 0] * stp.EARTH_RADIUS_KM):
            st = stations.index[k]
            r = float(stp.station_ratios(stations.iloc[[k]], gauges.drop(index=name)).r.iloc[0])
            a = self.lev.atlas.loc[st]
            sid = stations.station_id.loc[st]
            self.todo.append({"station": len(self.todo), "gtsm_index": int(stations.gtsm_index.loc[st]), "atlas": a,
                              "coast_rp": coast[k], "station_index": st, "hc": hc[a][k], "mhws_m": float(mh[a][k]),
                              "r": r, "tc": self.tc_levels.loc[sid].to_numpy() if sid in self.tc_levels.index else None,
                              "tc_shape": tc_shapes.get(st)})
            meta.append({"file_name": name, "station": st, "station_km": dist, "r_loo": r})
        self.meta = pd.DataFrame(meta)
        print(f"{len(self.todo)} of {len(ratio_gauges)} ratio gauges within {GAUGE_MAX_KM:.0f} km of a station",
              flush=True)

    def tc_hours(self, **mask) -> dict:
        with_tc = self.picked[self.picked.station_id.isin(self.tc_levels.index)]
        return stp.tc_hours(with_tc, len(tp.hourly_times()), Path(self.cfg["storm_tide_product"]["ibtracs_path"]),
                            tp.HOURLY_START, **mask)

    def levels(self, workers: int, tc_hours: dict | None = None, **station_keys) -> pd.DataFrame:
        """Rebuilt levels (gauge, rp{RP}_m / etc_raw_rp{RP}_m); `station_keys` go into every input."""
        hours = self.tc_hours() if tc_hours is None else tc_hours
        todo = [{**s, "tc_hours": hours.get(s["station_index"]), **station_keys} for s in self.todo]
        hyd = np.full((len(todo), len(stp.RPS), stp.N_STEPS), np.nan, dtype=np.float32)
        paths = {a: (self.cache / f"{a}_hourly_re.npy", self.cache / f"{a}_hourly_im.npy",
                     self.cache / f"{a}_nodal_re.npy", self.cache / f"{a}_nodal_im.npy") for a in ("fes2022", "fes2014")}
        rows, _ = stp.build_stations(todo, self.cache / "gtsm_surge_stations", paths, hyd, workers)
        loo = pd.DataFrame(rows).set_index("station").sort_index()
        loo.index = self.meta.file_name.to_numpy()[loo.index]
        return loo

    def table(self, loo: pd.DataFrame) -> pd.DataFrame:
        """Gauge table: reference, COAST-RP, product in sample and `loo` (r left out)."""
        res = self.meta.set_index("file_name").join(self.reference[["region", "station_class"]])
        at = self.lev.loc[res.station]
        res["has_tc"] = np.isfinite(at["coast_rp_tc_rp100_m"]).to_numpy()
        for rp in GAUGE_RPS:
            res[f"gauge_rp{rp}_m"] = self.reference.loc[res.index, f"gauge_gauge_rp{rp}_m"]
            res[f"gev_rp{rp}_m"] = self.reference.loc[res.index, f"gev_gauge_rp{rp}_m"]
            res[f"coast_rp_rp{rp}_m"] = at[f"coast_rp_rp{rp}_m"].to_numpy()
            res[f"new_rp{rp}_m"] = at[f"rp{rp}_m"].to_numpy()
            res[f"new_loo_rp{rp}_m"] = loo[f"rp{rp}_m"].reindex(res.index)
            res[f"new_loo_r1_rp{rp}_m"] = loo[f"etc_raw_rp{rp}_m"].reindex(res.index)
        return res


def group_mask(res: pd.DataFrame, label: str) -> pd.Series:
    return {"all": res.index == res.index, "non-TC-dominated gauges": res.station_class != "tc_dominated",
            "TC-dominated gauges": res.station_class == "tc_dominated", "stations with a TC part": res.has_tc}[label]


def scores(res: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Median, MAE, p10 and p90 of level - gauge per group, RP and column (label -> column prefix)."""
    rows = []
    for label in GROUPS:
        sel = group_mask(res, label)
        for rp in GAUGE_RPS:
            for name, col in columns.items():
                e = (res[f"{col}_rp{rp}_m"] - res[f"gauge_rp{rp}_m"])[sel].dropna()
                rows.append({"gauges": label, "rp": rp, "level": name, "n": len(e), "median": e.median(),
                             "mae": e.abs().mean(), "p10": e.quantile(0.1), "p90": e.quantile(0.9)})
    return pd.DataFrame(rows).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None, help="YAML merged over the config (e.g. a local ibtracs_path)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.product_dir)
    check = GaugeCheck(out, Path(args.sources_dir), load_config(args.config, args.override))
    res = check.table(check.levels(args.workers))
    res.to_parquet(out / "gauge_check.parquet")
    summary = scores(res, {"COAST-RP": "coast_rp", "new, r left out": "new_loo", "new, r = 1, no TC": "new_loo_r1",
                           "new, r in sample": "new"})
    summary.to_csv(out / "gauge_check_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
