"""Storm-tide product at the ratio gauges, with r left one out, against the gauge and COAST-RP (§8.4).

Each gauge with a ratio takes its nearest product station within stp.GAUGE_STATION_KM; that
station is rebuilt from production's inputs (1950-2024 record, IBTrACS mask, pooled GPD shape,
COAST-RP_TC shift, atlas) with r and the tide factor computed without the gauge, so no gauge
calibrates itself; the tide factor keeps production's through-water rule where the product has
its station links. The all-basin shift is fitted at all TC-dominated gauges, the scored one
among them (1 of ~170). The
reference is the gauge's own skew-surge convolution of test 9 (`surge_correction.py`: gauge surge
on the gauge tide). Reads the product directory of `run_preparation.py storm_tide_product` and its
cache. `open_guesses.py` reuses `GaugeCheck` for variants of the product's rules.

Writes <dest-dir>/gauge_check.parquet, gauge_check_summary.csv and gauge_check_breakdown.csv
(by region and by gauge spring tidal range); dest-dir defaults to the product directory.

Usage:
    python boundary_test/level_product_gauges.py --product-dir <storm_tide_product_dir> \
        [--dest-dir <dir>] [--sources-dir .cleanup_baseline/level_sources] [--config ...] [--override local.yml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from gesla import DEDUP_KM  # noqa: E402
import storm_tide_product as stp  # noqa: E402
import tide_correction as tdc  # noqa: E402
import tide_prediction as tp  # noqa: E402

GAUGE_RPS = (2, 10, 50, 100)
GROUPS = ("all", "non-TC-dominated gauges", "TC-dominated gauges", "stations with a TC part")
RANGE_BINS = (0, 1, 2, 4, 20)  # gauge spring tidal range (m), the classes of generator_validation
MIN_REGION_GAUGES = 15  # smaller AR6 regions are scored together as "other"
ATLASES = ("fes2022", "fes2014")
PRODUCTION_RECORD = "1950-2024"


class GaugeCheck:
    """Production's station inputs at the ratio gauges (r and tide factor left one out), rebuilt on demand."""

    def __init__(self, out: Path, sources: Path, cfg: dict):
        self.out, self.cfg = out, cfg
        self.cache = cache = Path(cfg["storm_tide_product"]["cache_dir"])
        # record -> (surge stores, FES bases, start, hours); 1979-2024 is the GTSM-ERA5 store alone
        self.records = {
            PRODUCTION_RECORD: ((cache / "gtsm_surge_stations_{}_{}".format(*stp.EARLY_YEARS), cache / "gtsm_surge_stations"),
                                {a: (cache / f"{a}_record_re.npy", cache / f"{a}_record_im.npy",
                                     cache / f"{a}_nodal_re.npy", cache / f"{a}_nodal_im.npy") for a in ATLASES},
                                stp.RECORD_START, len(stp.record_times())),
            "1979-2024": (cache / "gtsm_surge_stations",
                          {a: (cache / f"{a}_hourly_re.npy", cache / f"{a}_hourly_im.npy",
                               cache / f"{a}_nodal_re.npy", cache / f"{a}_nodal_im.npy") for a in ATLASES},
                          tp.HOURLY_START, len(tp.hourly_times()))}
        catalog = get_data_catalog(REPO / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
        coast_rp = Path(catalog.get_source("coast_rp").path)
        self.stations = stations = pd.read_csv(out / "stations.csv", index_col="station")
        self.lev = pd.read_parquet(out / "levels.parquet")
        self.gauges = gauges = pd.read_parquet(out / "gauges" / "gauges.parquet")
        tide_gauges = pd.read_parquet(out / "gauges" / "tide_gauges.parquet")
        # a product built with the through-water tide rule carries its station links
        links_path = out / "gauges" / "tide_links_stations.npz"
        links = load_npz(links_path).tocsr() if links_path.exists() else None
        self.reference = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
        lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
        c = np.load(cache / "fes_constants.npz")
        models = {"fes2022": tp.fes2022_model(cache / "fes2022"),
                  "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
        self.constituents = {a: list(models[a].constituents) for a in ATLASES}
        self.hc = hc = {a: tp.fill_masked(c[a], lonlat) for a in ATLASES}
        for a in hc:
            hc[a][:, tp.seasonal_columns(models[a])] = 0.0
        self.mh = mh = np.load(cache / "fes_mhws.npz")
        tc_levels = stp.coast_rp_levels(coast_rp.parent / "COAST-RP_TC.nc")
        self.tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
        self.coast = coast = stp.coast_rp_levels(coast_rp).to_numpy()
        self.donors = donors = pd.read_parquet(out / "gauges" / "tc_donors.parquet")
        self.shapes = shapes = np.load(out / "gauges" / "tc_donor_shapes.npy")
        ratio_gauges = gauges[np.isfinite(gauges.ratio_fes)].index.intersection(self.reference.index)
        d, i = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine").query(
            np.radians(gauges.loc[ratio_gauges, ["lat", "lon"]].to_numpy()), k=1)
        near = d[:, 0] * stp.EARTH_RADIUS_KM <= stp.GAUGE_STATION_KM
        self.picked = stations.iloc[i[near, 0]]
        tc_shapes = stp.station_tc_shapes(self.picked, donors, shapes)
        tg_lonlat = tide_gauges[["lon", "lat"]].to_numpy(dtype=float)
        tg_tree = BallTree(np.radians(tg_lonlat[:, ::-1]), metric="haversine")
        self.todo, meta = [], []
        for name, k, dist in zip(ratio_gauges[near], i[near, 0], d[near, 0] * stp.EARTH_RADIUS_KM):
            st = stations.index[k]
            others = gauges.drop(index=name)
            r = float(stp.station_ratios(stations.iloc[[k]], others).r_idw.iloc[0])
            a = self.lev.atlas.loc[st]
            sid = stations.station_id.loc[st]
            # the gauge's own site (and its duplicate records) leaves the tide factor
            site = tg_tree.query_radius(np.radians(gauges.loc[[name], ["lat", "lon"]].to_numpy(dtype=float)),
                                        r=DEDUP_KM / stp.EARTH_RADIUS_KM)[0]
            log_ratio = tide_gauges[f"log_ratio_{a}"].to_numpy(dtype=float).copy()
            log_ratio[site] = np.nan
            f = float(tdc.correction_factors(lonlat[[k]], tg_lonlat, log_ratio,
                                             links=None if links is None else links[[k]])[0][0])
            item = {"station": len(self.todo), "gtsm_index": int(stations.gtsm_index.loc[st]), "atlas": a,
                    "coast_rp": coast[k], "station_index": st, "hc": hc[a][k], "mhws_m": float(mh[a][k]),
                    "r_idw": r, "r_covariate": stp.ratio_covariate(others),
                    "tc": self.tc_levels.loc[sid].to_numpy() if sid in self.tc_levels.index else None,
                    "tc_shape": tc_shapes.get(st), "pooled_shape": float(self.lev.gpd_shape.loc[st])}
            if np.isfinite(self.lev.tc_shift_m.loc[st]):
                item["tc_shift_m"] = float(self.lev.tc_shift_m.loc[st])
            self.todo.append(item)
            meta.append({"file_name": name, "station": st, "station_km": dist, "r_loo": r, "tide_factor_loo": f})
        self.meta = pd.DataFrame(meta)
        print(f"{len(self.todo)} of {len(ratio_gauges)} ratio gauges within {stp.GAUGE_STATION_KM:.0f} km "
              "of a station", flush=True)

    def tc_hours(self, record: str = PRODUCTION_RECORD, **mask) -> dict:
        with_tc = self.picked[self.picked.station_id.isin(self.tc_levels.index)]
        _, _, start, n_hours = self.records[record]
        return stp.tc_hours(with_tc, n_hours, Path(self.cfg["storm_tide_product"]["ibtracs_path"]), start, **mask)

    def inputs(self, tide_corrected: bool = True, **station_keys) -> list[dict]:
        """Station inputs with the gauge-left-out tide factor (or none) and `station_keys` applied."""
        out = []
        for s, f in zip(self.todo, self.meta.tide_factor_loo):
            f = f if tide_corrected else 1.0
            out.append({**s, "hc": s["hc"] * f, "mhws_m": float(s["mhws_m"] * f), **station_keys})
        return out

    def levels(self, workers: int, mask_hours: dict | None = None, record: str = PRODUCTION_RECORD,
               tide_corrected: bool = True, **station_keys) -> pd.DataFrame:
        """Rebuilt levels (gauge, rp{RP}_m / etc_raw_rp{RP}_m); `station_keys` go into every input.

        `mask_hours` None applies production's IBTrACS mask on `record`; {} builds without a mask.
        """
        hours = self.tc_hours(record) if mask_hours is None else mask_hours
        stores, bases, start, _ = self.records[record]
        todo = [{**s, "tc_hours": hours.get(s["station_index"])} for s in self.inputs(tide_corrected, **station_keys)]
        hyd = np.full((len(todo), len(stp.RPS), stp.N_STEPS), np.nan, dtype=np.float32)
        rows, _ = stp.build_stations(todo, stores, bases, hyd, workers, start=start)
        loo = pd.DataFrame(rows).set_index("station").sort_index()
        loo.index = self.meta.file_name.to_numpy()[loo.index]
        return loo

    def table(self, loo: pd.DataFrame) -> pd.DataFrame:
        """Gauge table: reference, COAST-RP, product in sample and `loo` (r and tide factor left out)."""
        res = self.meta.set_index("file_name").join(self.reference[["region", "station_class"]])
        res["spring_range_m"] = 2.0 * self.gauges.mhws_gauge_m.reindex(res.index)
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


def breakdown_groups(res: pd.DataFrame) -> dict:
    """Gauge masks by AR6 region (small regions as "other") and by spring tidal range class."""
    counts = res.region.value_counts()
    region = res.region.where(res.region.isin(counts.index[counts >= MIN_REGION_GAUGES]), "other")
    groups = {("region", r): (region == r).to_numpy() for r in sorted(region.dropna().unique())}
    rc = pd.cut(res.spring_range_m, RANGE_BINS)
    groups |= {("spring tidal range (m)", str(c)): (rc == c).to_numpy() for c in rc.cat.categories}
    return groups


def scores(res: pd.DataFrame, columns: dict, groups: dict | None = None) -> pd.DataFrame:
    """Median, MAE, p10 and p90 of level - gauge per group, RP and column (label -> column prefix).

    `groups`: label -> gauge mask, None for GROUPS; (split, group) labels fill a "split" column.
    """
    groups = groups if groups is not None else {label: group_mask(res, label) for label in GROUPS}
    rows = []
    for label, sel in groups.items():
        key = {"split": label[0], "gauges": label[1]} if isinstance(label, tuple) else {"gauges": label}
        for rp in GAUGE_RPS:
            for name, col in columns.items():
                e = (res[f"{col}_rp{rp}_m"] - res[f"gauge_rp{rp}_m"])[sel].dropna()
                rows.append({**key, "rp": rp, "level": name, "n": len(e), "median": e.median(),
                             "mae": e.abs().mean(), "p10": e.quantile(0.1), "p90": e.quantile(0.9)})
    return pd.DataFrame(rows).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--dest-dir", default=None, help="output directory (default: the product directory)")
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None, help="YAML merged over the config (e.g. a local ibtracs_path)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.product_dir)
    dest = Path(args.dest_dir) if args.dest_dir else out
    dest.mkdir(parents=True, exist_ok=True)
    check = GaugeCheck(out, Path(args.sources_dir), load_config(args.config, args.override))
    res = check.table(check.levels(args.workers))
    res.to_parquet(dest / "gauge_check.parquet")
    columns = {"COAST-RP": "coast_rp", "new, r left out": "new_loo", "new, r = 1, no TC": "new_loo_r1",
               "new, r in sample": "new"}
    summary = scores(res, columns)
    summary.to_csv(dest / "gauge_check_summary.csv", index=False)
    scores(res, columns, breakdown_groups(res)).to_csv(dest / "gauge_check_breakdown.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
