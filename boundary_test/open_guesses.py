"""Sensitivity of the storm-tide product to the rules set by estimate (TODO.md open questions).

Stages (each writes <product-dir>/open_guesses/<stage>*.parquet for the figure script):
  cap       GPD shape cap: levels at the ratio gauges (r left one out) against the gauge, and
            RP100 / RP1000 at the stations the cap touches, for caps none / 0.5 / 0.35 / 0.25 / 0.2
  despike   GTSM residual jumps per station: largest hourly jump and the number above 1-3 m,
            i.e. how many stations each (jump, count) threshold sends to COAST-RP
  tc_mask   the IBTrACS mask: radius, time window, wind threshold and no mask, scored at the
            gauges with a COAST-RP TC part
  fes_rule  FES2022 / FES2014 MHWS at the gauges and at the stations the version rule flags
  peaks     hydrographs whose maximum is not at time 0: how many, how far, and the tide's
            diurnal inequality there
Gauge reference: test 9's convolution of the gauge's own skew surges on its own tide.

Usage:
    python boundary_test/open_guesses.py <stage> [...] --product-dir <storm_tide_product_dir> \
        [--sources-dir .cleanup_baseline/level_sources] [--override local.yml] [--workers 4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from gtsm_series import MISSING, load_series  # noqa: E402
from level_product_gauges import GaugeCheck, scores  # noqa: E402
import storm_tide_product as stp  # noqa: E402
import tide_prediction as tp  # noqa: E402

CAPS = {"none": None, "0.5": 0.5, "0.35 (production)": 0.35, "0.25": 0.25, "0.2": 0.2}
CAP_SAMPLE_MIN_SHAPE = 0.2  # stations whose MLE shape any tested cap would bind
TC_MASKS = {"no mask": None,
            "300 km, +-24 h, 34 kt (production)": {},
            "150 km, +-24 h, 34 kt": {"radius_km": 150.0},
            "500 km, +-24 h, 34 kt": {"radius_km": 500.0},
            "300 km, +-12 h, 34 kt": {"half_hr": 12},
            "300 km, +-48 h, 34 kt": {"half_hr": 48},
            "300 km, +-24 h, 64 kt": {"min_wind_kt": 64.0}}
JUMPS_M = (1.0, 2.0, 3.0)
ROWS_PER_READ = 400


def _station_inputs(check: GaugeCheck, ks: np.ndarray) -> list[dict]:
    st = check.stations.iloc[ks]
    shapes = stp.station_tc_shapes(st, check.donors, check.shapes)
    hours = stp.tc_hours(st[st.station_id.isin(check.tc_levels.index)], len(tp.hourly_times()),
                         Path(check.cfg["storm_tide_product"]["ibtracs_path"]), tp.HOURLY_START)
    todo = []
    for j, k in enumerate(ks):
        idx, s = check.stations.index[k], check.stations.iloc[k]
        a = check.lev.atlas.iloc[k]
        todo.append({"station": j, "gtsm_index": int(s.gtsm_index), "atlas": a, "coast_rp": check.coast[k],
                     "hc": check.hc[a][k], "mhws_m": float(check.mh[a][k]), "r": float(check.lev.r.iloc[k]),
                     "tc": check.tc_levels.loc[s.station_id].to_numpy() if s.station_id in check.tc_levels.index
                     else None, "tc_shape": shapes.get(idx), "tc_hours": hours.get(idx)})
    return todo


def _build(check: GaugeCheck, todo: list[dict], workers: int) -> pd.DataFrame:
    hyd = np.full((len(todo), len(stp.RPS), stp.N_STEPS), np.nan, dtype=np.float32)
    paths = {a: (check.cache / f"{a}_hourly_re.npy", check.cache / f"{a}_hourly_im.npy",
                 check.cache / f"{a}_nodal_re.npy", check.cache / f"{a}_nodal_im.npy") for a in ("fes2022", "fes2014")}
    rows, _ = stp.build_stations(todo, check.cache / "gtsm_surge_stations", paths, hyd, workers)
    return pd.DataFrame(rows).set_index("station").sort_index()


def cap(check: GaugeCheck, dest: Path, workers: int) -> None:
    gauge_rows, station_rows = [], []
    lev = check.lev
    ks = np.flatnonzero((lev.gpd_shape_mle > CAP_SAMPLE_MIN_SHAPE).to_numpy() & (lev.level_source == "product").to_numpy())
    print(f"cap: {len(ks)} stations with MLE shape > {CAP_SAMPLE_MIN_SHAPE}", flush=True)
    todo = _station_inputs(check, ks)
    hours = check.tc_hours()
    for name, value in CAPS.items():
        res = check.table(check.levels(workers, hours, shape_cap=value))
        gauge_rows.append(scores(res, {name: "new_loo"}))
        res[[f"new_loo_rp{rp}_m" for rp in (10, 100)] + ["gauge_rp100_m", "station_class"]].assign(cap=name) \
            .to_parquet(dest / f"cap_gauges_{name.split()[0]}.parquet")
        built = _build(check, [{**s, "shape_cap": value} for s in todo], workers)
        station_rows.append(pd.DataFrame({"station": lev.index[ks][built.index], "cap": name,
                                          "shape_mle": lev.gpd_shape_mle.iloc[ks[built.index]].to_numpy(),
                                          "rp100_m": built.rp100_m.to_numpy(), "rp1000_m": built.rp1000_m.to_numpy(),
                                          "lon": lev.lon.iloc[ks[built.index]].to_numpy(),
                                          "lat": lev.lat.iloc[ks[built.index]].to_numpy()}))
    pd.concat(gauge_rows).to_parquet(dest / "cap_scores.parquet")
    pd.concat(station_rows).to_parquet(dest / "cap_stations.parquet")
    gauge_shapes = check.reference[["shape_gauge", "shape_gtsm", "station_class"]]
    gauge_shapes.to_parquet(dest / "cap_gauge_shapes.parquet")
    print(pd.concat(gauge_rows).query("rp == 100").to_string(index=False), flush=True)


def despike(check: GaugeCheck, dest: Path) -> None:
    st_tab, _, series = load_series(check.cache / "gtsm_surge_stations")
    rows = []
    for i in range(0, len(st_tab), ROWS_PER_READ):
        block = np.asarray(series[i:i + ROWS_PER_READ]).astype(np.float32)
        block[block == MISSING] = np.nan
        jump = np.abs(np.diff(block, axis=1)) / 1000.0
        row = {"gtsm_index": st_tab.gtsm_index.to_numpy()[i:i + ROWS_PER_READ], "max_jump_m": np.nanmax(jump, axis=1)}
        for m in JUMPS_M:
            row[f"jumps_gt_{m:g}m"] = np.sum(jump > m, axis=1)
        rows.append(pd.DataFrame(row))
        if i % (20 * ROWS_PER_READ) == 0:
            print(f"  {i}/{len(st_tab)}", flush=True)
    jumps = pd.concat(rows).set_index("gtsm_index")
    out = check.stations[["station_id", "lon", "lat", "gtsm_index"]].join(jumps, on="gtsm_index")
    out["level_source"] = check.lev.level_source.to_numpy()
    out.to_parquet(dest / "despike.parquet")
    print(out.describe().round(3).to_string(), flush=True)


def tc_mask(check: GaugeCheck, dest: Path, workers: int) -> None:
    rows = []
    for name, mask in TC_MASKS.items():
        hours = {} if mask is None else check.tc_hours(**mask)
        res = check.table(check.levels(workers, hours))
        rows.append(scores(res, {name: "new_loo"}))
        print(f"  {name}", flush=True)
    s = pd.concat(rows)
    s.to_parquet(dest / "tc_mask_scores.parquet")
    print(s[s.gauges.isin(["TC-dominated gauges", "stations with a TC part"])].to_string(index=False), flush=True)


def fes_rule(check: GaugeCheck, dest: Path, workers: int) -> None:
    gauges = pd.read_parquet(check.out / "gauges" / "gauges.parquet")
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    model14 = tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))
    hc14 = tp.grouped_constants(model14, gauges[["lon", "lat"]].to_numpy(dtype=float))
    hc14[:, tp.seasonal_columns(model14)] = 0.0
    gauges["mhws_fes2014_m"] = stp.nodal_mhws((check.cache / "fes2014_nodal_re.npy", check.cache / "fes2014_nodal_im.npy"),
                                              hc14, workers)
    gauges[["lon", "lat", "mhws_gauge_m", "mhws_fes2022_m", "mhws_fes2014_m"]].to_parquet(dest / "fes_rule_gauges.parquet")
    lev = check.lev
    st = lev[["station_id", "lon", "lat", "atlas", "mhws_fes2022_m", "mhws_fes2014_m", "rp100_m", "coast_rp_rp100_m"]].copy()
    st["rule_flag"] = np.asarray(tp.version_flag(st.mhws_fes2022_m, st.mhws_fes2014_m), dtype=bool)
    st.to_parquet(dest / "fes_rule_stations.parquet")
    g = gauges.dropna(subset=["mhws_gauge_m", "mhws_fes2022_m", "mhws_fes2014_m"])
    flag = np.asarray(tp.version_flag(g.mhws_fes2022_m, g.mhws_fes2014_m), dtype=bool)
    print(f"gauges {len(g)}; the rule fires at {flag.sum()}: |FES2022 - gauge| "
          f"{(g.mhws_fes2022_m - g.mhws_gauge_m).abs()[flag].median():.2f} m, |FES2014 - gauge| "
          f"{(g.mhws_fes2014_m - g.mhws_gauge_m).abs()[flag].median():.2f} m (medians)", flush=True)


def peaks(check: GaugeCheck, dest: Path) -> None:
    hyd = np.load(check.out / "hydrographs.npy", mmap_mode="r")
    centre = hyd.shape[2] // 2
    rows = []
    for i in range(0, hyd.shape[0], 2000):
        block = np.asarray(hyd[i:i + 2000])
        filled = np.where(np.isfinite(block), block, -np.inf)
        k = filled.argmax(axis=2)
        mx = filled.max(axis=2)
        at0 = block[:, :, centre]
        for j in range(block.shape[0]):
            for r, rp in enumerate(stp.RPS):
                rows.append((i + j, rp, (k[j, r] - centre) / stp.STEPS_PER_HR, mx[j, r] - at0[j, r]))
    p = pd.DataFrame(rows, columns=["station", "rp", "offset_hr", "excess_m"])
    cons = list(tp.fes2022_model(check.cache / "fes2022").constituents)
    amp = np.abs(check.hc["fes2022"])
    form = (amp[:, cons.index("k1")] + amp[:, cons.index("o1")]) / (amp[:, cons.index("m2")] + amp[:, cons.index("s2")])
    p["form_factor"] = form[p.station]
    p["mhws_m"] = check.lev.mhws_m.to_numpy()[p.station]
    info = pd.read_parquet(check.out / "hydrograph_info.parquet").set_index(["station", "rp"])
    p = p.join(info[["shape_source", "half_window_hr"]], on=["station", "rp"])
    p["lon"], p["lat"] = check.lev.lon.to_numpy()[p.station], check.lev.lat.to_numpy()[p.station]
    p.to_parquet(dest / "peaks.parquet")
    off = p[p.offset_hr != 0]
    print(f"{len(off)} of {len(p)} hydrographs ({len(off) / len(p):.1%}) peak off time 0; by RP:", flush=True)
    print((p.assign(off=p.offset_hr != 0).groupby("rp").off.mean() * 100).round(1).to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("cap", "despike", "tc_mask", "fes_rule", "peaks"))
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.product_dir)
    dest = out / "open_guesses"
    dest.mkdir(exist_ok=True)
    check = GaugeCheck(out, Path(args.sources_dir), load_config(args.config, args.override))
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        if stage in ("cap", "tc_mask", "fes_rule"):
            {"cap": cap, "tc_mask": tc_mask, "fes_rule": fes_rule}[stage](check, dest, args.workers)
        else:
            {"despike": despike, "peaks": peaks}[stage](check, dest)


if __name__ == "__main__":
    main()
