"""Sensitivity of the storm-tide product to the rules set by estimate (TODO.md open questions).

Stages (each writes <product-dir>/open_guesses/<stage>*.parquet for the figure script):
  cap       GPD shape cap: levels at the ratio gauges (r left one out) against the gauge, and
            RP100 / RP1000 at the stations the cap touches, for caps none / 0.5 / 0.35 / 0.25 / 0.2
  despike   GTSM residual jumps per station: largest hourly jump and the number above 1-3 m,
            i.e. how many stations each (jump, count) threshold sends to COAST-RP
  tc_mask   an IBTrACS mask on the ETC fit (production has none): radius, time window, wind
            threshold, scored at the gauges with a COAST-RP TC part
  fes_rule  FES2022 / FES2014 MHWS at the gauges and at the stations the version rule flags
  peaks     hydrographs whose maximum is not at time 0: how many, how far, and the tide's
            diurnal inequality there
  tc_scale  the IBTrACS mask with COAST-RP_TC raised (scaled or shifted) against the double count;
            the raise is fitted per cyclone basin and applied leaving that basin out
  tc_jumps  every station with a TC part rebuilt masked with the TC part shifted by TC_JUMPS_SHIFT_M:
            RP100 differences between neighbouring stations, cap use and RP1000 against production
  r_variants  the gauge ratio r, left one out: production; fallback 1 or a covariate fit on
            GTSM's RP10 skew surge where fewer than MIN_NEIGHBOURS gauges are near; calibration
            without gauges whose GTSM skew surge is below R_MIN_SKEW_M; an additive offset
  long_record  every station rebuilt on GTSM-ERA5 1950-2024 (the 1950-1978 months joined to the
            product's 1979-2024 store): GPD shape, cap use and RP levels against the product, and
            each period's own skew-surge fit (an ERA5 homogeneity check)
  pooled_shape  on 1950-2024 with the IBTrACS mask (no shift): each station's GPD shape pooled
            over the stations within POOL_KM (summed profile likelihood, scale per station),
            its stability between 1979-2024 and 1950-2024, and the levels at the gauges
  tide_correction  FES tide amplitude against short harmonic fits of every usable coastal GESLA
            site: FES at the gauge, the current forcing (IDW of the stations), and both corrected by
            the other gauges within TIDE_KM (gauge left out)
  tile_interpolation  test 4 at the ratio gauges: the current forcing (IDW of the knn nearest
            stations' levels) against IDW of their skew-surge distributions (quantile average)
            convolved with the local FES tide at the gauge, each with and without the gauge tide
            correction (gauge left out)
  coastline_nodes  the production forcing with coastline nodes (src/coastline_nodes.py): IDW of the
            knn nearest stations and nodes together, as the tile seeds take them, for the tide at the
            tide-correction gauges and the levels at the ratio gauges (gauge left out of every factor)
  shape_noise  the stations' MLE GPD shapes against the shapes a light tail (shape 0, or the
            median shape) gives from each station's number of storms: is the heavy tail noise?
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
from scipy.stats import genpareto

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from gesla import DEDUP_KM  # noqa: E402
from gtsm_series import MISSING, extract_series, load_series, station_series  # noqa: E402
from level_product_gauges import GAUGE_RPS, GaugeCheck, scores, tc_hours  # noqa: E402
import skew_surge as ss  # noqa: E402
import storm_tide_product as stp  # noqa: E402
import tide_correction as tdc  # noqa: E402
import tide_prediction as tp  # noqa: E402

CAPS = {"none": None, "0.5": 0.5, "0.35 (production)": 0.35, "0.25": 0.25, "0.2": 0.2}
CAP_SAMPLE_MIN_SHAPE = 0.2  # stations whose MLE shape any tested cap would bind
TC_MASKS = {"no mask (production)": None,
            "300 km, +-24 h, 34 kt": {},
            "150 km, +-24 h, 34 kt": {"radius_km": 150.0},
            "500 km, +-24 h, 34 kt": {"radius_km": 500.0},
            "300 km, +-12 h, 34 kt": {"half_hr": 12},
            "300 km, +-48 h, 34 kt": {"half_hr": 48},
            "300 km, +-24 h, 64 kt": {"min_wind_kt": 64.0}}
JUMPS_M = (1.0, 2.0, 3.0)
ROWS_PER_READ = 400
NOISE_N_GRID = np.unique(np.geomspace(10, 400, 32).round().astype(int))
NOISE_REPS = 2000
NOISE_THRESHOLDS = (0.2, 0.35, 0.5)
CYCLE_COUNT_HOURS = 2 * 8766  # two years of the hourly tide set the cycles per year
TC_RAISES = {"tc_scale": (1.0, 1.1, 1.2, 1.3, 1.4), "tc_shift_m": (0.0, 0.1, 0.2, 0.3, 0.4)}
TC_JUMPS_SHIFT_M = 0.24  # the basin-left-out shift of tc_scale (0.18-0.25 m)
NEIGHBOUR_KM = 25.0
LONG_START = np.datetime64("1950-01-01T00:00")
EARLY_YEARS = (1950, 1978)  # the months before the product's store
R_MIN_SKEW_M = 0.2  # below it the ratio rests on a GTSM skew surge of a few decimetres
BASIN_KM = 300.0  # a gauge's basin: the most frequent IBTrACS basin of the track points this close
POOL_KM = (100.0, 300.0, 1000.0)
TIDE_KM = (10.0, 25.0, 50.0, 100.0, 200.0)
BLEND_QUANTILES = np.linspace(0.0, 1.0, 2001)
INTERP_CHUNK = 20
SHAPE_GRID = np.round(np.arange(-0.5, 0.805, 0.01), 2)


def _station_inputs(check: GaugeCheck, ks: np.ndarray) -> list[dict]:
    st = check.stations.iloc[ks]
    shapes = stp.station_tc_shapes(st, check.donors, check.shapes)
    todo = []
    for j, k in enumerate(ks):
        idx, s = check.stations.index[k], check.stations.iloc[k]
        a = check.lev.atlas.iloc[k]
        todo.append({"station": j, "gtsm_index": int(s.gtsm_index), "atlas": a, "coast_rp": check.coast[k],
                     "hc": check.hc[a][k], "mhws_m": float(check.mh[a][k]), "r": float(check.lev.r.iloc[k]),
                     "tc": check.tc_levels.loc[s.station_id].to_numpy() if s.station_id in check.tc_levels.index
                     else None, "tc_shape": shapes.get(idx)})
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
    for name, value in CAPS.items():
        res = check.table(check.levels(workers, shape_cap=value))
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


def _gauge_basins(check: GaugeCheck, gauges: pd.DataFrame) -> pd.Series:
    from sklearn.neighbors import BallTree
    tracks = pd.read_csv(check.cfg["storm_tide_product"]["ibtracs_path"], usecols=["BASIN", "LAT", "LON"],
                         skiprows=[1], keep_default_na=False, low_memory=False)
    tree = BallTree(np.radians(tracks[["LAT", "LON"]].to_numpy(dtype=float)), metric="haversine")
    near = tree.query_radius(np.radians(gauges[["lat", "lon"]].to_numpy()), r=BASIN_KM / stp.EARTH_RADIUS_KM)
    basin = tracks.BASIN.to_numpy()
    return pd.Series([pd.Series(basin[i]).mode().iloc[0] if len(i) else "none" for i in near], index=gauges.index)


def _loo_basin(levels: dict, gauge_rp100: pd.Series, basin: pd.Series, fit: np.ndarray) -> pd.Series:
    """Per gauge: the raise that zeroes the median RP100 bias at the `fit` gauges of the other basins."""
    values = np.array(sorted(levels))
    err = np.stack([levels[v]["new_loo_rp100_m"] - gauge_rp100 for v in values])
    raise_ = pd.Series(np.nan, index=gauge_rp100.index)
    for b in basin.unique():
        bias = np.median(err[:, fit & (basin != b).to_numpy()], axis=1)
        raise_[(basin == b).to_numpy()] = (float(np.interp(0.0, bias, values)) if np.all(np.diff(bias) > 0)
                                           else values[np.argmin(np.abs(bias))])
    return raise_


def tc_scale(check: GaugeCheck, dest: Path, workers: int) -> None:
    hours = check.tc_hours()
    base = check.table(check.levels(workers))
    gauges = pd.read_parquet(check.out / "gauges" / "gauges.parquet")
    basin = _gauge_basins(check, gauges.loc[base.index])
    tc_dom = (base.station_class == "tc_dominated").to_numpy()
    out = base[["station_class", "has_tc"] + [f"gauge_rp{rp}_m" for rp in GAUGE_RPS]
               + [f"new_loo_rp{rp}_m" for rp in GAUGE_RPS]].rename(
        columns={f"new_loo_rp{rp}_m": f"double_count_rp{rp}_m" for rp in GAUGE_RPS})
    out["basin"] = basin
    for key, values in TC_RAISES.items():
        levels = {}
        for v in values:
            levels[v] = check.table(check.levels(workers, hours, **{key: v}))
            out[f"masked_{key}_{v:g}_rp100_m"] = levels[v].new_loo_rp100_m
            print(f"  masked, {key} {v:g}", flush=True)
        values = np.array(sorted(levels))
        out[f"{key}_loo_basin"] = raise_ = _loo_basin(levels, base.gauge_rp100_m, basin, tc_dom)
        for rp in GAUGE_RPS:
            lev = np.stack([levels[v][f"new_loo_rp{rp}_m"] for v in values])
            out[f"masked_{key}_loo_basin_rp{rp}_m"] = [np.interp(x, values, lev[:, j]) for j, x in enumerate(raise_)]
    out.to_parquet(dest / "tc_scale.parquet")
    cols = {"double count (production)": "double_count"} | {f"masked + {k}, basin left out": f"masked_{k}_loo_basin"
                                                            for k in TC_RAISES}
    s = scores(out, cols)
    s.to_parquet(dest / "tc_scale_scores.parquet")
    print(out.groupby("basin").agg(gauges=("station_class", "size"), tc_dominated=("station_class",
          lambda c: (c == "tc_dominated").sum()), **{k: (f"{k}_loo_basin", "first") for k in TC_RAISES}).to_string(),
          flush=True)
    print(s[s.gauges.isin(["TC-dominated gauges", "stations with a TC part", "non-TC-dominated gauges"])
            & (s.rp == 100)].to_string(index=False), flush=True)


def tc_jumps(check: GaugeCheck, dest: Path, workers: int) -> None:
    from sklearn.neighbors import BallTree
    lev = check.lev
    ks = np.flatnonzero(lev.station_id.isin(check.tc_levels.index).to_numpy() & (lev.level_source == "product").to_numpy())
    st = check.stations.iloc[ks]
    hours = tc_hours(st, len(tp.hourly_times()), Path(check.cfg["storm_tide_product"]["ibtracs_path"]), tp.HOURLY_START)
    todo = [{**s, "tc_hours": hours.get(check.stations.index[k]), "tc_shift_m": TC_JUMPS_SHIFT_M}
            for s, k in zip(_station_inputs(check, ks), ks)]
    print(f"tc_jumps: {len(todo)} stations", flush=True)
    built = _build(check, todo, workers)
    out = lev.iloc[ks][["station_id", "lon", "lat", "gpd_shape_mle", "rp100_m", "rp1000_m", "coast_rp_rp1000_m"]].copy()
    out["masked_rp100_m"] = built.rp100_m.to_numpy()
    out["masked_rp1000_m"] = built.rp1000_m.to_numpy()
    out["masked_shape_mle"] = built.gpd_shape_mle.to_numpy()
    out.to_parquet(dest / "tc_jumps.parquet")
    xy = np.radians(out[["lat", "lon"]].to_numpy())
    d, i = BallTree(xy, metric="haversine").query(xy, k=2)
    pair = d[:, 1] * stp.EARTH_RADIUS_KM <= NEIGHBOUR_KM
    for name, col in (("double count (production)", "rp100_m"), ("masked + shift", "masked_rp100_m")):
        z = out[col].to_numpy()
        jump = np.abs(z - z[i[:, 1]])[pair]
        print(f"{name}: nearest-neighbour |dRP100| within {NEIGHBOUR_KM:.0f} km ({pair.sum()} stations): median "
              f"{np.median(jump):.3f}, p90 {np.quantile(jump, 0.9):.3f}, p99 {np.quantile(jump, 0.99):.3f}, "
              f"> 0.5 m {np.sum(jump > 0.5)}", flush=True)
    for name, col in (("production", "gpd_shape_mle"), ("masked", "masked_shape_mle")):
        print(f"{name}: MLE shape > {stp.SHAPE_CAP} at {(out[col] > stp.SHAPE_CAP).sum()} stations", flush=True)
    print("RP1000 - COAST-RP RP1000, p50 / p99 / max: " + "; ".join(
        f"{c}: {np.nanquantile(out[c] - out.coast_rp_rp1000_m, 0.5):.2f} / "
        f"{np.nanquantile(out[c] - out.coast_rp_rp1000_m, 0.99):.2f} / {np.nanmax(out[c] - out.coast_rp_rp1000_m):.2f}"
        for c in ("rp1000_m", "masked_rp1000_m")), flush=True)


def _idw(at: pd.DataFrame, cal: pd.DataFrame, value: str) -> tuple[float, int]:
    """IDW of `value` over `cal` within the production radius at one site, and the neighbour count."""
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(cal[["lat", "lon"]].to_numpy()), metric="haversine")
    i, d = tree.query_radius(np.radians(at[["lat", "lon"]].to_numpy()), r=stp.IDW_RADIUS_KM / stp.EARTH_RADIUS_KM,
                             return_distance=True)
    i, d = i[0], d[0]
    if len(i) < stp.MIN_NEIGHBOURS:
        return np.nan, len(i)
    w = 1.0 / np.maximum(d * stp.EARTH_RADIUS_KM, 1.0)
    return float(np.sum(w * cal[value].to_numpy()[i]) / w.sum()), len(i)


def r_variants(check: GaugeCheck, dest: Path, workers: int) -> None:
    gauges = pd.read_parquet(check.out / "gauges" / "gauges.parquet")
    gauges["offset_m"] = gauges.skew_gauge_m - gauges.skew_gtsmfes_m
    cal_all = gauges[gauges.calibrate & np.isfinite(gauges.ratio_fes)]
    names = check.meta.file_name.to_numpy()
    keys = {v: [] for v in ("fallback 1", "covariate fallback", f"calibration skew >= {R_MIN_SKEW_M:g} m", "additive")}
    neighbours = []
    for name in names:
        at, cal = gauges.loc[[name]], cal_all.drop(index=name, errors="ignore")
        r, n = _idw(at, cal, "ratio_fes")
        neighbours.append(n)
        fit = np.polyfit(np.log(cal.skew_gtsmfes_m), np.log(cal.ratio_fes), 1)
        cov = float(np.exp(np.polyval(fit, np.log(at.skew_gtsmfes_m.iloc[0]))))
        keys["fallback 1"].append({"r": min(r, stp.R_CAP) if n >= stp.MIN_NEIGHBOURS else 1.0})
        keys["covariate fallback"].append({"r": min(r if n >= stp.MIN_NEIGHBOURS else cov, stp.R_CAP)})
        big = cal[cal.skew_gtsmfes_m >= R_MIN_SKEW_M]
        rb, nb = _idw(at, big, "ratio_fes")
        keys[f"calibration skew >= {R_MIN_SKEW_M:g} m"].append(
            {"r": min(rb if nb >= stp.MIN_NEIGHBOURS else float(big.ratio_fes.median()), stp.R_CAP)})
        d, nd = _idw(at, cal, "offset_m")
        keys["additive"].append({"surge_offset_m": d if nd >= stp.MIN_NEIGHBOURS else float(cal.offset_m.median())})
    base_todo = check.todo
    out = check.table(check.levels(workers))
    out["neighbours"] = neighbours
    out = out.rename(columns={f"new_loo_rp{rp}_m": f"production_rp{rp}_m" for rp in GAUGE_RPS})
    for variant, extra in keys.items():
        check.todo = [{**s, **e} for s, e in zip(base_todo, extra)]
        loo = check.table(check.levels(workers))
        for rp in GAUGE_RPS:
            out[f"{variant}_rp{rp}_m"] = loo[f"new_loo_rp{rp}_m"]
        out[f"{variant}_value"] = [next(iter(e.values())) for e in extra]
        print(f"  {variant}", flush=True)
    check.todo = base_todo
    out.to_parquet(dest / "r_variants.parquet")
    rows = []
    sparse = out.neighbours < stp.MIN_NEIGHBOURS
    for label, sel in (("non-TC-dominated", out.station_class != "tc_dominated"),
                       (f"non-TC-dominated, < {stp.MIN_NEIGHBOURS} gauges within 500 km",
                        (out.station_class != "tc_dominated") & sparse)):
        for rp in (10, 100):
            for variant in ["production", *keys]:
                e = (out[f"{variant}_rp{rp}_m"] - out[f"gauge_rp{rp}_m"])[sel].dropna()
                rows.append({"gauges": label, "rp": rp, "variant": variant, "n": len(e), "median": e.median(),
                             "mae": e.abs().mean(), "p10": e.quantile(0.1), "p90": e.quantile(0.9)})
    s = pd.DataFrame(rows).round(3)
    s.to_parquet(dest / "r_variants_scores.parquet")
    print(f"covariate: log r = {fit[0]:.3f} log skew + {fit[1]:.3f}", flush=True)
    print(s.to_string(index=False), flush=True)


def _kept(skew: np.ndarray, hw_years: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, int]:
    sel = (hw_years >= lo) & (hw_years <= hi)
    cover = pd.Series(np.isfinite(skew[sel])).groupby(hw_years[sel]).mean()
    keep = cover.index[cover >= stp.YEAR_COVERAGE]
    return sel & np.isin(hw_years, keep) & np.isfinite(skew), len(keep)


def _period_fit(skew: np.ndarray, hw_years: np.ndarray, lo: int, hi: int) -> dict:
    ok, n_years = _kept(skew, hw_years, lo, hi)
    if n_years < 10:
        return {"n_years": n_years, "shape": np.nan, "skew_rp10_m": np.nan, "skew_rp100_m": np.nan}
    m = ss.surge_model(skew[ok], n_years)
    return {"n_years": n_years, "shape": m["shape"], "skew_rp10_m": ss.skew_level(m, 10),
            "skew_rp100_m": ss.skew_level(m, 100)}


def _long_stations(chunk, stores, bases, n_hours):
    loaded = []
    for store in stores:
        st_tab, time, series = load_series(Path(store))
        idx = ((time.to_numpy().astype("datetime64[h]") - LONG_START) // np.timedelta64(1, "h")).astype(int)
        loaded.append((dict(zip(st_tab.gtsm_index, range(len(st_tab)))), idx, series))
    years = (LONG_START + np.arange(n_hours).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    for atlas in bases:
        sts = [s for s in chunk if s["atlas"] == atlas]
        if not sts:
            continue
        b = [np.load(p, mmap_mode="r") for p in bases[atlas]]
        hc = np.vstack([s["hc"] for s in sts])
        th_all, t10_all = tp.predict(b[0], b[1], hc), tp.predict(b[2], b[3], hc)
        for j, st in enumerate(sts):
            g = np.full(n_hours, np.nan)
            for pos, idx, series in loaded:
                g[idx] = station_series(series, pos[st["gtsm_index"]])
            yield st, g, th_all[:, j], t10_all[:, j], years


def _product_row(st, g, th, t10, years) -> dict | None:
    try:
        return stp.station_product(st, g, th, t10, years)[0]
    except Exception as err:  # one bad station must not stop the pool; it is reported and skipped
        print(f"  skipped station {st['station']}: {type(err).__name__}: {err}", flush=True)
        return None


def _long_task(args) -> list[dict]:
    out = []
    for st, g, th, t10, years in _long_stations(*args):
        row = _product_row(st, g, th, t10, years)
        if row is None:
            continue
        hw, h = ss.high_waters(th, ss.HW_PROMINENCE)
        skew = ss.skew_surges(ss.despike(g) + th, hw, h)
        for name, (lo, hi) in (("early", EARLY_YEARS), ("late", (EARLY_YEARS[1] + 1, int(years[-1])))):
            row.update({f"{name}_{k}": v for k, v in _period_fit(skew, years[hw], lo, hi).items()})
        out.append(row)
    return out


def _long_inputs(check: GaugeCheck, dest: Path, workers: int) -> tuple[tuple[str, str], dict, int]:
    """Surge stores (1950-1978, the product's 1979-2024) and FES bases on the 1950-2024 hourly axis."""
    root = dest / "long_record"
    early = root / "gtsm_surge_1950_1978"
    main_store = check.cache / "gtsm_surge_stations"
    if not (early / "series.npy").exists():
        st_tab, _, _ = load_series(main_store)
        extract_series(Path(check.cfg["storm_tide_product"]["gtsm_dir"]), st_tab.gtsm_index.to_numpy(), early,
                       workers, years=EARLY_YEARS)
    times = np.arange(LONG_START, tp.HOURLY_END, np.timedelta64(1, "h"))
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    models = {"fes2022": tp.fes2022_model(check.cache / "fes2022"),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    bases = {a: (*tp.cached_basis(models[a], times, root / f"{a}_hourly_re.npy", root / f"{a}_hourly_im.npy"),
                 check.cache / f"{a}_nodal_re.npy", check.cache / f"{a}_nodal_im.npy") for a in models}
    return (str(early), str(main_store)), {a: tuple(str(p) for p in v) for a, v in bases.items()}, len(times)


def _run_long(task, todo: list[dict], stores, bases, n_hours: int, workers: int) -> list:
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(todo[i:i + stp.TASK_SIZE], stores, bases, n_hours) for i in range(0, len(todo), stp.TASK_SIZE)]
    out = []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(task, tasks)):
            out += res
            if k % 40 == 0:
                print(f"  {min((k + 1) * stp.TASK_SIZE, len(todo))}/{len(todo)} stations", flush=True)
    return out


def long_record(check: GaugeCheck, dest: Path, workers: int) -> None:
    stores, bases, n_hours = _long_inputs(check, dest, workers)
    root = dest / "long_record"
    lev = check.lev
    ks = np.flatnonzero((lev.level_source == "product").to_numpy())
    built = pd.DataFrame(_run_long(_long_task, _station_inputs(check, ks), stores, bases, n_hours, workers)) \
        .set_index("station").sort_index()
    keep = ["station_id", "lon", "lat", "gpd_shape_mle", "gpd_shape", "rp100_m", "rp1000_m", "etc_rp100_m",
            "etc_rp1000_m"]
    out = lev.iloc[ks[built.index]][keep].copy()
    out["tc_part"] = out.station_id.isin(check.tc_levels.index).to_numpy()
    for c in ("gpd_shape_mle", "gpd_shape", "n_years", "rp100_m", "rp1000_m", "etc_rp100_m", "etc_rp1000_m"):
        out[f"long_{c}"] = built[c].to_numpy()
    for c in built.columns:
        if c.startswith(("early_", "late_")):
            out[c] = built[c].to_numpy()
    out.to_parquet(root / "long_record.parquet")
    for label, sel in (("all", np.ones(len(out), bool)), ("TC part", out.tc_part.to_numpy()),
                       ("no TC part", ~out.tc_part.to_numpy())):
        s = out[sel]
        print(f"{label} ({len(s)}): MLE shape median {s.gpd_shape_mle.median():.3f} -> {s.long_gpd_shape_mle.median():.3f}"
              f", > 0.35 {(s.gpd_shape_mle > 0.35).sum()} -> {(s.long_gpd_shape_mle > 0.35).sum()}; "
              f"RP100 change p10/p50/p90 {np.nanquantile(s.long_rp100_m - s.rp100_m, [0.1, 0.5, 0.9]).round(3)}; "
              f"RP1000 change {np.nanquantile(s.long_rp1000_m - s.rp1000_m, [0.1, 0.5, 0.9]).round(3)}; "
              f"early/late skew RP10 median {np.nanmedian(s.early_skew_rp10_m / s.late_skew_rp10_m):.3f}",
              flush=True)


def _profile_loglik(x: np.ndarray) -> np.ndarray:
    """GPD log-likelihood of excesses x at each SHAPE_GRID shape, maximised over the scale."""
    xi = SHAPE_GRID[:, None]
    lo = np.log(np.maximum(np.where(SHAPE_GRID < 0, -SHAPE_GRID * x.max() * (1 + 1e-9), 0.0), 1e-4 * x.mean()))
    hi = np.full(len(SHAPE_GRID), np.log(20.0 * x.max()))

    def loglik(log_scale):
        s = np.exp(log_scale)[:, None]
        z = 1.0 + xi * x[None, :] / s
        with np.errstate(divide="ignore", invalid="ignore"):
            core = np.where(xi == 0, x[None, :] / s, (1.0 + 1.0 / xi) * np.log(z))
            v = -len(x) * np.log(s[:, 0]) - core.sum(axis=1)
        return np.where((z > 0).all(axis=1) & np.isfinite(v), v, -np.inf)

    golden = (np.sqrt(5.0) - 1.0) / 2.0  # the profile is unimodal in log scale
    for _ in range(60):
        c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
        left = loglik(c) >= loglik(d)
        hi, lo = np.where(left, d, hi), np.where(left, lo, c)
    return loglik((lo + hi) / 2.0)


def _pool_task(args) -> list:
    out = []
    for st, g, th, t10, years in _long_stations(*args):
        row = _product_row(st, g, th, t10, years)
        if row is None:
            continue
        hw, h = ss.high_waters(th, ss.HW_PROMINENCE)
        skew = ss.skew_surges(ss.despike(g) + th, hw, h)
        masked = np.isin(hw, st["tc_hours"]) if st.get("tc_hours") is not None else np.zeros(len(hw), bool)
        prof = {}
        for name, lo in (("full", EARLY_YEARS[0]), ("late", EARLY_YEARS[1] + 1)):
            ok, _ = _kept(skew, years[hw], lo, int(years[-1]))
            prof[name] = _profile_loglik(ss.cluster_excesses(skew[ok & ~masked])[1])
        out.append((row, prof))
    return out


def _pooled_shapes(lonlat: np.ndarray, prof: np.ndarray, query: np.ndarray, radius_km: float) -> np.ndarray:
    from sklearn.neighbors import BallTree
    near = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine").query_radius(
        np.radians(query[:, ::-1]), r=radius_km / stp.EARTH_RADIUS_KM)
    return np.array([SHAPE_GRID[np.argmax(prof[j].sum(axis=0))] for j in near])


def _long_tc_hours(check: GaugeCheck, stations: pd.DataFrame, n_hours: int) -> dict:
    with_tc = stations[stations.station_id.isin(check.tc_levels.index)]
    return tc_hours(with_tc, n_hours, Path(check.cfg["storm_tide_product"]["ibtracs_path"]), LONG_START)


def pooled_shape(check: GaugeCheck, dest: Path, workers: int) -> None:
    stores, bases, n_hours = _long_inputs(check, dest, workers)
    root = dest / "pooled_shape"
    root.mkdir(exist_ok=True)
    lev = check.lev
    ks = np.flatnonzero((lev.level_source == "product").to_numpy())
    if not (root / "profiles.npz").exists():
        hours = _long_tc_hours(check, check.stations.iloc[ks], n_hours)
        todo = [{**s, "tc_hours": hours.get(check.stations.index[k])}
                for s, k in zip(_station_inputs(check, ks), ks)]
        res = _run_long(_pool_task, todo, stores, bases, n_hours, workers)
        built = pd.DataFrame([r for r, _ in res]).set_index("station")
        np.savez(root / "profiles.npz", station=built.index.to_numpy(),
                 full=np.vstack([p["full"] for _, p in res]), late=np.vstack([p["late"] for _, p in res]))
        built.to_parquet(root / "masked_rows.parquet")
    prof = np.load(root / "profiles.npz")
    built = pd.read_parquet(root / "masked_rows.parquet").loc[prof["station"]]
    out = lev.iloc[ks[prof["station"]]][["station_id", "lon", "lat", "gpd_shape_mle", "rp100_m", "rp1000_m"]].copy()
    out["tc_part"] = out.station_id.isin(check.tc_levels.index).to_numpy()
    for c in ("gpd_shape_mle", "rp100_m", "rp1000_m", "tc_masked_share"):
        out[f"masked_{c}"] = built[c].to_numpy()
    lonlat = out[["lon", "lat"]].to_numpy(dtype=float)
    for name in ("full", "late"):
        out[f"shape_0km_{name}"] = SHAPE_GRID[np.argmax(prof[name], axis=1)]
        for km in POOL_KM:
            out[f"shape_{km:g}km_{name}"] = _pooled_shapes(lonlat, prof[name], lonlat, km)
    out.to_parquet(root / "stations.parquet")
    for label, sel in (("TC part", out.tc_part.to_numpy()), ("no TC part", ~out.tc_part.to_numpy())):
        s = out[sel]
        for km in (0.0, *POOL_KM):
            a, b = s[f"shape_{km:g}km_late"], s[f"shape_{km:g}km_full"]
            print(f"{label} ({len(s)}), {km:g} km: shape p10/p50/p90 {np.quantile(b, [0.1, 0.5, 0.9]).round(2)}, "
                  f"> 0.35 late {(a > 0.35).sum()} / full {(b > 0.35).sum()} / both {((a > 0.35) & (b > 0.35)).sum()}"
                  f"; |full - late| p50 {np.median(np.abs(b - a)):.3f} p90 {np.quantile(np.abs(b - a), 0.9):.3f}",
                  flush=True)

    gauge_st = check.stations.loc[check.meta.station.to_numpy()]
    hours = _long_tc_hours(check, check.picked.drop_duplicates(), n_hours)
    variants = {"75 yr, double count, cap 0.35": {}, "75 yr, masked, cap 0.35": {"mask": True},
                "75 yr, masked, no cap": {"mask": True, "shape_cap": None}}
    for km in POOL_KM:
        shapes = _pooled_shapes(lonlat, prof["full"], gauge_st[["lon", "lat"]].to_numpy(dtype=float), km)
        variants[f"75 yr, masked, pooled {km:g} km, no cap"] = {"mask": True, "shape_cap": None, "pooled_shape": shapes}
    table = check.table(check.levels(workers))
    table = table.rename(columns={f"new_loo_rp{rp}_m": f"production_rp{rp}_m" for rp in GAUGE_RPS})
    cols = {"46 yr, double count, cap 0.35 (production)": "production"}
    for k, (name, v) in enumerate(variants.items()):
        todo = []
        for j, s in enumerate(check.todo):
            extra = {key: (val[j] if key == "pooled_shape" else val) for key, val in v.items() if key != "mask"}
            todo.append({**s, **extra, "tc_hours": hours.get(s["station_index"]) if v.get("mask") else None})
        rows = pd.DataFrame(_run_long(_long_task, todo, stores, bases, n_hours, workers)).set_index("station")
        rows.index = check.meta.file_name.to_numpy()[rows.index]
        for rp in GAUGE_RPS:
            table[f"v{k}_rp{rp}_m"] = rows[f"rp{rp}_m"].reindex(table.index)
        if "pooled_shape" in v:
            table[f"v{k}_shape"] = pd.Series(v["pooled_shape"], index=check.meta.file_name.to_numpy())
        cols[name] = f"v{k}"
        print(f"  {name}", flush=True)
    table.to_parquet(root / "gauges.parquet")
    s = scores(table, cols)
    s.to_parquet(root / "scores.parquet")
    print(s[s.gauges.isin(["TC-dominated gauges", "non-TC-dominated gauges"]) & s.rp.isin([10, 100])]
          .to_string(index=False), flush=True)


def tide_correction(check: GaugeCheck, dest: Path, workers: int) -> None:
    from sklearn.neighbors import BallTree
    root = dest / "tide_correction"
    root.mkdir(exist_ok=True)
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    models = {"fes2022": tp.fes2022_model(check.cache / "fes2022"),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    table = check.out / "gauges" / "tide_gauges.parquet"
    if not table.exists():
        table = root / "tide_gauges.parquet"
        if not table.exists():
            tdc.gauge_table(Path(check.cfg["storm_tide_product"]["gesla_dir"]), models, workers).to_parquet(table)
    gt = pd.read_parquet(table)
    gt = gt[gt.usable].reset_index(drop=True)
    lev = check.lev[check.lev.level_source == "product"]
    st_lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    g_lonlat = gt[["lon", "lat"]].to_numpy(dtype=float)
    d, i = BallTree(np.radians(st_lonlat[:, ::-1]), metric="haversine").query(np.radians(g_lonlat[:, ::-1]),
                                                                            k=check.cfg["simulation"]["flooding"]["knn"])
    w = np.maximum(d * stp.EARTH_RADIUS_KM, 1e-3) ** -float(check.cfg["simulation"]["flooding"]["idw_power"])
    w /= w.sum(axis=1, keepdims=True)
    atlas = lev.atlas.to_numpy()[i[:, 0]]  # a gauge takes the atlas of its nearest station
    gt["atlas"] = atlas
    gt["near_km"] = d[:, 0] * stp.EARTH_RADIUS_KM
    gt["amp_gauge_m"] = tdc.tide_amplitude(gt)
    gt["amp_fes_m"] = np.where(atlas == "fes2014", gt.amp_fes2014_m, gt.amp_fes2022_m)
    lr = np.where(atlas == "fes2014", gt.log_ratio_fes2014, gt.log_ratio_fes2022)
    c = np.load(check.cache / "fes_constants.npz")
    st_all = check.stations[["lon", "lat"]].to_numpy(dtype=float)
    pos = check.stations.index.get_indexer(lev.index)
    amp = {a: tdc.fes_amplitude(tp.fill_masked(c[a], st_all), list(m.constituents))[pos] for a, m in models.items()}
    amp_station = np.where(lev.atlas.to_numpy() == "fes2014", amp["fes2014"], amp["fes2022"])
    gt["amp_station_idw_m"] = (amp_station[i] * w).sum(axis=1)
    for km in TIDE_KM:
        f, n = tdc.correction_factors(g_lonlat, g_lonlat, lr, km, exclude_km=DEDUP_KM)
        gt[f"amp_corrected_{km:g}km_m"], gt[f"neighbours_{km:g}km"] = gt.amp_fes_m * f, n
    # the current forcing with corrected stations; the gauge (and its site) is left out of every factor
    g_tree = BallTree(np.radians(g_lonlat[:, ::-1]), metric="haversine")
    same_site = g_tree.query_radius(np.radians(g_lonlat[:, ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)
    used = np.unique(i)
    near = dict(zip(used, g_tree.query_radius(np.radians(st_lonlat[used, ::-1]),
                                              r=tdc.CORRECTION_KM / stp.EARTH_RADIUS_KM)))
    factor = np.ones(i.shape)
    for j in range(len(gt)):
        for m, k in enumerate(i[j]):
            keep = np.setdiff1d(near[k], same_site[j])
            keep = keep[np.isfinite(lr[keep])]
            if len(keep):
                factor[j, m] = np.exp(np.median(lr[keep]))
    gt["amp_station_idw_corrected_m"] = (amp_station[i] * factor * w).sum(axis=1)
    gt.to_parquet(root / "gauges.parquet")
    cols = {"FES at the gauge": "amp_fes_m", "station IDW (current forcing)": "amp_station_idw_m",
            "station IDW, stations corrected": "amp_station_idw_corrected_m"} | {
        f"FES at the gauge, corrected within {km:g} km": f"amp_corrected_{km:g}km_m" for km in TIDE_KM}
    rows = []
    for label, sel in (("all", np.ones(len(gt), bool)), ("gauge tide > 2 m", gt.amp_gauge_m.to_numpy() > 2),
                       ("> 10 km from a station", gt.near_km.to_numpy() > 10)):
        for name, col in cols.items():
            e, rel = (gt[col] - gt.amp_gauge_m)[sel], (gt[col] / gt.amp_gauge_m - 1)[sel]
            rows.append({"gauges": label, "tide": name, "n": int(sel.sum()), "median_m": e.median(),
                         "mae_m": e.abs().mean(), "off_25pct": int((rel.abs() > 0.25).sum())})
    s = pd.DataFrame(rows).round(3)
    s.to_parquet(root / "scores.parquet")
    print(s.to_string(index=False), flush=True)


def _blend(models: list[dict], w: np.ndarray) -> dict:
    out = {"sorted": np.sum([wk * np.quantile(m["sorted"], BLEND_QUANTILES) for m, wk in zip(models, w)], axis=0)}
    for k in ("u", "p_u", "shape", "scale", "theta"):
        out[k] = float(np.sum([wk * m[k] for m, wk in zip(models, w)]))
    return out


def _interp_task(args) -> list[dict]:
    chunk, store, bases = args
    surge_st, surge_time, surge = load_series(Path(store))
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = tp.on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    loaded = {a: [np.load(p, mmap_mode="r") for p in bases[a]] for a in bases}
    n_hours = loaded["fes2022"][0].shape[0]
    years = tp.hour_years(np.arange(n_hours))
    out = []
    for gauge in chunk:
        models, raws, station_levels, station_levels_corr, tcs = [], [], [], [], []
        for st, f in zip(gauge["stations"], gauge["station_factors"]):
            b = loaded[st["atlas"]]
            th, t10 = tp.predict(b[0], b[1], st["hc"][None, :])[:, 0], tp.predict(b[2], b[3], st["hc"][None, :])[:, 0]
            g = np.full(n_hours, np.nan)
            g[surge_idx] = station_series(surge, pos[st["gtsm_index"]])
            _, raw, corrected, _ = stp.station_surge_model(st, g, th, years)
            raws.append(raw)
            models.append(corrected)
            tcs.append(st["tc"])
            station_levels.append(stp.rp_levels(raw, corrected, t10, st["tc"]))
            station_levels_corr.append(stp.rp_levels(raw, corrected, t10 * f, st["tc"]) if f != 1.0
                                       else station_levels[-1])
        w = gauge["weights"]
        row = {"file_name": gauge["file_name"]}
        for name, levels in (("current", station_levels), ("current_tide_corrected", station_levels_corr)):
            for rp in stp.RPS:
                row[f"{name}_rp{rp}_m"] = float(np.sum([wk * lv[f"rp{rp}_m"] for lv, wk in zip(levels, w)]))
        has_tc = [t is not None for t in tcs]
        tc = None
        if has_tc[0]:
            wt = w * np.array(has_tc)
            tc = np.sum([wk * t for t, wk in zip(tcs, wt) if t is not None], axis=0) / wt.sum()
        raw_b, corr_b = _blend(raws, w), _blend(models, w)
        b = loaded[gauge["atlas"]]
        t10 = tp.predict(b[2], b[3], gauge["hc"][None, :])[:, 0]
        for name, factor in (("decomposed", 1.0), ("decomposed_tide_corrected", gauge["factor"])):
            lv = stp.rp_levels(raw_b, corr_b, t10 * factor, tc)
            for rp in stp.RPS:
                row[f"{name}_rp{rp}_m"] = lv[f"rp{rp}_m"]
        out.append(row)
    return out


def tile_interpolation(check: GaugeCheck, dest: Path, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.neighbors import BallTree
    root = dest / "tile_interpolation"
    root.mkdir(exist_ok=True)
    tide_gauges = check.out / "gauges" / "tide_gauges.parquet"
    if not tide_gauges.exists():
        tide_gauges = dest / "tide_correction" / "tide_gauges.parquet"
    gt = pd.read_parquet(tide_gauges)
    gt = gt[gt.usable].reset_index(drop=True)
    gt_lonlat = gt[["lon", "lat"]].to_numpy(dtype=float)
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    models = {"fes2022": tp.fes2022_model(check.cache / "fes2022"),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    lev = check.lev[check.lev.level_source == "product"]
    k_near, power = check.cfg["simulation"]["flooding"]["knn"], float(check.cfg["simulation"]["flooding"]["idw_power"])
    st_lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    meta = check.meta
    gauges = pd.read_parquet(check.out / "gauges" / "gauges.parquet").loc[meta.file_name]
    g_lonlat = gauges[["lon", "lat"]].to_numpy(dtype=float)
    d, i = BallTree(np.radians(st_lonlat[:, ::-1]), metric="haversine").query(np.radians(g_lonlat[:, ::-1]), k=k_near)
    w = np.maximum(d * stp.EARTH_RADIUS_KM, 1e-3) ** -power
    w /= w.sum(axis=1, keepdims=True)
    pos = check.stations.index.get_indexer(lev.index)
    todo_all = _station_inputs(check, pos)
    atlas = lev.atlas.to_numpy()
    hc_gauge = {a: tp.fill_masked(tp.grouped_constants(m, g_lonlat), g_lonlat) for a, m in models.items()}
    for a, m in models.items():
        hc_gauge[a][:, tp.seasonal_columns(m)] = 0.0
    lr = {a: gt[f"log_ratio_{a}"].to_numpy() for a in models}
    gt_tree = BallTree(np.radians(gt_lonlat[:, ::-1]), metric="haversine")
    tasks = []
    for j, name in enumerate(meta.file_name):
        a = atlas[i[j, 0]]
        f, _ = tdc.correction_factors(g_lonlat[[j]], gt_lonlat, lr[a], exclude_km=DEDUP_KM)
        own = gt_tree.query_radius(np.radians(g_lonlat[[j], ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)[0]
        sf = []
        for k in i[j]:
            keep = np.ones(len(gt), bool)
            keep[own] = False
            fk, _ = tdc.correction_factors(st_lonlat[[k]], gt_lonlat[keep], lr[atlas[k]][keep])
            sf.append(float(fk[0]))
        tasks.append({"file_name": name, "atlas": a, "hc": hc_gauge[a][j], "factor": float(f[0]), "weights": w[j],
                      "stations": [todo_all[k] for k in i[j]], "station_factors": sf})
    bases = {a: tuple(str(check.cache / f"{a}_{kind}.npy") for kind in ("hourly_re", "hourly_im", "nodal_re", "nodal_im"))
             for a in models}
    chunks = [(tasks[n:n + INTERP_CHUNK], str(check.cache / "gtsm_surge_stations"), bases)
              for n in range(0, len(tasks), INTERP_CHUNK)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(_interp_task, chunks)):
            rows += res
            print(f"  {min((k + 1) * INTERP_CHUNK, len(tasks))}/{len(tasks)} gauges", flush=True)
    built = pd.DataFrame(rows).set_index("file_name")
    loo = built[[f"current_rp{rp}_m" for rp in stp.RPS]].rename(columns=lambda c: c.replace("current_", ""))
    for rp in stp.RPS:
        loo[f"etc_raw_rp{rp}_m"] = np.nan
    table = check.table(loo)
    variants = {"current: IDW of station levels": "current", "current, station tides corrected": "current_tide_corrected",
                "decomposed: IDW surge + local FES tide": "decomposed",
                "decomposed, local tide corrected": "decomposed_tide_corrected"}
    for col in variants.values():
        for rp in GAUGE_RPS:
            table[f"{col}_rp{rp}_m"] = built[f"{col}_rp{rp}_m"].reindex(table.index)
    table["tide_factor"] = [t["factor"] for t in tasks]
    table.to_parquet(root / "gauges.parquet")
    s = scores(table, variants)
    s.to_parquet(root / "scores.parquet")
    print(s[s.gauges.isin(["all", "non-TC-dominated gauges"]) & s.rp.isin([2, 100])].to_string(index=False), flush=True)


def _node_eval_task(args) -> list[dict]:
    chunk, store, bases = args
    surge_st, surge_time, surge = load_series(Path(store))
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = tp.on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    loaded = {a: [np.load(p, mmap_mode="r") for p in bases[a]] for a in bases}
    n_hours = loaded["fes2022"][0].shape[0]
    years = tp.hour_years(np.arange(n_hours))
    built: dict = {}

    def station(st: dict):
        if st["key"] not in built:
            b = loaded[st["atlas"]]
            th = tp.predict(b[0], b[1], st["hc"][None, :])[:, 0]
            g = np.full(n_hours, np.nan)
            g[surge_idx] = station_series(surge, pos[st["gtsm_index"]])
            _, raw, corrected, _ = stp.station_surge_model(st, g, th, years)
            built[st["key"]] = (raw, corrected, tp.predict(b[2], b[3], st["hc"][None, :])[:, 0])
        return built[st["key"]]

    out = []
    for gauge in chunk:
        levels = []
        for e in gauge["entities"]:
            if e["kind"] == "station":
                raw, corrected, t10 = station(e["st"])
                levels.append(stp.rp_levels(raw, corrected, t10 * e["factor"], e["st"]["tc"]))
            else:
                corr = [station(st)[1] for st in e["stations"]]
                tcs = [st["tc"] for st in e["stations"]]
                tc = None
                if tcs[0] is not None:
                    wt = e["weights"] * np.array([t is not None for t in tcs])
                    tc = np.sum([wk * t for t, wk in zip(tcs, wt) if t is not None], axis=0) / wt.sum()
                blend = _blend(corr, e["weights"])
                b = loaded[e["atlas"]]
                t10 = tp.predict(b[2], b[3], e["hc"][None, :])[:, 0] * e["factor"]
                levels.append(stp.rp_levels(blend, blend, t10, tc))
        row = {"file_name": gauge["file_name"], "node_weight": gauge["node_weight"]}
        for rp in stp.RPS:
            row[f"nodes_rp{rp}_m"] = float(np.sum([wk * lv[f"rp{rp}_m"] for lv, wk in zip(levels, gauge["weights"])]))
        out.append(row)
    return out


def coastline_nodes(check: GaugeCheck, dest: Path, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.neighbors import BallTree
    import coastline_nodes as cn
    root = dest / "coastline_nodes"
    root.mkdir(exist_ok=True)
    node_file = check.cache / "coastline_nodes.parquet"
    if not node_file.exists():
        node_file = root / "coastline_nodes.parquet"
        if not node_file.exists():
            catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
            cn.coastline_nodes(Path(catalog.get_source("deltadtm_mask").path).parent,
                               check.cfg["tile_generation"]["ocean_code"], check.cfg["tile_generation"]["river_code"],
                               workers).to_parquet(node_file)
    nodes = pd.read_parquet(node_file)
    n_lonlat = nodes[["lon", "lat"]].to_numpy(dtype=float)
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    models = {"fes2022": tp.fes2022_model(check.cache / "fes2022"),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    tg = pd.read_parquet(dest / "tide_correction" / "gauges.parquet")
    k_near, power = check.cfg["simulation"]["flooding"]["knn"], float(check.cfg["simulation"]["flooding"]["idw_power"])
    lev = check.lev[check.lev.level_source == "product"]
    st_lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    st_atlas = lev.atlas.to_numpy()
    st_tree = BallTree(np.radians(st_lonlat[:, ::-1]), metric="haversine")
    meta = check.meta
    rg = pd.read_parquet(check.out / "gauges" / "gauges.parquet").loc[meta.file_name]
    tg_lonlat, rg_lonlat = tg[["lon", "lat"]].to_numpy(dtype=float), rg[["lon", "lat"]].to_numpy(dtype=float)
    # the seed at a point: IDW over the knn nearest of the stations and nodes together
    both = np.vstack([st_lonlat, n_lonlat])
    both_tree = BallTree(np.radians(both[:, ::-1]), metric="haversine")
    seeds = {}
    for name, q in (("tide", tg_lonlat), ("ratio", rg_lonlat)):
        d, i = both_tree.query(np.radians(q[:, ::-1]), k=k_near)
        w = np.maximum(d * stp.EARTH_RADIUS_KM, 1e-3) ** -power
        seeds[name] = (i, w / w.sum(axis=1, keepdims=True))
    used = np.unique(np.concatenate([seeds[n][0].ravel() for n in seeds]))
    used_nodes = used[used >= len(st_lonlat)] - len(st_lonlat)
    print(f"{len(nodes)} coastline nodes, {len(used_nodes)} of them seed a gauge", flush=True)
    node_atlas = st_atlas[st_tree.query(np.radians(n_lonlat[used_nodes][:, ::-1]), k=1)[1][:, 0]]
    node_hc = {}
    for a, m in models.items():
        sel = node_atlas == a
        if sel.any():
            h = tp.fill_masked(tp.grouped_constants(m, n_lonlat[used_nodes[sel]]), n_lonlat[used_nodes[sel]])
            h[:, tp.seasonal_columns(m)] = 0.0
            node_hc.update(dict(zip(used_nodes[sel], h)))
    node_atlas = dict(zip(used_nodes, node_atlas))
    # tide gauges for the correction; each evaluated gauge's own site is left out of every factor
    gt = pd.read_parquet(check.out / "gauges" / "tide_gauges.parquet" if (check.out / "gauges" / "tide_gauges.parquet").exists()
                         else dest / "tide_correction" / "tide_gauges.parquet")
    gt = gt[gt.usable].reset_index(drop=True)
    gt_lonlat = gt[["lon", "lat"]].to_numpy(dtype=float)
    lr = {a: gt[f"log_ratio_{a}"].to_numpy() for a in models}
    gt_tree = BallTree(np.radians(gt_lonlat[:, ::-1]), metric="haversine")

    def entity_lonlat_atlas(e: int):
        return (st_lonlat[e], st_atlas[e]) if e < len(st_lonlat) else (n_lonlat[e - len(st_lonlat)],
                                                                        node_atlas[e - len(st_lonlat)])

    def factor(e: int, own: np.ndarray) -> float:
        ll, a = entity_lonlat_atlas(e)
        near = gt_tree.query_radius(np.radians(ll[None, ::-1]), r=tdc.CORRECTION_KM / stp.EARTH_RADIUS_KM)[0]
        keep = np.setdiff1d(near, own)
        keep = keep[np.isfinite(lr[a][keep])]
        return float(np.exp(np.median(lr[a][keep]))) if len(keep) else 1.0

    c = np.load(check.cache / "fes_constants.npz")
    st_all = check.stations[["lon", "lat"]].to_numpy(dtype=float)
    spos = check.stations.index.get_indexer(lev.index)
    amp_st = {a: tdc.fes_amplitude(tp.fill_masked(c[a], st_all), list(m.constituents))[spos] for a, m in models.items()}
    amp_station = np.where(st_atlas == "fes2014", amp_st["fes2014"], amp_st["fes2022"])
    amp_node = {n: float(tdc.fes_amplitude(node_hc[n][None, :], list(models[node_atlas[n]].constituents))[0])
                for n in used_nodes}
    i, w = seeds["tide"]
    own_tg = gt_tree.query_radius(np.radians(tg_lonlat[:, ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)
    amp = np.array([[amp_station[e] if e < len(st_lonlat) else amp_node[e - len(st_lonlat)] for e in row] for row in i])
    f = np.array([[factor(e, own_tg[j]) for e in row] for j, row in enumerate(i)])
    tg["amp_nodes_m"] = (amp * w).sum(axis=1)
    tg["amp_nodes_corrected_m"] = (amp * f * w).sum(axis=1)
    tg["node_weight"] = (w * (i >= len(st_lonlat))).sum(axis=1)
    tg.to_parquet(root / "tide_gauges.parquet")
    cols = {"stations (current forcing)": "amp_station_idw_m", "stations, corrected": "amp_station_idw_corrected_m",
            "stations + nodes": "amp_nodes_m", "stations + nodes, corrected": "amp_nodes_corrected_m",
            "FES at the gauge, corrected within 25 km": "amp_corrected_25km_m"}
    rows = []
    for label, sel in (("all", np.ones(len(tg), bool)), ("gauge tide > 2 m", tg.amp_gauge_m.to_numpy() > 2),
                       ("> 10 km from a station", tg.near_km.to_numpy() > 10)):
        for name, col in cols.items():
            e = (tg[col] - tg.amp_gauge_m)[sel]
            rows.append({"gauges": label, "tide": name, "n": int(sel.sum()), "median_m": e.median(),
                         "mae_m": e.abs().mean()})
    s = pd.DataFrame(rows).round(3)
    s.to_parquet(root / "tide_scores.parquet")
    print(s.to_string(index=False), flush=True)

    todo_all = _station_inputs(check, check.stations.index.get_indexer(lev.index))
    for k, st in enumerate(todo_all):
        st["key"] = k
    i, w = seeds["ratio"]
    own_rg = gt_tree.query_radius(np.radians(rg_lonlat[:, ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)
    tasks = []
    for j, name in enumerate(meta.file_name):
        entities = []
        for e in i[j]:
            if e < len(st_lonlat):
                entities.append({"kind": "station", "st": todo_all[e], "factor": factor(e, own_rg[j])})
                continue
            n = e - len(st_lonlat)
            dn, nb = st_tree.query(np.radians(n_lonlat[[n], ::-1]), k=k_near)
            wn = np.maximum(dn[0] * stp.EARTH_RADIUS_KM, 1e-3) ** -power
            entities.append({"kind": "node", "hc": node_hc[n], "atlas": node_atlas[n], "factor": factor(e, own_rg[j]),
                             "stations": [todo_all[k] for k in nb[0]], "weights": wn / wn.sum()})
        tasks.append({"file_name": name, "entities": entities, "weights": w[j],
                      "node_weight": float((w[j] * (i[j] >= len(st_lonlat))).sum())})
    order = np.lexsort((rg_lonlat[:, 1], rg_lonlat[:, 0]))  # neighbouring gauges share stations within a chunk
    tasks = [tasks[k] for k in order]
    bases = {a: tuple(str(check.cache / f"{a}_{kind}.npy") for kind in ("hourly_re", "hourly_im", "nodal_re", "nodal_im"))
             for a in models}
    chunks = [(tasks[n:n + INTERP_CHUNK], str(check.cache / "gtsm_surge_stations"), bases)
              for n in range(0, len(tasks), INTERP_CHUNK)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(_node_eval_task, chunks)):
            rows += res
            print(f"  {min((k + 1) * INTERP_CHUNK, len(tasks))}/{len(tasks)} gauges", flush=True)
    built = pd.DataFrame(rows).set_index("file_name")
    table = pd.read_parquet(dest / "tile_interpolation" / "gauges.parquet")
    for rp in GAUGE_RPS:
        table[f"nodes_rp{rp}_m"] = built[f"nodes_rp{rp}_m"].reindex(table.index)
    table["node_weight"] = built.node_weight.reindex(table.index)
    table.to_parquet(root / "gauges.parquet")
    variants = {"stations (current forcing)": "current", "stations, corrected": "current_tide_corrected",
                "stations + nodes, corrected": "nodes"}
    s = scores(table, variants)
    s.to_parquet(root / "scores.parquet")
    print(s[s.gauges.isin(["all", "non-TC-dominated gauges"]) & s.rp.isin([2, 100])].to_string(index=False), flush=True)


def _null_shapes(args) -> np.ndarray:
    n, shape0, seed = args
    rng = np.random.default_rng(seed)
    return np.array([genpareto.fit(genpareto.rvs(shape0, size=n, random_state=rng), floc=0)[0]
                     for _ in range(NOISE_REPS)])


def _cycles_per_year(check: GaugeCheck) -> np.ndarray:
    out = np.full(len(check.lev), np.nan)
    for atlas in ("fes2022", "fes2014"):
        ks = np.flatnonzero((check.lev.atlas == atlas).to_numpy())
        b_re, b_im = (np.load(check.cache / f"{atlas}_hourly_{p}.npy", mmap_mode="r")[:CYCLE_COUNT_HOURS]
                      for p in ("re", "im"))
        for i in range(0, len(ks), 2000):
            k = ks[i:i + 2000]
            tide = tp.predict(b_re, b_im, np.nan_to_num(check.hc[atlas][k]))
            out[k] = [len(ss.high_waters(tide[:, j], ss.HW_PROMINENCE)[0]) for j in range(len(k))]
    return out * 8766 / CYCLE_COUNT_HOURS


def shape_noise(check: GaugeCheck, dest: Path, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    lev = check.lev
    cycles = _cycles_per_year(check)
    cycles[cycles == 0] = np.nanmedian(cycles)
    n = np.maximum(np.round(lev.theta * (1 - ss.THRESHOLD_Q) * cycles * lev.n_years), 5).astype(int)
    product = (lev.level_source == "product").to_numpy()
    nulls = {"0": 0.0, "median": float(lev.gpd_shape_mle[product].median())}
    with ProcessPoolExecutor(workers) as pool:
        sims = {name: dict(zip(NOISE_N_GRID, pool.map(_null_shapes, [(int(m), s0, int(m)) for m in NOISE_N_GRID])))
                for name, s0 in nulls.items()}
    grid = NOISE_N_GRID.astype(float)
    st = lev[["station_id", "lon", "lat", "n_years", "theta", "gpd_shape_mle", "gpd_shape"]].copy()
    st["cycles_per_yr"], st["n_storms"] = cycles, n
    st["tc_part"] = st.station_id.isin(check.tc_levels.index).to_numpy()
    st = st[product]
    for name, sim in sims.items():
        # P(MLE shape >= observed) and P(MLE shape > threshold) under the null, interpolated in log n
        p_obs = np.array([[np.mean(sim[m] >= x) for m in NOISE_N_GRID] for x in st.gpd_shape_mle])
        st[f"p_null_{name}"] = [np.interp(np.log(v), np.log(grid), row) for v, row in zip(st.n_storms, p_obs)]
        for th in NOISE_THRESHOLDS:
            curve = np.array([np.mean(sim[m] > th) for m in NOISE_N_GRID])
            st[f"null_{name}_gt_{th:g}"] = np.interp(np.log(st.n_storms), np.log(grid), curve)
    st.to_parquet(dest / "shape_noise.parquet")
    pd.DataFrame([{"null": name, "n_storms": m, "median": np.median(s), "sd": np.std(s),
                   **{f"gt_{th:g}": np.mean(s > th) for th in NOISE_THRESHOLDS}}
                  for name, sim in sims.items() for m, s in sim.items()]).to_parquet(dest / "shape_noise_null.parquet")
    for label, sel in (("all", np.ones(len(st), bool)), ("TC part", st.tc_part.to_numpy()),
                       ("no TC part", ~st.tc_part.to_numpy())):
        s = st[sel]
        line = [f"{label} ({len(s)}): n_storms median {s.n_storms.median():.0f}, MLE shape median "
                f"{s.gpd_shape_mle.median():.3f} sd {s.gpd_shape_mle.std():.3f}"]
        for th in NOISE_THRESHOLDS:
            line.append(f"  > {th:g}: observed {(s.gpd_shape_mle > th).sum()}, expected "
                        + ", ".join(f"null {k} {s[f'null_{k}_gt_{th:g}'].sum():.0f}" for k in nulls))
        line.append("  share with p < 0.05: " + ", ".join(f"null {k} {(s[f'p_null_{k}'] < 0.05).mean():.3f}"
                                                          for k in nulls))
        print("\n".join(line), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("cap", "despike", "tc_mask", "fes_rule", "peaks", "shape_noise",
                                                       "tc_scale", "tc_jumps", "r_variants", "long_record",
                                                       "pooled_shape", "tide_correction",
                                                       "tile_interpolation", "coastline_nodes"))
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
        if stage in ("cap", "tc_mask", "fes_rule", "shape_noise", "tc_scale", "tc_jumps", "r_variants",
                     "long_record", "pooled_shape", "tide_correction", "tile_interpolation", "coastline_nodes"):
            {"cap": cap, "tc_mask": tc_mask, "fes_rule": fes_rule, "shape_noise": shape_noise,
             "tc_scale": tc_scale, "tc_jumps": tc_jumps, "r_variants": r_variants, "long_record": long_record,
             "pooled_shape": pooled_shape, "tide_correction": tide_correction,
             "tile_interpolation": tile_interpolation, "coastline_nodes": coastline_nodes}[stage](check, dest, args.workers)
        else:
            {"despike": despike, "peaks": peaks}[stage](check, dest)


if __name__ == "__main__":
    main()
