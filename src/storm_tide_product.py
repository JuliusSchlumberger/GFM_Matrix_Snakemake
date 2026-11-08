"""Storm-tide RP levels and per-RP hydrographs at the COAST-RP stations from GTSM-ERA5, FES and gauges.

ETC levels: skew surges of the GTSM-ERA5 surge residual (1979-2024) on the hourly FES tide, a
GPD on their declustered tail (shape capped at SHAPE_CAP), positive skew surges scaled by the
gauge ratio r, convolved with the FES high waters of one nodal cycle (`skew_surge`). r is the
inverse-distance mean, within IDW_RADIUS_KM, of the GESLA gauges' ratio of gauge to GTSM-on-FES
skew surge at RATIO_RP (non-TC-dominated gauges, at least MIN_NEIGHBOURS, else their median),
capped at R_CAP. Atlas: FES2022, FES2014 where `tide_prediction.consistent_version_flag` fires.
TC: COAST-RP_TC's annual exceedance rate is added to the ETC rate, as COAST-RP combines its
own two parts. The ETC fit keeps the cyclones of the ERA5 record: both TC sources are too low at
the gauges and counting them in both matches the gauges best (INPUT_ANALYSIS.md §8.6). RP
levels solve rate(z) = -ln(1 - 1/RP), m above local MSL.
Hydrographs: the FES tide around the high water nearest the median tide-at-peak of the
station's annual maxima of the RP window, plus their median 13-h smoothed surge, scaled so the
peak is the RP level (`storm_tide_hydrograph.rp_hydrograph`); a gauge's surge-dominated shape
or a Gaussian where the TC rate dominates.
Method and validation: boundary_test/INPUT_ANALYSIS.md §8.3-8.5. Stages are run by
preparation/prepare_storm_tide_product.py.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

from gesla import extract_record
from gtsm_series import load_series, station_series
from skew_surge import (
    HW_PROMINENCE, SPIKE_M, annual_rate, despike, exceedance, high_waters, scaled, shifted, skew_level, skew_surges,
    surge_model, tide_peaks)
from storm_tide_hydrograph import (
    STEPS_PER_HR, annual_maximum_events, gaussian_shape, high_water_indices, high_water_window, median_surge_shape,
    rp_event_selection, rp_hydrograph)
from tide_prediction import (
    FES2022_OK_M, STEP_MIN, fit_constants, hour_years, levels, on_hourly_grid, predict, remove_annual_means)

RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)
EARTH_RADIUS_KM = 6371.0
# Gauges
GAUGE_GOOD_YEARS = 20  # complete years since 1979 for a ratio gauge
GAUGE_GTSM_KM = 10.0  # gauge to its GTSM station
SAME_STATION_KM = 0.1  # COAST-RP / COAST-RP_TC rows of one station
YEAR_COVERAGE = 0.8  # share of a year's cycles observed for the year to count
MIN_YEARS = 15  # common gauge / GTSM years for a ratio
RATIO_RP = 10
# Stations
IDW_RADIUS_KM = 500.0
MIN_NEIGHBOURS = 3
R_CAP = 2.0  # ratios above it rest on skew surges of a few decimetres
SHAPE_CAP = 0.35  # GPD shape bound: Fathom RFA's cap (Collings et al., 2024), ~p95 of GTSM at the gauges
GTSM_QA_JUMPS = 5  # hourly residual jumps > SPIKE_M that mark a station's GTSM series unusable
# Hydrographs (accepted generator, INPUT_ANALYSIS.md §8.3)
BASE_HR, EXTENDED_HR = 72, 120
CENSORED_SHARE = 0.25  # share of annual maxima above MSL over a whole +-BASE_HR half: use EXTENDED_HR
SMOOTH_HR = 13  # one semidiurnal period: removes the tide-surge interaction ripple of the residual
MIN_EVENTS = 10
TC_GAUGE_KM, TC_SURGE_SHARE, TC_HALF_WIDTH_HR = 50.0, 0.5, 8.0
TC_SHAPE_MIN_RP = 10  # below it COAST-RP_TC is the tide floor of the synthetic tracks
EVENT_WINDOW_HR = 72
EVENT_WINDOW_COVERAGE = 0.9
N_STEPS = 2 * EXTENDED_HR * STEPS_PER_HR + 1
TASK_SIZE = 25


def coast_rp_levels(path: Path) -> pd.DataFrame:
    """RP levels (station row, RP) of a COAST-RP file, indexed by station_id."""
    with xr.open_dataset(path) as ds:
        return pd.DataFrame({rp: ds[f"storm_tide_rp_{rp:04d}"].values for rp in RPS},
                            index=ds.station_id.values.astype(str))


def nearest(src_lonlat: np.ndarray, query_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index into `src_lonlat` (finite rows only) and great-circle distance (km) of the nearest."""
    ok = np.isfinite(src_lonlat).all(axis=1)
    idx = np.flatnonzero(ok)
    d, i = BallTree(np.radians(src_lonlat[ok][:, ::-1]), metric="haversine").query(np.radians(query_lonlat[:, ::-1]), k=1)
    return idx[i[:, 0]], d[:, 0] * EARTH_RADIUS_KM


def match_stations(coast_rp_path: Path, gtsm_lonlat: np.ndarray) -> pd.DataFrame:
    """COAST-RP stations (row order) with their nearest GTSM output point."""
    with xr.open_dataset(coast_rp_path) as c:
        ll = np.column_stack([c.station_x_coordinate.values, c.station_y_coordinate.values]).astype(float)
        ids = c.station_id.values.astype(str)
    i, d = nearest(gtsm_lonlat, ll)
    return pd.DataFrame({"station_id": ids, "lon": ll[:, 0], "lat": ll[:, 1], "gtsm_index": i, "match_km": d},
                        index=pd.RangeIndex(len(ll), name="station"))


def pair_gauges(sites: pd.DataFrame, gtsm_lonlat: np.ndarray, coast_rp_path: Path) -> pd.DataFrame:
    """GESLA sites with their nearest GTSM output point and COAST-RP station."""
    q = sites[["lon", "lat"]].to_numpy(dtype=float)
    pairs = sites[["file_name", "site_name", "country", "lon", "lat"]].copy()
    pairs["time_zone_hr"] = pd.to_numeric(sites.time_zone_hr, errors="coerce").fillna(0.0).to_numpy()
    pairs["gtsm_index"], pairs["gtsm_dist_km"] = nearest(gtsm_lonlat, q)
    with netCDF4.Dataset(coast_rp_path) as ds:
        ll = np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)
    i, _ = nearest(ll, q)
    pairs["coast_rp_lon"], pairs["coast_rp_lat"] = ll[i, 0], ll[i, 1]
    return pairs.reset_index(drop=True)


def _record_task(args) -> dict:
    gesla_dir, file_name, tz_hr, target = args
    return extract_record(Path(gesla_dir), file_name, tz_hr, Path(target))


def extract_gauge_records(gesla_dir: Path, pairs: pd.DataFrame, dest: Path, workers: int) -> pd.DataFrame:
    """Write every gauge's record to dest/<file_name>.npz; return good_years per gauge."""
    dest.mkdir(parents=True, exist_ok=True)
    tasks = [(str(gesla_dir), f, tz, str(dest / f"{f}.npz")) for f, tz in zip(pairs.file_name, pairs.time_zone_hr)]
    with ProcessPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(_record_task, tasks))).set_index("file_name")


def _gauge_fit_task(args) -> list:
    records, names, hourly = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in hourly)
    out = []
    for f in names:
        data = np.load(Path(records) / f"{f}.npz")
        out.append(fit_constants(*on_hourly_grid(data["t"], data["z"]), b_re, b_im)[0])
    return out


def _mhws_task(args) -> list:
    paths, hc = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in paths)
    tide = predict(b_re, b_im, np.nan_to_num(hc))
    ok = np.isfinite(hc).all(axis=1)
    return [levels(tide[:, j], STEP_MIN * 60.0)["mhws_m"] if ok[j] else np.nan for j in range(len(hc))]


def nodal_mhws(paths: tuple[Path, Path], hc: np.ndarray, workers: int) -> np.ndarray:
    """MHWS (m) of constants (point, constituent) over the 10-min nodal-cycle basis `paths`."""
    tasks = [(paths, hc[i:i + TASK_SIZE]) for i in range(0, len(hc), TASK_SIZE)]
    with ProcessPoolExecutor(workers) as pool:
        return np.array([m for chunk in pool.map(_mhws_task, tasks) for m in chunk])


def fit_gauge_tides(pairs: pd.DataFrame, records: Path, hourly: tuple[Path, Path], workers: int) -> np.ndarray:
    """Harmonic constants (gauge, constituent) fitted to each record on the hourly FES2022 basis."""
    names = pairs.file_name.tolist()
    chunks = [(str(records), names[i:i + TASK_SIZE], hourly) for i in range(0, len(names), TASK_SIZE)]
    with ProcessPoolExecutor(workers) as pool:
        return np.vstack([h for chunk in pool.map(_gauge_fit_task, chunks) for h in chunk])


def gauge_station_class(pairs: pd.DataFrame, coast_rp_dir: Path) -> pd.Series:
    """tc_dominated / etc_with_tc / no_tc from COAST-RP's ETC and TC RP100 at each gauge's station."""
    target = np.radians(pairs[["coast_rp_lat", "coast_rp_lon"]].to_numpy(dtype=float))
    rp100 = {}
    for name in ("COAST-RP_ETC.nc", "COAST-RP_TC.nc"):
        with netCDF4.Dataset(Path(coast_rp_dir) / name) as ds:
            ll = np.column_stack([ds["station_y_coordinate"][:], ds["station_x_coordinate"][:]]).astype(float)
            d, i = BallTree(np.radians(ll), metric="haversine").query(target, k=1)
            same = d[:, 0] * EARTH_RADIUS_KM <= SAME_STATION_KM
            rp100[name] = np.where(same, np.ma.filled(ds["storm_tide_rp_0100"][:], np.nan)[i[:, 0]], np.nan)
    etc, tc = rp100["COAST-RP_ETC.nc"], rp100["COAST-RP_TC.nc"]
    return pd.Series(np.select([np.isfinite(tc) & (tc > etc), np.isfinite(tc)], ["tc_dominated", "etc_with_tc"],
                               "no_tc"), index=pairs.file_name)


def _ratio_site(site: dict, records: Path, surge, surge_idx: np.ndarray, pos: dict, b_re, b_im) -> dict | None:
    data = np.load(records / f"{site['file_name']}.npz")
    idx, z = on_hourly_grid(data["t"], data["z"])
    if len(idx) == 0:
        return None
    n_hours = b_re.shape[0]
    tg, tf = predict(b_re, b_im, site["hc_gauge"]), predict(b_re, b_im, site["hc_fes"])
    obs = np.full(n_hours, np.nan)
    obs[idx] = remove_annual_means(idx, z)
    gtsm = np.full(n_hours, np.nan)
    gtsm[surge_idx] = station_series(surge, pos[site["gtsm_index"]])
    year = hour_years(np.arange(n_hours))
    skew, years = {}, {}
    for name, total, tide in (("gauge", obs, tg), ("gtsmfes", gtsm + tf, tf)):
        hw, height = high_waters(tide, HW_PROMINENCE)
        skew[name] = skew_surges(total, hw, height)
        yr = pd.Series(np.isfinite(skew[name])).groupby(year[hw]).mean()
        years[name] = (year[hw], set(yr.index[yr >= YEAR_COVERAGE]))
    common = sorted(years["gauge"][1] & years["gtsmfes"][1])
    if len(common) < MIN_YEARS:
        return None
    level = {}
    for name in ("gauge", "gtsmfes"):
        yr, s = years[name][0], skew[name]
        level[name] = skew_level(surge_model(s[np.isin(yr, common) & np.isfinite(s)], len(common)), RATIO_RP)
    return {"file_name": site["file_name"], "n_years": len(common), "skew_gauge_m": level["gauge"],
            "skew_gtsmfes_m": level["gtsmfes"], "ratio_fes": level["gauge"] / level["gtsmfes"]}


def _ratio_task(args) -> list[dict]:
    sites, records, store, hourly = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in hourly)
    surge_st, surge_time, surge = load_series(Path(store))
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    rows = []
    for site in sites:
        try:
            r = _ratio_site(site, Path(records), surge, surge_idx, pos, b_re, b_im)
        except Exception as err:  # one bad record must not stop the pool; it is reported and skipped
            print(f"  skipped {site['file_name']}: {type(err).__name__}: {err}", flush=True)
            continue
        if r is not None:
            rows.append(r)
    return rows


def gauge_ratios(sites: list[dict], records: Path, store: Path, hourly: tuple[Path, Path], workers: int) -> pd.DataFrame:
    """ratio_fes = gauge / GTSM-on-FES skew surge at RATIO_RP per gauge (dicts: file_name,
    gtsm_index, hc_gauge, hc_fes, the latter two without Sa/Ssa)."""
    chunks = [(sites[i:i + TASK_SIZE // 2], str(records), str(store), hourly)
              for i in range(0, len(sites), TASK_SIZE // 2)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(_ratio_task, chunks)):
            rows += r
            print(f"  {min((k + 1) * (TASK_SIZE // 2), len(sites))}/{len(sites)} gauges", flush=True)
    return pd.DataFrame(rows).set_index("file_name")


def _donor_task(args) -> list:
    names, hcs, records, hourly = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in hourly)
    n_hours = b_re.shape[0]
    year = hour_years(np.arange(n_hours))
    out = []
    for f, hc in zip(names, hcs):
        data = np.load(Path(records) / f"{f}.npz")
        idx, z = on_hourly_grid(data["t"], data["z"])
        if not len(idx) or not np.isfinite(hc).all():
            out.append([])
            continue
        level = np.full(n_hours, np.nan)
        level[idx] = remove_annual_means(idx, z)
        surge = level - predict(b_re, b_im, hc)
        good = pd.Series(level).groupby(year).count() >= YEAR_COVERAGE * 8766
        events = []
        for y in good.index[good.to_numpy()]:
            sel = np.flatnonzero(year == y)
            c = int(sel[np.nanargmax(level[sel])])
            lo, hi = c - EVENT_WINDOW_HR, c + EVENT_WINDOW_HR + 1
            if lo < 0 or hi > n_hours or np.isfinite(level[lo:hi]).mean() < EVENT_WINDOW_COVERAGE:
                continue
            if surge[c] / level[c] >= TC_SURGE_SHARE:
                events.append(surge[lo:hi].astype(np.float32))
        out.append(events)
    return out


def tc_donors(pairs: pd.DataFrame, hc_full: np.ndarray, records: Path, hourly: tuple[Path, Path],
              workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Gauges with >= 2 surge-dominated annual maxima (surge >= TC_SURGE_SHARE of the level):
    (table lat/lon/n, median surge shapes (donor, 10-min step)). `hc_full` keeps Sa/Ssa, so the
    seasonal cycle is tide here, as in the shape analysis."""
    names = pairs.file_name.tolist()
    chunks = [(names[i:i + TASK_SIZE], hc_full[i:i + TASK_SIZE], str(records), hourly)
              for i in range(0, len(names), TASK_SIZE)]
    with ProcessPoolExecutor(workers) as pool:
        events = [e for chunk in pool.map(_donor_task, chunks) for e in chunk]
    rows, shapes = [], []
    for k, ev in enumerate(events):
        if len(ev) >= 2:
            rows.append({"file_name": names[k], "lat": pairs.lat.iloc[k], "lon": pairs.lon.iloc[k], "n_events": len(ev)})
            shapes.append(median_surge_shape(np.vstack(ev).astype(float)))
    n = max(len(s) for s in shapes)
    return pd.DataFrame(rows), np.vstack([fit_length(s, n) for s in shapes])


def station_ratios(stations: pd.DataFrame, gauges: pd.DataFrame) -> pd.DataFrame:
    """Gauge ratio r per station: IDW within IDW_RADIUS_KM, else the gauge median, capped."""
    cal = gauges[gauges.calibrate & np.isfinite(gauges.ratio_fes)]
    tree = BallTree(np.radians(cal[["lat", "lon"]].to_numpy()), metric="haversine")
    idx, dist = tree.query_radius(np.radians(stations[["lat", "lon"]].to_numpy()), r=IDW_RADIUS_KM / EARTH_RADIUS_KM,
                                  return_distance=True)
    median = float(cal.ratio_fes.median())
    ratio = cal.ratio_fes.to_numpy()
    r, n = np.full(len(stations), median), np.zeros(len(stations), dtype=int)
    for k, (i, d) in enumerate(zip(idx, dist)):
        n[k] = len(i)
        if len(i) >= MIN_NEIGHBOURS:
            w = 1.0 / np.maximum(d * EARTH_RADIUS_KM, 1.0)
            r[k] = float(np.sum(w * ratio[i]) / w.sum())
    return pd.DataFrame({"r_uncapped": r, "r": np.minimum(r, R_CAP), "r_neighbours": n}, index=stations.index)


def station_tc_shapes(stations: pd.DataFrame, donors: pd.DataFrame, shapes: np.ndarray) -> dict:
    """Station -> median surge shape of the nearest donor gauge within TC_GAUGE_KM."""
    if not len(donors):
        return {}
    tree = BallTree(np.radians(donors[["lat", "lon"]].to_numpy()), metric="haversine")
    d, i = tree.query(np.radians(stations[["lat", "lon"]].to_numpy()), k=1)
    return {st: shapes[j] for st, dist, j in zip(stations.index, d[:, 0] * EARTH_RADIUS_KM, i[:, 0])
            if dist <= TC_GAUGE_KM}


def running_mean(windows: np.ndarray, width_hr: int) -> np.ndarray:
    kernel = np.ones(width_hr) / width_hr
    return np.vstack([np.convolve(np.nan_to_num(w), kernel, mode="same") for w in windows])


def fit_length(shape: np.ndarray, n: int) -> np.ndarray:
    """Crop or zero-pad a centred shape to n steps."""
    if len(shape) == n:
        return shape
    c, h = len(shape) // 2, n // 2
    if len(shape) > n:
        return shape[c - h:c + h + 1]
    return np.pad(shape, h - c)


def _tc_rate(levels_tc: np.ndarray):
    lr = np.log(annual_rate(RPS))
    z = np.maximum.accumulate(levels_tc) + np.arange(len(RPS)) * 1e-6

    def rate(x: float) -> float:
        if x < z[0]:
            slope = (lr[1] - lr[0]) / (z[1] - z[0])
            return float(np.exp(min(lr[0] + (x - z[0]) * slope, 50.0)))
        if x > z[-1]:
            return float(np.exp(lr[-1] + (x - z[-1]) * (lr[-1] - lr[-2]) / (z[-1] - z[-2])))
        return float(np.exp(np.interp(x, z, lr)))
    return rate


def _solve(rate, target: float, lo: float, hi: float) -> float:
    for _ in range(50):
        z = 0.5 * (lo + hi)
        lo, hi = (z, hi) if rate(z) > target else (lo, z)
    return 0.5 * (lo + hi)


def station_surge_model(st: dict, g: np.ndarray, th: np.ndarray, years: np.ndarray) -> tuple[dict, dict, dict, np.ndarray]:
    """Row fields, raw and gauge-corrected skew-surge models and the despiked surge of one station.

    `g`: hourly GTSM surge residual (m, NaN missing); `th`: the station's hourly FES tide on the same
    axis. `st` may set "shape_cap" (None: no cap) in place of SHAPE_CAP, "pooled_shape" as a fixed
    GPD shape, and "surge_offset_m" adds a gauge offset to the positive skew surges in place of the
    ratio r.
    """
    row = {"station": st["station"], "atlas": st["atlas"], "mhws_m": st["mhws_m"]}
    row["gtsm_jumps"] = int(np.sum(np.abs(np.diff(g)) > SPIKE_M))
    g = despike(g)
    hw, h = high_waters(th, HW_PROMINENCE)
    skew = skew_surges(g + th, hw, h)
    cover = pd.Series(np.isfinite(skew)).groupby(years[hw]).mean()
    keep_years = cover.index[cover >= YEAR_COVERAGE]
    ok = np.isin(years[hw], keep_years) & np.isfinite(skew)
    # tc_hours (cycles left out of the ETC fit) is set only by the cyclone-mask test, boundary_test/open_guesses.py
    tc_cycles = np.isin(hw, st["tc_hours"]) if st.get("tc_hours") is not None else np.zeros(len(hw), bool)
    row["tc_masked_share"] = float(tc_cycles[ok].mean())
    model = surge_model(skew[ok & ~tc_cycles], len(keep_years), max_shape=st.get("shape_cap", SHAPE_CAP),
                        shape=st.get("pooled_shape"))
    corrected = shifted(model, st["surge_offset_m"]) if "surge_offset_m" in st else scaled(model, st["r"])
    row.update({"n_years": len(keep_years), "gpd_shape": model["shape"], "gpd_shape_mle": model["shape_mle"],
                "gpd_scale": model["scale"], "theta": model["theta"]})
    return row, model, corrected, g


def rp_levels(model: dict, corrected: dict, t10: np.ndarray, tc_levels: np.ndarray | None) -> dict:
    """ETC (raw and corrected) and total RP levels (m) of skew-surge models on the 10-min tide t10.

    `tc_levels`: COAST-RP_TC levels at RPS (m above MSL), added as a rate; None for no TC part.
    """
    v, w, cyc = tide_peaks(t10, STEP_MIN / 60.0, HW_PROMINENCE)

    def etc_rate(m):
        return lambda z: cyc * m["theta"] * float(np.sum(w * exceedance(m, z - v)))
    tc = _tc_rate(tc_levels) if tc_levels is not None else (lambda z: 0.0)
    lo, hi = float(v.min()), float(v.max()) + 20.0
    out = {}
    for rp, target in zip(RPS, annual_rate(RPS)):
        out[f"etc_raw_rp{rp}_m"] = _solve(etc_rate(model), target, lo, hi)
        out[f"etc_rp{rp}_m"] = _solve(etc_rate(corrected), target, lo, hi)
        out[f"rp{rp}_m"] = z = _solve(lambda x: etc_rate(corrected)(x) + tc(x), target, lo, hi)
        out[f"tc_dominant_rp{rp}"] = bool(tc(z) > etc_rate(corrected)(z))
    return out


def station_product(st: dict, g: np.ndarray, th: np.ndarray, t10: np.ndarray,
                    years: np.ndarray) -> tuple[dict, np.ndarray, list]:
    """RP levels (row), hydrographs (RP, N_STEPS) and their sources for one station.

    `t10`: the station's FES tide, 10-min over one nodal cycle; `g`, `th`, `years` and the `st` keys
    as in `station_surge_model`, plus "tc_scale" / "tc_shift_m" applied to the COAST-RP_TC levels
    (m above MSL) before they become a rate.
    """
    row, model, corrected, g = station_surge_model(st, g, th, years)
    hyd = np.full((len(RPS), N_STEPS), np.nan, dtype=np.float32)
    info = []
    jumps = row["gtsm_jumps"]
    tc_levels = st["tc"] * st.get("tc_scale", 1.0) + st.get("tc_shift_m", 0.0) if st["tc"] is not None else None
    row.update(rp_levels(model, corrected, t10, tc_levels))
    # GTSM's residual is numerically broken here (jumps of metres within an hour): COAST-RP's level
    row["level_source"] = "coast_rp" if jumps >= GTSM_QA_JUMPS else "product"
    if jumps >= GTSM_QA_JUMPS:
        for rp, z in zip(RPS, st["coast_rp"]):
            row[f"rp{rp}_m"] = float(z)
            row[f"etc_raw_rp{rp}_m"] = row[f"etc_rp{rp}_m"] = np.nan

    level = g + th
    ev = annual_maximum_events(level, years, EXTENDED_HR)
    if len(ev) < MIN_EVENTS:
        return row, hyd, info
    centres = ev.centre.to_numpy()
    win = np.arange(-EXTENDED_HR, EXTENDED_HR + 1)
    surge_win, level_win = g[centres[:, None] + win], level[centres[:, None] + win]
    tide_at_peak = th[centres]
    c = EXTENDED_HR
    above = level_win > 0
    censored = above[:, c - BASE_HR:c + 1].all(axis=1) | above[:, c:c + BASE_HR + 1].all(axis=1)
    half_hr = EXTENDED_HR if censored.mean() >= CENSORED_SHARE else BASE_HR
    hw10 = high_water_indices(t10)
    for j, rp in enumerate(RPS):
        sel = rp_event_selection(ev.rp_yr.to_numpy(), rp)
        sel = sel if len(sel) >= 2 else np.arange(len(ev))
        tide_level = float(np.median(tide_at_peak[sel]))
        tide_win = high_water_window(t10, hw10, tide_level, half_hr)
        if tide_win.max() >= row[f"rp{rp}_m"]:
            tide_win = high_water_window(t10, hw10, tide_level, half_hr, ceiling_m=row[f"rp{rp}_m"])
        if row[f"tc_dominant_rp{rp}"] and row[f"tc_dominant_rp{max(rp, TC_SHAPE_MIN_RP)}"]:
            source = "tc_gauge" if st["tc_shape"] is not None else "tc_gaussian"
            shape = st["tc_shape"] if st["tc_shape"] is not None else gaussian_shape(TC_HALF_WIDTH_HR, half_hr)
            shape = fit_length(shape, len(tide_win))
        elif jumps >= GTSM_QA_JUMPS:
            shape, source = gaussian_shape(TC_HALF_WIDTH_HR, half_hr), "gtsm_qa_gaussian"
        else:
            try:
                shape = median_surge_shape(running_mean(surge_win[sel][:, c - half_hr:c + half_hr + 1], SMOOTH_HR))
                source = "gtsm_events"
            except ValueError:  # every selected maximum is a spring tide with no positive surge
                shape, source = gaussian_shape(TC_HALF_WIDTH_HR, half_hr), "no_surge_gaussian"
        lev = rp_hydrograph(tide_win, shape, row[f"rp{rp}_m"])["level"]
        pad = (N_STEPS - len(lev)) // 2
        hyd[j, pad:pad + len(lev)] = lev
        info.append({"station": st["station"], "rp": rp, "half_window_hr": half_hr,
                     "tide_level_ratio": tide_level / st["mhws_m"], "n_events": len(sel), "shape_source": source})
    return row, hyd, info


def _station_task(args):
    chunk, store, bases = args
    surge_st, surge_time, surge = load_series(Path(store))
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    results = []
    for atlas in bases:
        sts = [s for s in chunk if s["atlas"] == atlas]
        if not sts:
            continue
        hc = np.vstack([s["hc"] for s in sts])
        b = [np.load(p, mmap_mode="r") for p in bases[atlas]]
        th_all, t10_all = predict(b[0], b[1], hc), predict(b[2], b[3], hc)
        n_hours = th_all.shape[0]
        years = hour_years(np.arange(n_hours))
        for j, st in enumerate(sts):
            g = np.full(n_hours, np.nan)
            g[surge_idx] = station_series(surge, pos[st["gtsm_index"]])
            try:
                results.append(station_product(st, g, th_all[:, j], t10_all[:, j], years))
            except Exception as err:  # one bad station must not stop the pool; it is reported and skipped
                print(f"  skipped station {st['station']}: {type(err).__name__}: {err}", flush=True)
    return results


def build_stations(todo: list[dict], store: Path, bases: dict, hyd: np.ndarray, workers: int) -> tuple[list, list]:
    """Run `station_product` for every station input in `todo`, writing hyd[row] in place.

    `bases`: atlas -> (hourly re, hourly im, 10-min re, 10-min im) .npy paths.
    """
    tasks = [(todo[i:i + TASK_SIZE], str(store), bases) for i in range(0, len(todo), TASK_SIZE)]
    rows, info = [], []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(_station_task, tasks)):
            for row, h, i in res:
                rows.append(row)
                info += i
                hyd[row["station"]] = h
            if k % 20 == 0:
                print(f"  {min((k + 1) * TASK_SIZE, len(todo))}/{len(todo)} stations", flush=True)
    return rows, info


def fes2022_ok_gauges(pairs: pd.DataFrame, mhws_gauge: np.ndarray, mhws_fes: np.ndarray) -> np.ndarray:
    """(lon, lat) of the gauges where FES2022's MHWS is within FES2022_OK_M of the gauge's own."""
    ok = np.abs(mhws_fes - mhws_gauge) <= FES2022_OK_M
    return pairs.loc[ok, ["lon", "lat"]].to_numpy(dtype=float)


def export_levels(lev: pd.DataFrame, coast_rp_path: Path, dest: Path) -> None:
    """Write the levels in COAST-RP's layout (stations, order, coordinates, storm_tide_rp_XXXX).

    Stations without a product level keep COAST-RP's (`level_source`). RP1 is not produced.
    """
    coast = xr.open_dataset(coast_rp_path)
    assert len(lev) == coast.sizes["stations"] and (lev.station_id.to_numpy() == coast.station_id.values.astype(str)).all()
    source = lev.level_source.fillna("coast_rp").to_numpy(dtype=object)
    data = {}
    for rp in RPS:
        z = lev[f"rp{rp}_m"].to_numpy(dtype=np.float64)
        fallback = coast[f"storm_tide_rp_{rp:04d}"].values.astype(np.float64)
        missing = ~np.isfinite(z)
        source[missing] = "coast_rp"
        data[f"storm_tide_rp_{rp:04d}"] = ("stations", np.where(missing, fallback, z),
                                           {"units": "m", "long_name": f"storm tide, return period {rp} y",
                                            "vertical_reference": "local mean sea level"})
    data["level_source"] = ("stations", source.astype(str))
    for col in ("r", "atlas", "gpd_shape"):
        data[col] = ("stations", lev[col].to_numpy())
    ds = xr.Dataset(data, coords={"station_id": ("stations", coast.station_id.values),
                                  "station_x_coordinate": ("stations", coast.station_x_coordinate.values),
                                  "station_y_coordinate": ("stations", coast.station_y_coordinate.values)},
                    attrs={"title": "GFM storm-tide return levels", "source": "src/storm_tide_product.py",
                           "return_periods_y": list(RPS)})
    coast.close()
    tmp = dest.with_suffix(".nc.part")
    ds.to_netcdf(tmp)
    tmp.replace(dest)
    print(f"{dest.name}: {len(lev)} stations, {int((source == 'coast_rp').sum())} with COAST-RP levels", flush=True)


def export_hydrographs(lev: pd.DataFrame, hyd: np.ndarray, dest: Path) -> None:
    """Write `storm_tide_hydrograph` (stations, return_period, time): 10-min steps around the
    peak at time 0, NaN outside each station's window (`event_volume.load_coast_hg` reads it)."""
    half = hyd.shape[2] // 2
    step = np.timedelta64(600, "s")
    ds = xr.Dataset({"storm_tide_hydrograph": (("stations", "return_period", "time"), hyd,
                                               {"units": "m", "vertical_reference": "local mean sea level"})},
                    coords={"station_id": ("stations", lev.station_id.to_numpy()),
                            "station_x_coordinate": ("stations", lev.lon.to_numpy()),
                            "station_y_coordinate": ("stations", lev.lat.to_numpy()),
                            "return_period": list(RPS),
                            "time": (np.arange(-half, half + 1) * step).astype("timedelta64[ns]")},
                    attrs={"title": "GFM per-RP storm-tide hydrographs", "source": "src/storm_tide_product.py"})
    enc = {"storm_tide_hydrograph": {"zlib": True, "complevel": 1, "chunksizes": (64, len(RPS), hyd.shape[2])}}
    tmp = dest.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(dest)
    print(f"{dest.name}: {hyd.shape}", flush=True)
