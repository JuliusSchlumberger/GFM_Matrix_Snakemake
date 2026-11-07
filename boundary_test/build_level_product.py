"""Tide-corrected RP levels and per-RP hydrographs at every COAST-RP station (INPUT_ANALYSIS.md §8).

Levels (tests 5 and 9): skew surges of the GTSM-ERA5 surge residual (1979-2024) laid on the
hourly FES tide, a GPD on their declustered tail (`surge_correction.surge_model`, shape capped at
SHAPE_CAP), positive skew surges scaled by the gauge ratio r, convolved with the FES high waters
of one nodal cycle. r is the inverse-distance mean of `ratio_fes` at the non-TC-dominated
gauges within IDW_RADIUS_KM (at least MIN_NEIGHBOURS, else their median), capped at R_CAP:
ratios above it rest on skew surges of a few decimetres. The atlas is FES2022, or FES2014 where
the version rule fires (`atlas_rule_sensitivity.version_flag`, test 6, grown to neighbours on the
same artefact by `consistent_version_flag`); gauges do not override it here. Stations on FES land cells take the constants of the nearest station with values. TC: the
annual exceedance rate of COAST-RP's TC component is added to the ETC rate, as COAST-RP combines
its own two parts (its total equals that rate sum to 1 mm); the TC levels keep GTSM's tide.
RP levels solve rate(z) = -ln(1 - 1/RP), relative to local MSL (seasonal cycle in the surge).

Hydrographs (§8.3, generator_smooth): the FES tide around the high water nearest the median
tide-at-peak of the station's annual maxima of the RP window (k x MHWS of the validation;
`rp_event_selection`; RP >= 50 takes the maxima of RP >= 20), plus the median 13-h smoothed
surge of those maxima, scaled so the peak is the RP level. Where a neighbouring high water in
the window already exceeds the RP level, the window moves to the nearest high water whose window
stays below it. Where the TC rate dominates at that level (and, below TC_SHAPE_MIN_RP, also at
TC_SHAPE_MIN_RP) the surge is a surge-dominated gauge's median event within TC_GAUGE_KM, else a
Gaussian.
Annual maxima are of GTSM surge + FES tide (GTSM's own tide is not extracted at every station).

Writes <out-dir>/levels.parquet (one row per station), hydrographs.npy (station, RP, 10-min step
over +-EXTENDED_HR, NaN outside the station's window), hydrograph_info.parquet and the FES caches;
the `gauges` stage (after `build`) writes gauge_check.parquet and gauge_check_summary.csv.

Usage:
    python boundary_test/build_level_product.py stations --out-dir .cleanup_baseline/level_product
    python boundary_test/extract_gtsm_series.py --out-dir .cleanup_baseline/level_product \
        --stations-csv .cleanup_baseline/level_product/stations.csv --variable surge
    python boundary_test/build_level_product.py build --out-dir .cleanup_baseline/level_product \
        --sources-dir .cleanup_baseline/level_sources [--workers 3] [--limit N | --stations K ...]
    (--stations rebuilds those station indices inside the existing outputs)
    python boundary_test/build_level_product.py gauges --out-dir .cleanup_baseline/level_product
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyTMD.io
import xarray as xr
from scipy.ndimage import median_filter
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from atlas_check import atlas_basis, grouped_constants, reference_model  # noqa: E402
from atlas_rule_sensitivity import version_flag  # noqa: E402
from extract_gtsm_series import GTSM_DIR, FOLDERS, load_series, station_series  # noqa: E402
from fes_tide_stats import FES_MODEL, STEP_MIN, ensure_local_fes  # noqa: E402
from generator_validation import (  # noqa: E402
    BASE_HR, CENSORED_SHARE, EXTENDED_HR, SMOOTH_HR, TC_GAUGE_KM, TC_HALF_WIDTH_HR, TC_SURGE_SHARE, fit_length,
    running_mean)
from levels_tides import FES_CACHE, HOURLY_START, SEASONAL, fes_constants, hourly_basis, levels, on_hourly_grid  # noqa: E402
from storm_tide_hydrograph import (  # noqa: E402
    STEPS_PER_HR, annual_maximum_events, gaussian_shape, high_water_indices, high_water_window, median_surge_shape,
    rp_event_selection, rp_hydrograph)
from surge_correction import (  # noqa: E402
    EARTH_RADIUS_KM, YEAR_COVERAGE, exceedance, high_waters, scaled, skew_surges, surge_model, tide_peaks)

COAST_RP = Path("P:/11212688-004-global-floodmaps/modelling/inputs/COAST-RP_dataset")
FES2014_CACHE = REPO / ".cleanup_baseline" / "fes2014_cache"
RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)
IDW_RADIUS_KM = 500.0
MIN_NEIGHBOURS = 3
R_CAP = 2.0
SHAPE_CAP = 0.35  # GPD shape bound: Fathom RFA's cap (Collings et al., 2024), ~p95 of GTSM at the gauges
MIN_EVENTS = 10
VERSION_GROW_KM = 25.0
VERSION_GROW_DIFF_M = 0.25
VERSION_VETO_KM = 50.0
FES2022_OK_M = 0.25  # |FES2022 - GESLA| MHWS at a gauge that vetoes the growth
TC_SHAPE_MIN_RP = 10  # below it COAST-RP_TC is the tide floor of the synthetic tracks
STATIONS_PER_TASK = 25
N_STEPS = 2 * EXTENDED_HR * STEPS_PER_HR + 1
GAUGE_MAX_KM = 25.0
IBTRACS = REPO / ".cleanup_baseline" / "ibtracs" / "ibtracs.since1980.list.v04r01.csv"
HW_PROMINENCE = 0.1  # of the 25-h tidal range; drops the low-water bumps of shallow-water tides
SPIKE_M = 2.0  # GTSM residual hours this far from their +-SPIKE_HALF_HR median are dropped
SPIKE_HALF_HR = 2
GTSM_QA_JUMPS = 5  # hourly residual jumps > SPIKE_M that mark a station's GTSM series unusable
TC_MASK_KM = 300.0  # cycles with a tropical storm (>= TC_MIN_WIND_KT) this close leave the ETC fit
TC_MASK_HR = 24
TC_MIN_WIND_KT = 34.0
GAUGE_RPS = (2, 10, 50, 100)


def annual_rate(rp) -> np.ndarray:
    return -np.log1p(-1.0 / np.asarray(rp, dtype=float))


def coast_rp_levels(name: str) -> pd.DataFrame:
    ds = xr.open_dataset(COAST_RP / name)
    return pd.DataFrame({rp: ds[f"storm_tide_rp_{rp:04d}"].values for rp in RPS},
                        index=ds.station_id.values.astype(str))


def match_stations(out: Path) -> None:
    folder, pattern, var, first_year = FOLDERS["surge"]
    first = next(f for f in sorted((GTSM_DIR / folder).glob(pattern)) if int(f.stem.split("_")[3]) >= first_year)
    with netCDF4.Dataset(first) as ds:
        gtsm = np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)
    c = xr.open_dataset(COAST_RP / "COAST-RP.nc")
    ll = np.column_stack([c.station_x_coordinate.values, c.station_y_coordinate.values]).astype(float)
    d, i = BallTree(np.radians(gtsm[:, ::-1]), metric="haversine").query(np.radians(ll[:, ::-1]), k=1)
    pd.DataFrame({"station_id": c.station_id.values.astype(str), "lon": ll[:, 0], "lat": ll[:, 1],
                  "gtsm_index": i[:, 0], "match_km": d[:, 0] * EARTH_RADIUS_KM}).to_csv(out / "stations.csv",
                                                                                      index_label="station")
    print(f"{len(ll)} stations, largest match distance {d.max() * EARTH_RADIUS_KM * 1000:.1f} m")


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


def tc_donor_shapes(sources: Path, stations: pd.DataFrame) -> dict:
    """Station -> median surge of the nearest surge-dominated gauge within TC_GAUGE_KM."""
    ev = pd.read_parquet(sources / "events.parquet").reset_index(drop=True)
    surge = np.load(sources / "event_profiles.npz")["surge_m"]
    pairs = pd.read_parquet(sources / "pairs.parquet").set_index("file_name")
    g = ev[ev.source == "gesla"]
    keep = g[(g.surge_at_peak_m / g.peak_m >= TC_SURGE_SHARE).to_numpy()]
    rows = keep.groupby("file_name").groups
    donors = [f for f, r in rows.items() if len(r) >= 2 and f in pairs.index]
    tree = BallTree(np.radians(pairs.loc[donors, ["lat", "lon"]].to_numpy()), metric="haversine")
    d, i = tree.query(np.radians(stations[["lat", "lon"]].to_numpy()), k=1)
    shapes, out = {}, {}
    for st, dist, j in zip(stations.index, d[:, 0] * EARTH_RADIUS_KM, i[:, 0]):
        if dist <= TC_GAUGE_KM:
            if j not in shapes:
                shapes[j] = median_surge_shape(surge[np.asarray(rows[donors[j]])].astype(float))
            out[st] = shapes[j]
    return out


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


def tc_hours(stations: pd.DataFrame, n_hours: int) -> dict:
    """Station -> hourly indices within TC_MASK_HR of a tropical storm passing within TC_MASK_KM."""
    tracks = pd.read_csv(IBTRACS, usecols=["ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND"], skiprows=[1],
                         na_values=[" "], low_memory=False)
    wind = tracks[["WMO_WIND", "USA_WIND"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    tracks = tracks[wind >= TC_MIN_WIND_KT]
    hour = ((pd.to_datetime(tracks.ISO_TIME).to_numpy() - np.datetime64(HOURLY_START)) //
            np.timedelta64(1, "h")).astype(int)
    tree = BallTree(np.radians(tracks[["LAT", "LON"]].to_numpy(dtype=float)), metric="haversine")
    near = tree.query_radius(np.radians(stations[["lat", "lon"]].to_numpy()), r=TC_MASK_KM / EARTH_RADIUS_KM)
    spread = np.arange(-TC_MASK_HR, TC_MASK_HR + 1)
    out = {}
    for st, idx in zip(stations.index, near):
        if len(idx):
            h = np.unique((hour[idx][:, None] + spread).ravel())
            out[st] = h[(h >= 0) & (h < n_hours)]
    return out


def despike(g: np.ndarray) -> np.ndarray:
    med = median_filter(np.nan_to_num(g), 2 * SPIKE_HALF_HR + 1, mode="nearest")
    return np.where(np.abs(g - med) > SPIKE_M, np.nan, g)


def _station(st: dict, g: np.ndarray, th: np.ndarray, t10: np.ndarray, years: np.ndarray, tc_shape) -> tuple[dict, np.ndarray, list]:
    row = {"station": st["station"], "atlas": st["atlas"], "mhws_m": st["mhws_m"]}
    hyd = np.full((len(RPS), N_STEPS), np.nan, dtype=np.float32)
    info = []
    row["gtsm_jumps"] = jumps = int(np.sum(np.abs(np.diff(g)) > SPIKE_M))
    g = despike(g)
    hw, h = high_waters(th, HW_PROMINENCE)
    skew = skew_surges(g + th, hw, h)
    cover = pd.Series(np.isfinite(skew)).groupby(years[hw]).mean()
    keep_years = cover.index[cover >= YEAR_COVERAGE]
    ok = np.isin(years[hw], keep_years) & np.isfinite(skew)
    tc_cycles = np.isin(hw, st["tc_hours"]) if st["tc_hours"] is not None else np.zeros(len(hw), bool)
    row["tc_masked_share"] = float(tc_cycles[ok].mean())
    model = surge_model(skew[ok & ~tc_cycles], len(keep_years), max_shape=SHAPE_CAP)
    corrected = scaled(model, st["r"])
    v, w, cyc = tide_peaks(t10, STEP_MIN / 60.0, HW_PROMINENCE)
    row.update({"n_years": len(keep_years), "gpd_shape": model["shape"], "gpd_shape_mle": model["shape_mle"],
                "gpd_scale": model["scale"], "theta": model["theta"]})

    def etc_rate(m):
        return lambda z: cyc * m["theta"] * float(np.sum(w * exceedance(m, z - v)))
    tc = _tc_rate(st["tc"]) if st["tc"] is not None else (lambda z: 0.0)
    lo, hi = float(v.min()), float(v.max()) + 20.0
    for rp, target in zip(RPS, annual_rate(RPS)):
        row[f"etc_raw_rp{rp}_m"] = _solve(etc_rate(model), target, lo, hi)
        row[f"etc_rp{rp}_m"] = _solve(etc_rate(corrected), target, lo, hi)
        row[f"rp{rp}_m"] = z = _solve(lambda x: etc_rate(corrected)(x) + tc(x), target, lo, hi)
        row[f"tc_dominant_rp{rp}"] = bool(tc(z) > etc_rate(corrected)(z))
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
            source = "tc_gauge" if tc_shape is not None else "tc_gaussian"
            shape = tc_shape if tc_shape is not None else gaussian_shape(TC_HALF_WIDTH_HR, half_hr)
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
        info.append({"station": st["station"], "rp": rp, "half_window_hr": half_hr, "tide_level_ratio": tide_level / st["mhws_m"],
                     "n_events": len(sel), "shape_source": source})
    return row, hyd, info


def _task(args):
    out, sources, chunk, use14 = args
    out, sources = Path(out), Path(sources)
    surge_st, surge_time, surge = load_series(out, "surge")
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    bases = {"fes2022": (sources / "hourly_basis_re.npy", sources / "hourly_basis_im.npy",
                         FES_CACHE / "basis_re.npy", FES_CACHE / "basis_im.npy"),
             "fes2014": (FES2014_CACHE / "hourly_basis_re.npy", FES2014_CACHE / "hourly_basis_im.npy",
                         FES2014_CACHE / "basis_re.npy", FES2014_CACHE / "basis_im.npy")}
    results = []
    for atlas in ("fes2022", "fes2014") if use14 else ("fes2022",):
        sts = [s for s in chunk if s["atlas"] == atlas]
        if not sts:
            continue
        hc = np.vstack([s["hc"] for s in sts]).T
        b = [np.load(p, mmap_mode="r") for p in bases[atlas]]
        th_all = np.asarray(b[0]) @ hc.real + np.asarray(b[1]) @ hc.imag
        t10_all = np.asarray(b[2]) @ hc.real + np.asarray(b[3]) @ hc.imag
        n_hours = th_all.shape[0]
        years = (HOURLY_START + np.arange(n_hours).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
        for j, st in enumerate(sts):
            g = np.full(n_hours, np.nan)
            g[surge_idx] = station_series(surge, pos[st["gtsm_index"]])
            try:
                results.append(_station(st, g, th_all[:, j], t10_all[:, j], years, st["tc_shape"]))
            except Exception as err:  # one bad station must not stop the pool; it is reported and skipped
                print(f"  skipped station {st['station']}: {type(err).__name__}: {err}", flush=True)
    return results


def _mhws_task(args):
    paths, hc = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in paths)
    tide = np.asarray(b_re) @ np.nan_to_num(hc.real).T + np.asarray(b_im) @ np.nan_to_num(hc.imag).T
    ok = np.isfinite(hc).all(axis=1)
    return [levels(tide[:, j], STEP_MIN * 60.0)["mhws_m"] if ok[j] else np.nan for j in range(len(hc))]


def _mhws(paths, hc: np.ndarray, workers: int) -> np.ndarray:
    tasks = [(paths, hc[i:i + STATIONS_PER_TASK]) for i in range(0, len(hc), STATIONS_PER_TASK)]
    with ProcessPoolExecutor(workers) as pool:
        return np.array([m for chunk in pool.map(_mhws_task, tasks) for m in chunk])


def fill_masked(hc: np.ndarray, lonlat: np.ndarray) -> np.ndarray:
    """Give stations on FES land cells the constants of the nearest station with values."""
    ok = np.isfinite(hc).all(axis=1)
    if ok.all():
        return hc
    tree = BallTree(np.radians(lonlat[ok][:, ::-1]), metric="haversine")
    _, i = tree.query(np.radians(lonlat[~ok][:, ::-1]), k=1)
    hc = hc.copy()
    hc[~ok] = hc[ok][i[:, 0]]
    return hc


def consistent_version_flag(mh22: np.ndarray, mh14: np.ndarray, lonlat: np.ndarray,
                            fes2022_ok_lonlat: np.ndarray | None = None) -> np.ndarray:
    """Version rule, extended to neighbours of flagged stations that fail only its minimum difference.

    A station within VERSION_GROW_KM of a flagged one is flagged too when FES2022 exceeds FES2014
    by the rule's ratio and by VERSION_GROW_DIFF_M; repeated until no station is added. Without
    it, neighbours 3 km apart on the same FES artefact take different atlases (~1 m of MHWS).
    Stations within VERSION_VETO_KM of a gauge in `fes2022_ok_lonlat` (FES2022 matches it) are
    never added.
    """
    flag = np.asarray(version_flag(mh22, mh14), dtype=bool)
    weaker = np.asarray(version_flag(mh22, mh14, diff_m=VERSION_GROW_DIFF_M), dtype=bool)
    if fes2022_ok_lonlat is not None and len(fes2022_ok_lonlat):
        d, _ = BallTree(np.radians(fes2022_ok_lonlat[:, ::-1]), metric="haversine").query(np.radians(lonlat[:, ::-1]))
        weaker &= d[:, 0] * EARTH_RADIUS_KM > VERSION_VETO_KM
    tree = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine")
    while True:
        cand = np.flatnonzero(weaker & ~flag)
        if not len(cand):
            return flag
        near = tree.query_radius(np.radians(lonlat[cand, ::-1]), r=VERSION_GROW_KM / EARTH_RADIUS_KM)
        add = cand[[flag[n].any() for n in near]]
        if not len(add):
            return flag
        flag[add] = True


def _prepare(out: Path, sources: Path, stations: pd.DataFrame, workers: int) -> dict:
    lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
    model22 = pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL)
    model14 = reference_model("fes2014")
    cache = out / "fes_constants.npz"
    if cache.exists() and np.isin(stations.index, np.load(cache)["station"]).all():
        # fill on every cached station, so that a subset run borrows from the true neighbour
        c = np.load(cache)
        cached = pd.read_csv(out / "stations.csv", index_col="station").loc[c["station"], ["lon", "lat"]]
        k = np.searchsorted(c["station"], stations.index)
        hc22, hc14 = (fill_masked(c[a], cached.to_numpy(dtype=float))[k] for a in ("fes2022", "fes2014"))
    else:
        hc22, hc14 = fes_constants(model22, lonlat), grouped_constants(model14, lonlat)
        np.savez(cache, station=stations.index.to_numpy(), fes2022=hc22, fes2014=hc14)
        hc22, hc14 = fill_masked(hc22, lonlat), fill_masked(hc14, lonlat)
    for hc, model in ((hc22, model22), (hc14, model14)):
        cons = list(model.constituents)
        hc[:, [cons.index(c) for c in SEASONAL if c in cons]] = 0.0
    hourly_basis(model22, sources)
    mhws_cache = out / "fes_mhws.npz"
    if mhws_cache.exists() and np.isin(stations.index, np.load(mhws_cache)["station"]).all():
        c = np.load(mhws_cache)
        k = np.searchsorted(c["station"], stations.index)
        mh22, mh14 = c["fes2022"][k], c["fes2014"][k]
        for mh, hc, paths in ((mh22, hc22, (FES_CACHE / "basis_re.npy", FES_CACHE / "basis_im.npy")),
                              (mh14, hc14, atlas_basis(model14, "fes2014"))):
            miss = ~np.isfinite(mh) & np.isfinite(hc).all(axis=1)
            if miss.any():
                mh[miss] = _mhws(paths, hc[miss], workers)
    else:
        print("MHWS of both atlases", flush=True)
        mh22 = _mhws((FES_CACHE / "basis_re.npy", FES_CACHE / "basis_im.npy"), hc22, workers)
        mh14 = _mhws(atlas_basis(model14, "fes2014"), hc14, workers)
        np.savez(mhws_cache, station=stations.index.to_numpy(), fes2022=mh22, fes2014=mh14)
    g = pd.read_csv(sources / "analysis" / "atlas_gauges.csv")
    g = g[(g.fes_gauge__mhws_m - g.gesla_harm__mhws_m).abs() <= FES2022_OK_M]
    use14 = consistent_version_flag(mh22, mh14, lonlat, g[["lon", "lat"]].to_numpy(dtype=float))
    if use14.any():
        hourly_basis(model14, FES2014_CACHE)
    print(f"FES2014 at {use14.sum()} stations", flush=True)
    tc_levels = coast_rp_levels("COAST-RP_TC.nc")
    tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
    with_tc = stations[stations.station_id.isin(tc_levels.index)]
    coast = coast_rp_levels("COAST-RP.nc").to_numpy()
    return {"coast": dict(zip(stations.index, coast[stations.index])),
            "tc_hours": tc_hours(with_tc, len(np.load(sources / "hourly_basis_re.npy", mmap_mode="r"))),
            "hc22": hc22, "hc14": hc14, "mh22": mh22, "mh14": mh14, "use14": use14, "tc_levels": tc_levels,
            "tc_shapes": tc_donor_shapes(sources, stations)}


def _station_input(prep: dict, stations: pd.DataFrame, k: int, r: float, key) -> dict:
    s, use14 = stations.iloc[k], prep["use14"][k]
    tc_levels = prep["tc_levels"]
    st = stations.index[k]
    return {"station": key, "gtsm_index": int(s.gtsm_index), "atlas": "fes2014" if use14 else "fes2022",
            "coast_rp": prep["coast"][st], "tc_hours": prep["tc_hours"].get(st),
            "hc": prep["hc14"][k] if use14 else prep["hc22"][k],
            "mhws_m": float(prep["mh14"][k] if use14 else prep["mh22"][k]), "r": r,
            "tc": tc_levels.loc[s.station_id].to_numpy() if s.station_id in tc_levels.index else None,
            "tc_shape": prep["tc_shapes"].get(st)}


def _run(out: Path, sources: Path, todo: list, use14: bool, workers: int, hyd=None, row_of=None):
    tasks = [(str(out), str(sources), todo[i:i + STATIONS_PER_TASK], use14)
             for i in range(0, len(todo), STATIONS_PER_TASK)]
    rows, info = [], []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(_task, tasks)):
            for row, h, i in res:
                rows.append(row)
                info += i
                if hyd is not None:
                    hyd[row_of[row["station"]]] = h
            if k % 20 == 0:
                print(f"  {min((k + 1) * STATIONS_PER_TASK, len(todo))}/{len(todo)} stations", flush=True)
    return rows, info


def build(out: Path, sources: Path, workers: int, limit: int | None, only: list[int] | None = None) -> None:
    """Build all stations (or the first `limit`), or rebuild `only` these in the existing outputs."""
    every = pd.read_csv(out / "stations.csv", index_col="station")
    every = every.iloc[:limit] if limit else every
    # the version rule looks at neighbours, so the atlas choice is always made on every station
    prep = _prepare(out, sources, every, workers)
    pos = [every.index.get_loc(s) for s in only] if only else list(range(len(every)))
    stations = every.iloc[pos]
    gauges = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
    ratios = station_ratios(stations, gauges)
    todo = [_station_input(prep, every, k, float(ratios.r.iloc[j]), every.index[k]) for j, k in enumerate(pos)]
    print(f"{len(todo)} stations", flush=True)
    if only:
        hyd = np.load(out / "hydrographs.npy", mmap_mode="r+")
        hyd[pos] = np.nan
        row_of = dict(zip(every.index, range(len(every))))
    else:
        hyd = np.lib.format.open_memmap(out / "hydrographs.npy", mode="w+", dtype=np.float32,
                                        shape=(len(stations), len(RPS), N_STEPS))
        hyd[:] = np.nan
        row_of = {st: k for k, st in enumerate(stations.index)}
    rows, info = _run(out, sources, todo, bool(prep["use14"].any()), workers, hyd, row_of)
    hyd.flush()
    info = pd.DataFrame(info)
    if only:
        kept = pd.read_parquet(out / "hydrograph_info.parquet")
        info = pd.concat([kept[~kept.station.isin(only)], info]).sort_values(["station", "rp"], kind="stable")
    info.reset_index(drop=True).to_parquet(out / "hydrograph_info.parquet")
    mh22, mh14, tc_levels = prep["mh22"][pos], prep["mh14"][pos], prep["tc_levels"]
    # station_id is not unique in COAST-RP (offshore "currents_*" points), so its rows join by position
    coast, etc = coast_rp_levels("COAST-RP.nc"), coast_rp_levels("COAST-RP_ETC.nc")
    lev = stations.join(ratios).join(pd.DataFrame(rows).set_index("station"))
    lev["mhws_fes2022_m"], lev["mhws_fes2014_m"] = mh22, mh14
    for rp in RPS:
        lev[f"coast_rp_rp{rp}_m"] = coast[rp].to_numpy()[lev.index]
        lev[f"coast_rp_etc_rp{rp}_m"] = etc[rp].to_numpy()[lev.index]
        lev[f"coast_rp_tc_rp{rp}_m"] = tc_levels[rp].reindex(lev.station_id).to_numpy()
    if only:
        kept = pd.read_parquet(out / "levels.parquet")
        lev = pd.concat([kept.drop(index=only), lev[kept.columns.intersection(lev.columns)]]).sort_index()
    lev.to_parquet(out / "levels.parquet")
    summarise(lev)


def gauge_check(out: Path, sources: Path, workers: int) -> None:
    """Levels at the test 9 gauges with r left one out, against the gauge and COAST-RP.

    Each gauge takes its nearest station within GAUGE_MAX_KM; r is rebuilt without that gauge.
    The reference is the gauge's own skew-surge convolution (gauge surge + gauge tide).
    """
    stations = pd.read_csv(out / "stations.csv", index_col="station")
    gauges = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
    lev = pd.read_parquet(out / "levels.parquet")
    prep = _prepare(out, sources, stations, workers)
    d, i = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine").query(
        np.radians(gauges[["lat", "lon"]].to_numpy()), k=1)
    todo, meta = [], []
    for name, k, dist in zip(gauges.index, i[:, 0], d[:, 0] * EARTH_RADIUS_KM):
        if dist > GAUGE_MAX_KM:
            continue
        r = float(station_ratios(stations.iloc[[k]], gauges.drop(index=name)).r.iloc[0])
        todo.append(_station_input(prep, stations, k, r, name))
        meta.append({"file_name": name, "station": stations.index[k], "station_km": dist, "r_loo": r})
    print(f"{len(todo)} of {len(gauges)} gauges within {GAUGE_MAX_KM:.0f} km of a station", flush=True)
    rows, _ = _run(out, sources, todo, bool(prep["use14"].any()), workers)
    loo = pd.DataFrame(rows).set_index("station")
    res = pd.DataFrame(meta).set_index("file_name")
    res = res.join(gauges[["region", "lat", "lon", "station_class", "calibrate", "ratio_fes"]])
    at = lev.loc[res.station]
    res["has_tc"] = np.isfinite(at["coast_rp_tc_rp100_m"]).to_numpy()
    for rp in GAUGE_RPS:
        res[f"gauge_rp{rp}_m"] = gauges.loc[res.index, f"gauge_gauge_rp{rp}_m"]
        res[f"gev_rp{rp}_m"] = gauges.loc[res.index, f"gev_gauge_rp{rp}_m"]
        res[f"coast_rp_rp{rp}_m"] = at[f"coast_rp_rp{rp}_m"].to_numpy()
        res[f"new_rp{rp}_m"] = at[f"rp{rp}_m"].to_numpy()
        res[f"new_loo_rp{rp}_m"] = loo.loc[res.index, f"rp{rp}_m"]
        res[f"new_loo_r1_rp{rp}_m"] = loo.loc[res.index, f"etc_raw_rp{rp}_m"]
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


def summarise(lev: pd.DataFrame) -> None:
    rows = []
    has_tc = np.isfinite(lev["coast_rp_tc_rp100_m"])
    for rp in RPS:
        for label, sel in (("all", np.ones(len(lev), bool)), ("no TC part", ~has_tc), ("TC part", has_tc)):
            for name, col, ref in (("new - COAST-RP", f"rp{rp}_m", f"coast_rp_rp{rp}_m"),
                                   ("ETC, r = 1 - COAST-RP ETC", f"etc_raw_rp{rp}_m", f"coast_rp_etc_rp{rp}_m"),
                                   ("ETC, r - COAST-RP ETC", f"etc_rp{rp}_m", f"coast_rp_etc_rp{rp}_m")):
                d = (lev[col] - lev[ref])[sel].dropna()
                rows.append({"rp": rp, "stations": label, "comparison": name, "n": len(d), "median": d.median(),
                             "p10": d.quantile(0.1), "p90": d.quantile(0.9), "abs_gt_0.5m": int((d.abs() > 0.5).sum())})
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"r: median {lev.r.median():.2f}, capped {int((lev.r_uncapped > R_CAP).sum())}, "
          f"global fallback {int((lev.r_neighbours < MIN_NEIGHBOURS).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("stations", "build", "gauges"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first N stations (trial run)")
    parser.add_argument("--stations", type=int, nargs="*", default=None, help="only these station indices")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "stations":
        match_stations(out)
    elif args.stage == "build":
        build(out, Path(args.sources_dir), args.workers, args.limit, args.stations)
    else:
        gauge_check(out, Path(args.sources_dir), args.workers)


if __name__ == "__main__":
    main()
