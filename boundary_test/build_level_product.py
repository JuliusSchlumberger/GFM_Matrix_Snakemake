"""Tide-corrected RP levels and per-RP hydrographs at every COAST-RP station (INPUT_ANALYSIS.md §8).

Levels (tests 5 and 9): skew surges of the GTSM-ERA5 surge residual (1979-2024) laid on the
hourly FES tide, a GPD on their declustered tail (`surge_correction.surge_model`), positive skew
surges scaled by the gauge ratio r, convolved with the FES high waters of one nodal cycle. r is
the inverse-distance mean of `ratio_fes` at the non-TC-dominated gauges within IDW_RADIUS_KM
(at least MIN_NEIGHBOURS, else their median), capped at R_CAP: ratios above it rest on skew
surges of a few decimetres. The atlas is FES2022, or FES2014 where the version rule fires
(`atlas_rule_sensitivity.version_flag`, test 6); gauges do not override it here. TC: the annual
exceedance rate of COAST-RP's TC component is added to the ETC rate, as COAST-RP combines its
own two parts (its total equals that rate sum to 1 mm); the TC levels keep GTSM's tide.
RP levels solve rate(z) = -ln(1 - 1/RP), relative to local MSL (seasonal cycle in the surge).

Hydrographs (§8.3, generator_smooth): the FES tide around the high water nearest the median
tide-at-peak of the station's annual maxima of the RP window (k x MHWS of the validation;
`rp_event_selection`; RP >= 50 takes the maxima of RP >= 20), plus the median 13-h smoothed
surge of those maxima, scaled so the peak is the RP level. Where the TC rate dominates at that
level the surge is a surge-dominated gauge's median event within TC_GAUGE_KM, else a Gaussian.
Annual maxima are of GTSM surge + FES tide (GTSM's own tide is not extracted at every station).

Writes <out-dir>/levels.parquet (one row per station), hydrographs.npy (station, RP, 10-min step
over +-EXTENDED_HR, NaN outside the station's window), hydrograph_info.parquet, fes_constants.npz.

Usage:
    python boundary_test/build_level_product.py stations --out-dir .cleanup_baseline/level_product
    python boundary_test/extract_gtsm_series.py --out-dir .cleanup_baseline/level_product \
        --stations-csv .cleanup_baseline/level_product/stations.csv --variable surge
    python boundary_test/build_level_product.py build --out-dir .cleanup_baseline/level_product \
        --sources-dir .cleanup_baseline/level_sources [--workers 3] [--limit N]
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
MIN_EVENTS = 10
STATIONS_PER_TASK = 25
N_STEPS = 2 * EXTENDED_HR * STEPS_PER_HR + 1


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


def _station(st: dict, g: np.ndarray, th: np.ndarray, t10: np.ndarray, years: np.ndarray, tc_shape) -> tuple[dict, np.ndarray, list]:
    row = {"station": st["station"], "atlas": st["atlas"], "mhws_m": st["mhws_m"]}
    hyd = np.full((len(RPS), N_STEPS), np.nan, dtype=np.float32)
    info = []
    hw, h = high_waters(th)
    skew = skew_surges(g + th, hw, h)
    cover = pd.Series(np.isfinite(skew)).groupby(years[hw]).mean()
    keep_years = cover.index[cover >= YEAR_COVERAGE]
    ok = np.isin(years[hw], keep_years) & np.isfinite(skew)
    model = surge_model(skew[ok], len(keep_years))
    corrected = scaled(model, st["r"])
    v, w, cyc = tide_peaks(t10, STEP_MIN / 60.0)
    row.update({"n_years": len(keep_years), "gpd_shape": model["shape"], "gpd_scale": model["scale"],
                "theta": model["theta"]})

    def etc_rate(m):
        return lambda z: cyc * m["theta"] * float(np.sum(w * exceedance(m, z - v)))
    tc = _tc_rate(st["tc"]) if st["tc"] is not None else (lambda z: 0.0)
    lo, hi = float(v.min()), float(v.max()) + 20.0
    for rp, target in zip(RPS, annual_rate(RPS)):
        row[f"etc_raw_rp{rp}_m"] = _solve(etc_rate(model), target, lo, hi)
        row[f"etc_rp{rp}_m"] = _solve(etc_rate(corrected), target, lo, hi)
        row[f"rp{rp}_m"] = z = _solve(lambda x: etc_rate(corrected)(x) + tc(x), target, lo, hi)
        row[f"tc_dominant_rp{rp}"] = bool(tc(z) > etc_rate(corrected)(z))

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
        if row[f"tc_dominant_rp{rp}"]:
            source = "tc_gauge" if tc_shape is not None else "tc_gaussian"
            shape = tc_shape if tc_shape is not None else gaussian_shape(TC_HALF_WIDTH_HR, half_hr)
            shape = fit_length(shape, len(tide_win))
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


def build(out: Path, sources: Path, workers: int, limit: int | None) -> None:
    stations = pd.read_csv(out / "stations.csv", index_col="station")
    stations = stations.iloc[:limit] if limit else stations
    lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
    model22 = pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL)
    model14 = reference_model("fes2014")
    cache = out / "fes_constants.npz"
    if cache.exists() and len(np.load(cache)["station"]) == len(stations):
        hc22, hc14 = np.load(cache)["fes2022"], np.load(cache)["fes2014"]
    else:
        hc22, hc14 = fes_constants(model22, lonlat), grouped_constants(model14, lonlat)
        np.savez(cache, station=stations.index.to_numpy(), fes2022=hc22, fes2014=hc14)
    for hc, model in ((hc22, model22), (hc14, model14)):
        cons = list(model.constituents)
        hc[:, [cons.index(c) for c in SEASONAL if c in cons]] = 0.0
    hourly_basis(model22, sources)
    print("MHWS of both atlases", flush=True)
    mh22 = _mhws((FES_CACHE / "basis_re.npy", FES_CACHE / "basis_im.npy"), hc22, workers)
    mh14 = _mhws(atlas_basis(model14, "fes2014"), hc14, workers)
    use14 = np.asarray(version_flag(mh22, mh14), dtype=bool)
    if use14.any():
        hourly_basis(model14, FES2014_CACHE)
    print(f"FES2014 at {use14.sum()} stations", flush=True)

    gauges = pd.read_parquet(sources / "surge_correction" / "sites.parquet")
    ratios = station_ratios(stations, gauges)
    tc_levels = coast_rp_levels("COAST-RP_TC.nc")
    tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
    tc_shapes = tc_donor_shapes(sources, stations)
    todo = []
    for k, (st, s) in enumerate(stations.iterrows()):
        hc = hc14[k] if use14[k] else hc22[k]
        if not np.isfinite(hc).all():
            continue
        todo.append({"station": st, "gtsm_index": int(s.gtsm_index), "atlas": "fes2014" if use14[k] else "fes2022",
                     "hc": hc, "mhws_m": float(mh14[k] if use14[k] else mh22[k]), "r": float(ratios.r.iloc[k]),
                     "tc": tc_levels.loc[s.station_id].to_numpy() if s.station_id in tc_levels.index else None,
                     "tc_shape": tc_shapes.get(st)})
    tasks = [(str(out), str(sources), todo[i:i + STATIONS_PER_TASK], bool(use14.any()))
             for i in range(0, len(todo), STATIONS_PER_TASK)]
    print(f"{len(todo)} stations with FES constants of {len(stations)}", flush=True)
    hyd = np.lib.format.open_memmap(out / "hydrographs.npy", mode="w+", dtype=np.float32,
                                    shape=(len(stations), len(RPS), N_STEPS))
    hyd[:] = np.nan
    row_of = {st: k for k, st in enumerate(stations.index)}
    rows, info = [], []
    with ProcessPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(_task, tasks)):
            for row, h, i in res:
                rows.append(row)
                info += i
                hyd[row_of[row["station"]]] = h
            if k % 20 == 0:
                print(f"  {min((k + 1) * STATIONS_PER_TASK, len(todo))}/{len(todo)} stations", flush=True)
    hyd.flush()
    pd.DataFrame(rows).to_parquet(out / "station_rows.parquet")
    pd.DataFrame(info).to_parquet(out / "hydrograph_info.parquet")
    # station_id is not unique in COAST-RP (offshore "currents_*" points), so its rows join by position
    coast, etc = coast_rp_levels("COAST-RP.nc"), coast_rp_levels("COAST-RP_ETC.nc")
    lev = stations.join(ratios).join(pd.DataFrame(rows).set_index("station"))
    lev["mhws_fes2022_m"], lev["mhws_fes2014_m"] = mh22, mh14
    for rp in RPS:
        lev[f"coast_rp_rp{rp}_m"] = coast[rp].to_numpy()[lev.index]
        lev[f"coast_rp_etc_rp{rp}_m"] = etc[rp].to_numpy()[lev.index]
        lev[f"coast_rp_tc_rp{rp}_m"] = tc_levels[rp].reindex(lev.station_id).to_numpy()
    lev.to_parquet(out / "levels.parquet")
    (out / "station_rows.parquet").unlink()
    summarise(lev)


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
    parser.add_argument("stage", choices=("stations", "build"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first N stations (trial run)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "stations":
        match_stations(out)
    else:
        build(out, Path(args.sources_dir), args.workers, args.limit)


if __name__ == "__main__":
    main()
