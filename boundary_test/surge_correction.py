"""Tests 2, 5 and 9: skew-surge convolution at the gauges and a gauge-calibrated surge correction.

One estimator for every combination of surge and tide (INPUT_ANALYSIS.md §8.1, tests 2 and 5):
skew surge per tidal cycle (highest level in the cycle minus the predicted high water, cycles
bounded halfway between predicted high waters), a GPD on its declustered exceedances above the
THRESHOLD_Q quantile (clusters separated by more than RUN_CYCLES cycles; extremal index =
clusters / exceedances), and the empirical distribution below. Assuming skew surge independent
of the tide, the annual rate of a level z is
    lambda(z) = cycles per year x theta x mean over predicted high waters t_k of P(S > z - t_k),
over the predicted high waters, and the RP level solves
lambda(z) = -ln(1 - 1/RP). Surge sources, on the same years (gauge years with >= YEAR_COVERAGE
of their cycles observed):
  gauge  GESLA record with each year's mean removed, minus its harmonic tide (`gesla_harm`)
  gtsm   GTSM surge residual + the GTSM harmonic tide (`gtsm_harm`), minus that tide
Tides (seasonal Sa / Ssa left in the surge): gauge and GTSM hourly over 1979-2024, FES2022 at the
gauge at FES_STEP_MIN over one nodal cycle. Combinations:
gauge_gauge (the reference), gtsm_gtsm (should reproduce COAST-RP's ETC levels), gtsm_fes (the
tide-corrected level of test 5) and gtsm_gaugetide (only the surge differs from the reference).

Test 9: the surge deficit as the ratio r = gauge / GTSM skew-surge level at RP 10, applied to
GTSM's positive skew surges. Predicted per gauge by leave-one-out from the other gauges: a global
median, or an inverse-distance mean within RADII_KM (gauges on the same GTSM station or within
SAME_SITE_KM excluded, at least MIN_NEIGHBOURS, else the global median). Scored on the total
levels with the gauge tide (surge error only) and with the FES tide (production). Stability: r
from the first and second half of each record. `ratio_fes` is r with GTSM's skew surges taken on
the hourly FES2022 tide instead of GTSM's own (surge + FES tide minus FES tide), the basis of
`build_level_product.py`, where GTSM's tide is not available at every station.

Writes <out-dir>/surge_correction/{sites.parquet, estimator.csv, deficit.csv, correction.csv}.

Usage:
    python boundary_test/surge_correction.py --out-dir .cleanup_baseline/level_sources [--workers 3]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks
from scipy.stats import genpareto
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from fes_tide_stats import STEP_MIN as FES_STEP_MIN  # noqa: E402
from levels_tides import FES_CACHE, HOURLY_START, SEASONAL, on_hourly_grid, remove_annual_means  # noqa: E402

RPS = (2, 10, 50, 100)
THRESHOLD_Q = 0.995
RUN_CYCLES = 3  # ~1.5 days: one storm raises several consecutive high waters
YEAR_COVERAGE = 0.8
MIN_YEARS = 15
RATIO_RP = 10
RADII_KM = (250, 500, 1000)
SAME_SITE_KM = 10.0
MIN_NEIGHBOURS = 3
SITES_PER_TASK = 10
EARTH_RADIUS_KM = 6371.0
COMBINATIONS = {"gauge_gauge": ("gauge", "gauge"), "gtsm_gtsm": ("gtsm", "gtsm"), "gtsm_fes": ("gtsm", "fes"),
                "gtsm_gaugetide": ("gtsm", "gauge")}


def high_waters(tide: np.ndarray, min_prominence: float | None = None,
                steps_per_hr: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the predicted high waters and their parabola-refined heights.

    With `min_prominence`, a local maximum counts only when its prominence is at least that
    fraction of the tidal range over the surrounding 25 h, which drops the low-water bumps of
    shallow-water tides (double low waters) that are otherwise taken for high waters.
    """
    if min_prominence is None:
        i = np.flatnonzero((tide[1:-1] > tide[:-2]) & (tide[1:-1] >= tide[2:])) + 1
    else:
        span = 25 * steps_per_hr + 1
        local_range = maximum_filter1d(tide, span) - minimum_filter1d(tide, span)
        i, props = find_peaks(tide, prominence=0)
        i = i[props["prominences"] >= min_prominence * local_range[i]]
    a, b, c = tide[i - 1], tide[i], tide[i + 1]
    curv = a - 2 * b + c
    return i, b - np.where(curv < 0, (c - a) ** 2 / (8 * curv), 0.0)


def skew_surges(total: np.ndarray, hw: np.ndarray, hw_height: np.ndarray) -> np.ndarray:
    """Skew surge per cycle around each high water; NaN where the cycle is not fully observed."""
    bounds = np.r_[0, (hw[:-1] + hw[1:]) // 2]
    peak = np.maximum.reduceat(np.where(np.isfinite(total), total, -np.inf), bounds)
    gaps = np.add.reduceat(~np.isfinite(total), bounds)
    return np.where(gaps == 0, peak - hw_height, np.nan)


def surge_model(s: np.ndarray, n_years: float, max_shape: float | None = None) -> dict:
    """Empirical body + GPD tail of per-cycle skew surges, with the extremal index.

    With `max_shape`, a fitted shape above it is fixed at max_shape and the scale refitted.
    """
    u = float(np.quantile(s, THRESHOLD_Q))
    exc = np.flatnonzero(s > u)
    starts = np.r_[0, np.flatnonzero(np.diff(exc) > RUN_CYCLES) + 1]
    cmax = np.maximum.reduceat(s[exc], starts)
    c, _, scale = genpareto.fit(cmax - u, floc=0)
    shape_mle = float(c)
    if max_shape is not None and c > max_shape:
        c, _, scale = genpareto.fit(cmax - u, f0=max_shape, floc=0)
    return {"sorted": np.sort(s), "u": u, "p_u": len(exc) / len(s), "shape": float(c), "scale": float(scale),
            "shape_mle": shape_mle, "theta": len(starts) / len(exc), "cycles_per_yr": len(s) / n_years,
            "n_years": n_years}


def scaled(model: dict, r: float) -> dict:
    """The model of r x S for S > 0 (the negative skew surges stay as they are)."""
    srt = model["sorted"]
    return {**model, "sorted": np.sort(np.where(srt > 0, r * srt, srt)), "u": model["u"] * r if model["u"] > 0
            else model["u"], "scale": model["scale"] * r}


def exceedance(model: dict, s: np.ndarray) -> np.ndarray:
    srt = model["sorted"]
    body = 1.0 - np.searchsorted(srt, s, side="right") / len(srt)
    tail = model["p_u"] * genpareto.sf(s - model["u"], model["shape"], 0, model["scale"])
    return np.where(s > model["u"], tail, body)


def total_level(model: dict, peaks: np.ndarray, weights: np.ndarray, cycles_per_yr: float, rp: float) -> float:
    target = -np.log1p(-1.0 / rp)
    lo, hi = peaks.min(), peaks.max() + 20.0
    for _ in range(50):
        z = 0.5 * (lo + hi)
        rate = cycles_per_yr * model["theta"] * np.sum(weights * exceedance(model, z - peaks))
        lo, hi = (z, hi) if rate > target else (lo, z)
    return 0.5 * (lo + hi)


def skew_level(model: dict, rp: float) -> float:
    """Skew surge exceeded at a rate of -ln(1 - 1/rp) clusters per year."""
    rate_u = model["cycles_per_yr"] * model["theta"] * model["p_u"]
    p = -np.log1p(-1.0 / rp) / rate_u
    return float(model["u"] + genpareto.isf(p, model["shape"], 0, model["scale"])) if p < 1 else np.nan


def tide_peaks(tide: np.ndarray, step_hr: float,
               min_prominence: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """High-water heights binned to 1 cm (values, weights) and high waters per year."""
    _, h = high_waters(tide, min_prominence, int(round(1.0 / step_hr)))
    v, n = np.unique(np.round(h, 2), return_counts=True)
    return v, n / n.sum(), len(h) / (len(tide) * step_hr / 8766.0)


def _site_models(out: Path, site: dict, surge: np.ndarray, surge_idx: np.ndarray, pos: dict, bases: dict) -> dict | None:
    data = np.load(out / "gesla" / f"{site['file_name']}.npz")
    idx, z = on_hourly_grid(data["t"], data["z"])
    if len(idx) == 0:
        return None
    n_hours = bases["b_re"].shape[0]
    tides = {k: np.asarray(bases[b + "_re"]) @ site[h].real + np.asarray(bases[b + "_im"]) @ site[h].imag
             for k, h, b in (("gauge", "hc_gauge", "b"), ("gtsm", "hc_gtsm", "b"), ("fes", "hc_fes", "f"),
                             ("fes_hourly", "hc_fes", "b"))}
    obs = np.full(n_hours, np.nan)
    obs[idx] = remove_annual_means(idx, z)
    gtsm = np.full(n_hours, np.nan)
    gtsm[surge_idx] = station_series(surge, pos[site["gtsm_index"]])
    year = (HOURLY_START + np.arange(n_hours).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    skew, years = {}, {}
    for name, total, tide in (("gauge", obs, tides["gauge"]), ("gtsm", gtsm + tides["gtsm"], tides["gtsm"]),
                              ("gtsmfes", gtsm + tides["fes_hourly"], tides["fes_hourly"])):
        hw, height = high_waters(tide)
        skew[name] = skew_surges(total, hw, height)
        yr = pd.Series(np.isfinite(skew[name])).groupby(year[hw]).mean()
        years[name] = (year[hw], set(yr.index[yr >= YEAR_COVERAGE]))
    common = sorted(years["gauge"][1] & years["gtsm"][1])
    if len(common) < MIN_YEARS:
        return None
    half = common[len(common) // 2]
    models = {}
    for name in ("gauge", "gtsm"):
        yr, s = years[name][0], skew[name]
        for part, keep in (("all", np.isin(yr, common)), ("first", np.isin(yr, common) & (yr < half)),
                           ("second", np.isin(yr, common) & (yr >= half))):
            ok = keep & np.isfinite(s)
            n_yr = len([y for y in common if (part == "all") or (part == "first") == (y < half)])
            models[(name, part)] = surge_model(s[ok], n_yr)
    yr, s = years["gtsmfes"][0], skew["gtsmfes"]
    models[("gtsmfes", "all")] = surge_model(s[np.isin(yr, common) & np.isfinite(s)], len(common))
    steps = {"gauge": 1.0, "gtsm": 1.0, "fes": FES_STEP_MIN / 60.0}
    return {"models": models, "peaks": {k: tide_peaks(t, steps[k]) for k, t in tides.items() if k in steps},
            "n_years": len(common)}


def _site_task(args) -> list[dict]:
    out, sites = args
    out = Path(out)
    bases = {"b_re": np.load(out / "hourly_basis_re.npy", mmap_mode="r"),
             "b_im": np.load(out / "hourly_basis_im.npy", mmap_mode="r"),
             "f_re": np.load(FES_CACHE / "basis_re.npy", mmap_mode="r"),
             "f_im": np.load(FES_CACHE / "basis_im.npy", mmap_mode="r")}
    surge_st, surge_time, surge = load_series(out, "surge")
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    rows = []
    for site in sites:
        try:
            m = _site_models(out, site, surge, surge_idx, pos, bases)
        except Exception as err:  # one bad site must not stop the pool; it is reported and skipped
            print(f"  skipped {site['file_name']}: {type(err).__name__}: {err}", flush=True)
            continue
        if m is not None:
            rows.append({"file_name": site["file_name"], **m})
    return rows


def levels_row(m: dict, ratio: float = 1.0) -> dict:
    """Total RP levels of every combination, the GTSM surge scaled by `ratio`."""
    row = {}
    for combo, (surge, tide) in COMBINATIONS.items():
        model = m["models"][(surge, "all")]
        if surge == "gtsm" and ratio != 1.0:
            model = scaled(model, ratio)
        v, w, cyc = m["peaks"][tide]
        for rp in RPS:
            row[f"{combo}_rp{rp}_m"] = total_level(model, v, w, cyc, rp)
    return row


def loo_ratio(sites: pd.DataFrame, radius_km: float | None) -> np.ndarray:
    """Leave-one-out prediction of each site's skew-surge ratio from the other calibration sites."""
    cal = sites[sites.calibrate & np.isfinite(sites.ratio)]
    ll = np.radians(cal[["lat", "lon"]].to_numpy())
    tree = BallTree(ll, metric="haversine")
    pred = np.full(len(sites), np.nan)
    for k, (_, s) in enumerate(sites.iterrows()):
        others = cal[(cal.index != s.name) & (cal.gtsm_index != s.gtsm_index)]
        fallback = float(others.ratio.median())
        if radius_km is None:
            pred[k] = fallback
            continue
        i, d = tree.query_radius(np.radians([[s.lat, s.lon]]), r=radius_km / EARTH_RADIUS_KM, return_distance=True)
        i, d = i[0], d[0] * EARTH_RADIUS_KM
        nb = cal.iloc[i]
        keep = (nb.index != s.name) & (nb.gtsm_index != s.gtsm_index).to_numpy() & (d > SAME_SITE_KM)
        if keep.sum() < MIN_NEIGHBOURS:
            pred[k] = fallback
            continue
        w = 1.0 / d[keep]
        pred[k] = float(np.sum(w * nb.ratio.to_numpy()[keep]) / w.sum())
    return pred


def _stats(x: pd.Series) -> pd.Series:
    x = x.dropna()
    return pd.Series({"n": len(x), "median": x.median(), "mae": x.abs().mean(), "p10": x.quantile(0.1),
                      "p90": x.quantile(0.9)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sites (trial run)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "surge_correction"
    dest.mkdir(exist_ok=True)
    sets = pd.read_csv(out / "analysis" / "site_sets.csv", index_col="file_name")
    sets = sets[sets.extremes]
    pairs = pd.read_parquet(out / "pairs.parquet").set_index("file_name")
    hc = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    pos = {f: i for i, f in enumerate(hc["file_name"])}
    seasonal = [list(hc["constituents"]).index(c) for c in SEASONAL]
    sites = []
    for f in sets.index:
        if f not in pos:
            continue
        h = {k: hc[src][pos[f]].copy() for k, src in (("hc_gauge", "gesla_harm"), ("hc_gtsm", "gtsm_harm"),
                                                         ("hc_fes", "fes_gauge"))}
        if not all(np.isfinite(v).all() for v in h.values()):
            continue
        for v in h.values():
            v[seasonal] = 0.0
        sites.append({"file_name": f, "gtsm_index": int(pairs.loc[f, "gtsm_index"]), **h})
    sites = sites[:args.limit] if args.limit else sites
    print(f"{len(sites)} sites", flush=True)
    chunks = [(str(out), sites[i:i + SITES_PER_TASK]) for i in range(0, len(sites), SITES_PER_TASK)]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(_site_task, chunks)):
            results += r
            print(f"  {min((k + 1) * SITES_PER_TASK, len(sites))}/{len(sites)} sites", flush=True)
    by_site = {r["file_name"]: r for r in results}

    ex = pd.read_parquet(out / "extremes.parquet").set_index(["file_name", "source"])
    rows = []
    for f, m in by_site.items():
        g, t = m["models"][("gauge", "all")], m["models"][("gtsm", "all")]
        e = ex.loc[(f, "gesla")]
        etc, tc = e[f"coast_rp_etc_rp{100}_m"], e[f"coast_rp_tc_rp{100}_m"]
        row = {"file_name": f, "n_years": m["n_years"], "region": sets.loc[f, "region"],
               "lat": sets.loc[f, "lat"], "lon": sets.loc[f, "lon"], "gtsm_index": int(pairs.loc[f, "gtsm_index"]),
               "station_class": ("tc_dominated" if np.isfinite(tc) and tc > etc else
                                 "etc_with_tc" if np.isfinite(tc) else "no_tc"),
               "theta_gauge": g["theta"], "theta_gtsm": t["theta"], "shape_gauge": g["shape"], "shape_gtsm": t["shape"]}
        for rp in RPS:
            row[f"skew_gauge_rp{rp}_m"] = skew_level(g, rp)
            row[f"skew_gtsm_rp{rp}_m"] = skew_level(t, rp)
            row[f"skew_gtsmfes_rp{rp}_m"] = skew_level(m["models"][("gtsmfes", "all")], rp)
            row[f"gev_gauge_rp{rp}_m"] = e[f"rp{rp}_m"]
            row[f"coast_rp_etc_rp{rp}_m"] = e.get(f"coast_rp_etc_rp{rp}_m", np.nan)
        for part in ("first", "second"):
            row[f"ratio_{part}"] = (skew_level(m["models"][("gauge", part)], RATIO_RP)
                                    / skew_level(m["models"][("gtsm", part)], RATIO_RP))
        rows.append(row | levels_row(m))
    s = pd.DataFrame(rows).set_index("file_name")
    s["ratio"] = s[f"skew_gauge_rp{RATIO_RP}_m"] / s[f"skew_gtsm_rp{RATIO_RP}_m"]
    s["ratio_fes"] = s[f"skew_gauge_rp{RATIO_RP}_m"] / s[f"skew_gtsmfes_rp{RATIO_RP}_m"]
    s["calibrate"] = s.station_class != "tc_dominated"

    est = []
    for rp in RPS:
        ref = s[f"gauge_gauge_rp{rp}_m"]
        for label, x in (("convolution - gauge GEV (estimator)", ref - s[f"gev_gauge_rp{rp}_m"]),
                         ("gtsm_gtsm - COAST-RP ETC (estimator)", s[f"gtsm_gtsm_rp{rp}_m"] - s[f"coast_rp_etc_rp{rp}_m"]),
                         ("gtsm_gtsm - gauge (model total)", s[f"gtsm_gtsm_rp{rp}_m"] - ref),
                         ("gtsm_gaugetide - gauge (surge)", s[f"gtsm_gaugetide_rp{rp}_m"] - ref),
                         ("gtsm_gtsm - gtsm_gaugetide (GTSM tide)", s[f"gtsm_gtsm_rp{rp}_m"] - s[f"gtsm_gaugetide_rp{rp}_m"]),
                         ("gtsm_fes - gtsm_gaugetide (FES tide)", s[f"gtsm_fes_rp{rp}_m"] - s[f"gtsm_gaugetide_rp{rp}_m"]),
                         ("gtsm_fes - gauge (production)", s[f"gtsm_fes_rp{rp}_m"] - ref),
                         ("skew surge on FES tide - on GTSM tide", s[f"skew_gtsmfes_rp{rp}_m"] - s[f"skew_gtsm_rp{rp}_m"])):
            for cls, x_c in x.groupby(s.station_class):
                est.append({"comparison": label, "rp": rp, "station_class": cls, **_stats(x_c)})
            est.append({"comparison": label, "rp": rp, "station_class": "all", **_stats(x)})
    est = pd.DataFrame(est)
    est.round(3).to_csv(dest / "estimator.csv", index=False)

    deficit = []
    for rp in RPS:
        d = s[f"skew_gauge_rp{rp}_m"] - s[f"skew_gtsm_rp{rp}_m"]
        r = s[f"skew_gauge_rp{rp}_m"] / s[f"skew_gtsm_rp{rp}_m"]
        for key, grp in ([("station_class", c) for c in s.station_class.unique()]
                         + [("region", c) for c in s.region.unique()]):
            sel = s[key] == grp
            deficit.append({"rp": rp, "group": f"{key}={grp}", "n": int(sel.sum()), "skew_gauge_m": s.loc[sel, f"skew_gauge_rp{rp}_m"].median(),
                            "skew_gtsm_m": s.loc[sel, f"skew_gtsm_rp{rp}_m"].median(), "deficit_m": d[sel].median(),
                            "ratio": r[sel].median(), "ratio_p10": r[sel].quantile(0.1), "ratio_p90": r[sel].quantile(0.9)})
    deficit = pd.DataFrame(deficit)
    deficit.round(3).to_csv(dest / "deficit.csv", index=False)

    corr = []
    preds = {"none": np.ones(len(s)), "global": loo_ratio(s, None),
             **{f"idw{r:g}km": loo_ratio(s, r) for r in RADII_KM}}
    for name, pred in preds.items():
        s[f"pred_ratio_{name}"] = pred
        lev = pd.DataFrame([levels_row(by_site[f], p) for f, p in zip(s.index, pred)], index=s.index)
        for rp in RPS:
            ref = s[f"gauge_gauge_rp{rp}_m"]
            for label, col in (("surge only (gauge tide)", "gtsm_gaugetide"), ("production (FES tide)", "gtsm_fes")):
                x = lev[f"{col}_rp{rp}_m"] - ref
                for cls in ("calibration", "tc_dominated"):
                    sel = s.calibrate if cls == "calibration" else ~s.calibrate
                    corr.append({"correction": name, "rp": rp, "levels": label, "sites": cls, **_stats(x[sel])})
    corr = pd.DataFrame(corr)
    corr.round(3).to_csv(dest / "correction.csv", index=False)
    s.to_parquet(dest / "sites.parquet")

    cal = s[s.calibrate]
    nn = BallTree(np.radians(cal[["lat", "lon"]].to_numpy()), metric="haversine")
    d, i = nn.query(np.radians(cal[["lat", "lon"]].to_numpy()), k=6)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print("\nEstimator and error decomposition (m)")
        print(est[est.rp.isin([10, 100])].round(3).to_string(index=False))
        print("\nSkew-surge deficit (gauge vs GTSM, same years)")
        print(deficit[deficit.rp.isin([10, 100])].round(3).to_string(index=False))
        print("\nTest 9: leave-one-out correction (total level - gauge convolution, m)")
        print(corr[corr.rp.isin([10, 100])].round(3).to_string(index=False))
        print("\nStability of the ratio (calibration sites)")
        ok = np.isfinite(cal.ratio_first) & np.isfinite(cal.ratio_second)
        print(f"  first vs second half: r = {np.corrcoef(cal.ratio_first[ok], cal.ratio_second[ok])[0, 1]:.2f}, "
              f"median |diff| {np.median(np.abs(cal.ratio_first[ok] - cal.ratio_second[ok])):.3f}")
        for k in range(1, 6):
            far = d[:, k] * EARTH_RADIUS_KM
            sel = far > SAME_SITE_KM
            r_nb = cal.ratio.to_numpy()[i[:, k]]
            print(f"  neighbour {k}: median distance {np.median(far[sel]):.0f} km, "
                  f"r = {np.corrcoef(cal.ratio.to_numpy()[sel], r_nb[sel])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
