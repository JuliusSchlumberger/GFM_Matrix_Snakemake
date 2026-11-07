"""Acceptance test of the per-RP storm-tide hydrograph generator against the gauge events (§8.3).

Per quality site (tide set, `levels_sites.csv`) and target RP in TARGET_RPS, hydrographs are built
at the paired station and compared with the gauge's annual maxima of the matching empirical-RP
window (`storm_tide_hydrograph.rp_event_selection`):

  generator        FES2022 tide at the gauge position around the high water nearest k x MHWS,
                   k = median tide-at-peak / MHWS of the station's GTSM annual maxima of that RP
                   window; surge = median normalised GTSM surge of those maxima, peak placed on
                   the high water; window +-72 h, +-EXTENDED_HR where >= CENSORED_SHARE of the
                   station's GTSM maxima stay above MSL from the peak to either +-72 h edge.
  generator_phase  as generator, but each surge keeps its observed timing relative to the
                   water-level peak.
  generator_durations  as generator, but the surge shape from the mean time the events stay above
                   each fraction of their peak (`duration_mean_shape`) instead of the median profile.
  generator_smooth as generator, with the surge smoothed by a SMOOTH_HR running mean before the
                   shape is built (removes the tidal-period ripple of the non-tidal residual).
  generator_gauge_peak  diagnostic: as generator, scaled to the median peak of the gauge's own events
                   of that RP window instead of COAST-RP (separates the shape from the peak level).
  swap_tide, swap_surge, gtsm_event  diagnostics, per GTSM maximum of the RP window (up to
                   SWAP_EVENTS, ETC RPs only), measures as the median over those events:
                   swap_tide = the event's own GTSM tide (high water nearest its peak, scaled by
                   MHWS_FES / MHWS_GTSM) + the generator_smooth surge shape; swap_surge = the
                   generator's FES tide window + the event's own smoothed surge (normalised on its
                   maximum); swap_both = the event's own tide and surge, scaled to COAST-RP;
                   gtsm_event = the event's GTSM level itself. Steps from gtsm_event to swap_both,
                   one change each: step_scaled (own tide and raw surge in their observed phase,
                   surge scaled to COAST-RP), step_smoothed (+ 13-h running mean of the surge),
                   step_recentred (+ surge maximum moved onto the nearest high water); swap_both
                   adds the MHWS_FES / MHWS_GTSM tide scaling. Which swap recovers
                   gtsm_event's measures locates a generator deficit in its tide or its surge.
  hgrapher         the HGRAPHER port (`cell_hydrograph`, per_rp): spring high water nearest MHWS,
                   level-by-level mean limb durations of GTSM surge events of matching size.
  coast_hg         COAST-HG spring tide (RP100 only).

The peak is COAST-RP at that RP. Where every selected GTSM maximum has no positive surge (pure
spring-tide maxima) the generator falls back to the Gaussian and flags no_surge_shape. At RPs where COAST-RP's TC part exceeds its ETC part the surge is
the median surge of surge-dominated events (surge share >= TC_SURGE_SHARE) at another quality
gauge within TC_GAUGE_KM, otherwise a Gaussian of TC_HALF_WIDTH_HR; the scored gauge's own events
are never used (no circularity). FES2014 is not applied: the version rule fires at no quality
gauge's own FES value in this set beyond Juneau, Skagway and Port Alberni.

Measures per hydrograph and per gauge event, on the hourly +-72 h profile relative to local MSL
and normalised by its peak: T(0.25), T(0.5) (hours within f x peak of the peak) and nb24 (highest
level / peak 21-28 h from the peak, both sides averaged), ex090 and ex075 (time integral of the
normalised level above 0.9 and 0.75, in peak-hours; test 11 found this overflow volume, not T,
predicts flood volume behind a sill); profile RMSE against the gauge's median profile over +-48 h,
with GTSM's own events of the same RP window as the benchmark.

Acceptance per tidal class x RP: pass_overflow when the median ex090 and ex075 lie within
OVERFLOW_TOL of the gauge medians (test 11: about +-7-10% in flood volume); pass_shape (T(0.25),
T(0.5), nb24 within +-15%) and pass_rmse are kept for comparison.

Writes <out-dir>/analysis/{generator_scores.parquet, generator_acceptance.csv} and
generator_profiles.npz (hourly normalised profiles per site, RP and variant).

Usage:
    python boundary_test/generator_validation.py --out-dir .cleanup_baseline/level_sources [--workers 3]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from levels_tides import FES_CACHE  # noqa: E402
from tide_prediction import HOURLY_START, on_hourly_grid  # noqa: E402
from storm_tide_hydrograph import (  # noqa: E402
    STEPS_PER_HR,
    annual_maximum_events,
    cell_hydrograph,
    duration_mean_shape,
    gaussian_shape,
    high_water_indices,
    high_water_window,
    median_surge_shape,
    rp_event_selection,
    rp_hydrograph,
    spring_tide_window,
    storm_events,
    to_10min,
)
from storm_tide_product import fit_length, running_mean  # noqa: E402
from tide_statistics import spring_high_waters  # noqa: E402

TARGET_RPS = (2, 10, 25, 50, 100)
BASE_HR, EXTENDED_HR = 72, 120
CENSORED_SHARE = 0.25
TC_GAUGE_KM, TC_SURGE_SHARE, TC_HALF_WIDTH_HR = 50.0, 0.5, 8.0
RANGE_BINS = (0, 1, 2, 4, 20)
NB24_HR = (21, 28)
RMSE_HR = 48
SWAP_EVENTS = 8
SEASONAL = ("sa", "ssa")
OVERFLOW_TOL = 0.10
EXCESS_LEVELS = (0.9, 0.75)
SMOOTH_HR = 13  # one semidiurnal period: removes the tide-surge interaction ripple of the residual
EARTH_RADIUS_KM = 6371.0
VARIANTS = ("generator", "generator_phase", "generator_durations", "generator_smooth", "generator_gauge_peak",
            "hgrapher", "coast_hg")


def profile_measures(prof: np.ndarray) -> dict:
    """T(0.25), T(0.5), nb24 and ex090 / ex075 of an hourly profile normalised to 1 at its centre."""
    c = len(prof) // 2
    finite = np.where(np.isfinite(prof), prof, -np.inf)
    nb = []
    for sign in (-1, 1):
        lo, hi = sorted((c + sign * NB24_HR[0], c + sign * NB24_HR[1]))
        seg = prof[lo:hi + 1]
        if np.isfinite(seg).any():
            nb.append(np.nanmax(seg))
    return {"t025_hr": float((finite >= 0.75).sum()), "t05_hr": float((finite >= 0.5).sum()),
            "nb24": float(np.mean(nb)) if nb else np.nan,
            **{f"ex{round(100 * f):03d}": float(np.clip(finite - f, 0.0, None).sum()) for f in EXCESS_LEVELS}}


def hourly_profile(level10: np.ndarray) -> np.ndarray:
    """Hourly +-BASE_HR profile around the maximum of a 10-min hydrograph, normalised by it."""
    k = int(np.argmax(level10))
    idx = k + np.arange(-BASE_HR, BASE_HR + 1) * STEPS_PER_HR
    out = np.full(len(idx), np.nan)
    ok = (idx >= 0) & (idx < len(level10))
    out[ok] = level10[idx[ok]]
    return out / level10[k]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    c = len(a) // 2
    d = (a - b)[c - RMSE_HR:c + RMSE_HR + 1]
    return float(np.sqrt(np.nanmean(d ** 2)))


def _site_task(args) -> tuple[list[dict], dict]:
    try:
        return _score_site(args)
    except Exception as err:  # one bad site must not stop the pool; it is reported and skipped
        print(f"  skipped {args[1]['file_name']}: {type(err).__name__}: {err}", flush=True)
        return [], {}


def _score_site(args) -> tuple[list[dict], dict]:
    out, site, fes_hc, gtsm_hc, mhws, coast, tc_shapes, hg_profile, gauge_ev, gtsm_ev = args
    out = Path(out)
    b_re = np.load(out / "hourly_basis_re.npy", mmap_mode="r")
    b_im = np.load(out / "hourly_basis_im.npy", mmap_mode="r")
    f_re = np.load(FES_CACHE / "basis_re.npy", mmap_mode="r")
    f_im = np.load(FES_CACHE / "basis_im.npy", mmap_mode="r")
    surge_st, surge_time, surge = load_series(out, "surge")
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    n_hours = b_re.shape[0]
    gtsm_surge = np.full(n_hours, np.nan)
    gtsm_surge[surge_idx] = station_series(surge, pos[site["gtsm_index"]])
    gtsm_tide = np.asarray(b_re) @ gtsm_hc.real + np.asarray(b_im) @ gtsm_hc.imag
    level = gtsm_surge + gtsm_tide
    years = (HOURLY_START + np.arange(n_hours).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    ev = annual_maximum_events(level, years, EXTENDED_HR)
    if len(ev) < 10:
        return [], {}
    centres = ev.centre.to_numpy()
    win = np.arange(-EXTENDED_HR, EXTENDED_HR + 1)
    surge_win = gtsm_surge[centres[:, None] + win]
    level_win = level[centres[:, None] + win]
    tide_ratio = gtsm_tide[centres] / mhws["gtsm"]
    c = EXTENDED_HR
    above = level_win > 0
    censored = above[:, c - BASE_HR:c + 1].all(axis=1) | above[:, c:c + BASE_HR + 1].all(axis=1)
    half_hr = EXTENDED_HR if censored.mean() >= CENSORED_SHARE else BASE_HR
    fes = np.asarray(f_re) @ fes_hc.real + np.asarray(f_im) @ fes_hc.imag
    hw = high_water_indices(fes)
    spring = spring_high_waters(fes, 600.0)
    surge10 = to_10min(pd.Series(np.nan_to_num(gtsm_surge), index=pd.date_range(HOURLY_START, periods=n_hours, freq="h")))
    rows, profiles = [], {}
    hg_events = None
    for rp in TARGET_RPS:
        target = coast[f"rp{rp}"]
        if not np.isfinite(target):
            continue
        sel = rp_event_selection(ev.rp_yr.to_numpy(), rp)
        if len(sel) < 2:
            continue
        k = float(np.nanmedian(tide_ratio[sel]))
        is_tc = bool(np.isfinite(coast[f"tc{rp}"]) and coast[f"tc{rp}"] > coast[f"etc{rp}"])
        tide_win = high_water_window(fes, hw, k * mhws["fes"], half_hr)
        built, no_surge = {}, False
        for variant, on_max in (("generator", True), ("generator_phase", False), ("generator_durations", None),
                                ("generator_smooth", True)):
            if is_tc:
                shape = tc_shapes if tc_shapes is not None else gaussian_shape(TC_HALF_WIDTH_HR, half_hr)
                shape = fit_length(shape, len(tide_win))
            else:
                try:
                    windows = surge_win[sel][:, c - half_hr:c + half_hr + 1]
                    if variant == "generator_smooth":
                        windows = running_mean(windows, SMOOTH_HR)
                    shape = (duration_mean_shape(windows) if on_max is None
                             else median_surge_shape(windows, centre_on_max=on_max))
                except ValueError:  # every selected maximum is a spring tide with no positive surge
                    shape = gaussian_shape(TC_HALF_WIDTH_HR, half_hr)
                    no_surge = True
            built[variant] = rp_hydrograph(tide_win, shape, target)["level"]
            if variant == "generator" and len(g_sel_early := rp_event_selection(gauge_ev["rp"], rp)):
                built["generator_gauge_peak"] = rp_hydrograph(tide_win, shape,
                                                              float(np.median(gauge_ev["peak"][g_sel_early])))["level"]
        if not is_tc and not no_surge:
            swaps = _swap_measures(sel[:SWAP_EVENTS], centres, gtsm_tide, surge_win, level_win, c, half_hr, tide_win,
                                   shape, target, mhws)
        else:
            swaps = {}
        if hg_events is None:
            hg_events = storm_events(surge10, BASE_HR, min_peak_m=0.5 * max(coast["rp2"] - mhws["fes"], 0.05))
        spring_win = spring_tide_window(fes, spring, mhws["fes"], BASE_HR)
        mode = "gaussian" if is_tc or len(hg_events["surge_peak_m"]) < 3 else "per_rp"
        built["hgrapher"] = cell_hydrograph(spring_win, hg_events, target, mhws["fes"], BASE_HR, mode)["level"]
        if rp == 100 and hg_profile is not None:
            built["coast_hg"] = None
        g_sel = rp_event_selection(gauge_ev["rp"], rp)
        t_sel = rp_event_selection(gtsm_ev["rp"], rp)
        if len(g_sel) == 0:
            continue
        g_med = np.nanmedian(gauge_ev["prof"][g_sel], axis=0)
        g_meas = pd.DataFrame([profile_measures(p) for p in gauge_ev["prof"][g_sel]]).median()
        gtsm_rmse = _rmse(np.nanmedian(gtsm_ev["prof"][t_sel], axis=0), g_med) if len(t_sel) else np.nan
        for variant, lev in built.items():
            prof = hg_profile if variant == "coast_hg" else hourly_profile(lev)
            m = profile_measures(prof)
            rows.append({"file_name": site["file_name"], "rp": rp, "variant": variant, "tc": is_tc,
                         "no_surge_shape": no_surge,
                         "half_window_hr": half_hr, "tide_level_ratio": k, "n_events": len(sel),
                         **m, "rmse_vs_gauge": _rmse(prof, g_med), "gtsm_rmse_vs_gauge": gtsm_rmse,
                         "n_gauge_events": len(g_sel), **{f"gauge_{key}": float(v) for key, v in g_meas.items()}})
            profiles[(site["file_name"], rp, variant)] = prof.astype(np.float32)
        for variant, m in swaps.items():
            rows.append({"file_name": site["file_name"], "rp": rp, "variant": variant, "tc": is_tc,
                         "no_surge_shape": no_surge, "half_window_hr": half_hr, "tide_level_ratio": k,
                         "n_events": len(sel), **m, "rmse_vs_gauge": np.nan, "gtsm_rmse_vs_gauge": gtsm_rmse,
                         "n_gauge_events": len(g_sel), **{f"gauge_{key}": float(v) for key, v in g_meas.items()}})
        profiles[(site["file_name"], rp, "gauge")] = g_med.astype(np.float32)
    return rows, profiles


def _swap_measures(sel, centres, gtsm_tide, surge_win, level_win, c, half_hr, fes_window, shape,
                   target, mhws) -> dict:
    hours = np.arange(-half_hr, half_hr + 1)
    t10 = np.arange(-half_hr * STEPS_PER_HR, half_hr * STEPS_PER_HR + 1) / STEPS_PER_HR
    out = {"swap_tide": [], "swap_surge": [], "swap_both": [], "gtsm_event": [], "step_scaled": [],
           "step_smoothed": [], "step_recentred": []}
    for j in sel:
        near = centres[j] + np.arange(-3, 4)
        hw = int(near[np.argmax(gtsm_tide[near])])
        if min(hw, centres[j]) - half_hr < 0 or max(hw, centres[j]) + half_hr + 1 > len(gtsm_tide):
            continue
        raw = surge_win[j, c - half_hr:c + half_hr + 1]
        smooth = running_mean(raw[None, :], SMOOTH_HR)[0]
        if not (raw[half_hr] > 0 and smooth[half_hr] > 0):
            continue
        tide_at_peak = np.interp(t10, hours, gtsm_tide[centres[j] - half_hr:centres[j] + half_hr + 1])
        tide_at_hw = np.interp(t10, hours, gtsm_tide[hw - half_hr:hw + half_hr + 1])
        for variant, tide_s, surge_s in (("step_scaled", tide_at_peak, raw / raw[half_hr]),
                                         ("step_smoothed", tide_at_peak, smooth / smooth[half_hr])):
            out[variant].append(profile_measures(hourly_profile(
                rp_hydrograph(tide_s, np.interp(t10, hours, surge_s), target)["level"])))
        tide = np.interp(t10, hours, gtsm_tide[hw - half_hr:hw + half_hr + 1]) * mhws["fes"] / mhws["gtsm"]
        out["swap_tide"].append(profile_measures(hourly_profile(rp_hydrograph(tide, shape, target)["level"])))
        own = running_mean(surge_win[[j]][:, c - half_hr:c + half_hr + 1], SMOOTH_HR)
        try:
            own_shape = median_surge_shape(own)
        except ValueError:
            continue
        out["swap_surge"].append(profile_measures(hourly_profile(rp_hydrograph(fes_window, own_shape, target)["level"])))
        out["swap_both"].append(profile_measures(hourly_profile(rp_hydrograph(tide, own_shape, target)["level"])))
        out["step_recentred"].append(profile_measures(hourly_profile(rp_hydrograph(tide_at_hw, own_shape,
                                                                                    target)["level"])))
        lev = level_win[j, c - half_hr:c + half_hr + 1]
        out["gtsm_event"].append(profile_measures(lev / lev[half_hr]))
    return {v: pd.DataFrame(m).median().to_dict() for v, m in out.items() if m}


def gauge_tc_shapes(ev: pd.DataFrame, prof: np.ndarray, surge: np.ndarray, sites: pd.DataFrame) -> dict:
    """Per site: the TC surge shape from the nearest other quality gauge within TC_GAUGE_KM, or None."""
    g = ev[(ev.source == "gesla")]
    share = g.surge_at_peak_m / g.peak_m
    keep = (share >= TC_SURGE_SHARE).to_numpy()
    by_site = {f: np.flatnonzero(keep & (g.file_name == f).to_numpy()) for f in g.file_name[keep].unique()}
    donors = [f for f in by_site if len(by_site[f]) >= 2 and f in sites.index]
    out = {}
    if not donors:
        return out
    tree = BallTree(np.radians(sites.loc[donors, ["lat", "lon"]].to_numpy()), metric="haversine")
    rows = g.index.to_numpy()
    for f in sites.index:
        d, i = tree.query(np.radians(sites.loc[[f], ["lat", "lon"]].to_numpy()), k=min(2, len(donors)))
        for dist, j in zip(d[0] * EARTH_RADIUS_KM, i[0]):
            donor = donors[j]
            if donor != f and dist <= TC_GAUGE_KM:
                out[f] = median_surge_shape(surge[rows[by_site[donor]]].astype(float))
                break
    return out


def acceptance(scores: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    s = scores.assign(range_class=pd.cut(scores.file_name.map(sites.gesla_harm__spring_range_m), RANGE_BINS))
    rows = []
    for (variant, rc, rp), g in s.groupby(["variant", "range_class", "rp"], observed=True):
        row = {"variant": variant, "range_class": str(rc), "rp": rp, "n_sites": len(g)}
        for m in ("t025_hr", "t05_hr", "nb24", "ex090", "ex075"):
            row[m] = g[m].median()
            row[f"gauge_{m}"] = g[f"gauge_{m}"].median()
            row[f"{m}_ratio"] = row[m] / row[f"gauge_{m}"]
        row["rmse_vs_gauge"] = g.rmse_vs_gauge.median()
        row["gtsm_rmse_vs_gauge"] = g.gtsm_rmse_vs_gauge.median()
        row["pass_shape"] = all(abs(row[f"{m}_ratio"] - 1) <= 0.15 for m in ("t025_hr", "t05_hr", "nb24"))
        row["pass_rmse"] = row["rmse_vs_gauge"] <= row["gtsm_rmse_vs_gauge"]
        row["pass_overflow"] = all(abs(row[f"{m}_ratio"] - 1) <= OVERFLOW_TOL for m in ("ex090", "ex075"))
        rows.append(row)
    t = pd.DataFrame(rows)
    rp_ratio = t[t.rp.isin([2, 10])].pivot_table(index=["variant", "range_class"], columns="rp", values=["t025_hr",
                                                                                                    "gauge_t025_hr"])
    trend = (rp_ratio[("t025_hr", 2)] / rp_ratio[("t025_hr", 10)]).rename("t025_rp2_over_rp10")
    gauge_trend = (rp_ratio[("gauge_t025_hr", 2)] / rp_ratio[("gauge_t025_hr", 10)]).rename("gauge_t025_rp2_over_rp10")
    return t.merge(pd.concat([trend, gauge_trend], axis=1).reset_index(), on=["variant", "range_class"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N sites (trial run)")
    parser.add_argument("--sample", type=int, default=None, help="score N random sites (seed 0)")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    sites = pd.read_csv(dest / "levels_sites.csv", index_col=0)
    sites = sites[sites.quality]
    pairs = pd.read_parquet(out / "pairs.parquet").set_index("file_name")
    hc = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    pos = {f: i for i, f in enumerate(hc["file_name"])}
    cons = list(hc["constituents"])
    seasonal = [cons.index(c) for c in SEASONAL]
    tl = pd.read_parquet(out / "tide_levels.parquet")
    tl = tl[tl.variant == "no_seasonal"].set_index(["file_name", "source"]).mhws_m
    ex = pd.read_parquet(out / "extremes.parquet")
    ex = ex[ex.source == "gesla"].set_index("file_name")
    ev = pd.read_parquet(out / "events.parquet").reset_index(drop=True)
    npz = np.load(out / "event_profiles.npz")
    prof, surge = npz["profiles"], npz["surge_m"]
    tc_shapes = gauge_tc_shapes(ev, prof, surge, sites)
    tasks = []
    for f in sites.index:
        if f not in pos or f not in ex.index or (f, "fes_gauge") not in tl.index or (f, "gtsm_harm") not in tl.index:
            continue
        fes_hc, gtsm_hc = hc["fes_gauge"][pos[f]].copy(), hc["gtsm_harm"][pos[f]]
        if not (np.isfinite(fes_hc).all() and np.isfinite(gtsm_hc).all()):
            continue
        fes_hc[seasonal] = 0.0
        e = ex.loc[f]
        coast = {**{f"rp{rp}": e[f"coast_rp_rp{rp}_m"] for rp in TARGET_RPS},
                 **{f"tc{rp}": e[f"coast_rp_tc_rp{rp}_m"] for rp in TARGET_RPS},
                 **{f"etc{rp}": e[f"coast_rp_etc_rp{rp}_m"] for rp in TARGET_RPS}}
        mh = {"fes": tl[(f, "fes_gauge")], "gtsm": tl[(f, "gtsm_harm")]}
        hg = ev.index[(ev.file_name == f) & (ev.source == "coast_hg_spring")]
        g = ev.index[(ev.file_name == f) & (ev.source == "gesla")]
        t = ev.index[(ev.file_name == f) & (ev.source == "gtsm")]
        gauge_ev = {"rp": ev.rp_empirical_yr[g].to_numpy(), "prof": prof[g], "peak": ev.peak_m[g].to_numpy()}
        gtsm_ev = {"rp": ev.rp_empirical_yr[t].to_numpy(), "prof": prof[t]}
        tasks.append((str(out), {"file_name": f, "gtsm_index": int(pairs.loc[f, "gtsm_index"])}, fes_hc, gtsm_hc, mh,
                      coast, tc_shapes.get(f), prof[hg[0]].astype(float) if len(hg) else None, gauge_ev, gtsm_ev))
    if args.sample:
        tasks = [tasks[i] for i in np.random.default_rng(0).choice(len(tasks), min(args.sample, len(tasks)),
                                                                    replace=False)]
    tasks = tasks[:args.limit] if args.limit else tasks
    rows, profiles = [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (r, p) in enumerate(pool.map(_site_task, tasks)):
            rows += r
            profiles.update(p)
            if k % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} sites", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_parquet(dest / f"generator_scores{args.tag}.parquet")
    keys = list(profiles)
    np.savez(dest / f"generator_profiles{args.tag}.npz", file_name=np.array([k[0] for k in keys]),
             rp=np.array([k[1] for k in keys]), variant=np.array([k[2] for k in keys]),
             profile=np.vstack([profiles[k] for k in keys]))
    table = acceptance(scores, sites)
    table.round(3).to_csv(dest / f"generator_acceptance{args.tag}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["variant", "range_class", "rp", "n_sites", "t025_hr", "gauge_t025_hr", "t05_hr", "gauge_t05_hr",
                     "nb24", "gauge_nb24", "ex090_ratio", "ex075_ratio", "pass_shape", "pass_rmse",
                     "pass_overflow"]]
              .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
