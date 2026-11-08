"""Uncertainty of the storm-tide boundary: RP100 level components per station and duration accuracy.

Entry point: main(). Reads the product and open_guesses outputs (no rebuild); writes
<product-dir>/open_guesses/uncertainty/ for make_boundary_forcing_figures.py fig29-fig31.

RP100 level components (m, one standard error or a typical absolute error, per station):
  sampling_station  GPD tail, shape free, own 1950-2024 masked fit: delta method on the return
                    level with the asymptotic GPD covariance (Coles 2001, §4.3.3) and a Poisson
                    cluster count
  sampling_pooled   the same with the 300 km pooled shape: the scale's variance given the shape,
                    plus the pooled shape's profile-likelihood standard error
  record            |RP100 1950-2024 - 1979-2024| of the unmasked production method
  surge_signal      leave-one-out error of the gauge ratio r around the station, times the surge
                    part of the level (RP100 ETC level - MHWS)
  tc                |RP100 error| of the TC-dominated gauges nearby (masked, basin shift left out),
                    where the TC part dominates at RP100
  tide              relative tide-amplitude error of the gauges nearby (gauge-corrected where the
                    station has a tide gauge within the correction radius, FES otherwise), times MHWS
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from regions import ar6_regions  # noqa: E402
from storm_tide_product import EARTH_RADIUS_KM, MIN_NEIGHBOURS, R_CAP  # noqa: E402
import tide_correction as tdc  # noqa: E402

SHAPE_GRID = np.round(np.arange(-0.5, 0.805, 0.01), 2)  # as open_guesses.pooled_shape's profiles
POOL_KM = 300.0
P_U = 0.005  # skew_surge.THRESHOLD_Q: exceedance share of the tail threshold
CHI2_95 = 1.92  # half the 95% chi-square(1) quantile: profile-likelihood interval
SEMIDIURNAL_CYCLES, DIURNAL_CYCLES = 705.8, 353.0  # high waters per year
# Form factor (K1 + O1) / (M2 + S2): semidiurnal < 0.25 < mixed, mainly semidiurnal < 1.5 <
# mixed, mainly diurnal < 3 < diurnal (Pugh & Woodworth 2014, §4.2.4)
FORM_BINS = (0.0, 0.25, 1.5, 3.0, np.inf)
FORM_LABELS = ("semidiurnal", "mixed, semidiurnal", "mixed, diurnal", "diurnal")
RANGE_BINS = (0.0, 1.0, 2.0, 4.0, 20.0)  # MHWS classes of the generator acceptance test
NEAR_GAUGES = 5
GAUGE_KM = 500.0  # gauges further away say nothing about the station's r or tide
TC_GAUGE_KM = 1500.0  # TC errors are basin-scale (one basin shift per basin)
COMPONENTS = ("sampling_pooled", "record", "surge_signal", "tc", "tide")
CELL_DEG = 5.0
R_VARIANT = "covariate fallback"  # the production rule for r in open_guesses.r_variants


def _tree(lonlat: np.ndarray) -> BallTree:
    return BallTree(np.radians(lonlat[:, ::-1]), metric="haversine")


def return_level_se(scale: np.ndarray, shape: np.ndarray, n_clusters: np.ndarray, rate_u: np.ndarray, rp: float,
                    shape_var: np.ndarray | None = None) -> np.ndarray:
    """Delta-method standard error (m) of the GPD return level above the threshold at `rp` years.

    x = scale / shape * ((rate_u / rate)^shape - 1), rate = -ln(1 - 1/rp) clusters per year,
    rate_u = clusters per year above the threshold. Shape free (`shape_var` None): asymptotic
    covariance (1 + shape) / n [[2 scale^2, scale], [scale, 1 + shape]] (valid for shape > -0.5).
    `shape_var` given: the scale's variance given the shape, scale^2 (1 + 2 shape) / n, and the
    shape's own variance, taken as independent of this station's scale (estimated from a pool).
    The cluster count is Poisson: var(rate_u) / rate_u^2 = 1 / n.
    """
    log_r = np.log(rate_u / -np.log1p(-1.0 / rp))
    small = np.abs(shape) < 1e-6
    safe = np.where(small, 1.0, shape)
    pw = np.exp(shape * log_r)
    g_scale = np.where(small, log_r, np.expm1(shape * log_r) / safe)
    g_shape = np.where(small, scale * log_r ** 2 / 2, -scale / safe ** 2 * np.expm1(shape * log_r)
                       + scale / safe * pw * log_r)
    g_rate = scale * pw / rate_u
    var = g_rate ** 2 * rate_u ** 2 / n_clusters
    if shape_var is None:
        c = (1 + shape) / n_clusters
        var += c * (g_scale ** 2 * 2 * scale ** 2 + 2 * g_scale * g_shape * scale + g_shape ** 2 * (1 + shape))
    else:
        var += g_scale ** 2 * scale ** 2 * (1 + 2 * shape) / n_clusters + g_shape ** 2 * shape_var
    return np.sqrt(var)


def profile_interval(loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) GPD shape of the 95% profile-likelihood interval per row of SHAPE_GRID log-likelihoods."""
    inside = loglik >= np.nanmax(loglik, axis=1, keepdims=True) - CHI2_95
    lo = SHAPE_GRID[np.argmax(inside, axis=1)]
    hi = SHAPE_GRID[len(SHAPE_GRID) - 1 - np.argmax(inside[:, ::-1], axis=1)]
    return lo, hi


def _form_factor(cache: Path, stations: pd.Index) -> np.ndarray:
    import tide_prediction as tp
    c = np.load(cache / "fes_constants.npz")
    names = list(tp.fes2022_model(cache / "fes2022").constituents)
    pos = pd.Index(c["station"]).get_indexer(stations)
    amp = np.abs(c["fes2022"][pos])
    col = {n: names.index(n) for n in ("k1", "o1", "m2", "s2")}
    return (amp[:, col["k1"]] + amp[:, col["o1"]]) / (amp[:, col["m2"]] + amp[:, col["s2"]])


def _near_median(src_lonlat: np.ndarray, values: np.ndarray, query: np.ndarray, km: float,
                 k: int = NEAR_GAUGES) -> np.ndarray:
    """Median of `values` at the k nearest sources within km of each query point (NaN if none)."""
    ok = np.isfinite(values)
    d, i = _tree(src_lonlat[ok]).query(np.radians(query[:, ::-1]), k=min(k, int(ok.sum())))
    v = np.where(d * EARTH_RADIUS_KM <= km, values[ok][i], np.nan)
    return np.array([np.nanmedian(row) if np.isfinite(row).any() else np.nan for row in v])


def station_components(product: Path, cache: Path, sources: Path) -> pd.DataFrame:
    g = product / "open_guesses"
    lev = pd.read_parquet(product / "levels.parquet")
    lev = lev[lev.level_source == "product"]
    masked = pd.read_parquet(g / "pooled_shape" / "masked_rows.parquet").reindex(lev.index)
    prof = np.load(g / "pooled_shape" / "profiles.npz")
    full = pd.DataFrame(prof["full"], index=prof["station"]).reindex(lev.index).to_numpy()
    lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    out = lev[["station_id", "lon", "lat", "mhws_m", "r", "r_neighbours", "tc_dominant_rp100"]].copy()
    out["form_factor"] = _form_factor(cache, lev.index)
    out["tide_type"] = pd.cut(out.form_factor, FORM_BINS, labels=FORM_LABELS).astype(str)
    out["range_class"] = pd.cut(out.mhws_m, RANGE_BINS).astype(str)

    # (i) sampling, 1950-2024 with the IBTrACS mask
    cycles = np.where(out.form_factor > FORM_BINS[3], DIURNAL_CYCLES, SEMIDIURNAL_CYCLES)
    rate_u = cycles * masked.theta.to_numpy() * P_U
    n_c = rate_u * masked.n_years.to_numpy()
    scale = masked.gpd_scale.to_numpy() * lev.r.to_numpy()
    shape = np.maximum(masked.gpd_shape_mle.to_numpy(), -0.49)
    lo, hi = profile_interval(full)
    out["shape_station"], out["shape_station_lo"], out["shape_station_hi"] = shape, lo, hi
    out["sampling_station_m"] = return_level_se(scale, shape, n_c, rate_u, 100.0)
    near = _tree(lonlat).query_radius(np.radians(lonlat[:, ::-1]), r=POOL_KM / EARTH_RADIUS_KM)
    pooled = np.vstack([np.nansum(full[n], axis=0) for n in near])
    p_lo, p_hi = profile_interval(pooled)
    p_shape = SHAPE_GRID[np.nanargmax(pooled, axis=1)]
    out["shape_pooled"], out["shape_pooled_lo"], out["shape_pooled_hi"] = p_shape, p_lo, p_hi
    out["pool_stations"] = [len(n) for n in near]
    out["sampling_pooled_m"] = return_level_se(scale, np.maximum(p_shape, -0.49), n_c, rate_u, 100.0,
                                               shape_var=((p_hi - p_lo) / 3.92) ** 2)
    out["n_clusters"] = n_c

    # (iii) record length
    lr = pd.read_parquet(g / "long_record" / "long_record.parquet").set_index("station_id")
    lr = lr[~lr.index.duplicated()]
    out["record_m"] = (lr.long_rp100_m - lr.rp100_m).abs().reindex(out.station_id).to_numpy()

    # (ii) surge signal: leave-one-out error of r at the ratio gauges around the station
    rv = pd.read_parquet(g / "r_variants.parquet")
    gg = pd.read_parquet(product / "gauges" / "gauges.parquet").reindex(rv.index)
    r_gauge = np.minimum(gg.ratio_fes.to_numpy(), R_CAP)
    log_err = np.abs(np.log(rv[f"{R_VARIANT}_value"].to_numpy() / r_gauge))
    g_lonlat = gg[["lon", "lat"]].to_numpy(dtype=float)
    rel = np.expm1(_near_median(g_lonlat, log_err, lonlat, GAUGE_KM))
    fallback = np.expm1(np.nanmedian(log_err[rv.neighbours.to_numpy() < MIN_NEIGHBOURS]))
    out["r_err_rel"] = np.where(np.isfinite(rel), rel, fallback)
    surge_part = np.maximum(lev.etc_rp100_m.to_numpy() - lev.mhws_m.to_numpy(), 0.0)
    out["surge_signal_m"] = out.r_err_rel * surge_part

    # (iv) TC part
    tc = pd.read_parquet(g / "tc_scale.parquet")
    tcd = (tc.station_class == "tc_dominated").to_numpy()
    tc_err = (tc.masked_tc_shift_m_loo_basin_rp100_m - tc.gauge_rp100_m).abs().to_numpy()[tcd]
    tc_lonlat = gg.reindex(tc.index)[["lon", "lat"]].to_numpy(dtype=float)[tcd]
    e = _near_median(tc_lonlat, tc_err, lonlat, TC_GAUGE_KM)
    out["tc_m"] = np.where(out.tc_dominant_rp100, np.where(np.isfinite(e), e, np.nanmedian(tc_err)), 0.0)

    # (v) tide
    tg = pd.read_parquet(g / "tide_correction" / "gauges.parquet")
    t_lonlat = tg[["lon", "lat"]].to_numpy(dtype=float)
    corrected = tg.neighbours_25km.to_numpy() > 0
    e_corr = np.abs(tg.amp_corrected_25km_m / tg.amp_gauge_m - 1).to_numpy()
    e_fes = np.abs(tg.amp_fes_m / tg.amp_gauge_m - 1).to_numpy()
    has_gauge = _tree(t_lonlat).query_radius(np.radians(lonlat[:, ::-1]), r=tdc.CORRECTION_KM / EARTH_RADIUS_KM,
                                             count_only=True) > 0
    rel_c = _near_median(t_lonlat, np.where(corrected, e_corr, np.nan), lonlat, GAUGE_KM)
    rel_f = _near_median(t_lonlat, e_fes, lonlat, GAUGE_KM)
    rel = np.where(has_gauge, rel_c, rel_f)
    out["tide_corrected"] = has_gauge
    out["tide_err_rel"] = np.where(np.isfinite(rel), rel, np.nanmedian(e_fes))
    out["tide_m"] = out.tide_err_rel * out.mhws_m

    comp = out[[f"{c}_m" for c in COMPONENTS]].fillna(0.0).to_numpy()
    out["combined_m"] = np.sqrt((comp ** 2).sum(axis=1))
    out["dominant"] = np.array(COMPONENTS)[comp.argmax(axis=1)]

    # durations: product RP100 hydrograph against COAST-HG, and the shape source
    dur = pd.read_parquet(product / "durations.parquet").reindex(lev.index)
    for f in ("t025", "t05"):
        out[f"rp100_{f}_hr"] = dur[f"rp100_{f}_hr"].to_numpy()
        out[f"coast_hg_{f}_hr"] = dur[f"coast_hg_{f}_hr"].to_numpy()
    info = pd.read_parquet(product / "hydrograph_info.parquet")
    src = info[info.rp == 100].set_index("station").shape_source
    out["shape_source_rp100"] = src.reindex(lev.index).to_numpy()
    reg = ar6_regions(out.lon.to_numpy(), out.lat.to_numpy())
    out["region"] = reg.region.to_numpy()
    return out


def gauge_surge_accuracy(product: Path) -> pd.DataFrame:
    """Per ratio gauge: GTSM-on-FES against gauge skew surge (RP10), r left out, and level residuals."""
    g = product / "open_guesses"
    rv = pd.read_parquet(g / "r_variants.parquet")
    gg = pd.read_parquet(product / "gauges" / "gauges.parquet").reindex(rv.index)
    out = gg[["lon", "lat", "mhws_gauge_m", "n_years", "skew_gauge_m", "skew_gtsmfes_m", "ratio_fes"]].copy()
    out["region"], out["station_class"], out["neighbours"] = rv.region, rv.station_class, rv.neighbours
    out["r_loo"] = rv[f"{R_VARIANT}_value"]
    out["r_loo_err"] = out.r_loo / np.minimum(gg.ratio_fes, R_CAP) - 1
    for rp in (2, 10, 100):
        out[f"raw_rp{rp}_err_m"] = rv[f"new_loo_r1_rp{rp}_m"] - rv[f"gauge_rp{rp}_m"]
        out[f"product_rp{rp}_err_m"] = rv[f"{R_VARIANT}_rp{rp}_m"] - rv[f"gauge_rp{rp}_m"]
        out[f"coast_rp_rp{rp}_err_m"] = rv[f"coast_rp_rp{rp}_m"] - rv[f"gauge_rp{rp}_m"]
    tc = pd.read_parquet(g / "tc_scale.parquet").reindex(rv.index)
    out["basin"] = tc.basin
    out["decided_rp100_err_m"] = np.where(out.station_class == "tc_dominated",
                                          tc.masked_tc_shift_m_loo_basin_rp100_m - tc.gauge_rp100_m,
                                          out.product_rp100_err_m)
    out["range_class"] = pd.cut(out.mhws_gauge_m, RANGE_BINS).astype(str)
    return out


def gauge_duration_accuracy(sources: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """Per GESLA site and RP: production generator against the gauge's events, and event spread."""
    sc = pd.read_parquet(sources / "analysis" / "generator_scores.parquet")
    sc = sc[sc.variant == "generator_smooth"]
    ev = pd.read_parquet(sources / "events.parquet")
    pairs = pd.read_parquet(sources / "pairs.parquet").set_index("file_name")
    spread = []
    for src in ("gesla", "gtsm"):
        e = ev[ev.source == src].groupby("file_name")
        for f, col in (("t025", "t_within_0.25"), ("t05", "t_within_0.5")):
            q = e[col].quantile([0.25, 0.5, 0.75]).unstack()
            spread.append(pd.DataFrame({f"{src}_{f}_median_hr": q[0.5], f"{src}_{f}_iqr_rel": (q[0.75] - q[0.25]) / q[0.5]}))
    spread = pd.concat(spread, axis=1)
    out = sc[["file_name", "rp", "tc", "no_surge_shape", "t025_hr", "t05_hr", "gauge_t025_hr", "gauge_t05_hr",
              "n_gauge_events"]].copy()
    out = out.join(pairs[["lon", "lat"]], on="file_name").join(spread, on="file_name")
    for f in ("t025", "t05"):
        out[f"{f}_err_rel"] = out[f"{f}_hr"] / out[f"gauge_{f}_hr"] - 1
    s_lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
    _, i = _tree(s_lonlat).query(np.radians(out[["lon", "lat"]].to_numpy(dtype=float)[:, ::-1]), k=1)
    for col in ("mhws_m", "form_factor", "tide_type", "range_class", "region"):
        out[col] = stations[col].to_numpy()[i[:, 0]]
    return out


def _summary(df: pd.DataFrame, err: str, by: str) -> pd.DataFrame:
    rows = []
    for key, s in [("all", df)] + list(df.groupby(by, observed=True)):
        e = s[err].dropna()
        if len(e):
            rows.append({by: str(key), "n": len(e), "median": e.median(), "mae": e.abs().mean(),
                         "p10": e.quantile(0.1), "p90": e.quantile(0.9)})
    return pd.DataFrame(rows).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-dir", default=str(REPO / ".cleanup_baseline" / "storm_tide_product"))
    parser.add_argument("--sources-dir", default=str(REPO / ".cleanup_baseline" / "level_sources"))
    parser.add_argument("--cache-dir", required=True, help="storm_tide_product.cache_dir (FES constants, FES2022)")
    args = parser.parse_args()
    product, sources = Path(args.product_dir), Path(args.sources_dir)
    dest = product / "open_guesses" / "uncertainty"
    dest.mkdir(parents=True, exist_ok=True)

    st = station_components(product, Path(args.cache_dir), sources)
    st.to_parquet(dest / "stations.parquet")
    surge = gauge_surge_accuracy(product)
    surge.to_parquet(dest / "gauges_surge.parquet")
    dur = gauge_duration_accuracy(sources, st)
    dur.to_parquet(dest / "gauges_duration.parquet")

    tables = {"components": pd.concat([st[[f"{c}_m" for c in ("sampling_station", *COMPONENTS, "combined")]]
                                       .describe(percentiles=[0.1, 0.5, 0.9]).T,
                                       st.groupby("dominant").size().rename("dominant_n")], axis=1).round(3),
              "components_by_region": st.groupby("region")[[f"{c}_m" for c in (*COMPONENTS, "combined")]]
              .median().join(st.groupby("region").size().rename("n")).round(3),
              "surge_ratio": _summary(surge.assign(log_r=np.log(surge.ratio_fes)), "log_r", "station_class"),
              "surge_rloo": _summary(surge, "r_loo_err", "station_class"),
              "surge_rp100_raw_by_region": _summary(surge, "raw_rp100_err_m", "region"),
              "surge_rp100_by_class": pd.concat([_summary(surge, f"{v}_rp100_err_m", "station_class").assign(level=v)
                                                 for v in ("coast_rp", "raw", "product", "decided")]),
              "surge_rp100_by_range": pd.concat([_summary(surge, f"{v}_rp100_err_m", "range_class").assign(level=v)
                                                 for v in ("raw", "decided")]),
              "duration_by_range": pd.concat([_summary(dur[dur.rp == rp], f"{f}_err_rel", "range_class")
                                             .assign(rp=rp, measure=f) for rp in (2, 100) for f in ("t025", "t05")]),
              "duration_by_tide_type": pd.concat([_summary(dur[dur.rp == rp], f"{f}_err_rel", "tide_type")
                                                 .assign(rp=rp, measure=f) for rp in (2, 100) for f in ("t025", "t05")]),
              "duration_by_tc": pd.concat([_summary(dur[dur.rp == rp], f"{f}_err_rel", "tc")
                                          .assign(rp=rp, measure=f) for rp in (2, 100) for f in ("t025", "t05")])}
    sites = dur[dur.rp == 2].drop_duplicates("file_name")
    for f in ("t025", "t05"):
        sites = sites.assign(**{f"gtsm_vs_gesla_{f}": sites[f"gtsm_{f}_median_hr"] / sites[f"gesla_{f}_median_hr"] - 1})
    tables["event_spread"] = pd.concat([_summary(sites, f"gesla_{f}_iqr_rel", "range_class").assign(measure=f)
                                        for f in ("t025", "t05")])
    tables["gtsm_vs_gesla_events"] = pd.concat([_summary(sites, f"gtsm_vs_gesla_{f}", "range_class").assign(measure=f)
                                                for f in ("t025", "t05")])
    for f in ("t025", "t05"):
        st[f"product_vs_coast_hg_{f}"] = st[f"rp100_{f}_hr"] / st[f"coast_hg_{f}_hr"] - 1
    tables["product_vs_coast_hg"] = pd.concat([_summary(st, f"product_vs_coast_hg_{f}", "range_class").assign(measure=f)
                                               for f in ("t025", "t05")])
    tables["shape_source_rp100"] = st.groupby(["tide_type", "shape_source_rp100"]).size().unstack(fill_value=0)
    st.to_parquet(dest / "stations.parquet")
    # The delta-method sampling error against the GEV 90% interval of the gauges' own annual maxima
    ext = pd.read_parquet(sources / "extremes.parquet")
    ext = ext[ext.source == "gesla"].set_index("file_name")
    se_gev = ((ext.rp100_p95_m - ext.rp100_p05_m) / (2 * 1.645)).reindex(surge.index)
    _, i = _tree(st[["lon", "lat"]].to_numpy(dtype=float)).query(
        np.radians(surge[["lon", "lat"]].to_numpy(dtype=float)[:, ::-1]), k=1)
    tables["sampling_check"] = pd.DataFrame({
        "gauge_gev_se_m": se_gev.describe(percentiles=[0.1, 0.5, 0.9]),
        "station_delta_se_m": pd.Series(st.sampling_station_m.to_numpy()[i[:, 0]]).describe(percentiles=[0.1, 0.5, 0.9]),
        "pooled_delta_se_m": pd.Series(st.sampling_pooled_m.to_numpy()[i[:, 0]]).describe(percentiles=[0.1, 0.5, 0.9]),
    }).round(3)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for name, t in tables.items():
            t.to_csv(dest / f"{name}.csv")
            print(f"== {name}\n{t.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
