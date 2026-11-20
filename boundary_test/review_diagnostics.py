"""Diagnostics of the storm-tide product review (TODO.md, review plan phase 2) on production's inputs.

Stages (each writes <dest-dir>/<stage>/ parquet and csv tables):
  tc_raise     the COAST-RP_TC raise in three forms, each one parameter per basin fitted at RP100
               on the TC-dominated gauges (basin left out, or the basin's own gauges), scored at RP
               2-100: shift (production), scale of the whole level, and scale of the level above
               COAST-RP_TC's own RP2 level (its tide floor), which leaves the tide untouched
  tc_mask_et   the IBTrACS mask with and without extratropical track points (NATURE "ET"): share of
               the stations' annual maxima each masks, and the levels at the gauges
  shift_bootstrap  the production basin shift resampled over its gauges (tc_shift_gauges.parquet)
  tide_neighbours  the tide factor's neighbour gauges within tide_correction.CORRECTION_KM: all
               (production), connected through GEBCO water (elevation < 0), FES amplitude within a
               factor TIDE_SIMILAR_RULE of the point's (TIDE_SIMILAR as sensitivity), and both; the
               tide amplitude at the usable tide gauges and the levels at the ratio gauges, gauge
               site left out
  gauge_series one pass over the ratio gauges' GESLA and GTSM records (1979-2024, common years):
               r at R_RPS, skew-surge quantiles by high-water height tertile (gauge and GTSM on FES),
               monthly climatologies of the gauge residual, GTSM and FES Sa+Ssa, the months of the
               gauge's annual maxima, and GEBCO depth at the GTSM point and distance to the shelf
               edge (SHELF_EDGE_M)
  gauge_analysis  from gauge_series: r_rp2 and r_rp50 against r_rp10; ln r against S10 and the
               independent covariates, and the fallback options for r scored gauge left out;
               skew-surge quantiles of the top against the bottom high-water tertile; the gauge-minus-
               GTSM seasonal cycle at the months of the gauge's annual maxima, beside FES Sa+Ssa
  despike_tracks  GTSM hours the despike drops within the IBTrACS mask hours and outside, at the
               stations with a TC part
Gauge reference and r / tide factor left out as in `level_product_gauges.GaugeCheck`.

Usage:
    python boundary_test/review_diagnostics.py <stage> [...] --product-dir <storm_tide_product_dir> \
        --dest-dir <dir> [--sources-dir .cleanup_baseline/level_sources] [--override local.yml] [--workers 4]
"""

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from along_water import along_water_distances  # noqa: E402
from config_utils import get_data_catalog, load_config  # noqa: E402
from gesla import DEDUP_KM  # noqa: E402
from gtsm_series import gtsm_stations, load_series, station_series  # noqa: E402
from level_product_gauges import (  # noqa: E402
    GAUGE_RPS,
    PRODUCTION_RECORD,
    GaugeCheck,
    breakdown_groups,
    group_mask,
    scores,
)
import storm_tide_product as stp  # noqa: E402
import skew_surge as ss  # noqa: E402
import tide_correction as tdc  # noqa: E402
import tide_prediction as tp  # noqa: E402

RAISE_FORMS = {"shift": ("tc_shift_m", (0.0, 0.1, 0.2, 0.3, 0.4)),
               "level scale": ("tc_scale", (1.0, 1.1, 1.2, 1.3, 1.4)),
               "surge scale": ("tc_surge_scale", (1.0, 1.25, 1.5, 1.75, 2.0))}
BOOTSTRAP_REPS = 2000
SEED = 0
TC_GROUPS = ("TC-dominated gauges", "stations with a TC part")
TIDE_SIMILAR_RULE = 1.2  # a neighbour shares the point's tidal regime if FES's amplitudes differ by < 20%
TIDE_SIMILAR = (1.1, 1.2, 1.5)
KM_PER_DEG = 111.2
R_RPS = (2, 10, 50)
TERTILE_QUANTILES = (0.99, 0.999)
SHELF_EDGE_M = 200.0  # depth of the shelf break
SHELF_GRID_DEG = 1 / 20
LONG_YEARS = 40
WINDOW_MARGIN_KM = 5.0


def _subset(check: GaugeCheck, keep: np.ndarray) -> GaugeCheck:
    """A GaugeCheck over the gauges in `keep` (renumbered inputs)."""
    sub = copy.copy(check)
    idx = np.flatnonzero(keep)
    sub.todo = [{**check.todo[j], "station": n} for n, j in enumerate(idx)]
    sub.meta = check.meta.iloc[idx].reset_index(drop=True)
    sub.picked = check.picked.iloc[idx]
    return sub


def _tc_part(check: GaugeCheck) -> GaugeCheck:
    return _subset(check, np.array([s["tc"] is not None for s in check.todo]))


def _raised(todo: list[dict], key: str, value: float) -> list[dict]:
    """Inputs with production's shift replaced by one raise; "tc_surge_scale" scales tc above tc[0]."""
    out = []
    for s in todo:
        s = {**s, "tc_shift_m": 0.0}
        if key == "tc_surge_scale":
            if s["tc"] is not None:
                s["tc"] = s["tc"][0] + value * (s["tc"] - s["tc"][0])
        else:
            s[key] = value
        out.append(s)
    return out


def _levels_with(check: GaugeCheck, todo: list[dict], workers: int, **kw) -> pd.DataFrame:
    base, check.todo = check.todo, todo
    try:
        return check.table(check.levels(workers, **kw))
    finally:
        check.todo = base


def _basin_values(trials: np.ndarray, err: np.ndarray, basin: np.ndarray) -> tuple[dict, dict, float]:
    """Per basin: the basin-left-out value, the basin's own value, and the all-gauge value."""
    loo, all_value = stp.loo_basin_fit(trials, err, basin)
    own = {str(b): stp.trial_root(trials, np.median(err[:, basin == b], axis=1)) for b in np.unique(basin)}
    return loo, own, all_value


def tc_raise(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "tc_raise"
    root.mkdir(parents=True, exist_ok=True)
    sub = _tc_part(check)
    hours = sub.tc_hours()
    basin = sub.lev.tc_basin.loc[sub.meta.station].to_numpy().astype(str)
    out = None
    fits = []
    for form, (key, trials) in RAISE_FORMS.items():
        levels = []
        for v in trials:
            levels.append(_levels_with(sub, _raised(sub.todo, key, v), workers, mask_hours=hours))
            print(f"  {form} {v:g}", flush=True)
        if out is None:
            out = levels[0].drop(columns=[c for c in levels[0] if c.startswith("new_loo")])
            out["basin"] = basin
            for rp in GAUGE_RPS:
                out[f"none_rp{rp}_m"] = levels[0][f"new_loo_rp{rp}_m"]
        tcd = (out.station_class == "tc_dominated").to_numpy()
        err = np.stack([lv["new_loo_rp100_m"] - lv["gauge_rp100_m"] for lv in levels])[:, tcd]
        loo, own, all_value = _basin_values(np.array(trials), err, basin[tcd])
        for fit, values in (("basin left out", loo), ("own basin", own)):
            per_gauge = np.array([values.get(b, all_value) for b in basin])
            name = f"{form}, {fit}"
            col = "_".join(name.replace(",", "").split())
            for rp in GAUGE_RPS:
                lev = np.stack([lv[f"new_loo_rp{rp}_m"].to_numpy() for lv in levels])
                out[f"{col}_rp{rp}_m"] = [np.interp(x, trials, lev[:, j]) for j, x in enumerate(per_gauge)]
            fits += [{"form": form, "fit": fit, "basin": b, "value": v, "gauges": int((basin[tcd] == b).sum())}
                     for b, v in values.items()] + [{"form": form, "fit": "all gauges", "basin": "all",
                                                       "value": all_value, "gauges": int(tcd.sum())}]
    out.to_parquet(root / "gauges.parquet")
    pd.DataFrame(fits).drop_duplicates().to_csv(root / "fits.csv", index=False)
    cols = {"no raise": "none"} | {f"{f}, {fit}": "_".join(f"{f} {fit}".split())
                                   for f in RAISE_FORMS for fit in ("basin left out", "own basin")}
    groups = {g: group_mask(out, g) for g in TC_GROUPS} | breakdown_groups(out)
    s = scores(out, cols, groups)
    s["split"] = s.split.fillna("TC group")
    s.to_csv(root / "scores.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(pd.DataFrame(fits).drop_duplicates().pivot_table(index=["form", "fit"], columns="basin",
                                                                values="value").round(3).to_string())
        for v in ("median", "mae"):
            print(v)
            print(s.pivot_table(index=["split", "gauges", "rp"], columns="level", values=v, sort=False)[list(cols)]
                  .round(3).to_string(), flush=True)


def _tc_hours_nature(stations: pd.DataFrame, n_hours: int, ibtracs_path: Path, start: np.datetime64,
                     drop_natures: tuple = ()) -> dict:
    """`stp.tc_hours` (production radius, window and wind) without the track points of `drop_natures`."""
    tracks = pd.read_csv(ibtracs_path, usecols=["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND", "NATURE"],
                         skiprows=[1], na_values=[" "], low_memory=False)
    wind = tracks[["WMO_WIND", "USA_WIND"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    no_wind = ~wind.notna().groupby(tracks.SID).transform("any")
    tracks = tracks[((wind >= stp.TC_MIN_WIND_KT) | no_wind) & ~tracks.NATURE.isin(drop_natures)]
    hour = ((pd.to_datetime(tracks.ISO_TIME).to_numpy() - np.datetime64(start)) // np.timedelta64(1, "h")).astype(int)
    tree = BallTree(np.radians(tracks[["LAT", "LON"]].to_numpy(dtype=float)), metric="haversine")
    near = tree.query_radius(np.radians(stations[["lat", "lon"]].to_numpy()), r=stp.TC_MASK_KM / stp.EARTH_RADIUS_KM)
    spread = np.arange(-stp.TC_MASK_HR, stp.TC_MASK_HR + 1)
    out = {}
    for st, idx in zip(stations.index, near):
        if len(idx):
            h = np.unique((hour[idx][:, None] + spread).ravel())
            out[st] = h[(h >= 0) & (h < n_hours)]
    return out


def _maxima_task(args) -> list[dict]:
    chunk, stores, bases, start = args
    out = []
    for st, g, th, _, years in stp.station_records(chunk, stores, bases, start, nodal=False):
        level = g + th
        row = {"station": st["station"], "years": 0, "masked": 0, "masked_tropical": 0}
        for y in np.unique(years):
            sel = np.flatnonzero(years == y)
            if np.isfinite(level[sel]).sum() < 0.8 * len(sel):
                continue
            h = sel[np.nanargmax(level[sel])]
            row["years"] += 1
            row["masked"] += int(h in st["hours_all"])
            row["masked_tropical"] += int(h in st["hours_tropical"])
        out.append(row)
    return out


def tc_mask_et(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "tc_mask_et"
    root.mkdir(parents=True, exist_ok=True)
    sub = _tc_part(check)
    stores, bases, start, n_hours = sub.records[PRODUCTION_RECORD]
    path = Path(sub.cfg["storm_tide_product"]["ibtracs_path"])
    with_tc = sub.picked[sub.picked.station_id.isin(sub.tc_levels.index)]
    hours_all = sub.tc_hours()
    hours_tropical = _tc_hours_nature(with_tc, n_hours, path, start, drop_natures=("ET",))
    empty = np.array([], dtype=int)
    todo = [{**s, "hours_all": set(hours_all.get(s["station_index"], empty).tolist()),
             "hours_tropical": set(hours_tropical.get(s["station_index"], empty).tolist())}
            for s in sub.inputs()]
    tasks = [(todo[i:i + stp.TASK_SIZE], [str(p) for p in stores], bases, start)
             for i in range(0, len(todo), stp.TASK_SIZE)]
    with ProcessPoolExecutor(workers) as pool:
        maxima = pd.DataFrame([r for res in pool.map(_maxima_task, tasks) for r in res]).set_index("station")
    maxima.index = sub.meta.file_name.to_numpy()[maxima.index]
    full = sub.table(sub.levels(workers, hours_all))
    tropical = sub.table(sub.levels(workers, hours_tropical))
    out = full.drop(columns=[c for c in full if c.startswith("new_loo")]).join(maxima)
    out["basin"] = sub.lev.tc_basin.loc[sub.meta.station].to_numpy().astype(str)
    for rp in GAUGE_RPS:
        out[f"mask_all_rp{rp}_m"] = full[f"new_loo_rp{rp}_m"]
        out[f"mask_tropical_rp{rp}_m"] = tropical[f"new_loo_rp{rp}_m"]
    out.to_parquet(root / "gauges.parquet")
    groups = {g: group_mask(out, g) for g in TC_GROUPS} | breakdown_groups(out)
    s = scores(out, {"mask, all track points (production)": "mask_all", "mask, no ET points": "mask_tropical"},
               groups)
    s["split"] = s.split.fillna("TC group")
    s.to_csv(root / "scores.csv", index=False)
    loss = out.assign(region=out.region.where(out.region.map(out.region.value_counts()) >= 15, "other")) \
        .groupby("region")[["years", "masked", "masked_tropical"]].sum()
    loss["masked_share"] = loss.masked / loss.years
    loss["et_share_of_masked"] = 1 - loss.masked_tropical / loss.masked
    loss.to_csv(root / "maxima_loss.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(loss.round(3).to_string())
        print(s[s.rp.isin([2, 100])].pivot_table(index=["split", "gauges", "rp"], columns="level",
                                                  values=["median", "mae"], sort=False).round(3).to_string(), flush=True)


def shift_bootstrap(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "shift_bootstrap"
    root.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(check.out / "tc_shift_gauges.parquet")
    trials = np.array(stp.TC_TRIAL_SHIFTS_M)
    ref = f"gauge_rp{stp.REFERENCE_RP}_m"
    err = np.stack([g[f"rp{stp.REFERENCE_RP}_shift_{t:g}_m"] - g[ref] for t in trials])
    basin = g.basin.to_numpy().astype(str)
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(BOOTSTRAP_REPS):
        k = rng.integers(0, len(g), len(g))
        loo, own, all_value = _basin_values(trials, err[:, k], basin[k])
        rows += [{"rep": rep, "fit": "basin left out", "basin": b, "value": v} for b, v in loo.items()]
        rows += [{"rep": rep, "fit": "own basin", "basin": b, "value": v} for b, v in own.items()]
        rows.append({"rep": rep, "fit": "all gauges", "basin": "all", "value": all_value})
    boot = pd.DataFrame(rows)
    loo, own, all_value = _basin_values(trials, err, basin)
    point = pd.DataFrame([{"fit": "basin left out", "basin": b, "value": v} for b, v in loo.items()]
                         + [{"fit": "own basin", "basin": b, "value": v} for b, v in own.items()]
                         + [{"fit": "all gauges", "basin": "all", "value": all_value}])
    q = boot.groupby(["fit", "basin"]).value.quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = ["p05", "p50", "p95"]
    table = point.set_index(["fit", "basin"]).join(q)
    table["gauges"] = [int((basin == b).sum()) if b != "all" else len(g) for _, b in table.index]
    table["at_trial_edge_share"] = boot.groupby(["fit", "basin"]).value.apply(
        lambda v: float(np.mean((v <= trials[0]) | (v >= trials[-1]))))
    table.to_csv(root / "shifts.csv")
    print(table.round(3).to_string(), flush=True)


def _water_window(src, lon: float, lat: float, half_km: float):
    """GEBCO water (elevation < 0) around a point on ~square cells, its cell size (m) and a lon/lat -> (row, col)."""
    dlat = half_km / KM_PER_DEG
    dlon = dlat / max(np.cos(np.radians(lat)), 0.05)
    # clipped at the raster edge: neighbours across the antimeridian are not connected
    window = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform).round_offsets()         .round_lengths().intersection(Window(0, 0, src.width, src.height))
    z = src.read(1, window=window)
    water = (z < 0) & (z != src.nodata)
    t = window_transform(window, src.transform)
    # merge columns to ~square cells; a merged column is water where any of its columns is, so channels stay open
    n_cols = max(2, int(round(water.shape[1] * np.cos(np.radians(lat)))))
    edges = np.linspace(0, water.shape[1], n_cols + 1).astype(int)
    square = np.stack([water[:, a:b].any(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    col_deg = (water.shape[1] * t.a) / n_cols

    def rc(lonlat: np.ndarray) -> np.ndarray:
        return np.column_stack([np.clip(((t.f - lonlat[:, 1]) / -t.e).astype(int), 0, square.shape[0] - 1),
                                np.clip(((lonlat[:, 0] - t.c) / col_deg).astype(int), 0, n_cols - 1)])
    return square, abs(t.e) * KM_PER_DEG * 1000.0, rc


def _along_km(src, point: np.ndarray, others: np.ndarray, radius_km: float) -> np.ndarray:
    """Water-path distance (km) from `point` (lon, lat) to each of `others`; inf where not connected."""
    if not len(others):
        return np.zeros(0)
    water, cell_m, rc = _water_window(src, float(point[0]), float(point[1]), radius_km + WINDOW_MARGIN_KM)
    return along_water_distances(water, cell_m, rc(point[None, :]), rc(others), limit_m=radius_km * 1000.0)[0] / 1000.0


def _neighbour_factors(points: np.ndarray, fes_amp_points: np.ndarray, g_lonlat: np.ndarray, g_fes_amp: np.ndarray,
                       log_ratio: np.ndarray, exclude: list[np.ndarray], src) -> pd.DataFrame:
    """Tide factor per point under each neighbour rule; `exclude`: gauge rows left out per point."""
    tree = BallTree(np.radians(g_lonlat[:, ::-1]), metric="haversine")
    near, dist = tree.query_radius(np.radians(points[:, ::-1]), r=tdc.CORRECTION_KM / stp.EARTH_RADIUS_KM,
                                   return_distance=True)
    rows = []
    for j, (i, d) in enumerate(zip(near, dist)):
        keep = ~np.isin(i, exclude[j]) & np.isfinite(log_ratio[i])
        i = i[keep]
        along = _along_km(src, points[j], g_lonlat[i], tdc.CORRECTION_KM)
        similar = np.abs(np.log(g_fes_amp[i] / fes_amp_points[j]))
        rules = {"25 km (production)": np.ones(len(i), bool), "along water": np.isfinite(along)}
        rules |= {f"FES amplitude within x{x:g}": similar <= np.log(x) for x in TIDE_SIMILAR}
        rules[f"along water + FES within x{TIDE_SIMILAR_RULE:g}"] = rules["along water"] & (similar <= np.log(TIDE_SIMILAR_RULE))
        row = {}
        for name, sel in rules.items():
            row[name] = float(np.exp(np.median(log_ratio[i[sel]]))) if sel.any() else 1.0
            row[f"n {name}"] = int(sel.sum())
        rows.append(row)
        if j % 250 == 0:
            print(f"  {j}/{len(points)} points", flush=True)
    return pd.DataFrame(rows)


def tide_neighbours(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "tide_neighbours"
    root.mkdir(parents=True, exist_ok=True)
    gt = pd.read_parquet(check.out / "gauges" / "tide_gauges.parquet")
    g_lonlat = gt[["lon", "lat"]].to_numpy(dtype=float)
    st_lonlat = check.stations[["lon", "lat"]].to_numpy(dtype=float)
    site_tree = BallTree(np.radians(g_lonlat[:, ::-1]), metric="haversine")
    src = rasterio.open(get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"],
                                         root=check.cfg["paths"]["root"]).get_source("gebco").path)

    # amplitude at the usable tide gauges, each with its own site left out; atlas of the nearest station
    use = gt[gt.usable].reset_index(drop=True)
    u_lonlat = use[["lon", "lat"]].to_numpy(dtype=float)
    _, k = BallTree(np.radians(st_lonlat[:, ::-1]), metric="haversine").query(np.radians(u_lonlat[:, ::-1]), k=1)
    atlas = check.lev.atlas.to_numpy()[k[:, 0]]
    fes_amp = np.where(atlas == "fes2014", use.amp_fes2014_m, use.amp_fes2022_m)
    same = site_tree.query_radius(np.radians(u_lonlat[:, ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)
    # the neighbours' ratios to the target's own atlas
    factors = []
    for a in ("fes2022", "fes2014"):
        sel = np.flatnonzero(atlas == a)
        f = _neighbour_factors(u_lonlat[sel], fes_amp[sel], g_lonlat, gt[f"amp_{a}_m"].to_numpy(dtype=float),
                               gt[f"log_ratio_{a}"].to_numpy(dtype=float), [same[j] for j in sel], src)
        factors.append(f.set_index(sel))
    factors = pd.concat(factors).sort_index()
    amp = use[["file_name", "lon", "lat"]].assign(atlas=atlas, amp_gauge_m=tdc.tide_amplitude(use), amp_fes_m=fes_amp,
                                                  country=use.file_name.str.split("-").str[-2].str.upper())
    amp = amp.join(factors)
    amp.to_parquet(root / "amplitude.parquet")
    rules = [c for c in factors if not c.startswith("n ")]
    rows = []
    groups = {"all": np.ones(len(amp), bool), "gauge amplitude > 2 m": amp.amp_gauge_m.to_numpy() > 2.0,
              "rule changes the neighbours": (factors[rules[1]] != factors[rules[0]]).to_numpy()} | {
        f"country {c}": (amp.country == c).to_numpy() for c in ("CAN", "USA", "GBR", "FRA", "NOR", "AUS", "JPN")}
    for label, sel in groups.items():
        for name in ["FES, uncorrected"] + rules:
            corr = amp.amp_fes_m * (1.0 if name == "FES, uncorrected" else amp[name])
            rel = (corr / amp.amp_gauge_m - 1)[sel]
            rows.append({"gauges": label, "tide": name, "n": int(sel.sum()), "median_rel": rel.median(),
                         "mae_rel": rel.abs().mean(), "off_25pct": int((rel.abs() > 0.25).sum())})
    amp_scores = pd.DataFrame(rows).round(3)
    amp_scores.to_csv(root / "amplitude_scores.csv", index=False)

    # levels at the ratio gauges: the station's factor under each rule, the scored gauge's site left out
    st = check.meta.station.to_numpy()
    pos = check.stations.index.get_indexer(st)
    s_atlas = check.lev.atlas.loc[st].to_numpy()
    s_amp = np.array([tdc.fes_amplitude(check.hc[a][[p]], check.constituents[a])[0] for a, p in zip(s_atlas, pos)])
    g_pts = check.gauges.loc[check.meta.file_name, ["lon", "lat"]].to_numpy(dtype=float)
    sites = site_tree.query_radius(np.radians(g_pts[:, ::-1]), r=DEDUP_KM / stp.EARTH_RADIUS_KM)
    st_factors = []
    for a in ("fes2022", "fes2014"):
        sel = np.flatnonzero(s_atlas == a)
        f = _neighbour_factors(st_lonlat[pos[sel]], s_amp[sel], g_lonlat, gt[f"amp_{a}_m"].to_numpy(dtype=float),
                               gt[f"log_ratio_{a}"].to_numpy(dtype=float), [sites[j] for j in sel], src)
        st_factors.append(f.set_index(sel))
    st_factors = pd.concat(st_factors).sort_index()
    st_factors.index = check.meta.file_name.to_numpy()
    st_factors.to_parquet(root / "station_factors.parquet")
    table, cols = None, {}
    level_rules = ["25 km (production)", "along water", f"FES amplitude within x{TIDE_SIMILAR_RULE:g}",
                   f"along water + FES within x{TIDE_SIMILAR_RULE:g}"]
    for n, name in enumerate(level_rules):
        sub = copy.copy(check)
        sub.meta = check.meta.assign(tide_factor_loo=st_factors[name].to_numpy())
        res = sub.table(sub.levels(workers))
        if table is None:
            table = res.drop(columns=[c for c in res if c.startswith("new_loo")])
        for rp in GAUGE_RPS:
            table[f"v{n}_rp{rp}_m"] = res[f"new_loo_rp{rp}_m"]
        cols[name] = f"v{n}"
        print(f"  levels: {name}", flush=True)
    table.to_parquet(root / "levels.parquet")
    s = pd.concat([scores(table, cols).assign(split="TC group"), scores(table, cols, breakdown_groups(table))])
    s.to_csv(root / "level_scores.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(amp_scores.pivot_table(index="gauges", columns="tide", values="mae_rel", sort=False).round(3).to_string())
        print(amp_scores.groupby("gauges").n.first().to_string())
        print(s[s.rp.isin([2, 100])].pivot_table(index=["split", "gauges", "rp"], columns="level", values="mae",
                                                 sort=False)[list(cols)].round(3).to_string(), flush=True)


def _gauge_row(site: dict, records: Path, surge, surge_idx, pos: dict, b_re, b_im, month: np.ndarray) -> dict | None:
    data = np.load(records / f"{site['file_name']}.npz")
    idx, z = tp.on_hourly_grid(data["t"], data["z"])
    if not len(idx):
        return None
    n_hours = b_re.shape[0]
    tg, tf = tp.predict(b_re, b_im, site["hc_gauge"]), tp.predict(b_re, b_im, site["hc_fes"])
    obs = np.full(n_hours, np.nan)
    obs[idx] = tp.remove_annual_means(idx, z)
    gtsm = np.full(n_hours, np.nan)
    gtsm[surge_idx] = station_series(surge, pos[site["gtsm_index"]])
    year = tp.hour_years(np.arange(n_hours))
    skew, hw_at, height_at, years = {}, {}, {}, {}
    for name, total, tide in (("gauge", obs, tg), ("gtsmfes", gtsm + tf, tf)):
        hw, height = ss.high_waters(tide, ss.HW_PROMINENCE)
        skew[name], hw_at[name], height_at[name] = ss.skew_surges(total, hw, height), hw, height
        yr = pd.Series(np.isfinite(skew[name])).groupby(year[hw]).mean()
        years[name] = set(yr.index[yr >= stp.YEAR_COVERAGE])
    common = sorted(years["gauge"] & years["gtsmfes"])
    if len(common) < stp.MIN_YEARS:
        return None
    row = {"file_name": site["file_name"], "n_years": len(common)}
    models = {}
    for name in skew:
        keep = np.isin(year[hw_at[name]], common) & np.isfinite(skew[name])
        models[name] = ss.surge_model(skew[name][keep], len(common))
        s, h = skew[name][keep], height_at[name][keep]
        # skew surge by high-water height tertile: independence of surge and tide
        edges = np.quantile(h, [1 / 3, 2 / 3])
        for t, sel in enumerate((h <= edges[0], (h > edges[0]) & (h <= edges[1]), h > edges[1])):
            for q in TERTILE_QUANTILES:
                row[f"{name}_t{t}_q{q * 1000:g}_m"] = float(np.quantile(s[sel], q))
    for rp in R_RPS:
        row[f"r_rp{rp}"] = ss.skew_level(models["gauge"], rp) / ss.skew_level(models["gtsmfes"], rp)
    in_common = np.isin(year, common)
    resid = obs - tg
    for name, series in (("gauge", resid), ("gtsm", gtsm), ("fes_sa_ssa", tp.predict(b_re, b_im, site["hc_seasonal"]))):
        ok = in_common & np.isfinite(series)
        clim = pd.Series(series[ok]).groupby(month[ok]).mean().reindex(range(12))
        for m, v in enumerate(clim - clim.mean()):
            row[f"{name}_month{m + 1:02d}_m"] = v
    am = []
    for y in common:
        sel = np.flatnonzero((year == y) & np.isfinite(obs))
        if len(sel):
            am.append(month[sel[np.argmax(obs[sel])]])
    for m in range(12):
        row[f"annual_max_month{m + 1:02d}"] = int(np.sum(np.array(am) == m))
    return row


def _gauge_task(args) -> list[dict]:
    sites, records, store, hourly = args
    b_re, b_im = (np.load(p, mmap_mode="r") for p in hourly)
    surge_st, surge_time, surge = load_series(Path(store))
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = tp.on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    month = (tp.HOURLY_START + np.arange(b_re.shape[0]).astype("timedelta64[h]")).astype("datetime64[M]") \
        .astype(int) % 12
    rows = []
    for site in sites:
        try:
            r = _gauge_row(site, Path(records), surge, surge_idx, pos, b_re, b_im, month)
        except (ValueError, IndexError) as err:
            print(f"  skipped {site['file_name']}: {type(err).__name__}: {err}", flush=True)
            continue
        if r is not None:
            rows.append(r)
    return rows


def _shelf_grid(src, path: Path) -> tuple[np.ndarray, float]:
    """Global GEBCO elevation on a SHELF_GRID_DEG grid (cached to `path`)."""
    if not path.exists():
        shape = (int(round(180 / SHELF_GRID_DEG)), int(round(360 / SHELF_GRID_DEG)))
        np.save(path, src.read(1, out_shape=shape))
    return np.load(path), SHELF_GRID_DEG


def _covariates(check: GaugeCheck, gauges: pd.DataFrame, dest: Path) -> pd.DataFrame:
    """GEBCO depth (m, positive down) at each gauge's GTSM point and its distance to the shelf edge (km)."""
    catalog = get_data_catalog(REPO / check.cfg["paths"]["hydromt_data_catalog"], root=check.cfg["paths"]["root"])
    src = rasterio.open(catalog.get_source("gebco").path)
    lonlat, _ = gtsm_stations(Path(check.cfg["storm_tide_product"]["gtsm_dir"]))
    pts = lonlat[gauges.gtsm_index.to_numpy()]
    depth = -np.array([v[0] for v in src.sample([tuple(p) for p in pts])], dtype=float)
    z, res = _shelf_grid(src, dest / "gebco_shelf_grid.npy")
    deep = z < -SHELF_EDGE_M
    # shelf-edge cells: deep cells next to shallower ones
    edge = deep & ~(np.roll(deep, 1, 0) & np.roll(deep, -1, 0) & np.roll(deep, 1, 1) & np.roll(deep, -1, 1))
    r, c = np.nonzero(edge)
    edge_ll = np.column_stack([90 - (r + 0.5) * res, -180 + (c + 0.5) * res])
    d, _ = BallTree(np.radians(edge_ll), metric="haversine").query(np.radians(pts[:, ::-1]), k=1)
    return pd.DataFrame({"gtsm_depth_m": depth, "shelf_edge_km": d[:, 0] * stp.EARTH_RADIUS_KM}, index=gauges.index)


def gauge_series(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "gauge_series"
    root.mkdir(parents=True, exist_ok=True)
    g = check.gauges[np.isfinite(check.gauges.ratio_fes)]
    hc = np.load(check.out / "gauges" / "gauge_constants.npz")
    pos = {f: i for i, f in enumerate(hc["file_name"])}
    seasonal = [check.constituents["fes2022"].index(c) for c in tp.SEASONAL if c in check.constituents["fes2022"]]
    sites = []
    for f, gi in zip(g.index, g.gtsm_index):
        hg, hf = hc["gauge"][pos[f]].copy(), hc["fes"][pos[f]].copy()
        hs = np.zeros_like(hf)
        hs[seasonal] = hf[seasonal]
        hg[seasonal] = hf[seasonal] = 0.0
        sites.append({"file_name": f, "gtsm_index": int(gi), "hc_gauge": hg, "hc_fes": hf, "hc_seasonal": hs})
    hourly = (str(check.cache / "fes2022_hourly_re.npy"), str(check.cache / "fes2022_hourly_im.npy"))
    tasks = [(sites[i:i + stp.TASK_SIZE], str(check.cache / "gesla"), str(check.cache / "gtsm_surge_gauges"), hourly)
             for i in range(0, len(sites), stp.TASK_SIZE)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(_gauge_task, tasks)):
            rows += r
            if k % 10 == 0:
                print(f"  {min((k + 1) * stp.TASK_SIZE, len(sites))}/{len(sites)} gauges", flush=True)
    out = g.join(pd.DataFrame(rows).set_index("file_name"), how="inner", rsuffix="_diag")
    out = out.join(_covariates(check, out, root))
    out["region"] = check.reference.region.reindex(out.index)
    out.to_parquet(root / "gauges.parquet")
    print(f"{len(out)} gauges", flush=True)


def _loo_predictions(cal: pd.DataFrame) -> pd.DataFrame:
    """ln r per calibration gauge predicted with that gauge left out, per fallback option."""
    x = {"S10": np.log(cal.skew_gtsmfes_m), "depth": np.log(cal.gtsm_depth_m.clip(lower=1.0)),
         "shelf": np.log(cal.shelf_edge_km.clip(lower=1.0)), "range": np.log((2 * cal.mhws_gauge_m).clip(lower=0.1))}
    y = np.log(cal.ratio_fes.to_numpy())
    designs = {"power law on S10 (production)": ["S10"], "depth + shelf + range": ["depth", "shelf", "range"],
               "S10 + depth + shelf + range": ["S10", "depth", "shelf", "range"]}
    tree = BallTree(np.radians(cal[["lat", "lon"]].to_numpy()), metric="haversine")
    idx, dist = tree.query_radius(np.radians(cal[["lat", "lon"]].to_numpy()), r=stp.IDW_RADIUS_KM / stp.EARTH_RADIUS_KM,
                                  return_distance=True)
    out = {"r = 1": np.zeros(len(cal)), "median r": np.empty(len(cal))} | {k: np.empty(len(cal)) for k in designs}
    out["IDW (production, >= 3 gauges)"] = np.full(len(cal), np.nan)
    neighbours = np.zeros(len(cal), dtype=int)
    for j in range(len(cal)):
        other = np.arange(len(cal)) != j
        out["median r"][j] = np.median(y[other])
        for name, cols in designs.items():
            a = np.column_stack([np.ones(len(cal))] + [x[c].to_numpy() for c in cols])
            coef = np.linalg.lstsq(a[other], y[other], rcond=None)[0]
            out[name][j] = a[j] @ coef
        keep = idx[j] != j
        i, d = idx[j][keep], dist[j][keep]
        neighbours[j] = len(i)
        if len(i) >= stp.MIN_NEIGHBOURS:
            w = 1.0 / np.maximum(d * stp.EARTH_RADIUS_KM, 1.0)
            out["IDW (production, >= 3 gauges)"][j] = np.log(np.sum(w * np.exp(y[i])) / w.sum())
    pred = pd.DataFrame(out, index=cal.index)
    pred["neighbours"] = neighbours
    pred["ln_r"] = y
    return pred


def gauge_analysis(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "gauge_series"
    g = pd.read_parquet(root / "gauges.parquet")
    g["tc_dominated"] = g.station_class == "tc_dominated"
    lines = []

    # r across return periods
    for label, sel in (("all", g.n_years >= stp.MIN_YEARS), (f">= {LONG_YEARS} years", g.n_years >= LONG_YEARS)):
        for cls, s in g[sel].groupby("tc_dominated"):
            for rp in (2, 50):
                q = (s[f"r_rp{rp}"] / s.r_rp10).quantile([0.1, 0.5, 0.9]).round(3).tolist()
                lines.append(f"r_rp{rp} / r_rp10, {label}, {'TC-dominated' if cls else 'not TC-dominated'} "
                             f"(n={len(s)}): p10/p50/p90 {q}")
    r_rp = g[["region", "n_years", "station_class"] + [f"r_rp{rp}" for rp in R_RPS]]
    r_rp.to_csv(root / "r_by_rp.csv")

    # r covariates and the fallback options, gauge left out
    cal = g[g.calibrate & (g.skew_gtsmfes_m > 0) & np.isfinite(g.gtsm_depth_m)]
    for c, v in (("S10", np.log(cal.skew_gtsmfes_m)), ("GTSM depth", np.log(cal.gtsm_depth_m.clip(lower=1.0))),
                 ("shelf edge distance", np.log(cal.shelf_edge_km.clip(lower=1.0))),
                 ("spring range", np.log((2 * cal.mhws_gauge_m).clip(lower=0.1)))):
        lines.append(f"Spearman ln r vs ln {c}: {pd.Series(np.log(cal.ratio_fes)).corr(v, method='spearman'):.2f}")
    pred = _loo_predictions(cal)
    pred.to_parquet(root / "r_fallback_loo.parquet")
    rows = []
    options = [c for c in pred if c not in ("neighbours", "ln_r")]
    for label, sel in (("all calibration gauges", np.ones(len(pred), bool)),
                       (f"< {stp.MIN_NEIGHBOURS} other gauges within {stp.IDW_RADIUS_KM:g} km",
                        pred.neighbours.to_numpy() < stp.MIN_NEIGHBOURS)):
        for name in options:
            e = (pred[name] - pred.ln_r)[sel].dropna()
            rows.append({"gauges": label, "r from": name, "n": len(e), "mae_ln_r": e.abs().mean(),
                         "median_ln_r": e.median(), "share_off_20pct": float((np.abs(np.expm1(e)) > 0.2).mean())})
    fallback = pd.DataFrame(rows).round(3)
    fallback.to_csv(root / "r_fallback_scores.csv", index=False)

    # surge-tide independence: skew-surge quantile in the top vs the bottom high-water tertile
    dep = g[["region", "station_class", "mhws_gauge_m"]].copy()
    for name in ("gauge", "gtsmfes"):
        for q in TERTILE_QUANTILES:
            dep[f"{name}_q{q * 1000:g}_top_over_bottom"] = g[f"{name}_t2_q{q * 1000:g}_m"] / g[f"{name}_t0_q{q * 1000:g}_m"]
    dep["range_class"] = pd.cut(2 * dep.mhws_gauge_m, (0, 1, 2, 4, 20)).astype(str)
    dep.to_csv(root / "tide_dependence.csv")
    tide_dep = dep.groupby("range_class")[[c for c in dep if c.endswith("top_over_bottom")]].median().round(3)
    tide_dep["n"] = dep.groupby("range_class").size()

    # seasonal cycle: gauge residual minus GTSM, weighted by the months of the gauge's annual maxima
    months = [f"{m:02d}" for m in range(1, 13)]
    gap = g[[f"gauge_month{m}_m" for m in months]].to_numpy() - g[[f"gtsm_month{m}_m" for m in months]].to_numpy()
    fes = g[[f"fes_sa_ssa_month{m}_m" for m in months]].to_numpy()
    w = g[[f"annual_max_month{m}" for m in months]].to_numpy(dtype=float)
    w /= w.sum(axis=1, keepdims=True)
    season = pd.DataFrame({"region": g.region, "gap_amplitude_m": gap.max(axis=1) - gap.min(axis=1),
                           "gap_at_annual_maxima_m": (gap * w).sum(axis=1),
                           "fes_sa_ssa_at_annual_maxima_m": (fes * w).sum(axis=1)}, index=g.index)
    season.to_csv(root / "seasonal_gap.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print("\n".join(lines))
        print(fallback.to_string(index=False))
        print(tide_dep.to_string())
        print(season.describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())
        reg = season.groupby(season.region.where(season.region.map(season.region.value_counts()) >= 15, "other"))
        print(reg.gap_at_annual_maxima_m.describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string(), flush=True)


def _despike_task(args) -> list[dict]:
    chunk, stores, bases, start = args
    out = []
    for st, g, _, _, _ in stp.station_records(chunk, stores, bases, start, nodal=False):
        dropped = np.isfinite(g) & ~np.isfinite(ss.despike(g))
        near = np.zeros(len(g), bool)
        near[st["hours"]] = True
        out.append({"station": st["station"], "hours_near_track": int((near & np.isfinite(g)).sum()),
                    "hours_away": int((~near & np.isfinite(g)).sum()), "dropped_near_track": int((dropped & near).sum()),
                    "dropped_away": int((dropped & ~near).sum()),
                    "largest_dropped_near_track_m": float(np.nanmax(np.where(dropped & near, g, np.nan)))
                    if (dropped & near).any() else np.nan})
    return out


def despike_tracks(check: GaugeCheck, dest: Path, workers: int) -> None:
    root = dest / "despike_tracks"
    root.mkdir(parents=True, exist_ok=True)
    lev = check.lev
    ks = np.flatnonzero(lev.station_id.isin(check.tc_levels.index).to_numpy() & (lev.level_source == "product").to_numpy())
    stores, bases, start, n_hours = check.records[PRODUCTION_RECORD]
    hours = stp.tc_hours(check.stations.iloc[ks], n_hours, Path(check.cfg["storm_tide_product"]["ibtracs_path"]), start)
    empty = np.array([], dtype=int)
    todo = [{"station": int(st), "gtsm_index": int(check.stations.gtsm_index.iloc[k]), "atlas": lev.atlas.iloc[k],
             "hc": check.hc[lev.atlas.iloc[k]][k], "hours": hours.get(st, empty)}
            for k, st in zip(ks, check.stations.index[ks])]
    tasks = [(todo[i:i + stp.TASK_SIZE], [str(p) for p in stores], bases, start)
             for i in range(0, len(todo), stp.TASK_SIZE)]
    with ProcessPoolExecutor(workers) as pool:
        out = pd.DataFrame([r for res in pool.map(_despike_task, tasks) for r in res]).set_index("station")
    out = lev[["station_id", "lon", "lat", "tc_basin"]].join(out, how="inner")
    out.to_parquet(root / "stations.parquet")
    near_rate = out.dropped_near_track.sum() / out.hours_near_track.sum()
    away_rate = out.dropped_away.sum() / out.hours_away.sum()
    print(f"{len(out)} stations with a TC part: despiked share of hours near tracks {near_rate:.2e}, away "
          f"{away_rate:.2e}; stations with any despiked hour near a track {(out.dropped_near_track > 0).sum()}; "
          f"largest despiked surge near a track p50/p99/max "
          f"{np.nanquantile(out.largest_dropped_near_track_m, [0.5, 0.99]).round(2)} "
          f"{np.nanmax(out.largest_dropped_near_track_m):.2f} m", flush=True)
    print(out.groupby("tc_basin")[["dropped_near_track", "dropped_away", "hours_near_track"]].sum().to_string())


STAGES = {"tc_raise": tc_raise, "tc_mask_et": tc_mask_et, "shift_bootstrap": shift_bootstrap,
          "tide_neighbours": tide_neighbours, "gauge_series": gauge_series, "gauge_analysis": gauge_analysis,
          "despike_tracks": despike_tracks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    check = GaugeCheck(Path(args.product_dir), Path(args.sources_dir), load_config(args.config, args.override))
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        STAGES[stage](check, dest, args.workers)


if __name__ == "__main__":
    main()
