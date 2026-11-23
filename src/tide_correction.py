"""Gauge correction of the FES tide: GESLA/FES amplitude ratios spread to points within CORRECTION_KM.

Entry points: `gauge_table` (short harmonic fits of the GESLA sites and their ratio to FES),
`water_links` (which gauges a point reaches through water) and `correction_factors`. A point
takes only the gauges it reaches through water within CORRECTION_KM, so a gauge in the next fjord
or across a peninsula does not correct it. The tide amplitude is the sum of the TIDE_SPEEDS constituent amplitudes, so
semidiurnal and diurnal regimes are both measured; the factor scales every FES constant alike
(amplitude only, phases kept). Method and the radius test: input analysis, `open_guesses.py
tide_correction`; the water rule: `review_diagnostics.py tide_neighbours`.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely import STRtree, box, unary_union
from sklearn.neighbors import BallTree

import tide_prediction as tp
from along_water import along_water_distances, land_water_window, window_bounds
from process_pool import process_pool

from gesla import EARTH_RADIUS_KM, GOOD_QC, deduplicate, read_metadata

TIDE_SPEEDS = {"m2": 28.9841042, "s2": 30.0, "n2": 28.4397295, "k1": 15.0410686, "o1": 13.9430356}
FIT_SPEEDS = {**TIDE_SPEEDS, "m4": 57.9682084}  # M4 is fitted so shallow-water distortion does not leak into M2
FIT_DAYS = 2 * 366  # the first two years of a record; nodal modulation (M2 +-4%, K1/O1 +-11%) is left in
MIN_DAYS = 28  # separates M2 from N2 (Rayleigh period 27.6 d)
MIN_COVER = 0.7  # samples per hour of record span
MAX_FLOOR_SHARE = 0.03  # a gauge that dries at low water records a flat floor and fits too small a tide
MAX_RANGE_M = 20.0  # above the largest tidal range observed (~16 m, Bay of Fundy): a record in feet
CORRECTION_KM = 25.0  # leave-one-out best of 10 / 25 / 50 / 100 / 200 km over the GESLA gauges
# water outside the OSM land polygons: 100 m cells keep the graph small, 25 m sub-cells keep
# fjord and inlet channels open (GEBCO's ~460 m cells cut them)
WATER_CELL_M, WATER_SUPERSAMPLE = 100.0, 4
WATER_MARGIN_KM = 5.0
CHUNK = 8
LINK_CHUNK = 16


def tide_sites(gesla_dir: Path) -> pd.DataFrame:
    """Coastal GESLA sites with no obvious issues, any record length, one per DEDUP_KM."""
    df = read_metadata(gesla_dir)
    ok = (df.gauge_type == "Coastal") & (df.overall_quality == "No obvious issues")
    return deduplicate(df[ok]).reset_index(drop=True)


def fit_record(path: Path) -> dict | None:
    """Constituent amplitudes (m) and record checks of the first FIT_DAYS of one GESLA file."""
    try:
        df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=["date", "time", "z", "qc", "use"],
                         dtype={"date": str, "time": str}, nrows=FIT_DAYS * 24 * 4)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    df = df[(df.use == 1) & df.qc.isin(GOOD_QC)]
    if len(df) < 2:
        return None
    t = pd.to_datetime(df.date + " " + df.time, format="%Y/%m/%d %H:%M:%S")
    hours = ((t - t.iloc[0]) / pd.Timedelta(hours=1)).to_numpy()
    hours, z = hours[hours <= FIT_DAYS * 24], df.z.to_numpy(float)[hours <= FIT_DAYS * 24]
    span = float(np.ptp(hours))
    if span < MIN_DAYS * 24:
        return None
    t0 = (t.iloc[0] - pd.Timestamp("1900-01-01")) / pd.Timedelta(hours=1)
    cols = [np.ones_like(hours)]
    for w in FIT_SPEEDS.values():
        a = np.radians(w) * (hours + t0)
        cols += [np.cos(a), np.sin(a)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z - z.mean(), rcond=None)
    amp = {f"amp_{k}_m": float(np.hypot(coef[1 + 2 * i], coef[2 + 2 * i])) for i, k in enumerate(FIT_SPEEDS)}
    floor = np.quantile(z, 0.01)
    return {**amp, "days": span / 24.0, "cover": len(z) / max(span, 1.0),
            "floor_share": float(np.mean(z < floor + 0.05)), "range_m": float(np.ptp(np.quantile(z, [0.001, 0.999])))}


def _fit_task(args) -> list:
    gesla_dir, names = args
    return [fit_record(Path(gesla_dir) / n) for n in names]


def gauge_tides(gesla_dir: Path, sites: pd.DataFrame, workers: int) -> pd.DataFrame:
    """One row per site with a fit: file_name, lon, lat, amp_<constituent>_m and the record checks."""
    names = sites.file_name.tolist()
    tasks = [(str(gesla_dir), names[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    with process_pool(workers) as pool:
        fits = [f for chunk in pool.map(_fit_task, tasks) for f in chunk]
    rows = [{"file_name": n, "lon": float(s.lon), "lat": float(s.lat), **f}
            for n, (_, s), f in zip(names, sites.iterrows(), fits) if f is not None]
    return pd.DataFrame(rows)


def tide_amplitude(amps: pd.DataFrame) -> np.ndarray:
    return amps[[f"amp_{k}_m" for k in TIDE_SPEEDS]].sum(axis=1).to_numpy()


def fes_amplitude(hc: np.ndarray, constituents: list[str]) -> np.ndarray:
    """Sum of the TIDE_SPEEDS amplitudes (m) of FES constants (point, constituent)."""
    return np.abs(hc[:, [constituents.index(k) for k in TIDE_SPEEDS]]).sum(axis=1)


def usable(gt: pd.DataFrame) -> np.ndarray:
    return ((gt.cover >= MIN_COVER) & (gt.floor_share < MAX_FLOOR_SHARE) & (gt.range_m <= MAX_RANGE_M)).to_numpy()


def log_ratios(gt: pd.DataFrame, fes_amp: np.ndarray) -> np.ndarray:
    """log(gauge / FES tide amplitude); NaN where either is missing or zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(tide_amplitude(gt) / fes_amp)
    return np.where(np.isfinite(lr), lr, np.nan)


def gauge_table(gesla_dir: Path, models: dict, workers: int) -> pd.DataFrame:
    """Fits of the coastal GESLA sites, `usable`, and per atlas (name -> pyTMD model) amp_<atlas>_m
    and log_ratio_<atlas> (NaN where the gauge is not usable)."""
    gt = gauge_tides(Path(gesla_dir), tide_sites(Path(gesla_dir)), workers)
    gt["usable"] = usable(gt)
    lonlat = gt[["lon", "lat"]].to_numpy(dtype=float)
    for a, model in models.items():
        hc = tp.fill_masked(tp.grouped_constants(model, lonlat), lonlat)
        gt[f"amp_{a}_m"] = fes_amplitude(hc, list(model.constituents))
        gt[f"log_ratio_{a}"] = np.where(gt.usable, log_ratios(gt, gt[f"amp_{a}_m"].to_numpy()), np.nan)
    return gt


def _links_task(args) -> list[tuple[int, np.ndarray]]:
    out = []
    for g, lonlat, cand, cand_lonlat, land in args:
        water, rc = land_water_window(land, lonlat[0], lonlat[1], CORRECTION_KM + WATER_MARGIN_KM,
                                      WATER_CELL_M, WATER_SUPERSAMPLE)
        if water.any():
            d = along_water_distances(water, WATER_CELL_M, rc(lonlat[None, :]), rc(cand_lonlat),
                                      limit_m=CORRECTION_KM * 1000.0)[0]
            out.append((g, cand[np.isfinite(d)]))
    return out


def water_links(points_lonlat: np.ndarray, gauge_lonlat: np.ndarray, land_path: Path, workers: int) -> csr_matrix:
    """(point, gauge) True where the gauge reaches the point within CORRECTION_KM through water.

    Water lies outside the polygons of `land_path` (OSM land polygons, lon/lat) on WATER_CELL_M
    cells; both ends snap to their nearest water cell. One shortest-path search per gauge.
    """
    tree = BallTree(np.radians(points_lonlat[:, ::-1]), metric="haversine")
    near = tree.query_radius(np.radians(gauge_lonlat[:, ::-1]), r=CORRECTION_KM / EARTH_RADIUS_KM)
    gauges = [g for g in range(len(gauge_lonlat)) if len(near[g])]
    half_km = CORRECTION_KM + WATER_MARGIN_KM
    windows = [box(*window_bounds(*gauge_lonlat[g], half_km)) for g in gauges]
    land = gpd.read_file(land_path, mask=unary_union(windows)).geometry.to_numpy() if gauges else np.array([])
    land_tree = STRtree(land)
    items = [(g, gauge_lonlat[g], near[g], points_lonlat[near[g]], land[land_tree.query(w)])
             for g, w in zip(gauges, windows)]
    tasks = [items[i:i + LINK_CHUNK] for i in range(0, len(items), LINK_CHUNK)]
    with process_pool(workers) as pool:
        found = [r for chunk in pool.map(_links_task, tasks) for r in chunk]
    rows = np.concatenate([p for _, p in found]) if found else np.zeros(0, int)
    cols = np.concatenate([np.full(len(p), g) for g, p in found]) if found else np.zeros(0, int)
    return csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(points_lonlat), len(gauge_lonlat)))


def correction_factors(points_lonlat: np.ndarray, gauge_lonlat: np.ndarray, log_ratio: np.ndarray,
                       radius_km: float = CORRECTION_KM, exclude_km: float = 0.0,
                       links: csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tide factor (exp of the median gauge log ratio within radius_km; 1 without a gauge) and gauge count.

    `exclude_km` > 0 leaves out gauges that close to the point (leave-one-out at a gauge: DEDUP_KM,
    which also drops records of the same site). `links` (point, gauge; `water_links`) keeps only
    the gauges that reach the point through water.
    """
    ok = np.isfinite(log_ratio)
    gauge_rows = np.flatnonzero(ok)
    glonlat, lr = gauge_lonlat[ok], log_ratio[ok]
    factor, count = np.ones(len(points_lonlat)), np.zeros(len(points_lonlat), dtype=int)
    if not len(lr):
        return factor, count
    tree = BallTree(np.radians(glonlat[:, ::-1]), metric="haversine")
    idx, dist = tree.query_radius(np.radians(points_lonlat[:, ::-1]), r=radius_km / EARTH_RADIUS_KM,
                                  return_distance=True)
    for j, (i, d) in enumerate(zip(idx, dist)):
        keep = i[d * EARTH_RADIUS_KM > exclude_km] if exclude_km > 0 else i
        if links is not None:
            keep = keep[links[j, gauge_rows[keep]].toarray()[0]]
        if len(keep):
            factor[j], count[j] = float(np.exp(np.median(lr[keep]))), len(keep)
    return factor, count
