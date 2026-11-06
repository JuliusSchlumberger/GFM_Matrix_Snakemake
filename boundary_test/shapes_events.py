"""Step 2a: storm-tide event shapes at the GESLA sites from GESLA, GTSM and COAST-HG.

Events are the annual maxima of total water level (hourly, relative to local MSL as in
`levels_extremes.py`), kept when +-WINDOW_HR around the peak is >= WINDOW_COVERAGE complete.
Per event, in hours:
  d_above_<level>    contiguous time above MSL / MHW / MHWS that contains the peak; `censored_*`
                     marks runs that reach the window edge (true duration is longer)
  cum_above_msl      cumulative time above MSL in the window
  t_within_<f>       time the level is within f x (peak - MSL) of the peak, cumulative over the
                     window: the T of correction B (`event_volume.time_within_peak`) for a cell
                     whose head over its spill level is that fraction of the storm tide
  surge_dur50        contiguous time the surge residual stays above half its maximum
plus the peak, the tide and surge at the peak, and the empirical return period of the event
among the site's annual maxima ((n + 1) / rank). COAST-HG gives one RP100 event per site (both
tide variants, 10-min series sampled hourly). GTSM tide: re-predicted from its harmonic fit.
MHW and MHWS: each source's own levels (`tide_levels.parquet`, variant full).

Writes <out-dir>/events.parquet and <out-dir>/event_profiles.npz (level / peak per event and the
surge residual in m, hourly offsets -WINDOW_HR..WINDOW_HR; row i belongs to events row i). The
surge is the observed level minus the site's harmonic tide (GESLA) or the CDS surge series (GTSM).

Usage:
    python boundary_test/shapes_events.py --out-dir .cleanup_baseline/level_sources [--workers 3]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from levels_tides import HOURLY_END, HOURLY_START, on_hourly_grid  # noqa: E402

COAST_HG = Path("P:/11212688-004-global-floodmaps/modelling/inputs/COAST_HG/COAST-HG_RP100.nc")
WINDOW_HR = 72  # +-72 h, the length of a COAST-HG hydrograph (149 h)
WINDOW_COVERAGE = 0.9
YEAR_COVERAGE = 0.8
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
SITES_PER_TASK = 10
N_HOURS = int((HOURLY_END - HOURLY_START) / np.timedelta64(1, "h"))


def contiguous_run(above: np.ndarray, centre: int) -> tuple[float, bool]:
    if not above[centre]:
        return 0.0, False
    lo, hi = centre, centre
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    return float(hi - lo + 1), bool(lo == 0 or hi == len(above) - 1)


def event_metrics(level: np.ndarray, surge: np.ndarray | None, centre: int, mhw: float, mhws: float) -> dict:
    """Metrics of one hourly window with the peak at `centre` (NaN gaps count as not above)."""
    peak = level[centre]
    finite = np.where(np.isfinite(level), level, -np.inf)
    row = {"peak_m": float(peak), "cum_above_msl": float((finite > 0).sum())}
    for name, thr in (("msl", 0.0), ("mhw", mhw), ("mhws", mhws)):
        d, cens = contiguous_run(finite > thr, centre) if np.isfinite(thr) else (np.nan, False)
        row[f"d_above_{name}"], row[f"censored_{name}"] = d, cens
    for f in FRACTIONS:
        row[f"t_within_{f:g}"] = float((finite >= peak - f * max(peak, 0.0)).sum())
    if surge is not None:
        s = np.where(np.isfinite(surge), surge, -np.inf)
        k = int(np.argmax(s))
        row.update({"surge_at_peak_m": float(surge[centre]), "tide_at_peak_m": float(peak - surge[centre]),
                    "surge_max_m": float(s[k]), "surge_dur50": contiguous_run(s > 0.5 * s[k], k)[0] if s[k] > 0 else 0.0,
                    "surge_max_lag_hr": float(k - centre)})
    return row


def series_events(level: np.ndarray, surge: np.ndarray | None, mhw: float,
                  mhws: float) -> tuple[list[dict], list, list]:
    """Annual-maximum events of a full-length hourly series (NaN where missing)."""
    years = (HOURLY_START + np.arange(N_HOURS).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    rows, profiles, surges = [], [], []
    df = pd.DataFrame({"z": level, "year": years})
    good = df.groupby("year").z.count() >= YEAR_COVERAGE * 8766
    for year in good[good].index:
        sel = np.flatnonzero(years == year)
        centre = int(sel[np.nanargmax(level[sel])])
        lo, hi = centre - WINDOW_HR, centre + WINDOW_HR + 1
        if lo < 0 or hi > N_HOURS or np.isfinite(level[lo:hi]).mean() < WINDOW_COVERAGE:
            continue
        w = level[lo:hi]
        rows.append({"year": int(year), **event_metrics(w, None if surge is None else surge[lo:hi], WINDOW_HR, mhw, mhws)})
        profiles.append((w / level[centre]).astype(np.float32))
        surges.append((np.full(len(w), np.nan) if surge is None else surge[lo:hi]).astype(np.float32))
    if rows:
        peaks = np.array([r["peak_m"] for r in rows])
        rank = (-peaks).argsort().argsort() + 1
        for r, k in zip(rows, rank):
            r["rp_empirical_yr"] = (len(rows) + 1) / k
    return rows, profiles, surges


def _full_hourly(idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    full = np.full(N_HOURS, np.nan)
    full[idx] = z
    return full


def _site_task(args) -> tuple[list[dict], list, list]:
    out, sites, hc, levels = args
    out = Path(out)
    b_re = np.load(out / "hourly_basis_re.npy", mmap_mode="r")
    b_im = np.load(out / "hourly_basis_im.npy", mmap_mode="r")
    surge_st, surge_time, surge = load_series(out, "surge")
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    rows, profiles, surges = [], [], []
    for site in sites:
        f = site["file_name"]
        for source in ("gesla", "gtsm"):
            h = hc[f"{source}_harm"][site["row"]]
            lev = levels.get((f, f"{source}_harm"), {})
            if not np.isfinite(h).all():
                continue
            tide = np.asarray(b_re) @ h.real + np.asarray(b_im) @ h.imag
            if source == "gesla":
                data = np.load(out / "gesla" / f"{f}.npz")
                idx, z = on_hourly_grid(data["t"], data["z"])
                if not len(idx):
                    continue
                year = (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]")
                z = z - pd.Series(z).groupby(year).transform("mean").to_numpy()
                level = _full_hourly(idx, z)
            else:
                level = _full_hourly(surge_idx, station_series(surge, pos[site["gtsm_index"]])) + tide
            r, p, s = series_events(level, level - tide, lev.get("mhw_m", np.nan), lev.get("mhws_m", np.nan))
            rows += [{"file_name": f, "source": source, **x} for x in r]
            profiles += p
            surges += s
    return rows, profiles, surges


def coast_hg_events(pairs: pd.DataFrame, levels: dict) -> tuple[list[dict], list, list]:
    """COAST-HG events; the dataset holds total level only, so the surge rows are NaN."""
    rows, profiles = [], []
    with netCDF4.Dataset(COAST_HG) as ds:
        for var, label in (("hydrograph_average_tide_signal", "coast_hg_average"),
                           ("hydrograph_spring_tide_signal", "coast_hg_spring")):
            h10 = np.ma.filled(ds[var][:], np.nan)[pairs.coast_hg_index.to_numpy()]
            for f, w10 in zip(pairs.file_name, h10):
                lev = levels.get((f, "gtsm_harm"), {})
                k10 = int(np.nanargmax(w10))
                w = w10[k10 % 6::6]  # hourly samples through the peak
                centre = k10 // 6
                src = np.arange(2 * WINDOW_HR + 1) + centre - WINDOW_HR
                win = np.where((src >= 0) & (src < len(w)), w[np.clip(src, 0, len(w) - 1)], np.nan)
                rows.append({"file_name": f, "source": label, "rp_empirical_yr": 100.0,
                             **event_metrics(win, None, WINDOW_HR, lev.get("mhw_m", np.nan), lev.get("mhws_m", np.nan))})
                profiles.append((win / win[WINDOW_HR]).astype(np.float32))
    return rows, profiles, [np.full(2 * WINDOW_HR + 1, np.nan, dtype=np.float32)] * len(profiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out_dir)
    pairs = pd.read_parquet(out / "pairs.parquet").reset_index(drop=True)
    hc_file = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    hc = {k: hc_file[k] for k in ("gesla_harm", "gtsm_harm")}
    tl = pd.read_parquet(out / "tide_levels.parquet")
    tl = tl[tl.variant == "full"]
    levels = {(r.file_name, r.source): {"mhw_m": r.mhw_m, "mhws_m": r.mhws_m} for r in tl.itertuples()}
    sites = [{"file_name": f, "gtsm_index": g, "row": i} for i, (f, g) in enumerate(zip(pairs.file_name, pairs.gtsm_index))]
    tasks = [(str(out), sites[i:i + SITES_PER_TASK], hc, levels) for i in range(0, len(sites), SITES_PER_TASK)]
    rows, profiles, surges = coast_hg_events(pairs, levels)
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (r, p, s) in enumerate(pool.map(_site_task, tasks)):
            rows += r
            profiles += p
            surges += s
            if k % 10 == 0:
                print(f"  {min((k + 1) * SITES_PER_TASK, len(sites))}/{len(sites)} sites", flush=True)
    events = pd.DataFrame(rows)
    events.to_parquet(out / "events.parquet")
    np.savez(out / "event_profiles.npz", profiles=np.vstack(profiles), surge_m=np.vstack(surges),
             offset_hr=np.arange(-WINDOW_HR, WINDOW_HR + 1))
    print(events.groupby("source")[["d_above_msl", "t_within_0.25", "t_within_0.5"]].median().round(1).to_string())


if __name__ == "__main__":
    main()
