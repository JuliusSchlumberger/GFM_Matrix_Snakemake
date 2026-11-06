"""Test 7: split COAST-HG into its tide and surge parts and score each against the gauge events.

COAST-HG (HGRAPHER) is a GTSM surge shape scaled so that surge + tide peaks at COAST-RP RP100,
laid on a GTSM tide stretch; the dataset holds only the total. The tide stretch is recovered
from the station's GTSM harmonic tide (`harmonic_constants.npz`, predicted hourly 1979-2024):

- spring variant: the hourly stretch and constant offset that best match the hydrograph where
  the surge has died out (|t| >= EDGE_HR from the peak), by least squares;
- average variant: its surge is larger and still non-zero at the edges, so the stretch, an
  offset and a multiple of the spring variant's surge shape are fitted jointly over the whole
  window (both variants share one HGRAPHER surge shape).

The surge part is the hydrograph minus the fitted tide. For each part the same measures as the
gauge events are taken (`shapes_events.py`, now with surge residuals): tide at the peak relative
to the station's MHWS, the surge share of the peak, the surge duration above half its maximum
(surge_dur50), and the next-day high water nb24 split into its tide and surge contributions.
Gauge and GTSM events use their own tide (level - surge) and surge residual.

Writes <out-dir>/analysis/{coast_hg_decomposition.parquet, coast_hg_parts.npz,
coast_hg_vs_events.csv}.

Usage:
    python boundary_test/coast_hg_decompose.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes_events import COAST_HG, WINDOW_HR, contiguous_run  # noqa: E402

EDGE_HR = 48
NB24_HR = (21, 28)
RANGE_BINS = (0, 1, 2, 4, 20)
VARIANTS = {"spring": "hydrograph_spring_tide_signal", "average": "hydrograph_average_tide_signal"}


def hourly_window(w10: np.ndarray) -> np.ndarray:
    """Hourly samples through the 10-min peak, on offsets -WINDOW_HR..WINDOW_HR."""
    k10 = int(np.nanargmax(w10))
    w = w10[k10 % 6::6]
    src = np.arange(2 * WINDOW_HR + 1) + k10 // 6 - WINDOW_HR
    return np.where((src >= 0) & (src < len(w)), w[np.clip(src, 0, len(w) - 1)], np.nan)


def _corr(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.correlate(x, kernel, mode="valid")


def fit_edges(tide: np.ndarray, w: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
    """Start index, offset and RMSE of the tide stretch matching w on mask (least squares)."""
    m = mask.astype(float)
    wm = np.where(mask, w, 0.0)
    n, w1, w2 = m.sum(), wm.sum(), (wm ** 2).sum()
    s1, s2, s3 = _corr(tide, m), _corr(tide ** 2, m), _corr(tide, wm)
    sse = w2 - 2 * s3 + s2 - (w1 - s1) ** 2 / n
    j = int(np.argmin(sse))
    return j, float((w1 - s1[j]) / n), float(np.sqrt(max(sse[j], 0.0) / n))


def fit_joint(tide: np.ndarray, w: np.ndarray, shape: np.ndarray, mask: np.ndarray) -> tuple[int, float, float, float]:
    """Start index, offset, surge multiple and RMSE of w = stretch + offset + k * shape on mask."""
    m = mask.astype(float)
    wm, sm = np.where(mask, w, 0.0), np.where(mask, shape, 0.0)
    n = m.sum()
    sx, sxx, swx, ssx = _corr(tide, m), _corr(tide ** 2, m), _corr(tide, wm), _corr(tide, sm)
    rr = (wm ** 2).sum() - 2 * swx + sxx
    p1, p2 = wm.sum() - sx, (sm * wm).sum() - ssx
    a11, a12, a22 = n, sm.sum(), (sm ** 2).sum()
    det = a11 * a22 - a12 ** 2
    c = (a22 * p1 - a12 * p2) / det
    k = (a11 * p2 - a12 * p1) / det
    sse = rr - (c * p1 + k * p2)
    j = int(np.argmin(sse))
    return j, float(c[j]), float(k[j]), float(np.sqrt(max(sse[j], 0.0) / n))


def part_metrics(level: np.ndarray, surge: np.ndarray, mhws: float) -> dict:
    """Tide / surge measures of one hourly window with the peak at WINDOW_HR."""
    c = WINDOW_HR
    peak = level[c]
    tide = level - surge
    s = np.where(np.isfinite(surge), surge, -np.inf)
    k = int(np.argmax(s))
    row = {"peak_m": float(peak), "tide_at_peak_m": float(tide[c]), "surge_at_peak_m": float(surge[c]),
           "surge_share": float(surge[c] / peak), "tide_peak_over_mhws": float(tide[c] / mhws),
           "surge_max_m": float(s[k]), "surge_dur50": contiguous_run(s > 0.5 * s[k], k)[0] if s[k] > 0 else 0.0}
    nb_level, nb_tide, nb_surge = [], [], []
    for sign in (-1, 1):
        lo, hi = sorted((c + sign * NB24_HR[0], c + sign * NB24_HR[1]))
        seg = level[lo:hi + 1]
        if np.isfinite(seg).any():
            i = lo + int(np.nanargmax(seg))
            nb_level.append(level[i] / peak)
            nb_tide.append(tide[i] / peak)
            nb_surge.append(surge[i] / peak)
    row.update({"nb24": float(np.mean(nb_level)) if nb_level else np.nan,
                "nb24_tide": float(np.mean(nb_tide)) if nb_tide else np.nan,
                "nb24_surge": float(np.mean(nb_surge)) if nb_surge else np.nan})
    return row


def decompose(out: Path) -> tuple[pd.DataFrame, dict]:
    pairs = pd.read_parquet(out / "pairs.parquet").reset_index(drop=True)
    hc = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    pos = {f: i for i, f in enumerate(hc["file_name"])}
    gtsm = hc["gtsm_harm"]
    b_re = np.load(out / "hourly_basis_re.npy", mmap_mode="r")
    b_im = np.load(out / "hourly_basis_im.npy", mmap_mode="r")
    tl = pd.read_parquet(out / "tide_levels.parquet")
    mhws = tl[(tl.variant == "full") & (tl.source == "gtsm_harm")].set_index("file_name").mhws_m
    t = np.arange(-WINDOW_HR, WINDOW_HR + 1)
    edge = np.abs(t) >= EDGE_HR
    rows, parts = [], {f"{v}_{p}": [] for v in VARIANTS for p in ("tide", "surge")}
    keep = []
    with netCDF4.Dataset(COAST_HG) as ds:
        hg = {v: np.ma.filled(ds[var][:], np.nan)[pairs.coast_hg_index.to_numpy()] for v, var in VARIANTS.items()}
    for i, f in enumerate(pairs.file_name):
        h = gtsm[pos[f]] if f in pos else None
        if h is None or not np.isfinite(h).all() or f not in mhws.index:
            continue
        tide = np.asarray(b_re) @ h.real + np.asarray(b_im) @ h.imag
        w_s, w_a = hourly_window(hg["spring"][i]), hourly_window(hg["average"][i])
        if not (np.isfinite(w_s).all() and np.isfinite(w_a).all()):
            continue
        j, c, rmse_s = fit_edges(tide, w_s, edge)
        tide_s = tide[j:j + len(t)] + c
        surge_s = w_s - tide_s
        shape = surge_s / max(surge_s.max(), 1e-6)
        ja, ca, ka, rmse_a = fit_joint(tide, w_a, shape, np.ones(len(t), bool))
        tide_a = tide[ja:ja + len(t)] + ca
        surge_a = w_a - tide_a
        for v, lev, sur, rmse in (("spring", w_s, surge_s, rmse_s), ("average", w_a, surge_a, rmse_a)):
            rows.append({"file_name": f, "variant": v, "fit_rmse_m": rmse, **part_metrics(lev, sur, mhws[f])})
        for name, arr in (("spring_tide", tide_s), ("spring_surge", surge_s), ("average_tide", tide_a),
                          ("average_surge", surge_a)):
            parts[name].append(arr.astype(np.float32))
        keep.append(f)
    return pd.DataFrame(rows), {k: np.vstack(v) for k, v in parts.items()} | {"file_name": np.array(keep)}


def event_parts(out: Path) -> pd.DataFrame:
    ev = pd.read_parquet(out / "events.parquet").reset_index(drop=True)
    prof = np.load(out / "event_profiles.npz")
    tl = pd.read_parquet(out / "tide_levels.parquet")
    tl = tl[tl.variant == "full"]
    mhws = dict(zip(zip(tl.file_name, tl.source), tl.mhws_m))
    files, sources = ev.file_name.to_numpy(), ev.source.to_numpy()
    peaks, rps = ev.peak_m.to_numpy(), ev.rp_empirical_yr.to_numpy()
    profiles, surges = prof["profiles"], prof["surge_m"]
    rows = []
    for i in np.flatnonzero(np.isin(sources, ["gesla", "gtsm"])):
        mh = mhws.get((files[i], f"{sources[i]}_harm"), np.nan)
        rows.append({"file_name": files[i], "source": sources[i], "rp_empirical_yr": rps[i],
                     **part_metrics(profiles[i] * peaks[i], surges[i].astype(float), mh)})
    return pd.DataFrame(rows)


def comparison(hg: pd.DataFrame, ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    rng = sites.gesla_harm__spring_range_m
    cols = ["tide_peak_over_mhws", "surge_share", "surge_dur50", "nb24", "nb24_tide", "nb24_surge"]
    frames = [hg.assign(source="coast_hg_" + hg.variant)]
    frames.append(ev[ev.rp_empirical_yr >= 10].assign(source=lambda d: d.source + "_rp10plus"))
    t = pd.concat(frames, ignore_index=True)
    t = t[t.file_name.isin(sites.index[sites.quality])]
    t["range_class"] = pd.cut(rng.reindex(t.file_name).to_numpy(), RANGE_BINS)
    site_med = t.groupby(["source", "range_class", "file_name"], observed=True)[cols].median()
    return site_med.groupby(["source", "range_class"], observed=True).median().assign(
        n_sites=site_med.groupby(["source", "range_class"], observed=True).size())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    hg, parts = decompose(out)
    hg.to_parquet(dest / "coast_hg_decomposition.parquet")
    np.savez(dest / "coast_hg_parts.npz", **parts)
    print(hg.groupby("variant")[["fit_rmse_m", "tide_peak_over_mhws", "surge_share", "surge_dur50"]].describe(
        percentiles=[0.1, 0.5, 0.9]).T.round(3).to_string())
    sites = pd.read_csv(dest / "levels_sites.csv", index_col=0)
    table = comparison(hg, event_parts(out), sites)
    table.round(3).to_csv(dest / "coast_hg_vs_events.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
