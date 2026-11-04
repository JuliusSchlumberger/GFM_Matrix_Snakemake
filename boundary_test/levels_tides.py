"""Step 1b: tidal levels at the GESLA sites from GESLA, GTSM and FES2022 on one basis.

Sources per site (`source` column):
  gesla_harm   harmonic fit to the gauge record
  gtsm_harm    harmonic fit to the GTSM tide at the paired station
  gtsm_raw     the GTSM tide series itself (10 min, 1996-2014), no fit
  fes_gauge    FES2022 atlas constants at the gauge
  fes_station  FES2022 atlas constants at the paired GTSM station
Fits use the 34 FES2022 constituents with pyTMD's FES nodal corrections and inferred minor
constituents, on hourly samples in FIT_YEARS (else every year from 1979), after removing each
calendar year's mean (local MSL of that year, which also removes the GTSM tide's MSL trend and
GESLA's datum). Every source except gtsm_raw is then predicted over the same 18.61-y epoch at
10 min (`fes_tide_stats.prediction_basis`), and all go through `tide_statistics`; so method,
epoch and sampling cannot explain a difference. Variant `no_seasonal` drops Sa and Ssa, which
FES2022 takes from altimetry and the barotropic GTSM tide does not have.
Levels are relative to local MSL: MHWS, MHW, MLW, MLWS (low waters: the same statistics on -tide),
HAT, LAT, and the spring range MHWS - MLWS.

Writes <out-dir>/tide_levels.parquet and <out-dir>/harmonic_constants.npz.

Usage:
    python boundary_test/levels_tides.py --out-dir .cleanup_baseline/level_sources [--workers 3]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.arguments
import pyTMD.io
import pyTMD.predict
import timescale.time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from fes_tide_stats import FES_MODEL, STEP_MIN, ensure_local_fes, extract_constants  # noqa: E402
from tide_statistics import high_water_statistics  # noqa: E402

FES_CACHE = REPO / ".cleanup_baseline" / "fes2022_cache"
HOURLY_START = np.datetime64("1979-01-01T00:00")
HOURLY_END = np.datetime64("2025-01-01T00:00")
FIT_YEARS = (1996, 2014)  # the GTSM tide period
MIN_FIT_YEARS = 3.0  # >= 1 y separates S2/K2/T2; 3 y keeps the long-period terms stable
SEASONAL = ("sa", "ssa")
GROUP_DEG = 5.0
SITES_PER_TASK = 20


def hourly_basis(model, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """(B_re, B_im) at every hour HOURLY_START..HOURLY_END, built as `prediction_basis`."""
    if not (cache / "hourly_basis_re.npy").exists():
        constituents = list(model.constituents)
        hours = np.arange(HOURLY_START, HOURLY_END, np.timedelta64(1, "h"))
        ts = timescale.time.Timescale().from_datetime(hours)
        pu, pf, G = pyTMD.arguments.arguments(ts.tide + pyTMD.predict._mjd_tide, constituents,
                                              deltat=ts.tt_ut1, corrections=model.corrections)
        th = G * np.pi / 180.0 + pu
        b_re, b_im = pf * np.cos(th), -pf * np.sin(th)
        for k in range(len(constituents)):
            for unit, dst in ((1.0, b_re), (1j, b_im)):
                z = np.ma.zeros((1, len(constituents)), dtype=np.complex128)
                z[0, k] = unit
                dst[:, k] += np.ma.filled(pyTMD.predict.infer_minor(ts.tide, z, constituents, deltat=ts.tt_ut1,
                                                                    corrections=model.corrections,
                                                                    minor=model.minor), 0.0)
        np.save(cache / "hourly_basis_im.npy", b_im)
        np.save(cache / "hourly_basis_re.npy", b_re)
    return (np.load(cache / "hourly_basis_re.npy", mmap_mode="r"), np.load(cache / "hourly_basis_im.npy", mmap_mode="r"))


def on_hourly_grid(t: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hour index and value of the samples that fall on a full hour inside the basis period."""
    t = t.astype("datetime64[s]")
    full = (t.astype(np.int64) % 3600 == 0) & (t >= HOURLY_START) & (t < HOURLY_END) & np.isfinite(z)
    idx = ((t[full] - HOURLY_START.astype("datetime64[s]")).astype(np.int64) // 3600)
    return idx, z[full].astype(np.float64)


def remove_annual_means(idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    year = (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]")
    return z - pd.Series(z).groupby(year).transform("mean").to_numpy()


def fit_constants(idx: np.ndarray, z: np.ndarray, b_re, b_im) -> tuple[np.ndarray, dict]:
    """Least-squares complex constants (constituent,) for an hourly sample set, and fit info."""
    years = (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    window = (years >= FIT_YEARS[0]) & (years <= FIT_YEARS[1])
    use = window if window.sum() >= MIN_FIT_YEARS * 8766 else np.ones_like(window)
    idx, z = idx[use], z[use]
    nan_info = {"fit_years": len(idx) / 8766.0, "fit_in_window": bool(use is window)}
    if len(idx) < MIN_FIT_YEARS * 8766:
        return np.full(b_re.shape[1], np.nan + 0j), nan_info
    z = remove_annual_means(idx, z)
    order = np.argsort(idx)
    idx, z = idx[order], z[order]
    a = np.hstack([np.asarray(b_re[idx]), np.asarray(b_im[idx]), np.ones((len(idx), 1))])
    x, *_ = np.linalg.lstsq(a, z, rcond=None)
    nc = b_re.shape[1]
    resid = z - a @ x
    return x[:nc] + 1j * x[nc:2 * nc], {**nan_info, "fit_rms_m": float(resid.std()),
                                        "fit_explained": float(1.0 - resid.var() / z.var())}


def levels(tide: np.ndarray, dt_s: float) -> dict:
    hi = high_water_statistics(tide, dt_s)
    lo = high_water_statistics(-tide, dt_s)
    return {"mhws_m": hi["mhws_m"], "mhw_m": hi["mhw_m"], "hat_m": hi["hat_m"], "mlws_m": -lo["mhws_m"],
            "mlw_m": -lo["mhw_m"], "lat_m": -lo["hat_m"], "spring_range_m": hi["mhws_m"] + lo["mhws_m"],
            "n_spring_high_waters": hi["n_spring_high_waters"]}


def _predict_task(args) -> list[dict]:
    basis_dir, keys, hc, seasonal_cols = args
    b_re = np.load(Path(basis_dir) / "basis_re.npy", mmap_mode="r")
    b_im = np.load(Path(basis_dir) / "basis_im.npy", mmap_mode="r")
    rows = []
    for variant in ("full", "no_seasonal"):
        h = hc.copy()
        if variant == "no_seasonal":
            h[:, seasonal_cols] = 0.0
        ok = np.isfinite(h).all(axis=1)
        tide = np.asarray(b_re) @ np.nan_to_num(h.real).T + np.asarray(b_im) @ np.nan_to_num(h.imag).T
        for j, key in enumerate(keys):
            rows.append({**key, "variant": variant, **(levels(tide[:, j], STEP_MIN * 60.0) if ok[j] else {})})
    return rows


def _gtsm_raw_task(args) -> list[dict]:
    out, keys, positions = args
    _, time, series = load_series(Path(out), "tide")
    year = time.year.to_numpy()
    rows = []
    for key, j in zip(keys, positions):
        x = station_series(series, j)
        ok = np.isfinite(x)
        if ok.mean() < 0.9:
            rows.append({**key, "variant": "full"})
            continue
        x = np.where(ok, x - pd.Series(x).groupby(year).transform("mean").to_numpy(), 0.0)
        rows.append({**key, "variant": "full", **levels(x, STEP_MIN * 60.0),
                     "gtsm_tide_mean_m": float(np.nanmean(station_series(series, j))),
                     "gtsm_tide_trend_mm_yr": float(np.polyfit(*_annual(station_series(series, j), year), 1)[0] * 1000)})
    return rows


def _annual(x: np.ndarray, year: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = pd.Series(x).groupby(year).mean().dropna()
    return m.index.to_numpy(dtype=float), m.to_numpy()


def fes_constants(model, lonlat: np.ndarray) -> np.ndarray:
    """FES2022 constants (point, constituent) at arbitrary points, cropped per GROUP_DEG cell."""
    cell = np.floor(lonlat / GROUP_DEG).astype(int)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    hc = extract_constants(model, [lonlat[g] for g in groups])
    out = np.empty_like(hc)
    out[np.concatenate(groups)] = hc
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out_dir)
    pairs = pd.read_parquet(out / "pairs.parquet")
    model = pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL)
    constituents = list(model.constituents)
    seasonal_cols = [constituents.index(c) for c in SEASONAL]
    b_re, b_im = hourly_basis(model, out)

    print("fitting GESLA and GTSM tide", flush=True)
    gtsm_st, gtsm_time, gtsm_series = load_series(out, "tide")
    gtsm_pos = dict(zip(gtsm_st.gtsm_index, range(len(gtsm_st))))
    hourly = np.asarray(gtsm_time.minute == 0)
    gtsm_idx, _ = on_hourly_grid(gtsm_time.to_numpy()[hourly], np.zeros(hourly.sum()))
    hc_gesla, hc_gtsm, fit_rows = [], [], []
    for k, site in enumerate(pairs.itertuples()):
        data = np.load(out / "gesla" / f"{site.file_name}.npz")
        h, info_g = fit_constants(*on_hourly_grid(data["t"], data["z"]), b_re, b_im)
        hc_gesla.append(h)
        x = station_series(gtsm_series, gtsm_pos[site.gtsm_index])[hourly]
        ok = np.isfinite(x)
        h, info_m = fit_constants(gtsm_idx[ok], x[ok], b_re, b_im)
        hc_gtsm.append(h)
        fit_rows += [{"file_name": site.file_name, "source": "gesla_harm", **info_g},
                     {"file_name": site.file_name, "source": "gtsm_harm", **info_m}]
        if k % 100 == 0:
            print(f"  {k + 1}/{len(pairs)} sites", flush=True)
    print("extracting FES2022 constants", flush=True)
    hc_fes_gauge = fes_constants(model, pairs[["lon", "lat"]].to_numpy(dtype=float))
    hc_fes_station = fes_constants(model, pairs[["gtsm_lon", "gtsm_lat"]].to_numpy(dtype=float))
    hc = {"gesla_harm": np.vstack(hc_gesla), "gtsm_harm": np.vstack(hc_gtsm), "fes_gauge": hc_fes_gauge,
          "fes_station": hc_fes_station}
    np.savez(out / "harmonic_constants.npz", constituents=np.array(constituents), file_name=pairs.file_name.to_numpy(),
             **hc)

    print("predicting the nodal cycle and computing levels", flush=True)
    tasks = []
    for source, h in hc.items():
        for i in range(0, len(pairs), SITES_PER_TASK):
            keys = [{"file_name": f, "source": source} for f in pairs.file_name.iloc[i:i + SITES_PER_TASK]]
            tasks.append((str(FES_CACHE), keys, h[i:i + SITES_PER_TASK], seasonal_cols))
    raw_tasks = []
    for i in range(0, len(pairs), SITES_PER_TASK):
        chunk = pairs.iloc[i:i + SITES_PER_TASK]
        raw_tasks.append((str(out), [{"file_name": f, "source": "gtsm_raw"} for f in chunk.file_name],
                          [gtsm_pos[g] for g in chunk.gtsm_index]))
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(list(pool.map(_predict_task, tasks)) + list(pool.map(_gtsm_raw_task, raw_tasks))):
            rows += r
    table = pd.DataFrame(rows).merge(pd.DataFrame(fit_rows), on=["file_name", "source"], how="left")
    table.to_parquet(out / "tide_levels.parquet")
    print(table.groupby(["source", "variant"]).mhws_m.describe().round(3).to_string())


if __name__ == "__main__":
    main()
