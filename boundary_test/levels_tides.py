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
import pyTMD.io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from fes_tide_stats import FES_MODEL, STEP_MIN, ensure_local_fes  # noqa: E402
from tide_prediction import (  # noqa: E402
    cached_basis, fit_constants, hourly_times, levels, on_hourly_grid)
from tide_prediction import grouped_constants as fes_constants  # noqa: E402

FES_CACHE = REPO / ".cleanup_baseline" / "fes2022_cache"
SEASONAL = ("sa", "ssa")
SITES_PER_TASK = 20


def hourly_basis(model, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """(B_re, B_im) at every hour HOURLY_START..HOURLY_END, cached in `cache`."""
    paths = cached_basis(model, hourly_times(), cache / "hourly_basis_re.npy", cache / "hourly_basis_im.npy")
    return tuple(np.load(p, mmap_mode="r") for p in paths)


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
