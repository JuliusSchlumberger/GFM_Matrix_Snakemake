"""FES tide prediction as a matrix product, harmonic fits and the FES2022 / FES2014 version rule.

Prediction is linear in the harmonic constants, so a (time, constituent) basis with pyTMD's
nodal corrections and inferred minor constituents is built once (`tide_basis`, checked against
`pyTMD.compute.tide_elevations` to 1e-8 m) and tide = B_re @ hc.real + B_im @ hc.imag.
"""

import lzma
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.arguments
import pyTMD.io
import pyTMD.io.FES
import pyTMD.predict
import timescale.time
from sklearn.neighbors import BallTree

from tide_statistics import high_water_statistics

FES2022_MODEL = "FES2022_extrapolated"
FES2014_MODEL = "FES2014_extrapolated"
FES2022_SUBDIR = Path("fes2022b") / "ocean_tide_extrapolated"
SEASONAL = ("sa", "ssa")  # FES2022 takes them from altimetry; they belong to the surge here
CUTOFF_KM = 20.0  # extrapolation to the nearest valid ocean node
CROP_BUFFER_DEG = 0.5  # > CUTOFF_KM, so extrapolation sees every valid cell within the cutoff
GROUP_DEG = 5.0
NODAL_CYCLE_YR = 18.61
NODAL_START = np.datetime64("1996-01-01T00:00")
STEP_MIN = 10
HOURLY_START = np.datetime64("1979-01-01T00:00")
HOURLY_END = np.datetime64("2025-01-01T00:00")
FIT_YEARS = (1996, 2014)  # the GTSM tide period
MIN_FIT_YEARS = 3.0  # >= 1 y separates S2/K2/T2; 3 y keeps the long-period terms stable
# Version rule (boundary_test/INPUT_ANALYSIS.md test 6): FES2014 where FES2022's MHWS exceeds it
# by both; grown to neighbours on the same artefact by `consistent_version_flag`.
VERSION_RATIO = 1.5
VERSION_DIFF_M = 1.25
VERSION_GROW_KM = 25.0
VERSION_GROW_DIFF_M = 0.25
VERSION_VETO_KM = 50.0
FES2022_OK_M = 0.25  # |FES2022 - gauge| MHWS at a gauge that vetoes the growth
EARTH_RADIUS_KM = 6371.0


def ensure_local_fes(source_dir: Path, cache: Path) -> Path:
    """Decompress the FES2022 .nc.xz files of `source_dir` into cache/FES2022_SUBDIR once."""
    dest = cache / FES2022_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(Path(source_dir).iterdir()):
        name = src.name[:-3] if src.name.endswith(".xz") else src.name
        target = dest / name
        if target.exists() or not name.endswith(".nc"):
            continue
        tmp = target.with_suffix(".nc.part")
        opener = lzma.open if src.name.endswith(".xz") else open
        with opener(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 << 20)
        tmp.replace(target)
    return cache


def fes2022_model(cache: Path):
    return pyTMD.io.model(str(cache)).elevation(FES2022_MODEL)


def fes2014_model(files_dir: Path):
    # pyTMD expects its own folder layout; the inputs keep the provider's, so the paths are remapped.
    model = pyTMD.io.model(verify=False).elevation(FES2014_MODEL)
    model.model_file = [Path(files_dir) / Path(f).name for f in model.model_file]
    return model


def seasonal_columns(model) -> list[int]:
    cons = list(model.constituents)
    return [cons.index(c) for c in SEASONAL if c in cons]


def extract_constants(model, groups: list[np.ndarray], method: str = "spline") -> np.ndarray:
    """Complex harmonic constants (point, constituent) for the (lon, lat) point groups, stacked.

    As `model.extract_constants(crop=True, extrapolate=True)` per group, with each constituent
    grid read once: `pyTMD.io.FES.read_netcdf_file` is swapped for a one-file cache meanwhile.
    Points with no valid node within CUTOFF_KM are NaN.
    """
    fes, original = pyTMD.io.FES, pyTMD.io.FES.read_netcdf_file
    cache: dict = {}
    window: dict = {}

    def cached(path, **kwargs):
        if str(path) not in cache:
            cache.clear()
            cache[str(path)] = original(path, **kwargs)
        # pyTMD crops and extrapolates the returned grid in place, so each call gets its own copy;
        # without it every later group sees the damaged grid and comes back NaN. Only the window
        # around the current group is copied (a global 1/30 deg grid is ~1 GB).
        hc, lon, lat = cache[str(path)]
        if window:
            margin = CROP_BUFFER_DEG + 4 * abs(float(lon[1] - lon[0]))
            ii = np.flatnonzero((lon >= window["lon0"] - margin) & (lon <= window["lon1"] + margin))
            jj = np.flatnonzero((lat >= window["lat0"] - margin) & (lat <= window["lat1"] + margin))
            return (hc[jj[0]:jj[-1] + 1, ii[0]:ii[-1] + 1].copy(), lon[ii[0]:ii[-1] + 1].copy(),
                    lat[jj[0]:jj[-1] + 1].copy())
        return hc.copy(), lon.copy(), lat.copy()

    # 0-360 longitudes (the FES grid's) avoid shifting the global grid per crop, except for
    # groups that straddle 0 deg.
    groups = [np.column_stack([np.where(g[:, 0] < 0, g[:, 0] + 360, g[:, 0]), g[:, 1]])
              if np.ptp(np.where(g[:, 0] < 0, g[:, 0] + 360, g[:, 0])) < 180 else g for g in groups]
    hc = [np.full((len(g), len(model.model_file)), np.nan + 1j * np.nan) for g in groups]
    fes.read_netcdf_file = cached
    try:
        for k, model_file in enumerate(model.model_file):
            for g, out in zip(groups, hc):
                window.clear()
                if (g[:, 0] >= 0).all():  # groups straddling 0 deg keep the global grid
                    window.update(lon0=g[:, 0].min(), lon1=g[:, 0].max(), lat0=g[:, 1].min(), lat1=g[:, 1].max())
                amp, ph = fes.extract_constants(g[:, 0], g[:, 1], [model_file], version=model.version,
                                                compressed=model.compressed, type=model.type, scale=model.scale,
                                                crop=True, buffer=CROP_BUFFER_DEG, method=method,
                                                extrapolate=True, cutoff=CUTOFF_KM)
                out[:, k] = np.ma.filled(amp[:, 0] * np.exp(-1j * ph[:, 0] * np.pi / 180.0), np.nan)
            print(f"  constituent {k + 1}/{len(model.model_file)}", flush=True)
    finally:
        fes.read_netcdf_file = original
    return np.vstack(hc)


def grouped_constants(model, lonlat: np.ndarray, method: str = "spline") -> np.ndarray:
    """Constants (point, constituent) at arbitrary (lon, lat) points, cropped per GROUP_DEG cell."""
    cell = np.floor(lonlat / GROUP_DEG).astype(int)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    hc = extract_constants(model, [lonlat[g] for g in groups], method)
    out = np.empty_like(hc)
    out[np.concatenate(groups)] = hc
    return out


def fill_masked(hc: np.ndarray, lonlat: np.ndarray) -> np.ndarray:
    """Give points without constants (atlas land cells) those of the nearest point with them."""
    ok = np.isfinite(hc).all(axis=1)
    if ok.all() or not ok.any():
        return hc
    tree = BallTree(np.radians(lonlat[ok][:, ::-1]), metric="haversine")
    _, i = tree.query(np.radians(lonlat[~ok][:, ::-1]), k=1)
    hc = hc.copy()
    hc[~ok] = hc[ok][i[:, 0]]
    return hc


def nodal_times() -> np.ndarray:
    """10-min steps over one nodal cycle from NODAL_START."""
    steps = np.arange(int(NODAL_CYCLE_YR * 365.25 * 24 * 60 / STEP_MIN))
    return NODAL_START + (steps * STEP_MIN).astype("timedelta64[m]")


def hourly_times() -> np.ndarray:
    return np.arange(HOURLY_START, HOURLY_END, np.timedelta64(1, "h"))


def tide_basis(model, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (B_re, B_im), (time, constituent): tide = B_re @ hc.real + B_im @ hc.imag.

    Major part as `pyTMD.predict.time_series`; minor part from `pyTMD.predict.infer_minor`
    evaluated on unit constants, which is exact because the inference is linear in them.
    """
    constituents = list(model.constituents)
    ts = timescale.time.Timescale().from_datetime(times)
    pu, pf, G = pyTMD.arguments.arguments(ts.tide + pyTMD.predict._mjd_tide, constituents,
                                          deltat=ts.tt_ut1, corrections=model.corrections)
    th = G * np.pi / 180.0 + pu
    b_re, b_im = pf * np.cos(th), -pf * np.sin(th)
    del pu, pf, G, th
    nc = len(constituents)
    for k in range(nc):
        for unit, dst in ((1.0, b_re), (1j, b_im)):
            z = np.ma.zeros((1, nc), dtype=np.complex128)
            z[0, k] = unit
            dst[:, k] += np.ma.filled(pyTMD.predict.infer_minor(ts.tide, z, constituents, deltat=ts.tt_ut1,
                                                                corrections=model.corrections, minor=model.minor), 0.0)
    return b_re, b_im


def cached_basis(model, times: np.ndarray, re_path: Path, im_path: Path, dtype=np.float64) -> tuple[Path, Path]:
    """Build `tide_basis` once into two .npy files (read them with mmap_mode="r")."""
    if not (re_path.exists() and im_path.exists()):
        re_path.parent.mkdir(parents=True, exist_ok=True)
        b_re, b_im = tide_basis(model, times)
        np.save(im_path, b_im.astype(dtype))
        np.save(re_path, b_re.astype(dtype))
    return re_path, im_path


def on_hourly_grid(t: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hour index and value of the samples that fall on a full hour inside the hourly basis period."""
    t = t.astype("datetime64[s]")
    full = (t.astype(np.int64) % 3600 == 0) & (t >= HOURLY_START) & (t < HOURLY_END) & np.isfinite(z)
    idx = ((t[full] - HOURLY_START.astype("datetime64[s]")).astype(np.int64) // 3600)
    return idx, z[full].astype(np.float64)


def hour_years(idx: np.ndarray) -> np.ndarray:
    return (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970


def remove_annual_means(idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    year = (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]")
    return z - pd.Series(z).groupby(year).transform("mean").to_numpy()


def fit_constants(idx: np.ndarray, z: np.ndarray, b_re, b_im) -> tuple[np.ndarray, dict]:
    """Least-squares complex constants (constituent,) for an hourly sample set, and fit info.

    Fits FIT_YEARS when they hold MIN_FIT_YEARS of samples, else every year; each calendar
    year's mean is removed first (local MSL of that year).
    """
    years = hour_years(idx)
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
    """Tidal levels (m, relative to the series' zero) of a nodal-cycle prediction."""
    hi = high_water_statistics(tide, dt_s)
    lo = high_water_statistics(-tide, dt_s)
    return {"mhws_m": hi["mhws_m"], "mhw_m": hi["mhw_m"], "hat_m": hi["hat_m"], "mlws_m": -lo["mhws_m"],
            "mlw_m": -lo["mhw_m"], "lat_m": -lo["hat_m"], "spring_range_m": hi["mhws_m"] + lo["mhws_m"],
            "n_spring_high_waters": hi["n_spring_high_waters"]}


def predict(b_re, b_im, hc: np.ndarray) -> np.ndarray:
    """Tide (time,) or (time, point) of constants (constituent,) or (point, constituent)."""
    hc = np.asarray(hc)
    return np.asarray(b_re) @ hc.real.T + np.asarray(b_im) @ hc.imag.T


def version_flag(fes2022_m, fes2014_m, ratio: float = VERSION_RATIO, diff_m: float = VERSION_DIFF_M):
    """Flag where FES2022 is replaced by FES2014; NaN in either value never flags."""
    return (fes2022_m > ratio * fes2014_m) & (fes2022_m - fes2014_m > diff_m)


def consistent_version_flag(mh22: np.ndarray, mh14: np.ndarray, lonlat: np.ndarray,
                            fes2022_ok_lonlat: np.ndarray | None = None) -> np.ndarray:
    """Version rule, extended to neighbours of flagged stations that fail only its minimum difference.

    A station within VERSION_GROW_KM of a flagged one is flagged too when FES2022 exceeds FES2014
    by the rule's ratio and by VERSION_GROW_DIFF_M; repeated until no station is added. Without
    it, neighbours 3 km apart on the same FES artefact take different atlases (~1 m of MHWS).
    Stations within VERSION_VETO_KM of a gauge in `fes2022_ok_lonlat` (FES2022 matches it) are
    never added.
    """
    flag = np.asarray(version_flag(mh22, mh14), dtype=bool)
    weaker = np.asarray(version_flag(mh22, mh14, diff_m=VERSION_GROW_DIFF_M), dtype=bool)
    if fes2022_ok_lonlat is not None and len(fes2022_ok_lonlat):
        d, _ = BallTree(np.radians(fes2022_ok_lonlat[:, ::-1]), metric="haversine").query(np.radians(lonlat[:, ::-1]))
        weaker &= d[:, 0] * EARTH_RADIUS_KM > VERSION_VETO_KM
    tree = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine")
    while True:
        cand = np.flatnonzero(weaker & ~flag)
        if not len(cand):
            return flag
        near = tree.query_radius(np.radians(lonlat[cand, ::-1]), r=VERSION_GROW_KM / EARTH_RADIUS_KM)
        add = cand[[flag[n].any() for n in near]]
        if not len(add):
            return flag
        flag[add] = True
