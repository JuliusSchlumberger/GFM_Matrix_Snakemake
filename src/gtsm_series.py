"""Extract GTSM-ERA5 series at chosen stations into a station-major int16 store, and read it back.

Monthly files hold all ~43,000 stations in chunks of 8,000-22,000 stations, so a subset needs
every file read in full; contiguous station runs are read as one hyperslab. The store is int16
mm, station-major, so one station reads contiguously (23,000 stations x 46 y hourly is ~19 GB).
Main entry: `extract_series`; `load_series` / `station_series` read it.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

# Surge from 1979, the ERA5 period of COAST-RP and HGRAPHER-RP; tide: every downloaded year.
FOLDERS = {"surge": ("storm_surge_residual", "reanalysis_surge_hourly_*_v3.nc", "surge", 1979),
           "tide": ("tidal_elevation", "historical_tide_*_v1.nc", "tide", None)}
SLAB_GAP = 2000
MISSING = np.iinfo(np.int16).min


def gtsm_files(gtsm_dir: Path, variable: str = "surge") -> list[Path]:
    folder, pattern, _, first_year = FOLDERS[variable]
    files = [f for f in sorted((Path(gtsm_dir) / folder).glob(pattern))
             if first_year is None or int(f.stem.split("_")[3]) >= first_year]
    if not files:
        raise FileNotFoundError(f"no {pattern} under {Path(gtsm_dir) / folder}")
    return files


def gtsm_stations(gtsm_dir: Path, variable: str = "surge") -> tuple[np.ndarray, np.ndarray]:
    """Return (lonlat (n, 2), station ids) of the GTSM output points; the same in every file."""
    with netCDF4.Dataset(gtsm_files(gtsm_dir, variable)[0]) as ds:
        lonlat = np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)
        ids = np.asarray(ds["stations"][:]).astype(int)
    return lonlat, ids


def load_series(store: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Return the stations table, the time axis and the (station, time) int16 mm memmap."""
    store = Path(store)
    return (pd.read_parquet(store / "stations.parquet"), pd.DatetimeIndex(np.load(store / "time.npy")),
            np.load(store / "series.npy", mmap_mode="r"))


def station_series(series: np.ndarray, j: int) -> np.ndarray:
    """Station j of a `load_series` memmap in m, NaN where missing."""
    raw = np.asarray(series[j])
    return np.where(raw == MISSING, np.nan, raw / 1000.0)


def _read_month(args) -> tuple[np.ndarray, np.ndarray]:
    path, var, station_idx = args
    order = np.argsort(station_idx)
    sorted_idx = station_idx[order]
    with netCDF4.Dataset(path) as ds:
        v = ds[var]
        time = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        out = np.empty((len(station_idx), len(time)), dtype=np.int16)
        start = 0
        while start < len(sorted_idx):
            stop = start + 1
            while stop < len(sorted_idx) and sorted_idx[stop] - sorted_idx[stop - 1] < SLAB_GAP:
                stop += 1
            lo = int(sorted_idx[start])
            slab = np.ma.filled(v[:, lo: int(sorted_idx[stop - 1]) + 1], np.nan)[:, sorted_idx[start:stop] - lo]
            out[order[start:stop]] = np.where(np.isfinite(slab), np.round(slab * 1000.0), MISSING).T
            start = stop
    return np.asarray(time, dtype="datetime64[s]"), out


def extract_series(gtsm_dir: Path, gtsm_indices: np.ndarray, store: Path, workers: int = 4,
                   variable: str = "surge") -> None:
    """Write the series of the GTSM stations `gtsm_indices` (unique, sorted) to `store`."""
    files = gtsm_files(gtsm_dir, variable)
    var = FOLDERS[variable][2]
    lonlat, ids = gtsm_stations(gtsm_dir, variable)
    lengths = []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            lengths.append(ds.dimensions["time"].size)
    unique_idx = np.unique(np.asarray(gtsm_indices))
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    series = np.lib.format.open_memmap(store / "series.npy.part", mode="w+", dtype=np.int16,
                                       shape=(len(unique_idx), sum(lengths)))
    times, pos = [], 0
    with ProcessPoolExecutor(workers) as pool:
        for k, (time, block) in enumerate(pool.map(_read_month, [(str(f), var, unique_idx) for f in files])):
            keep = time > times[-1][-1] if times else np.ones(len(time), dtype=bool)  # monthly files may overlap
            series[:, pos: pos + keep.sum()] = block[:, keep]
            times.append(time[keep])
            pos += int(keep.sum())
            if k % 24 == 0:
                print(f"  {k + 1}/{len(files)} files", flush=True)
    series.flush()
    del series
    if pos == sum(lengths):
        (store / "series.npy.part").replace(store / "series.npy")
    else:
        final = np.lib.format.open_memmap(store / "series.npy", mode="w+", dtype=np.int16,
                                          shape=(len(unique_idx), pos))
        part = np.load(store / "series.npy.part", mmap_mode="r")
        for j in range(0, len(unique_idx), 500):
            final[j: j + 500] = part[j: j + 500, :pos]
        final.flush()
        del final, part
        (store / "series.npy.part").unlink()
    np.save(store / "time.npy", np.concatenate(times))
    pd.DataFrame({"gtsm_index": unique_idx, "gtsm_station": ids[unique_idx], "lon": lonlat[unique_idx, 0],
                  "lat": lonlat[unique_idx, 1]}).to_parquet(store / "stations.parquet")
    print(f"{len(unique_idx)} stations, {len(files)} files, {pos} steps", flush=True)
