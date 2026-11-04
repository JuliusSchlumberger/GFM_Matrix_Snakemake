"""Extract GTSM reanalysis series (surge residual hourly, tide 10-min) at the test-tile stations.

Monthly files hold all ~43,000 stations in chunks of 8,000-22,000 stations, so a subset needs
every file read in full; contiguous station runs are read as one hyperslab (as GCFM_UUdev
workflow/src/gtsm.py). Stations: the boundary stations of the tiles in --tiles-csv, or the GTSM
indices in --stations-csv (column gtsm_index).
Series are stored station-major as int16 mm (`load_series`), so one station reads contiguously:
10,000 stations x 19 y of 10-min tide is ~22 GB, which does not fit in memory as float32.

Writes <out-dir>/gtsm_<variable>/{series.npy, time.npy, stations.parquet} and, with --tiles-csv,
<out-dir>/test_station_tiles.csv.

Usage:
    python boundary_test/extract_gtsm_series.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_representative.csv --variable surge
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

GTSM_DIR = Path("P:/11212688-004-global-floodmaps/modelling/inputs/GTSM_storm_tide_hourly")
BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
# Surge from 1979, the ERA5 period of COAST-RP and HGRAPHER-RP; tide: every downloaded year.
FOLDERS = {"surge": ("storm_surge_residual", "reanalysis_surge_hourly_*_v3.nc", "surge", 1979),
           "tide": ("tidal_elevation", "historical_tide_*_v1.nc", "tide", None)}
SLAB_GAP = 2000
MATCH_KM = 0.1
EARTH_RADIUS_KM = 6371.0
MISSING = np.iinfo(np.int16).min


def load_series(out: Path, variable: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Return the stations table, the time axis and the (station, time) int16 mm memmap."""
    d = out / f"gtsm_{variable}"
    return (pd.read_parquet(d / "stations.parquet"), pd.DatetimeIndex(np.load(d / "time.npy")),
            np.load(d / "series.npy", mmap_mode="r"))


def station_series(series: np.ndarray, j: int) -> np.ndarray:
    """Station j of a `load_series` memmap in m, NaN where missing."""
    raw = np.asarray(series[j])
    return np.where(raw == MISSING, np.nan, raw / 1000.0)


def _file_year(path: Path) -> int:
    return int(path.stem.split("_")[3])


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


def test_station_indices(tiles: pd.DataFrame, gtsm_lonlat: np.ndarray) -> pd.DataFrame:
    rows = []
    tree = BallTree(np.radians(gtsm_lonlat[:, ::-1]), metric="haversine")
    for tile_id in tiles.tile_id.astype(str):
        g = gpd.read_file(BASE / tile_id / "inputs" / "boundaries_RP100_SLR_0.gpkg")
        ll = np.column_stack([g.geometry.x, g.geometry.y])
        d, i = tree.query(np.radians(ll[:, ::-1]), k=1)
        for k in range(len(ll)):
            if d[k, 0] * EARTH_RADIUS_KM <= MATCH_KM:
                rows.append({"tile_id": tile_id, "gtsm_index": int(i[k, 0]),
                             "lon": gtsm_lonlat[i[k, 0], 0], "lat": gtsm_lonlat[i[k, 0], 1]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    stations_from = parser.add_mutually_exclusive_group(required=True)
    stations_from.add_argument("--tiles-csv")
    stations_from.add_argument("--stations-csv")
    parser.add_argument("--variable", choices=list(FOLDERS), default="surge")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    folder, pattern, var, first_year = FOLDERS[args.variable]
    files = [f for f in sorted((GTSM_DIR / folder).glob(pattern)) if first_year is None or _file_year(f) >= first_year]
    if not files:
        raise FileNotFoundError(f"no {pattern} under {GTSM_DIR / folder}")
    with netCDF4.Dataset(files[0]) as ds:
        lonlat = np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)
        ids = np.asarray(ds["stations"][:]).astype(int)
    lengths = []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            lengths.append(ds.dimensions["time"].size)
    out = Path(args.out_dir)
    stations = (test_station_indices(pd.read_csv(args.tiles_csv, dtype={"tile_id": str}), lonlat) if args.tiles_csv
                else pd.read_csv(args.stations_csv))
    unique_idx = np.unique(stations.gtsm_index.to_numpy())

    d = out / f"gtsm_{args.variable}"
    d.mkdir(exist_ok=True)
    series = np.lib.format.open_memmap(d / "series.npy.part", mode="w+", dtype=np.int16,
                                       shape=(len(unique_idx), sum(lengths)))
    times, pos = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
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
        (d / "series.npy.part").replace(d / "series.npy")
    else:
        final = np.lib.format.open_memmap(d / "series.npy", mode="w+", dtype=np.int16, shape=(len(unique_idx), pos))
        part = np.load(d / "series.npy.part", mmap_mode="r")
        for j in range(0, len(unique_idx), 500):
            final[j: j + 500] = part[j: j + 500, :pos]
        final.flush()
        del final, part
        (d / "series.npy.part").unlink()
    np.save(d / "time.npy", np.concatenate(times))
    pd.DataFrame({"gtsm_index": unique_idx, "gtsm_station": ids[unique_idx], "lon": lonlat[unique_idx, 0],
                  "lat": lonlat[unique_idx, 1]}).to_parquet(d / "stations.parquet")
    if args.tiles_csv:
        stations.to_csv(out / "test_station_tiles.csv", index=False)
    print(f"{len(unique_idx)} stations, {len(files)} files, {pos} steps")


if __name__ == "__main__":
    main()
