"""Extract hourly GTSM reanalysis series (surge residual, tide) at the test-tile stations.

Monthly files hold all ~43,000 stations, so a subset needs every file; contiguous station
runs are read as one hyperslab (as GCFM_UUdev workflow/src/gtsm.py). Stations: the boundary
stations of the tiles in --tiles-csv. Writes `<out-dir>/gtsm_<variable>_test_stations.nc`.

Usage:
    python boundary_test/extract_gtsm_series.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_provisional.csv --variable surge
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

GTSM_DIR = Path("P:/11212688-004-global-floodmaps/modelling/inputs/GTSM_storm_tide_hourly")
BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
FOLDERS = {"surge": ("storm_surge_residual", "reanalysis_surge_hourly_*_v3.nc", "surge"),
           "tide": ("tidal_elevation", "reanalysis_tide_hourly_*_v3.nc", "tide")}
SLAB_GAP = 2000
MATCH_KM = 0.1
EARTH_RADIUS_KM = 6371.0


def _read_month(args) -> tuple[np.ndarray, np.ndarray]:
    path, var, station_idx = args
    order = np.argsort(station_idx)
    sorted_idx = station_idx[order]
    with netCDF4.Dataset(path) as ds:
        v = ds[var]
        time = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        out = np.empty((len(time), len(station_idx)), dtype=np.float32)
        start = 0
        while start < len(sorted_idx):
            stop = start + 1
            while stop < len(sorted_idx) and sorted_idx[stop] - sorted_idx[stop - 1] < SLAB_GAP:
                stop += 1
            lo = int(sorted_idx[start])
            slab = np.ma.filled(v[:, lo: int(sorted_idx[stop - 1]) + 1], np.nan)
            out[:, order[start:stop]] = slab[:, sorted_idx[start:stop] - lo]
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
    parser.add_argument("--tiles-csv", required=True)
    parser.add_argument("--variable", choices=list(FOLDERS), default="surge")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    folder, pattern, var = FOLDERS[args.variable]
    files = sorted((GTSM_DIR / folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {GTSM_DIR / folder}")
    with netCDF4.Dataset(files[0]) as ds:
        lonlat = np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)
        ids = np.asarray(ds["stations"][:]).astype(int)
    stations = test_station_indices(pd.read_csv(args.tiles_csv, dtype={"tile_id": str}), lonlat)
    unique_idx = np.unique(stations.gtsm_index.to_numpy())
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(_read_month, [(str(f), var, unique_idx) for f in files], chunksize=4))
    time = np.concatenate([p[0] for p in parts])
    data = np.concatenate([p[1] for p in parts])
    keep = ~pd.Index(time).duplicated()
    out = xr.Dataset({args.variable: (("time", "station"), data[keep], {"units": "m"})},
                     coords={"time": time[keep], "gtsm_index": ("station", unique_idx),
                             "gtsm_station": ("station", ids[unique_idx]),
                             "lon": ("station", lonlat[unique_idx, 0]), "lat": ("station", lonlat[unique_idx, 1])},
                     attrs={"source": f"GTSM v3 reanalysis hourly, {folder}", "n_files": len(files)})
    out_dir = Path(args.out_dir)
    out.to_netcdf(out_dir / f"gtsm_{args.variable}_test_stations.nc")
    stations.to_csv(out_dir / "test_station_tiles.csv", index=False)
    print(f"{len(unique_idx)} stations, {len(files)} files, {keep.sum()} hours")


if __name__ == "__main__":
    main()
