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
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gtsm_series import FOLDERS, extract_series, gtsm_stations, station_series  # noqa: E402, F401 (re-exported)
from gtsm_series import load_series as _load_series  # noqa: E402

GTSM_DIR = Path("P:/11212688-004-global-floodmaps/modelling/inputs/GTSM_storm_tide_hourly")
BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
MATCH_KM = 0.1
EARTH_RADIUS_KM = 6371.0


def load_series(out: Path, variable: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Return the stations table, the time axis and the (station, time) int16 mm memmap."""
    return _load_series(out / f"gtsm_{variable}")


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
    out = Path(args.out_dir)
    lonlat, _ = gtsm_stations(GTSM_DIR, args.variable)
    stations = (test_station_indices(pd.read_csv(args.tiles_csv, dtype={"tile_id": str}), lonlat) if args.tiles_csv
                else pd.read_csv(args.stations_csv))
    extract_series(GTSM_DIR, stations.gtsm_index.to_numpy(), out / f"gtsm_{args.variable}", args.workers, args.variable)
    if args.tiles_csv:
        stations.to_csv(out / "test_station_tiles.csv", index=False)


if __name__ == "__main__":
    main()
