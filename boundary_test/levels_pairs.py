"""Step 1: pair every GESLA site with its nearest GTSM, COAST-RP and COAST-HG station.

GTSM stations are all ~43,000 output points of the tide files; COAST-RP and COAST-HG each have
their own station rows (not aligned with each other). Distances are great-circle, in km.

Writes <out-dir>/pairs.parquet and <out-dir>/gtsm_stations.csv (the GTSM indices to extract).

Usage:
    python boundary_test/levels_pairs.py --out-dir .cleanup_baseline/level_sources \
        --gesla-sites .cleanup_baseline/boundary_test/gesla_sites.parquet
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

INPUTS = Path("P:/11212688-004-global-floodmaps/modelling/inputs")
GTSM_TIDE_FILE = INPUTS / "GTSM_storm_tide_hourly" / "tidal_elevation" / "historical_tide_2000_01_v1.nc"
COAST_RP = INPUTS / "COAST-RP_dataset" / "COAST-RP.nc"
COAST_HG = INPUTS / "COAST_HG" / "COAST-HG_RP100.nc"
EARTH_RADIUS_KM = 6371.0


def station_lonlat(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        return np.column_stack([ds["station_x_coordinate"][:], ds["station_y_coordinate"][:]]).astype(float)


def nearest(src_lonlat: np.ndarray, query_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(src_lonlat).all(axis=1)
    idx = np.flatnonzero(ok)
    d, i = BallTree(np.radians(src_lonlat[ok][:, ::-1]), metric="haversine").query(np.radians(query_lonlat[:, ::-1]), k=1)
    return idx[i[:, 0]], d[:, 0] * EARTH_RADIUS_KM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gesla-sites", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(args.gesla_sites).reset_index(drop=True)
    q = sites[["lon", "lat"]].to_numpy(dtype=float)
    pairs = sites[["file_name", "site_name", "country", "lon", "lat", "n_years", "datum_information"]].copy()
    for name, path in (("gtsm", GTSM_TIDE_FILE), ("coast_rp", COAST_RP), ("coast_hg", COAST_HG)):
        ll = station_lonlat(path)
        i, d = nearest(ll, q)
        pairs[f"{name}_index"], pairs[f"{name}_dist_km"] = i, d
        pairs[f"{name}_lon"], pairs[f"{name}_lat"] = ll[i, 0], ll[i, 1]
        print(f"{name}: {len(ll)} stations; distance to GESLA median {np.median(d):.1f} km, "
              f"p90 {np.quantile(d, 0.9):.1f} km, max {d.max():.1f} km")
    pairs.to_parquet(out / "pairs.parquet")
    pd.DataFrame({"gtsm_index": np.unique(pairs.gtsm_index)}).to_csv(out / "gtsm_stations.csv", index=False)
    print(f"{len(pairs)} GESLA sites, {pairs.gtsm_index.nunique()} distinct GTSM stations")


if __name__ == "__main__":
    main()
