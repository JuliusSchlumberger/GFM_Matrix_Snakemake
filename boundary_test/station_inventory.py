"""Station inventory for the boundary test: COAST-RP levels, TC/ETC split, GTSM match, tide proxy.

Pre-check 2 of the boundary-test plan. Reads COAST-RP (combined, TC, ETC), COAST-HG RP100 and
the GTSM hourly station coordinates read-only; writes `<out-dir>/stations.parquet`.
The tide proxy is the highest level of COAST-HG's spring-tide hydrograph more than
TIDE_TAIL_HR from the surge peak, where the surge has decayed: a GTSM spring high water.

Usage:
    python boundary_test/station_inventory.py --out-dir .cleanup_baseline/boundary_test
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

INPUTS = Path("P:/11212688-004-global-floodmaps/modelling/inputs")
COAST_RP_DIR = INPUTS / "COAST-RP_dataset"
COAST_HG_NC = INPUTS / "COAST_HG" / "COAST-HG_RP100.nc"
GTSM_SURGE_DIR = INPUTS / "GTSM_storm_tide_hourly" / "storm_surge_residual"
RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
MATCH_KM = 0.1  # COAST-RP stations are GTSM output stations: coincidence tolerance
TIDE_TAIL_HR = 36.0
EARTH_RADIUS_KM = 6371.0


def _lonlat(ds) -> np.ndarray:
    return np.column_stack([np.asarray(ds["station_x_coordinate"][:], dtype=float),
                            np.asarray(ds["station_y_coordinate"][:], dtype=float)])


def nearest(src_lonlat: np.ndarray, query_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest src point to each query point and its distance (km)."""
    tree = BallTree(np.radians(src_lonlat[:, ::-1]), metric="haversine")
    d, i = tree.query(np.radians(query_lonlat[:, ::-1]), k=1)
    return i[:, 0], d[:, 0] * EARTH_RADIUS_KM


def build_inventory() -> pd.DataFrame:
    with xr.open_dataset(COAST_RP_DIR / "COAST-RP.nc") as ds:
        ll = _lonlat(ds)
        df = pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1],
                           "station_id": ds["station_id"].values.astype(str)})
        for rp in RPS:
            df[f"rp{rp}_m"] = ds[f"storm_tide_rp_{rp:04d}"].values.astype(float)
    for part in ("TC", "ETC"):
        with xr.open_dataset(COAST_RP_DIR / f"COAST-RP_{part}.nc") as ds:
            idx, dist = nearest(_lonlat(ds), ll)
            rp100 = ds["storm_tide_rp_0100"].values.astype(float)[idx]
            df[f"{part.lower()}_rp100_m"] = np.where(dist <= MATCH_KM, rp100, np.nan)
    # COAST-RP combines TC and ETC probabilistically; the larger part sets the RP100 level.
    df["tc_dominated"] = df.tc_rp100_m.notna() & (df.tc_rp100_m > df.etc_rp100_m.fillna(-np.inf))

    with netCDF4.Dataset(sorted(GTSM_SURGE_DIR.glob("reanalysis_surge_hourly_*_v3.nc"))[0]) as ds:
        gtsm_ll = _lonlat(ds)
        gtsm_ids = np.asarray(ds["stations"][:]).astype(int)
    idx, dist = nearest(gtsm_ll, ll)
    df["gtsm_index"], df["gtsm_station"], df["gtsm_dist_km"] = idx, gtsm_ids[idx], dist

    with xr.open_dataset(COAST_HG_NC) as ds:
        idx, dist = nearest(_lonlat(ds), ll)
        spring = ds["hydrograph_spring_tide_signal"].values
        t_hr = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "h")
    peak_t = t_hr[np.nanargmax(spring, axis=1)]
    tail = np.abs(t_hr[None, :] - peak_t[:, None]) > TIDE_TAIL_HR
    hw = np.where(tail, spring, np.nan)
    df["coast_hg_dist_km"] = dist
    df["spring_hw_proxy_m"] = np.nanmax(hw, axis=1)[idx]
    df["spring_lw_proxy_m"] = np.nanmin(hw, axis=1)[idx]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_inventory()
    df.to_parquet(out / "stations.parquet")
    print(f"{len(df)} stations; GTSM match <= {MATCH_KM} km: {(df.gtsm_dist_km <= MATCH_KM).mean():.1%}; "
          f"COAST-HG match <= {MATCH_KM} km: {(df.coast_hg_dist_km <= MATCH_KM).mean():.1%}; "
          f"TC-dominated: {df.tc_dominated.sum()}")


if __name__ == "__main__":
    main()
