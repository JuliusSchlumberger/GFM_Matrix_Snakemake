"""Leave-one-out test of station-to-cell interpolation schemes on the level product (§8.4-8.5).

Each station's RP level is predicted from the other stations by inverse-distance weighting of
the k nearest (power p; haversine distance floored at 1 km), as `flood_model.idw_seed_values_at`
does for the eikonal seeds (`simulation.flooding.knn`, `idw_power`), and by the nearest station alone. The
error between stations bounds the interpolation error at the coastline cells between them.
Production also restricts the neighbours to the tile's ocean component; that is not reproduced
here. Stations below PRODUCTION_MIN_LAT are left out, as in the preparation.

Writes <out-dir>/interpolation.csv (scheme x RP), interpolation_spacing.csv (MAE by distance to
the nearest other station) and interpolation_outliers.csv (stations whose RP100 is more than
OUTLIER_M from the k = 5, p = 2 prediction).

Usage:
    python boundary_test/station_interpolation.py --out-dir <storm_tide_product_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
PRODUCTION_MIN_LAT = -60.0
TEST_RPS = (10, 100, 1000)
KS = (1, 2, 3, 5, 8, 15, 30)
POWERS = (1, 2, 3)
SPACING_BINS_KM = (0, 5, 10, 20, 50, np.inf)
OUTLIER_M = 1.0


def loo_predictions(lonlat: np.ndarray, values: np.ndarray, k: int, power: float) -> np.ndarray:
    tree = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine")
    d, i = tree.query(np.radians(lonlat[:, ::-1]), k=k + 1)
    d, i = np.maximum(d[:, 1:] * EARTH_RADIUS_KM, 1.0), i[:, 1:]
    w = d ** -power
    return (w * values[i]).sum(axis=1) / w.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    out = Path(parser.parse_args().out_dir)
    lev = pd.read_parquet(out / "levels.parquet")
    lev = lev[(lev.lat > PRODUCTION_MIN_LAT) & lev[[f"rp{rp}_m" for rp in TEST_RPS]].notna().all(axis=1)]
    lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    rows = []
    for rp in TEST_RPS:
        v = lev[f"rp{rp}_m"].to_numpy()
        for power in POWERS:
            for k in KS:
                if k == 1 and power != POWERS[0]:
                    continue
                e = loo_predictions(lonlat, v, k, power) - v
                rows.append({"rp": rp, "k": k, "power": power if k > 1 else 0, "mae_m": np.abs(e).mean(),
                             "p90_abs_m": np.quantile(np.abs(e), 0.9), "n_abs_gt_0.5m": int((np.abs(e) > 0.5).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "interpolation.csv", index=False)

    tree = BallTree(np.radians(lonlat[:, ::-1]), metric="haversine")
    spacing = tree.query(np.radians(lonlat[:, ::-1]), k=2)[0][:, 1] * EARTH_RADIUS_KM
    bins = pd.cut(spacing, SPACING_BINS_KM)
    v = lev.rp100_m.to_numpy()
    by_spacing = []
    for name, k, power in (("nearest", 1, 1), ("k = 15, p = 2 (former)", 15, 2), ("k = 5, p = 3 (production)", 5, 3)):
        e = np.abs(loo_predictions(lonlat, v, k, power) - v)
        by_spacing.append(pd.Series(e).groupby(bins, observed=True).mean().rename(name))
    by_spacing = pd.concat(by_spacing, axis=1)
    by_spacing["stations"] = pd.Series(spacing).groupby(bins, observed=True).size()
    by_spacing.to_csv(out / "interpolation_spacing.csv", index_label="spacing_km")

    e = loo_predictions(lonlat, v, 5, 2) - v
    outliers = lev.assign(loo_error_m=e, spacing_km=spacing)[np.abs(e) > OUTLIER_M]
    outliers[["station_id", "lon", "lat", "atlas", "level_source", "mhws_m", "rp100_m", "coast_rp_rp100_m",
              "loo_error_m", "spacing_km"]].to_csv(out / "interpolation_outliers.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table[table.rp == 100].round(4).to_string(index=False))
        print(by_spacing.round(3))
    print(f"{len(outliers)} stations more than {OUTLIER_M} m from the k = 5 prediction at RP100")


if __name__ == "__main__":
    main()
