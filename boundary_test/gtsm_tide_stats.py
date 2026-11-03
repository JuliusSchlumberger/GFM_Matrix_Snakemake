"""Plan §5.B: MHWS, MHW, HW99, HAT of the GTSM tide (tidal_elevation) at the test-tile stations.

Reads <out-dir>/gtsm_tide_test_stations.nc (extract_gtsm_series.py --variable tide, 10-min) and
writes <out-dir>/tide_stats_gtsm.parquet, one row per GTSM station. GTSM tide is relative to
the 1986-2005 MSL, the datum of COAST-RP.

Usage:
    python boundary_test/gtsm_tide_stats.py --out-dir .cleanup_baseline/boundary_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tide_statistics import high_water_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    ds = xr.open_dataset(out / "gtsm_tide_test_stations.nc")
    dt_s = float(np.median(np.diff(ds.time.values)) / np.timedelta64(1, "s"))
    rows = []
    for j in range(ds.sizes["station"]):
        series = ds["tide"].values[:, j].astype(np.float64)
        rec = {"gtsm_station": int(ds.gtsm_station[j]), "gtsm_index": int(ds.gtsm_index[j]),
               "lon": float(ds.lon[j]), "lat": float(ds.lat[j]), "n_years": len(series) * dt_s / 3.15576e7}
        ok = np.isfinite(series)
        if ok.mean() < 0.99:
            rows.append({**rec, "note": f"{1 - ok.mean():.1%} missing"})
            continue
        series = pd.Series(series).interpolate(limit_area="inside").to_numpy()
        stats = high_water_statistics(series, dt_s)
        rows.append({**rec, **{f"gtsm_{k}": v for k, v in stats.items() if k != "high_waters_m"}})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "tide_stats_gtsm.parquet")
    print(df.drop(columns=["lon", "lat"]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
