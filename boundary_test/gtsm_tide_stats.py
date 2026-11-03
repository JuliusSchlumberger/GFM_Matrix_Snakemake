"""Plan §5.B: MHWS, MHW, HW99, HAT of the GTSM tide (tidal_elevation) at the test-tile stations.

Reads <out-dir>/gtsm_tide (extract_gtsm_series.py --variable tide, 10-min) and writes
<out-dir>/tide_stats_gtsm.parquet, one row per GTSM station. GTSM tide is relative to the
1986-2005 MSL, the datum of COAST-RP.

Usage:
    python boundary_test/gtsm_tide_stats.py --out-dir .cleanup_baseline/boundary_test [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from tide_statistics import high_water_statistics  # noqa: E402

YEAR_S = 3.15576e7


def _station_stats(args) -> list[dict]:
    out, rows = args
    stations, time, series = load_series(Path(out), "tide")
    dt_s = float(np.median(np.diff(time.values[:1000])) / np.timedelta64(1, "s"))
    recs = []
    for j in rows:
        st = stations.iloc[j]
        s = station_series(series, j)
        rec = {"gtsm_station": int(st.gtsm_station), "gtsm_index": int(st.gtsm_index),
               "lon": float(st.lon), "lat": float(st.lat), "n_years": len(s) * dt_s / YEAR_S}
        ok = np.isfinite(s)
        if ok.mean() < 0.99:
            recs.append({**rec, "note": f"{1 - ok.mean():.1%} missing"})
            continue
        stats = high_water_statistics(pd.Series(s).interpolate(limit_area="inside").to_numpy(), dt_s)
        recs.append({**rec, **{f"gtsm_{k}": v for k, v in stats.items() if k != "high_waters_m"}})
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out_dir)
    n = len(load_series(out, "tide")[0])
    chunks = [(str(out), range(i, min(i + 200, n))) for i in range(0, n, 200)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for recs in pool.map(_station_stats, chunks):
            rows += recs
            print(f"  {len(rows)}/{n} stations", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "tide_stats_gtsm.parquet")
    print(df.drop(columns=["lon", "lat"]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
