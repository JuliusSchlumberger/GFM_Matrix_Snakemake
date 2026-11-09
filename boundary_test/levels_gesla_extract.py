"""Step 1: read the GESLA 4.1 records of the paired sites into compact per-site arrays.

Kept: samples from FIRST_YEAR on with the use-in-analysis flag 1 and QC flag 1 (correct) or
2 (interpolated). Times are shifted to UTC with the header's TIME ZONE HOURS. Levels stay in
the gauge's own datum (chart datum, national datum, ...); every analysis removes a local MSL.

Writes <out-dir>/gesla/<file_name>.npz (t: datetime64[s], z: float32 m) and
<out-dir>/gesla_records.csv (samples, years with >= 80% coverage, median step).

Usage:
    python boundary_test/levels_gesla_extract.py --out-dir .cleanup_baseline/level_sources \
        --gesla-sites .cleanup_baseline/boundary_test/gesla_sites.parquet [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gesla import extract_record  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sfincs_tiles"))
from gfm_config import read_root  # noqa: E402

DATA_ROOT = read_root(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")

GESLA_DIR = DATA_ROOT / "inputs" / "GESLA4.1_ALL"
FIRST_YEAR = 1979  # start of the GTSM ERA5 surge series


def read_site(args) -> dict:
    file_name, tz_hr, out = args
    return extract_record(GESLA_DIR, file_name, tz_hr, Path(out) / "gesla" / f"{file_name}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gesla-sites", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "gesla").mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(args.gesla_sites)
    tz = dict(zip(sites.file_name, pd.to_numeric(sites.time_zone_hr, errors="coerce").fillna(0.0)))
    pairs = pd.read_parquet(out / "pairs.parquet")
    tasks = [(f, tz.get(f, 0.0), str(out)) for f in pairs.file_name]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(read_site, tasks)):
            rows.append(row)
            if k % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} sites", flush=True)
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "gesla_records.csv", index=False)
    print(f"{len(rec)} sites; >= 10 good years since {FIRST_YEAR}: {(rec.good_years >= 10).sum()}; "
          f"median step {rec.step_min.median():.0f} min")


if __name__ == "__main__":
    main()
