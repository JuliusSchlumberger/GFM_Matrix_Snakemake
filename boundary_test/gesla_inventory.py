"""Deduplicated GESLA 4.1 gauge list for the boundary test (pre-check 1).

The metadata header lacks the comma between its last two columns, so pandas would shift every
column by one; it is read with explicit names. Gauges within DEDUP_KM of each other are one
site, kept by preference: research-quality / national contributor over fast-delivery, then
record length. `NUMBER OF YEARS` is the record span, not the count of flag-1 years, which
needs the data files. Writes `<out-dir>/gesla_sites.parquet`.

Usage:
    python boundary_test/gesla_inventory.py --out-dir .cleanup_baseline/boundary_test
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gesla import DEDUP_KM, deduplicate, read_metadata  # noqa: E402

GESLA_META = Path("P:/11212688-004-global-floodmaps/modelling/inputs/GESLA4.1_ALL/GESLA4-1_ALL.csv")
MIN_YEARS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    df = read_metadata(GESLA_META.parent)
    ok = (df.gauge_type == "Coastal") & (df.n_years >= MIN_YEARS) & (df.overall_quality == "No obvious issues")
    sites = deduplicate(df[ok])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(out / "gesla_sites.parquet")
    print(f"{len(df)} records; {int(ok.sum())} coastal, >= {MIN_YEARS} y, no obvious issues; "
          f"{len(sites)} sites after {DEDUP_KM:g} km dedup")


if __name__ == "__main__":
    main()
