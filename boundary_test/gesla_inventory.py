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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

GESLA_META = Path("P:/11212688-004-global-floodmaps/modelling/inputs/GESLA4.1_ALL/GESLA4-1_ALL.csv")
COLUMNS = ["file_name", "site_name", "site_code", "country", "contributor_abbr", "contributor_full",
           "contributor_website", "contributor_contact", "originator", "originator_website",
           "originator_contact", "lat", "lon", "coordinate_system", "start", "end", "n_years",
           "time_zone_hr", "datum_information", "instrument", "precision", "null_value", "gauge_type",
           "overall_quality", "download_link"]
MIN_YEARS = 20
DEDUP_KM = 1.0
EARTH_RADIUS_KM = 6371.0


def read_metadata(path: Path = GESLA_META) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, names=COLUMNS, index_col=False, skipinitialspace=True)
    if not (df.lat.between(-90, 90).all() and df.lon.between(-180, 360).all()):
        raise ValueError("GESLA metadata columns are misaligned: latitude/longitude out of range")
    return df


def contributor_rank(file_name: str) -> int:
    """0 = research quality / national, 1 = other, 2 = fast delivery (lower is preferred)."""
    suffix = file_name.rsplit("-", 1)[-1]
    if suffix.endswith("_fd"):
        return 2
    if suffix.endswith("_rq") or not suffix.startswith(("uhslc", "gloss")):
        return 0
    return 1


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(rank=df.file_name.map(contributor_rank)).sort_values(["rank", "n_years"],
                                                                        ascending=[True, False])
    tree = BallTree(np.radians(df[["lat", "lon"]].to_numpy()), metric="haversine")
    taken = np.zeros(len(df), dtype=bool)
    keep = []
    for i, neighbours in enumerate(tree.query_radius(np.radians(df[["lat", "lon"]].to_numpy()),
                                                     r=DEDUP_KM / EARTH_RADIUS_KM)):
        if taken[i]:
            continue
        keep.append(i)
        taken[neighbours] = True
    return df.iloc[keep].drop(columns="rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    df = read_metadata()
    ok = (df.gauge_type == "Coastal") & (df.n_years >= MIN_YEARS) & (df.overall_quality == "No obvious issues")
    sites = deduplicate(df[ok])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(out / "gesla_sites.parquet")
    print(f"{len(df)} records; {int(ok.sum())} coastal, >= {MIN_YEARS} y, no obvious issues; "
          f"{len(sites)} sites after {DEDUP_KM:g} km dedup")


if __name__ == "__main__":
    main()
