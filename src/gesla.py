"""GESLA-4.1 tide gauges: deduplicated site list and compact per-site records since 1979.

The metadata header lacks the comma between its last two columns, so pandas would shift every
column by one; it is read with explicit names. Gauges within DEDUP_KM of each other are one
site, kept by preference: research-quality / national contributor over fast-delivery, then
record span. Records keep the samples flagged for use with QC 1 (correct) or 2 (interpolated),
shifted to UTC, in the gauge's own datum (every analysis removes a local MSL).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

METADATA_FILE = "GESLA4-1_ALL.csv"
COLUMNS = ["file_name", "site_name", "site_code", "country", "contributor_abbr", "contributor_full",
           "contributor_website", "contributor_contact", "originator", "originator_website",
           "originator_contact", "lat", "lon", "coordinate_system", "start", "end", "n_years",
           "time_zone_hr", "datum_information", "instrument", "precision", "null_value", "gauge_type",
           "overall_quality", "download_link"]
MIN_SPAN_YEARS = 20  # `NUMBER OF YEARS` is the record span, not the count of complete years
DEDUP_KM = 1.0
FIRST_YEAR = 1979  # start of the GTSM ERA5 surge series
GOOD_QC = (1, 2)
YEAR_COVERAGE = 0.8
EARTH_RADIUS_KM = 6371.0


def read_metadata(gesla_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(gesla_dir) / METADATA_FILE, header=0, names=COLUMNS, index_col=False,
                     skipinitialspace=True)
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


def coastal_sites(gesla_dir: Path) -> pd.DataFrame:
    """Coastal gauges spanning >= MIN_SPAN_YEARS with no obvious issues, one per DEDUP_KM."""
    df = read_metadata(gesla_dir)
    ok = (df.gauge_type == "Coastal") & (df.n_years >= MIN_SPAN_YEARS) & (df.overall_quality == "No obvious issues")
    return deduplicate(df[ok]).reset_index(drop=True)


def extract_record(gesla_dir: Path, file_name: str, tz_hr: float, target: Path) -> dict:
    """Write one site's samples to `target` (.npz: t datetime64[s] UTC, z float32 m); return stats."""
    if not target.exists():
        df = pd.read_csv(Path(gesla_dir) / file_name, comment="#", sep=r"\s+", header=None,
                         names=["date", "time", "z", "qc", "use"], dtype={"date": str, "time": str})
        df = df[(df.use == 1) & df.qc.isin(GOOD_QC) & (df.date.str[:4].astype(int) >= FIRST_YEAR)]
        t = pd.to_datetime(df.date + " " + df.time, format="%Y/%m/%d %H:%M:%S") - pd.Timedelta(hours=float(tz_hr))
        np.savez(target, t=t.to_numpy(dtype="datetime64[s]"), z=df.z.to_numpy(dtype=np.float32))
    t = np.load(target)["t"]
    if len(t) < 2:
        return {"file_name": file_name, "n_samples": len(t), "good_years": 0, "step_min": np.nan}
    step = np.median(np.diff(t).astype(float)) / 60.0
    years = t.astype("datetime64[Y]").astype(int) + 1970
    counts = pd.Series(years).value_counts()
    good = int((counts >= YEAR_COVERAGE * 365.25 * 24 * 60 / step).sum())
    return {"file_name": file_name, "n_samples": len(t), "good_years": good, "step_min": step}
