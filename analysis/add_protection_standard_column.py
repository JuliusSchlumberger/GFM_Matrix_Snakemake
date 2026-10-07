"""Adds a `max_coastal_protection_standard_years` column to
`compute_delta_flood_extent.py`'s own output CSV: the maximum FLOPROS
coastal design return period (years) among the WRI geogunit_107 units
intersecting each delta's real polygon - i.e. "what's the best-protected
point inside this delta, by design standard?".

Reuses the same FLOPROS / geogunit_107 protection-standard data source
already used by the exposure pipeline (`compute_exposure_analysis.py`,
`compute_flood_totals.py`): `flopros_protection_standards` (Tiggeloven et
al. 2020 design return periods, per geogunit, `Coastal` column with
`Riverine` fallback - same convention those scripts already use) joined
against the `geogunit_protection_units` raster (WRI geogunit_107 IDs, ~30
arcsec, EPSG:4326). Unlike `compute_exposure_analysis.py`'s own use of this
same table (which SNAPS each value to the nearest simulated return period,
for picking which simulated RP counts as "protected" in its EAI
calculation), this column reports the RAW FLOPROS design standard - a
delta-level reporting figure, not an input to any threshold logic here.

Per delta: the geogunit raster is read clipped to the delta's own bbox
(`DataCatalog.get_rasterdataset(..., bbox=...)`), the delta's real polygon
is rasterized onto THAT raster's own native grid with
`validation.tri_domain_mask` (same reusable building block
`compute_delta_flood_extent.py` already uses for the flood-extent mask),
and the MAX of `coastal_rp` across every distinct geogunit ID touching the
polygon is taken. A delta with no resolvable geogunit coverage (offshore
polygon slivers, data gaps) gets NaN here - never silently defaulted to 0
or `default_rp`, which would misrepresent "no data" as "no protection".

This is a standalone enrichment step, matching `add_slr_year_column.py`'s
own convention (reads/writes the CSV only; not wired into the Snakemake
DAG, since neither this script nor `compute_delta_flood_extent.py` nor
`add_slr_year_column.py` are Snakemake rules today - all three are run
manually, in sequence, after the HPC postprocess stage finishes).

Usage:
    python add_protection_standard_column.py
    python add_protection_standard_column.py --csv path/to/delta_flood_extent.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from validation import tri_domain_mask  # noqa: E402

from compute_delta_flood_extent import DELTAS_PATH, VOLGA_ID  # noqa: E402  (sibling script - reuse, avoid re-duplicating path/exclusion)

_REPO_ROOT = Path(__file__).resolve().parent.parent
GEOGUNIT_SOURCE = "geogunit_protection_units"  # catalog key (data_catalog_gfm.yml)


def max_coastal_protection_for_delta(delta_geometry, catalog, coastal_rp: pd.Series) -> float:
    """Max FLOPROS coastal design RP (years) among geogunits intersecting
    `delta_geometry`. NaN if no geogunit resolves within the delta's own
    polygon (not silently defaulted - see module docstring)."""
    bbox = list(delta_geometry.bounds)
    try:
        geo_da = catalog.get_rasterdataset(GEOGUNIT_SOURCE, bbox=bbox, variables=["Geogunits"]).squeeze(drop=True)
    except Exception:
        return float("nan")
    if geo_da is None or geo_da.size == 0:
        return float("nan")

    geo_arr = geo_da.values.astype("float64")
    nodata = geo_da.raster.nodata
    valid = np.isfinite(geo_arr) & (geo_arr >= 0)
    if nodata is not None:
        valid &= geo_arr != nodata

    gdf_single = gpd.GeoDataFrame({"geometry": [delta_geometry]}, crs="EPSG:4326")
    domain = tri_domain_mask(gdf_single, geo_da.raster.transform, geo_da.raster.crs, geo_arr.shape)

    ids = np.unique(geo_arr[valid & domain].astype("int32"))
    if ids.size == 0:
        return float("nan")

    rps = [coastal_rp.get(int(gid)) for gid in ids]
    rps = [float(r) for r in rps if r is not None and not pd.isna(r)]
    return max(rps) if rps else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--csv", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    csv_path = Path(args.csv) if args.csv else root / "deltas_floodmaps" / "delta_flood_extent.csv"
    default_rp = float(cfg.get("protection", {}).get("default_rp", 5.0))

    df = pd.read_csv(csv_path)
    print(f"{len(df)} row(s) from {csv_path}")

    deltas = retry_transient_io(gpd.read_file, DELTAS_PATH)
    deltas = deltas[deltas["Id"] != VOLGA_ID].reset_index(drop=True)

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")
    coastal_rp = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)

    print(f"Computing max coastal protection standard for {len(deltas)} delta(s)...")
    values: dict[int, float] = {}
    for i, (_, delta_row) in enumerate(deltas.iterrows(), start=1):
        delta_id = int(delta_row["Id"])
        values[delta_id] = max_coastal_protection_for_delta(delta_row.geometry, catalog, coastal_rp)
        print(f"  [{i}/{len(deltas)}] delta {delta_id:2d}: {values[delta_id]} yr", flush=True)

    df["max_coastal_protection_standard_years"] = df["delta_id"].map(values)
    n_missing = df["max_coastal_protection_standard_years"].isna().sum()
    print(f"{n_missing}/{len(df)} row(s) have no resolvable geogunit coverage - left NaN (not defaulted)")

    retry_transient_io(df.to_csv, csv_path, index=False)
    print(f"Wrote {csv_path} (added max_coastal_protection_standard_years column)")


if __name__ == "__main__":
    main()
