"""Load the first N rows of each hazard table into a real GeoDataFrame.

Proves the no-database parsing path works and reports real schema / values.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from nor_dump import block_for, iter_rows, load_index

N = 300
HERE = Path(__file__).parent
IDX = load_index()
DUMP = Path(IDX["dump"])

TABLES = [
    "dekningsomrade",
    "middelhoyvann_klimaarna",
    "stormflo20ar_klimaarna",
    "stormflo200ar_klimaarna",
    "stormflo1000ar_klimaarna",
    "stormflo200ar_klimaar2100",
    "stormflo1000ar_klimaar2100",
    "stormfloovreestimat_klimaarna",
]


def to_gdf(table: str, n: int = N) -> gpd.GeoDataFrame:
    block = block_for(IDX, table)
    rows = list(iter_rows(DUMP, block, limit=n))
    df = pd.DataFrame(rows)
    geom = shapely.from_wkb([bytes.fromhex(h) for h in df.pop("omrade")])
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:25833")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    for table in TABLES:
        gdf = to_gdf(table)
        print("=" * 90)
        print(f"{table}   (first {len(gdf)} rows)")
        print("=" * 90)
        print("dtypes:")
        print(gdf.dtypes.to_string())
        print(f"\ncrs                : {gdf.crs}")
        print(f"geom types         : {sorted(gdf.geom_type.unique())}")
        print(f"bbox (EPSG:25833)  : {[round(v, 1) for v in gdf.total_bounds]}")
        print(f"bbox (EPSG:4326)   : {[round(v, 4) for v in gdf.to_crs(4326).total_bounds]}")
        print(f"has_z              : {bool(shapely.has_z(gdf.geometry.values).any())}")
        for col in gdf.columns:
            if col == "geometry":
                continue
            u = gdf[col].unique()
            head = list(u[:6])
            print(f"  {col:<22} nunique={len(u):<7} e.g. {head}")
        if "vannstandovernn2000" in gdf:
            v = pd.to_numeric(gdf["vannstandovernn2000"])
            print(f"  -> vannstandovernn2000 min={v.min()} max={v.max()} (integers)")
        print()

    # write one small head sample so it can be opened in QGIS
    out = HERE / "norway_head_sample.gpkg"
    for table in ["stormflo20ar_klimaarna", "stormflo200ar_klimaarna", "stormflo1000ar_klimaarna"]:
        to_gdf(table, 200).to_file(out, layer=table, driver="GPKG")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
