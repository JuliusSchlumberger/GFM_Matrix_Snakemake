"""One-time converter: PostGIS SQL dump table -> GeoPackage (no database needed).

This is the prototype of the recommended ingest path.  It streams one COPY
block out of the 28 GB plain-SQL pg_dump, parses the hex-EWKB geometry column
with shapely, and appends batches to a GeoPackage, so peak memory stays flat
regardless of table size.

The resulting .gpkg is a normal OGR vector file, i.e. a plain
``data_type: GeoDataFrame`` / ``driver: vector`` hydromt catalog entry, exactly
like Spain's and France's shapefiles.

Usage:
    python 06_convert_table.py stormflo200ar_klimaarna            # whole table
    python 06_convert_table.py stormflo200ar_klimaarna --limit 20000
    python 06_convert_table.py stormflo200ar_klimaarna --to-4326
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from nor_dump import block_for, columns, iter_raw_lines, load_index, unescape

HERE = Path(__file__).parent
BATCH = 5000

# Columns that carry no information (constant for the whole table) and only
# bloat the output. Keep objid/lokalid as identifiers.
DROP = ["navnerom", "malemetode", "opphav", "objtype"]


def convert(table: str, out: Path, limit: int | None, to_4326: bool) -> None:
    index = load_index()
    dump = Path(index["dump"])
    block = block_for(index, table)
    cols = columns(block)
    gi = cols.index("omrade")

    if out.exists():
        out.unlink()

    t0 = time.time()
    n = 0
    batch: list[list[str]] = []
    first = True

    def flush() -> None:
        nonlocal n, first
        if not batch:
            return
        geoms = shapely.from_wkb([bytes.fromhex(r[gi]) for r in batch])
        data = {
            c: [unescape(r[j]) for r in batch]
            for j, c in enumerate(cols)
            if c != "omrade" and c not in DROP
        }
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry=geoms, crs="EPSG:25833")
        for c in ("objid", "klimaar", "vannstandovernn2000"):
            if c in gdf:
                gdf[c] = pd.to_numeric(gdf[c], downcast="integer")
        if to_4326:
            gdf = gdf.to_crs(4326)
        gdf.to_file(out, layer=table, driver="GPKG", mode="w" if first else "a")
        first = False
        n += len(batch)
        batch.clear()
        print(f"  {n:>9,} rows  {time.time() - t0:6.0f}s", flush=True)

    for line in iter_raw_lines(dump, block):
        batch.append(line.decode("utf-8").split("\t"))
        if len(batch) >= BATCH:
            flush()
        if limit is not None and n >= limit:
            break
    flush()

    print(f"\n{table}: {n:,} rows -> {out}  ({out.stat().st_size / 1024**2:,.0f} MB, "
          f"{time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--to-4326", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / f"{a.table}.gpkg"
    convert(a.table, out, a.limit, a.to_4326)


if __name__ == "__main__":
    main()
