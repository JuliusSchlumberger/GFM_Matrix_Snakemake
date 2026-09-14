"""Full streaming profile of selected hazard tables + AOI extraction.

Reads each ~2.4 GiB COPY block once, in batches, and accumulates statistics
(bbox, area, geometry types, vertex counts, value histogram of
``vannstandovernn2000``) without keeping every geometry in memory.  In the same
pass it collects the polygons intersecting two AOIs (Oslofjord, Bergen) and
writes them to a GeoPackage for visual inspection.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from nor_dump import block_for, columns, iter_raw_lines, load_index, unescape

HERE = Path(__file__).parent
IDX = load_index()
DUMP = Path(IDX["dump"])
BATCH = 2000

TABLES = [
    "middelhoyvann_klimaarna",
    "stormflo20ar_klimaarna",
    "stormflo200ar_klimaarna",
    "stormflo1000ar_klimaarna",
]

# AOIs in lon/lat -> EPSG:25833
_T = Transformer.from_crs(4326, 25833, always_xy=True)


def aoi(name: str, lon0, lat0, lon1, lat1):
    x0, y0 = _T.transform(lon0, lat0)
    x1, y1 = _T.transform(lon1, lat1)
    return name, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


AOIS = [
    aoi("oslofjord", 10.55, 59.80, 10.90, 59.98),
    aoi("bergen", 5.20, 60.33, 5.42, 60.45),
]


def profile(table: str) -> gpd.GeoDataFrame:
    block = block_for(IDX, table)
    cols = columns(block)
    gi = cols.index("omrade")

    n = 0
    minx = miny = np.inf
    maxx = maxy = -np.inf
    area = 0.0
    verts = 0
    gtypes: Counter = Counter()
    level_rows: Counter = Counter()
    level_area: Counter = Counter()
    invalid = 0
    keep: dict[str, list] = {a: [] for a, _ in AOIS}

    t0 = time.time()
    batch: list[list[str]] = []

    def flush():
        nonlocal n, minx, miny, maxx, maxy, area, verts, invalid
        if not batch:
            return
        geoms = shapely.from_wkb([bytes.fromhex(r[gi]) for r in batch])
        b = shapely.bounds(geoms)
        minx = min(minx, np.nanmin(b[:, 0]))
        miny = min(miny, np.nanmin(b[:, 1]))
        maxx = max(maxx, np.nanmax(b[:, 2]))
        maxy = max(maxy, np.nanmax(b[:, 3]))
        a = shapely.area(geoms)
        area += float(a.sum())
        verts += int(shapely.get_num_coordinates(geoms).sum())
        invalid += int((~shapely.is_valid(geoms)).sum())
        gtypes.update(shapely.get_type_id(geoms).tolist())
        if "vannstandovernn2000" in cols:
            li = cols.index("vannstandovernn2000")
            for row, ar in zip(batch, a):
                level_rows[int(row[li])] += 1
                level_area[int(row[li])] += float(ar)
        for name, bb in AOIS:
            hit = np.where(
                (b[:, 0] <= bb[2]) & (b[:, 2] >= bb[0]) & (b[:, 1] <= bb[3]) & (b[:, 3] >= bb[1])
            )[0]
            for i in hit:
                rec = {c: unescape(batch[i][j]) for j, c in enumerate(cols) if c != "omrade"}
                rec["geometry"] = geoms[i]
                keep[name].append(rec)
        n += len(batch)
        batch.clear()

    for line in iter_raw_lines(DUMP, block):
        batch.append(line.decode("utf-8").split("\t"))
        if len(batch) >= BATCH:
            flush()
    flush()

    el = time.time() - t0
    print("=" * 92)
    print(f"{table}   FULL TABLE  ({n:,} rows, {el:.0f}s)")
    print("=" * 92)
    tid = {0: "Point", 3: "Polygon", 6: "MultiPolygon"}
    print(f"  geometry types      : { {tid.get(k, k): v for k, v in gtypes.items()} }")
    print(f"  invalid geometries  : {invalid:,}")
    print(f"  total vertices      : {verts:,}  (mean {verts / max(n,1):,.0f} per feature)")
    print(f"  bbox EPSG:25833     : [{minx:,.0f}, {miny:,.0f}, {maxx:,.0f}, {maxy:,.0f}]")
    ll = Transformer.from_crs(25833, 4326, always_xy=True)
    lo0, la0 = ll.transform(minx, miny)
    lo1, la1 = ll.transform(maxx, maxy)
    print(f"  bbox EPSG:4326      : [{lo0:.3f}, {la0:.3f}, {lo1:.3f}, {la1:.3f}]")
    print(f"  total polygon area  : {area / 1e6:,.1f} km2")
    if level_rows:
        print(f"  vannstandovernn2000 : {len(level_rows)} distinct values (cm above NN2000)")
        print(f"     min={min(level_rows)} max={max(level_rows)}")
        print(f"     {'cm':>6} {'features':>12} {'area km2':>14}")
        for k in sorted(level_rows):
            print(f"     {k:>6} {level_rows[k]:>12,} {level_area[k] / 1e6:>14,.1f}")
    print()

    for name, _ in AOIS:
        if keep[name]:
            g = gpd.GeoDataFrame(keep[name], geometry="geometry", crs="EPSG:25833")
            out = HERE / f"norway_aoi_{name}.gpkg"
            g.to_file(out, layer=table, driver="GPKG")
            print(f"  AOI {name:<10} {len(g):>6} features, "
                  f"{g.area.sum() / 1e6:8.2f} km2 -> {out.name}:{table}")
    print()
    return None


def main() -> None:
    for t in TABLES:
        profile(t)


if __name__ == "__main__":
    main()
