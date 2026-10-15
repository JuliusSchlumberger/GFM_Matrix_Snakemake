"""Download the Environment Agency's "Flood Map for Planning: Flood Zones 2
& 3 (Rivers and Sea)" dataset for England, filtered to the sea-source,
modelled-origin subset this framework's coastal validation needs, and save
it alongside Wales/Scotland's own downloaded benchmarks.

Flood Zone 3 (sea source): >=1-in-200-year (0.5% AEP) annual probability -
no exact native model RP (model has RP100/RP250, not RP200) - bracket it the
same way Norway's own ~200yr-class benchmark is bracketed (see
data_catalog_validation.yml's comparison_return_periods on that entry).
Flood Zone 2 (sea source): >=1-in-1000-year (0.1% AEP) - matches the model's
own native RP1000 exactly, no bracketing needed.

England's real coastal extent (not all of GBR - Wales/Scotland have their
own separate agencies/datasets) is restricted at validate_country.py's own
scoring time via geogunit_ids, same mechanism as Wales(3367)/Scotland(3366);
this script's own bbox is deliberately generous (doesn't need to be exact).

Usage:
    python validation/fetch_england_flood_zones.py
"""

import os
import time

import geopandas as gpd
import pandas as pd
import requests

BASE = (
    "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-flood-zones/"
    "ogc/features/v1/collections/Flood_Zones_2_3_Rivers_and_Sea/items"
)
# England's real extent (lon/lat, CRS84) - generous, geogunit_ids does the real restriction.
ENGLAND_BBOX = (-6.5, 49.8, 2.0, 55.9)
TILE_DEG = 1.0  # smaller per-request bbox - easier on the server, clear incremental progress
OUT_DIR = "P:/11212688-004-global-floodmaps/modelling/inputs/validation/England"
CACHE_DIR = f"{OUT_DIR}/_tile_cache"


def _tiles(bbox, step):
    minx, miny, maxx, maxy = bbox
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            yield (x, y, min(x + step, maxx), min(y + step, maxy))
            y += step
        x += step


def fetch_bbox(bbox, page=2000):
    url = BASE
    params = {"f": "json", "bbox": ",".join(map(str, bbox)), "limit": page}
    parts = []
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js["features"]:
            parts.append(gpd.GeoDataFrame.from_features(js["features"], crs=4326))
        url = next((link["href"] for link in js.get("links", []) if link["rel"] == "next"), None)
        params = None
    return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame()


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    tiles = list(_tiles(ENGLAND_BBOX, TILE_DEG))
    print(f"Fetching England flood zones in {len(tiles)} tile(s) of {TILE_DEG} deg ...", flush=True)

    all_parts = []
    for i, tile in enumerate(tiles):
        cache_path = f"{CACHE_DIR}/tile_{i:03d}.gpkg"
        if os.path.exists(cache_path):
            gdf = gpd.read_file(cache_path)
            print(f"[{i + 1}/{len(tiles)}] {tile}: {len(gdf)} feature(s) (cached)", flush=True)
        else:
            t0 = time.time()
            gdf = fetch_bbox(tile)
            if not gdf.empty:
                gdf.to_file(cache_path, driver="GPKG")
            print(f"[{i + 1}/{len(tiles)}] {tile}: {len(gdf)} feature(s) ({time.time() - t0:.1f}s)", flush=True)
        if not gdf.empty:
            all_parts.append(gdf)

    gdf = pd.concat(all_parts, ignore_index=True) if all_parts else gpd.GeoDataFrame()
    print(f"\nTotal features fetched: {len(gdf)}", flush=True)
    if gdf.empty:
        return
    print(gdf.groupby(["flood_zone", "flood_source", "origin"]).size(), flush=True)

    sea_modelled = gdf.query("flood_source == 'sea' and origin == 'modelled'")
    for zone, name in [("FZ3", "fz3_sea_modelled"), ("FZ2", "fz2_sea_modelled")]:
        subset = sea_modelled[sea_modelled["flood_zone"] == zone].drop_duplicates(subset="geometry")
        out_path = f"{OUT_DIR}/{name}.gpkg"
        subset.to_file(out_path, driver="GPKG")
        print(f"wrote {out_path}: {len(subset)} feature(s)", flush=True)


if __name__ == "__main__":
    main()
