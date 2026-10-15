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

import geopandas as gpd
import requests

BASE = (
    "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-flood-zones/"
    "ogc/features/v1/collections/Flood_Zones_2_3_Rivers_and_Sea/items"
)
# England's real extent (lon/lat, CRS84) - generous, geogunit_ids does the real restriction.
ENGLAND_BBOX = (-6.5, 49.8, 2.0, 55.9)
OUT_DIR = "P:/11212688-004-global-floodmaps/modelling/inputs/validation/England"


def fetch(bbox, page=2000):
    url = BASE
    params = {"f": "json", "bbox": ",".join(map(str, bbox)), "limit": page}
    parts = []
    n_pages = 0
    while url:
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        js = r.json()
        if js["features"]:
            parts.append(gpd.GeoDataFrame.from_features(js["features"], crs=4326))
        n_pages += 1
        print(f"  page {n_pages}: {len(js.get('features', []))} feature(s) (running total: {sum(len(p) for p in parts)})", flush=True)
        url = next((link["href"] for link in js.get("links", []) if link["rel"] == "next"), None)
        params = None
    return gpd.pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame()


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching England flood zones for bbox {ENGLAND_BBOX} ...")
    gdf = fetch(ENGLAND_BBOX)
    print(f"\nTotal features fetched: {len(gdf)}")
    print(gdf.groupby(["flood_zone", "flood_source", "origin"]).size())

    sea_modelled = gdf.query("flood_source == 'sea' and origin == 'modelled'")
    for zone, name in [("FZ3", "fz3_sea_modelled"), ("FZ2", "fz2_sea_modelled")]:
        subset = sea_modelled[sea_modelled["flood_zone"] == zone]
        out_path = f"{OUT_DIR}/{name}.gpkg"
        subset.to_file(out_path, driver="GPKG")
        print(f"wrote {out_path}: {len(subset)} feature(s)")


if __name__ == "__main__":
    main()
