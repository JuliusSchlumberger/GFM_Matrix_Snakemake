"""Download the Environment Agency's "Flood Map for Planning: Flood Zones 2
& 3 (Rivers and Sea)" dataset for England, filtered to the coastal-relevant,
modelled-origin subset this framework's coastal validation needs, and save
it alongside Wales/Scotland's own downloaded benchmarks.

`flood_source` is a free-text field, not a clean sea/river enum (confirmed
2026-10-16 by sampling a real cached tile: 'river', 'sea', 'river and sea',
'unknown' all occur) - 'river and sea' is the estuarine/tidal-river class,
genuinely coastal (tidally/storm-surge driven) even though it also carries
fluvial influence. An earlier version of this script filtered to
flood_source=='sea' only, which silently dropped every estuary/tidal-river
reach - exactly the stretches a coastal model's own boundary forcing would
plausibly flood too. Fixed 2026-10-16 (user-caught) to also keep 'river and
sea'; plain 'river' and 'unknown' remain excluded (not comparable to this
model's pure coastal-storm-surge physics).

Flood Zone 3 (sea source): >=1-in-200-year (0.5% AEP) annual probability -
no exact native model RP (model has RP100/RP250, not RP200) - bracket it with
RP100/RP250 (see `comparison_return_periods` on england_ea_floodzone3_sea in
data_catalog_validation.yml).
Flood Zone 2 (sea source): >=1-in-1000-year (0.1% AEP) - matches the model's
own native RP1000 exactly, no bracketing needed.

England's real coastal extent (not all of GBR - Wales/Scotland have their
own separate agencies/datasets) is restricted at validate_country.py's own
scoring time via geogunit_ids, same mechanism as Wales(3367)/Scotland(3366);
this script's own bbox is deliberately generous (doesn't need to be exact).

Usage:
    python validation/fetch_england_flood_zones.py [--config path/to/config.yml]
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import load_config  # noqa: E402

BASE = (
    "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-flood-zones/"
    "ogc/features/v1/collections/Flood_Zones_2_3_Rivers_and_Sea/items"
)
# England's real extent (lon/lat, CRS84) - generous, geogunit_ids does the real restriction.
ENGLAND_BBOX = (-6.5, 49.8, 2.0, 55.9)
TILE_DEG = 1.0  # smaller per-request bbox - easier on the server, clear incremental progress
_SEA_SOURCES = ["sea", "river and sea"]  # see module docstring - excludes plain "river"/"unknown"


def _tiles(bbox, step):
    minx, miny, maxx, maxy = bbox
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            yield (x, y, min(x + step, maxx), min(y + step, maxy))
            y += step
        x += step


def _get_with_retry(url, params, max_attempts=6, delay=10):
    # The API's own "next" pagination links redirect to a different backend
    # host (confirmed live: api-k8s-dsp-prod.agrimetrics.co.uk) that hit a
    # transient DNS resolution failure mid-run - retry rather than let one
    # flaky page kill the whole tiled fetch (per-tile caching already makes
    # a full restart safe, but there's no reason to lose an in-progress
    # tile's own later pages over a passing blip).
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_attempts:
                raise
            print(f"    [retry {attempt}/{max_attempts}] {e.__class__.__name__}: {e} - retrying in {delay}s...", flush=True)
            time.sleep(delay)


def fetch_bbox(bbox, page=2000):
    url = BASE
    params = {"f": "json", "bbox": ",".join(map(str, bbox)), "limit": page}
    parts = []
    while url:
        js = _get_with_retry(url, params)
        if js["features"]:
            parts.append(gpd.GeoDataFrame.from_features(js["features"], crs=4326))
        url = next((link["href"] for link in js.get("links", []) if link["rel"] == "next"), None)
        params = None
    return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame()


def _read_cached_tile(cache_path: str, local_scratch: str) -> gpd.GeoDataFrame:
    """Read a cached tile via a LOCAL staged copy, not a direct P:\\ open.

    Confirmed 2026-10-16: reading these GeoPackages (several are 100-450 MB)
    directly off the P:\\ network share took over 2 hours to get through
    46/63 tiles before being killed - the same GDAL/SQLite-over-network-share
    pattern diagnosed earlier this session on Hydrax (SQLite does many small
    random reads, each paying a full network round trip; a plain bulk byte
    copy does not hit that path). Staging via shutil.copy2 first, then
    reading the local copy, is the same fix that solved it there.
    """
    local_path = os.path.join(local_scratch, "tile.gpkg")
    shutil.copy2(cache_path, local_path)
    try:
        return gpd.read_file(local_path)
    finally:
        os.remove(local_path)


def main(out_dir: str, cache_dir: str) -> None:
    """Fetch, filter, and write England's coastal-relevant flood zones.

    Parameters
    ----------
    out_dir : str
        Directory the filtered per-zone GeoPackages are written to.
    cache_dir : str
        Directory used to cache each tile's raw (unfiltered) download, so a
        re-run only has to fetch tiles that failed or are missing.
    """
    os.makedirs(cache_dir, exist_ok=True)

    tiles = list(_tiles(ENGLAND_BBOX, TILE_DEG))
    print(f"Fetching England flood zones in {len(tiles)} tile(s) of {TILE_DEG} deg ...", flush=True)

    all_parts = []
    failed_tiles = []
    with tempfile.TemporaryDirectory(prefix="england_fz_") as local_scratch:
        for i, tile in enumerate(tiles):
            cache_path = f"{cache_dir}/tile_{i:03d}.gpkg"
            if os.path.exists(cache_path):
                t0 = time.time()
                gdf = _read_cached_tile(cache_path, local_scratch)
                print(f"[{i + 1}/{len(tiles)}] {tile}: {len(gdf)} feature(s) (cached, staged, {time.time() - t0:.1f}s)", flush=True)
            else:
                t0 = time.time()
                try:
                    gdf = fetch_bbox(tile)
                except requests.exceptions.RequestException as e:
                    print(f"[{i + 1}/{len(tiles)}] {tile}: FAILED after retries ({e.__class__.__name__}) - "
                          f"continuing, re-run this script afterward to pick it up", flush=True)
                    failed_tiles.append(tile)
                    continue
                if not gdf.empty:
                    gdf.to_file(cache_path, driver="GPKG")  # raw/unfiltered - stays reusable for any future filter change
                print(f"[{i + 1}/{len(tiles)}] {tile}: {len(gdf)} feature(s) ({time.time() - t0:.1f}s)", flush=True)
            # Filter to the coastal-relevant subset HERE, before accumulating -
            # keeps memory bounded to the small filtered subset instead of the
            # full ~4GB national raw dataset (confirmed 2026-10-16: holding the
            # raw concatenation grew one process to >10GB RSS).
            if not gdf.empty:
                gdf = gdf[gdf["flood_source"].isin(_SEA_SOURCES) & (gdf["origin"] == "modelled")]
            if not gdf.empty:
                all_parts.append(gdf)

    if failed_tiles:
        print(f"\n{len(failed_tiles)} tile(s) failed and were skipped: {failed_tiles}", flush=True)
        print("Re-run this script to retry just those (everything else is cached).", flush=True)

    gdf = pd.concat(all_parts, ignore_index=True) if all_parts else gpd.GeoDataFrame()
    print(f"\nTotal coastal-relevant features (sea / river and sea, modelled): {len(gdf)}", flush=True)
    if gdf.empty:
        return
    print(gdf.groupby(["flood_zone", "flood_source"]).size(), flush=True)

    for zone, name in [("FZ3", "fz3_sea_modelled"), ("FZ2", "fz2_sea_modelled")]:
        subset = gdf[gdf["flood_zone"] == zone].drop_duplicates(subset="geometry")
        out_path = f"{out_dir}/{name}.gpkg"
        subset.to_file(out_path, driver="GPKG")
        print(f"wrote {out_path}: {len(subset)} feature(s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"),
        help="Path to config.yml (merged with the machine's config_local.yml).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    _out_dir = f"{cfg['validation']['benchmark_root']}/England"
    _cache_dir = f"{_out_dir}/_tile_cache"
    main(_out_dir, _cache_dir)
