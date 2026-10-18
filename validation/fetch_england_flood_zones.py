"""Download England's EA Flood Zones 2 & 3, filtered to modelled coastal sources.

Keeps flood_source 'sea' and 'river and sea' (estuarine/tidal-river, surge-driven);
drops 'river' and 'unknown'. FZ3: >=1-in-200yr (0.5% AEP), bracketed by RP100/RP250.
FZ2: >=1-in-1000yr (0.1% AEP). The bbox is generous; validate_country.py restricts
to England via geogunit_ids.

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
# lon/lat, CRS84
ENGLAND_BBOX = (-6.5, 49.8, 2.0, 55.9)
TILE_DEG = 1.0
_SEA_SOURCES = ["sea", "river and sea"]


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
    # "next" pagination links redirect to another backend host
    # (api-k8s-dsp-prod.agrimetrics.co.uk) with transient DNS failures.
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
    """Fetch all features in a lon/lat bbox, following pagination."""
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
    # SQLite's many small random reads are very slow over the P:\ network share;
    # a bulk copy to local disk first is much faster.
    local_path = os.path.join(local_scratch, "tile.gpkg")
    shutil.copy2(cache_path, local_path)
    try:
        return gpd.read_file(local_path)
    finally:
        os.remove(local_path)


def main(out_dir: str, cache_dir: str) -> None:
    """Fetch, filter and write the per-zone GeoPackages.

    Raw tiles are cached unfiltered in `cache_dir`; a re-run fetches only missing ones.
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
                    gdf.to_file(cache_path, driver="GPKG")
                print(f"[{i + 1}/{len(tiles)}] {tile}: {len(gdf)} feature(s) ({time.time() - t0:.1f}s)", flush=True)
            # Filter per tile: the raw national dataset (~4GB) grows RSS past 10GB.
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
