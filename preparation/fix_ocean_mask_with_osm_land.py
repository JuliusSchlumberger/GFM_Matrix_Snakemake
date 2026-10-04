"""Correct ocean_code mask misclassification using real OSM land polygons.

DeltaDTM's mask raster regularly miscodes inland water (wide river
confluences, aquaculture ponds/paddies, thermokarst lakes - the retired
pre-2026-10 pipeline's river_mouth_min_coastal_component_cells speckle
filter was a previously-documented, narrower symptom of the same root
cause) as ocean_code even far from any real coast. Confirmed directly on the Nakhon
Sawan, Thailand four-river confluence (2026-10, connectivity-tiling
prototype investigation): cells there carry mask==ocean_code but sit at
real DEM elevations around +10m, and only ~10% of them carry the DEM's own
nodata sentinel (-9999) - vs. ~99.8% for a genuinely offshore ocean_code
cell checked the same way. That elevation/nodata evidence is informative
but not used here as the correction rule itself, because it needs a
threshold choice; this script instead uses real ground-truth coastline
geometry (OSM land polygons), which needs none: a cell strictly inside an
OSM land polygon can never be real open ocean, regardless of which
DeltaDTM mask sub-code it carries, since OSM's land/sea boundary IS the
coastline by construction.

Corrects the ORIGINAL per-tile mask .tif files themselves (not just the VRT
mosaic built over them) - wherever a cell is mask==ocean_code AND falls
inside a real OSM land polygon, it is reclassified to river_code. A tile
with zero affected cells is left untouched on disk (mtime unchanged) -
deliberately, to avoid cascading unrelated reruns downstream the way a
blanket rewrite of all ~7400 tiles would (see the Wales/Scotland
friction=9 re-run investigation, 2026-10, for a real instance of exactly
that cascade risk from unnecessary mtime churn upstream).

Not a standalone entry point - exposes `run(config, ...)`, called from
run_preparation.py as the `fix_ocean_mask` step (runs AFTER sync_deltadtm,
BEFORE build_deltadtm_vrt, so the VRT mosaic is built from the already-
corrected tiles). Can also be run directly for ad hoc testing/dry runs -
see the `__main__` block at the bottom.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402
from rasters import _atomic_raster_write  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_land_index(land_path: Path) -> gpd.GeoDataFrame:
    """Read the full OSM land-polygons dataset once and return it with its
    spatial index already built (`.sindex` is built lazily on first access -
    triggering it here means the first per-tile query isn't the one that
    pays for it, and makes the one-time cost visible in the log).

    Loaded once per `run()` call, not per tile - 873k features is cheap to
    hold in memory and query by index compared to re-reading/re-indexing it
    ~7400 times (once per DeltaDTM mask tile), and the plain shapefile here
    has no .qix/.sbn spatial-index sidecar, so a per-tile bbox-filtered
    `read_file(..., bbox=...)` would otherwise linear-scan all 873k
    features' bounding boxes on every single tile.
    """
    print(f"Loading OSM land polygons from {land_path} ...")
    gdf = retry_transient_io(gpd.read_file, land_path)
    _ = gdf.sindex  # force index build now, not on the first tile query
    print(f"  {len(gdf)} feature(s) loaded, spatial index built.")
    return gdf


def _correct_tile(
    tile_path: Path, land_gdf: gpd.GeoDataFrame, ocean_code: int, river_code: int, dry_run: bool,
) -> int:
    """Reclassify ocean_code cells inside OSM land polygons to river_code in
    one mask tile. Returns the number of cells reclassified (0 if the tile
    is left untouched). Writes nothing at all if n_fix == 0, even when
    dry_run=False - no-op tiles must never have their mtime touched."""
    with retry_transient_io(rasterio.open, tile_path) as src:
        profile = src.profile
        transform = src.transform
        bounds = src.bounds
        arr = src.read(1)

    candidate_idx = land_gdf.sindex.query(box(*bounds), predicate="intersects")
    if len(candidate_idx) == 0:
        return 0  # no land anywhere near this tile - nothing to correct

    land_raster = rasterize(
        ((geom, 1) for geom in land_gdf.geometry.iloc[candidate_idx]),
        out_shape=arr.shape, transform=transform, fill=0, dtype=np.uint8, all_touched=False,
    )
    to_fix = (arr == ocean_code) & (land_raster == 1)
    n_fix = int(to_fix.sum())
    if n_fix == 0:
        return 0

    print(f"  {tile_path.name}: {n_fix} cell(s) ocean_code -> river_code" + (" (dry run)" if dry_run else ""))
    if not dry_run:
        corrected = arr.copy()
        corrected[to_fix] = river_code
        _atomic_raster_write(tile_path, profile, lambda dst: dst.write(corrected, indexes=1))
    return n_fix


def run(config: dict, tile_filter: set[str] | None = None, dry_run: bool = False) -> None:
    """Correct every DeltaDTM mask tile in place (unless dry_run=True).

    `tile_filter`, if given, restricts the run to tiles whose filename is in
    the set (ad hoc testing - not used by run_preparation.py's normal call).
    """
    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    mask_dir = Path(catalog.get_source("deltadtm_mask").path).parent
    land_path = Path(catalog.get_source("land_polygons").path)
    ocean_code = config["tile_generation"]["ocean_code"]
    river_code = config["tile_generation"]["river_code"]

    print("=== Fixing ocean_code mask misclassification against OSM land polygons ===")
    print(f"Mask tiles: {mask_dir}")
    print(f"OSM land polygons: {land_path}")
    print(f"ocean_code={ocean_code} -> river_code={river_code}" + ("  [DRY RUN - no files written]" if dry_run else ""))

    land_gdf = _load_land_index(land_path)

    tile_paths = sorted(mask_dir.glob("*.tif"))
    if tile_filter is not None:
        tile_paths = [p for p in tile_paths if p.name in tile_filter]
    print(f"\nScanning {len(tile_paths)} mask tile(s) ...")

    n_tiles_scanned = 0
    n_tiles_modified = 0
    n_cells_total = 0
    for tile_path in tile_paths:
        n_tiles_scanned += 1
        n_fix = _correct_tile(tile_path, land_gdf, ocean_code, river_code, dry_run)
        if n_fix:
            n_tiles_modified += 1
            n_cells_total += n_fix

    print("\nDone.")
    print(f"  Tiles scanned:  {n_tiles_scanned}")
    print(f"  Tiles with misclassified ocean cells found: {n_tiles_modified}")
    print(f"  Total cells reclassified ocean_code -> river_code: {n_cells_total}")
    if dry_run and n_tiles_modified:
        print("  (dry run - no files were actually written)")


if __name__ == "__main__":
    import argparse

    from config_utils import load_config  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--tiles", default=None, help="comma-separated mask .tif filenames to restrict to (ad hoc testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_filter = set(args.tiles.split(",")) if args.tiles else None
    run(cfg, tile_filter=tile_filter, dry_run=args.dry_run)
