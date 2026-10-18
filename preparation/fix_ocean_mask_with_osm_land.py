"""Reclassify ocean_code mask cells inside OSM land polygons to river_code.

DeltaDTM miscodes inland water (river confluences, aquaculture ponds,
thermokarst lakes) as ocean, e.g. Nakhon Sawan cells at ~+10 m with only ~10%
DEM nodata vs ~99.8% offshore. OSM land is threshold-free ground truth: a cell
inside it cannot be open ocean. Edits the per-tile mask .tifs in place; tiles
with no change keep their mtime to avoid downstream reruns.

Entry point: run(config) as run_preparation.py's fix_ocean_mask step (after
sync_deltadtm, before build_deltadtm_vrt); __main__ for ad hoc dry runs.
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
    # Load once: the shapefile has no spatial-index sidecar, so per-tile bbox
    # reads would linear-scan all 873k features ~7400 times.
    print(f"Loading OSM land polygons from {land_path} ...")
    gdf = retry_transient_io(gpd.read_file, land_path)
    _ = gdf.sindex  # force index build now, not on the first tile query
    print(f"  {len(gdf)} feature(s) loaded, spatial index built.")
    return gdf


def _correct_tile(
    tile_path: Path, land_gdf: gpd.GeoDataFrame, ocean_code: int, river_code: int, dry_run: bool,
) -> int:
    with retry_transient_io(rasterio.open, tile_path) as src:
        profile = src.profile
        transform = src.transform
        bounds = src.bounds
        arr = src.read(1)

    candidate_idx = land_gdf.sindex.query(box(*bounds), predicate="intersects")
    if len(candidate_idx) == 0:
        return 0

    land_raster = rasterize(
        ((geom, 1) for geom in land_gdf.geometry.iloc[candidate_idx]),
        out_shape=arr.shape, transform=transform, fill=0, dtype=np.uint8, all_touched=False,
    )
    to_fix = (arr == ocean_code) & (land_raster == 1)
    n_fix = int(to_fix.sum())
    if n_fix == 0:
        return 0  # write nothing so untouched tiles keep their mtime

    print(f"  {tile_path.name}: {n_fix} cell(s) ocean_code -> river_code" + (" (dry run)" if dry_run else ""))
    if not dry_run:
        corrected = arr.copy()
        corrected[to_fix] = river_code
        _atomic_raster_write(tile_path, profile, lambda dst: dst.write(corrected, indexes=1))
    return n_fix


def run(config: dict, tile_filter: set[str] | None = None, dry_run: bool = False) -> None:
    """Reclassify ocean_code cells inside OSM land polygons in every mask tile.

    tile_filter restricts to these tile filenames (ad hoc testing).
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
    parser.add_argument("--mask-files", default=None, help="comma-separated mask .tif filenames to restrict to (ad hoc testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_filter = set(args.mask_files.split(",")) if args.mask_files else None
    run(cfg, tile_filter=tile_filter, dry_run=args.dry_run)
