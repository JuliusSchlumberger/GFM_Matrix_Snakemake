"""Build a small, bounded fine-grained tile cluster for SF Bay's local-input validation.

SF Bay has no catalog `regions` dict to borrow a named test area from the way Wales did
(`wales_nrw_floodzone_seas`'s severn_estuary/menai_strait) - there is no SF Bay entry in
data_catalog_validation.yml at all (validation/validate_sf_bay.py is a standalone path).
Alameda County is used instead (user, 2026-11-06): a per-county CoSMoS depth raster is
already staged at
`{root}/inputs/validation/California/alameda_flooding_slr000.zip` - its own bounds
(read directly via rasterio's zip:// VFS, no extraction needed) are the candidate-grid
footprint here, in place of a catalog bbox.

Otherwise follows build_wales_fine_tiles.py exactly: a uniform 5km candidate grid (see
that script's own docstring for why 5km, not coarser), a per-cell real-coverage + exact-
min-elevation filter against the DEM (the catalog's `sf_bay_lidar_dem` entry, EPSG:26910 -
no reprojection needed between the benchmark's own CRS and the DEM's, unlike Wales's
BNG<->4326 step), then the same hop-distance/tile-assignment pipeline from
connectivity_tiling.py, unchanged.

`_MAX_WATERLEVEL_M` here is derived from COAST-RP (not a local design-levels product - SF
Bay's boundary stays global, datasource.txt): max storm_tide_rp_0250 among COAST-RP stations
within the Bay Area bbox is 1.725 m (queried directly this session), vs England/Wales's
~9.1-9.5 m (Severn/Bristol Channel has one of the world's largest tidal ranges) - SF Bay's
real levels are far smaller, so a looser 5.0 m ceiling (comfortably covers MDT + any local
amplification) is used rather than reusing England/Wales's 10.0 m by rote.

Usage:
    python build_sf_bay_fine_tiles.py --config <config.yml> --region alameda
"""

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from osgeo import gdal
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from connectivity_tiling import ConnectivityConfig, Domain, assign_tile_id, compute_hop_distances  # noqa: E402
from tiles import _scan_mask_dir  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SF_BAY_CRS = "EPSG:26910"
_CELL_SIZE_M = 5000.0
_OVERLAP_M = 1000.0

# See module docstring for derivation.
_MAX_WATERLEVEL_M = 5.0

_ALAMEDA_BENCHMARK_ZIP_MEMBER = "flooding/v2.1/county_alameda_flddepth_slr000_w100.tif"
_REGIONS = {
    "alameda": lambda root: (
        f"zip://{root}/inputs/validation/California/alameda_flooding_slr000.zip!{_ALAMEDA_BENCHMARK_ZIP_MEMBER}"
    ),
}


def _candidate_grid_cells(
    minx: float, miny: float, maxx: float, maxy: float, cell_size_m: float,
) -> list[tuple[float, float, float, float]]:
    """Uniform cell_size_m grid covering a bbox, snapped to a multiple of cell_size_m."""
    x0 = math.floor(minx / cell_size_m) * cell_size_m
    y0 = math.floor(miny / cell_size_m) * cell_size_m
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + cell_size_m, y + cell_size_m))
            y += cell_size_m
        x += cell_size_m
    return cells


def _real_sf_bay_cells(
    dem_path: Path, region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """5km candidate cells over the region's bbox with real DEM coverage that could flood.

    Same exactness requirement as build_wales_fine_tiles.py's own filter (plain numpy min
    over the pixels actually read, no GDAL approximate-stats subsampling).
    """
    minx, miny, maxx, maxy = region_bbox
    candidates = _candidate_grid_cells(minx, miny, maxx, maxy, _CELL_SIZE_M)
    ds = gdal.Open(str(dem_path))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    px_size = gt[1]
    kept = []
    n_no_coverage = n_above_max_waterlevel = 0
    for cminx, cminy, cmaxx, cmaxy in candidates:
        xoff = int((cminx - gt[0]) / px_size)
        yoff = int((gt[3] - cmaxy) / px_size)
        xsize = int((cmaxx - cminx) / px_size)
        ysize = int((cmaxy - cminy) / px_size)
        if xoff < 0 or yoff < 0 or xoff + xsize > ds.RasterXSize or yoff + ysize > ds.RasterYSize:
            n_no_coverage += 1
            continue
        arr = band.ReadAsArray(xoff, yoff, xsize, ysize)
        valid = arr != nodata
        if not valid.any():
            n_no_coverage += 1
            continue
        min_elevation_m = float(arr[valid].min())
        if min_elevation_m > _MAX_WATERLEVEL_M:
            n_above_max_waterlevel += 1
            continue
        kept.append((cminx, cminy, cmaxx, cmaxy))
    print(f"  _real_sf_bay_cells: {len(candidates)} candidate 5km cell(s), {n_no_coverage} dropped "
          f"(no valid DEM coverage), {n_above_max_waterlevel} dropped (min elevation > "
          f"{_MAX_WATERLEVEL_M} m - can never flood), {len(kept)} kept")
    return kept


def run(config: dict, region: str) -> None:
    root = Path(config["paths"]["root"])

    catalog_paths = config["paths"]["hydromt_data_catalog"]
    if isinstance(catalog_paths, str):
        catalog_paths = [catalog_paths]
    catalog = get_data_catalog([_REPO_ROOT / p for p in catalog_paths], root=config["paths"]["root"])
    dem_path = Path(catalog["sf_bay_lidar_dem"].path)

    benchmark_path = _REGIONS[region](root)
    with rasterio.open(benchmark_path) as src:
        if str(src.crs) != _SF_BAY_CRS:
            raise ValueError(f"{region} benchmark raster CRS {src.crs} != expected {_SF_BAY_CRS}")
        region_bbox = tuple(src.bounds)
    print(f"region {region!r} bbox ({_SF_BAY_CRS}): {region_bbox}")

    cells = _real_sf_bay_cells(dem_path, region_bbox)
    if not cells:
        raise ValueError(f"no real LIDAR coverage/floodable cells found for region {region!r}")

    domains = []
    for minx, miny, maxx, maxy in cells:
        padded_bbox = shapely_box(minx - _OVERLAP_M, miny - _OVERLAP_M, maxx + _OVERLAP_M, maxy + _OVERLAP_M)
        padded_bounds_deg = gpd.GeoSeries([padded_bbox], crs=_SF_BAY_CRS).to_crs("EPSG:4326").total_bounds
        domains.append(Domain(
            bbox=tuple(padded_bounds_deg), split_reason="sf_bay_5km_cell_plus_overlap",
            approx_native_cells=(maxx - minx + 2 * _OVERLAP_M) * (maxy - miny + 2 * _OVERLAP_M),
            component_id=0,
        ))

    mask_dir = Path(catalog["deltadtm_mask"].path).parent
    mask_index = _scan_mask_dir(mask_dir)

    tg_cfg = config["tile_generation"]
    conn_cfg = ConnectivityConfig(elev_threshold_m=tg_cfg["elev_threshold_m"], ocean_code=tg_cfg["ocean_code"])

    hop, unreachable = compute_hop_distances(domains, mask_index, conn_cfg)
    n_hop0 = sum(1 for h in hop if h == 0)
    print(f"hop distances: hop=0 {n_hop0}/{len(domains)}, unreachable (dropped) {len(unreachable)}")
    if unreachable:
        for i in unreachable:
            print(f"  dropped (no path to ocean): cell at {cells[i]}")

    unreachable_set = set(unreachable)
    final_domains = [d for i, d in enumerate(domains) if i not in unreachable_set]
    final_hop = [h for i, h in enumerate(hop) if i not in unreachable_set]
    final_gdf = assign_tile_id(final_domains, final_hop)

    out_dir = Path(config["paths"]["processed_inputs_dir"]) / "mask" / "validation"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    tiles_out = out_dir / f"sf_bay_fine_tiles_{region}.gpkg"
    retry_transient_io(final_gdf.to_file, tiles_out, driver="GPKG")
    print(f"wrote {tiles_out} ({len(final_gdf)} tiles)")

    manifest_path = out_dir / f"sf_bay_fine_dem_source_manifest_{region}.csv"

    def _write_manifest(f) -> None:
        f.write("tile_id,dem_source\n")
        for tid in sorted(final_gdf["tile_id"]):
            f.write(f"{tid},sf_bay_lidar_dem\n")

    atomic_write(manifest_path, _write_manifest, encoding="utf-8", newline="")
    print(f"wrote {manifest_path} ({len(final_gdf)} sf_bay_lidar_dem)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--region", required=True, choices=list(_REGIONS),
                        help="hardcoded test area - no catalog regions dict exists for SF Bay")
    args = parser.parse_args()
    run(load_config(args.config), args.region)


if __name__ == "__main__":
    main()
