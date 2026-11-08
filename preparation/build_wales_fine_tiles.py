"""Build a small, bounded fine-grained tile cluster for Wales's local-input validation.

Wales has no equivalent of England's `coastal_tiles.gpkg` (a natural 5km OS-grid tile
index baked in by England's own WCS download step, `ea_coastal_dtm.py`) -
`wales_dtm_32bit_cog.tif` is one pre-built 45GB COG with no tile index at all. This
script instead tiles a uniform 5km candidate grid directly over one named benchmark
region (`severn_estuary` or `menai_strait`, `wales_nrw_floodzone_seas`'s own `regions`
meta), then runs the same per-cell padding/hop-distance/tile-assignment pipeline as
`build_england_fine_tiles.py`, reusing `connectivity_tiling.py`'s primitives UNCHANGED.

5km cells (not coarser) matter here: the elevation filter below keys off each
candidate's MIN elevation, not its mean - a coarser cell spanning both a low floodable
strip and high never-flooding terrain still has a low MIN and gets kept whole, since
nothing downstream re-applies the filter to a sub-piece. 5km also keeps results directly
comparable to England's own 5km-EA-core tile-count/hop characteristics.

Deliberately scoped to ONE named region at a time (TODO.md: validate the local sample
first, don't commit to a full-country build yet) - there is no "whole Wales" option;
that needs its own country-wide footprint design as a separate, later step.

Usage:
    python build_wales_fine_tiles.py --config <config.yml> --region severn_estuary
"""

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from connectivity_tiling import ConnectivityConfig, Domain, assign_tile_id, compute_hop_distances  # noqa: E402
from tiles import _scan_mask_dir  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BNG = "EPSG:27700"
_CELL_SIZE_M = 5000.0      # see module docstring - matches England's EA-core elevation-filter granularity
_OVERLAP_M = 1000.0        # same per-tile domain padding as England (~7km domains, ~49M cells at 1m)
_REGION_BUFFER_M = 5000.0  # margin added around the named region's own bbox before gridding

# Severn Estuary's own extreme sea levels (ea_coastal_design_sea_levels, "UK MAINLAND" series)
# are the SAME stations that set England's own _MAX_WATERLEVEL_M - confirmed this session by
# directly querying stations within 30km of the severn_estuary bbox (max t250 9.12 m, max
# c2_t250 9.54 m - identical to build_england_fine_tiles.py's own derivation). Reused directly
# here, not re-derived. menai_strait has NOT been checked - re-query before using this
# constant for that region.
_MAX_WATERLEVEL_M = 10.0


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


def _real_wales_cells(
    cog_path: Path, region_bbox_bng: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """5km candidate cells over the region's bbox with real COG coverage that could flood.

    No on-disk tile index exists for Wales (module docstring), so validity is checked by
    reading each candidate's own windowed pixels directly from `wales_dtm_32bit_cog.tif` (a
    COG, so this is a cheap tiled+overview-aware read, not a full-file scan) rather than
    England's per-tile nodata-tag check. The elevation floor uses the exact min of the pixels
    actually read (plain numpy min - no GDAL approximate-stats subsampling, same exactness
    requirement as England's `ComputeStatistics(0)` call).
    """
    minx, miny, maxx, maxy = region_bbox_bng
    candidates = _candidate_grid_cells(
        minx - _REGION_BUFFER_M, miny - _REGION_BUFFER_M, maxx + _REGION_BUFFER_M, maxy + _REGION_BUFFER_M,
        _CELL_SIZE_M,
    )
    ds = gdal.Open(str(cog_path))
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
    print(f"  _real_wales_cells: {len(candidates)} candidate 5km cell(s), {n_no_coverage} dropped "
          f"(no valid COG coverage), {n_above_max_waterlevel} dropped (min elevation > "
          f"{_MAX_WATERLEVEL_M} m - can never flood), {len(kept)} kept")
    return kept


def run(config: dict, region: str) -> None:
    root = Path(config["paths"]["root"])
    cog_path = root / "inputs" / "validation" / "Wales" / "new_inputs" / "lidar_dem" / "wales_dtm_32bit_cog.tif"

    val_cfg = config["validation"]
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])
    spec = v.load_benchmark_spec(bench_catalog, "wales_nrw_floodzone_seas")
    if not spec.regions or region not in spec.regions:
        raise ValueError(f"region {region!r} not in wales_nrw_floodzone_seas's regions: {sorted(spec.regions or {})}")
    region_bbox_4326 = spec.regions[region]
    region_bbox_bng = tuple(gpd.GeoSeries([shapely_box(*region_bbox_4326)], crs="EPSG:4326").to_crs(_BNG).total_bounds)

    cells = _real_wales_cells(cog_path, region_bbox_bng)
    if not cells:
        raise ValueError(f"no real LIDAR coverage/floodable cells found for region {region!r}")

    domains = []
    for minx, miny, maxx, maxy in cells:
        padded_bbox_bng = shapely_box(minx - _OVERLAP_M, miny - _OVERLAP_M, maxx + _OVERLAP_M, maxy + _OVERLAP_M)
        padded_bounds_deg = gpd.GeoSeries([padded_bbox_bng], crs=_BNG).to_crs("EPSG:4326").total_bounds
        domains.append(Domain(
            bbox=tuple(padded_bounds_deg), split_reason="wales_5km_cell_plus_overlap",
            approx_native_cells=(maxx - minx + 2 * _OVERLAP_M) * (maxy - miny + 2 * _OVERLAP_M),
            component_id=0,
        ))

    catalog_paths = config["paths"]["hydromt_data_catalog"]
    if isinstance(catalog_paths, str):
        catalog_paths = [catalog_paths]
    catalog = get_data_catalog([_REPO_ROOT / p for p in catalog_paths], root=config["paths"]["root"])
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
    tiles_out = out_dir / f"wales_fine_tiles_{region}.gpkg"
    retry_transient_io(final_gdf.to_file, tiles_out, driver="GPKG")
    print(f"wrote {tiles_out} ({len(final_gdf)} tiles)")

    manifest_path = out_dir / f"wales_fine_dem_source_manifest_{region}.csv"

    def _write_manifest(f) -> None:
        f.write("tile_id,dem_source\n")
        for tid in sorted(final_gdf["tile_id"]):
            f.write(f"{tid},wales_lidar_dem\n")

    atomic_write(manifest_path, _write_manifest, encoding="utf-8", newline="")
    print(f"wrote {manifest_path} ({len(final_gdf)} wales_lidar_dem)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--region", required=True, choices=["severn_estuary", "menai_strait"],
                        help="one of wales_nrw_floodzone_seas's named regions - no whole-country option yet")
    args = parser.parse_args()
    run(load_config(args.config), args.region)


if __name__ == "__main__":
    main()
