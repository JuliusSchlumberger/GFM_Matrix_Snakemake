"""Build a fine-grained, overlapping tile grid for England's local-input validation.

Production's `tile_grid` is sized for DeltaDTM's 30m global data (~25M cells/tile,
tile_generation.budget_cells) - reusing it unmodified for England's 1m LIDAR DEM
would need ~50-200GB per tile just to read the DEM, and ~1-2 billion solver cells
per tile, neither of which is feasible (TODO.md, local-input validation Section 4
revision). This script instead tiles EA's own real-LIDAR 5km OS-grid cores
directly, pads each by `_OVERLAP_M` on every side (~7km domains, ~49M cells at 1m
- the same per-tile budget philosophy as production, just at 1m instead of 30m),
and computes hop_distance by reusing `connectivity_tiling.compute_hop_distances`
UNMODIFIED against the existing global DeltaDTM mask.

`--production-tile-id` (optional) scopes the cores to one selected production
tile's footprint, for a small, cheap first test (the Isles of Scilly test, e.g.
--production-tile-id 3263 - TODO.md). Omit it to run every real EA core across
all of England in ONE pass - not per-production-tile: production's own tiles
overlap each other substantially by design (confirmed geometrically: 74% of EA
cores fall inside 2+ of the 28 production tiles selected by an earlier,
superseded per-production-tile-closure approach, since deleted - TODO.md),
so looping this script once per production tile would solve most of England 2-3x
over. Running every core together instead gives each one exactly one fine tile
(no deduplication step needed) and lets hop_distance come out >=1 wherever
England's coastline actually needs it (e.g. an estuary), rather than the
single-production-tile test's every-tile-hop=0 result.

Simplifications vs. production's full pipeline (`connectivity_tiling.py`), each
a deliberate scope cut, not a behavioural change to anything reused:
  - No Phase 1/2 floodable-land connectivity graph across these new tiles - every
    tile gets the same `component_id` (0). `compute_hop_distances`'s own adjacency
    test (`_bbox_touches`) still runs unmodified; two tiles that only touch via
    the 1km overlap padding (not real floodable land) could in principle be
    treated as hydraulically connected when they are not - low risk for coastal/
    tidal LIDAR tiles (all directly sea-exposed, unlike inland river networks),
    but not a guarantee.
  - No Phase 3/4 budget-splitting or merging - tile cores are EA's own existing
    5km squares, not recursively split/merged.
  - No Phase 6 hop=0-only coastal pad - every tile is padded uniformly instead
    (the user's own overlap requirement), not just hop=0 ones.

Usage:
    python build_england_fine_tiles.py --config <config.yml>                       # all England
    python build_england_fine_tiles.py --config <config.yml> --production-tile-id 3263  # one tile, cheap test
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from connectivity_tiling import ConnectivityConfig, Domain, assign_tile_id, compute_hop_distances  # noqa: E402
from tiles import _scan_mask_dir  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BNG = "EPSG:27700"
_OVERLAP_M = 1000.0  # padding added to every side of an EA 5km core (~7km domains, ~49M cells at 1m)

# Max England RP250 design sea level (ea_coastal_design_sea_levels, native ODN, the 3 named
# local-datum exceptions excluded): t250 max 9.12 m, 95% CI upper bound (c2_t250) 9.54 m -
# rounded up for a clean safety margin. RP250, not RP1000: TODO.md/this session's validation
# only ever runs RP100/RP250 (D5's "RP100 first" step) - no need to size this against a return
# period this framework never actually solves. A core whose own minimum elevation sits above
# this can never flood under any scenario this framework runs and is dropped before tiling -
# cheaper than the ocean-connectivity check, and before it (no point checking hop_distance for
# a tile that can never flood regardless of ocean adjacency).
_MAX_WATERLEVEL_M = 10.0


def _real_ea_tiles(lidar_dir: Path) -> gpd.GeoDataFrame:
    """EA 5km tiles that are on disk, carry a real nodata tag, and could plausibly flood.

    Drops the degenerate (no-coverage, no nodata tag) WCS tiles, and any tile whose own
    minimum elevation sits above `_MAX_WATERLEVEL_M` - see that constant's own docstring.
    GDAL's exact (non-approximate) ComputeStatistics is used for the minimum: approximate
    stats can use an internal subsample and under-count a real but spatially small low
    point (confirmed empirically: approx min -4.17 m vs exact -5.25 m on one real tile) -
    the wrong direction for a filter whose failure mode must never be "wrongly excluded a
    tile that could actually flood".
    """
    candidates = gpd.read_file(lidar_dir / "coastal_tiles.gpkg")
    on_disk = {p.stem for p in lidar_dir.glob("*.tif")}
    real_names = set()
    n_no_nodata_tag = n_above_max_waterlevel = 0
    for name in on_disk:
        ds = gdal.Open(str(lidar_dir / f"{name}.tif"))
        band = ds.GetRasterBand(1)
        if band.GetNoDataValue() is None:
            n_no_nodata_tag += 1
            continue
        min_elevation_m = band.ComputeStatistics(0)[0]  # approx_ok=0 (exact) - see docstring
        if min_elevation_m > _MAX_WATERLEVEL_M:
            n_above_max_waterlevel += 1
            continue
        real_names.add(name)
    print(f"  _real_ea_tiles: {len(on_disk)} on disk, {n_no_nodata_tag} dropped (no nodata tag - "
          f"degenerate WCS response), {n_above_max_waterlevel} dropped (min elevation > "
          f"{_MAX_WATERLEVEL_M} m - can never flood), {len(real_names)} kept")
    return candidates[candidates["tile"].isin(real_names)]


def run(config: dict, production_tile_id: int | None) -> None:
    root = Path(config["paths"]["root"])
    lidar_dir = root / "inputs" / "validation" / "England" / "new_inputs" / "lidar_dem"

    real_tiles = _real_ea_tiles(lidar_dir)
    if production_tile_id is not None:
        prod_tiles = gpd.read_file(config["tile_grid"]["path"])
        prod_tiles["tile_id"] = prod_tiles["tile_id"].astype(int)
        prod_tile = prod_tiles.loc[prod_tiles["tile_id"] == production_tile_id].to_crs(_BNG)
        if prod_tile.empty:
            raise ValueError(f"production tile_id {production_tile_id} not found in {config['tile_grid']['path']}")
        prod_geom = prod_tile.geometry.iloc[0]
        cores = real_tiles[real_tiles.geometry.intersects(prod_geom)]
        print(f"production tile {production_tile_id}: {len(cores)} real EA 5km tile(s) inside its footprint")
        if cores.empty:
            raise ValueError(f"production tile {production_tile_id} has no real EA LIDAR coverage")
    else:
        cores = real_tiles
        print(f"all England: {len(cores)} real EA 5km tile(s)")

    domains = []
    for _, row in cores.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        padded_bbox_bng = shapely_box(minx - _OVERLAP_M, miny - _OVERLAP_M, maxx + _OVERLAP_M, maxy + _OVERLAP_M)
        padded_bounds_deg = gpd.GeoSeries([padded_bbox_bng], crs=_BNG).to_crs("EPSG:4326").total_bounds
        domains.append(Domain(
            bbox=tuple(padded_bounds_deg), split_reason="ea_5km_core_plus_overlap",
            approx_native_cells=(maxx - minx + 2 * _OVERLAP_M) * (maxy - miny + 2 * _OVERLAP_M),
            component_id=0,
        ))

    catalog_path = _REPO_ROOT / config["paths"]["hydromt_data_catalog"] if isinstance(
        config["paths"]["hydromt_data_catalog"], str,
    ) else [_REPO_ROOT / p for p in config["paths"]["hydromt_data_catalog"]]
    catalog = get_data_catalog(catalog_path, root=config["paths"]["root"])
    mask_dir = Path(catalog["deltadtm_mask"].path).parent
    mask_index = _scan_mask_dir(mask_dir)

    tg_cfg = config["tile_generation"]
    cfg = ConnectivityConfig(elev_threshold_m=tg_cfg["elev_threshold_m"], ocean_code=tg_cfg["ocean_code"])

    hop, unreachable = compute_hop_distances(domains, mask_index, cfg)
    n_hop0 = sum(1 for h in hop if h == 0)
    print(f"hop distances: hop=0 {n_hop0}/{len(domains)}, unreachable (dropped) {len(unreachable)}")
    if unreachable:
        for i in unreachable:
            print(f"  dropped (no path to ocean): {cores.iloc[i]['tile']}")

    unreachable_set = set(unreachable)
    final_domains = [d for i, d in enumerate(domains) if i not in unreachable_set]
    final_hop = [h for i, h in enumerate(hop) if i not in unreachable_set]
    final_gdf = assign_tile_id(final_domains, final_hop)

    # Distinct names per mode: the single-tile test grid's outputs on <data_root> use the same
    # tile_ids for other tiles, so one grid must never overwrite the other. The England
    # scenario configs read the all-England files.
    name_suffix = "" if production_tile_id is not None else "_all_england"
    out_dir = Path(config["paths"]["processed_inputs_dir"]) / "mask" / "validation"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    tiles_out = out_dir / f"england_fine_tiles{name_suffix}.gpkg"
    retry_transient_io(final_gdf.to_file, tiles_out, driver="GPKG")
    print(f"wrote {tiles_out} ({len(final_gdf)} tiles)")

    manifest_path = out_dir / f"england_fine_dem_source_manifest{name_suffix}.csv"

    def _write_manifest(f) -> None:
        f.write("tile_id,dem_source\n")
        for tid in sorted(final_gdf["tile_id"]):
            f.write(f"{tid},england_lidar_dem\n")

    atomic_write(manifest_path, _write_manifest, encoding="utf-8", newline="")
    print(f"wrote {manifest_path} ({len(final_gdf)} england_lidar_dem)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--production-tile-id", type=int, default=None,
                        help="scope to one production tile's footprint (e.g. 3263, the Isles of "
                             "Scilly test); omit to run every real EA core across all of England")
    args = parser.parse_args()
    run(load_config(args.config), args.production_tile_id)


if __name__ == "__main__":
    main()
