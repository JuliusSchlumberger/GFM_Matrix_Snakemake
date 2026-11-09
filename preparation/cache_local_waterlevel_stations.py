"""Cache EA's GB-wide design sea levels as a local-input scenario's SLR_0 stations.

`ea_coastal_design_sea_levels` is reused unchanged for England/Wales/Scotland - the
`--local-dem-key` argument says which catalog entry's `native_datum`/`hybrid_grid_raster_path`
to bridge the station levels through (see `run()`'s own comment on why this mirrors the DEM's
own native_datum rather than hardcoding a bridge here). For England and Wales both legs are
native ODN already (native_datum="native" in data_catalog_local_input.yml - TODO.md), so
bridging would only add the geoid grids' own resampling noise to freeboard, confirmed
empirically to cancel out anyway.

Scotland has no local DEM to key off (its local-input scenario keeps the global DeltaDTM
DEM and only swaps the boundary) - `--boundary-native-datum` bridges the stations directly,
bypassing the `--local-dem-key` lookup entirely, since the stations' own native datum (ODN)
has nothing to do with whatever datum a paired DEM happens to use.

Writes the station caches under boundary_conditions.stations_cache_dir (an isolated
directory for this scenario, not the shared global cache every other scenario reads). The
scenario sets boundary_conditions.stations_cache_prebuilt, so the workflow takes these files
as inputs and never rebuilds them from the global levels. Run once, manually, before
`snakemake preprocess|simulate|all` on the scenario's materialized config; rerun it if the EA
shapefile changes.

Usage:
    python cache_local_waterlevel_stations.py --config <england_local_input_materialized.yml>
    python cache_local_waterlevel_stations.py --config <wales_local_input_materialized.yml> \\
        --local-dem-key wales_lidar_dem
    python cache_local_waterlevel_stations.py --config <scotland_local_input_materialized.yml> \\
        --boundary-native-datum odn_osgm15 \\
        --hybrid-grid-raster-path "{root}/inputs/validation/England/new_inputs/lidar_dem/uk_os_OSGM15_GB.tif"
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from vertical_datum import compute_vertical_datum_offset_at_points  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Present-day only (TODO.md D4): EA's design sea levels are a single present-day estimate,
# with no SLR scenario of their own.
WATERLEVEL_NAME = "SLR_0"

# These 3 `location` groups anchor to their own local chart datum, not ODN (Newlyn) -
# vertical_datum_to_goco06s.md's named exception. Outer Hebrides' and Shetland's reference
# tide gauges are Stornoway and Lerwick respectively; St Mary's (Scilly) is named directly.
_LOCAL_DATUM_EXCLUDED_LOCATIONS = {"OUTER HEBRIDES", "Shetland", "St. Mary's (Scilly)"}


def run(
    config: dict, local_dem_key: str, boundary_native_datum: str | None = None,
    hybrid_grid_raster_path: str | None = None,
) -> None:
    catalog_paths = config["paths"]["hydromt_data_catalog"]
    if isinstance(catalog_paths, str):
        catalog_paths = [catalog_paths]
    catalog = get_data_catalog([_REPO_ROOT / p for p in catalog_paths], root=config["paths"]["root"])

    stations = catalog.get_geodataframe("ea_coastal_design_sea_levels")
    n_total = len(stations)
    stations = stations[~stations["location"].isin(_LOCAL_DATUM_EXCLUDED_LOCATIONS)].reset_index(drop=True)
    print(f"ea_coastal_design_sea_levels: {len(stations)}/{n_total} points kept "
          f"({n_total - len(stations)} excluded - local chart datum, not ODN)")

    if boundary_native_datum is not None:
        # No local DEM to key off (Scotland: global DeltaDTM DEM, only the boundary is
        # local) - bridge the stations' own native datum (ODN) directly. odn_osgm15 still
        # needs the GB-wide OSGM15 grid raster; --hybrid-grid-raster-path supplies it since
        # there is no local DEM catalog entry to read it from.
        native_datum = boundary_native_datum
        raw_hybrid_grid_raster_path = hybrid_grid_raster_path
    else:
        # Mirrors local_dem_key's own native_datum (data_catalog_local_input.yml) rather than
        # hardcoding a bridge here, so the DEM and boundary legs can never drift apart: "native"
        # (England's and Wales's current case) is an explicit no-op - their DEM and boundary
        # levels are both already-native ODN and mutually consistent, so bridging both would only
        # add the geoid grids' own resampling noise to freeboard (confirmed empirically, TODO.md).
        dem_source = catalog.get_source(local_dem_key)
        native_datum = dem_source.meta.get("native_datum", "egm2008")
        raw_hybrid_grid_raster_path = dem_source.meta.get("hybrid_grid_raster_path")
    vc_cfg = config["vertical_datum_correction"]

    stations_4326 = stations.to_crs("EPSG:4326")
    offset_m = compute_vertical_datum_offset_at_points(
        native_datum,
        stations_4326.geometry.x.to_numpy(),
        stations_4326.geometry.y.to_numpy(),
        geoid_offset_raster=vc_cfg["offset_raster_path"],  # unused by odn_osgm15/native, required positionally
        hybrid_grid_raster=raw_hybrid_grid_raster_path.format(root=config["paths"]["root"]) if raw_hybrid_grid_raster_path else None,
        goco06s_absolute_raster=vc_cfg["goco06s_absolute_raster_path"],
    )
    print(f"{native_datum} offset at {len(offset_m)} stations: "
          f"[{offset_m.min():.3f}, {offset_m.max():.3f}] m, mean {offset_m.mean():.3f} m "
          f"(0 for every station under native_datum=native)")

    out_dir = Path(config["boundary_conditions"]["stations_cache_dir"])
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    for rp in config["boundary_conditions"]["return_periods"]:
        column = f"t{rp}"
        if column not in stations.columns:
            raise KeyError(f"ea_coastal_design_sea_levels has no {column!r} column for return_period={rp}")
        waterlevel_m = stations[column].to_numpy() + offset_m  # native ODN, unchanged, when native_datum=native

        out = gpd.GeoDataFrame(
            {WATERLEVEL_NAME: waterlevel_m}, geometry=stations_4326.geometry, crs="EPSG:4326",
        )
        out_path = out_dir / f"stations_RP{rp}_{WATERLEVEL_NAME}.gpkg"
        retry_transient_io(out.to_file, out_path, driver="GPKG")
        print(f"  RP{rp}: wrote {out_path} ({len(out)} stations, "
              f"[{waterlevel_m.min():.2f}, {waterlevel_m.max():.2f}] m)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-dem-key", default="england_lidar_dem",
                        help="data_catalog_local_input.yml entry whose native_datum/"
                             "hybrid_grid_raster_path bridges the station levels - ignored "
                             "when --boundary-native-datum is given")
    parser.add_argument("--boundary-native-datum", default=None,
                        help="bridge name (vertical_datum.DATUM_BRIDGES) to apply to the "
                             "stations directly, bypassing --local-dem-key - use when there "
                             "is no local DEM sharing a datum with the boundary (Scotland: "
                             "global DeltaDTM DEM + local ODN boundary -> odn_osgm15)")
    parser.add_argument("--hybrid-grid-raster-path", default=None,
                        help="OSGM15 grid raster path, {root}-templated - required with "
                             "--boundary-native-datum odn_osgm15 (unused by egm2008/native)")
    args = parser.parse_args()
    run(load_config(args.config), args.local_dem_key, args.boundary_native_datum, args.hybrid_grid_raster_path)


if __name__ == "__main__":
    main()
