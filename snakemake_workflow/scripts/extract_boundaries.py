"""Extract water level boundary points for a single tile, return period and SLR scenario."""

import sys
from pathlib import Path

import geopandas as gpd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from boundaries import (  # noqa: E402
    filter_stations_by_ocean_connectivity,
    save_boundary_points,
    select_stations_for_tile,
    station_search_bounds,
)
from config_utils import retry_transient_io  # noqa: E402

bc_cfg = snakemake.params.bc_cfg  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

tile = retry_transient_io(gpd.read_file, snakemake.input.tile_geometry)  # noqa: F821

search = {"buffer_deg": bc_cfg["station_search_buffer_deg"],
          "min_search_size_deg": bc_cfg["station_search_min_size_deg"]}
# Pre-parsed once per scenario by cache_waterlevel_stations. The cache holds every station of the
# product (~480k), so only the search box is read, through the GeoPackage's spatial index. That
# read returns rows in index order and leaves an empty result untyped: restore file order and the
# cache's dtypes, as a full read gives them.
cache_path = snakemake.input.stations_cache  # noqa: F821
cache_info = pyogrio.read_info(cache_path)
stations = (retry_transient_io(pyogrio.read_dataframe, cache_path, bbox=station_search_bounds(tile, **search),
                               fid_as_index=True)
            .sort_index()
            .astype(dict(zip(cache_info["fields"], cache_info["dtypes"]))))
selected = select_stations_for_tile(stations, tile, **search)

# select_stations_for_tile is distance-only and cannot see thin land barriers;
# this coarse long-range ocean-connectivity pass removes such stations.
mask_dir = Path(snakemake.params.mask_dir)  # noqa: F821
selected = filter_stations_by_ocean_connectivity(
    selected,
    tile,
    mask_dir=mask_dir,
    target_resolution_m=bc_cfg["connectivity_resolution_m"],
    water_fraction_threshold=bc_cfg["connectivity_water_fraction_threshold"],
)

# No station found: write an explicit empty placeholder (`selected` keeps its
# schema/crs with zero rows). Station selection does not consult lower-hop
# neighbours; hinterland forcing from them happens only at run time.
if selected.empty:
    print(f"  no COAST-RP station found for this tile/scenario - writing an empty "  # noqa: T201
          f"placeholder boundaries file (tile_id={snakemake.wildcards.tile_id})")  # noqa: F821

save_boundary_points(selected, snakemake.output.boundaries, column_name=waterlevel_name)  # noqa: F821
