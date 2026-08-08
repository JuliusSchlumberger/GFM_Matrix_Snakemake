"""Extract water level boundary points for a single tile, return period and SLR scenario."""

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boundaries import (  # noqa: E402
    filter_stations_by_ocean_connectivity,
    load_waterlevel_stations,
    save_boundary_points,
    select_stations_for_tile,
)
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402

bc_cfg = snakemake.params.bc_cfg  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

nc_filename = bc_cfg["nc_filename_template"].format(
    return_period=return_period, waterlevel_name=waterlevel_name
)
variable = bc_cfg["nc_variable_template"].format(
    return_period=return_period, waterlevel_name=waterlevel_name
)
nc_path = Path(bc_cfg["waterlevel_nc_dir"]) / nc_filename

tile = retry_transient_io(gpd.read_file, snakemake.input.tile_geometry)  # noqa: F821

stations = load_waterlevel_stations(
    nc_path,
    variable=variable,
    x_var=bc_cfg["station_x_var"],
    y_var=bc_cfg["station_y_var"],
    column_name=waterlevel_name,
)

selected = select_stations_for_tile(
    stations, tile,
    buffer_deg=bc_cfg["station_search_buffer_deg"],
    min_search_size_deg=bc_cfg["station_search_min_size_deg"],
)  # noqa: F821

# Coarse, long-range ocean-connectivity filter - select_stations_for_tile above
# is purely distance-based and can't tell a station across a thin land
# barrier from a genuinely nearby one; see filter_stations_by_ocean_connectivity's
# own docstring for why this has to be a separate, coarser pass rather than
# folded into the native-resolution IDW connectivity fix in flood_model.py.
data_catalog = get_data_catalog(snakemake.params.data_catalog, root=snakemake.params.data_catalog_root)  # noqa: F821
mask_dir = Path(data_catalog["deltadtm_mask"].path).parent
selected = filter_stations_by_ocean_connectivity(
    selected,
    tile,
    mask_dir=mask_dir,
    target_resolution_m=bc_cfg["connectivity_resolution_m"],
    water_fraction_threshold=bc_cfg["connectivity_water_fraction_threshold"],
)

# A tile that finds no real COAST-RP station at all (small/hinterland
# chunks, or genuinely isolated ones) used to be dropped from tile_grid.path
# entirely by the now-retired connectivity_map step. That step is gone, so
# this is now an explicit, intentional EMPTY placeholder instead - `selected`
# already has the right schema/crs even with zero rows, so save_boundary_points
# writes it as-is. Revisit once neighbour-derived boundary forcing (see
# src/tile_chunking.py's compute_run_order / the "NOT IMPLEMENTED" note in
# build_tile_manifest.py's Stage 13) exists to actually fill this in.
if selected.empty:
    print(f"  no COAST-RP station found for this tile/scenario - writing an empty "  # noqa: T201
          f"placeholder boundaries file (tile_id={snakemake.wildcards.tile_id})")  # noqa: F821

save_boundary_points(selected, snakemake.output.boundaries, column_name=waterlevel_name)  # noqa: F821
