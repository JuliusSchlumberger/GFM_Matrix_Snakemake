"""Extract water level boundary points for a single tile, return period and SLR scenario."""

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boundaries import load_waterlevel_stations, save_boundary_points, select_stations_for_tile  # noqa: E402
from config_utils import retry_transient_io  # noqa: E402

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
selected = select_stations_for_tile(stations, tile, buffer_deg=bc_cfg["station_search_buffer_deg"])  # noqa: F821

save_boundary_points(selected, snakemake.output.boundaries)  # noqa: F821
