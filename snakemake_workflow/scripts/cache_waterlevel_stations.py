"""Cache one (return_period, waterlevel_name) scenario's global water-level
stations once, as a lightweight pre-parsed GeoPackage - see
rules/preprocessing.smk's cache_waterlevel_stations docstring for why."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boundaries import load_waterlevel_stations  # noqa: E402
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

stations = load_waterlevel_stations(
    nc_path,
    variable=variable,
    x_var=bc_cfg["station_x_var"],
    y_var=bc_cfg["station_y_var"],
    column_name=waterlevel_name,
)

out_path = Path(snakemake.output.stations_cache)  # noqa: F821
retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
retry_transient_io(stations.to_file, out_path, driver="GPKG")
