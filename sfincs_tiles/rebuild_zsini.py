"""Rebuild ONLY sfincs.ini (zsini) for an already-built SFINCS tile model,
using the corrected land/river/lake-vs-ocean rule (see
build_sfincs_tile.py::_compute_zsini_array's own docstring for the bug this
fixes), without redoing the whole (expensive) grid/subgrid/boundary-forcing
build. Opens the existing model in "r+" mode, recomputes zsini from the
tile's own already-written matched_boundary_points.gpkg/
corrected_hydrographs.csv, and writes only the "zs" grid variable
(sf.grid.write(data_vars=["zs"]) - still always rewrites sfincs.msk/sfincs.ind
too, hydromt_sfincs's own write() requirement, but with unchanged content
since mask itself isn't touched here).

Usage:
    python rebuild_zsini.py --tile-id 1702
    python rebuild_zsini.py --config <resolved_config.yml> --tile-id 1702
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from hydromt_sfincs import SfincsModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sfincs_tile import TRUNCATE_WINDOW_HR_DEFAULT, _compute_zsini_array  # noqa: E402
from gfm_config import read_root  # noqa: E402


def rebuild_zsini(tile_id: str, root: Path, truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT) -> None:
    tile_dir = root / "model_outputs" / tile_id / "inputs"
    sfincs_dir = root / "validation_sfincs" / tile_id / "sfincs_model"

    sf = SfincsModel(root=str(sfincs_dir), mode="r+")
    sf.grid.read()

    matched_points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")

    # same truncation build_sfincs_tile() applies before taking "first_vals" -
    # must match, since the truncated series' own first row is what zsini uses
    if truncate_window_hr is not None:
        t_start, t_end = truncate_window_hr
        keep = (hydrographs["elapsed_hr"] >= t_start) & (hydrographs["elapsed_hr"] <= t_end)
        hydrographs = hydrographs.loc[keep].reset_index(drop=True)

    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]
    locations_utm = matched_points.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)

    grid_coords = sf.grid.data["mask"]
    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]

    zsini_arr = _compute_zsini_array(
        tile_dir / "mask.tif", station_x, station_y, first_vals,
        grid_coords, main_transform, main_crs, (main_height, main_width),
    )
    zsini_da = xr.DataArray(zsini_arr, dims=grid_coords.dims, coords=grid_coords.coords)
    zsini_da = zsini_da.rio.write_crs(sf.crs)
    zsini_da = zsini_da.rio.write_transform(grid_coords.rio.transform())

    sf.initial_conditions.create(zsini=zsini_da, fill_value=-9999.0, reproj_method="nearest")
    sf.grid.write(data_vars=["zs"])

    is_wet = zsini_arr > -9999.0
    n_wet = int(is_wet.sum())
    if n_wet:
        print(f"tile {tile_id}: zsini rebuilt - {n_wet} ocean cell(s) initialized "
              f"{float(zsini_arr[is_wet].min()):.4f} to {float(zsini_arr[is_wet].max()):.4f} m "
              f"(interpolated from {len(station_x)} station(s))")
    else:
        print(f"tile {tile_id}: zsini rebuilt - WARNING: 0 ocean cells found, every cell left dry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=None, help="config.yml or a resolved_config.yml; defaults to the main repo config.yml")
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        root = Path(config["paths"]["root"])
    else:
        _repo_root = Path(__file__).resolve().parent.parent
        root = read_root(_repo_root / "snakemake_workflow" / "config" / "config.yml")

    rebuild_zsini(args.tile_id, root)


if __name__ == "__main__":
    main()
