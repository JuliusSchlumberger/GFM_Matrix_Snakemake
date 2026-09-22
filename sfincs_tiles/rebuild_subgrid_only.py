"""Rebuild ONLY the subgrid table (dep_subgrid.tif/manning_subgrid.tif/
sfincs_subgrid.nc) for an already-built SFINCS tile whose
elevation_combined.tif was replaced (e.g. by the GEBCO-ceiling/lake-floor
fix in build_elevation.py), without redoing boundary-forcing/zsini - those
don't depend on elevation and are left untouched. Mirrors
build_sfincs_tile.py's own steps 1-3 exactly (grid/mask are deterministic
from tile_gdf/ocean_poly, both unchanged, so recreating them here
reproduces the same grid/mask - required anyway since a SfincsModel's
DataCatalog is parsed once at construction, not re-readable after adding
the subgrid_src entries mid-session, same constraint build_sfincs_tile.py
itself works around the same way).

Usage:
    python rebuild_subgrid_only.py --tile-id 1751 --sfincs-dir <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from hydromt_sfincs import SfincsModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sfincs_tile import (
    MAIN_RES_M_DEFAULT,
    SUBGRID_NR_LEVELS_DEFAULT,
    SUBGRID_NR_PIXELS_DEFAULT,
    SUBGRID_NRMAX_DEFAULT,
    _ocean_polygon_wgs84,
    _reproject_nearest_to_grid,
)
from gfm_config import read_root  # noqa: E402


def rebuild_subgrid(
    tile_id: str, root: Path, sfincs_dir: Path,
    resolution_m: float = MAIN_RES_M_DEFAULT,
    subgrid_nr_pixels: int = SUBGRID_NR_PIXELS_DEFAULT,
    subgrid_nr_levels: int = SUBGRID_NR_LEVELS_DEFAULT,
    subgrid_nrmax: int = SUBGRID_NRMAX_DEFAULT,
) -> None:
    tile_dir = root / "model_outputs" / tile_id / "inputs"
    local_catalog_path = sfincs_dir / "data_catalog_local.yml"
    with open(local_catalog_path) as fh:
        local_catalog = yaml.safe_load(fh)

    tile_gdf = gpd.read_file(tile_dir / "tile_geometry.gpkg")
    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")

    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False)

    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    fine_transform = main_transform * main_transform.scale(1.0 / subgrid_nr_pixels)
    fine_height, fine_width = main_height * subgrid_nr_pixels, main_width * subgrid_nr_pixels

    subgrid_sources = {}
    for name, src_uri in [("local_elevation_subgrid", "elevation_combined.tif"), ("local_roughness_subgrid", "manning_n.tif")]:
        out_path = sfincs_dir / f"{Path(src_uri).stem}_subgrid_src.tif"
        fine_arr = _reproject_nearest_to_grid(sfincs_dir / src_uri, fine_transform, main_crs, (fine_height, fine_width))
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": fine_height, "width": fine_width,
            "transform": fine_transform, "crs": main_crs, "nodata": np.nan, "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(fine_arr, 1)
        subgrid_sources[name] = out_path.name

    local_catalog["local_elevation_subgrid"] = {"data_type": "RasterDataset", "uri": subgrid_sources["local_elevation_subgrid"], "driver": "rasterio"}
    local_catalog["local_roughness_subgrid"] = {"data_type": "RasterDataset", "uri": subgrid_sources["local_roughness_subgrid"], "driver": "rasterio"}
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False)

    sf.subgrid.create(
        elevation_list=[{"elevation": "local_elevation_subgrid"}],
        roughness_list=[{"manning": "local_roughness_subgrid"}],
        nr_subgrid_pixels=subgrid_nr_pixels,
        nr_levels=subgrid_nr_levels,
        nrmax=subgrid_nrmax,
        write_dep_tif=True,
        write_man_tif=True,
    )
    # NOTE: deliberately no sf.grid.write() here - "zs" (zsini) was never set on
    # this freshly-recreated in-memory grid (only mask/dep/ind would write), and
    # the already-correct sfincs.ini from the earlier zsini fix must stay
    # untouched; mask/ind are deterministic from tile_gdf/ocean_poly (unchanged)
    # so the ones already on disk already match.
    print(f"tile {tile_id} at {sfincs_dir}: subgrid table rebuilt from updated elevation_combined.tif "
          f"({subgrid_nr_pixels}x refinement, {subgrid_nr_levels} levels)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--sfincs-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        root = Path(config["paths"]["root"])
    else:
        _repo_root = Path(__file__).resolve().parent.parent
        root = read_root(_repo_root / "snakemake_workflow" / "config" / "config.yml")

    rebuild_subgrid(args.tile_id, root, Path(args.sfincs_dir))


if __name__ == "__main__":
    main()
