"""Build a runnable SFINCS model for one GFM tile, from the prep-stage
outputs (build_elevation.py, build_roughness.py, build_boundary_forcing.py).

Every sfincs_tiles/ script, including the prep stage, runs under the SAME
single hydromt-sfincs-dev env - none of them import src/config_utils.py
(confirmed 2026-09: importing it at all fails under this env's hydromt
1.4.1, `ImportError: cannot import name 'setuplog' from 'hydromt.log'` -
removed in hydromt's 1.x rewrite; config_utils.py was written against the
main pipeline's hydromt 0.9.3). gfm_config.py's read_root/resolve_catalog_path
cover the small amount sfincs_tiles/ actually needs from it. No separate
gfm_python_preprocessing env or `conda run` needed for anything in this
folder:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\hydromt-sfincs-dev\\python.exe build_sfincs_tile.py --tile-id 1907

No weirs, no restart, no subgrid, no discharge - see sfincs_tiles' own plan
doc for the full design/reasoning.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry
import xarray as xr
import yaml
from hydromt_sfincs import SfincsModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_boundary_forcing import idw_interpolate_to_grid  # noqa: E402
from gfm_config import read_root  # noqa: E402


def _ocean_polygon_wgs84(mask_path: Path, ocean_code: int = 1) -> gpd.GeoDataFrame:
    """Vectorize mask.tif's ocean cells into a polygon GeoDataFrame (EPSG:4326)."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
    ocean = (mask == ocean_code).astype(np.uint8)
    shapes = rasterio.features.shapes(ocean, mask=ocean.astype(bool), transform=transform)
    geoms = [shapely.geometry.shape(geom) for geom, val in shapes]
    if not geoms:
        raise ValueError(f"{mask_path}: no ocean (code={ocean_code}) cells found")
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def build_sfincs_tile(
    tile_id: str, root: Path, resolution_m: float = 30.0,
    dtmapout_s: float = 1800.0, tref: datetime | None = None,
) -> Path:
    tile_dir = root / "model_outputs" / tile_id / "inputs"
    sfincs_dir = root / "validation_sfincs" / tile_id / "sfincs_model"
    sfincs_dir.mkdir(parents=True, exist_ok=True)

    tile_gdf = gpd.read_file(tile_dir / "tile_geometry.gpkg")

    # -- local data catalog for elevation.create/roughness.create (both need
    # catalog-keyed sources, confirmed via the real hydromt_sfincs API -
    # not a raw-DataArray-accepting signature) --
    local_catalog_path = sfincs_dir / "data_catalog_local.yml"
    local_catalog = {
        "meta": {"root": str(sfincs_dir)},
        "local_elevation": {"data_type": "RasterDataset", "uri": "elevation_combined.tif", "driver": "rasterio"},
        "local_roughness": {"data_type": "RasterDataset", "uri": "manning_n.tif", "driver": "rasterio"},
    }
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)

    # -- 1. grid --
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    print(f"[1/8] grid created: {dict(sf.grid.data.sizes)} cells, crs={sf.crs}")

    # -- 2. elevation (combined DeltaDTM+MDT-corrected-GEBCO, built by build_elevation.py) --
    sf.elevation.create(elevation_list=[{"elevation": "local_elevation"}])
    print(f"[2/8] elevation set - range {float(sf.grid.data['dep'].min()):.2f} to {float(sf.grid.data['dep'].max()):.2f} m")

    # -- 3. mask: active cells (whole tile) + waterlevel boundary (ocean edge only) --
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False)
    n_active = int((sf.grid.data["mask"] > 0).sum())
    n_bnd = int((sf.grid.data["mask"] == 2).sum())
    print(f"[3/8] mask: {n_active} active cell(s), {n_bnd} waterlevel-boundary cell(s)")
    if n_bnd == 0:
        raise RuntimeError(
            f"tile {tile_id}: 0 waterlevel-boundary cells after create_boundary - "
            "the ocean polygon didn't reach any active-domain edge cell on this grid. "
            "No weir/subgrid/discharge fallback exists in this pipeline - this tile can't be forced."
        )

    # -- 4. roughness (friction.tif decoded to real Manning's n by build_roughness.py) --
    sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])
    print(f"[4/8] roughness set - range {float(sf.grid.data['manning'].min()):.4f} to {float(sf.grid.data['manning'].max()):.4f}")

    # -- 5. boundary forcing (COAST-HG hydrographs, empirically MDT-corrected -
    # build_boundary_forcing.py's own prep output) --
    matched_points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")
    elapsed_hr = hydrographs["elapsed_hr"].to_numpy()
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]

    tref = tref or datetime(2026, 1, 1)  # arbitrary but fixed reference - only elapsed time
    # within the hydrograph is physically meaningful (see coast_hg's own catalog caveats)
    times = pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in elapsed_hr])
    wl_df = hydrographs[station_cols].copy()
    wl_df.index = times
    wl_df.columns = range(len(station_cols))  # water_level.create expects positional columns matching locations' row order

    # matched_points.gpkg and corrected_hydrographs.csv's columns were written
    # in the same row order by build_boundary_forcing.py's own CLI - no
    # re-matching needed here. water_level.create()'s own docstring requires
    # an explicit "index" column matching timeseries' column names (its
    # internal get_geodataframe/set_index("index") call) - a plain
    # positional/row-order match is not enough.
    locations_gdf = matched_points.copy()
    locations_gdf["index"] = range(len(station_cols))

    sf.config.set("tref", tref)
    sf.config.set("tstart", tref)
    tstop = times[-1]
    sf.config.set("tstop", tstop)

    # buffer: build_boundary_forcing.py's own select_k_nearest_boundary_points
    # already restricts these to the k stations nearest the tile itself
    # (default k=20 - see K_NEAREST_STATIONS there for why: the tile's own
    # pre-selected boundaries_{RP}_{SLR}.gpkg set can otherwise include real
    # GTSM stations 50-125 km away, too far to physically represent this
    # tile's own local storm-tide forcing), but water_level.create() still
    # spatially clips `locations` to within `buffer` metres of the model's
    # own waterlevel-boundary cells - a small buffer (e.g. a few
    # resolution-cell-widths) would silently drop the farther of those k
    # points again. 100 km errs generous since k-nearest already keeps this
    # set small and locally relevant - no risk of pulling in unrelated
    # stations, just headroom so none of the k get clipped a second time.
    sf.water_level.create(timeseries=wl_df, locations=locations_gdf, buffer=100_000.0)
    print(f"[5/8] water-level forcing: {len(station_cols)} station(s), {len(times)} timestep(s), "
          f"{tref} -> {tstop} ({elapsed_hr[-1]:.1f} h)")

    # -- 6. initial conditions (zsini): IDW of the matched stations' own
    # first-timestep corrected value, onto this model's own UTM grid --
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)
    locations_utm = locations_gdf.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()

    dep = sf.grid.data["dep"]
    yy, xx = np.meshgrid(dep["y"].values, dep["x"].values, indexing="ij")
    zsini_arr = idw_interpolate_to_grid(station_x, station_y, first_vals, xx, yy)
    zsini_da = xr.DataArray(zsini_arr.astype(np.float32), dims=dep.dims, coords=dep.coords)
    zsini_da = zsini_da.rio.write_crs(sf.crs)
    zsini_da = zsini_da.rio.write_transform(dep.rio.transform())

    sf.initial_conditions.create(zsini=zsini_da, fill_value=-9999.0, reproj_method="nearest")
    print(f"[6/8] zsini set - range {float(np.nanmin(zsini_arr)):.4f} to {float(np.nanmax(zsini_arr)):.4f} m "
          f"(interpolated from {len(station_x)} station(s))")

    # -- 7. output config: dtmaxout must span the WHOLE simulation, so the
    # zsmax envelope is one true whole-run maximum, not reset partway
    # through (SFINCS default dtmaxout=86400s=1 day, shorter than most of
    # our ~6.2-day COAST-HG events) - only max-inundation/extent is wanted
    # (see plan doc), so history/velocity/wet-duration outputs stay off. --
    total_span_s = elapsed_hr[-1] * 3600.0
    sf.config.set("dtmaxout", total_span_s + 3600.0)  # +1h margin, never triggers a reset
    sf.config.set("dtmapout", dtmapout_s)
    sf.config.set("baro", 0)
    print(f"[7/8] output config: dtmaxout={total_span_s + 3600.0:.0f}s (whole-run max), dtmapout={dtmapout_s:.0f}s")

    # -- 8. write --
    sf.write()
    print(f"[8/8] wrote SFINCS model to {sfincs_dir}")
    return sfincs_dir


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--resolution-m", type=float, default=30.0)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    build_sfincs_tile(args.tile_id, root, resolution_m=args.resolution_m)


if __name__ == "__main__":
    main()
