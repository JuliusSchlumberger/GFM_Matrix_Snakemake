"""Runs a built SFINCS model (build_sfincs_tile.py's own output) and
postprocesses it: max inundation depth + max flood extent, reprojected to
EPSG:4326 to match the eikonal model's own `waterdepth_{RP}_{SLR}.tif`
global-grid convention.

Run under the hydromt-sfincs-dev env:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\hydromt-sfincs-dev\\python.exe run_sfincs_tile.py --tile-id 1907 --sfincs-exe <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows.downscaling import downscale_floodmap
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sfincs_tile import _ocean_polygon_wgs84  # noqa: E402 - generic despite the name (any mask code)
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402
from sfincs_run import run_sfincs_subprocess  # noqa: E402


class _SimpleLog:
    """No-op logger: run_sfincs_subprocess already prints every forwarded
    line to stderr itself; a log object that also prints here would double
    every SFINCS output line."""

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


def _apply_land_mask_doublecheck(hmax: np.ndarray, transform, crs, land_mask_path: Path) -> np.ndarray:
    """Second, independent land check on top of `downscale_floodmap()`'s own
    `gdf_mask`: reprojects `land_mask_path` directly onto `hmax`'s own fine
    grid (nearest-neighbour) and requires both checks to agree a cell is
    land before it counts, masking every other cell to NaN.

    `gdf_mask`'s own polygon-vs-raster rasterization at the coastline can
    let a small number of edge cells slip through as land; this raster-vs-
    raster check has no such vector/raster boundary to disagree about.
    """
    with retry_transient_io(rasterio.open, land_mask_path) as src:
        land_on_subgrid = np.empty(hmax.shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=land_on_subgrid,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return np.where(land_on_subgrid == 0.0, hmax, np.nan)


def compute_max_inundation(sfincs_dir: Path, land_mask_path: Path) -> tuple[np.ndarray, dict]:
    """Returns (hmax_m, grid_info): max flood depth on the model's own fine
    subgrid resolution via hydromt_sfincs's own downscale_floodmap()
    utility, which downscales the coarse-grid zsmax onto the real
    fine-resolution subgrid DEM.

    `land_mask_path` (the tile's own mask.tif, EPSG:4326) is vectorised into
    a land-only polygon passed as downscale_floodmap()'s own gdf_mask,
    excluding ocean/lake/river cells from the output. It is also reprojected
    directly onto the subgrid's own fine grid and intersected with
    gdf_mask's output - see `_apply_land_mask_doublecheck`.
    """
    map_path = sfincs_dir / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(f"SFINCS ran but no map output found at {map_path}")

    dep_subgrid_path = sfincs_dir / "subgrid" / "dep_subgrid.tif"
    if not dep_subgrid_path.exists():
        raise FileNotFoundError(
            f"{dep_subgrid_path} not found - build_sfincs_tile.py's own sf.subgrid.create() "
            "call needs write_dep_tif=True for this postprocessing step to have a fine-"
            "resolution DEM to downscale onto."
        )

    # Not a raw xr.open_dataset(map_path): SFINCS's own sfincs_map.nc stores zsmax
    # on its native staggered (n, m) index dims with x/y as 2D coordinate arrays,
    # which hydromt's own raster accessor (needed by downscale_floodmap()) can't
    # recognize as spatial dims. SfincsOutput.read_map_file() translates this
    # staggered format into a proper regular-grid DataArray.
    sf_out = SfincsModel(root=str(sfincs_dir), mode="r")
    retry_transient_io(sf_out.output.read)
    if "zsmax" not in sf_out.output.data:
        raise KeyError(f"'zsmax' not found in {map_path} - variables present: {list(sf_out.output.data.keys())}")
    zsmax = sf_out.output.data["zsmax"]
    if "timemax" in zsmax.dims:
        zsmax = zsmax.max(dim="timemax", skipna=True)
    zsmax = zsmax.squeeze().load()

    land_gdf = _ocean_polygon_wgs84(land_mask_path, ocean_code=0)  # code=0 -> land cells, despite the function's name

    hmax_subgrid_path = sfincs_dir / "hmax_subgrid.tif"
    downscale_floodmap(
        zsmax=zsmax, dep=dep_subgrid_path, reproj_method="nearest", subtract_dem=True,
        hmin=0.05, gdf_mask=land_gdf, floodmap_fn=hmax_subgrid_path,
    )

    with retry_transient_io(rasterio.open, hmax_subgrid_path) as src:
        hmax = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    hmax = np.where(hmax == nodata, np.nan, hmax) if nodata is not None else hmax

    hmax = _apply_land_mask_doublecheck(hmax, transform, crs, land_mask_path)

    return hmax, {"transform": transform, "crs": crs}


def reproject_to_4326(arr: np.ndarray, transform, crs, out_path: Path, nodata: float = -9999.0) -> None:
    dst_crs = "EPSG:4326"
    src_bounds = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], transform)

    # Explicit latitude-corrected target resolution, not calculate_default_transform()'s
    # own default guess: a UTM source's isotropic metre resolution converted to degrees
    # is square in real ground distance only at the equator, so degrees are computed
    # separately per axis from the tile's own centre latitude and passed explicitly as
    # calculate_default_transform()'s own `resolution` argument.
    src_res_m = abs(transform.a)
    center_x = (src_bounds[0] + src_bounds[2]) / 2.0
    center_y = (src_bounds[1] + src_bounds[3]) / 2.0
    _, center_lat = warp_transform(crs, dst_crs, [center_x], [center_y])
    res_x_deg = src_res_m / (111320.0 * np.cos(np.radians(center_lat[0])))
    res_y_deg = src_res_m / 110540.0

    dst_transform, width, height = calculate_default_transform(
        crs, dst_crs, arr.shape[1], arr.shape[0], *src_bounds, resolution=(res_x_deg, res_y_deg),
    )
    dst = np.full((height, width), nodata, dtype=np.float32)
    src_arr = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
    reproject(
        source=src_arr, destination=dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=nodata, dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": height, "width": width,
        "transform": dst_transform, "crs": dst_crs, "nodata": nodata, "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(dst, 1)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--sfincs-exe", default=None, help="required unless --skip-run")
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument(
        "--skip-run", action="store_true",
        help="skip run_sfincs_subprocess and go straight to postprocessing - for a tile "
             "already run elsewhere (e.g. run_one_tile.sh's own HPC batch dispatch), "
             "whose sfincs_map.nc has already been copied back into sfincs_model/.",
    )
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2", help="output root directory name under paths.root (default: validation_sfincs_v2)")
    args = parser.parse_args()
    if not args.skip_run and not args.sfincs_exe:
        parser.error("--sfincs-exe is required unless --skip-run is given")

    root = read_root(Path(args.config))
    sfincs_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    out_dir = root / args.base_dir_name / args.tile_id / "outputs"
    # Reads from the tile's working copy, not model_outputs/ directly: the SFINCS
    # model is built from base_dir_name's own mask.tif, so the land-mask doublecheck
    # here must use the same one.
    land_mask_path = root / args.base_dir_name / args.tile_id / "inputs" / "mask.tif"

    if args.skip_run:
        map_path = sfincs_dir / "sfincs_map.nc"
        if not map_path.exists():
            raise FileNotFoundError(f"--skip-run given but {map_path} doesn't exist - has the HPC batch job for this tile actually finished and copied its output back?")
        print(f"--skip-run: using existing {map_path}")
    else:
        log = _SimpleLog()
        run_sfincs_subprocess(Path(args.sfincs_exe), sfincs_dir, args.timeout_s, log, label=f"SFINCS tile {args.tile_id}")

    hmax, grid_info = compute_max_inundation(sfincs_dir, land_mask_path)
    n_flooded = int(np.isfinite(hmax).sum())
    print(f"Flooded cells (UTM grid): {n_flooded} of {hmax.size} ({100 * n_flooded / hmax.size:.1f}%)")
    if n_flooded:
        print(f"Max inundation depth: {np.nanmax(hmax):.3f} m")

    reproject_to_4326(hmax, grid_info["transform"], grid_info["crs"], out_dir / "hmax.tif")
    extent = np.where(np.isfinite(hmax), 1.0, np.nan)
    reproject_to_4326(extent, grid_info["transform"], grid_info["crs"], out_dir / "flood_extent.tif")
    print(f"Wrote {out_dir / 'hmax.tif'}")
    print(f"Wrote {out_dir / 'flood_extent.tif'}")


if __name__ == "__main__":
    main()
