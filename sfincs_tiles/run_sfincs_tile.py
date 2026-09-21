"""Run a built SFINCS model (build_sfincs_tile.py's own output) and
postprocess: max inundation depth + max flood extent, reprojected back to
EPSG:4326 for direct comparability with the eikonal model's own
`waterdepth_{RP}_{SLR}.tif` global-grid convention.

Run under the hydromt-sfincs-dev env (needs rioxarray/rasterio consistent
with the model build, and xarray for reading sfincs_map.nc):
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
from sfincs_run import run_sfincs_subprocess  # noqa: E402


class _SimpleLog:
    """No-op: sfincs_run.py's run_sfincs_subprocess already prints every
    forwarded line directly to stderr itself (for live console output in
    its original logger-based context, where log.info/.warning write to a
    real log file, not the console) - a log object that also prints here
    would double every SFINCS output line."""

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


def _apply_land_mask_doublecheck(hmax: np.ndarray, transform, crs, land_mask_path: Path) -> np.ndarray:
    """Second, independent land check on top of `downscale_floodmap()`'s own
    `gdf_mask` - reproject `land_mask_path` directly onto `hmax`'s own fine
    grid (nearest-neighbour) and require BOTH checks to agree a cell is
    land before it counts, masking every other cell to NaN.

    Real, confirmed leak fixed here (2026-09, tile 1907's first subgrid
    run): 32 cells still passed `gdf_mask` alone with clearly ocean-like
    negative subgrid elevation (down to -9 m) - traced the worst one back
    to its real lon/lat and confirmed the NATIVE mask.tif value there is
    1.0 (ocean), not land. `gdf_mask`'s own polygon-vs-raster
    rasterization at the coastline boundary lets a small number of edge
    cells slip through - this independent raster-vs-raster check has no
    such vector/raster boundary to disagree about.
    """
    with rasterio.open(land_mask_path) as src:
        land_on_subgrid = np.empty(hmax.shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=land_on_subgrid,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return np.where(land_on_subgrid == 0.0, hmax, np.nan)


def compute_max_inundation(sfincs_dir: Path, land_mask_path: Path) -> tuple[np.ndarray, dict]:
    """(hmax_m, grid_info) - max flood depth on the model's own FINE
    SUBGRID resolution, not the coarse computational grid, via
    hydromt_sfincs's own downscale_floodmap() utility (replaces this
    function's own earlier hand-rolled `zsmax - dep` check entirely, now
    that the model has a real subgrid - see build_sfincs_tile.py's own
    subgrid.create() comment for why: subgrid tables exist specifically so
    a coarse-grid zsmax can be downscaled onto real fine-resolution terrain,
    which is a fundamentally more correct approach than this function's own
    old single-resolution "one dep value per computational cell" model ever
    was, including for the land/ocean-cliff-smoothing artifact class of bug
    that model used to need its own dep_m > -2.0 plausibility filter for
    (see git history) - downscale_floodmap() reads the real fine-resolution
    subgrid DEM directly, no separately-reprojected coarse land mask to
    disagree with it in the first place.

    `land_mask_path` (the tile's own mask.tif, EPSG:4326) is vectorised into
    a land-only polygon and passed as downscale_floodmap()'s own gdf_mask,
    excluding ocean/lake/river cells from the flood-depth output - the
    subgrid-resolution equivalent of this function's old is_land check.

    gdf_mask alone isn't quite enough, though - real, confirmed leak (2026-09,
    tile 1907's first subgrid run): 32 cells still passed the mask with clearly
    ocean-like negative subgrid elevation (down to -9 m); traced the worst one
    back to its real lon/lat and confirmed the NATIVE mask.tif value there is
    1.0 (ocean), not land - gdf_mask's own polygon-vs-raster rasterization at
    the coastline boundary lets a small number of edge cells slip through
    (a smaller-magnitude version of the same class of coastline-alignment bug
    already fixed once for the old single-resolution code). Belt-and-suspenders
    fix: ALSO reproject land_mask_path directly onto the subgrid's own fine
    grid (nearest-neighbour) and intersect that with gdf_mask's own output -
    two independent land checks, from two different code paths, must both
    agree a cell is land before it counts.
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

    # NOT a raw xr.open_dataset(map_path) - real bug found and fixed here (2026-09,
    # first subgrid run): SFINCS's own sfincs_map.nc stores zsmax on its native
    # staggered (n, m) index dims with x/y as 2D COORDINATE arrays, not proper 1D
    # x/y dims - hydromt's own raster accessor (which downscale_floodmap() needs)
    # can't recognize spatial dims from that shape at all ("x dimension not found").
    # hydromt_sfincs's own SfincsOutput.read_map_file() translates this staggered
    # format into a proper regular-grid DataArray via
    # readers.read_sfincs_map_results(fn_map, ds_like=model.grid.mask, ...) - reuse
    # that real, tested reader instead of hand-rolling the same translation.
    sf_out = SfincsModel(root=str(sfincs_dir), mode="r")
    sf_out.output.read()
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

    with rasterio.open(hmax_subgrid_path) as src:
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
    # own default guess. Real bug found and fixed here (2026-09, comparing tile 1573
    # against eikonal): letting calculate_default_transform() pick its own degree
    # resolution from a projected (UTM, isotropic-METRE) source produces a
    # SQUARE-IN-DEGREES output grid, which is NOT square in real ground distance
    # except right at the equator - confirmed live at ~55 deg N: the true-metre UTM
    # subgrid resolution (30x30 m) reprojected out to ~29 x ~50 m in real ground
    # distance, inflating flooded-AREA comparisons against the eikonal model's own
    # (already correctly latitude-corrected) grid by ~1.6x despite a nearly identical
    # flooded CELL COUNT - not a real physical model disagreement, a reprojection
    # resolution bug. Fix: get the source's own real metre resolution directly (UTM
    # is isotropic, so this is just abs(transform.a)), convert to degrees separately
    # per axis using the tile's own centre latitude, and pass that explicitly as
    # calculate_default_transform()'s own `resolution` argument - keeps the
    # reprojected grid square in real ground distance, like the eikonal model's own.
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
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--skip-run", action="store_true",
        help="skip run_sfincs_subprocess and go straight to postprocessing - for a tile "
             "already run elsewhere (e.g. generate_sfincs_hpc_jobs.py's own HPC batch jobs), "
             "whose sfincs_map.nc has already been copied back into sfincs_model/.",
    )
    args = parser.parse_args()
    if not args.skip_run and not args.sfincs_exe:
        parser.error("--sfincs-exe is required unless --skip-run is given")

    root = read_root(Path(args.config))
    sfincs_dir = root / "validation_sfincs" / args.tile_id / "sfincs_model"
    out_dir = root / "validation_sfincs" / args.tile_id / "outputs"
    land_mask_path = root / "model_outputs" / args.tile_id / "inputs" / "mask.tif"

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
