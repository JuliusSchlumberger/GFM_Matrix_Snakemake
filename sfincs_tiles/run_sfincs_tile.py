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
import xarray as xr
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def compute_max_inundation(sfincs_dir: Path, land_mask_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(hmax_m, dep_m, grid_info) from sfincs_map.nc's own zsmax envelope,
    on the model's own UTM grid - hmax = zsmax - dep, masked to positive
    depth AND restricted to LAND cells only (matching 16_run_event.py's own
    `da_hmax = zsmax - bed` pattern from the reference project, which
    similarly requires a `sea_mask_path`).

    Real bug found and fixed here (first tile-1907 run): computing
    zsmax - dep over EVERY cell, including deep ocean ones (dep down to
    -38m for this tile), reports "38 m of inundation" at cells that were
    never dry to begin with - the same permanent-water exclusion this
    pipeline's own eikonal-model validation (src/validation.py's
    permanent_water_mask, used throughout validate_country.py) already
    handles for exactly this reason. `land_mask_path` is the tile's own
    mask.tif (EPSG:4326, land=0/ocean=1/lake=2/river=3), reprojected onto
    this model's own UTM grid (nearest-neighbour, categorical) before
    masking.
    """
    map_path = sfincs_dir / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(f"SFINCS ran but no map output found at {map_path}")

    with xr.open_dataset(map_path) as ds:
        if "zsmax" not in ds:
            raise KeyError(f"'zsmax' not found in {map_path} - variables present: {list(ds.data_vars)}")
        zsmax = ds["zsmax"]
        if "timemax" in zsmax.dims:
            zsmax = zsmax.max(dim="timemax", skipna=True)
        zsmax_arr = np.squeeze(zsmax.values)

    # sfincs.dep is a raw binary grid, not georeferenced on its own - read
    # the georeferencing from the model's own exported GeoTIFF instead
    # (gis/ subfolder, written alongside sfincs.dep by sf.write()).
    gis_dep_candidates = list((sfincs_dir / "gis").glob("*dep*.tif")) + list((sfincs_dir / "gis").glob("*elevation*.tif"))
    if not gis_dep_candidates:
        raise FileNotFoundError(f"No elevation GeoTIFF found under {sfincs_dir / 'gis'} to read the model's own georeferencing from")
    with rasterio.open(gis_dep_candidates[0]) as src:
        dep_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if zsmax_arr.shape != dep_arr.shape:
        raise ValueError(f"zsmax shape {zsmax_arr.shape} != dep shape {dep_arr.shape} from {gis_dep_candidates[0].name}")

    dep_m = np.where(dep_arr == nodata, np.nan, dep_arr) if nodata is not None else dep_arr.astype(np.float64)

    with rasterio.open(land_mask_path) as src:
        land_on_grid = np.empty(dep_arr.shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=land_on_grid,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
    is_land = land_on_grid == 0.0

    # No depth-plausibility filter here (2026-09: an earlier version of this
    # function had one, dep_m > -2.0 - removed). That was a band-aid for a
    # real bug that's now fixed at the source instead: build_sfincs_tile.py
    # used to let hydromt_sfincs reproject the elevation grid with bilinear
    # smoothing while this function's own land_mask_path reprojects with
    # nearest-neighbour, independently, on the same UTM grid - elevation_
    # combined.tif has a DELIBERATE hard step at the coastline (real land
    # directly abutting a GEBCO+MIN_BATHYMETRY_M-floored ocean value, not a
    # physically continuous surface), so bilinear smoothing across it could
    # produce a UTM cell the mask still calls "land" with a deeply negative
    # interpolated dep - confirmed live: as low as -32.7 m, reporting >33 m
    # of "inundation" at a perfectly normal zsmax~1 m open-water cell. A
    # fixed numeric cutoff patching that symptom was itself not physically
    # justified (real coastal land/polders genuinely do sit several metres
    # below the reference datum) and, worse, wasn't even robust to a
    # different MIN_BATHYMETRY_M value (confirmed: it let through a whole
    # different set of artifacts when the floor changed from -50 to -10 m -
    # see build_sfincs_tile.py's own elevation-reprojection comment for the
    # real fix: pre-reproject with nearest-neighbour ourselves before
    # elevation.create() ever runs, so dep_m here is never smoothed across
    # that cliff in the first place, and no depth threshold is needed to
    # compensate for it downstream).
    hmax = zsmax_arr - dep_m
    hmax = np.where((hmax > 0.0) & is_land, hmax, np.nan)

    return hmax, dep_m, {"transform": transform, "crs": crs}


def reproject_to_4326(arr: np.ndarray, transform, crs, out_path: Path, nodata: float = -9999.0) -> None:
    dst_crs = "EPSG:4326"
    dst_transform, width, height = calculate_default_transform(
        crs, dst_crs, arr.shape[1], arr.shape[0], *rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], transform)
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

    hmax, dep_m, grid_info = compute_max_inundation(sfincs_dir, land_mask_path)
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
