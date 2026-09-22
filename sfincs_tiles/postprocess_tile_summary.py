"""Per-tile results summary for the v2 validation batch - writes one small
{base_dir_name}/{tile_id}/outputs/summary.json per tile, so the whole batch's
results can be merged with a trivial glob+concat (see
aggregate_tile_summaries.py) instead of re-deriving everything from raw
rasters after the fact.

Needs the hydromt-sfincs-dev env (imports hydromt_sfincs.SfincsModel to read
the SFINCS grid's own boundary-cell mask) - run this as the LAST step of a
tile's own pipeline, after the env has already switched back for the SFINCS
run stage, so no extra env switch is needed just for postprocessing.
Deliberately avoids importing src/rasters.py (older-hydromt-env only) - the
waterdepth int16-cm decode (WATERDEPTH_SCALE=100, WATERDEPTH_NODATA_INT16=
32767) is copied directly from its own documented convention instead.

Usage:
    python postprocess_tile_summary.py --tile-id 12345 --set A
    python postprocess_tile_summary.py --tile-id 12345 --set B --base-dir-name validation_sfincs_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from hydromt_sfincs import SfincsModel
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

LAND_CODE = 0
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767


def _decode_waterdepth_cm(path: Path) -> tuple[np.ndarray, object, object, tuple]:
    """(depth_m, transform, crs, shape) - NaN where not flooded/nodata."""
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
        transform = src.transform
        crs = src.crs
        shape = src.shape
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m, transform, crs, shape


def _pixel_area_km2_by_row(transform, height: int, crs) -> np.ndarray:
    """Latitude-corrected per-row pixel area (km2) for an EPSG:4326 raster -
    same convention used throughout this session's own area comparisons."""
    assert str(crs).upper() == "EPSG:4326", crs
    px_w_deg = abs(transform.a)
    px_h_deg = abs(transform.e)
    rows = np.arange(height)
    lat_top = transform.f
    lat_center = lat_top + transform.e * (rows + 0.5)
    w_m = px_w_deg * 111320.0 * np.cos(np.radians(lat_center))
    h_m = px_h_deg * 110540.0
    return (w_m * h_m) / 1e6


def _depth_stats(depth_m: np.ndarray) -> dict:
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if finite.size == 0:
        return {"mean_m": None, "median_m": None, "max_m": None}
    return {"mean_m": float(finite.mean()), "median_m": float(np.median(finite)), "max_m": float(finite.max())}


def summarize_tile(tile_id: str, root: Path, base_dir_name: str, tile_set: str) -> dict:
    tile_dir = root / base_dir_name / tile_id
    native_mask_path = tile_dir / "inputs" / "mask.tif"
    out_dir = tile_dir / "outputs"

    with rasterio.open(native_mask_path) as src:
        native_mask = src.read(1)
        native_transform = src.transform
        native_crs = src.crs
        native_height = src.height
    ocean_frac = float(np.mean(native_mask == 1))
    row_area_native = _pixel_area_km2_by_row(native_transform, native_height, native_crs)
    domain_area_km2 = float((np.ones(native_mask.shape[0]) * native_mask.shape[1] * row_area_native).sum())

    result: dict = {
        "tile_id": tile_id, "set": tile_set,
        "domain_area_km2": domain_area_km2, "ocean_frac": ocean_frac,
    }

    # -- bathtub + eikonal: both on the SFINCS subgrid UTM grid (30m, isotropic) --
    for name, fname in [
        ("bathtub", "bathtub_waterdepth_RP100_SLR_0.tif"),
        ("eikonal", "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif"),
    ]:
        path = out_dir / fname
        if not path.exists():
            result[f"{name}_km2"] = None
            result.update({f"{name}_depth_{k}": None for k in ("mean_m", "median_m", "max_m")})
            continue
        depth_m, transform, crs, shape = _decode_waterdepth_cm(path)
        mog = np.empty(shape, dtype=np.float32)
        with rasterio.open(native_mask_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=mog,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
            )
        land = mog == LAND_CODE
        flooded_land = np.isfinite(depth_m) & (depth_m > 0) & land
        cell_km2 = abs(transform.a) * abs(transform.e) / 1e6
        result[f"{name}_km2"] = float(flooded_land.sum() * cell_km2)
        stats = _depth_stats(np.where(flooded_land, depth_m, np.nan))
        result.update({f"{name}_depth_{k}": v for k, v in stats.items()})

    # -- SFINCS: hmax.tif, already reprojected to EPSG:4326 land-only via run_sfincs_tile.py's own doublecheck --
    hmax_path = out_dir / "hmax.tif"
    if hmax_path.exists():
        with rasterio.open(hmax_path) as src:
            hmax = src.read(1)
            hmax_nodata = src.nodata
            hmax_transform = src.transform
            hmax_crs = src.crs
            hmax_height = src.height
        valid = hmax != hmax_nodata if hmax_nodata is not None else np.isfinite(hmax)
        row_area_hmax = _pixel_area_km2_by_row(hmax_transform, hmax_height, hmax_crs)
        n_per_row = valid.sum(axis=1)
        result["sfincs_km2"] = float((n_per_row * row_area_hmax).sum())
        depth_vals = np.where(valid, hmax, np.nan)
        stats = _depth_stats(depth_vals)
        result.update({f"sfincs_depth_{k}": v for k, v in stats.items()})
    else:
        result["sfincs_km2"] = None
        result.update({f"sfincs_depth_{k}": None for k in ("mean_m", "median_m", "max_m")})

    # -- SFINCS boundary-cell distance to real land (today's central metric) --
    sfincs_model_dir = tile_dir / "sfincs_model"
    try:
        sf = SfincsModel(root=str(sfincs_model_dir), mode="r")
        sf.grid.read()
        sfincs_mask = sf.grid.data["mask"].values
        sfincs_transform = sf.grid.data.raster.transform
        sfincs_crs = sf.grid.data.raster.crs
        shape = sfincs_mask.shape
        mog = np.empty(shape, dtype=np.float32)
        with rasterio.open(native_mask_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=mog,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=sfincs_transform, dst_crs=sfincs_crs, resampling=Resampling.nearest,
            )
        real_land = mog == LAND_CODE
        bnd = sfincs_mask == 2
        if real_land.any() and bnd.any():
            dist_to_land = ndimage.distance_transform_edt(
                ~real_land, sampling=(abs(sfincs_transform.e), abs(sfincs_transform.a)),
            )
            d_bnd_km = dist_to_land[bnd] / 1000.0
            result["sfincs_boundary_dist_mean_km"] = float(d_bnd_km.mean())
            result["sfincs_boundary_dist_median_km"] = float(np.median(d_bnd_km))
            result["sfincs_boundary_dist_max_km"] = float(d_bnd_km.max())
        else:
            result["sfincs_boundary_dist_mean_km"] = None
            result["sfincs_boundary_dist_median_km"] = None
            result["sfincs_boundary_dist_max_km"] = None
    except Exception as e:
        result["sfincs_boundary_dist_mean_km"] = None
        result["sfincs_boundary_dist_median_km"] = None
        result["sfincs_boundary_dist_max_km"] = None
        result["sfincs_boundary_dist_error"] = f"{type(e).__name__}: {e}"

    return result


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--set", required=True, choices=["A", "B"])
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    result = summarize_tile(args.tile_id, root, args.base_dir_name, args.set)

    out_path = root / args.base_dir_name / args.tile_id / "outputs" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"tile {args.tile_id}: bathtub={result.get('bathtub_km2')} eikonal={result.get('eikonal_km2')} "
          f"sfincs={result.get('sfincs_km2')} km2, boundary_dist_median={result.get('sfincs_boundary_dist_median_km')} km")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
