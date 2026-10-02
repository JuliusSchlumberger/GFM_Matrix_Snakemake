"""Per-tile, per-model results summary for a validation batch - writes one
small {base_dir_name}/{tile_id}/outputs/summary_{model}.json per requested
model (bathtub, eikonal, sfincs). Each model can be postprocessed
independently without the others being present. The whole batch's per-model
files are combined into one row per tile by the consuming analysis code
(plot_validation_results.py's collect_summaries()).

Needs the hydromt-sfincs-dev env (imports hydromt_sfincs.SfincsModel to read
the SFINCS grid's boundary-cell mask).

Usage:
    python postprocess_tile_summary.py --tile-id 12345 --base-dir-name validation_sfincs_v4
    python postprocess_tile_summary.py --tile-id 12345 --models eikonal --base-dir-name validation_sfincs_v4
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
from flood_agreement import (  # noqa: E402
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, WET_THRESHOLD_M,
    confusion_counts, depth_corr_sufficient_stats, depth_joint_hist, prune_to_ocean_connected,
)
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

LAND_CODE = 0
OCEAN_CODE = 1
RIVER_CODE = 3  # matches config.yml's tile_generation.ocean_code/river_code
WATERDEPTH_SCALE = 100.0
WATERDEPTH_NODATA_INT16 = 32767
MODEL_WATERDEPTH_FILENAME = {
    "bathtub": "bathtub_waterdepth_RP100_SLR_0.tif",
    "eikonal": "eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif",
}


def _decode_waterdepth_cm(path: Path) -> tuple[np.ndarray, object, object, tuple]:
    """(depth_m, transform, crs, shape) - NaN where not flooded/nodata."""
    with retry_transient_io(rasterio.open, path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA_INT16
        transform = src.transform
        crs = src.crs
        shape = src.shape
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    depth_m[raw == nodata] = np.nan
    return depth_m, transform, crs, shape


def _pixel_area_km2_by_row(transform, height: int, crs) -> np.ndarray:
    """Latitude-corrected per-row pixel area (km2) for an EPSG:4326 raster."""
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


def summarize_domain(native_mask_path: Path) -> dict:
    """Domain stats shared by every model's own summary file."""
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        native_mask = src.read(1)
        native_transform = src.transform
        native_crs = src.crs
        native_height = src.height
    ocean_frac = float(np.mean(native_mask == 1))
    row_area_native = _pixel_area_km2_by_row(native_transform, native_height, native_crs)
    domain_area_km2 = float((np.ones(native_mask.shape[0]) * native_mask.shape[1] * row_area_native).sum())
    return {"domain_area_km2": domain_area_km2, "ocean_frac": ocean_frac}


def _load_sfincs_wet_subgrid(
    tile_dir: Path, native_mask_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple] | None:
    """SFINCS's own subgrid-resolution wet mask and depth (hmax_subgrid.tif,
    native UTM subgrid CRS, pixel-identical to bathtub/eikonal's own
    waterdepth rasters). Returns (sfincs_wet_subgrid, sfincs_depth_subgrid,
    sg_shape), or None if SFINCS hasn't been run/postprocessed yet.
    """
    hmax_subgrid_path = tile_dir / "sfincs_model" / "hmax_subgrid.tif"
    if not hmax_subgrid_path.exists():
        return None
    with retry_transient_io(rasterio.open, hmax_subgrid_path) as src:
        sfincs_depth_subgrid = src.read(1).astype(np.float32)
        sg_nodata = src.nodata
        sg_transform = src.transform
        sg_crs = src.crs
        sg_shape = src.shape
    if sg_nodata is not None and not np.isnan(sg_nodata):
        sfincs_depth_subgrid = np.where(sfincs_depth_subgrid == sg_nodata, np.nan, sfincs_depth_subgrid)
    mog_sg = np.empty(sg_shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=mog_sg,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=sg_transform, dst_crs=sg_crs, resampling=Resampling.nearest,
        )
    land_sg = mog_sg == LAND_CODE
    sfincs_wet_subgrid = land_sg & np.isfinite(sfincs_depth_subgrid) & (sfincs_depth_subgrid > WET_THRESHOLD_M)
    return sfincs_wet_subgrid, sfincs_depth_subgrid, sg_shape


def summarize_extent_model(name: str, tile_dir: Path, native_mask_path: Path, sfincs_wet: tuple | None) -> dict:
    """bathtub/eikonal's own flooded area + depth stats, plus (if SFINCS's
    own hmax_subgrid.tif is already available) agreement counts against it.
    Every field is null if this model's own waterdepth raster doesn't exist
    yet for this tile."""
    result = {f"{name}_km2": None}
    result.update({f"{name}_depth_{k}": None for k in ("mean_m", "median_m", "max_m")})
    result.update({f"{name}_matched_km2": None, f"{name}_only_km2": None, f"{name}_sfincs_only_km2": None})
    result[f"{name}_depth_joint"] = None

    path = tile_dir / "outputs" / MODEL_WATERDEPTH_FILENAME[name]
    if not path.exists():
        return result

    depth_m, transform, crs, shape = _decode_waterdepth_cm(path)
    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, native_mask_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    land = mog == LAND_CODE
    flooded_land = np.isfinite(depth_m) & (depth_m > WET_THRESHOLD_M) & land
    if name == "bathtub":
        # Naive bathtub has no connectivity/friction - restrict to
        # ocean-connected flooding (eikonal/SFINCS enforce this upstream).
        flooded_land = prune_to_ocean_connected(
            flooded_land, mog, ocean_code=OCEAN_CODE, land_code=LAND_CODE, river_code=RIVER_CODE,
        )
    cell_km2 = abs(transform.a) * abs(transform.e) / 1e6
    result[f"{name}_km2"] = float(flooded_land.sum() * cell_km2)
    stats = _depth_stats(np.where(flooded_land, depth_m, np.nan))
    result.update({f"{name}_depth_{k}": v for k, v in stats.items()})

    if sfincs_wet is not None:
        sfincs_wet_subgrid, sfincs_depth_subgrid, sg_shape = sfincs_wet
        if shape == sg_shape:
            model_wet = flooded_land  # same (possibly ocean-pruned) mask used for {name}_km2 above
            weight = np.full(shape, cell_km2, dtype=np.float64)
            matched, model_only, sfincs_only = confusion_counts(model_wet, sfincs_wet_subgrid, land, weight)
            result[f"{name}_matched_km2"] = matched
            result[f"{name}_only_km2"] = model_only
            result[f"{name}_sfincs_only_km2"] = sfincs_only

            # Cell-level depth agreement at every mutually-wet cell:
            # SFINCS's own depth (x) vs this model's (y), pixel-identical
            # grids, no reprojection needed.
            both_wet = model_wet & sfincs_wet_subgrid & land
            if both_wet.any():
                x = sfincs_depth_subgrid[both_wet].astype(np.float64)
                y = depth_m[both_wet].astype(np.float64)
                result[f"{name}_depth_joint"] = {
                    **depth_corr_sufficient_stats(x, y),
                    "hist_fine": depth_joint_hist(x, y, DEPTH_CORR_FINE_EDGES).tolist(),
                    "hist_category": depth_joint_hist(x, y, DEPTH_CATEGORY_EDGES).tolist(),
                }

    return result


def summarize_sfincs(tile_dir: Path, native_mask_path: Path) -> dict:
    """SFINCS's own hmax.tif (already reprojected to EPSG:4326 land-only via
    run_sfincs_tile.py's own doublecheck) + boundary-cell distance to real
    land. Every field is null if SFINCS hasn't been run for this tile yet."""
    result: dict = {"sfincs_km2": None}
    result.update({f"sfincs_depth_{k}": None for k in ("mean_m", "median_m", "max_m")})

    hmax_path = tile_dir / "outputs" / "hmax.tif"
    if hmax_path.exists():
        with retry_transient_io(rasterio.open, hmax_path) as src:
            hmax = src.read(1)
            hmax_nodata = src.nodata
            hmax_transform = src.transform
            hmax_crs = src.crs
            hmax_height = src.height
        valid = np.isfinite(hmax) & (hmax > WET_THRESHOLD_M)
        if hmax_nodata is not None:
            valid &= hmax != hmax_nodata
        row_area_hmax = _pixel_area_km2_by_row(hmax_transform, hmax_height, hmax_crs)
        n_per_row = valid.sum(axis=1)
        result["sfincs_km2"] = float((n_per_row * row_area_hmax).sum())
        depth_vals = np.where(valid, hmax, np.nan)
        stats = _depth_stats(depth_vals)
        result.update({f"sfincs_depth_{k}": v for k, v in stats.items()})

    # -- SFINCS boundary-cell distance to real land --
    sfincs_model_dir = tile_dir / "sfincs_model"
    try:
        sf = SfincsModel(root=str(sfincs_model_dir), mode="r")
        retry_transient_io(sf.grid.read)
        sfincs_mask = sf.grid.data["mask"].values
        sfincs_transform = sf.grid.data.raster.transform
        sfincs_crs = sf.grid.data.raster.crs
        shape = sfincs_mask.shape
        mog = np.empty(shape, dtype=np.float32)
        with retry_transient_io(rasterio.open, native_mask_path) as src:
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


def summarize_tile(
    tile_id: str, root: Path, base_dir_name: str, models: list[str], tile_set: str | None = None,
) -> dict[str, dict]:
    """Returns {model_name: result_dict} for each of `models` - each dict is
    already a complete, standalone summary_{model}.json payload (tile_id,
    optional set, domain stats, and that model's own fields)."""
    tile_dir = root / base_dir_name / tile_id
    native_mask_path = tile_dir / "inputs" / "mask.tif"

    domain = summarize_domain(native_mask_path)
    header: dict = {"tile_id": tile_id}
    if tile_set is not None:
        header["set"] = tile_set
    header.update(domain)

    extent_models = [m for m in models if m in ("bathtub", "eikonal")]
    sfincs_wet = _load_sfincs_wet_subgrid(tile_dir, native_mask_path) if extent_models else None

    results: dict[str, dict] = {}
    for name in extent_models:
        results[name] = {**header, **summarize_extent_model(name, tile_dir, native_mask_path, sfincs_wet)}
    if "sfincs" in models:
        results["sfincs"] = {**header, **summarize_sfincs(tile_dir, native_mask_path)}
    return results


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--models", default="bathtub,eikonal,sfincs",
                         help="comma-separated subset to (re-)summarize - writes one summary_{model}.json per "
                              "entry, independent of whether the other models have been run for this tile yet")
    parser.add_argument("--set", required=False, default=None, choices=["A", "B"],
                         help="optional bookkeeping label for two-set batches")
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v4")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ("bathtub", "eikonal", "sfincs"):
            parser.error(f"--models: unknown model {m!r} (choose from bathtub, eikonal, sfincs)")

    root = read_root(Path(args.config))
    results = summarize_tile(args.tile_id, root, args.base_dir_name, models, args.set)

    out_dir = root / args.base_dir_name / args.tile_id / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_line = []
    for name, result in results.items():
        out_path = out_dir / f"summary_{name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")
        km2 = result.get(f"{name}_km2")
        summary_line.append(f"{name}={km2}")
    print(f"tile {args.tile_id}: " + ", ".join(summary_line) + " km2")


if __name__ == "__main__":
    main()
