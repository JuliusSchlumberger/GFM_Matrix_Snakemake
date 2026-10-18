"""Validate SF Bay flood depth against USGS CoSMoS (RP100, SLR000) at two grids.

A) "coarsen": CoSMoS 2m EPSG:26910 depth averaged onto the model grid (~30m, EPSG:4326).
B) "refine": model depth bilinear-resampled onto the CoSMoS 2m grid.
Each gives HT/FAR/CSI/bias at primary_threshold_m and RMSE/MAE (m) over the whole
domain and over benchmark-wet pixels only.

Domain: union of the 9 counties' rectangular raster extents
(sf_bay_county_bbox_union.gpkg). CoSMoS only has values inside the flood extent,
so its nodata inside the union counts as dry (depth 0). Permanent water is
excluded: CoSMoS depth includes Bay bathymetry.

Usage:
    python validation/validate_sf_bay.py [--config path/to/config.yml] [--out-csv path.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox, _mosaic_read  # noqa: E402


def _rasterize_surveyed_domain(surveyed_domain_path: Path, out_transform, out_shape, out_crs) -> np.ndarray:
    gdf = gpd.read_file(surveyed_domain_path).to_crs(out_crs)
    return rasterize(
        [(geom, 1) for geom in gdf.geometry], out_shape=out_shape,
        transform=out_transform, fill=0, dtype="uint8",
    ).astype(bool)


def _permanent_water_mask_lean(land_use: np.ndarray, exclude_codes) -> np.ndarray:
    """Like validation.permanent_water_mask without np.isin, which runs out of memory at 2m."""
    out = np.zeros(land_use.shape, dtype=bool)
    for code in exclude_codes:
        out |= land_use == code
    return out

NODATA_SENTINEL = -9999.0
# EPSG:4326, footprint of the v2.1 sfb_fldhazd benchmark.
BAY_BBOX_4326 = (-122.64, 37.40, -121.88, 38.28)


def _read_model_bay(chunk_paths: list[Path], bbox):
    if not chunk_paths:
        raise FileNotFoundError(f"no merged waterdepth chunk overlaps the SF Bay bbox {bbox}")
    if len(chunk_paths) == 1:
        return _read_model_window(chunk_paths[0], bbox)
    mosaic, transform = _mosaic_read(chunk_paths, list(bbox))
    with rasterio.open(chunk_paths[0]) as src:
        crs = src.crs
    return mosaic.astype("float64"), transform, crs, -9999.0


def _read_model_window(path: Path, bbox):
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        depth = src.read(1, window=window).astype("float64")
        nodata = src.nodata
        transform = src.window_transform(window)
        crs = src.crs
    return depth, transform, crs, nodata


def _read_benchmark_full(path: Path):
    with rasterio.open(path) as src:
        depth = src.read(1).astype("float32")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    depth = np.where(np.isnan(depth) if nodata is None or np.isnan(nodata) else depth == nodata, NODATA_SENTINEL, depth)
    return depth, transform, crs


def _permanent_water_on_grid(gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, out_shape, out_crs):
    """Like validation.read_permanent_water_mask, which hardcodes EPSG:4326, for any CRS."""
    try:
        lu_da = retry_transient_io(gfm_catalog.get_rasterdataset, permanent_water_source, bbox=bbox).squeeze(drop=True)
    except Exception:
        return np.zeros(out_shape, dtype=bool)
    if lu_da is None or lu_da.size == 0:
        return np.zeros(out_shape, dtype=bool)
    lu_arr = lu_da.values.astype("float32")
    lu_nodata = lu_da.raster.nodata
    dst = np.full(out_shape, -1.0, dtype="float32")
    reproject(
        source=lu_arr, destination=dst,
        src_transform=lu_da.raster.transform, src_crs=lu_da.raster.crs,
        dst_transform=out_transform, dst_crs=out_crs,
        src_nodata=lu_nodata, dst_nodata=-1.0,
        resampling=Resampling.nearest,
    )
    return _permanent_water_mask_lean(dst, tuple(permanent_water_codes))


def _confusion_metrics(model_wet: np.ndarray, bench_wet: np.ndarray, domain: np.ndarray) -> dict:
    d = domain
    tp = float((model_wet & bench_wet & d).sum())
    fp = float((model_wet & ~bench_wet & d).sum())
    fn = float((~model_wet & bench_wet & d).sum())
    tn = float((~model_wet & ~bench_wet & d).sum())
    ht = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"n_domain_px": int(d.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "HT": ht, "FAR": far, "CSI": csi, "bias": bias}


def _depth_error_metrics(model_depth: np.ndarray, bench_depth: np.ndarray, domain: np.ndarray, bench_wet: np.ndarray) -> dict:
    err_all = (model_depth - bench_depth)[domain]
    rmse_all = float(np.sqrt(np.mean(err_all ** 2))) if err_all.size else np.nan
    mae_all = float(np.mean(np.abs(err_all))) if err_all.size else np.nan

    flooded = domain & bench_wet
    err_fl = (model_depth - bench_depth)[flooded]
    rmse_fl = float(np.sqrt(np.mean(err_fl ** 2))) if err_fl.size else np.nan
    mae_fl = float(np.mean(np.abs(err_fl))) if err_fl.size else np.nan

    return {
        "RMSE_all_m": rmse_all, "MAE_all_m": mae_all, "n_all_px": int(err_all.size),
        "RMSE_flooded_m": rmse_fl, "MAE_flooded_m": mae_fl, "n_flooded_px": int(err_fl.size),
    }


def validate_sf_bay(config_path: str, threshold_m: float | None = None) -> dict[str, dict]:
    """Score both directions; `threshold_m` defaults to validation.primary_threshold_m."""
    cfg = load_config(Path(config_path))
    val_cfg = cfg["validation"]
    threshold_m = threshold_m if threshold_m is not None else val_cfg["primary_threshold_m"]
    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]

    benchmark_root = Path(val_cfg["benchmark_root"])
    surveyed_domain_path = benchmark_root / "California" / "flooding" / "sf_bay_county_bbox_union.gpkg"
    benchmark_path = benchmark_root / "California" / "flooding" / "sf_bay_flddepth_slr000_w100_mosaic.tif"
    model_chunk_paths = _chunks_overlapping_bbox(
        list(BAY_BBOX_4326), cfg["postprocessing"]["chunk_size_deg"],
        Path(cfg["postprocessing"]["merged_outputs"]) / "chunks", "RP100", "SLR_0",
    )

    bench_depth_2m, bench_transform_2m, bench_crs_2m = _read_benchmark_full(benchmark_path)
    bench_shape_2m = bench_depth_2m.shape

    model_depth_native, model_transform_native, model_crs, model_nodata = _read_model_bay(model_chunk_paths, BAY_BBOX_4326)
    model_shape_native = model_depth_native.shape
    model_valid_native = model_depth_native != model_nodata if model_nodata is not None else np.isfinite(model_depth_native)
    model_depth_native = np.where(model_valid_native, model_depth_native, NODATA_SENTINEL)

    results: dict[str, dict] = {}

    bench_on_model = np.full(model_shape_native, NODATA_SENTINEL, dtype="float64")
    reproject(
        source=bench_depth_2m, destination=bench_on_model,
        src_transform=bench_transform_2m, src_crs=bench_crs_2m,
        dst_transform=model_transform_native, dst_crs=model_crs,
        src_nodata=NODATA_SENTINEL, dst_nodata=NODATA_SENTINEL,
        resampling=Resampling.average,
    )
    water_on_model = _permanent_water_on_grid(
        gfm_catalog, permanent_water_source, permanent_water_codes,
        list(BAY_BBOX_4326), model_transform_native, model_shape_native, model_crs,
    )
    surveyed_a = _rasterize_surveyed_domain(surveyed_domain_path, model_transform_native, model_shape_native, model_crs)
    # CoSMoS nodata inside the surveyed domain = modeled dry.
    bench_on_model = np.where((bench_on_model == NODATA_SENTINEL) & surveyed_a, 0.0, bench_on_model)
    domain_a = (model_depth_native != NODATA_SENTINEL) & surveyed_a & ~water_on_model
    model_wet_a = model_depth_native > threshold_m
    bench_wet_a = bench_on_model > threshold_m
    results["A_coarsen_to_model_grid"] = {
        "resolution": f"{abs(model_transform_native.a):.7f} deg (~model native, {model_shape_native})",
        **_confusion_metrics(model_wet_a, bench_wet_a, domain_a),
        **_depth_error_metrics(model_depth_native, bench_on_model, domain_a, bench_wet_a),
    }

    model_on_2m = np.full(bench_shape_2m, NODATA_SENTINEL, dtype="float32")
    reproject(
        source=model_depth_native, destination=model_on_2m,
        src_transform=model_transform_native, src_crs=model_crs,
        dst_transform=bench_transform_2m, dst_crs=bench_crs_2m,
        src_nodata=NODATA_SENTINEL, dst_nodata=NODATA_SENTINEL,
        resampling=Resampling.bilinear,
    )
    water_on_2m = _permanent_water_on_grid(
        gfm_catalog, permanent_water_source, permanent_water_codes,
        list(BAY_BBOX_4326), bench_transform_2m, bench_shape_2m, bench_crs_2m,
    )
    surveyed_b = _rasterize_surveyed_domain(surveyed_domain_path, bench_transform_2m, bench_shape_2m, bench_crs_2m)
    bench_depth_2m = np.where((bench_depth_2m == NODATA_SENTINEL) & surveyed_b, 0.0, bench_depth_2m)
    domain_b = (model_on_2m != NODATA_SENTINEL) & surveyed_b & ~water_on_2m
    model_wet_b = model_on_2m > threshold_m
    bench_wet_b = bench_depth_2m > threshold_m
    results["B_refine_model_to_2m"] = {
        "resolution": f"2m (EPSG:26910, {bench_shape_2m})",
        **_confusion_metrics(model_wet_b, bench_wet_b, domain_b),
        **_depth_error_metrics(model_on_2m, bench_depth_2m, domain_b, bench_wet_b),
    }

    return results


def main() -> None:
    """Print the SF Bay results and optionally write them to --out-csv."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    results = validate_sf_bay(args.config)

    import pandas as pd
    df = pd.DataFrame(results).T
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df.to_string())

    if args.out_csv:
        df.to_csv(args.out_csv)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
