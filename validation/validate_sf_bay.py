"""San Francisco Bay flood-depth validation against USGS CoSMoS (RP100,
no SLR / "existing conditions", W100/SLR000), at TWO resolutions:

  A) "coarsen" - the CoSMoS depth mosaic (2m, EPSG:26910) averaged onto the
     model's own native grid (~30m, EPSG:4326) - apples-to-apples at model
     resolution.
  B) "refine" - the model's own depth output bilinear-resampled onto
     CoSMoS's native 2m grid - apples-to-apples at benchmark resolution.

At each resolution, computes the standard confusion-matrix metrics (CSI,
HT/POD, FAR, bias) at the model's own primary_threshold_m wet-cell
threshold, PLUS continuous-depth RMSE/MAE - both over the whole evaluation
domain and restricted to benchmark-flooded pixels only (where the ground-
truth CoSMoS data itself shows wet - the depth-accuracy question is only
meaningful where there's real flooding to compare against).

Unlike every other benchmark in validate_country.py (vector or categorical-
raster hazard extent, scored via the soft/fractional-coverage machinery in
src/validation.py), CoSMoS's SF Bay product is a continuous DEPTH raster at
much finer resolution than the model grid, and the user wants BOTH an
up-sampled and a down-sampled comparison rather than a single fractional-
coverage score - that's genuinely different machinery (explicit reprojection
to one common grid per direction, then plain array-level confusion counts
and RMSE/MAE), not a variant of the existing soft-fraction path; this script
is SF Bay's own driver/CLI (there is no reusable version in validate_country.py).

Evaluation domain (2026-10-15, revised): NOT the benchmark raster's own
valid-pixel footprint - confirmed live that CoSMoS depth rasters only carry
a value WITHIN the mapped flood extent itself (nodata means "outside the
hazard zone", not "assessed, dry"), so that footprint is ~100% benchmark-wet
by construction and makes FAR/bias vacuous (no real dry ground to test a
false alarm against). The domain is instead the UNION of all 9 counties'
own RECTANGULAR raster extents (sf_bay_county_bbox_union.gpkg, built by
build_sf_bay_benchmark_mosaic.py's sibling step) - each county's modeled
coastal-zone footprint, confirmed far larger (2500-9700 km^2 each) than the
flood extent alone. CoSMoS nodata WITHIN this union is now correctly
treated as "modeled, found dry" (depth=0), not excluded - this is what
actually gives FAR/bias real negative-space to measure against. Permanent
water (DeltaDTM land/ocean/lake/river mask, validation.permanent_water_source)
is still excluded from the domain entirely at both resolutions, same as
every other benchmark - CoSMoS's depth values include the Bay's own real
bathymetric depth, not just new flood depth, and scoring that as "model
under-prediction" would be meaningless.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # validation/ (for _chunks)
from _chunks import _chunks_overlapping_bbox, _mosaic_read  # noqa: E402


def _rasterize_surveyed_domain(surveyed_domain_path: Path, out_transform, out_shape, out_crs) -> np.ndarray:
    """Rasterize the surveyed-domain polygon(s) onto the given output grid.

    Parameters
    ----------
    surveyed_domain_path : Path
        Path to the surveyed-domain vector file (e.g. the 9-county bbox
        union gpkg).
    out_transform : affine.Affine
        Affine transform of the destination grid.
    out_shape : tuple[int, int]
        Shape (rows, cols) of the destination grid.
    out_crs : rasterio.crs.CRS
        CRS of the destination grid.

    Returns
    -------
    np.ndarray
        Boolean mask, True inside the surveyed domain.
    """
    gdf = gpd.read_file(surveyed_domain_path).to_crs(out_crs)
    return rasterize(
        [(geom, 1) for geom in gdf.geometry], out_shape=out_shape,
        transform=out_transform, fill=0, dtype="uint8",
    ).astype(bool)


def _permanent_water_mask_lean(land_use: np.ndarray, exclude_codes) -> np.ndarray:
    """Same semantics as src/validation.py's permanent_water_mask, but avoids
    np.isin's internal temporary arrays - at the SF Bay 2m grid's ~1.7 billion
    pixels, np.isin's own machinery (built for an arbitrary-length second
    array, overkill for 3 known codes) was enough to exhaust available memory
    on its own, on top of the several other full-resolution arrays already
    live at that point."""
    out = np.zeros(land_use.shape, dtype=bool)
    for code in exclude_codes:
        out |= land_use == code
    return out

NODATA_SENTINEL = -9999.0
# SF Bay bbox (EPSG:4326) - same footprint the v2.1 sfb_fldhazd benchmark covers.
BAY_BBOX_4326 = (-122.64, 37.40, -121.88, 38.28)


def _read_model_bay(chunk_paths: list[Path], bbox):
    """Read the model's merged waterdepth over `bbox` (EPSG:4326) from every overlapping chunk.

    One chunk (the case for SF Bay with 5deg chunks): a windowed read, exactly as before.
    Several chunks (smaller chunk_size_deg, or a bbox across a chunk edge): mosaicked with
    validation/_chunks._mosaic_read, the same path validate_country.py uses.

    Returns
    -------
    tuple
        (depth_m float64, transform, crs, nodata).
    """
    if not chunk_paths:
        raise FileNotFoundError(f"no merged waterdepth chunk overlaps the SF Bay bbox {bbox}")
    if len(chunk_paths) == 1:
        return _read_model_window(chunk_paths[0], bbox)
    mosaic, transform = _mosaic_read(chunk_paths, list(bbox))
    with rasterio.open(chunk_paths[0]) as src:
        crs = src.crs
    return mosaic.astype("float64"), transform, crs, -9999.0


def _read_model_window(path: Path, bbox):
    """Windowed-read the model chunk to `bbox` (EPSG:4326), return (depth_m, transform, crs, nodata)."""
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
    """Like src/validation.py's read_permanent_water_mask, but with a caller-chosen dst_crs
    (that function hardcodes EPSG:4326, which doesn't fit this script's EPSG:26910 2m grid)."""
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

    # ── Direction A: benchmark averaged ("coarsened") onto the model's own grid ──
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
    # Within the surveyed footprint, CoSMoS nodata means "modeled, found dry" - fill it
    # as a real 0.0 depth rather than excluding it, so it can register as a TN or FP.
    bench_on_model = np.where((bench_on_model == NODATA_SENTINEL) & surveyed_a, 0.0, bench_on_model)
    domain_a = (model_depth_native != NODATA_SENTINEL) & surveyed_a & ~water_on_model
    model_wet_a = model_depth_native > threshold_m
    bench_wet_a = bench_on_model > threshold_m
    results["A_coarsen_to_model_grid"] = {
        "resolution": f"{abs(model_transform_native.a):.7f} deg (~model native, {model_shape_native})",
        **_confusion_metrics(model_wet_a, bench_wet_a, domain_a),
        **_depth_error_metrics(model_depth_native, bench_on_model, domain_a, bench_wet_a),
    }

    # ── Direction B: model bilinear-resampled ("refined") onto the benchmark's 2m grid ──
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
