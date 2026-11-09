"""Plot side-by-side agreement maps: a local-input scenario vs its global-input counterpart.

Builds the same agree/under/over category raster validate_country.py writes
(_CAT_DRY/_CAT_AGREE/_CAT_UNDER/_CAT_OVER), directly from each scenario's own
merged output (native-CRS for local, the standard EPSG:4326 chunks for global -
rescore_england_fz3_local.py / rescore_england_fz3.py's block loops, extended to
also fill a category array instead of just accumulating confusion counts). The
1 km false-positive buffer (rescore_1km_fp_buffer.py) only changes which false
positives are COUNTED in the aggregate metrics - it does not change this
pixel-level classification, so the map looks the same either way; the annotated
HR/FAR/CSI/bias use the buffered metrics for consistency with TODO.md's reported
numbers. Local's category raster (small, uint8) is reprojected to EPSG:4326 only
for display, alongside global's - the underlying scoring never reprojects the
big depth raster (TODO.md).

Defaults reproduce England's original behaviour exactly. --region picks which of the
benchmark's named regions bounds the scoring window - required whenever the benchmark
has more than one (e.g. wales_nrw_floodzone_seas's severn_estuary/menai_strait/wales).

Usage:
    python validation/plot_local_vs_global_agreement.py \\
        --local-config <england_local_input_materialized.yml> \\
        --global-config <england_global_input_materialized.yml>
    python validation/plot_local_vs_global_agreement.py \\
        --local-config <wales_local_input_materialized.yml> \\
        --global-config <wales_global_input_materialized.yml> \\
        --benchmark-key wales_nrw_floodzone_seas --region severn_estuary
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox  # noqa: E402
from plot_agreement_map import _draw_agreement_panel, _draw_annotation_box, _format_metrics_lines, _legend_handles  # noqa: E402
from rescore_1km_fp_buffer_local import _fp_buffer_mask as _fp_buffer_mask_native  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BLOCK_SIZE = 8192
_CAT_DRY, _CAT_AGREE, _CAT_UNDER, _CAT_OVER = 0, 1, 2, 3
_CAT_NODATA = 255
_MAX_CATEGORY_PIXELS = 4_000_000_000  # one uint8 category array; all of England at 1 m is ~1e11


def _crop_to_valid(category: np.ndarray, transform) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Crop to the bounding box of non-nodata pixels - the scoring window is often much
    larger than where the model actually has data (e.g. a whole benchmark region's bbox
    intersected with one 5deg chunk, for a model covering only a handful of small tiles)."""
    valid = category != _CAT_NODATA
    if not valid.any():
        return category, transform
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    cropped = category[r0:r1, c0:c1]
    cropped_transform = transform * rasterio.Affine.translation(c0, r0)
    return cropped, cropped_transform


def _fill_category(category, model_wet, benchmark_wet, domain_mask, row_slice, col_slice) -> None:
    block = np.full(model_wet.shape, _CAT_DRY, dtype="uint8")
    block[domain_mask & model_wet & ~benchmark_wet] = _CAT_OVER
    block[domain_mask & ~model_wet & benchmark_wet] = _CAT_UNDER
    block[domain_mask & model_wet & benchmark_wet] = _CAT_AGREE
    block[~domain_mask] = _CAT_NODATA
    category[row_slice, col_slice] = block


def _category_and_metrics_local(
    cfg, gfm_catalog, bench_catalog, rp, benchmark_key, country_iso, region_name, bbox=None,
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS, dict]:
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    full_sindex = full_gdf.sindex
    region = region_name if region_name is not None else next(iter(spec.regions))
    region_bbox = spec.regions[region] if bbox is None else bbox

    native_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "local_native"
    raster_path = native_dir / f"waterdepth_{rp}_{slr}.tif"

    tp = fp = fn = tn = 0.0
    with retry_transient_io(rasterio.open, raster_path) as src:
        native_crs = src.crs
        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
        region_bbox_native = transform_bounds("EPSG:4326", native_crs, *region_bbox)
        window = rasterio.windows.from_bounds(*region_bbox_native, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        window = window.round_offsets().round_lengths()
        out_transform = src.window_transform(window)
        if window.height * window.width > _MAX_CATEGORY_PIXELS:
            raise ValueError(
                f"local window is {window.height} x {window.width} px at native resolution, too large to "
                f"classify in memory: pass a smaller --bbox"
            )
        category = np.full((window.height, window.width), _CAT_NODATA, dtype="uint8")

        for row_off in range(0, window.height, _BLOCK_SIZE):
            bh = min(_BLOCK_SIZE, window.height - row_off)
            for col_off in range(0, window.width, _BLOCK_SIZE):
                bw = min(_BLOCK_SIZE, window.width - col_off)
                block_window = rasterio.windows.Window(window.col_off + col_off, window.row_off + row_off, bw, bh)
                depth = src.read(1, window=block_window)
                block_transform = src.window_transform(block_window)

                model_domain = v.model_domain_mask(depth, nodata=src.nodata)
                if not model_domain.any():
                    continue

                native_bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], block_transform))
                bbox_4326 = list(transform_bounds(native_crs, "EPSG:4326", *native_bbox))
                candidate_idx = list(full_sindex.query(box(*bbox_4326), predicate="intersects"))
                fraction = v.benchmark_fraction_from_vector(
                    full_gdf.iloc[candidate_idx], block_transform, native_crs, depth.shape, supersample=supersample,
                )
                not_water, domain_mask = v.window_scoring_domain(
                    gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                    iso_lookup, country_iso, bbox_4326, block_transform, model_domain,
                    geogunit_ids=spec.geogunit_ids, dst_crs=native_crs,
                )
                model_wet = model_domain & (depth > primary_threshold_m) & not_water
                benchmark_wet_display = v.wet_mask_from_fraction(fraction, wet_fraction)
                _fill_category(category, model_wet, benchmark_wet_display, domain_mask,
                               slice(row_off, row_off + bh), slice(col_off, col_off + bw))

                c_tp, _c_fp_unbuffered, c_fn, c_tn = v.confusion_counts_soft(
                    model_wet, fraction, domain_mask, pixel_area_km2,
                )
                buf = _fp_buffer_mask_native(fraction, block_transform, 1.0)
                _, c_fp, _, _ = v.confusion_counts_soft(model_wet, fraction, domain_mask & buf, pixel_area_km2)
                tp += c_tp
                fp += c_fp
                fn += c_fn
                tn += c_tn

    metrics = v.metrics_from_counts(tp, fp, fn, tn)
    category, out_transform = _crop_to_valid(category, out_transform)
    return category, out_transform, native_crs, metrics


def _category_and_metrics_global(
    cfg, gfm_catalog, bench_catalog, rp, benchmark_key, country_iso, region_name, bbox=None,
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS, dict]:
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    full_sindex = full_gdf.sindex
    region = region_name if region_name is not None else next(iter(spec.regions))
    region_bbox = spec.regions[region] if bbox is None else bbox
    chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)

    pieces = []  # (category_block, transform) - mosaicked below
    tp = fp = fn = tn = 0.0
    for chunk_path in chunk_paths:
        with retry_transient_io(rasterio.open, chunk_path) as src:
            window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            window = window.round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                continue
            crs = src.crs
            full_transform = src.window_transform(window)
            category = np.full((window.height, window.width), _CAT_NODATA, dtype="uint8")

            for row_off in range(0, window.height, _BLOCK_SIZE):
                bh = min(_BLOCK_SIZE, window.height - row_off)
                for col_off in range(0, window.width, _BLOCK_SIZE):
                    bw = min(_BLOCK_SIZE, window.width - col_off)
                    block_window = rasterio.windows.Window(window.col_off + col_off, window.row_off + row_off, bw, bh)
                    depth = src.read(1, window=block_window)
                    block_transform = src.window_transform(block_window)
                    model_domain = v.model_domain_mask(depth, nodata=-9999.0)
                    if not model_domain.any():
                        continue
                    bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], block_transform))
                    candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
                    fraction = v.benchmark_fraction_from_vector(
                        full_gdf.iloc[candidate_idx], block_transform, "EPSG:4326", depth.shape, supersample=supersample,
                    )
                    not_water, domain_mask = v.window_scoring_domain(
                        gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                        iso_lookup, country_iso, bbox, block_transform, model_domain, geogunit_ids=spec.geogunit_ids,
                    )
                    model_wet = model_domain & (depth > primary_threshold_m) & not_water
                    benchmark_wet_display = v.wet_mask_from_fraction(fraction, wet_fraction)
                    _fill_category(category, model_wet, benchmark_wet_display, domain_mask,
                                   slice(row_off, row_off + bh), slice(col_off, col_off + bw))

                    from pipeline_plots import pixel_area_km2_grid
                    area_km2 = pixel_area_km2_grid(block_transform, depth.shape[1], depth.shape[0])
                    c_tp, c_fp_unbuf, c_fn, c_tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, area_km2)
                    buf = _fp_buffer_mask_global(fraction, block_transform)
                    _, c_fp, _, _ = v.confusion_counts_soft(model_wet, fraction, domain_mask & buf, area_km2)
                    tp += c_tp
                    fp += c_fp
                    fn += c_fn
                    tn += c_tn
            pieces.append((category, full_transform, crs))

    if not pieces:
        raise RuntimeError(f"no real model data found for {rp} in region bbox {region_bbox}")
    # Only one 5deg chunk practically overlaps our tiny test area at a time; if more than
    # one does, keep the first (mosaicking is out of scope for this plotting helper).
    category, out_transform, crs = pieces[0]
    category, out_transform = _crop_to_valid(category, out_transform)
    metrics = v.metrics_from_counts(tp, fp, fn, tn)
    return category, out_transform, crs, metrics


_KM_PER_DEG = 111.32


def _fp_buffer_mask_global(fraction, out_transform, buffer_km: float = 1.0):
    from scipy import ndimage
    bench_wet = fraction > 1e-9
    if not bench_wet.any():
        return np.zeros(fraction.shape, dtype=bool)
    if bench_wet.all():
        return np.ones(fraction.shape, dtype=bool)
    lat_center = out_transform.f + (fraction.shape[0] / 2.0) * out_transform.e
    dy_km = abs(out_transform.e) * _KM_PER_DEG
    dx_km = abs(np.cos(np.radians(lat_center)) * out_transform.a) * _KM_PER_DEG
    dist_km = ndimage.distance_transform_edt(~bench_wet, sampling=(dy_km, dx_km))
    return dist_km <= buffer_km


def _reproject_category_to_4326(category, transform, crs):
    bounds = rasterio.transform.array_bounds(category.shape[0], category.shape[1], transform)
    dst_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
    # Keep roughly the same pixel count as the native array - plenty for display.
    dst_h, dst_w = category.shape
    px = (dst_bounds[2] - dst_bounds[0]) / dst_w
    py = (dst_bounds[3] - dst_bounds[1]) / dst_h
    dst_transform = rasterio.Affine(px, 0.0, dst_bounds[0], 0.0, -py, dst_bounds[3])
    dst = np.full((dst_h, dst_w), _CAT_NODATA, dtype="uint8")
    reproject(
        source=category, destination=dst, src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs="EPSG:4326", src_nodata=_CAT_NODATA, dst_nodata=_CAT_NODATA,
        resampling=Resampling.nearest,
    )
    return dst, dst_transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-config", required=True)
    parser.add_argument("--global-config", required=True)
    parser.add_argument("--benchmark-key", default="england_ea_floodzone3_sea")
    parser.add_argument("--country-iso", default="GBR")
    parser.add_argument("--region", default=None,
                        help="named region bounding the scoring window - required if the "
                             "benchmark has more than one (e.g. wales_nrw_floodzone_seas)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="score and plot this EPSG:4326 window instead of the whole region (needed for a "
                             "country-scale 1 m local merge, whose full-resolution map does not fit in memory)")
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    local_cfg = load_config(args.local_config)
    global_cfg = load_config(args.global_config)
    val_cfg = local_cfg["validation"]

    local_catalog_paths = local_cfg["paths"]["hydromt_data_catalog"]
    if isinstance(local_catalog_paths, str):
        local_catalog_paths = [local_catalog_paths]
    local_gfm_catalog = get_data_catalog([_REPO_ROOT / p for p in local_catalog_paths], root=local_cfg["paths"]["root"])
    global_catalog_paths = global_cfg["paths"]["hydromt_data_catalog"]
    if isinstance(global_catalog_paths, str):
        global_catalog_paths = [global_catalog_paths]
    global_gfm_catalog = get_data_catalog([_REPO_ROOT / p for p in global_catalog_paths], root=global_cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    rp = args.return_period
    print(f"Scoring+classifying local ({rp})...", flush=True)
    local_cat, local_t, local_crs, local_metrics = _category_and_metrics_local(
        local_cfg, local_gfm_catalog, bench_catalog, rp, args.benchmark_key, args.country_iso, args.region, args.bbox,
    )
    print(f"  local: {local_metrics}")
    print(f"Scoring+classifying global ({rp})...", flush=True)
    global_cat, global_t, global_crs, global_metrics = _category_and_metrics_global(
        global_cfg, global_gfm_catalog, bench_catalog, rp, args.benchmark_key, args.country_iso, args.region, args.bbox,
    )
    print(f"  global: {global_metrics}")

    local_cat_4326, local_t_4326 = _reproject_category_to_4326(local_cat, local_t, local_crs)

    import matplotlib.pyplot as plt
    plots_cfg = val_cfg["plots"]
    colors = plots_cfg["agreement_colors"]
    fig, axes = plt.subplots(1, 2, figsize=(plots_cfg["figsize"][0] * 2, plots_cfg["figsize"][1]), sharey=True)

    for ax, cat, transform, catalog, title, metrics in [
        (axes[0], local_cat_4326, local_t_4326, local_gfm_catalog, "Local input (EA LIDAR + EA boundary)", local_metrics),
        (axes[1], global_cat, global_t, global_gfm_catalog, "Global input (DeltaDTM + COAST-RP)", global_metrics),
    ]:
        tmp_path = Path(tempfile.gettempdir()) / f"plot_local_vs_global_agreement_{abs(hash(title))}.tif"
        profile = {
            "driver": "GTiff", "dtype": "uint8", "nodata": _CAT_NODATA, "width": cat.shape[1],
            "height": cat.shape[0], "count": 1, "crs": "EPSG:4326", "transform": transform,
        }
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(cat, 1)
        _draw_agreement_panel(ax, tmp_path, colors, catalog, val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"])
        tmp_path.unlink(missing_ok=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        _draw_annotation_box(ax, _format_metrics_lines(metrics))
    axes[0].set_ylabel("Latitude")

    region_label = args.region if args.region is not None else "default region"
    fig.suptitle(f"{args.country_iso} agreement vs {args.benchmark_key}, {rp}, SLR_0 ({region_label})", fontsize=12)
    fig.subplots_adjust(wspace=0.04, bottom=0.2, top=0.88)
    fig.legend(handles=_legend_handles(colors, {
        "agree": "Agree (both wet)", "under": "Model under-predicts", "over": "Model over-predicts",
    }), loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

    out_path = Path(args.out) if args.out else Path(val_cfg["output_dir"]).parent / f"agreement_local_vs_global_{rp}_SLR_0.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(plots_cfg["dpi"]), bbox_inches="tight")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
