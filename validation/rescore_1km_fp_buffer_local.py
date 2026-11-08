"""Side analysis: england_local_input's CSI under a 1 km false-positive buffer.

Native-CRS counterpart to rescore_1km_fp_buffer.py's England-FZ3 section (same
Bates et al. 2021 / Wing et al. 2017 restriction: false positives only count
within 1 km of the benchmark's own flood extent) for a local-input scenario
whose merged raster stays in its own native projected CRS (rescore_england_fz3_local.py
- never reprojected onto the global grid, TODO.md). Simpler than the EPSG:4326
version's buffer-mask step: a projected CRS's pixel size is already in metres,
no per-block latitude correction needed.

Does NOT touch src/validation.py, src/merge.py or the production scorers - read-only,
ad-hoc, reuses their unmodified building blocks. Prints a before/after table; writes
nothing.

Defaults reproduce England's original behaviour exactly (bare --config call, no other
flags). --region picks which of the benchmark's named regions bounds the scoring window -
required whenever the benchmark has more than one (e.g. wales_nrw_floodzone_seas's
severn_estuary/menai_strait/wales); England's single-region benchmark doesn't need it.

Usage:
    python validation/rescore_1km_fp_buffer_local.py --config <england_local_input_materialized.yml>
    python validation/rescore_1km_fp_buffer_local.py --config <wales_local_input_materialized.yml> \\
        --benchmark-key wales_nrw_floodzone_seas --region severn_estuary
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BLOCK_SIZE = 8192
_FP_BUFFER_KM = 1.0


def _fp_buffer_mask(fraction: np.ndarray, out_transform, buffer_km: float) -> np.ndarray:
    """True within `buffer_km` (real metres) of any benchmark-wet cell (fraction > 0).

    Projected native CRS: pixel size is already in metres, no latitude correction.
    """
    bench_wet = fraction > 1e-9
    if not bench_wet.any():
        return np.zeros(fraction.shape, dtype=bool)
    if bench_wet.all():
        return np.ones(fraction.shape, dtype=bool)
    dy_km = abs(out_transform.e) / 1000.0
    dx_km = abs(out_transform.a) / 1000.0
    dist_km = ndimage.distance_transform_edt(~bench_wet, sampling=(dy_km, dx_km))
    return dist_km <= buffer_km


def _score_one_rp(
    rp, slr, cfg, gfm_catalog, native_crs, native_dir, spec, full_gdf, full_sindex, country_iso, region_name,
) -> dict | None:
    val_cfg = cfg["validation"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    region = region_name if region_name is not None else next(iter(spec.regions))
    region_bbox = spec.regions[region]
    region_bbox_native = transform_bounds("EPSG:4326", native_crs, *region_bbox)

    raster_path = native_dir / f"waterdepth_{rp}_{slr}.tif"
    tp = fp = fp_unbuffered = fn = tn = 0.0
    n_processed = 0

    with retry_transient_io(rasterio.open, raster_path) as src:
        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
        window = rasterio.windows.from_bounds(*region_bbox_native, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            return None

        for row_off in range(0, window.height, _BLOCK_SIZE):
            bh = min(_BLOCK_SIZE, window.height - row_off)
            for col_off in range(0, window.width, _BLOCK_SIZE):
                bw = min(_BLOCK_SIZE, window.width - col_off)
                block_window = rasterio.windows.Window(
                    window.col_off + col_off, window.row_off + row_off, bw, bh,
                )
                depth = src.read(1, window=block_window)
                out_transform = src.window_transform(block_window)

                model_domain = v.model_domain_mask(depth, nodata=src.nodata)
                if not model_domain.any():
                    continue
                n_processed += 1

                native_bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], out_transform))
                bbox_4326 = list(transform_bounds(native_crs, "EPSG:4326", *native_bbox))

                candidate_idx = list(full_sindex.query(box(*bbox_4326), predicate="intersects"))
                block_bench_gdf = full_gdf.iloc[candidate_idx]
                fraction = v.benchmark_fraction_from_vector(
                    block_bench_gdf, out_transform, native_crs, depth.shape, supersample=supersample,
                )
                del block_bench_gdf

                not_water, domain_mask = v.window_scoring_domain(
                    gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                    iso_lookup, country_iso, bbox_4326, out_transform, model_domain,
                    geogunit_ids=spec.geogunit_ids, dst_crs=native_crs,
                )

                model_wet = model_domain & (depth > primary_threshold_m) & not_water
                c_tp, c_fp_unbuf, c_fn, c_tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, pixel_area_km2)
                buf = _fp_buffer_mask(fraction, out_transform, _FP_BUFFER_KM)
                _, c_fp, _, _ = v.confusion_counts_soft(model_wet, fraction, domain_mask & buf, pixel_area_km2)

                tp += c_tp
                fp += c_fp
                fp_unbuffered += c_fp_unbuf
                fn += c_fn
                tn += c_tn

    if n_processed == 0:
        return None
    baseline = v.metrics_from_counts(tp, fp_unbuffered, fn, tn)
    buffered = v.metrics_from_counts(tp, fp, fn, tn)
    return {
        "return_period": rp, "waterlevel_name": slr,
        "HR": buffered["HR"],  # fp-independent
        "FAR_baseline": baseline["FAR"], "FAR_1km_fp_buffer": buffered["FAR"],
        "CSI_baseline": baseline["CSI"], "CSI_1km_fp_buffer": buffered["CSI"],
        "bias_baseline": baseline["bias"], "bias_1km_fp_buffer": buffered["bias"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-key", default="england_ea_floodzone3_sea")
    parser.add_argument("--country-iso", default="GBR")
    parser.add_argument("--rps", nargs="+", default=["RP100", "RP250"])
    parser.add_argument("--region", default=None,
                        help="named region bounding the scoring window - required if the "
                             "benchmark has more than one (e.g. wales_nrw_floodzone_seas)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]

    _catalog_paths = cfg["paths"]["hydromt_data_catalog"]
    if isinstance(_catalog_paths, str):
        _catalog_paths = [_catalog_paths]
    gfm_catalog = get_data_catalog([_REPO_ROOT / p for p in _catalog_paths], root=cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    spec = v.load_benchmark_spec(bench_catalog, args.benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    if full_gdf.empty:
        sys.exit(f"{args.benchmark_key}: empty after filtering - nothing to score.")
    full_sindex = full_gdf.sindex

    native_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "local_native"
    with retry_transient_io(rasterio.open, native_dir / f"waterdepth_{args.rps[0]}_{slr}.tif") as probe:
        native_crs = probe.crs

    rows = []
    for rp in args.rps:
        print(f"=== {rp} ===", flush=True)
        row = _score_one_rp(
            rp, slr, cfg, gfm_catalog, native_crs, native_dir, spec, full_gdf, full_sindex,
            args.country_iso, args.region,
        )
        if row is not None:
            rows.append(row)
            print(f"  {row}")

    if not rows:
        print("No rows produced.")
        return
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
