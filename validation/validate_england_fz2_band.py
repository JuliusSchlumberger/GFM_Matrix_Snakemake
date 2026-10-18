"""England Flood Zone 2 band comparison.

EA's own Flood Zones Product Description (confirmed 2026-10-16 via
WebSearch of environment.data.gov.uk/support/faqs/778338325) defines
Flood Zone 2 (sea) as the INCREMENTAL 0.1%-0.5% AEP band (1-in-1000 to
1-in-200 annual probability), i.e. land between the Zone 3 (~1-in-200)
outline and the 1-in-1000 outline - NOT the cumulative extent out to
1-in-1000. The normal `england_ea_floodzone2_sea` catalog entry
(`comparison_return_periods: [1000]`, scored by the ordinary
validate_country_national_coverage path) compares this thin benchmark band
against the model's FULL CUMULATIVE RP1000 extent, which is why that
comparison gives CSI=0.023/bias=26.2x (see ENGLAND_FZ2_CAVEAT in
compare_against_published.py) - a benchmark-definition mismatch, not a
real model failure. This script tests that theory directly, by building
the model's OWN matching incremental band - wet at RP1000 AND NOT wet at
RP100 (RP100 is the model's nearest native return period below the
benchmark's own ~RP200 inner edge; RP200 itself is not a native scenario)
- and re-scoring against the SAME Flood Zone 2 benchmark polygons.

Intentionally a standalone script, not a new general-purpose code path in
validate_country.py: every other benchmark in the catalog is a plain
single-RP or bracketed-RP (comparison_return_periods) comparison;
subtracting two RPs' own wet masks before scoring is a genuine one-off
need unique to this single benchmark's own band semantics.

Usage:
    python validation/validate_england_fz2_band.py --config snakemake_workflow/config/config.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
import rasterio.windows
from shapely.geometry import box

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_country import _chunks_overlapping_bbox  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_KEY = "england_ea_floodzone2_sea"
_COUNTRY_ISO = "GBR"
_RP_INNER = "RP100"   # model's nearest native RP below the benchmark's real ~RP200 inner edge
_RP_OUTER = "RP1000"  # exact native match for the benchmark's own outer edge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, _BENCHMARK_KEY)
    full_gdf = v.load_benchmark_full(bench_catalog, spec)
    if full_gdf.empty:
        sys.exit(f"{_BENCHMARK_KEY}: empty after filtering - nothing to score.")
    if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
        full_gdf = full_gdf.to_crs(4326)
    full_sindex = full_gdf.sindex

    region, region_bbox = next(iter(spec.regions.items()))
    print(f"{_BENCHMARK_KEY} / {region}: band = model({_RP_OUTER}) minus model({_RP_INNER})")

    outer_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, _RP_OUTER, slr)

    tp = fp = fn = tn = 0.0
    model_band_km2 = benchmark_wet_km2 = 0.0
    n_processed = 0

    for outer_path in outer_paths:
        chunk_id = outer_path.name.split("_")[1]  # waterdepth_{chunk_id}_{RP}_{SLR}.tif
        inner_path = merged_chunks_dir / f"waterdepth_{chunk_id}_{_RP_INNER}_{slr}.tif"
        if not inner_path.exists():
            print(f"  {chunk_id}: missing {_RP_INNER} chunk - skipped.")
            continue

        with retry_transient_io(rasterio.open, outer_path) as src:
            window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            window = window.round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                continue
            depth_outer = src.read(1, window=window)
            out_transform = src.window_transform(window)
        with retry_transient_io(rasterio.open, inner_path) as src:
            depth_inner = src.read(1, window=window)

        model_domain = v.model_domain_mask(depth_outer, nodata=-9999.0) & v.model_domain_mask(depth_inner, nodata=-9999.0)
        if not model_domain.any():
            continue
        n_processed += 1

        bbox = list(rasterio.transform.array_bounds(depth_outer.shape[0], depth_outer.shape[1], out_transform))
        candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
        chunk_bench_gdf = full_gdf.iloc[candidate_idx]
        fraction = v.benchmark_fraction_from_vector(
            chunk_bench_gdf, out_transform, "EPSG:4326", depth_outer.shape, supersample=supersample,
        )

        water_mask = v.read_permanent_water_mask(
            gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth_outer.shape,
        )
        not_water = ~water_mask
        in_country = v.read_country_mask(
            gfm_catalog, geogunit_source, iso_lookup, _COUNTRY_ISO, bbox, out_transform, depth_outer.shape,
            geogunit_ids=spec.geogunit_ids,
        )
        not_water = not_water & in_country
        domain_mask = model_domain & not_water

        wet_outer = model_domain & (depth_outer > primary_threshold_m) & not_water
        wet_inner = model_domain & (depth_inner > primary_threshold_m) & not_water
        model_band = wet_outer & ~wet_inner  # the model's own ~RP200-to-RP1000 incremental band

        area_km2 = pixel_area_km2_grid(out_transform, depth_outer.shape[1], depth_outer.shape[0])
        model_band_km2 += float((model_band * area_km2).sum())
        benchmark_wet_km2 += float((fraction * area_km2 * not_water).sum())

        c_tp, c_fp, c_fn, c_tn = v.confusion_counts_soft(model_band, fraction, domain_mask, area_km2)
        tp += c_tp
        fp += c_fp
        fn += c_fn
        tn += c_tn

    print(f"  {n_processed}/{len(outer_paths)} chunk(s) had real model data in both RPs.")
    metrics = v.metrics_from_counts(tp, fp, fn, tn)
    print(
        f"  model_band_km2={model_band_km2:.1f}  benchmark_wet_km2={benchmark_wet_km2:.1f}\n"
        f"  HR={metrics['HR']:.3f}  FAR={metrics['FAR']:.3f}  CSI={metrics['CSI']:.3f}  bias={metrics['bias']:.3f}"
    )

    out_dir = Path(val_cfg["output_dir"]) / _COUNTRY_ISO
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    out_path = out_dir / "england_fz2_band_result.csv"
    row = {
        "benchmark_key": _BENCHMARK_KEY, "region": region,
        "band_inner_rp": _RP_INNER, "band_outer_rp": _RP_OUTER,
        "model_band_km2": model_band_km2, "benchmark_wet_km2": benchmark_wet_km2,
        "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"], "bias": metrics["bias"],
    }
    atomic_write(out_path, lambda f: pd.DataFrame([row]).to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
