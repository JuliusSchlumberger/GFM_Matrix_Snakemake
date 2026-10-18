"""Re-score England Flood Zone 3 (RP100/RP250 bracket) in bounded memory.

A narrower, memory-safe path than run_multi_rp_summary.py. Chunks are scored in
8192-px blocks because `benchmark_fraction_from_vector` rasterizes at
out_shape*supersample per side: a full 5deg chunk (~18,500 px) at supersample=5
needs a ~8.6 GB uint8 array.

Usage:
    python validation/rescore_england_fz3.py --config snakemake_workflow/config/config.yml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.windows
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_KEY = "england_ea_floodzone3_sea"
_COUNTRY_ISO = "GBR"
_RPS = ["RP100", "RP250"]


def _score_one_rp(rp, slr, cfg, gfm_catalog, bench_catalog, spec, full_gdf, full_sindex) -> dict | None:
    val_cfg = cfg["validation"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    # Not postprocessing.block_size (2048, tuned for raster writes): per-block catalog
    # reads make 2048 too slow. 8192 keeps the supersampled array at ~1.7GB.
    block_size = 8192
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    region, region_bbox = next(iter(spec.regions.items()))
    chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)

    tp = fp = fn = tn = 0.0
    model_wet_km2 = benchmark_wet_km2 = 0.0
    n_processed = 0

    for chunk_path in chunk_paths:
        with retry_transient_io(rasterio.open, chunk_path) as src:
            chunk_window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
            chunk_window = chunk_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            chunk_window = chunk_window.round_offsets().round_lengths()
            if chunk_window.width <= 0 or chunk_window.height <= 0:
                continue
            chunk_has_data = False

            for row_off in range(0, chunk_window.height, block_size):
                bh = min(block_size, chunk_window.height - row_off)
                for col_off in range(0, chunk_window.width, block_size):
                    bw = min(block_size, chunk_window.width - col_off)
                    block_window = rasterio.windows.Window(
                        chunk_window.col_off + col_off, chunk_window.row_off + row_off, bw, bh,
                    )
                    depth = src.read(1, window=block_window)
                    out_transform = src.window_transform(block_window)

                    model_domain = v.model_domain_mask(depth, nodata=-9999.0)
                    if not model_domain.any():
                        continue
                    chunk_has_data = True

                    bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], out_transform))
                    candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
                    block_bench_gdf = full_gdf.iloc[candidate_idx]
                    fraction = v.benchmark_fraction_from_vector(
                        block_bench_gdf, out_transform, "EPSG:4326", depth.shape, supersample=supersample,
                    )
                    del block_bench_gdf

                    not_water, domain_mask = v.window_scoring_domain(
                        gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                        iso_lookup, _COUNTRY_ISO, bbox, out_transform, model_domain,
                        geogunit_ids=spec.geogunit_ids,
                    )

                    model_wet = model_domain & (depth > primary_threshold_m) & not_water
                    area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
                    model_wet_km2 += float((model_wet * area_km2).sum())
                    benchmark_wet_km2 += float((fraction * area_km2 * not_water).sum())

                    c_tp, c_fp, c_fn, c_tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, area_km2)
                    tp += c_tp
                    fp += c_fp
                    fn += c_fn
                    tn += c_tn

        if chunk_has_data:
            n_processed += 1
        print(f"    chunk {chunk_path.name}: done", flush=True)

    print(f"  {rp}: {n_processed}/{len(chunk_paths)} chunk(s) had real model data.")
    if n_processed == 0:
        return None
    metrics = v.metrics_from_counts(tp, fp, fn, tn)
    print(
        f"  {rp}: model_wet_km2={model_wet_km2:.1f} benchmark_wet_km2={benchmark_wet_km2:.1f}\n"
        f"  {rp}: HR={metrics['HR']:.3f} FAR={metrics['FAR']:.3f} CSI={metrics['CSI']:.3f} bias={metrics['bias']:.3f}"
    )
    return {
        "country": _COUNTRY_ISO, "benchmark_key": _BENCHMARK_KEY, "return_period": rp,
        "waterlevel_name": slr, "region": region,
        "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"], "bias": metrics["bias"],
        "model_wet_km2": model_wet_km2, "benchmark_wet_km2": benchmark_wet_km2,
    }


def main() -> None:
    """Score FZ3 at RP100/RP250; write metrics_GBR_england_ea_floodzone3_sea_{RP}_{SLR}.csv per RP."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    spec = v.load_benchmark_spec(bench_catalog, _BENCHMARK_KEY)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    if full_gdf.empty:
        sys.exit(f"{_BENCHMARK_KEY}: empty after filtering - nothing to score.")
    full_sindex = full_gdf.sindex

    rows = []
    for rp in _RPS:
        print(f"=== {rp} ===", flush=True)
        row = _score_one_rp(rp, slr, cfg, gfm_catalog, bench_catalog, spec, full_gdf, full_sindex)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No rows produced - nothing written.")
        return
    write_results(rows, val_cfg["output_dir"])


def write_results(rows: list[dict], output_dir: str | Path) -> list[Path]:
    """Write each RP's row to its own standalone metrics CSV (see validation.standalone_metrics_path)."""
    paths = []
    for row in rows:
        out_path = v.standalone_metrics_path(output_dir, _COUNTRY_ISO, _BENCHMARK_KEY, row["return_period"], row["waterlevel_name"])
        retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
        atomic_write(out_path, lambda f, row=row: pd.DataFrame([row]).to_csv(f, index=False),
                     mode="w", encoding="utf-8", newline="")
        print(f"wrote {out_path}")
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    main()
