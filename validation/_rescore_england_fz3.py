"""Fast, memory-safe re-score of England Flood Zone 3 (RP100/RP250 bracket)
only - NOT a replacement for run_multi_rp_summary.py, a narrower path to the
same two numbers after the benchmark was corrected (2026-10-16) to include
estuarine 'river and sea' features alongside pure 'sea' ones (see
fetch_england_flood_zones.py's own docstring).

Three earlier attempts at just re-running run_multi_rp_summary.py --countries
GBR all failed on this machine: (1) the full run (scoring + plotting) stalled
for 3+ hours with CPU going nearly flat, almost certainly inside
plot_country() rendering the now much denser (73,821-feature) FZ3 vector
layer; (2) a version that skipped plotting but still called
validate_country_national_coverage directly (which loads and holds ALL of
Wales+Scotland+England-FZ2+England-FZ3 - ~180k combined features - in memory
at once) spiked to 14GB RSS; (3) a version mirroring
validate_england_fz2_band.py's per-chunk (but NOT per-block) loop also
spiked, this time to 16GB, on whichever chunk happens to be large/dense
enough that `benchmark_fraction_from_vector`'s supersampled rasterize array
(`out_shape*supersample` in each dimension - confirmed in src/validation.py:
at this model's working resolution, a full un-clipped 5deg postprocessing
chunk can be ~18,500x18,500 px, and at supersample=5 that is a ~92,500x92,500
uint8 array, ~8.6 GB, BEFORE the block-average reshape's own temporary
buffers) is big enough on its own to blow past safe memory - a pre-existing
sizing risk in that shared function for any sufficiently large/unclipped
chunk, not something specific to this benchmark's feature count. Fixed here
by processing each chunk in `postprocessing.block_size`-sized row blocks
(the same block-size convention compute_flood_fraction_chunk.py already uses
for exactly this reason - see that script's own Step 1 comment), so the
supersampled array is always bounded to one block's size regardless of how
large the parent chunk is.

Usage:
    python validation/_rescore_england_fz3.py --config snakemake_workflow/config/config.yml
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
from plotting import pixel_area_km2_grid  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_country import _chunks_overlapping_bbox  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_KEY = "england_ea_floodzone3_sea"
_COUNTRY_ISO = "GBR"
_RPS = ["RP100", "RP250"]


def _score_one_rp(rp, slr, cfg, gfm_catalog, bench_catalog, spec, full_gdf, full_sindex) -> dict | None:
    val_cfg = cfg["validation"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    # NOT cfg["postprocessing"]["block_size"] (2048) - that value is tuned for
    # raster-WRITE memory, unrelated to this script's own supersampled-
    # rasterize cost. 2048 keeps the fine array tiny (~105MB at supersample=5)
    # but confirmed too slow in practice: hundreds of blocks/chunk, each
    # paying its own read_permanent_water_mask/read_country_mask catalog-read
    # overhead, stretched a single rescore past 3 hours. 8192 keeps the fine
    # array at ~1.7GB (8192*5=40960 per side) - still far below the 14-16GB
    # spikes that forced a kill with no blocking at all - while cutting the
    # block count by ~16x.
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

            # Row-block the chunk's own window (same block_size convention as
            # compute_flood_fraction_chunk.py's Step 1) - bounds the
            # supersampled rasterize array in benchmark_fraction_from_vector
            # to one block's size regardless of how large this chunk's own
            # window is (see module docstring).
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

                    water_mask = v.read_permanent_water_mask(
                        gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth.shape,
                    )
                    not_water = ~water_mask
                    in_country = v.read_country_mask(
                        gfm_catalog, geogunit_source, iso_lookup, _COUNTRY_ISO, bbox, out_transform, depth.shape,
                        geogunit_ids=spec.geogunit_ids,
                    )
                    not_water = not_water & in_country
                    domain_mask = model_domain & not_water

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
    full_gdf = v.load_benchmark_full(bench_catalog, spec)
    if full_gdf.empty:
        sys.exit(f"{_BENCHMARK_KEY}: empty after filtering - nothing to score.")
    if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
        full_gdf = full_gdf.to_crs(4326)
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

    out_path = Path(val_cfg["output_dir"]) / "england_fz3_rescore.csv"
    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    atomic_write(out_path, lambda f: pd.DataFrame(rows).to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
