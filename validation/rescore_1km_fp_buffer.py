"""Side analysis: England/Wales CSI under a 1 km false-positive buffer (Bates et al., 2021).

Bates et al. (2021) (following Wing et al., 2017) restrict false positives to within
1 km of the benchmark's own flood extent, and drop false positives caused by a
pluvial/surface-water component. This GFM has no pluvial component (coastal/tidal
eikonal solver only - rivers and lakes are masked out, not simulated as a separate
mechanism), so only the 1 km restriction is reproducible here; the pluvial clause is
not applicable.

Does NOT touch src/validation.py or the production scorers (rescore_england_fz3.py,
validate_england_fz2_band.py, validate_country.py) - this is read-only, ad-hoc, and
reuses their unmodified building blocks. tp/fn/tn are the existing (unbuffered)
confusion_counts_soft result; only fp is recomputed with domain_mask additionally
restricted to a 1 km buffer around benchmark-wet cells (fraction > 0), via
scipy.ndimage.distance_transform_edt in real metres (per-block centre-latitude pixel
size - a block spans at most a few km of latitude, so this is a small approximation,
same spirit as pixel_area_km2_grid's own per-row approximation elsewhere). The buffer
is computed independently per processing block (8192 px for England, per-chunk for
Wales): a benchmark polygon within 1 km of a block edge can make this undercount the
buffer there slightly, same edge-effect the existing coastline_tolerance_m
diagnostic already accepts.

Prints a before/after table; writes nothing (no CSVs, no agreement rasters).

Usage:
    python validation/rescore_1km_fp_buffer.py --config snakemake_workflow/config/config.yml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from scipy import ndimage
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COUNTRY_ISO = "GBR"
_FP_BUFFER_KM = 1.0
_KM_PER_DEG = 111.32  # matches src/pipeline_plots.py's own constant


def _fp_buffer_mask(fraction: np.ndarray, out_transform, buffer_km: float) -> np.ndarray:
    """True within `buffer_km` (real metres) of any benchmark-wet cell (fraction > 0)."""
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


def _confusion_buffered(model_wet, fraction, domain_mask, weight, out_transform):
    """(tp, fp, fn, tn): unbuffered tp/fn/tn, fp restricted to the 1 km buffer."""
    tp, fp_unbuffered, fn, tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, weight)
    buf = _fp_buffer_mask(fraction, out_transform, _FP_BUFFER_KM)
    _, fp, _, _ = v.confusion_counts_soft(model_wet, fraction, domain_mask & buf, weight)
    return tp, fp, fn, tn, fp_unbuffered


class _Acc:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = self.fp_unbuffered = 0.0

    def add(self, tp, fp, fn, tn, fp_unbuffered):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.fp_unbuffered += fp_unbuffered

    def row(self, label):
        buffered = v.metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        baseline = v.metrics_from_counts(self.tp, self.fp_unbuffered, self.fn, self.tn)
        return {
            "region": label,
            "CSI_baseline": baseline["CSI"], "CSI_1km_fp_buffer": buffered["CSI"],
            "FAR_baseline": baseline["FAR"], "FAR_1km_fp_buffer": buffered["FAR"],
            "HR": buffered["HR"],  # HR does not depend on fp, so unchanged by the buffer
            "fp_km2_baseline": self.fp_unbuffered, "fp_km2_1km_fp_buffer": self.fp,
        }


def _score_england_fz3(cfg, gfm_catalog, bench_catalog, val_cfg, slr) -> list[dict]:
    benchmark_key = "england_ea_floodzone3_sea"
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    block_size = 8192
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    if full_gdf.empty:
        print(f"{benchmark_key}: empty after filtering - skipped.")
        return []
    full_sindex = full_gdf.sindex
    region, region_bbox = next(iter(spec.regions.items()))

    rows = []
    for rp in ("RP100", "RP250"):
        acc = _Acc()
        chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
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
                        fraction = v.benchmark_fraction_from_vector(
                            full_gdf.iloc[candidate_idx], out_transform, "EPSG:4326", depth.shape,
                            supersample=supersample,
                        )
                        not_water, domain_mask = v.window_scoring_domain(
                            gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                            iso_lookup, _COUNTRY_ISO, bbox, out_transform, model_domain,
                            geogunit_ids=spec.geogunit_ids,
                        )
                        model_wet = model_domain & (depth > primary_threshold_m) & not_water
                        area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
                        acc.add(*_confusion_buffered(model_wet, fraction, domain_mask, area_km2, out_transform))
            if chunk_has_data:
                n_processed += 1
            print(f"    FZ3 {rp} chunk {chunk_path.name}: done", flush=True)
        print(f"  FZ3 {rp}: {n_processed}/{len(chunk_paths)} chunk(s) had real model data.")
        if n_processed:
            rows.append({**acc.row(f"England FZ3 ({rp})"), "benchmark": "england_ea_floodzone3_sea"})
    return rows


def _score_england_fz2_band(cfg, gfm_catalog, bench_catalog, val_cfg, slr) -> list[dict]:
    benchmark_key = "england_ea_floodzone2_sea"
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    if full_gdf.empty:
        print(f"{benchmark_key}: empty after filtering - skipped.")
        return []
    full_sindex = full_gdf.sindex
    region, region_bbox = next(iter(spec.regions.items()))

    rp_inner, rp_outer = "RP100", "RP1000"
    outer_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp_outer, slr)
    acc = _Acc()
    n_processed = 0
    for outer_path in outer_paths:
        chunk_id = outer_path.name.split("_")[1]
        inner_path = merged_chunks_dir / f"waterdepth_{chunk_id}_{rp_inner}_{slr}.tif"
        if not inner_path.exists():
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
        fraction = v.benchmark_fraction_from_vector(
            full_gdf.iloc[candidate_idx], out_transform, "EPSG:4326", depth_outer.shape, supersample=supersample,
        )
        not_water, domain_mask = v.window_scoring_domain(
            gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
            iso_lookup, _COUNTRY_ISO, bbox, out_transform, model_domain, geogunit_ids=spec.geogunit_ids,
        )
        wet_outer = model_domain & (depth_outer > primary_threshold_m) & not_water
        wet_inner = model_domain & (depth_inner > primary_threshold_m) & not_water
        model_band = wet_outer & ~wet_inner
        area_km2 = pixel_area_km2_grid(out_transform, depth_outer.shape[1], depth_outer.shape[0])
        acc.add(*_confusion_buffered(model_band, fraction, domain_mask, area_km2, out_transform))
        print(f"    FZ2 band chunk {chunk_id}: done", flush=True)
    print(f"  FZ2 band: {n_processed}/{len(outer_paths)} chunk(s) had real model data in both RPs.")
    if not n_processed:
        return []
    return [{**acc.row("England FZ2 band (RP100-RP1000)"), "benchmark": "england_ea_floodzone2_sea"}]


def _score_wales(cfg, gfm_catalog, bench_catalog, val_cfg, slr) -> list[dict]:
    benchmark_key = "wales_nrw_floodzone_seas"
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
    full_gdf = v.load_benchmark_4326(bench_catalog, spec)
    if full_gdf.empty:
        print(f"{benchmark_key}: empty after filtering - skipped.")
        return []
    full_sindex = full_gdf.sindex

    rows = []
    for rp in ("RP100", "RP250"):
        for region, region_bbox in spec.regions.items():
            acc = _Acc()
            chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
            n_processed = 0
            for chunk_path in chunk_paths:
                with retry_transient_io(rasterio.open, chunk_path) as src:
                    window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
                    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    window = window.round_offsets().round_lengths()
                    if window.width <= 0 or window.height <= 0:
                        continue
                    depth = src.read(1, window=window)
                    out_transform = src.window_transform(window)
                model_domain = v.model_domain_mask(depth, nodata=-9999.0)
                if not model_domain.any():
                    continue
                n_processed += 1
                bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], out_transform))
                candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
                fraction = v.benchmark_fraction_from_vector(
                    full_gdf.iloc[candidate_idx], out_transform, "EPSG:4326", depth.shape, supersample=supersample,
                )
                not_water, domain_mask = v.window_scoring_domain(
                    gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                    iso_lookup, _COUNTRY_ISO, bbox, out_transform, model_domain, geogunit_ids=spec.geogunit_ids,
                )
                model_wet = model_domain & (depth > primary_threshold_m) & not_water
                area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
                acc.add(*_confusion_buffered(model_wet, fraction, domain_mask, area_km2, out_transform))
            print(f"  Wales {region} {rp}: {n_processed}/{len(chunk_paths)} chunk(s) had real model data.")
            if n_processed:
                rows.append({**acc.row(f"Wales - {region} ({rp})"), "benchmark": benchmark_key})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--merged-outputs", default=None,
                        help="override postprocessing.merged_outputs (config.yml currently points at "
                             "merged_results_v2, which does not exist on disk yet - the completed run "
                             "lives under merged_results)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.merged_outputs:
        cfg["postprocessing"]["merged_outputs"] = args.merged_outputs
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    rows = []
    print("=== England FZ3 ===", flush=True)
    rows += _score_england_fz3(cfg, gfm_catalog, bench_catalog, val_cfg, slr)
    print("=== England FZ2 band ===", flush=True)
    rows += _score_england_fz2_band(cfg, gfm_catalog, bench_catalog, val_cfg, slr)
    print("=== Wales ===", flush=True)
    rows += _score_wales(cfg, gfm_catalog, bench_catalog, val_cfg, slr)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
