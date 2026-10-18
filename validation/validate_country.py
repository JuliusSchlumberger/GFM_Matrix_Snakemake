"""Score one country's national-coverage coastal benchmarks against model output.

Soft confusion matrix (`v.confusion_counts_soft`) at validation.primary_threshold_m,
per `meta.regions` bbox, plus a blended "ALL" row. CSI_tol/HR_tol/FAR_tol
(`validation.coastline_tolerance_cells`) are a boundary-registration diagnostic
only, read next to pct_disagreement_forgiven. Design: docs/methods_04b_MapsValidation.md.

Writes metrics_{ISO}_{RP}_{SLR}.csv and agreement_{ISO}_{region}_{RP}_{SLR}.tif
per region under {validation.output_dir}/{ISO}/.

Usage:
    python validation/validate_country.py \\
        --config snakemake_workflow/config/config.yml --country GBR
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from tiles import load_tile_grid  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import (  # noqa: E402
    _chunks_overlapping_bbox,
    _mosaic_read,
    _tile_paths_overlapping_bbox,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Ordered so Resampling.max gives priority over > under > agree > dry when coarsening.
_CAT_DRY = 0
_CAT_AGREE = 1
_CAT_UNDER = 2
_CAT_OVER = 3

# Regions larger than this (px, either side) are coarsened to plots.resolution_m;
# sub-regions stay native because MAX-resampling overstates disagreement.
_AGREEMENT_RASTER_MAX_DIM_PX = 2500


def _read_flood_totals(flood_totals_dir: Path, country_iso: str, rp: str, slr: str) -> tuple[float, float]:
    """Country-wide flooded km2 and exposed population from flood_totals_{ISO}.csv.

    Uses analysis/compute_flood_totals.py output at exposure.exceedance_threshold_m.
    Returns (NaN, NaN) if the CSV or the (rp, slr) row is missing.
    """
    csv_path = flood_totals_dir / f"flood_totals_{country_iso}.csv"
    if not csv_path.exists():
        return float("nan"), float("nan")
    df = pd.read_csv(csv_path)
    rp_num = int(str(rp).upper().lstrip("RP"))
    row = df[(df["return_period"] == rp_num) & (df["waterlevel_name"] == slr)]
    if row.empty:
        return float("nan"), float("nan")
    r = row.iloc[0]
    return float(r["total_wet_km2"]), float(r["total_exposed_pop"])


def _round_row(row: dict) -> dict:
    # Areas: integers (precision beyond the ~30m grid is spurious), but 2 decimals
    # below 1 so small nonzero values do not read as 0. Ratios: 3 decimals.
    int_cols = (
        "tp_km2", "fp_km2", "fn_km2", "tn_km2",
        "benchmark_wet_km2", "model_wet_km2", "benchmark_wet_outside_model_domain_km2",
        "model_total_wet_km2",
        "tp_tol_km2", "fp_tol_km2", "fn_tol_km2", "tn_tol_km2",
        "fp_forgiven_km2", "fn_forgiven_km2",
    )
    ratio_cols = (
        "benchmark_wet_outside_model_domain_pct",
        "HR", "FAR", "CSI", "EB", "EB_ratio", "bias",
        "HR_tol", "FAR_tol", "CSI_tol", "pct_disagreement_forgiven",
    )
    out = dict(row)
    for col in int_cols:
        val = out.get(col)
        if val is not None and val == val:  # skip NaN
            out[col] = round(val, 2) if abs(val) < 1 else int(round(val))
    for col in ratio_cols:
        if col in out and out[col] == out[col]:
            out[col] = round(out[col], 3)
    return out


def _new_acc() -> dict[str, float]:
    return {
        "tp_km2": 0.0, "fp_km2": 0.0, "fn_km2": 0.0, "tn_km2": 0.0,
        "tp_cells": 0, "fp_cells": 0, "fn_cells": 0, "tn_cells": 0,
        "tp_tol_km2": 0.0, "fp_tol_km2": 0.0, "fn_tol_km2": 0.0, "tn_tol_km2": 0.0,
        "fp_forgiven_km2": 0.0, "fn_forgiven_km2": 0.0,
    }


def _find_benchmark_keys(catalog, country_iso: str) -> list[str]:
    keys = []
    for key in catalog.sources.keys():
        meta = catalog.get_source(key).meta or {}
        if meta.get("country_iso") == country_iso and meta.get("hazard_type") == "coastal":
            keys.append(key)
    return sorted(keys)


_BLENDED_REGION = "ALL"


def _blend_regions(rows: list[dict], regions_spec: dict[str, list[float]] | None) -> dict | None:
    # Rows share one benchmark/RP/SLR. If one region bbox contains all others (nested
    # whole territory + sub-areas), its row is the total; else the regions are
    # disjoint and counts are summed. None for <= 1 row.
    if len(rows) <= 1:
        return None

    if regions_spec and len(regions_spec) > 1:
        def _contains(outer: list[float], inner: list[float]) -> bool:
            return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

        whole_name = next(
            (name for name, bbox in regions_spec.items()
             if all(_contains(bbox, other) for other_name, other in regions_spec.items() if other_name != name)),
            None,
        )
        if whole_name is not None:
            whole_row = next((r for r in rows if r.get("region") == whole_name), None)
            if whole_row is not None:
                blended = dict(whole_row)
                blended["region"] = _BLENDED_REGION
                return blended

    tp = sum(r.get("tp_km2", 0.0) for r in rows)
    fp = sum(r.get("fp_km2", 0.0) for r in rows)
    fn = sum(r.get("fn_km2", 0.0) for r in rows)
    tn = sum(r.get("tn_km2", 0.0) for r in rows)
    tp_c = sum(r.get("tp_cells", 0.0) for r in rows)
    fp_c = sum(r.get("fp_cells", 0.0) for r in rows)
    fn_c = sum(r.get("fn_cells", 0.0) for r in rows)
    tn_c = sum(r.get("tn_cells", 0.0) for r in rows)
    metrics = v.metrics_from_counts(tp, fp, fn, tn)

    tp_tol = sum(r.get("tp_tol_km2", 0.0) for r in rows)
    fp_tol = sum(r.get("fp_tol_km2", 0.0) for r in rows)
    fn_tol = sum(r.get("fn_tol_km2", 0.0) for r in rows)
    tn_tol = sum(r.get("tn_tol_km2", 0.0) for r in rows)
    fp_forgiven = sum(r.get("fp_forgiven_km2", 0.0) for r in rows)
    fn_forgiven = sum(r.get("fn_forgiven_km2", 0.0) for r in rows)
    tol_metrics = v.metrics_from_counts(tp_tol, fp_tol, fn_tol, tn_tol)
    hard_disagreement = fp_tol + fn_tol + fp_forgiven + fn_forgiven
    pct_forgiven = 100.0 * (fp_forgiven + fn_forgiven) / hard_disagreement if hard_disagreement > 0 else float("nan")

    first = rows[0]
    return _round_row({
        "country": first["country"], "iso": first["iso"], "benchmark_key": first["benchmark_key"],
        "region": _BLENDED_REGION,
        "return_period": first["return_period"], "waterlevel_name": first["waterlevel_name"],
        "tp_km2": tp, "fp_km2": fp, "fn_km2": fn, "tn_km2": tn,
        "tp_cells": tp_c, "fp_cells": fp_c, "fn_cells": fn_c, "tn_cells": tn_c,
        "benchmark_wet_km2": sum(r.get("benchmark_wet_km2", 0.0) for r in rows),
        "model_wet_km2": sum(r.get("model_wet_km2", 0.0) for r in rows),
        "benchmark_wet_outside_model_domain_km2": sum(
            r.get("benchmark_wet_outside_model_domain_km2", 0.0) for r in rows
        ),
        "benchmark_wet_outside_model_domain_pct": float("nan"),
        "model_total_wet_km2": first.get("model_total_wet_km2", float("nan")),  # already country-wide
        "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"],
        "EB": metrics["EB"], "EB_ratio": metrics["EB_ratio"], "bias": metrics["bias"],
        "tp_tol_km2": tp_tol, "fp_tol_km2": fp_tol, "fn_tol_km2": fn_tol, "tn_tol_km2": tn_tol,
        "fp_forgiven_km2": fp_forgiven, "fn_forgiven_km2": fn_forgiven,
        "HR_tol": tol_metrics["HR"], "FAR_tol": tol_metrics["FAR"], "CSI_tol": tol_metrics["CSI"],
        "pct_disagreement_forgiven": pct_forgiven,
    })


def validate_country_national_coverage(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
    read_tiles_directly: bool = False,
) -> pd.DataFrame:
    """Score every national-coverage coastal benchmark for one country, chunk by chunk.

    The benchmark assessed the whole coastline, so model-wet cells outside its polygons
    are false positives. The domain is the model domain minus permanent water, masked to
    the country (or `spec.geogunit_ids`). Scoring per chunk bounds memory on long
    coastlines. `read_tiles_directly` reads per-tile results instead of merged chunks.
    Side effect: writes one agreement raster per region.
    """
    val_cfg = cfg["validation"]
    rp = val_cfg["return_period"]
    slr = val_cfg["waterlevel_name"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    tolerance_cells = int(val_cfg["coastline_tolerance_cells"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])
    flood_totals_dir = Path(val_cfg["flood_totals_dir"])
    model_total_wet_km2, _ = _read_flood_totals(flood_totals_dir, country_iso, rp, slr)
    model_outputs = cfg["simulation"]["model_outputs"]
    tile_grid = load_tile_grid(cfg["tile_grid"]["path"]) if read_tiles_directly else None

    def _is_national_extent(key: str) -> bool:
        s = v.load_benchmark_spec(bench_catalog, key)
        return s.variable == "extent" and s.coverage != "partial"

    benchmark_keys = [key for key in _find_benchmark_keys(bench_catalog, country_iso) if _is_national_extent(key)]
    if not benchmark_keys:
        return pd.DataFrame()

    all_rows: list[dict] = []

    for benchmark_key in benchmark_keys:
        spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
        if spec.data_type != "GeoDataFrame":
            raise ValueError(
                f"{benchmark_key}: data_type={spec.data_type!r} not supported - only vector "
                "(GeoDataFrame) benchmarks are scored by validate_country_national_coverage."
            )
        if not spec.regions:
            raise ValueError(
                f"{benchmark_key}: coverage=national benchmarks require an explicit "
                "meta.regions block (no evaluation-cluster geometry exists here to derive "
                "a working extent from) - see BenchmarkSpec.regions' own docstring."
            )
        print(f"  National-coverage benchmark: {benchmark_key} ({spec.data_type})")

        full_gdf = v.load_benchmark_4326(bench_catalog, spec)
        if full_gdf.empty:
            print(f"    {benchmark_key}: empty after filtering - skipping.")
            continue
        full_sindex = full_gdf.sindex

        acc: dict[str, dict[str, float]] = defaultdict(_new_acc)
        benchmark_wet_km2_total: dict[str, float] = defaultdict(float)
        benchmark_wet_outside_model_domain_km2: dict[str, float] = defaultdict(float)
        model_wet_km2: dict[str, float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        n_processed = 0
        n_chunks_total = 0

        for region, region_bbox in spec.regions.items():
            pieces: list[tuple[np.ndarray, Affine]] = []
            if read_tiles_directly:
                tile_paths = _tile_paths_overlapping_bbox(region_bbox, tile_grid, model_outputs, rp, slr)
                n_chunks_total += len(tile_paths)
                depth, out_transform = _mosaic_read(tile_paths, region_bbox)
                if depth is not None:
                    pieces.append((depth, out_transform))
            else:
                chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
                n_chunks_total += len(chunk_paths)
                for chunk_path in chunk_paths:
                    with retry_transient_io(rasterio.open, chunk_path) as src:
                        # A chunk can hold several regions; window to region_bbox so each
                        # region scores its own data.
                        window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
                        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                        window = window.round_offsets().round_lengths()
                        if window.width <= 0 or window.height <= 0:
                            continue
                        depth = src.read(1, window=window)
                        out_transform = src.window_transform(window)
                    pieces.append((depth, out_transform))

            for depth, out_transform in pieces:
                model_domain = v.model_domain_mask(depth, nodata=-9999.0)
                if not model_domain.any():
                    continue
                n_processed += 1

                bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], out_transform))
                candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
                chunk_bench_gdf = full_gdf.iloc[candidate_idx]
                fraction = v.benchmark_fraction_from_vector(
                    chunk_bench_gdf, out_transform, "EPSG:4326", depth.shape, supersample=supersample,
                )
                # Scoring uses the soft `fraction`; the thresholded mask is display only.
                benchmark_wet_display = v.wet_mask_from_fraction(fraction, wet_fraction)

                not_water, domain_mask = v.window_scoring_domain(
                    gfm_catalog, permanent_water_source, permanent_water_codes, geogunit_source,
                    iso_lookup, country_iso, bbox, out_transform, model_domain,
                    geogunit_ids=spec.geogunit_ids,
                )

                area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
                benchmark_wet_km2_total[region] += float((fraction * area_km2 * not_water).sum())
                benchmark_wet_outside_model_domain_km2[region] += float(
                    (fraction * area_km2 * not_water * (~model_domain)).sum()
                )

                model_wet = model_domain & (depth > primary_threshold_m) & not_water
                model_wet_km2[region] += float((model_wet * area_km2).sum())

                tp, fp, fn, tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, area_km2)
                a = acc[region]
                a["tp_km2"] += tp
                a["fp_km2"] += fp
                a["fn_km2"] += fn
                a["tn_km2"] += tn
                # Cell counts are fraction-weighted (float).
                tp_c, fp_c, fn_c, tn_c = v.confusion_counts_soft(
                    model_wet, fraction, domain_mask, np.ones_like(area_km2),
                )
                a["tp_cells"] += tp_c
                a["fp_cells"] += fp_c
                a["fn_cells"] += fn_c
                a["tn_cells"] += tn_c

                tol = v.confusion_counts_tolerant(model_wet, fraction, domain_mask, area_km2, tolerance_cells)
                a["tp_tol_km2"] += tol["tp"]
                a["fp_tol_km2"] += tol["fp"]
                a["fn_tol_km2"] += tol["fn"]
                a["tn_tol_km2"] += tol["tn"]
                a["fp_forgiven_km2"] += tol["fp_forgiven"]
                a["fn_forgiven_km2"] += tol["fn_forgiven"]

                chunk_agreement = np.full(depth.shape, _CAT_DRY, dtype="uint8")
                chunk_agreement[domain_mask & model_wet & ~benchmark_wet_display] = _CAT_OVER
                chunk_agreement[domain_mask & ~model_wet & benchmark_wet_display] = _CAT_UNDER
                still_dry = chunk_agreement == _CAT_DRY
                chunk_agreement[still_dry & domain_mask & model_wet & benchmark_wet_display] = _CAT_AGREE
                agreement_pieces_by_region[region].append((chunk_agreement, out_transform))

        print(f"    {n_processed}/{n_chunks_total} chunk(s) had real model data.")

        region_rows: list[dict] = []
        for region in sorted(acc.keys()):
            a = acc[region]
            metrics = v.metrics_from_counts(a["tp_km2"], a["fp_km2"], a["fn_km2"], a["tn_km2"])
            bm_total = benchmark_wet_km2_total[region]
            bm_outside = benchmark_wet_outside_model_domain_km2[region]
            bm_wet_pct = 100.0 * bm_outside / bm_total if bm_total > 0 else float("nan")
            tol_metrics = v.metrics_from_counts(a["tp_tol_km2"], a["fp_tol_km2"], a["fn_tol_km2"], a["tn_tol_km2"])
            hard_disagreement = a["fp_tol_km2"] + a["fn_tol_km2"] + a["fp_forgiven_km2"] + a["fn_forgiven_km2"]
            pct_forgiven = (
                100.0 * (a["fp_forgiven_km2"] + a["fn_forgiven_km2"]) / hard_disagreement
                if hard_disagreement > 0 else float("nan")
            )
            region_rows.append(_round_row({
                "country": country_iso, "iso": country_iso, "benchmark_key": benchmark_key,
                "region": region,
                "return_period": rp, "waterlevel_name": slr,
                "tp_km2": a["tp_km2"], "fp_km2": a["fp_km2"], "fn_km2": a["fn_km2"], "tn_km2": a["tn_km2"],
                "tp_cells": a["tp_cells"], "fp_cells": a["fp_cells"],
                "fn_cells": a["fn_cells"], "tn_cells": a["tn_cells"],
                "benchmark_wet_km2": bm_total,
                "model_wet_km2": model_wet_km2[region],
                "benchmark_wet_outside_model_domain_km2": bm_outside,
                "benchmark_wet_outside_model_domain_pct": bm_wet_pct,
                "model_total_wet_km2": model_total_wet_km2,
                "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"],
                "EB": metrics["EB"], "EB_ratio": metrics["EB_ratio"], "bias": metrics["bias"],
                "tp_tol_km2": a["tp_tol_km2"], "fp_tol_km2": a["fp_tol_km2"],
                "fn_tol_km2": a["fn_tol_km2"], "tn_tol_km2": a["tn_tol_km2"],
                "fp_forgiven_km2": a["fp_forgiven_km2"], "fn_forgiven_km2": a["fn_forgiven_km2"],
                "HR_tol": tol_metrics["HR"], "FAR_tol": tol_metrics["FAR"], "CSI_tol": tol_metrics["CSI"],
                "pct_disagreement_forgiven": pct_forgiven,
            }))

        blended = _blend_regions(region_rows, spec.regions)
        if blended is not None:
            region_rows.append(blended)
        all_rows.extend(region_rows)

        for region_name, pieces in agreement_pieces_by_region.items():
            _write_agreement_raster(pieces, cfg, country_iso, rp, slr, region_name, metric="agreement")

    return pd.DataFrame(all_rows)


def _write_agreement_raster(
    agreement_pieces: list[tuple[np.ndarray, Affine]],
    cfg: dict,
    country_iso: str,
    rp: str,
    slr: str,
    region_name: str,
    metric: str = "agreement",
) -> None:
    # Display only: metrics come from the native grids. Native resolution unless the
    # region exceeds _AGREEMENT_RASTER_MAX_DIM_PX, then plots.resolution_m with MAX.
    if not agreement_pieces:
        return
    val_cfg = cfg["validation"]

    all_bounds = [rasterio.transform.array_bounds(a.shape[0], a.shape[1], t) for a, t in agreement_pieces]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    native_res_deg = abs(agreement_pieces[0][1].a)  # all pieces share the model grid
    native_w = (maxx - minx) / native_res_deg
    native_h = (maxy - miny) / native_res_deg
    if max(native_w, native_h) <= _AGREEMENT_RASTER_MAX_DIM_PX:
        res_deg = native_res_deg
    else:
        res_deg = float(val_cfg["plots"]["resolution_m"]) / (111.32 * 1000.0)

    out_w = max(1, int(math.ceil((maxx - minx) / res_deg)))
    out_h = max(1, int(math.ceil((maxy - miny) / res_deg)))
    out_transform = Affine(res_deg, 0, minx, 0, -res_deg, maxy)

    mosaic = np.zeros((out_h, out_w), dtype="uint8")
    for arr, transform in agreement_pieces:
        piece = np.zeros((out_h, out_w), dtype="uint8")
        reproject(
            source=arr, destination=piece,
            src_transform=transform, src_crs="EPSG:4326",
            dst_transform=out_transform, dst_crs="EPSG:4326",
            resampling=Resampling.max,
        )
        mosaic = np.maximum(mosaic, piece)

    out_dir = Path(val_cfg["output_dir"]) / country_iso
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    out_path = out_dir / f"{metric}_{country_iso}_{region_name}_{rp}_{slr}.tif"
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": 255,
        "crs": "EPSG:4326", "transform": out_transform,
        "width": out_w, "height": out_h, "compress": "lzw",
    }
    with retry_transient_io(rasterio.open, out_path, "w", **profile) as dst:
        dst.write(mosaic, 1)
    print(f"    {metric.replace('_', ' ').title()} raster ({region_name}): {out_path}")


def main() -> None:
    """Score one country, write its metrics CSV and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--country", required=True, help="ISO-3 country code, e.g. GBR")
    parser.add_argument(
        "--run-tag", default=None,
        help="tags every output row with this value in a run_tag column - for "
             "aggregating results across multiple runs (e.g. a {group}__{sweep_point} "
             "identity for some future calibration sweep). "
             "Optional - omit for a normal single/production validation run.",
    )
    parser.add_argument(
        "--read-tiles-directly", action="store_true",
        help="score national-coverage benchmarks (validate_country_national_coverage) "
             "straight from production's own per-tile results/waterdepth_{rp}_{slr}.tif "
             "files instead of merge_chunk's merged chunks - for an ad hoc tile subset "
             "(e.g. a country-validation run over a handful of tiles) that was never run "
             "through merge_chunk, see that function's own docstring.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    bench_catalog = get_data_catalog(
        _REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"],
    )
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    country_iso = args.country.upper()
    out_dir = Path(val_cfg["output_dir"]) / country_iso
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    rp, slr = val_cfg["return_period"], val_cfg["waterlevel_name"]

    print(f"=== Validating {country_iso} (extent, national coverage) ===")
    df = validate_country_national_coverage(
        country_iso, cfg, gfm_catalog, bench_catalog, read_tiles_directly=args.read_tiles_directly,
    )
    if df.empty:
        print("No extent rows produced.")
    else:
        if args.run_tag:
            df["run_tag"] = args.run_tag
        out_path = out_dir / f"metrics_{country_iso}_{rp}_{slr}.csv"
        atomic_write(out_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
        print(f"\nMetrics written: {out_path} ({len(df)} row(s))")

        has_blended = df["region"].eq(_BLENDED_REGION)
        printed = df[has_blended] if has_blended.any() else df
        for _, row in printed.iterrows():
            print(
                f"\n[{row['region']}] Primary threshold ({val_cfg['primary_threshold_m']}m):\n"
                f"  HR={row['HR']:.3f} FAR={row['FAR']:.3f} CSI={row['CSI']:.3f} EB={row['EB']:.3f}\n"
                f"  CSI_tol={row['CSI_tol']:.3f} (tolerance={val_cfg['coastline_tolerance_cells']} cell(s), "
                f"{row['pct_disagreement_forgiven']:.1f}% of strict disagreement forgiven - diagnostic only, "
                "not the headline number)\n"
                f"  benchmark_wet_outside_model_domain_pct={row['benchmark_wet_outside_model_domain_pct']:.1f}%"
                " - health check: if this is high, don't quote the rest of this row as a model result."
            )


if __name__ == "__main__":
    main()
