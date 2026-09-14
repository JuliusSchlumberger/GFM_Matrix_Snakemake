"""Per-country coastal flood-extent validation driver.

For one country, finds every applicable coastal benchmark in the validation
catalog, groups its geometries into evaluation CLUSTERS (buffer each
polygon, merge any that overlap - see src/validation.py::build_evaluation_clusters),
and for each cluster: mosaics the merged waterdepth chunk(s) covering its
bounding box, rasterizes the cluster's own buffered geometry as the
evaluation domain directly (no block-local distance transform needed - the
buffer is exact), excludes permanent water bodies via the Copernicus
land-use layer, and accumulates area-/population-weighted TP/FP/FN/TN
across every (threshold_m x domain) combination. This cluster-based path
(validate_country()) is only used for `coverage == "partial"` benchmarks
(the default - Spain/France's isolated designated survey zones). Benchmarks
whose source assessed the ENTIRE coastline (`coverage == "national"`, e.g.
Norway's Kartverket data) instead use validate_country_national_coverage()'s
per-postprocessing-chunk path, with no buffer/cluster step at all - see that
function's own docstring for why partial-coverage's buffering logic neither
applies nor is safe for a national benchmark's long, lightly-fragmented
coastline (confirmed 2026-09: it collapses into one cluster spanning almost
the whole country and crashes on the resulting bbox-sized array allocation).

2026-09 redesign: replaced an earlier uniform-0.5deg-grid-of-blocks design.
That approach did real, wasted work (window reads, per-block distance
transforms, ~64 reproject() calls per block for population disaggregation)
on the ~60% of grid cells that had no benchmark data at all (Spain's Q100
map only covers surveyed coastal stretches, plan doc §3), and split single
contiguous benchmark polygons across multiple blocks needing independent
halo-padding. Evaluation units now come directly from the benchmark's own
geometry instead of an arbitrary grid overlaid on top of it.

If the benchmark's catalog entry defines `meta.regions: {name: [minx,miny,maxx,maxy]}`,
each evaluation cluster is tagged by which named region its centroid falls in (e.g.
Spain's `mainland` vs `canary_islands`) and metrics are reported/plotted per region
instead of blended into one national row; a country with no `regions` configured yet
gets a single implicit region (its own ISO code). Every region-row also carries the
same country-WIDE "model total" flooded area/exposed population (_read_flood_totals,
read from analysis/compute_flood_totals.py's precomputed per-country CSV, not
region-specific) as a denominator independent of any cluster's own local working
window - see caveats doc §1.2.

Writes:
  {validation.output_dir}/{country}/metrics_{country}_{RP}_{SLR}.csv
  {validation.output_dir}/{country}/agreement_{country}_{region}_{RP}_{SLR}.tif (one per region)
    (plots.resolution_m, priority-reduced: over > under > agree > dry -
    see _write_agreement_raster)

See docs/flood_extent_validation_plan.md for the full design; src/validation.py
for the stateless computational building blocks this file just orchestrates.

Usage:
    python validation/validate_country.py \\
        --config snakemake_workflow/config/config.yml --country ESP
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
from rasterio.features import rasterize
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from plotting import pixel_area_km2_grid  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Agreement-category codes, ordered so Resampling.max correctly implements
# the plan's "priority: over > under > agree > dry" display-downsampling
# rule (plan doc §5.4) - the highest code present in a display cell wins.
_CAT_DRY = 0
_CAT_AGREE = 1
_CAT_UNDER = 2
_CAT_OVER = 3

# Label for a cluster whose centroid falls in none of a country's `regions:`
# bboxes (config error / gap in the partition, not expected in normal operation -
# see validation.region_for_point).
_UNCLASSIFIED_REGION = "unclassified"


def _postprocess_chunk_id(x: float, y: float, chunk_size_deg: float) -> str:
    """Which postprocessing (chunk_size_deg, typically 5deg) chunk a point falls in."""
    xi = int(math.floor(x / chunk_size_deg) * chunk_size_deg)
    yi = int(math.floor(y / chunk_size_deg) * chunk_size_deg)
    lat = f"N{yi:02d}" if yi >= 0 else f"S{-yi:02d}"
    lon = f"E{xi:03d}" if xi >= 0 else f"W{-xi:03d}"
    return f"{lat}{lon}"


def _chunks_overlapping_bbox(
    bbox: list[float], pp_chunk_deg: float, merged_chunks_dir: Path, rp: str, slr: str,
) -> list[Path]:
    """Every existing merged waterdepth chunk file whose 5deg cell overlaps `bbox`."""
    minx, miny, maxx, maxy = bbox
    eps = 1e-9
    x0 = math.floor(minx / pp_chunk_deg) * pp_chunk_deg
    x1 = math.floor((maxx - eps) / pp_chunk_deg) * pp_chunk_deg
    y0 = math.floor(miny / pp_chunk_deg) * pp_chunk_deg
    y1 = math.floor((maxy - eps) / pp_chunk_deg) * pp_chunk_deg
    paths = []
    x = x0
    while x <= x1 + eps:
        y = y0
        while y <= y1 + eps:
            chunk_id = _postprocess_chunk_id(x, y, pp_chunk_deg)
            p = merged_chunks_dir / f"waterdepth_{chunk_id}_{rp}_{slr}.tif"
            if p.exists():
                paths.append(p)
            y += pp_chunk_deg
        x += pp_chunk_deg
    return paths


def _mosaic_read(chunk_paths: list[Path], bbox: list[float]) -> tuple[np.ndarray | None, Affine | None]:
    """Mosaic (possibly several) merged waterdepth chunk files onto one array covering `bbox`.

    A cluster's buffered bounding box is not bounded to a single 5deg
    postprocessing chunk the way a 0.5deg block was guaranteed to be, so
    this can genuinely span more than one source file - rasterio.merge
    handles that directly instead of hand-rolled multi-file windowing.
    """
    if not chunk_paths:
        return None, None
    srcs = [retry_transient_io(rasterio.open, p) for p in chunk_paths]
    try:
        mosaic, out_transform = rasterio_merge(srcs, bounds=bbox, nodata=-9999.0)
    finally:
        for s in srcs:
            s.close()
    return mosaic[0], out_transform


def _read_flood_totals(flood_totals_dir: Path, country_iso: str, rp: str, slr: str) -> tuple[float, float]:
    """Country-wide total flooded area (km2) + exposed population for (rp, slr),
    at the single fixed exposure threshold (exposure.exceedance_threshold_m) -
    read from analysis/compute_flood_totals.py's precomputed
    flood_totals_{country}.csv (analysis.compute_flood_totals switch), rather than
    re-derived from raw depth here.

    Addresses caveats doc §1.2: a benchmark cluster only ever sees its own small
    local window, so it understates how much the model floods/exposes across the
    WHOLE country. compute_flood_totals.py already computes this correctly and
    cheaply (same WRI geogunit + FLOPROS ISO lookup used for exposure aggregation
    - exact country membership, not a `regions:` bbox needing its own mask) as
    part of the general analysis pipeline, streamed once for every country at
    once - re-deriving it per validation run from raw depth chunks would just
    duplicate that work far more slowly and with a bbox-approximated country
    boundary. Country-wide, not per-region: every region-row of this country's
    metrics CSV gets the SAME two values.

    Returns (NaN, NaN), never a silent 0.0, if the CSV or the matching (rp, slr)
    row doesn't exist yet - most likely because analysis/run_analysis.py hasn't
    been (re-)run since this feature was added.
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
    """Round the CSV's numeric columns to a sane display precision.

    Area (km²), population, and cell counts round to whole numbers - the
    model's own resolution (~30-50m, plan doc §2.1/§7.1) and the underlying
    population/benchmark data don't support the 15 significant digits raw
    float64 arithmetic produces (e.g. `1657.027066374921` km²), and
    reporting that many is actively misleading about the real precision of
    these estimates. EXCEPT: a value with `abs() < 1` rounds to 2 decimals
    instead of a bare integer, so a real (if small) nonzero quantity - e.g.
    `benchmark_wet_outside_model_domain_km2 = 0.34` - doesn't silently
    round away to `0` and read as "none" when it's actually "a little."

    Ratio metrics (HR/FAR/CSI/EB/EB_ratio/bias, the pop_* ratios, and the
    outside-domain %) are proportions, not counts - rounding THOSE to whole
    numbers would destroy virtually all their information (0.93 vs 0.0 read
    identically as "1" vs "0"), so they always keep 3 decimal places
    instead (matching this script's own console summary), regardless of
    magnitude.
    """
    int_cols = (
        "tp_km2", "fp_km2", "fn_km2", "tn_km2",
        "benchmark_wet_km2", "model_wet_km2", "benchmark_wet_outside_model_domain_km2",
        "model_total_wet_km2",
        "pop_tp", "pop_fp", "pop_fn", "pop_benchmark", "pop_model",
        "pop_over", "pop_under", "pop_diff_net", "pop_model_total",
        # depth-band comparison (validate_country_depth_bands)
        "agree_km2", "under_km2", "over_km2", "benchmark_band_coverage_km2",
        "pop_agree", "pop_under", "pop_over",
    )
    ratio_cols = (
        "benchmark_wet_outside_model_domain_pct",
        "HR", "FAR", "CSI", "EB", "EB_ratio", "bias",
        "pop_HR", "pop_FAR", "pop_CSI",
        # depth-band comparison (validate_country_depth_bands)
        "pct_agree", "pct_under", "pct_over", "depth_EB",
        "pop_pct_agree", "pop_pct_under", "pop_pct_over", "pop_depth_EB",
    )
    out = dict(row)
    for col in int_cols:
        val = out.get(col)
        if val is not None and val == val:  # NaN != NaN - leave NaN as-is
            out[col] = round(val, 2) if abs(val) < 1 else int(round(val))
    for col in ratio_cols:
        if col in out and out[col] == out[col]:
            out[col] = round(out[col], 3)
    return out


def _new_acc() -> dict[str, float]:
    return {
        "tp_km2": 0.0, "fp_km2": 0.0, "fn_km2": 0.0, "tn_km2": 0.0,
        "tp_cells": 0, "fp_cells": 0, "fn_cells": 0, "tn_cells": 0,
        "pop_tp": 0.0, "pop_fp": 0.0, "pop_fn": 0.0, "pop_tn": 0.0,
    }


def _find_benchmark_keys(catalog, country_iso: str) -> list[str]:
    """Every catalog entry whose meta.country_iso matches and hazard_type == 'coastal'.

    Scans the whole catalog rather than a hardcoded per-country key list -
    adding a new country/dataset is a catalog entry, not a code change
    (plan doc §5.2). `catalog.sources` is a plain dict (confirmed against
    this repo's installed hydromt version, 2026-09).
    """
    keys = []
    for key in catalog.sources.keys():
        meta = catalog.get_source(key).meta or {}
        if meta.get("country_iso") == country_iso and meta.get("hazard_type") == "coastal":
            keys.append(key)
    return sorted(keys)


def validate_country(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
) -> pd.DataFrame:
    val_cfg = cfg["validation"]
    rp = val_cfg["return_period"]
    slr = val_cfg["waterlevel_name"]
    thresholds_m = [float(t) for t in val_cfg["depth_thresholds_m"]]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    buffer_km = float(val_cfg["eval_domain"]["buffer_km"])
    also_model_only = bool(val_cfg["eval_domain"]["also_report_model_domain_only"])
    domains = ["buffered", "model_only"] if also_model_only else ["buffered"]
    simplify_tol_m = float(val_cfg["vector_simplify_tolerance_m"])
    quad_segs = int(val_cfg["buffer_quad_segs"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    population_source = val_cfg["population_source"]
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    flood_totals_dir = Path(val_cfg["flood_totals_dir"])
    model_total_wet_km2, pop_model_total = _read_flood_totals(flood_totals_dir, country_iso, rp, slr)
    if model_total_wet_km2 != model_total_wet_km2:  # NaN
        print(
            f"  NOTE: no flood_totals_{country_iso}.csv (or no matching {rp}/{slr} row) in "
            f"{flood_totals_dir} - model_total_wet_km2/pop_model_total will be NaN. Run "
            "analysis/run_analysis.py (analysis.compute_flood_totals) to populate it."
        )

    benchmark_keys = _find_benchmark_keys(bench_catalog, country_iso)
    if not benchmark_keys:
        print(f"  No coastal benchmark found for {country_iso} - nothing to validate.")
        return pd.DataFrame()

    all_rows: list[dict] = []

    for benchmark_key in benchmark_keys:
        spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
        if spec.variable != "extent":
            continue  # depth-band benchmarks are handled by validate_country_depth_bands
        if spec.coverage != "partial":
            continue  # national-coverage benchmarks are handled by validate_country_national_coverage
        if spec.data_type != "GeoDataFrame":
            raise NotImplementedError(
                f"Raster benchmark dispatch not yet implemented for {benchmark_key} "
                f"(data_type={spec.data_type}) - only GeoDataFrame benchmarks (Spain Q100) "
                "are wired up so far."
            )
        print(f"  Benchmark: {benchmark_key} ({spec.data_type})")

        full_gdf = v.load_benchmark_full(bench_catalog, spec)
        if full_gdf.empty:
            print(f"    {benchmark_key}: empty after filtering - skipping.")
            continue
        if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
            full_gdf = full_gdf.to_crs(4326)

        clusters = v.build_evaluation_clusters(full_gdf, buffer_km, simplify_tol_m, quad_segs)
        print(f"    {len(clusters)} cluster(s) from {len(full_gdf)} benchmark polygon(s).")

        # Spatial index once, reused per cluster to find that cluster's own
        # (unbuffered) benchmark polygons - no repeated catalog I/O per cluster.
        full_sindex = full_gdf.sindex

        # Keyed by (threshold_m, domain_name, region) - region tagging (below) is
        # per-cluster and dynamic (whatever spec.regions defines, or just
        # country_iso if a country has no regions configured yet), so the set of
        # keys actually populated isn't known up front the way (threshold, domain)
        # alone was pre-redesign.
        acc: dict[tuple[float, str, str], dict[str, float]] = defaultdict(_new_acc)
        benchmark_wet_km2_total: dict[str, float] = defaultdict(float)
        benchmark_wet_outside_model_domain_km2: dict[str, float] = defaultdict(float)
        model_wet_km2: dict[tuple[float, str], float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        n_processed = 0
        n_unclassified = 0

        for _, cluster in clusters.iterrows():
            cluster_geom = cluster.geometry
            bbox = list(cluster_geom.bounds)

            if spec.regions:
                centroid = cluster_geom.centroid
                region = v.region_for_point(centroid.x, centroid.y, spec.regions) or _UNCLASSIFIED_REGION
                if region == _UNCLASSIFIED_REGION:
                    n_unclassified += 1
            else:
                region = country_iso

            chunk_paths = _chunks_overlapping_bbox(bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
            if not chunk_paths:
                continue
            depth, out_transform = _mosaic_read(chunk_paths, bbox)
            if depth is None or depth.size == 0:
                continue

            model_domain = v.model_domain_mask(depth, nodata=-9999.0)
            if not model_domain.any():
                continue
            n_processed += 1

            cluster_domain_mask = rasterize(
                [(cluster_geom, 1)], out_shape=depth.shape, transform=out_transform,
                fill=0, dtype="uint8", all_touched=False,
            ).astype(bool)

            candidate_idx = list(full_sindex.query(cluster_geom, predicate="intersects"))
            cluster_bench_gdf = full_gdf.iloc[candidate_idx]
            fraction = v.benchmark_fraction_from_vector(
                cluster_bench_gdf, out_transform, "EPSG:4326", depth.shape, supersample=supersample,
            )
            benchmark_wet = v.wet_mask_from_fraction(fraction, wet_fraction)

            water_mask = v.read_permanent_water_mask(
                gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth.shape,
            )
            not_water = ~water_mask

            buffered_domain = model_domain & cluster_domain_mask & not_water
            model_only_domain = model_domain & not_water
            domain_masks = {"buffered": buffered_domain, "model_only": model_only_domain}

            area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])

            benchmark_wet_km2_total[region] += float((fraction * area_km2 * not_water).sum())
            benchmark_wet_outside_model_domain_km2[region] += float(
                (fraction * area_km2 * not_water * (~model_domain)).sum()
            )

            try:
                pop_da = retry_transient_io(
                    gfm_catalog.get_rasterdataset, population_source, bbox=bbox,
                ).squeeze(drop=True)
            except Exception:
                pop_da = None
            if pop_da is not None and pop_da.size > 0:
                pop_arr = np.nan_to_num(pop_da.values.astype("float64"), nan=0.0)
                pop_nodata = pop_da.raster.nodata
                if pop_nodata is not None:
                    pop_arr[pop_arr == pop_nodata] = 0.0
                pop_transform, pop_crs = pop_da.raster.transform, pop_da.raster.crs
            else:
                pop_arr, pop_transform, pop_crs = None, None, None

            cluster_agreement = np.full(depth.shape, _CAT_DRY, dtype="uint8")

            for threshold_m in thresholds_m:
                model_wet = model_domain & (depth > threshold_m) & not_water
                model_wet_km2[(threshold_m, region)] += float((model_wet * area_km2).sum())

                for domain_name in domains:
                    d = domain_masks[domain_name]
                    tp, fp, fn, tn = v.confusion_counts(model_wet, benchmark_wet, d, area_km2)
                    a = acc[(threshold_m, domain_name, region)]
                    a["tp_km2"] += tp
                    a["fp_km2"] += fp
                    a["fn_km2"] += fn
                    a["tn_km2"] += tn
                    a["tp_cells"] += int((d & model_wet & benchmark_wet).sum())
                    a["fp_cells"] += int((d & model_wet & ~benchmark_wet).sum())
                    a["fn_cells"] += int((d & ~model_wet & benchmark_wet).sum())
                    a["tn_cells"] += int((d & ~model_wet & ~benchmark_wet).sum())

                    if pop_arr is not None:
                        for cls_name, cls_mask in (
                            ("pop_tp", model_wet & benchmark_wet),
                            ("pop_fp", model_wet & ~benchmark_wet),
                            ("pop_fn", ~model_wet & benchmark_wet),
                            ("pop_tn", ~model_wet & ~benchmark_wet),
                        ):
                            pop_class = v.population_by_class(
                                cls_mask, d, out_transform, "EPSG:4326",
                                pop_arr, pop_transform, pop_crs,
                            )
                            a[cls_name] += float(pop_class.sum())

                # Agreement category at the PRIMARY threshold only (one map,
                # not one per threshold - plan §5.4 shows a single map).
                if threshold_m == primary_threshold_m:
                    buffered = domain_masks["buffered"]
                    cluster_agreement[buffered & model_wet & ~benchmark_wet] = _CAT_OVER
                    cluster_agreement[buffered & ~model_wet & benchmark_wet] = _CAT_UNDER
                    still_dry = cluster_agreement == _CAT_DRY
                    cluster_agreement[still_dry & buffered & model_wet & benchmark_wet] = _CAT_AGREE

            agreement_pieces_by_region[region].append((cluster_agreement, out_transform))

        print(f"    {n_processed}/{len(clusters)} cluster(s) had real model data.")
        if n_unclassified:
            print(
                f"    WARNING: {n_unclassified} cluster(s) fell outside every bbox in this "
                f"benchmark's 'regions' - reported under region='{_UNCLASSIFIED_REGION}'."
            )

        for threshold_m, domain_name, region in sorted(acc.keys()):
            a = acc[(threshold_m, domain_name, region)]
            metrics = v.metrics_from_counts(a["tp_km2"], a["fp_km2"], a["fn_km2"], a["tn_km2"])
            pop_metrics = v.metrics_from_counts(a["pop_tp"], a["pop_fp"], a["pop_fn"], a["pop_tn"])
            bm_total = benchmark_wet_km2_total[region]
            bm_outside = benchmark_wet_outside_model_domain_km2[region]
            bm_wet_pct = 100.0 * bm_outside / bm_total if bm_total > 0 else float("nan")
            # model_total_wet_km2/pop_model_total are country-wide (not per-region -
            # see _read_flood_totals) and only meaningful at the threshold the
            # underlying flood_totals CSV was actually built at (the single fixed
            # exposure.exceedance_threshold_m) - NaN at every other threshold in
            # this sweep, rather than repeating a number that doesn't apply to it.
            is_primary = threshold_m == primary_threshold_m
            all_rows.append(_round_row({
                "country": country_iso, "iso": country_iso, "benchmark_key": benchmark_key,
                "region": region,
                "return_period": rp, "waterlevel_name": slr,
                "threshold_m": threshold_m, "domain": domain_name,
                "tp_km2": a["tp_km2"], "fp_km2": a["fp_km2"], "fn_km2": a["fn_km2"], "tn_km2": a["tn_km2"],
                "tp_cells": a["tp_cells"], "fp_cells": a["fp_cells"],
                "fn_cells": a["fn_cells"], "tn_cells": a["tn_cells"],
                "benchmark_wet_km2": bm_total,
                "model_wet_km2": model_wet_km2[(threshold_m, region)],
                "benchmark_wet_outside_model_domain_km2": bm_outside,
                "benchmark_wet_outside_model_domain_pct": bm_wet_pct,
                "model_total_wet_km2": model_total_wet_km2 if is_primary else float("nan"),
                "pop_model_total": pop_model_total if is_primary else float("nan"),
                "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"],
                "EB": metrics["EB"], "EB_ratio": metrics["EB_ratio"], "bias": metrics["bias"],
                "pop_tp": a["pop_tp"], "pop_fp": a["pop_fp"], "pop_fn": a["pop_fn"],
                "pop_benchmark": a["pop_tp"] + a["pop_fn"], "pop_model": a["pop_tp"] + a["pop_fp"],
                "pop_over": a["pop_fp"], "pop_under": a["pop_fn"],
                "pop_diff_net": a["pop_fp"] - a["pop_fn"],
                "pop_HR": pop_metrics["HR"], "pop_FAR": pop_metrics["FAR"], "pop_CSI": pop_metrics["CSI"],
            }))

        for region_name, pieces in agreement_pieces_by_region.items():
            _write_agreement_raster(pieces, cfg, country_iso, rp, slr, region_name, metric="agreement")

    return pd.DataFrame(all_rows)


def validate_country_depth_bands(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
) -> pd.DataFrame:
    """Depth-band comparison: model depth vs. benchmark [ht_min, ht_max] bands
    (`variable == "depth"` catalog entries, e.g. France's `n_iso_ht_*` layers),
    the continuous-depth companion to validate_country()'s binary extent
    comparison. Shares that function's cluster/region/land-use-masking
    machinery (same helpers: _chunks_overlapping_bbox, _mosaic_read,
    v.build_evaluation_clusters, v.region_for_point, v.read_permanent_water_mask,
    _write_agreement_raster) but classifies each cell as agree/under/over
    against a rasterized depth band instead of thresholding depth into a
    binary wet/dry mask - see src/validation.py::rasterize_depth_bands/
    depth_band_counts/depth_band_metrics_from_counts.

    No threshold sweep (unlike validate_country's 4-threshold loop) - the
    model's own continuous depth is compared directly against each cell's
    band, so there is nothing to sweep a threshold over.
    """
    val_cfg = cfg["validation"]
    rp = val_cfg["return_period"]
    slr = val_cfg["waterlevel_name"]
    supersample = int(val_cfg["benchmark_supersample"])  # unused here (no fractional
    # coverage needed for depth bands - see rasterize_depth_bands' own docstring), kept
    # only so this function's config surface visibly mirrors validate_country's.
    del supersample
    buffer_km = float(val_cfg["eval_domain"]["buffer_km"])
    also_model_only = bool(val_cfg["eval_domain"]["also_report_model_domain_only"])
    domains = ["buffered", "model_only"] if also_model_only else ["buffered"]
    simplify_tol_m = float(val_cfg["vector_simplify_tolerance_m"])
    quad_segs = int(val_cfg["buffer_quad_segs"])
    open_ended_min_m = float(val_cfg["depth_band_open_ended_min_m"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    population_source = val_cfg["population_source"]
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]

    benchmark_keys = [
        key for key in _find_benchmark_keys(bench_catalog, country_iso)
        if v.load_benchmark_spec(bench_catalog, key).variable == "depth"
    ]
    if not benchmark_keys:
        print(f"  No coastal depth-band benchmark found for {country_iso} - skipping depth-band comparison.")
        return pd.DataFrame()

    all_rows: list[dict] = []

    for benchmark_key in benchmark_keys:
        spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
        if spec.coverage != "partial":
            # No chunk-based depth-band path exists yet (mirroring
            # validate_country_national_coverage's extent-only one) - a national-coverage
            # depth benchmark would hit the exact same cluster-bbox-blowup risk documented
            # on validate_country_national_coverage, so refuse loudly rather than silently
            # attempting the cluster-based path and risking the same crash.
            raise NotImplementedError(
                f"{benchmark_key}: coverage={spec.coverage!r} depth-band benchmarks have no "
                "chunk-based comparison path yet (only extent benchmarks do - see "
                "validate_country_national_coverage) - the cluster-based path this function "
                "uses is known to crash on a national-coverage benchmark's long coastline."
            )
        if spec.data_type != "GeoDataFrame":
            raise NotImplementedError(
                f"Raster depth-band benchmark dispatch not implemented for {benchmark_key} "
                f"(data_type={spec.data_type}) - only GeoDataFrame benchmarks are wired up."
            )
        print(f"  Depth-band benchmark: {benchmark_key} ({spec.data_type})")

        full_gdf = v.load_benchmark_full(bench_catalog, spec)
        if full_gdf.empty:
            print(f"    {benchmark_key}: empty after filtering - skipping.")
            continue
        if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
            full_gdf = full_gdf.to_crs(4326)

        clusters = v.build_evaluation_clusters(full_gdf, buffer_km, simplify_tol_m, quad_segs)
        print(f"    {len(clusters)} cluster(s) from {len(full_gdf)} benchmark band polygon(s).")

        full_sindex = full_gdf.sindex

        # Keyed by (domain_name, region) - no threshold dimension here.
        acc: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {
            "agree_km2": 0.0, "under_km2": 0.0, "over_km2": 0.0,
            "pop_agree": 0.0, "pop_under": 0.0, "pop_over": 0.0,
        })
        band_coverage_km2: dict[str, float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        n_processed = 0
        n_unclassified = 0

        for _, cluster in clusters.iterrows():
            cluster_geom = cluster.geometry
            bbox = list(cluster_geom.bounds)

            if spec.regions:
                centroid = cluster_geom.centroid
                region = v.region_for_point(centroid.x, centroid.y, spec.regions) or _UNCLASSIFIED_REGION
                if region == _UNCLASSIFIED_REGION:
                    n_unclassified += 1
            else:
                region = country_iso

            chunk_paths = _chunks_overlapping_bbox(bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
            if not chunk_paths:
                continue
            depth, out_transform = _mosaic_read(chunk_paths, bbox)
            if depth is None or depth.size == 0:
                continue

            model_domain = v.model_domain_mask(depth, nodata=-9999.0)
            if not model_domain.any():
                continue
            n_processed += 1

            cluster_domain_mask = rasterize(
                [(cluster_geom, 1)], out_shape=depth.shape, transform=out_transform,
                fill=0, dtype="uint8", all_touched=False,
            ).astype(bool)

            candidate_idx = list(full_sindex.query(cluster_geom, predicate="intersects"))
            cluster_bench_gdf = full_gdf.iloc[candidate_idx]
            ht_min_grid, ht_max_grid = v.rasterize_depth_bands(
                cluster_bench_gdf, spec.ht_min_col, spec.ht_max_col,
                out_transform, "EPSG:4326", depth.shape, open_ended_min_m,
            )
            has_band = ~np.isnan(ht_min_grid)

            water_mask = v.read_permanent_water_mask(
                gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth.shape,
            )
            not_water = ~water_mask

            buffered_domain = model_domain & cluster_domain_mask & not_water
            model_only_domain = model_domain & not_water
            domain_masks = {"buffered": buffered_domain, "model_only": model_only_domain}

            area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
            band_coverage_km2[region] += float((has_band * area_km2 * not_water).sum())

            try:
                pop_da = retry_transient_io(
                    gfm_catalog.get_rasterdataset, population_source, bbox=bbox,
                ).squeeze(drop=True)
            except Exception:
                pop_da = None
            if pop_da is not None and pop_da.size > 0:
                pop_arr = np.nan_to_num(pop_da.values.astype("float64"), nan=0.0)
                pop_nodata = pop_da.raster.nodata
                if pop_nodata is not None:
                    pop_arr[pop_arr == pop_nodata] = 0.0
                pop_transform, pop_crs = pop_da.raster.transform, pop_da.raster.crs
            else:
                pop_arr, pop_transform, pop_crs = None, None, None

            class_agree = has_band & (depth >= ht_min_grid) & (depth <= ht_max_grid)
            class_under = has_band & (depth < ht_min_grid)
            class_over = has_band & (depth > ht_max_grid)

            cluster_agreement = np.full(depth.shape, _CAT_DRY, dtype="uint8")
            buffered = domain_masks["buffered"]
            cluster_agreement[buffered & class_over] = _CAT_OVER
            cluster_agreement[buffered & class_under] = _CAT_UNDER
            still_dry = cluster_agreement == _CAT_DRY
            cluster_agreement[still_dry & buffered & class_agree] = _CAT_AGREE
            agreement_pieces_by_region[region].append((cluster_agreement, out_transform))

            for domain_name in domains:
                d = domain_masks[domain_name]
                agree_km2, under_km2, over_km2 = v.depth_band_counts(depth, ht_min_grid, ht_max_grid, d, area_km2)
                a = acc[(domain_name, region)]
                a["agree_km2"] += agree_km2
                a["under_km2"] += under_km2
                a["over_km2"] += over_km2

                if pop_arr is not None:
                    d_band = d & has_band
                    for cls_name, cls_mask in (
                        ("pop_agree", class_agree), ("pop_under", class_under), ("pop_over", class_over),
                    ):
                        pop_class = v.population_by_class(
                            cls_mask, d_band, out_transform, "EPSG:4326",
                            pop_arr, pop_transform, pop_crs,
                        )
                        a[cls_name] += float(pop_class.sum())

        print(f"    {n_processed}/{len(clusters)} cluster(s) had real model data.")
        if n_unclassified:
            print(
                f"    WARNING: {n_unclassified} cluster(s) fell outside every bbox in this "
                f"benchmark's 'regions' - reported under region='{_UNCLASSIFIED_REGION}'."
            )

        for domain_name, region in sorted(acc.keys()):
            a = acc[(domain_name, region)]
            metrics = v.depth_band_metrics_from_counts(a["agree_km2"], a["under_km2"], a["over_km2"])
            pop_metrics = v.depth_band_metrics_from_counts(a["pop_agree"], a["pop_under"], a["pop_over"])
            all_rows.append(_round_row({
                "country": country_iso, "iso": country_iso, "benchmark_key": benchmark_key,
                "region": region,
                "return_period": rp, "waterlevel_name": slr, "domain": domain_name,
                "benchmark_band_coverage_km2": band_coverage_km2[region],
                "agree_km2": a["agree_km2"], "under_km2": a["under_km2"], "over_km2": a["over_km2"],
                "pct_agree": metrics["pct_agree"], "pct_under": metrics["pct_under"],
                "pct_over": metrics["pct_over"], "depth_EB": metrics["depth_EB"],
                "pop_agree": a["pop_agree"], "pop_under": a["pop_under"], "pop_over": a["pop_over"],
                "pop_pct_agree": pop_metrics["pct_agree"], "pop_pct_under": pop_metrics["pct_under"],
                "pop_pct_over": pop_metrics["pct_over"], "pop_depth_EB": pop_metrics["depth_EB"],
            }))

        for region_name, pieces in agreement_pieces_by_region.items():
            _write_agreement_raster(pieces, cfg, country_iso, rp, slr, region_name, metric="depth_agreement")

    return pd.DataFrame(all_rows)


def validate_country_national_coverage(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
) -> pd.DataFrame:
    """Extent comparison for `coverage == "national"` benchmarks (the ENTIRE
    coastline was assessed, e.g. Norway's Kartverket storm-surge polygons -
    see BenchmarkSpec.coverage's own docstring) - processed per POSTPROCESSING
    CHUNK directly, with NO evaluation-cluster buffer/merge step at all.

    Two independent reasons this is a separate code path from validate_country()'s
    cluster-based one, not a variant of it:

    1. Correctness: build_evaluation_clusters' buffer-around-each-polygon step
       exists specifically to work around PARTIAL coverage (Spain/France survey
       only isolated designated zones, so "outside any benchmark polygon" is
       genuinely ambiguous between "surveyed and dry" and "never looked at" -
       caveats doc §1.1). National coverage has no such ambiguity - a
       model-wet cell outside the benchmark polygon (but within the country's
       assessed domain) is a real, unambiguous over-prediction, not a maybe.
       There is nothing for a buffer to protect against here.
    2. Robustness: confirmed 2026-09 that a national benchmark's long, only-
       lightly-fragmented coastline can make build_evaluation_clusters'
       buffer+merge collapse into ONE cluster spanning almost the entire
       country (Norway: a single cluster with bbox 27deg x 13.5deg). The
       cluster-based path reads/rasterizes at that cluster's own bounding-box
       size, which crashed with a 65.9 GiB allocation. Chunk-sized processing
       (mirroring analysis/compute_flood_totals.py's own chunk-streaming
       architecture) bounds memory per iteration regardless of how far a
       national benchmark's footprint spans.

    Domain is always effectively "the model's own domain, nothing more
    restrictive" (no buffered/model_only distinction - there is no cluster
    window to be "narrower than intended" the way caveats doc §1.2 describes
    for the partial-coverage path) - reported under `domain="model_only"` for
    schema/column continuity with validate_country()'s own output, so both
    functions' rows can be concatenated into one metrics CSV (see main()) -
    EXCEPT it is also masked to `country_iso`'s own territory
    (validation.read_country_mask), since a region's bbox is just a rectangle
    and can genuinely overlap a neighbouring country (confirmed 2026-09:
    Norway's own `mainland` bbox overlaps real Swedish/Danish coastline) -
    without this, GFM's own real flooding in that neighbour would be scored as
    a false positive against a benchmark that never covered it. The
    partial-coverage path never needs this guard because its clusters ARE the
    benchmark's own geometry, not a bbox.
    """
    val_cfg = cfg["validation"]
    rp = val_cfg["return_period"]
    slr = val_cfg["waterlevel_name"]
    thresholds_m = [float(t) for t in val_cfg["depth_thresholds_m"]]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    population_source = val_cfg["population_source"]
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])
    flood_totals_dir = Path(val_cfg["flood_totals_dir"])
    model_total_wet_km2, pop_model_total = _read_flood_totals(flood_totals_dir, country_iso, rp, slr)

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
            raise NotImplementedError(
                f"Raster benchmark dispatch not implemented for {benchmark_key} "
                f"(data_type={spec.data_type}) - only GeoDataFrame benchmarks are wired up."
            )
        if not spec.regions:
            raise ValueError(
                f"{benchmark_key}: coverage=national benchmarks require an explicit "
                "meta.regions block (no evaluation-cluster geometry exists here to derive "
                "a working extent from) - see BenchmarkSpec.regions' own docstring."
            )
        print(f"  National-coverage benchmark: {benchmark_key} ({spec.data_type})")

        full_gdf = v.load_benchmark_full(bench_catalog, spec)
        if full_gdf.empty:
            print(f"    {benchmark_key}: empty after filtering - skipping.")
            continue
        if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
            full_gdf = full_gdf.to_crs(4326)
        full_sindex = full_gdf.sindex

        acc: dict[tuple[float, str], dict[str, float]] = defaultdict(_new_acc)
        benchmark_wet_km2_total: dict[str, float] = defaultdict(float)
        benchmark_wet_outside_model_domain_km2: dict[str, float] = defaultdict(float)
        model_wet_km2: dict[tuple[float, str], float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        n_processed = 0
        n_chunks_total = 0

        for region, region_bbox in spec.regions.items():
            chunk_paths = _chunks_overlapping_bbox(region_bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
            n_chunks_total += len(chunk_paths)

            for chunk_path in chunk_paths:
                with retry_transient_io(rasterio.open, chunk_path) as src:
                    depth = src.read(1)
                    out_transform = src.transform

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
                benchmark_wet = v.wet_mask_from_fraction(fraction, wet_fraction)

                water_mask = v.read_permanent_water_mask(
                    gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth.shape,
                )
                not_water = ~water_mask
                # region_bbox is a rectangle, not the benchmark's own real coastline - it can
                # genuinely overlap a neighbouring country (confirmed 2026-09: Norway's own
                # `mainland` bbox overlaps real Swedish/Danish territory, whose own real
                # GFM-modelled flooding would otherwise be scored as a Norwegian false
                # positive). The partial-coverage path never needs this because its clusters
                # ARE the benchmark's own geometry; there is no equivalent guardrail here
                # without it.
                in_country = v.read_country_mask(
                    gfm_catalog, geogunit_source, iso_lookup, country_iso, bbox, out_transform, depth.shape,
                )
                not_water = not_water & in_country
                domain_mask = model_domain & not_water

                area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])
                benchmark_wet_km2_total[region] += float((fraction * area_km2 * not_water).sum())
                benchmark_wet_outside_model_domain_km2[region] += float(
                    (fraction * area_km2 * not_water * (~model_domain)).sum()
                )

                try:
                    pop_da = retry_transient_io(
                        gfm_catalog.get_rasterdataset, population_source, bbox=bbox,
                    ).squeeze(drop=True)
                except Exception:
                    pop_da = None
                if pop_da is not None and pop_da.size > 0:
                    pop_arr = np.nan_to_num(pop_da.values.astype("float64"), nan=0.0)
                    pop_nodata = pop_da.raster.nodata
                    if pop_nodata is not None:
                        pop_arr[pop_arr == pop_nodata] = 0.0
                    pop_transform, pop_crs = pop_da.raster.transform, pop_da.raster.crs
                else:
                    pop_arr, pop_transform, pop_crs = None, None, None

                chunk_agreement = np.full(depth.shape, _CAT_DRY, dtype="uint8")

                for threshold_m in thresholds_m:
                    model_wet = model_domain & (depth > threshold_m) & not_water
                    model_wet_km2[(threshold_m, region)] += float((model_wet * area_km2).sum())

                    tp, fp, fn, tn = v.confusion_counts(model_wet, benchmark_wet, domain_mask, area_km2)
                    a = acc[(threshold_m, region)]
                    a["tp_km2"] += tp
                    a["fp_km2"] += fp
                    a["fn_km2"] += fn
                    a["tn_km2"] += tn
                    a["tp_cells"] += int((domain_mask & model_wet & benchmark_wet).sum())
                    a["fp_cells"] += int((domain_mask & model_wet & ~benchmark_wet).sum())
                    a["fn_cells"] += int((domain_mask & ~model_wet & benchmark_wet).sum())
                    a["tn_cells"] += int((domain_mask & ~model_wet & ~benchmark_wet).sum())

                    if pop_arr is not None:
                        for cls_name, cls_mask in (
                            ("pop_tp", model_wet & benchmark_wet),
                            ("pop_fp", model_wet & ~benchmark_wet),
                            ("pop_fn", ~model_wet & benchmark_wet),
                            ("pop_tn", ~model_wet & ~benchmark_wet),
                        ):
                            pop_class = v.population_by_class(
                                cls_mask, domain_mask, out_transform, "EPSG:4326",
                                pop_arr, pop_transform, pop_crs,
                            )
                            a[cls_name] += float(pop_class.sum())

                    if threshold_m == primary_threshold_m:
                        chunk_agreement[domain_mask & model_wet & ~benchmark_wet] = _CAT_OVER
                        chunk_agreement[domain_mask & ~model_wet & benchmark_wet] = _CAT_UNDER
                        still_dry = chunk_agreement == _CAT_DRY
                        chunk_agreement[still_dry & domain_mask & model_wet & benchmark_wet] = _CAT_AGREE

                agreement_pieces_by_region[region].append((chunk_agreement, out_transform))

        print(f"    {n_processed}/{n_chunks_total} chunk(s) had real model data.")

        for threshold_m, region in sorted(acc.keys()):
            a = acc[(threshold_m, region)]
            metrics = v.metrics_from_counts(a["tp_km2"], a["fp_km2"], a["fn_km2"], a["tn_km2"])
            pop_metrics = v.metrics_from_counts(a["pop_tp"], a["pop_fp"], a["pop_fn"], a["pop_tn"])
            bm_total = benchmark_wet_km2_total[region]
            bm_outside = benchmark_wet_outside_model_domain_km2[region]
            bm_wet_pct = 100.0 * bm_outside / bm_total if bm_total > 0 else float("nan")
            is_primary = threshold_m == primary_threshold_m
            all_rows.append(_round_row({
                "country": country_iso, "iso": country_iso, "benchmark_key": benchmark_key,
                "region": region,
                "return_period": rp, "waterlevel_name": slr,
                "threshold_m": threshold_m, "domain": "model_only",
                "tp_km2": a["tp_km2"], "fp_km2": a["fp_km2"], "fn_km2": a["fn_km2"], "tn_km2": a["tn_km2"],
                "tp_cells": a["tp_cells"], "fp_cells": a["fp_cells"],
                "fn_cells": a["fn_cells"], "tn_cells": a["tn_cells"],
                "benchmark_wet_km2": bm_total,
                "model_wet_km2": model_wet_km2[(threshold_m, region)],
                "benchmark_wet_outside_model_domain_km2": bm_outside,
                "benchmark_wet_outside_model_domain_pct": bm_wet_pct,
                "model_total_wet_km2": model_total_wet_km2 if is_primary else float("nan"),
                "pop_model_total": pop_model_total if is_primary else float("nan"),
                "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"],
                "EB": metrics["EB"], "EB_ratio": metrics["EB_ratio"], "bias": metrics["bias"],
                "pop_tp": a["pop_tp"], "pop_fp": a["pop_fp"], "pop_fn": a["pop_fn"],
                "pop_benchmark": a["pop_tp"] + a["pop_fn"], "pop_model": a["pop_tp"] + a["pop_fp"],
                "pop_over": a["pop_fp"], "pop_under": a["pop_fn"],
                "pop_diff_net": a["pop_fp"] - a["pop_fn"],
                "pop_HR": pop_metrics["HR"], "pop_FAR": pop_metrics["FAR"], "pop_CSI": pop_metrics["CSI"],
            }))

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
    """Mosaic one region's clusters' native-resolution agreement category grids
    down to `plots.resolution_m`, using MAX resampling on the category codes -
    since codes are ordered dry(0) < agree(1) < under(2) < over(3), MAX
    exactly implements the plan's "priority: over > under > agree > dry"
    rule (plan doc §5.4) without a hand-rolled block-reduce. Shared by both
    the extent comparison (`metric="agreement"`, the default) and the
    depth-band comparison (`metric="depth_agreement"`, validate_country_depth_bands)
    - the category codes/priority-reduction logic and output shape are
    identical, only the classification rule that produced the category grid
    differs, and the output filename prefix keeps the two from colliding.

    One raster PER REGION, not one combined per country - a country's named
    regions (e.g. Spain's mainland vs. Canary Islands) can be thousands of km
    apart, so mosaicking them into one array would mean a huge, mostly-empty
    raster spanning the gap between them. plot_agreement_map.py reads all of a
    country's per-region rasters and shows the largest as the main map, the
    rest as small inset panels.
    """
    if not agreement_pieces:
        return
    val_cfg = cfg["validation"]
    res_m = float(val_cfg["plots"]["resolution_m"])
    res_deg = res_m / (111.32 * 1000.0)

    all_bounds = [rasterio.transform.array_bounds(a.shape[0], a.shape[1], t) for a, t in agreement_pieces]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--country", required=True, help="ISO-3 country code, e.g. ESP")
    parser.add_argument(
        "--run-tag", default=None,
        help="tags every output row with this value in a run_tag column - for "
             "aggregating results across multiple runs (e.g. the calibration sweep's "
             "{group}__{sweep_point} identity, see scripts/calibration/aggregate_calibration_results.py). "
             "Optional - omit for a normal single/production validation run.",
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

    print(f"=== Validating {country_iso} (extent) ===")
    df_partial = validate_country(country_iso, cfg, gfm_catalog, bench_catalog)
    print(f"\n=== Validating {country_iso} (extent, national coverage) ===")
    df_national = validate_country_national_coverage(country_iso, cfg, gfm_catalog, bench_catalog)
    df = pd.concat([df_partial, df_national], ignore_index=True)
    if df.empty:
        print("No extent rows produced.")
    else:
        if args.run_tag:
            df["run_tag"] = args.run_tag
        out_path = out_dir / f"metrics_{country_iso}_{rp}_{slr}.csv"
        atomic_write(out_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
        print(f"\nMetrics written: {out_path} ({len(df)} row(s))")

        # coverage=national benchmarks only ever produce "model_only" rows (no
        # cluster/buffer concept - see validate_country_national_coverage's own
        # docstring), so prefer "buffered" per (benchmark_key, region) but fall
        # back to whatever domain that combination actually has, rather than
        # silently printing nothing for a national-coverage country.
        at_primary = df[df["threshold_m"] == float(val_cfg["primary_threshold_m"])]
        preferred = (
            # "buffered" > "model_only" alphabetically, so descending-sort puts
            # "model_only" first and "buffered" last per (benchmark_key, region)
            # group; keep="last" then picks "buffered" when both exist.
            at_primary.sort_values("domain", ascending=False)
            .drop_duplicates(subset=["benchmark_key", "region"], keep="last")
        )
        for _, row in preferred.iterrows():
            print(
                f"\n[{row['region']}] Primary threshold ({val_cfg['primary_threshold_m']}m), {row['domain']} domain:\n"
                f"  HR={row['HR']:.3f} FAR={row['FAR']:.3f} CSI={row['CSI']:.3f} EB={row['EB']:.3f}\n"
                f"  benchmark_wet_outside_model_domain_pct={row['benchmark_wet_outside_model_domain_pct']:.1f}%"
                " - health check: if this is high, don't quote the rest of this row as a model result."
            )

    print(f"\n=== Validating {country_iso} (depth bands) ===")
    df_depth = validate_country_depth_bands(country_iso, cfg, gfm_catalog, bench_catalog)
    if df_depth.empty:
        print("No depth-band rows produced.")
        return

    if args.run_tag:
        df_depth["run_tag"] = args.run_tag
    depth_out_path = out_dir / f"depth_metrics_{country_iso}_{rp}_{slr}.csv"
    atomic_write(depth_out_path, lambda f: df_depth.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"\nDepth-band metrics written: {depth_out_path} ({len(df_depth)} row(s))")

    depth_primary = df_depth[df_depth["domain"] == "buffered"]
    for _, row in depth_primary.iterrows():
        print(
            f"\n[{row['region']}] Depth-band comparison, buffered domain:\n"
            f"  pct_agree={row['pct_agree']:.3f} pct_under={row['pct_under']:.3f} "
            f"pct_over={row['pct_over']:.3f} depth_EB={row['depth_EB']:.3f}\n"
            f"  benchmark_band_coverage_km2={row['benchmark_band_coverage_km2']}"
        )


if __name__ == "__main__":
    main()
