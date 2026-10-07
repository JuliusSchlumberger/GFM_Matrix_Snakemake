"""Per-country coastal flood-extent validation driver.

For one country, finds every applicable coastal benchmark in the validation
catalog and scores it against production model output. Partial-coverage
benchmarks (`coverage: "partial"`, the default - Spain/France's isolated
designated survey zones) use `validate_country()`: the evaluation DOMAIN
comes from the benchmark agency's own official study design
(`meta.study_area` in data_catalog_validation.yml), not an arbitrary
distance buffer -

- `geometry_type: "perimeter"` (France's TRI zones): each zone's own polygon
  used AS-IS (point-in-polygon, no buffer) as the domain - one evaluation
  unit per TRI zone that actually has coastal extent rows in this specific
  benchmark (`src/validation.py::units_with_extent_coverage`/`tri_domain_mask`).
- `geometry_type: "segments"` (Spain's ARPSI coastal seed lines): a SEEDED
  CONNECTIVITY domain instead - two independent 8-connected-component
  analyses (one on the model's own wet mask, one on the benchmark's),
  keeping only components that touch a seed cell, domain = union of both
  (`src/validation.py::connectivity_domain_mask`). A line is a 1D seed, not
  an area with its own extent - "near the line" was never the right
  domain-construction question the way it is for a perimeter polygon.

(2026-10 - replaced the earlier flat `eval_domain.buffer_km` margin and its
"buffered"/"model_only" domain-variant reporting; `build_evaluation_clusters`
survives only as a chunk-windowing optimization for the "segments" path, not
as the scoring domain - see its own docstring.)

Benchmarks whose source assessed the ENTIRE coastline (`coverage: "national"`,
e.g. Norway's Kartverket data) instead use `validate_country_national_coverage()`'s
per-postprocessing-chunk path - see that function's own docstring.

Scoring itself is the soft/continuous confusion matrix throughout
(`v.confusion_counts_soft`) at a single config-driven threshold
(`validation.primary_threshold_m`) - no threshold sweep, no population
weighting (removed 2026-10 - not needed by this pipeline).

Every region row also carries a tolerant-confusion diagnostic (CSI_tol/
HR_tol/FAR_tol, `v.confusion_counts_tolerant`, `validation.coastline_tolerance_cells`
cells) alongside the strict CSI/HR/FAR - NOT a substitute for them, never
used for any decision this pipeline makes on its own. It answers "how much
of the strict disagreement sits right next to a cell where model and
benchmark actually agree" (a small-scale boundary-registration effect, e.g.
the permanent-water-mask-vs-benchmark-coastline mismatch described in
methods_04b_MapsValidation.md's caveats), reported together with
pct_disagreement_forgiven so the size of the effect is always visible next
to the number it's explaining, not hidden behind it. Applies uniformly to
every benchmark (partial and national coverage) - it is a generic
registration check, not a per-country special case.

If the benchmark's catalog entry defines `meta.regions: {name: [minx,miny,maxx,maxy]}`,
each evaluation unit is tagged by which named region its centroid falls in (e.g.
Spain's `mainland` vs `canary_islands`) and metrics are reported/plotted per region
instead of blended into one national row; a country with no `regions` configured yet
gets a single implicit region (its own ISO code). `_blend_regions` additionally folds
every region into one country-level row (auto-detecting whether the regions are a
disjoint partition - summed - or a nested whole-territory + sub-area set, like Wales/
Scotland/New Brunswick - the whole-territory region's own numbers ARE the total, not
re-summed with its nested sub-areas, which would double-count them).

Writes:
  {validation.output_dir}/{country}/metrics_{country}_{RP}_{SLR}.csv
  {validation.output_dir}/{country}/agreement_{country}_{region}_{RP}_{SLR}.tif (one per region)
    (plots.resolution_m, priority-reduced: over > under > agree > dry -
    see _write_agreement_raster)

See docs/methods_04b_MapsValidation.md for the current design; src/validation.py
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

import geopandas as gpd
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
from tiles import load_tile_grid  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Agreement-category codes, ordered so Resampling.max correctly implements
# a "priority: over > under > agree > dry" rule when a display raster does
# need coarsening (_write_agreement_raster, whole-country-scale regions
# only) - the highest code present in a coarse cell wins.
_CAT_DRY = 0
_CAT_AGREE = 1
_CAT_UNDER = 2
_CAT_OVER = 3

# Label for a cluster whose centroid falls in none of a country's `regions:`
# bboxes (config error / gap in the partition, not expected in normal operation -
# see validation.region_for_point).
_UNCLASSIFIED_REGION = "unclassified"

# _write_agreement_raster's own native-vs-coarsened resolution cutoff (pixels,
# either dimension) - every evaluation unit/sub-region today (TRI zones, ARPSI
# components, Wales/Scotland/New Brunswick's own nested sub-areas) is well
# under this at the model's native 30m resolution (a few hundred to ~1000px
# across), so stays at native resolution; a whole-country region (Norway,
# Scotland mainland) is thousands of px across at 30m and falls back to
# `plots.resolution_m` instead. See that function's own docstring for why
# this matters (MAX-resampling a coarsened category raster visually
# overstates disagreement - confirmed on Firth of Forth).
_AGREEMENT_RASTER_MAX_DIM_PX = 2500


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


def _tile_paths_overlapping_bbox(
    bbox: list[float], tile_grid: gpd.GeoDataFrame, model_outputs: str, rp: str, slr: str,
) -> list[Path]:
    """Every production tile's own raw `results/waterdepth_{rp}_{slr}.tif` whose
    tile intersects `bbox` AND already exists on disk - the direct per-tile
    counterpart to `_chunks_overlapping_bbox`, for scoring a small, ad hoc
    tile subset that was never (and is not meant to be) run through
    `merge_chunk` (2026-10-08 - a country-validation run over a handful of
    tiles would otherwise require simulating every OTHER tile sharing their
    5deg postprocessing chunk too, see merge_chunk's own `waterdepth_tiles_for_chunk`
    input function in the Snakefile). A tile simply not yet simulated is
    silently skipped here, same spirit as `_chunks_overlapping_bbox` skipping
    a chunk that was never merged - not an error, just not part of this run.
    """
    region_geom = box(*bbox)
    hits = tile_grid[tile_grid.geometry.intersects(region_geom)]
    paths = []
    for tid in hits["tile_id"].astype(int):
        p = Path(model_outputs) / str(tid) / "results" / f"waterdepth_{rp}_{slr}.tif"
        if p.exists():
            paths.append(p)
    return paths


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

    CSI_tol/HR_tol/FAR_tol (v.confusion_counts_tolerant) are a diagnostic
    companion to CSI/HR/FAR, not a substitute - always read them next to
    pct_disagreement_forgiven (how much of the STRICT disagreement the
    tolerance check actually excused): a high CSI_tol with a high
    pct_disagreement_forgiven means "mostly boundary-registration noise,"
    the same CSI_tol with a low pct_disagreement_forgiven would be
    suspicious (shouldn't happen by construction - see that function's own
    docstring - but the column is there so it's checkable, not assumed).
    """
    int_cols = (
        "tp_km2", "fp_km2", "fn_km2", "tn_km2",
        "benchmark_wet_km2", "model_wet_km2", "benchmark_wet_outside_model_domain_km2",
        "model_total_wet_km2",
        "tp_tol_km2", "fp_tol_km2", "fn_tol_km2", "tn_tol_km2",
        "fp_forgiven_km2", "fn_forgiven_km2",
        # depth-band comparison (validate_country_depth_bands)
        "agree_km2", "under_km2", "over_km2", "benchmark_band_coverage_km2",
    )
    ratio_cols = (
        "benchmark_wet_outside_model_domain_pct",
        "HR", "FAR", "CSI", "EB", "EB_ratio", "bias",
        "HR_tol", "FAR_tol", "CSI_tol", "pct_disagreement_forgiven",
        # depth-band comparison (validate_country_depth_bands)
        "pct_agree", "pct_under", "pct_over", "depth_EB",
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
        # Tolerant confusion matrix (v.confusion_counts_tolerant) - a
        # diagnostic companion to the tp/fp/fn/tn above, never a substitute.
        # See _round_row's own CSI_tol/pct_disagreement_forgiven comment.
        "tp_tol_km2": 0.0, "fp_tol_km2": 0.0, "fn_tol_km2": 0.0, "tn_tol_km2": 0.0,
        "fp_forgiven_km2": 0.0, "fn_forgiven_km2": 0.0,
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


_BLENDED_REGION = "ALL"  # country-level blended row's own region label, from _blend_regions


def _blend_regions(rows: list[dict], regions_spec: dict[str, list[float]] | None) -> dict | None:
    """One country-level row, folding every region row in `rows` (all sharing
    the same benchmark_key/return_period/waterlevel_name - callers group by
    that first) into a single total - "region pooling" (2026-09 through
    2026-10: left to the consumer, see methods_04b_MapsValidation.md's own
    former note to that effect) is now part of the pipeline itself.

    Two region-partition shapes exist in this catalog and need different
    handling, auto-detected by bbox containment (no new catalog field):

    - DISJOINT partition (Spain mainland/canary_islands, France metropole +
      5 overseas territories, Norway/Finland's own single region): every
      region covers genuinely separate ground, so summing tp/fp/fn/tn
      across all of them and recomputing HR/FAR/CSI/bias from the pooled
      sums gives a real country-wide number.
    - NESTED whole-territory + sub-area(s) (Wales wales+severn_estuary+
      menai_strait, Scotland scotland+firth_of_forth, New Brunswick
      new_brunswick+cumberland_basin+petitcodiac): a sub-area's bbox sits
      entirely inside the whole-territory bbox and is independently
      re-scored over the same ground (see each catalog entry's own
      known_caveats) - summing would double-count the sub-area's cells.
      The bbox that CONTAINS every sibling's bbox is the whole-territory
      region; its own row already IS the country total, used as-is.

    Returns None if `rows` is empty (nothing to blend) or there is only one
    region row already (blending a single row would just be a relabeled
    copy of it - not useful, and ambiguous with the real "ALL" semantics
    once a country legitimately has only one region).
    """
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
        "benchmark_wet_outside_model_domain_pct": float("nan"),  # not a meaningful sum - see per-region rows
        "model_total_wet_km2": first.get("model_total_wet_km2", float("nan")),  # already country-wide, not per-region
        "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"],
        "EB": metrics["EB"], "EB_ratio": metrics["EB_ratio"], "bias": metrics["bias"],
        "tp_tol_km2": tp_tol, "fp_tol_km2": fp_tol, "fn_tol_km2": fn_tol, "tn_tol_km2": tn_tol,
        "fp_forgiven_km2": fp_forgiven, "fn_forgiven_km2": fn_forgiven,
        "HR_tol": tol_metrics["HR"], "FAR_tol": tol_metrics["FAR"], "CSI_tol": tol_metrics["CSI"],
        "pct_disagreement_forgiven": pct_forgiven,
    })


def validate_country(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
) -> pd.DataFrame:
    val_cfg = cfg["validation"]
    rp = val_cfg["return_period"]
    slr = val_cfg["waterlevel_name"]
    primary_threshold_m = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    wet_fraction = float(val_cfg["benchmark_wet_fraction"])
    tolerance_cells = int(val_cfg["coastline_tolerance_cells"])
    windowing_buffer_km = float(val_cfg["eval_domain"]["buffer_km"])
    simplify_tol_m = float(val_cfg["vector_simplify_tolerance_m"])
    quad_segs = int(val_cfg["buffer_quad_segs"])
    pp_chunk_deg = float(cfg["postprocessing"]["chunk_size_deg"])
    merged_chunks_dir = Path(cfg["postprocessing"]["merged_outputs"]) / "chunks"
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    flood_totals_dir = Path(val_cfg["flood_totals_dir"])
    model_total_wet_km2, _ = _read_flood_totals(flood_totals_dir, country_iso, rp, slr)
    if model_total_wet_km2 != model_total_wet_km2:  # NaN
        print(
            f"  NOTE: no flood_totals_{country_iso}.csv (or no matching {rp}/{slr} row) in "
            f"{flood_totals_dir} - model_total_wet_km2 will be NaN. Run "
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
        if not spec.study_area:
            raise ValueError(
                f"{benchmark_key}: coverage='partial' benchmarks require meta.study_area "
                "(2026-10 - the earlier flat eval_domain.buffer_km margin was removed; the "
                "evaluation domain always comes from the benchmark agency's own study design "
                "now) - see BenchmarkSpec.study_area's own docstring."
            )
        print(f"  Benchmark: {benchmark_key} ({spec.data_type})")

        full_gdf = v.load_benchmark_full(bench_catalog, spec)
        if full_gdf.empty:
            print(f"    {benchmark_key}: empty after filtering - skipping.")
            continue
        if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
            full_gdf = full_gdf.to_crs(4326)

        study = spec.study_area
        study_gdf = bench_catalog.get_geodataframe(study["source"])
        if study_gdf is None or study_gdf.empty:
            print(f"    study_area source {study['source']!r}: empty - skipping.")
            continue
        if study.get("attribute_filter"):
            mask = np.ones(len(study_gdf), dtype=bool)
            for col, val in study["attribute_filter"].items():
                mask &= (study_gdf[col] == val).to_numpy()
            study_gdf = study_gdf.loc[mask]
        if study_gdf.crs is None or study_gdf.crs.to_epsg() != 4326:
            study_gdf = study_gdf.to_crs(4326)

        geometry_type = study["geometry_type"]
        full_sindex = full_gdf.sindex  # reused per unit to find that unit's own benchmark polygons

        acc: dict[str, dict[str, float]] = defaultdict(_new_acc)  # keyed by region only (no more threshold/domain dims)
        benchmark_wet_km2_total: dict[str, float] = defaultdict(float)
        benchmark_wet_outside_model_domain_km2: dict[str, float] = defaultdict(float)
        model_wet_km2: dict[str, float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        unit_rows: list[dict] = []  # per-evaluation-unit CSI, for the CSI-dot map (plot_agreement_map.py)
        n_processed = 0
        n_units = 0
        n_unclassified = 0

        def _record_unit(unit_id, region, lon, lat, model_wet, fraction, unit_mask, area_km2) -> None:
            """One row per evaluation unit (a TRI zone, or - for the
            "segments" geometry_type - a distinct connected component) for
            the CSI-dot map: that unit's OWN CSI, restricted to its own
            cells only (`unit_mask`), not the region/country-level blend.
            """
            tp, fp, fn, _tn = v.confusion_counts_soft(model_wet, fraction, unit_mask, area_km2)
            csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
            hr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
            unit_rows.append({
                "unit_id": unit_id, "region": region, "lon": lon, "lat": lat,
                "csi": csi, "hr": hr, "far": far, "n_cells": int(unit_mask.sum()),
            })

        def _region_for(geom) -> str:
            nonlocal n_unclassified
            if spec.regions:
                c = geom.centroid
                r = v.region_for_point(c.x, c.y, spec.regions) or _UNCLASSIFIED_REGION
                if r == _UNCLASSIFIED_REGION:
                    n_unclassified += 1
                return r
            return country_iso

        def _window_fraction(depth_transform, depth_shape, bbox) -> np.ndarray:
            """This window's own benchmark coverage fraction (0-1 per cell) -
            computed once per window and reused for both scoring (_score,
            soft-weighted) and, for the "segments" branch, the connectivity
            domain's own binary benchmark-wet mask - avoids rasterizing the
            same benchmark geometries twice over the same grid.
            """
            candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
            window_bench_gdf = full_gdf.iloc[candidate_idx]
            return v.benchmark_fraction_from_vector(
                window_bench_gdf, depth_transform, "EPSG:4326", depth_shape, supersample=supersample,
            )

        def _score(depth, depth_transform, fraction, domain_mask, not_water, region):
            """Shared scoring for one already-mosaicked, already-domain-resolved
            window: classifies model_wet at the single config threshold,
            scores this window's own (already-rasterized) benchmark coverage
            `fraction` via confusion_counts_soft, accumulates into `acc`/
            `benchmark_wet_km2_total`/`model_wet_km2` by region, and appends
            this window's own agreement-category array. `depth`/`depth_transform`,
            `fraction` (_window_fraction), and `domain_mask`/`not_water` are all
            built by the caller - the two geometry_type branches below need
            `fraction`/the depth grid before this point anyway, to build the
            domain itself, so no redundant second computation here.

            Returns `(model_wet, d, area_km2)` (or `None` if this window had
            no real model data) so callers can compute per-unit CSI
            (_record_unit) from the same arrays without recomputing them.
            """
            nonlocal n_processed
            model_domain = v.model_domain_mask(depth, nodata=-9999.0)
            if not model_domain.any():
                return None
            n_processed += 1

            benchmark_wet_display = v.wet_mask_from_fraction(fraction, wet_fraction)  # display only

            d = domain_mask & model_domain
            model_wet = model_domain & (depth > primary_threshold_m) & not_water
            area_km2 = pixel_area_km2_grid(depth_transform, depth.shape[1], depth.shape[0])

            benchmark_wet_km2_total[region] += float((fraction * area_km2 * not_water).sum())
            benchmark_wet_outside_model_domain_km2[region] += float(
                (fraction * area_km2 * not_water * (~model_domain)).sum()
            )
            model_wet_km2[region] += float((model_wet * area_km2).sum())

            tp, fp, fn, tn = v.confusion_counts_soft(model_wet, fraction, d, area_km2)
            a = acc[region]
            a["tp_km2"] += tp
            a["fp_km2"] += fp
            a["fn_km2"] += fn
            a["tn_km2"] += tn
            tp_c, fp_c, fn_c, tn_c = v.confusion_counts_soft(model_wet, fraction, d, np.ones_like(area_km2))
            a["tp_cells"] += tp_c
            a["fp_cells"] += fp_c
            a["fn_cells"] += fn_c
            a["tn_cells"] += tn_c

            tol = v.confusion_counts_tolerant(model_wet, fraction, d, area_km2, tolerance_cells)
            a["tp_tol_km2"] += tol["tp"]
            a["fp_tol_km2"] += tol["fp"]
            a["fn_tol_km2"] += tol["fn"]
            a["tn_tol_km2"] += tol["tn"]
            a["fp_forgiven_km2"] += tol["fp_forgiven"]
            a["fn_forgiven_km2"] += tol["fn_forgiven"]

            cat = np.full(depth.shape, _CAT_DRY, dtype="uint8")
            cat[d & model_wet & ~benchmark_wet_display] = _CAT_OVER
            cat[d & ~model_wet & benchmark_wet_display] = _CAT_UNDER
            still_dry = cat == _CAT_DRY
            cat[still_dry & d & model_wet & benchmark_wet_display] = _CAT_AGREE
            agreement_pieces_by_region[region].append((cat, depth_transform))
            return model_wet, d, area_km2

        if geometry_type == "perimeter":
            # France: one evaluation unit per TRI zone that actually has
            # coastal extent rows in THIS benchmark (drops the fluvial-only
            # zones france_tri_perimeters also contains) - that zone's own
            # polygon, used as-is, no buffer, IS the domain.
            id_col = study.get("id_col", "id_tri")
            units_gdf = v.units_with_extent_coverage(study_gdf, full_gdf, id_col=id_col)
            n_units = len(units_gdf)
            print(f"    {n_units} TRI zone(s) with real coastal extent coverage (of {len(study_gdf)} total).")
            for _, unit in units_gdf.iterrows():
                geom = unit.geometry
                bbox = list(geom.bounds)
                region = _region_for(geom)

                chunk_paths = _chunks_overlapping_bbox(bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
                if not chunk_paths:
                    continue
                depth, depth_transform = _mosaic_read(chunk_paths, bbox)
                if depth is None or depth.size == 0:
                    continue
                water_mask = v.read_permanent_water_mask(
                    gfm_catalog, permanent_water_source, permanent_water_codes, bbox, depth_transform, depth.shape,
                )
                not_water = ~water_mask
                tri_mask = v.tri_domain_mask(
                    gpd.GeoDataFrame(geometry=[geom], crs=4326), depth_transform, "EPSG:4326", depth.shape,
                )
                domain_mask = tri_mask & not_water
                fraction = _window_fraction(depth_transform, depth.shape, bbox)
                result = _score(depth, depth_transform, fraction, domain_mask, not_water, region)
                if result is not None:
                    model_wet, d, area_km2 = result
                    zone_id = unit[id_col]
                    centroid = geom.centroid
                    _record_unit(zone_id, region, centroid.x, centroid.y, model_wet, fraction, d, area_km2)

        elif geometry_type == "segments":
            # Spain: no natural one-row-per-unit structure in the seed table
            # (hundreds of ARPSI segments) - group nearby segments into
            # windows purely for efficient chunk reading (build_evaluation_clusters,
            # windowing only - see its own docstring), then score each window's
            # FULL benchmark-wet/model-wet connectivity domain using every seed
            # line that falls in it.
            windows_gdf = v.build_evaluation_clusters(study_gdf, windowing_buffer_km, simplify_tol_m, quad_segs)
            n_units = len(windows_gdf)
            print(f"    {n_units} window(s) from {len(study_gdf)} ARPSI seed segment(s).")
            study_sindex = study_gdf.sindex
            component_counter = 0
            for _, window in windows_gdf.iterrows():
                window_geom = window.geometry
                bbox = list(window_geom.bounds)
                region = _region_for(window_geom)

                chunk_paths = _chunks_overlapping_bbox(bbox, pp_chunk_deg, merged_chunks_dir, rp, slr)
                if not chunk_paths:
                    continue
                depth, depth_transform = _mosaic_read(chunk_paths, bbox)
                if depth is None or depth.size == 0:
                    continue
                water_mask = v.read_permanent_water_mask(
                    gfm_catalog, permanent_water_source, permanent_water_codes, bbox, depth_transform, depth.shape,
                )
                not_water = ~water_mask
                model_wet_raw = (depth != -9999.0) & (depth > primary_threshold_m)

                seed_idx = list(study_sindex.query(window_geom, predicate="intersects"))
                window_seed_gdf = study_gdf.iloc[seed_idx]
                fraction = _window_fraction(depth_transform, depth.shape, bbox)
                benchmark_wet_binary = v.wet_mask_from_fraction(fraction, wet_fraction)

                domain_mask = v.connectivity_domain_mask(
                    model_wet_raw, benchmark_wet_binary, window_seed_gdf, depth_transform, "EPSG:4326", not_water,
                )
                result = _score(depth, depth_transform, fraction, domain_mask, not_water, region)
                if result is not None:
                    model_wet, d, area_km2 = result
                    # One evaluation unit per distinct connected component of
                    # the scored domain - see connectivity_components' own
                    # docstring for why this (not per-original-ARPSI-row).
                    labels = v.connectivity_components(d)
                    for lbl in range(1, int(labels.max()) + 1):
                        component_mask = labels == lbl
                        rows, cols = np.nonzero(component_mask)
                        if rows.size == 0:
                            continue
                        component_counter += 1
                        lon, lat = rasterio.transform.xy(depth_transform, float(rows.mean()), float(cols.mean()))
                        _record_unit(
                            f"component_{component_counter}", region, lon, lat,
                            model_wet, fraction, component_mask, area_km2,
                        )
        else:
            raise ValueError(
                f"{benchmark_key}: meta.study_area.geometry_type={geometry_type!r} not recognised "
                "(expected 'perimeter' or 'segments')."
            )

        print(f"    {n_processed}/{n_units} unit(s)/window(s) had real model data.")
        if n_unclassified:
            print(
                f"    WARNING: {n_unclassified} unit(s) fell outside every bbox in this "
                f"benchmark's 'regions' - reported under region='{_UNCLASSIFIED_REGION}'."
            )

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

        if unit_rows:
            out_dir = Path(val_cfg["output_dir"]) / country_iso
            retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
            units_path = out_dir / f"units_{benchmark_key}_{rp}_{slr}.csv"
            units_df = pd.DataFrame(unit_rows)
            atomic_write(units_path, lambda f: units_df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
            print(f"    Per-unit CSI written: {units_path} ({len(units_df)} unit(s))")

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

        print(f"    {n_processed}/{len(clusters)} cluster(s) had real model data.")
        if n_unclassified:
            print(
                f"    WARNING: {n_unclassified} cluster(s) fell outside every bbox in this "
                f"benchmark's 'regions' - reported under region='{_UNCLASSIFIED_REGION}'."
            )

        for domain_name, region in sorted(acc.keys()):
            a = acc[(domain_name, region)]
            metrics = v.depth_band_metrics_from_counts(a["agree_km2"], a["under_km2"], a["over_km2"])
            all_rows.append(_round_row({
                "country": country_iso, "iso": country_iso, "benchmark_key": benchmark_key,
                "region": region,
                "return_period": rp, "waterlevel_name": slr, "domain": domain_name,
                "benchmark_band_coverage_km2": band_coverage_km2[region],
                "agree_km2": a["agree_km2"], "under_km2": a["under_km2"], "over_km2": a["over_km2"],
                "pct_agree": metrics["pct_agree"], "pct_under": metrics["pct_under"],
                "pct_over": metrics["pct_over"], "depth_EB": metrics["depth_EB"],
            }))

        for region_name, pieces in agreement_pieces_by_region.items():
            _write_agreement_raster(pieces, cfg, country_iso, rp, slr, region_name, metric="depth_agreement")

    return pd.DataFrame(all_rows)


def validate_country_national_coverage(
    country_iso: str,
    cfg: dict,
    gfm_catalog,
    bench_catalog,
    read_tiles_directly: bool = False,
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

    Domain is always "the model's own domain, nothing more restrictive"
    (there is no cluster/buffer concept here at all, unlike the
    partial-coverage path's TRI-zone/connectivity domains - every cell the
    model computed something for is in scope) - EXCEPT it is also masked to
    `country_iso`'s own territory
    (validation.read_country_mask), since a region's bbox is just a rectangle
    and can genuinely overlap a neighbouring country (confirmed 2026-09:
    Norway's own `mainland` bbox overlaps real Swedish/Danish coastline) -
    without this, GFM's own real flooding in that neighbour would be scored as
    a false positive against a benchmark that never covered it. The
    partial-coverage path never needs this guard because its clusters ARE the
    benchmark's own geometry, not a bbox.

    `read_tiles_directly` (default False): read each region's model depth by
    mosaicking production's own per-tile `results/waterdepth_{rp}_{slr}.tif`
    files directly (`_tile_paths_overlapping_bbox` + `_mosaic_read`) instead
    of `merge_chunk`'s merged 5deg chunk files. For a small, ad hoc tile
    subset (e.g. a country-validation run over a handful of tiles) that was
    deliberately never run through `merge_chunk` - doing so would require
    simulating every OTHER tile sharing the same 5deg chunk too, see
    `_tile_paths_overlapping_bbox`'s own docstring. One mosaic read per
    region instead of one open+window per overlapping chunk file; everything
    downstream (scoring, agreement rasters) is unchanged either way.
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
        if spec.data_type not in ("GeoDataFrame", "RasterDataset"):
            raise NotImplementedError(
                f"Benchmark dispatch not implemented for {benchmark_key} "
                f"(data_type={spec.data_type}) - only GeoDataFrame and RasterDataset "
                "benchmarks are wired up."
            )
        if not spec.regions:
            raise ValueError(
                f"{benchmark_key}: coverage=national benchmarks require an explicit "
                "meta.regions block (no evaluation-cluster geometry exists here to derive "
                "a working extent from) - see BenchmarkSpec.regions' own docstring."
            )
        print(f"  National-coverage benchmark: {benchmark_key} ({spec.data_type})")

        # RasterDataset benchmarks (e.g. Denmark's continuous-depth GeoTIFFs) are read
        # windowed, per-chunk, directly from the catalog below - no upfront full-extent
        # load needed (unlike GeoDataFrame, which needs the WHOLE benchmark in memory
        # first to build a spatial index for per-chunk candidate queries).
        full_gdf = full_sindex = None
        if spec.data_type == "GeoDataFrame":
            full_gdf = v.load_benchmark_full(bench_catalog, spec)
            if full_gdf.empty:
                print(f"    {benchmark_key}: empty after filtering - skipping.")
                continue
            if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
                full_gdf = full_gdf.to_crs(4326)
            full_sindex = full_gdf.sindex

        acc: dict[str, dict[str, float]] = defaultdict(_new_acc)
        benchmark_wet_km2_total: dict[str, float] = defaultdict(float)
        benchmark_wet_outside_model_domain_km2: dict[str, float] = defaultdict(float)
        model_wet_km2: dict[str, float] = defaultdict(float)
        agreement_pieces_by_region: dict[str, list[tuple[np.ndarray, Affine]]] = defaultdict(list)
        n_processed = 0
        n_chunks_total = 0

        for region, region_bbox in spec.regions.items():
            # Each piece is one (depth, out_transform) to score - either one
            # per overlapping merge_chunk file (windowed to region_bbox,
            # since a chunk can hold multiple named regions, see comment
            # below) or, read_tiles_directly, a single mosaic read spanning
            # the whole region_bbox directly from production's per-tile
            # results (see this function's own docstring).
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
                        # Window the read to region_bbox ∩ this chunk's own bounds - a
                        # chunk is a fixed 5deg cell that can contain MULTIPLE named
                        # regions (e.g. Wales' severn_estuary and menai_strait both
                        # fall in the same chunk), so reading the whole chunk here
                        # would evaluate the SAME data for every region sharing it
                        # (confirmed 2026-09-30: produced byte-identical metrics for
                        # two genuinely different, ~150km-apart Welsh regions before
                        # this fix). region_bbox is a rectangle, not real geometry, so
                        # this still relies on read_country_mask below to exclude any
                        # non-target territory within the window.
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
                if spec.data_type == "GeoDataFrame":
                    candidate_idx = list(full_sindex.query(box(*bbox), predicate="intersects"))
                    chunk_bench_gdf = full_gdf.iloc[candidate_idx]
                    fraction = v.benchmark_fraction_from_vector(
                        chunk_bench_gdf, out_transform, "EPSG:4326", depth.shape, supersample=supersample,
                    )
                else:
                    fraction = v.read_benchmark_raster_fraction(bench_catalog, spec, bbox, out_transform, depth.shape)
                # Scoring uses `fraction` directly (confusion_counts_soft) -
                # never thresholded into a binary wet/dry call.
                # benchmark_wet_display is ONLY for the agreement-map
                # category raster below (one discrete colour per cell) - it
                # plays no part in any number in the metrics CSV.
                benchmark_wet_display = v.wet_mask_from_fraction(fraction, wet_fraction)

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
                    geogunit_ids=spec.geogunit_ids,
                )
                not_water = not_water & in_country
                domain_mask = model_domain & not_water

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
                # "cells" columns are soft/effective counts (float,
                # fraction-weighted) too - see validate_country()'s own
                # comment on this.
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

                # Display only - benchmark_wet_display's own threshold plays
                # no part in the scoring above.
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
    """Mosaic one region's clusters' native-resolution agreement category
    grids into one raster - at the model's own NATIVE resolution when the
    region is small enough for that to stay a tractable file size/pixel
    count (every partial-coverage unit, every nested sub-area like Wales'
    menai_strait/severn_estuary or Scotland's firth_of_forth), falling back
    to the coarser `plots.resolution_m` (with MAX resampling on the
    category codes - dry(0) < agree(1) < under(2) < over(3), so MAX
    implements "priority: over > under > agree > dry") only for a
    whole-country-scale region where native resolution would make the
    raster impractically large (Norway/Scotland's own mainland, hundreds of
    km across).

    2026-10: downsampling to `plots.resolution_m` UNCONDITIONALLY, as this
    function used to, visually overstates disagreement on a close-up panel
    - MAX-resampling the category codes means a single native cell inside a
    coarse output cell (at 200m vs. the model's own 30m, one output cell
    covers ~44 native cells) paints the WHOLE coarse cell with that single
    cell's own category, so one real, small, isolated patch of
    over-/under-prediction visually balloons to the size of the coarse
    cell - confirmed directly on Firth of Forth. This never affected the
    real CSI/HR/FAR numbers (computed earlier, from the native-resolution
    depth/fraction grids, never from this downsampled display raster) - it
    was purely a visualization artifact.

    Shared by both the extent comparison (`metric="agreement"`, the
    default) and the depth-band comparison (`metric="depth_agreement"`,
    validate_country_depth_bands) - the category codes/priority-reduction
    logic and output shape are identical, only the classification rule that
    produced the category grid differs, and the output filename prefix
    keeps the two from colliding.

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

    all_bounds = [rasterio.transform.array_bounds(a.shape[0], a.shape[1], t) for a, t in agreement_pieces]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    native_res_deg = abs(agreement_pieces[0][1].a)  # every piece shares the same native (model) grid resolution
    native_w = (maxx - minx) / native_res_deg
    native_h = (maxy - miny) / native_res_deg
    if max(native_w, native_h) <= _AGREEMENT_RASTER_MAX_DIM_PX:
        res_deg = native_res_deg  # small enough region - no coarsening, no priority-max artifact at all
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
    parser.add_argument(
        "--read-tiles-directly", action="store_true",
        help="score national-coverage benchmarks (validate_country_national_coverage) "
             "straight from production's own per-tile results/waterdepth_{rp}_{slr}.tif "
             "files instead of merge_chunk's merged chunks - for an ad hoc tile subset "
             "(e.g. a country-validation run over a handful of tiles) that was never run "
             "through merge_chunk, see that function's own docstring. No effect on "
             "validate_country/validate_country_depth_bands (unchanged, still chunk-based).",
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
    df_national = validate_country_national_coverage(
        country_iso, cfg, gfm_catalog, bench_catalog, read_tiles_directly=args.read_tiles_directly,
    )
    df = pd.concat([df_partial, df_national], ignore_index=True)
    if df.empty:
        print("No extent rows produced.")
    else:
        if args.run_tag:
            df["run_tag"] = args.run_tag
        out_path = out_dir / f"metrics_{country_iso}_{rp}_{slr}.csv"
        atomic_write(out_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
        print(f"\nMetrics written: {out_path} ({len(df)} row(s))")

        # One row per region already (no more threshold_m/domain dimensions to
        # dedup across - single config-driven threshold, one well-justified
        # domain per benchmark). Print the blended country-level row
        # (_BLENDED_REGION, "ALL") where one exists, else every region row.
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
