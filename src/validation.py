"""Coastal flood-extent validation against national hazard maps.

Benchmark loading/dispatch, benchmark-to-model-grid rasterization,
evaluation-domain construction (model domain, country mask, permanent-water
exclusion), and area-weighted contingency metrics. See
docs/methods_04b_MapsValidation.md for the current design - this module holds
the stateless computational building blocks; the per-benchmark iteration,
accumulation, and I/O live in validation/validate_country.py.

Per-country/per-dataset benchmark semantics (e.g. attribute_filter,
regions, geogunit_ids - national-coverage vector benchmarks only) live in
each catalog entry's own `meta:` block
(snakemake_workflow/config/data_catalog_validation.yml), read here via
load_benchmark_spec - adding a new country/dataset is a catalog entry, not a
new `if country == ...` branch in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from config_utils import retry_transient_io


# ── Benchmark loading/dispatch ──────────────────────────────────────────────

@dataclass
class BenchmarkSpec:
    """Parsed from one data_catalog_validation.yml entry's `meta:` block."""

    key: str
    data_type: str              # "GeoDataFrame" (vector benchmarks only)
    country_iso: str
    hazard_type: str            # coastal | fluvial | mixed - only "coastal" is comparable
    variable: str = "extent"    # extent - binary wet/dry comparison
    coverage: str = "national"  # national - the benchmark's own source assessed the ENTIRE
    # coastline, so "outside the polygon" unambiguously means "surveyed, found dry" and a
    # model-wet cell there is a genuine over-prediction - see
    # validate_country.validate_country_national_coverage (per-postprocessing-chunk
    # processing, bounded regardless of how far a national benchmark's footprint spans).
    attribute_filter: dict | None = None    # vector: {column: value} rows to KEEP
    regions: dict[str, list[float]] | None = None  # {name: [minx, miny, maxx, maxy]} -
    # named sub-regions (e.g. a whole-territory bbox plus sub-areas) covering the
    # WHOLE country, used to tag each evaluation unit for per-region CSV reporting instead of
    # one blended national row (validate_country.py's own _blend_regions folds them back into
    # one country-level row automatically). Every region a country cares about needs its own
    # explicit bbox here - there is no implicit "everything else" catch-all.
    geogunit_ids: list[int] | None = None  # explicit WRI geogunit_107 unit ID(s) to mask
    # to, bypassing the country_iso->ISO-lookup match in read_country_mask entirely -
    # for a benchmark that covers a SUB-national unit sharing its ISO code with
    # siblings the benchmark does NOT cover (e.g. Wales: geogunit_107 ID 3367, ISO
    # "GBR" like England/Scotland/N.Ireland's own separate IDs 3364/3366/3365 - using
    # country_iso="GBR" alone would wrongly include their coastline too). None (the
    # default) keeps the existing ISO-string lookup behaviour for every other country.
    comparison_return_periods: list[int] | None = None  # extra NATIVE return periods
    # (years, e.g. [100, 250] - must each have a real merged chunk, see
    # boundary_conditions.return_periods in config.yml) to also score this benchmark
    # against, beyond the single global `validation.return_period` every country is
    # scored at by default - e.g. for a benchmark whose own return period has no exact
    # native model scenario, both bracketing native RPs are reported instead of an
    # interpolated point (only the model's own real, simulated return periods). See validation/run_multi_rp_summary.py, the orchestration that actually
    # sweeps this list; `validate_country.py` itself still only ever reads the single
    # global `validation.return_period`. None (the default) means this benchmark is
    # only ever scored at that one global RP.


def fix_catalog_meta_encoding(catalog, yml_path: str | Path) -> None:
    """Re-read `yml_path` as UTF-8 and patch each already-loaded source's `meta` dict in-place.

    `hydromt.DataCatalog.from_yml` opens catalog yaml files with a bare
    `open(...)` (no explicit encoding), so it inherits the OS default
    (`locale.getpreferredencoding()`) - cp1252 on Windows. Any non-ASCII
    byte in a UTF-8-encoded catalog file (e.g. an accented character inside
    an `attribute_filter` value) is then silently mojibake-corrupted
    ('Ñ' -> 'Ã' + U+2018) before it ever reaches Python. There is no
    exception anywhere in that path - the only symptom is
    `filter_benchmark`'s exact-match `==` against the (correctly
    UTF-8-decoded) shapefile attribute never matching anything, so every
    benchmark row gets silently dropped. Re-parsing the same file explicitly as UTF-8
    and overwriting `meta` for every already-loaded source fixes this
    regardless of the host OS's default locale, without touching hydromt
    itself.

    Call this once, right after building any DataCatalog over
    data_catalog_validation.yml (or any other catalog whose `meta:` blocks
    may contain non-ASCII text) - both validate_country.py and
    run_validation.py's country-discovery scan need it, so it lives here
    rather than as a private helper duplicated in just one of them.
    """
    with open(yml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "meta" not in entry or key not in catalog.sources:
            continue
        catalog.get_source(key).meta.update(entry["meta"])


def load_benchmark_spec(catalog, key: str) -> BenchmarkSpec:
    """Read one catalog entry's `meta:` block into a BenchmarkSpec.

    `source.meta`/`source.data_type` are real hydromt DataAdapter attributes
    (confirmed against this repo's installed hydromt version, 2026-09) - not
    a re-parse of the YAML.
    """
    source = catalog.get_source(key)
    meta = dict(source.meta or {})
    return BenchmarkSpec(
        key=key,
        data_type=source.data_type,
        country_iso=str(meta.get("country_iso", "")),
        hazard_type=str(meta.get("hazard_type", "")),
        variable=str(meta.get("variable", "extent")),
        coverage=str(meta.get("coverage", "national")),
        attribute_filter=meta.get("attribute_filter"),
        regions=meta.get("regions"),
        geogunit_ids=meta.get("geogunit_ids"),
        comparison_return_periods=meta.get("comparison_return_periods"),
    )


def primary_region_name(regions: dict[str, list[float]] | None) -> str | None:
    """Which named region is "the main one" - the largest by bbox area
    (lon/lat degrees - an approximation, fine for picking one region out of
    a handful, not a real-area computation). Works uniformly for both
    `regions:` usage patterns: a disjoint partition (a mainland vs. much
    smaller islands/territories) and a nested whole-territory + sub-area
    set (Wales/Scotland/New Brunswick - the whole-territory bbox is by
    construction bigger than any sub-area bbox it contains). Returns None
    if `regions` is empty/None - caller should fall back to "the whole
    benchmark, no region split" in that case.

    Used to pick a single representative region for a benchmark-wide
    diagnostic that doesn't need (or want) one output per region - e.g. the
    tile-coverage plot - distinct from `_blend_regions`' own bbox-
    containment check in validate_country.py, which exists to avoid
    double-counting nested regions, not to pick a display region.
    """
    if not regions:
        return None

    def _area(bbox: list[float]) -> float:
        minx, miny, maxx, maxy = bbox
        return max(0.0, maxx - minx) * max(0.0, maxy - miny)

    return max(regions, key=lambda name: _area(regions[name]))


def load_benchmark_full(catalog, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Load and filter ALL of a vector benchmark's geometries (no bbox).

    Loads everything once per country/benchmark_key rather than per
    evaluation unit - there is no repeated bbox-scoped catalog read to make
    nodata-safe here.

    Applies `spec.attribute_filter` (keep only matching rows) - read from
    the catalog entry's own `meta`, not hardcoded per country.
    """
    gdf = catalog.get_geodataframe(spec.key)
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return filter_benchmark(gdf, spec)


def filter_benchmark(gdf: gpd.GeoDataFrame, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Apply `spec.attribute_filter` to an already-loaded benchmark GeoDataFrame."""
    if gdf.empty:
        return gdf
    if spec.attribute_filter:
        mask = np.ones(len(gdf), dtype=bool)
        for col, val in spec.attribute_filter.items():
            mask &= (gdf[col] == val).to_numpy()
        gdf = gdf.loc[mask]
    return gdf


# ── Bringing the benchmark onto the model grid ──────────────────────────────

def benchmark_fraction_from_vector(
    gdf: gpd.GeoDataFrame,
    dst_transform: Affine,
    dst_crs,
    dst_shape: tuple[int, int],
    supersample: int = 5,
) -> np.ndarray:
    """Coverage fraction (0-1) of `gdf`'s geometries within each dst cell.

    Rasterizes at `supersample`x finer resolution than the destination grid
    (reprojecting gdf to dst_crs first if needed), then block-averages each
    supersample x supersample sub-pixel block down to one destination cell.
    Unbiased, unlike `rasterize(all_touched=False)` (systematically loses
    thin coastal strips) or `all_touched=True` (systematically inflates
    them). Also gives an exact benchmark area for the
    coverage diagnostic, independent of any wet-fraction threshold, if the
    caller sums this directly (before thresholding).
    """
    if gdf.crs is not None and dst_crs is not None and gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    out_h, out_w = dst_shape
    if gdf.empty:
        return np.zeros((out_h, out_w), dtype="float32")

    fine_h, fine_w = out_h * supersample, out_w * supersample
    fine_transform = dst_transform * Affine.scale(1.0 / supersample)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.zeros((out_h, out_w), dtype="float32")
    fine_mask = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(fine_h, fine_w), transform=fine_transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    fraction = fine_mask.reshape(out_h, supersample, out_w, supersample).mean(axis=(1, 3))
    return fraction.astype("float32")


def wet_mask_from_fraction(fraction: np.ndarray, wet_fraction: float = 0.5) -> np.ndarray:
    """Binary benchmark-wet decision from a coverage fraction grid (the 0.5-rule threshold)."""
    return fraction >= wet_fraction


# ── Evaluation domain ────────────────────────────────────────────────────────

def model_domain_mask(depth: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """True where the model actually computed a value here (not outside its domain)."""
    return depth != nodata


def tri_domain_mask(
    perimeter_gdf: gpd.GeoDataFrame, dst_transform: Affine, dst_crs, dst_shape: tuple[int, int],
) -> np.ndarray:
    """Evaluation domain for a "perimeter" study area: the perimeter
    polygon(s) rasterized exactly as-is (point-in-polygon containment,
    `all_touched=False`) - no buffer at all: an official study-area
    perimeter IS the agency's own study design, so there is nothing a
    buffer would be compensating for.

    `perimeter_gdf` should already be filtered to the zones relevant to
    the caller before calling this - this function itself does no
    filtering, it only rasterizes whatever geometries it's given.
    """
    if dst_crs is not None and perimeter_gdf.crs is not None and perimeter_gdf.crs != dst_crs:
        perimeter_gdf = perimeter_gdf.to_crs(dst_crs)
    geoms = [g for g in perimeter_gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(dst_shape, dtype=bool)
    return rasterize(
        [(g, 1) for g in geoms], out_shape=dst_shape, transform=dst_transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)


def permanent_water_mask(land_use: np.ndarray, exclude_codes: tuple[int, ...]) -> np.ndarray:
    """True where a cell is permanent water (rivers/lakes/sea) per whichever
    categorical source `land_use` came from and its own `exclude_codes`
    (`validation.permanent_water_source`/`permanent_water_codes` in
    config.yml - DeltaDTM's own land/ocean/lake/river mask, codes {1,2,3},
    since 2026-09; was Copernicus Global Land Cover, codes {80,200} - the
    DeltaDTM switch is a real, measured coastal-strip-masking improvement,
    same DEM the model itself is built from, no second independently-
    registered dataset in the loop).

    These cells must be excluded from the evaluation domain entirely, not
    scored as agree/dry/wet - "is this pixel flooded" is not a meaningful
    question over water that is always wet regardless of any storm event
    (2026-09 addition, following real QGIS inspection of the benchmark vs.
    model rasters during design review).
    """
    return np.isin(land_use, np.asarray(exclude_codes))


def read_permanent_water_mask(
    gfm_catalog, permanent_water_source: str, permanent_water_codes,
    bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
) -> np.ndarray:
    """Permanent-water (rivers/lakes/sea) mask on an arbitrary target grid.

    `permanent_water_source` is a categorical raster (originally Copernicus
    land_use, ~100m; DeltaDTM's own native-resolution land/ocean/lake/river
    mask, ~25-31m, is the default since 2026-09 - a real, measured
    resolution improvement - either works, this function doesn't care which), coarser than the model
    grid either way, so this is always a nearest-neighbour reproject onto
    `out_transform`/`out_shape` (same convention as
    protection.load_geogunit_ids - never interpolate a categorical raster).
    Shared by validate_country.py (evaluation-domain construction) and
    plot_agreement_map.py (land background for the agreement map, in place of
    the separate `land_polygons` vector source - 2026-09, reuses the SAME
    raster the validation domain itself is already built from, instead of a
    second, independent coastline dataset).
    """
    try:
        lu_da = retry_transient_io(
            gfm_catalog.get_rasterdataset, permanent_water_source, bbox=bbox,
        ).squeeze(drop=True)
    except Exception:
        return np.zeros(out_shape, dtype=bool)
    if lu_da is None or lu_da.size == 0:
        return np.zeros(out_shape, dtype=bool)

    lu_arr = lu_da.values.astype("float64")
    lu_nodata = lu_da.raster.nodata
    dst = np.full(out_shape, -1.0, dtype="float64")
    reproject(
        source=lu_arr, destination=dst,
        src_transform=lu_da.raster.transform, src_crs=lu_da.raster.crs,
        dst_transform=out_transform, dst_crs="EPSG:4326",
        src_nodata=lu_nodata, dst_nodata=-1.0,
        resampling=Resampling.nearest,
    )
    return permanent_water_mask(dst, tuple(permanent_water_codes))


def load_iso_lookup(gfm_catalog, iso_lookup_source: str) -> dict[int, str]:
    """Geogunit-107 ID -> ISO-3 country code, the SAME table/convention as
    analysis/compute_exposure_analysis.py's/compute_flood_totals.py's own
    `iso_lookup` (FLOPROS's dataframe is indexed by geogunit ID, with an "ISO"
    column) - reused here rather than building a second, different mapping.
    """
    flopros = gfm_catalog.get_dataframe(iso_lookup_source)
    return {
        int(gid): str(row["ISO"]) for gid, row in flopros.iterrows()
        if pd.notna(row.get("ISO"))
    }


def read_country_mask(
    gfm_catalog, geogunit_source: str, iso_lookup: dict[int, str], country_iso: str,
    bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
    geogunit_ids: list[int] | None = None,
) -> np.ndarray:
    """True where a cell's WRI geogunit (nearest-neighbour reprojected onto this
    call's own grid - same convention as read_permanent_water_mask, never
    interpolate a categorical raster) resolves to `country_iso`.

    Needed anywhere a comparison's working extent is a rectangular bbox rather
    than the benchmark's own real geometry - a bbox can genuinely overlap a
    neighbouring country's territory (validate_country_national_coverage
    processes chunks found from a region bbox directly, with no
    benchmark-geometry step to naturally exclude them). Without
    this mask, GFM's own real flooding in a neighbouring country would be
    scored as a false positive against a benchmark that was never meant to
    cover that territory at all.

    `geogunit_ids`, when given (BenchmarkSpec.geogunit_ids), matches those
    geogunit_107 IDs directly instead of resolving `country_iso` through
    `iso_lookup` - for a benchmark covering a sub-national unit that shares
    its ISO code with sibling units it does NOT cover (Wales: ID 3367, ISO
    "GBR" like England/Scotland/N.Ireland's own separate IDs).
    """
    try:
        geo_da = retry_transient_io(
            gfm_catalog.get_rasterdataset, geogunit_source, bbox=bbox, variables=["Geogunits"],
        ).squeeze(drop=True)
    except Exception:
        return np.zeros(out_shape, dtype=bool)
    if geo_da is None or geo_da.size == 0:
        return np.zeros(out_shape, dtype=bool)

    geo_arr = geo_da.values.astype("float64")
    geo_nodata = geo_da.raster.nodata
    dst = np.full(out_shape, -1.0, dtype="float64")
    reproject(
        source=geo_arr, destination=dst,
        src_transform=geo_da.raster.transform, src_crs=geo_da.raster.crs,
        dst_transform=out_transform, dst_crs="EPSG:4326",
        src_nodata=geo_nodata, dst_nodata=-1.0,
        resampling=Resampling.nearest,
    )
    geo_ids = dst.astype("int32")
    target_ids = list(geogunit_ids) if geogunit_ids else [
        gid for gid, iso in iso_lookup.items() if iso == country_iso
    ]
    if not target_ids:
        return np.zeros(out_shape, dtype=bool)
    return np.isin(geo_ids, target_ids)


# ── Metrics ───────────────────────────────────────────────────────────────

def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float, float]:
    """Weighted (tp, fp, fn, tn) sums within `domain_mask`.

    `weight` is whatever consistent unit the caller wants weighted metrics
    in (km² from plotting.pixel_area_km2_grid is the production use today).
    Raw cell counts (weight=1 everywhere) work too, for a reproducibility
    figure alongside the area-weighted ones.

    tp = model wet AND benchmark wet (hit)
    fp = model wet AND NOT benchmark wet (over-prediction)
    fn = NOT model wet AND benchmark wet (under-prediction)
    tn = NOT model wet AND NOT benchmark wet
    """
    d = domain_mask
    tp = float(weight[d & model_wet & benchmark_wet].sum())
    fp = float(weight[d & model_wet & ~benchmark_wet].sum())
    fn = float(weight[d & ~model_wet & benchmark_wet].sum())
    tn = float(weight[d & ~model_wet & ~benchmark_wet].sum())
    return tp, fp, fn, tn


def confusion_counts_soft(
    model_wet: np.ndarray, fraction: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float, float]:
    """Weighted (tp, fp, fn, tn) sums within `domain_mask`, using the
    benchmark coverage FRACTION directly as continuous credit instead of
    thresholding it into a binary wet/dry call first
    (wet_mask_from_fraction) - a cell 20% covered by the benchmark
    contributes 0.2 of its weight to the wet side and 0.8 to the dry side,
    regardless of the model's own (still binary) wet/dry call at that cell.

    Replaces confusion_counts + wet_mask_from_fraction as the primary
    scoring path 2026-09-30: the optimal hard wet_fraction threshold varied
    by benchmark, and soft credit avoids having to choose one.

    tp = model wet, weighted by fraction (partial hit)
    fp = model wet, weighted by (1 - fraction) (partial over-prediction)
    fn = model dry, weighted by fraction (partial miss)
    tn = model dry, weighted by (1 - fraction)

    wet_mask_from_fraction (the 0.5-style threshold call this function
    itself never uses) survives for a narrow, still-real purpose, NOT as a
    scoring shortcut: the agreement-map DISPLAY category raster
    (_CAT_AGREE/_CAT_UNDER/_CAT_OVER need one discrete colour per cell; a
    soft credit can't render as a single colour). `confusion_counts` itself (the hard-threshold tp/fp/fn/tn
    function this replaces) is no longer called anywhere in
    validate_country.py at all - it remains a real, independently useful
    function (generic weighted confusion counts over any two boolean
    masks), still exercised by its own unit tests
    (tests/flood_extent_validation/test_metrics.py).
    """
    d = domain_mask
    f = np.clip(fraction, 0.0, 1.0)
    tp = float((weight * f)[d & model_wet].sum())
    fp = float((weight * (1.0 - f))[d & model_wet].sum())
    fn = float((weight * f)[d & ~model_wet].sum())
    tn = float((weight * (1.0 - f))[d & ~model_wet].sum())
    return tp, fp, fn, tn


def confusion_counts_tolerant(
    model_wet: np.ndarray,
    fraction: np.ndarray,
    domain_mask: np.ndarray,
    weight: np.ndarray,
    tolerance_cells: int,
) -> dict[str, float]:
    """Diagnostic companion to confusion_counts_soft - NOT a replacement, and
    never part of the primary scoring path (HR/FAR/CSI/EB/bias everywhere
    else stay confusion_counts_soft's own continuous-credit numbers,
    untouched). Answers one narrow question: how much of the STRICT (hard,
    zero-tolerance) disagreement between model and benchmark sits within
    `tolerance_cells` pixels of a cell where the two genuinely agree - i.e.
    looks like the same flood boundary drawn with a small spatial offset,
    rather than a real disagreement about whether an area floods at all.

    Written for the permanent-water-mask-vs-benchmark-coastline
    misalignment caveat (methods_04b_MapsValidation.md), but not specific to
    that caveat, to any one country, or to the coastline specifically - it
    is a generic small-scale-registration check applied uniformly over the
    whole domain. In practice it mostly fires along a coastline because
    that is where two independently-drawn wet/dry boundaries disagree by a
    pixel or two; the mechanism itself has no notion of "coastline."

    Guards against the obvious failure mode - silently inflating CSI:
      - Operates on a HARD benchmark-wet basis (`fraction > 0`, "any real
        benchmark-reported wet area in this cell at all" - an existence
        test, not a reintroduced 0.5-style majority threshold) and a hard
        tp/fp/fn/tn partition, entirely separate from confusion_counts_soft's
        own continuous credit. CSI itself is never touched by this function.
      - A disagreement cell is forgiven only if a REAL opposite-type cell
        exists nearby (binary dilation of the actual wet masks, not a
        blanket buffer drawn around the benchmark's extent) - an isolated
        model-wet patch with no nearby benchmark-wet cell at all gets zero
        credit regardless of `tolerance_cells`.
      - Symmetric: forgives a false positive against nearby benchmark-wet
        cells AND a false negative against nearby model-wet cells equally -
        a one-directional version would just mechanically inflate precision
        or recall, not test registration agreement.
      - Can only ever move weight from fp/fn into fp_forgiven/fn_forgiven;
        tp and tn are untouched, so CSI_tol >= the hard CSI this function's
        own tp/fp/fn/tn would give at tolerance_cells=0, always, by
        construction - never by estimation noise.
      - `tolerance_cells` is meant to be set from a real, independently
        justified registration-uncertainty distance (this pipeline's own
        native 30m grid resolution and the permanent-water mask's own
        ~25-31m native resolution are both close to one cell), not tuned
        upward until the number looks good. Callers are expected to also
        report fp_forgiven/fn_forgiven (or the share of strict disagreement
        they represent) alongside CSI_tol, so a reader can see how much
        work the tolerance is doing rather than just the headline number.

    Returns a dict (not a 4-tuple like confusion_counts_soft - there are two
    extra quantities here): tp, fp, fn, tn (feed straight into
    metrics_from_counts for CSI_tol/HR_tol/FAR_tol), plus fp_forgiven/
    fn_forgiven (the weight moved out of fp/fn by the tolerance check).
    """
    d = domain_mask
    bench_wet = fraction > 1e-9
    fp = d & model_wet & ~bench_wet
    fn = d & ~model_wet & bench_wet
    tp = d & model_wet & bench_wet
    tn = d & ~model_wet & ~bench_wet

    if tolerance_cells > 0:
        struct = np.ones((3, 3), dtype=bool)
        bench_wet_near = ndimage.binary_dilation(bench_wet, structure=struct, iterations=tolerance_cells)
        model_wet_near = ndimage.binary_dilation(model_wet, structure=struct, iterations=tolerance_cells)
    else:
        bench_wet_near = bench_wet
        model_wet_near = model_wet

    fp_forgiven = fp & bench_wet_near
    fn_forgiven = fn & model_wet_near
    fp_tol = fp & ~bench_wet_near
    fn_tol = fn & ~model_wet_near

    return {
        "tp": float(weight[tp].sum()),
        "fp": float(weight[fp_tol].sum()),
        "fn": float(weight[fn_tol].sum()),
        "tn": float(weight[tn].sum()),
        "fp_forgiven": float(weight[fp_forgiven].sum()),
        "fn_forgiven": float(weight[fn_forgiven].sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float = 0.0) -> dict[str, float]:
    """HR, FAR, CSI, EB, EB_ratio, bias from (weighted) TP/FP/FN(/TN) counts.

    Zero denominators return NaN, never a silent 0 - "no benchmark wet area
    in this domain" must not read as "CSI = 0". `EB == 0.5`
    exactly whenever `FP == FN` (both nonzero) - the requested
    over/under-prediction form; `EB_ratio = FP/FN` is the Wing et al. (2017)
    form, more legible at extremes (kept alongside, not instead of, EB).
    """
    return {
        "HR": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "EB": _safe_div(fp, fp + fn),
        "EB_ratio": _safe_div(fp, fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
