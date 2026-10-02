"""Coastal flood-extent validation against national hazard maps.

Benchmark loading/dispatch, benchmark-to-model-grid rasterization,
evaluation-domain construction (TRI-zone/connectivity domains for
partial-coverage benchmarks, permanent-water exclusion for every benchmark),
and area-weighted contingency metrics. See docs/methods_04b_MapsValidation.md
for the current design - this module holds the stateless computational
building blocks; the per-benchmark iteration, accumulation, and I/O live in
validation/validate_country.py.

Per-country/per-dataset benchmark semantics (currently just
attribute_filter, exclude_bbox - only vector/GeoDataFrame benchmarks are
implemented so far) live in each catalog entry's own `meta:` block
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
    """Parsed from one data_catalog_validation.yml entry's `meta:` block.

    Both GeoDataFrame and RasterDataset benchmarks are supported (RasterDataset
    added 2026-09 for Denmark's continuous-depth GeoTIFFs - see
    read_benchmark_raster_fraction and validate_country_national_coverage's
    data_type dispatch).
    """

    key: str
    data_type: str              # "GeoDataFrame" | "RasterDataset"
    country_iso: str
    hazard_type: str            # coastal | fluvial | mixed - only "coastal" is comparable
    variable: str = "extent"    # extent | depth - dispatches validate_country.py to the
    # binary wet/dry comparison (extent) or the continuous-depth-vs-band comparison
    # (depth, France's n_iso_ht_* layers - see validate_country.validate_country_depth_bands)
    wet_values: list | None = None       # RasterDataset only: categorical pixel values counted
    # as "wet" (e.g. a discrete hazard-class raster) - mutually exclusive with
    # depth_threshold_m; exactly one must be set for a RasterDataset benchmark, see
    # read_benchmark_raster_fraction.
    depth_threshold_m: float | None = None  # RasterDataset only: pixel value > this counts as
    # "wet" (a continuous depth raster, e.g. Denmark's "Oversvømmelsesfare" GeoTIFFs) -
    # mutually exclusive with wet_values.
    ht_min_col: str = "ht_min"  # variable="depth" only: column holding each band's lower bound (m)
    ht_max_col: str = "ht_max"  # variable="depth" only: column holding each band's upper bound (m) -
    # open-ended top bands use inconsistent, region/zone-specific sentinel values (confirmed
    # 2026-09 against France's real data: 9999/999/99, or NaN entirely) rather than a real depth -
    # see rasterize_depth_bands' open_ended_min_m param, not this column's raw values, for how
    # those get resolved to "no upper bound".
    coverage: str = "partial"   # partial | national - dispatches validate_country.py to the
    # TRI-zone/connectivity-domain comparison (partial, the default - Spain/France's
    # benchmarks only survey isolated designated zones, so a per-benchmark study-area domain
    # is used instead of "everywhere the model computed something" - see meta.study_area/
    # tri_domain_mask/connectivity_domain_mask) or the per-postprocessing-chunk comparison
    # (national - the benchmark's own source assessed the ENTIRE coastline, so "outside the
    # polygon" unambiguously means "surveyed, found dry" and a model-wet cell there is a
    # genuine over-prediction - see validate_country.validate_country_national_coverage,
    # added 2026-09 after Norway's long, only-lightly-fragmented coastline collapsed into one
    # giant buffer/merge cluster spanning almost the whole country and crashed the then-
    # cluster-based path's per-cluster bbox-sized array allocation - chunk-sized processing
    # is bounded regardless of how far a national benchmark's footprint spans).
    attribute_filter: dict | None = None    # vector: {column: value} rows to KEEP
    exclude_bbox: list[float] | None = None  # vector: [minx, miny, maxx, maxy] to DROP
    regions: dict[str, list[float]] | None = None  # {name: [minx, miny, maxx, maxy]} -
    # named sub-regions (mainland, each overseas territory/archipelago) covering the
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
    study_area: dict | None = None  # {source, geometry_type, attribute_filter?} - the
    # benchmark agency's own official study design, used as the partial-coverage
    # evaluation domain (2026-10, replacing the earlier flat eval_domain.buffer_km):
    # geometry_type "perimeter" (France's TRI zones) -> that zone's own polygon used
    # AS-IS (point-in-polygon, no buffer) as the domain, one evaluation unit per zone
    # still present in this benchmark's own extent data (tri_domain_mask). geometry_type
    # "segments" (Spain's ARPSI lines) -> a seeded-connectivity domain instead, NOT a
    # buffer around the line (connectivity_domain_mask) - a line is a 1D seed, not an
    # area, so "point near the line" was never the right domain-construction question
    # for this geometry type the way it is for a perimeter polygon. `source` is a
    # separate catalog key (category: study_area_source) holding the raw geometry;
    # `attribute_filter` (optional) narrows it the same way BenchmarkSpec's own
    # attribute_filter does (e.g. Spain's source has both marine and fluvial segments,
    # filtered here to ORIGEN_INU=="Marina"). None (the default, no benchmark currently
    # has this unset) means this benchmark has no study-area domain defined - see
    # validate_country.py for what that does.
    comparison_return_periods: list[int] | None = None  # extra NATIVE return periods
    # (years, e.g. [100, 250] - must each have a real merged chunk, see
    # boundary_conditions.return_periods in config.yml) to also score this benchmark
    # against, beyond the single global `validation.return_period` every country is
    # scored at by default - e.g. Norway, whose own real-world benchmark return period
    # (~200yr class) has no exact native model scenario, so both of the model's own
    # bracketing native RPs (100/250) are reported instead of estimating the 200yr
    # point itself (2026-10 - an earlier log-linear-interpolated RP200 point was
    # removed by user decision: report the model's own real, simulated return periods
    # only). See validation/run_multi_rp_summary.py, the orchestration that actually
    # sweeps this list; `validate_country.py` itself still only ever reads the single
    # global `validation.return_period`. None (the default) means this benchmark is
    # only ever scored at that one global RP.


def fix_catalog_meta_encoding(catalog, yml_path: str | Path) -> None:
    """Re-read `yml_path` as UTF-8 and patch each already-loaded source's `meta` dict in-place.

    `hydromt.DataCatalog.from_yml` opens catalog yaml files with a bare
    `open(...)` (no explicit encoding), so it inherits the OS default
    (`locale.getpreferredencoding()`) - cp1252 on Windows. Any non-ASCII
    byte in a UTF-8-encoded catalog file (e.g. the accented "AÑOS" inside
    Spain's `attribute_filter` value in data_catalog_validation.yml) is then
    silently mojibake-corrupted ('Ñ' -> 'Ã' + U+2018) before it ever reaches
    Python. There is no exception anywhere in that path - the only symptom
    is `filter_benchmark`'s exact-match `==` against the (correctly
    UTF-8-decoded) shapefile attribute never matching anything, so every
    benchmark row gets silently dropped (confirmed 2026-09: 397 -> 0 rows
    for spain_zi_marina_q100). Re-parsing the same file explicitly as UTF-8
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
        ht_min_col=str(meta.get("ht_min_col", "ht_min")),
        ht_max_col=str(meta.get("ht_max_col", "ht_max")),
        coverage=str(meta.get("coverage", "partial")),
        attribute_filter=meta.get("attribute_filter"),
        exclude_bbox=meta.get("exclude_bbox"),
        regions=meta.get("regions"),
        wet_values=meta.get("wet_values"),
        depth_threshold_m=meta.get("depth_threshold_m"),
        geogunit_ids=meta.get("geogunit_ids"),
        study_area=meta.get("study_area"),
        comparison_return_periods=meta.get("comparison_return_periods"),
    )


def _exclude_bbox_mask(gdf: gpd.GeoDataFrame, exclude_bbox: list[float]) -> np.ndarray:
    """True for rows whose centroid falls OUTSIDE `exclude_bbox` (EPSG:4326) - i.e. rows to KEEP."""
    ex_minx, ex_miny, ex_maxx, ex_maxy = exclude_bbox
    gdf_4326 = gdf if gdf.crs is not None and gdf.crs.to_epsg() == 4326 else gdf.to_crs(4326)
    centroids = gdf_4326.geometry.centroid
    inside = (
        (centroids.x >= ex_minx) & (centroids.x <= ex_maxx)
        & (centroids.y >= ex_miny) & (centroids.y <= ex_maxy)
    )
    return ~inside.to_numpy()


def region_for_point(x: float, y: float, regions: dict[str, list[float]] | None) -> str | None:
    """Which named region (EPSG:4326 bbox) `(x, y)` falls in, or None if none match.

    First match wins if bboxes ever overlap (they shouldn't - regions are meant to
    partition the country, e.g. Spain's mainland vs canary_islands). Returning None
    (rather than a guessed default) lets the caller decide how to label a genuine gap
    in the partition - see validate_country.py's "unclassified" fallback.
    """
    if not regions:
        return None
    for name, bbox in regions.items():
        minx, miny, maxx, maxy = bbox
        if minx <= x <= maxx and miny <= y <= maxy:
            return name
    return None


def primary_region_name(regions: dict[str, list[float]] | None) -> str | None:
    """Which named region is "the main one" - the largest by bbox area
    (lon/lat degrees - an approximation, fine for picking one region out of
    a handful, not a real-area computation). Works uniformly for both
    `regions:` usage patterns this catalog has: a disjoint partition (Spain
    mainland vs. the much smaller canary_islands; France metropole vs. its
    5 small overseas territories) and a nested whole-territory + sub-area
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

    2026-09 redesign: clustering (`build_evaluation_clusters`) needs the
    WHOLE benchmark up front to know what overlaps what, so this loads
    everything once per country/benchmark_key rather than per evaluation
    unit - unlike the earlier per-block design, there is no repeated
    bbox-scoped catalog read to make nodata-safe here.

    Applies `spec.attribute_filter` (keep only matching rows) and
    `spec.exclude_bbox` (drop rows whose centroid falls inside it, e.g.
    Spain's Canary Islands) - both read from the catalog entry's own `meta`,
    not hardcoded per country.
    """
    gdf = catalog.get_geodataframe(spec.key)
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return filter_benchmark(gdf, spec)


def filter_benchmark(gdf: gpd.GeoDataFrame, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Apply `spec.attribute_filter`/`spec.exclude_bbox` to an already-loaded benchmark GeoDataFrame."""
    if gdf.empty:
        return gdf
    if spec.attribute_filter:
        mask = np.ones(len(gdf), dtype=bool)
        for col, val in spec.attribute_filter.items():
            mask &= (gdf[col] == val).to_numpy()
        gdf = gdf.loc[mask]
    if spec.exclude_bbox and not gdf.empty:
        gdf = gdf.loc[_exclude_bbox_mask(gdf, spec.exclude_bbox)]
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


def rasterize_depth_bands(
    gdf: gpd.GeoDataFrame,
    ht_min_col: str,
    ht_max_col: str,
    dst_transform: Affine,
    dst_crs,
    dst_shape: tuple[int, int],
    open_ended_min_m: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a depth-band benchmark's ht_min/ht_max onto an arbitrary grid.

    Unlike `benchmark_fraction_from_vector`'s supersampled fractional coverage
    (needed for a binary wet/dry decision right at a polygon edge), depth-band
    membership is inherently categorical - a cell belongs to exactly one band
    polygon, since bands tile the benchmark's own extent polygons exactly
    (confirmed 2026-09 against real France data: the union of a TRI zone's
    depth-band sub-polygons matches that zone's own extent-polygon area to
    within floating-point noise) - so this rasterizes once at the
    destination's own resolution, no supersampling needed.

    `ht_max_col` values are NOT used literally where they exceed
    `open_ended_min_m` (default 50 m - comfortably above any real closed band,
    comfortably below every open-bucket sentinel found in real French data:
    99, 999, 9999) or are NaN - both are resolved to `np.inf` ("no upper
    bound") rather than rasterizing an absurd literal depth. This is a
    magnitude-threshold rule, not an enumeration of specific sentinel values,
    so it doesn't need updating if a not-yet-seen country/dataset uses a
    different sentinel - confirmed (2026-09) that within any single French TRI
    zone the sentinel is internally consistent even though it varies by zone
    (9999 most zones, 999 for Mayotte/Réunion, 99 for Martinique, NaN for
    several Brittany/Normandy zones with no max recorded at all) - a fixed
    magnitude threshold handles all of these uniformly without needing to
    enumerate them.

    Returns (ht_min_grid, ht_max_grid), both NaN outside every band polygon
    (no benchmark information there - not the same as "band [0, 0]").
    """
    if gdf.crs is not None and dst_crs is not None and gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    out_h, out_w = dst_shape
    ht_min_grid = np.full((out_h, out_w), np.nan, dtype="float32")
    ht_max_grid = np.full((out_h, out_w), np.nan, dtype="float32")
    if gdf.empty:
        return ht_min_grid, ht_max_grid

    ht_min_vals = gdf[ht_min_col].to_numpy(dtype="float64")
    ht_max_raw = gdf[ht_max_col].to_numpy(dtype="float64")
    ht_max_vals = np.where(np.isnan(ht_max_raw) | (ht_max_raw > open_ended_min_m), np.inf, ht_max_raw)

    geoms = list(gdf.geometry)
    valid = [
        (i, geom) for i, geom in enumerate(geoms)
        if geom is not None and not geom.is_empty and not np.isnan(ht_min_vals[i])
    ]
    if not valid:
        return ht_min_grid, ht_max_grid

    ht_min_grid = rasterize(
        [(geom, float(ht_min_vals[i])) for i, geom in valid],
        out_shape=(out_h, out_w), transform=dst_transform,
        fill=np.nan, dtype="float32", all_touched=False,
    )
    ht_max_grid = rasterize(
        [(geom, float(ht_max_vals[i])) for i, geom in valid],
        out_shape=(out_h, out_w), transform=dst_transform,
        fill=np.nan, dtype="float32", all_touched=False,
    )
    return ht_min_grid, ht_max_grid


# ── Multi-return-period comparison ──────────────────────────────────────────

# ── Evaluation domain ────────────────────────────────────────────────────────

def model_domain_mask(depth: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """True where the model actually computed a value here (not outside its domain)."""
    return depth != nodata


def build_evaluation_clusters(
    gdf: gpd.GeoDataFrame,
    buffer_km: float,
    simplify_tolerance_m: float = 100.0,
    quad_segs: int = 2,
) -> gpd.GeoDataFrame:
    """Group `gdf`'s geometries into evaluation clusters: buffer each by
    `buffer_km`, merge any that overlap, leave isolated ones as their own
    single-polygon cluster. Each cluster's own (already-buffered) geometry
    IS its evaluation domain - this replaces the earlier per-block
    `scipy.ndimage.distance_transform_edt` buffer approach entirely
    (2026-09 redesign): the benchmark's own geometry defines the processing
    units instead of an arbitrary uniform grid, so there's no wasted work on
    empty area, no halo-padding bookkeeping, and the buffer is exact instead
    of a block-local raster approximation.

    Reprojects to an estimated local UTM zone (via geopandas'
    `estimate_utm_crs()`) before buffering - buffering directly in degrees
    would make `buffer_km` meaningless, and hardcoding a per-country metric
    CRS doesn't scale to new benchmark countries.

    Simplifies with `preserve_topology=False` BEFORE buffering - NOT
    geopandas' own default topology-preserving simplify, which is itself
    pathologically slow on the same degenerate high-vertex-count records
    that make raw buffering slow. Confirmed empirically on the real Spain
    benchmark (2026-09): one of 397 records is a 596,191-vertex MultIPolygon
    (9,850 sub-parts, 27,189 holes - almost certainly a raster-to-vector
    artifact from a dense urban reach) that never finished buffering at
    default settings (>160s) or under topology-preserving simplify, but
    `simplify(100m, preserve_topology=False)` + `buffer(quad_segs=2)` +
    `union_all()` processed the WHOLE 397-polygon benchmark in <1s and
    matched the shapefile's own AREA_KM2 field almost exactly. A single
    other pathological record could in principle still blow up runtime on a
    future, larger benchmark dataset (e.g. France's 74,547-feature TRI
    layers) - there is no automatic vertex-count guard yet, so a new
    country's first real run is where that would surface, not something
    caught ahead of time.

    Returns a GeoDataFrame (EPSG:4326), one row per cluster, with a
    `cluster_id` column and geometry = that cluster's buffered domain.

    2026-10: no longer the SCORING domain for any benchmark with a
    meta.study_area block (every partial-coverage benchmark today - see
    tri_domain_mask/connectivity_domain_mask below) - kept only as a
    chunk-windowing optimization (which merged chunks to mosaic/read
    together for a group of nearby evaluation units), since reading one
    chunk per TRI zone/ARPSI segment independently would re-read the same
    chunk file many times over for geographically close units. The actual
    domain scored inside a window always comes from tri_domain_mask/
    connectivity_domain_mask, never from this function's own buffered
    geometry.
    """
    if gdf.empty:
        return gpd.GeoDataFrame({"cluster_id": []}, geometry=[], crs=4326)

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    simplified = gdf_utm.geometry.simplify(simplify_tolerance_m, preserve_topology=False)
    # geopandas' GeoSeries.buffer(distance, resolution=...) renames its own
    # `resolution` kwarg to shapely's `quad_segs` internally (geopandas
    # 0.14.1 _vectorized.buffer: `shapely.buffer(data, distance,
    # quad_segs=resolution, **kwargs)`) - passing `quad_segs=` here directly
    # collides with that and raises "got multiple values for keyword
    # argument 'quad_segs'". Pass it through geopandas' own kwarg name.
    buffered = simplified.buffer(buffer_km * 1000.0, resolution=quad_segs)
    merged = buffered.union_all() if hasattr(buffered, "union_all") else buffered.unary_union

    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    clusters = gpd.GeoDataFrame(
        {"cluster_id": range(len(parts))}, geometry=parts, crs=utm_crs,
    )
    return clusters.to_crs(4326)


def units_with_extent_coverage(
    perimeter_gdf: gpd.GeoDataFrame, extent_benchmark_gdf: gpd.GeoDataFrame, id_col: str = "id_tri",
) -> gpd.GeoDataFrame:
    """Filters a perimeter study-area source (e.g. `france_tri_perimeters`,
    all 131 TRI zones including fluvial-only ones) down to only the rows
    whose `id_col` value actually appears in the real coastal extent
    benchmark (e.g. `france_inondable_02moy`'s own `id_tri` column - both
    files share this field, confirmed 2026-10 via pyogrio.read_info against
    the real shapefiles, no join ambiguity). This is what drops the
    fluvial-only TRI zones without needing a generic proximity-based
    run-filter - a zone with a perimeter but zero real coastal flood rows in
    this specific benchmark was never meant to be evaluated against it.

    Returns one row per surviving zone - this IS the evaluation-unit
    granularity for a "perimeter" study area (one TRI zone = one unit),
    unlike the "segments" case (connectivity_domain_mask), which does not
    have a natural one-row-per-unit structure the same way.
    """
    if perimeter_gdf.empty or extent_benchmark_gdf.empty or id_col not in perimeter_gdf.columns:
        return perimeter_gdf.iloc[0:0]
    real_ids = set(extent_benchmark_gdf[id_col].dropna().unique()) if id_col in extent_benchmark_gdf.columns else set()
    if not real_ids:
        return perimeter_gdf.iloc[0:0]
    return perimeter_gdf[perimeter_gdf[id_col].isin(real_ids)]


def tri_domain_mask(
    perimeter_gdf: gpd.GeoDataFrame, dst_transform: Affine, dst_crs, dst_shape: tuple[int, int],
) -> np.ndarray:
    """Evaluation domain for a "perimeter" study area: the perimeter
    polygon(s) rasterized exactly as-is (point-in-polygon containment,
    `all_touched=False`) - no buffer at all. Replaces the earlier flat
    `eval_domain.buffer_km` margin for France: the TRI zone's own official
    boundary IS the agency's own study design, so there is nothing a buffer
    would be compensating for the way there was for an arbitrary distance
    margin around a benchmark polygon that was never meant to define the
    survey's own extent.

    `perimeter_gdf` should already be filtered to the zones relevant to
    this specific benchmark (units_with_extent_coverage) before calling
    this - this function itself does no filtering, it only rasterizes
    whatever geometries it's given.
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


def connectivity_domain_mask(
    model_wet: np.ndarray,
    benchmark_wet: np.ndarray,
    seed_gdf: gpd.GeoDataFrame,
    dst_transform: Affine,
    dst_crs,
    not_water: np.ndarray,
) -> np.ndarray:
    """Evaluation domain for a "segments" study area (Spain's ARPSI coastal
    seed lines): NOT a buffer around the lines - a line is a 1D seed marking
    "flooding starting here is real," not an area with its own extent the
    way a TRI perimeter polygon is. Instead, two independent 8-connected
    connected-component analyses (scipy.ndimage.label), run on the FULL
    (not yet water-masked) model-wet and benchmark-wet masks separately,
    each keeping only the components that touch a seed cell - domain =
    union of both connected results, so a real miss (one map connects to a
    seed, the other doesn't reach there at all) stays detectable, unlike a
    domain that was tautologically restricted to model_wet alone.

    Seed cells: `seed_gdf`'s geometries rasterized with `all_touched=True`
    (an exact rasterized touch, not a distance/snap-tolerance buffer - a
    cell the line geometry doesn't actually pass through is never a seed).

    `not_water` (rivers/lakes/ocean mask, already inverted - True means
    NOT water) is applied only to the FINAL connected result, never before
    labeling - a river/lake is a real physical flood conduit the
    connectivity graph must be allowed to pass through; masking it out
    first would sever genuine connectivity paths to real flooding on its
    far side. A water-body cell itself still never counts as in-domain
    (can't be "correctly flooded" over permanently-wet water), only the
    connectivity PATH through it is preserved.
    """
    if seed_gdf.crs is not None and dst_crs is not None and seed_gdf.crs != dst_crs:
        seed_gdf = seed_gdf.to_crs(dst_crs)
    geoms = [g for g in seed_gdf.geometry if g is not None and not g.is_empty]
    shape = model_wet.shape
    if not geoms:
        return np.zeros(shape, dtype=bool)
    seed_mask = rasterize(
        [(g, 1) for g in geoms], out_shape=shape, transform=dst_transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)

    structure = ndimage.generate_binary_structure(2, 2)  # 8-connected

    model_labels, _ = ndimage.label(model_wet, structure=structure)
    model_seed_ids = set(np.unique(model_labels[seed_mask & model_wet])) - {0}
    model_connected = np.isin(model_labels, list(model_seed_ids)) if model_seed_ids else np.zeros(shape, dtype=bool)

    bench_labels, _ = ndimage.label(benchmark_wet, structure=structure)
    bench_seed_ids = set(np.unique(bench_labels[seed_mask & benchmark_wet])) - {0}
    bench_connected = np.isin(bench_labels, list(bench_seed_ids)) if bench_seed_ids else np.zeros(shape, dtype=bool)

    return (model_connected | bench_connected) & not_water


def connectivity_components(domain_mask: np.ndarray) -> np.ndarray:
    """Re-labels a `connectivity_domain_mask` result into physically
    contiguous components (8-connected) - this is the evaluation-UNIT
    granularity for a "segments" study area's own per-unit CSI reporting
    (e.g. the CSI-per-ARPSI dot map), since the seed table itself has no
    natural one-row-per-unit structure the way a "perimeter" study area's
    TRI zones already do (one call to tri_domain_mask each). Each
    contiguous patch of the final (already seed-connected, already
    water-masked) domain becomes one dot - not one dot per original ARPSI
    table row, which would need a nontrivial nearest-seed attribution step
    to split a component fed by multiple nearby segments.

    Returns an int32 label array, 0 = outside the domain, 1..N = component
    ID - pass each `labels == i` slice to confusion_counts_soft (restricted
    further by this same domain_mask, which callers already have) for that
    component's own tp/fp/fn/tn.
    """
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connected, same as the labeling above
    labels, _ = ndimage.label(domain_mask, structure=structure)
    return labels.astype("int32")


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
    mask, ~25-31m in Norway, is the default since 2026-09 - a real, measured
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


def read_benchmark_raster_fraction(
    bench_catalog, spec: BenchmarkSpec,
    bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
) -> np.ndarray:
    """RasterDataset counterpart to benchmark_fraction_from_vector - classifies
    a benchmark raster (e.g. Denmark's continuous-depth "Oversvømmelsesfare"
    GeoTIFFs, ~5m native resolution) into a wet/dry mask and reprojects it onto
    an arbitrary target grid, returning a continuous 0-1 coverage FRACTION per
    destination cell - the same semantics `benchmark_fraction_from_vector`
    gives every vector benchmark, so `confusion_counts_soft` credits a Denmark
    cell exactly like a France/Spain/Norway one (partial coverage = partial
    credit), not as a degenerate always-0-or-1 input.

    Classifies at the benchmark's OWN native resolution first (via
    spec.wet_values for a categorical raster, or spec.depth_threshold_m for a
    continuous depth one - exactly one must be set), THEN reprojects the
    resulting binary mask with Resampling.average - deliberately NOT the
    reverse order (reproject raw values with nearest-neighbour, then
    threshold). The model grid is typically much coarser than a 5m source
    (Denmark: ~30m model cells, ~36 native sub-pixels each) - thresholding
    after a nearest-neighbour reproject would sample only ONE of those 36
    sub-pixels per destination cell, silently missing real benchmark flooding
    elsewhere in that cell. Resampling.average on the pre-classified mask
    instead computes the real proportion of covered native pixels that were
    wet (e.g. 8/36 -> 0.22), the direct raster-grid equivalent of
    `benchmark_supersample`'s sub-cell coverage estimate for a vector
    benchmark (2026-10 - replaced an earlier Resampling.max choice, which
    collapsed this to a binary "any wet sub-pixel -> whole cell counted
    wet" call; harmless for the old hard-threshold `wet_mask_from_fraction`
    comparison this function predates, but silently kept Denmark on
    degenerate 0/1 scoring after confusion_counts_soft became the one
    production scoring path for every other benchmark).

    Nodata/dry pixels are folded into "not wet" at the native-resolution
    classification step, so (unlike read_permanent_water_mask) no separate
    src_nodata handling is needed at reproject time - the binary mask has
    nothing left to distinguish.
    """
    try:
        da = retry_transient_io(bench_catalog.get_rasterdataset, spec.key, bbox=bbox).squeeze(drop=True)
    except Exception:
        return np.zeros(out_shape, dtype="float64")
    if da is None or da.size == 0:
        return np.zeros(out_shape, dtype="float64")

    arr = da.values
    nodata = da.raster.nodata
    valid = np.isfinite(arr) if nodata is None else (np.isfinite(arr) & (arr != nodata))

    if spec.wet_values is not None:
        wet = valid & np.isin(arr, spec.wet_values)
    elif spec.depth_threshold_m is not None:
        wet = valid & (arr > spec.depth_threshold_m)
    else:
        raise ValueError(
            f"{spec.key}: RasterDataset benchmark needs meta.wet_values or "
            "meta.depth_threshold_m (exactly one) - see BenchmarkSpec's own docstring."
        )

    dst = np.zeros(out_shape, dtype="float64")
    reproject(
        source=wet.astype("float32"), destination=dst,
        src_transform=da.raster.transform, src_crs=da.raster.crs,
        dst_transform=out_transform, dst_crs="EPSG:4326",
        resampling=Resampling.average,
    )
    return dst


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
    neighbouring country's territory (confirmed 2026-09, twice: Spain's
    `mainland` bbox overlaps Portugal/France/Morocco, via the flood_totals
    computation's own earlier bbox-leakage bug; Norway's
    `mainland` bbox, generous enough to cover the country's full latitude
    range, overlaps real Swedish and Danish territory too -
    validate_country_national_coverage processes chunks found from that bbox
    directly, with no benchmark-geometry clustering step to naturally exclude
    them the way the partial-coverage path's cluster polygons do). Without
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
    scoring path 2026-09-30, after a per-country threshold sensitivity
    sweep (single-tile spot-checks, tau in [0,1]) found the optimal
    wet_fraction threshold diverges by country with no consistent winner -
    Denmark/Finland preferred a lenient threshold (~0.1: CSI fell as tau
    rose), Norway/France preferred a strict one (~1.0: CSI rose as tau
    rose) - so no single global tau serves every country. This soft
    formulation removes the threshold entirely; empirically it reproduces
    the old tau=0.5 default almost exactly for all four tiles tested,
    without needing to pick a value at all.

    tp = model wet, weighted by fraction (partial hit)
    fp = model wet, weighted by (1 - fraction) (partial over-prediction)
    fn = model dry, weighted by fraction (partial miss)
    tn = model dry, weighted by (1 - fraction)

    wet_mask_from_fraction (the 0.5-style threshold call this function
    itself never uses) survives for two narrow, still-real purposes, NOT
    as a scoring shortcut: the agreement-map DISPLAY category raster
    (_CAT_AGREE/_CAT_UNDER/_CAT_OVER need one discrete colour per cell; a
    soft credit can't render as a single colour), and - 2026-10 -
    connectivity_domain_mask's own binary benchmark-wet mask (a seeded
    connected-component analysis needs a binary mask to label; this is
    domain CONSTRUCTION, not scoring - the resulting domain is then scored
    by this function, confusion_counts_soft, exactly like any other
    benchmark). `confusion_counts` itself (the hard-threshold tp/fp/fn/tn
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


# ── Depth-band comparison ────────────────────────────────────────────────

def depth_band_counts(
    model_depth: np.ndarray, ht_min_grid: np.ndarray, ht_max_grid: np.ndarray,
    domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted (agree, under, over) sums within `domain_mask`, further
    restricted to cells the benchmark actually assigns a depth band to
    (`~np.isnan(ht_min_grid)`) - outside any band polygon there is nothing to
    compare the model's depth against, unlike the extent comparison, where
    "benchmark dry" is itself a scoreable state everywhere in the domain.

    agree = model depth within [ht_min, ht_max] (ht_max may be `inf` - see
            rasterize_depth_bands' open-ended-band handling)
    under = model depth < ht_min (model under-predicts the flood depth)
    over  = model depth > ht_max (model over-predicts the flood depth; never
            true where ht_max is `inf`)
    """
    has_band = ~np.isnan(ht_min_grid)
    d = domain_mask & has_band
    agree = float(weight[d & (model_depth >= ht_min_grid) & (model_depth <= ht_max_grid)].sum())
    under = float(weight[d & (model_depth < ht_min_grid)].sum())
    over = float(weight[d & (model_depth > ht_max_grid)].sum())
    return agree, under, over


def depth_band_metrics_from_counts(agree: float, under: float, over: float) -> dict[str, float]:
    """pct_agree/pct_under/pct_over (a 3-way distribution over compared area,
    sums to 1) and depth_EB (over/(over+under) - the SAME over-vs-under
    framing as metrics_from_counts' own EB, 0.5 exactly when over==under) from
    (weighted) agree/under/over counts.

    Zero denominators return NaN, never a silent 0 - same reasoning as
    metrics_from_counts: "no comparable area in this domain" must not read as
    "0% agreement".
    """
    total = agree + under + over
    return {
        "pct_agree": _safe_div(agree, total),
        "pct_under": _safe_div(under, total),
        "pct_over": _safe_div(over, total),
        "depth_EB": _safe_div(over, over + under),
    }


