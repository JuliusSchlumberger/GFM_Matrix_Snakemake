"""Coastal flood-extent validation against national hazard maps.

Benchmark loading/dispatch, evaluation-cluster construction,
benchmark-to-model-grid rasterization, evaluation-domain construction
(including permanent-water exclusion), and area-/population-weighted
contingency metrics. See docs/flood_extent_validation_plan.md for the full
design - this module holds the stateless computational building blocks; the
per-cluster iteration, accumulation, and I/O live in
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

from config_utils import retry_transient_io


# ── Benchmark loading/dispatch ──────────────────────────────────────────────

@dataclass
class BenchmarkSpec:
    """Parsed from one data_catalog_validation.yml entry's `meta:` block.

    Only the fields the vector-benchmark path (the only one implemented so
    far - see validate_country.py) actually reads. Raster-benchmark-only
    fields (`wet_values`, `depth_threshold_m`, a `nodata` override,
    `return_period`, `variable`) were removed 2026-09 as dead code along
    with `benchmark_fraction_from_raster` - re-add them together with that
    function's caller when a raster benchmark is actually wired up.
    """

    key: str
    data_type: str              # "GeoDataFrame" | "RasterDataset"
    country_iso: str
    hazard_type: str            # coastal | fluvial | mixed - only "coastal" is comparable
    variable: str = "extent"    # extent | depth - dispatches validate_country.py to the
    # binary wet/dry comparison (extent) or the continuous-depth-vs-band comparison
    # (depth, France's n_iso_ht_* layers - see validate_country.validate_country_depth_bands)
    ht_min_col: str = "ht_min"  # variable="depth" only: column holding each band's lower bound (m)
    ht_max_col: str = "ht_max"  # variable="depth" only: column holding each band's upper bound (m) -
    # open-ended top bands use inconsistent, region/zone-specific sentinel values (confirmed
    # 2026-09 against France's real data: 9999/999/99, or NaN entirely) rather than a real depth -
    # see rasterize_depth_bands' open_ended_min_m param, not this column's raw values, for how
    # those get resolved to "no upper bound".
    coverage: str = "partial"   # partial | national - dispatches validate_country.py to the
    # cluster-based comparison (partial, the default - Spain/France's benchmarks only survey
    # isolated designated zones, so "outside any benchmark polygon" is genuinely ambiguous
    # between "surveyed and dry" and "never looked at" - see caveats doc §1.1, the whole
    # reason build_evaluation_clusters' buffer/merge exists) or the per-postprocessing-chunk
    # comparison (national - the benchmark's own source assessed the ENTIRE coastline, so
    # "outside the polygon" unambiguously means "surveyed, found dry" and a model-wet cell
    # there is a genuine over-prediction, not an ambiguous FP - see
    # validate_country.validate_country_national_coverage, added 2026-09 after Norway's
    # long, only-lightly-fragmented coastline collapsed into one giant buffer/merge cluster
    # spanning almost the whole country and crashed the cluster-based path's per-cluster
    # bbox-sized array allocation - chunk-sized processing is bounded regardless of how far a
    # national benchmark's footprint spans).
    attribute_filter: dict | None = None    # vector: {column: value} rows to KEEP
    exclude_bbox: list[float] | None = None  # vector: [minx, miny, maxx, maxy] to DROP
    regions: dict[str, list[float]] | None = None  # {name: [minx, miny, maxx, maxy]} -
    # named sub-regions (mainland, each overseas territory/archipelago) covering the
    # WHOLE country, used to (a) tag each evaluation cluster for per-region CSV
    # reporting instead of one blended national row, and (b) as the read extent for
    # compute_region_model_totals's benchmark-independent "how much does the model
    # flood/expose in this region, full stop" figures. Every region a country cares
    # about needs its own explicit bbox here - there is no implicit "everything else"
    # catch-all, since that would just be the cluster union again (see caveats doc
    # §1.2 for why that's not wide enough to be a useful comparison).


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


# ── Bringing the benchmark onto the model grid (plan §4.2) ─────────────────

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
    them) - see plan doc §4.2. Also gives an exact benchmark area for the
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
    """Binary benchmark-wet decision from a coverage fraction grid (plan §4.2's 0.5 rule)."""
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


# ── Evaluation domain (plan §4.3) ───────────────────────────────────────────

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


def permanent_water_mask(land_use: np.ndarray, exclude_codes: tuple[int, ...]) -> np.ndarray:
    """True where a cell is permanent water (rivers/lakes/sea) per whichever
    categorical source `land_use` came from and its own `exclude_codes`
    (`validation.permanent_water_source`/`permanent_water_codes` in
    config.yml - DeltaDTM's own land/ocean/lake/river mask, codes {1,2,3},
    since 2026-09; was Copernicus Global Land Cover, codes {80,200} - see
    docs/flood_extent_validation_caveats.md §1.3 for why the switch).

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
    resolution improvement, see docs/flood_extent_validation_caveats.md §1.3
    - either works, this function doesn't care which), coarser than the model
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
) -> np.ndarray:
    """True where a cell's WRI geogunit (nearest-neighbour reprojected onto this
    call's own grid - same convention as read_permanent_water_mask, never
    interpolate a categorical raster) resolves to `country_iso`.

    Needed anywhere a comparison's working extent is a rectangular bbox rather
    than the benchmark's own real geometry - a bbox can genuinely overlap a
    neighbouring country's territory (confirmed 2026-09, twice: Spain's
    `mainland` bbox overlaps Portugal/France/Morocco - see the flood_totals
    history in docs/flood_extent_validation_caveats.md §1.2; Norway's
    `mainland` bbox, generous enough to cover the country's full latitude
    range, overlaps real Swedish and Danish territory too -
    validate_country_national_coverage processes chunks found from that bbox
    directly, with no benchmark-geometry clustering step to naturally exclude
    them the way the partial-coverage path's cluster polygons do). Without
    this mask, GFM's own real flooding in a neighbouring country would be
    scored as a false positive against a benchmark that was never meant to
    cover that territory at all.
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
    target_ids = [gid for gid, iso in iso_lookup.items() if iso == country_iso]
    if not target_ids:
        return np.zeros(out_shape, dtype=bool)
    return np.isin(geo_ids, target_ids)


# ── Metrics (plan §4.4) ─────────────────────────────────────────────────────

def confusion_counts(
    model_wet: np.ndarray, benchmark_wet: np.ndarray, domain_mask: np.ndarray, weight: np.ndarray,
) -> tuple[float, float, float, float]:
    """Weighted (tp, fp, fn, tn) sums within `domain_mask`.

    `weight` is whatever consistent unit the caller wants area- or
    population-weighted metrics in (km² from plotting.pixel_area_km2_grid,
    or people from a population grid - plan doc §4.4/§4.5). Raw cell counts
    (weight=1 everywhere) work too, for the reproducibility figure the plan
    also wants reported alongside the area-weighted ones.

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


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float = 0.0) -> dict[str, float]:
    """HR, FAR, CSI, EB, EB_ratio, bias from (weighted) TP/FP/FN(/TN) counts.

    Zero denominators return NaN, never a silent 0 - "no benchmark wet area
    in this domain" must not read as "CSI = 0" (plan doc §4.4). `EB == 0.5`
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


# ── Exposure difference / population weighting (plan §4.5) ─────────────────

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


def population_by_class(
    class_mask: np.ndarray,
    domain_mask: np.ndarray,
    src_transform: Affine,
    src_crs,
    population: np.ndarray,
    pop_transform: Affine,
    pop_crs,
) -> np.ndarray:
    """Population (on the population grid) disaggregated into `class_mask`.

    Implements the plan doc §4.5 identity without materialising a
    fine-resolution population raster:

        sum over fine cells of class C of (pop_coarse / n_subcells)
            == sum over coarse cells of pop_coarse x (fraction of that
               coarse cell in class C)

    `rasters.average_pool_to_grid` computes exactly that fraction grid (fine
    class_mask, restricted to domain_mask -> coarse fraction); multiplying
    by `population` gives this class's disaggregated population per coarse
    cell. Sum the result for pop_tp/pop_fp/pop_fn/pop_tn (e.g. class_mask =
    model_wet & ~benchmark_wet for pop_fp/"pop_over").

    Imports `average_pool_to_grid` lazily to avoid a hard rasters.py
    dependency for callers that only need the non-population metrics.
    """
    from rasters import average_pool_to_grid

    numerator = np.where(domain_mask, np.where(class_mask, 1.0, 0.0), np.nan).astype("float32")
    domain = domain_mask.astype("float32")
    fraction = average_pool_to_grid(
        numerator=numerator, domain=domain,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=pop_transform, dst_crs=pop_crs, dst_shape=population.shape,
        numerator_nodata=np.nan,
    )
    fraction = np.nan_to_num(fraction, nan=0.0)
    return population * fraction
