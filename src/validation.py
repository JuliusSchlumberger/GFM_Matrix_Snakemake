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
    """Per-benchmark validation settings, parsed from one catalog entry's `meta:` block.

    Attributes
    ----------
    key : str
        The catalog entry's own key, used to re-fetch the source data later.
    data_type : str
        Hydromt data type; only "GeoDataFrame" (vector benchmarks) is supported.
    country_iso : str
        ISO-3 country code the benchmark covers.
    hazard_type : str
        "coastal", "fluvial", or "mixed"; only "coastal" is comparable to this model.
    variable : str
        Benchmark variable; only "extent" (binary wet/dry) is supported.
    coverage : str
        "national" (the source surveyed the whole coastline, so outside-polygon means
        surveyed-dry) or "partial".
    attribute_filter : dict or None
        ``{column: value}`` rows to keep when loading the benchmark; None keeps all rows.
    regions : dict[str, list[float]] or None
        ``{name: [minx, miny, maxx, maxy]}`` named sub-regions covering the whole
        country, used to tag evaluation units for per-region reporting.
    geogunit_ids : list[int] or None
        Explicit WRI geogunit_107 unit ID(s) to mask to, bypassing the
        ``country_iso``-based lookup for benchmarks covering a sub-national unit that
        shares its ISO code with sibling units the benchmark does not cover.
    comparison_return_periods : list[int] or None
        Extra native return periods (years) to also score this benchmark against,
        beyond the single global ``validation.return_period``.
    """

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

    `hydromt.DataCatalog.from_yml` opens catalog YAML with a bare `open(...)`, so it
    inherits the OS default encoding (cp1252 on Windows); any non-ASCII byte (e.g. an
    accented `attribute_filter` value) is then silently mojibake-corrupted with no
    exception, and `filter_benchmark`'s exact `==` match against it never matches,
    silently dropping every row. Re-parsing explicitly as UTF-8 and overwriting `meta`
    fixes this regardless of host locale, without touching hydromt itself. Call once
    right after building any DataCatalog over a catalog whose `meta:` blocks may
    contain non-ASCII text.

    Parameters
    ----------
    catalog : hydromt.DataCatalog
        Already-loaded catalog whose sources' `meta` dicts are patched in-place.
    yml_path : str or Path
        Path to the same catalog YAML file `catalog` was loaded from.
    """
    with open(yml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "meta" not in entry or key not in catalog.sources:
            continue
        catalog.get_source(key).meta.update(entry["meta"])


def load_benchmark_spec(catalog, key: str) -> BenchmarkSpec:
    """Read one catalog entry's `meta:` block into a BenchmarkSpec.

    Reads `source.meta`/`source.data_type`, real hydromt DataAdapter attributes
    (confirmed against this repo's installed hydromt version, 2026-09), not a
    re-parse of the YAML.

    Parameters
    ----------
    catalog : hydromt.DataCatalog
        Catalog to look `key` up in.
    key : str
        The catalog entry's key.

    Returns
    -------
    BenchmarkSpec
        The parsed per-benchmark settings for `key`.
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
    """Pick the "main" named region: the one with the largest bbox area.

    Area is lon/lat-degree bbox area (an approximation, fine for picking one region
    out of a handful). Works for both `regions:` usage patterns: a disjoint partition
    (mainland vs. smaller islands) and a nested whole-territory + sub-area set
    (the whole-territory bbox is by construction bigger than any sub-area it
    contains). Used to pick one representative region for a benchmark-wide
    diagnostic that wants a single output (e.g. the tile-coverage plot), not one
    per region.

    Parameters
    ----------
    regions : dict[str, list[float]] or None
        ``{name: [minx, miny, maxx, maxy]}``.

    Returns
    -------
    str or None
        The largest region's name, or None if `regions` is empty/None (caller should
        fall back to "the whole benchmark, no region split").
    """
    if not regions:
        return None

    def _area(bbox: list[float]) -> float:
        minx, miny, maxx, maxy = bbox
        return max(0.0, maxx - minx) * max(0.0, maxy - miny)

    return max(regions, key=lambda name: _area(regions[name]))


def load_benchmark_full(catalog, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Load and filter ALL of a vector benchmark's geometries (no bbox).

    Loads everything once per country/benchmark_key rather than per evaluation
    unit, then applies `spec.attribute_filter` (keep only matching rows).

    Parameters
    ----------
    catalog : hydromt.DataCatalog
        Catalog to read `spec.key` from.
    spec : BenchmarkSpec
        The benchmark's parsed settings; `spec.key` and `spec.attribute_filter` are used.

    Returns
    -------
    geopandas.GeoDataFrame
        The filtered benchmark geometries, or an empty GeoDataFrame (CRS EPSG:4326)
        if the source has none.
    """
    gdf = catalog.get_geodataframe(spec.key)
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return filter_benchmark(gdf, spec)


def filter_benchmark(gdf: gpd.GeoDataFrame, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Keep only the rows matching `spec.attribute_filter` in an already-loaded benchmark.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Benchmark geometries to filter.
    spec : BenchmarkSpec
        Only `spec.attribute_filter` (``{column: value}`` rows to keep) is used; no
        filtering is applied if it is None.

    Returns
    -------
    geopandas.GeoDataFrame
        `gdf` restricted to the matching rows (unchanged if `gdf` is empty or
        `spec.attribute_filter` is None).
    """
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
    """Compute the coverage fraction (0-1) of `gdf`'s geometries within each dst cell.

    Rasterizes at `supersample`x finer resolution than the destination grid
    (reprojecting `gdf` to `dst_crs` first if needed), then block-averages each
    supersample x supersample sub-pixel block down to one destination cell. Unbiased,
    unlike `rasterize(all_touched=False)` (systematically loses thin coastal strips)
    or `all_touched=True` (systematically inflates them). Summing the result directly,
    before any thresholding, also gives an exact benchmark area for a coverage
    diagnostic.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Benchmark geometries.
    dst_transform : affine.Affine
        Affine transform of the destination grid.
    dst_crs
        CRS of the destination grid.
    dst_shape : tuple[int, int]
        ``(height, width)`` of the destination grid.
    supersample : int, default 5
        Sub-pixel rasterization factor per destination-grid axis.

    Returns
    -------
    numpy.ndarray
        float32 array of shape `dst_shape`, each cell's covered fraction in [0, 1].
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
    """Threshold a coverage-fraction grid into a binary benchmark-wet mask.

    Parameters
    ----------
    fraction : numpy.ndarray
        Per-cell benchmark coverage fraction (0-1), e.g. from
        `benchmark_fraction_from_vector`.
    wet_fraction : float, default 0.5
        Minimum fraction for a cell to count as wet.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `fraction`; True where ``fraction >= wet_fraction``.
    """
    return fraction >= wet_fraction


# ── Evaluation domain ────────────────────────────────────────────────────────

def model_domain_mask(depth: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Mark cells where the model actually computed a value (inside its domain).

    Parameters
    ----------
    depth : numpy.ndarray
        Model flood-depth grid.
    nodata : float, default -9999.0
        Sentinel value marking cells outside the model domain.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `depth`; True where ``depth != nodata``.
    """
    return depth != nodata


def permanent_water_mask(land_use: np.ndarray, exclude_codes: tuple[int, ...]) -> np.ndarray:
    """Mark cells that are permanent water (rivers/lakes/sea) per a categorical land-use grid.

    These cells must be excluded from the evaluation domain entirely, not scored as
    agree/dry/wet: "is this pixel flooded" is not meaningful over water that is
    always wet regardless of any storm event. `exclude_codes` depends on the source
    (`validation.permanent_water_source`/`permanent_water_codes` in config.yml) -
    DeltaDTM's land/ocean/lake/river mask uses codes {1,2,3}; the older Copernicus
    Global Land Cover source used {80,200}.

    Parameters
    ----------
    land_use : numpy.ndarray
        Categorical land-use/land-cover grid.
    exclude_codes : tuple[int, ...]
        Category codes that count as permanent water.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `land_use`; True where the code is in
        `exclude_codes`.
    """
    return np.isin(land_use, np.asarray(exclude_codes))


def read_permanent_water_mask(
    gfm_catalog, permanent_water_source: str, permanent_water_codes,
    bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
) -> np.ndarray:
    """Read and reproject a permanent-water (rivers/lakes/sea) mask onto an arbitrary grid.

    `permanent_water_source` is a categorical raster (DeltaDTM's native-resolution
    land/ocean/lake/river mask, ~25-31m, by default; Copernicus land_use, ~100m, is
    the older alternative), always coarser than the model grid, so this always
    nearest-neighbour reprojects onto `out_transform`/`out_shape` (never interpolate a
    categorical raster - same convention as `protection.load_geogunit_ids`). Returns an
    all-False mask on any read failure or empty source, rather than raising. Shared by
    validate_country.py (evaluation-domain construction) and plot_agreement_map.py
    (land background for the agreement map).

    Parameters
    ----------
    gfm_catalog : hydromt.DataCatalog
        Catalog to read `permanent_water_source` from.
    permanent_water_source : str
        Catalog key of the categorical land-use raster.
    permanent_water_codes : Sequence[int]
        Category codes that count as permanent water.
    bbox : list[float]
        ``[minx, miny, maxx, maxy]`` bounding box to read.
    out_transform : affine.Affine
        Affine transform of the destination grid.
    out_shape : tuple[int, int]
        ``(height, width)`` of the destination grid.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape `out_shape`; True where permanent water.
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
    """Build a WRI geogunit_107 ID -> ISO-3 country code lookup.

    Reads FLOPROS's dataframe (indexed by geogunit ID, with an "ISO" column), the
    same table/convention as `compute_exposure_analysis.py`/`compute_flood_totals.py`'s
    own `iso_lookup`, reused here rather than building a second, different mapping.
    Rows with a missing ISO are dropped.

    Parameters
    ----------
    gfm_catalog : hydromt.DataCatalog
        Catalog to read `iso_lookup_source` from.
    iso_lookup_source : str
        Catalog key of the FLOPROS dataframe.

    Returns
    -------
    dict[int, str]
        Geogunit_107 ID -> ISO-3 country code.
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
    """Mask cells whose WRI geogunit resolves to `country_iso` (or `geogunit_ids`).

    The geogunit raster is nearest-neighbour reprojected onto this call's own grid
    (same convention as `read_permanent_water_mask`, never interpolate a categorical
    raster). Needed wherever a comparison's working extent is a rectangular bbox
    rather than the benchmark's own real geometry - a bbox can genuinely overlap a
    neighbouring country's territory, and without this mask GFM's own real flooding
    there would be scored as a false positive against a benchmark never meant to
    cover it. Returns an all-False mask on any read failure or empty source.

    Parameters
    ----------
    gfm_catalog : hydromt.DataCatalog
        Catalog to read `geogunit_source` from.
    geogunit_source : str
        Catalog key of the WRI geogunit_107 raster.
    iso_lookup : dict[int, str]
        Geogunit_107 ID -> ISO-3 country code, from `load_iso_lookup`.
    country_iso : str
        ISO-3 country code to resolve, via `iso_lookup`, when `geogunit_ids` is None.
    bbox : list[float]
        ``[minx, miny, maxx, maxy]`` bounding box to read.
    out_transform : affine.Affine
        Affine transform of the destination grid.
    out_shape : tuple[int, int]
        ``(height, width)`` of the destination grid.
    geogunit_ids : list[int] or None, default None
        Explicit geogunit_107 IDs to match directly instead of resolving `country_iso`
        through `iso_lookup` - for a benchmark covering a sub-national unit that
        shares its ISO code with sibling units it does not cover (e.g. Wales).

    Returns
    -------
    numpy.ndarray
        Boolean array of shape `out_shape`; True where the cell belongs to the
        target country/unit(s).
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
    """Compute weighted (tp, fp, fn, tn) confusion sums within `domain_mask`.

    `weight` is whatever consistent unit the caller wants the metrics in (km², from
    `plotting.pixel_area_km2_grid`, is the production use today; raw cell counts,
    weight=1 everywhere, work too for a reproducibility figure). Hard-threshold
    counterpart to `confusion_counts_soft`; not used in the current production
    scoring path, but still a generic, independently useful building block.

    Parameters
    ----------
    model_wet : numpy.ndarray
        Boolean model wet/dry mask.
    benchmark_wet : numpy.ndarray
        Boolean benchmark wet/dry mask.
    domain_mask : numpy.ndarray
        Boolean mask of cells to include in the sums.
    weight : numpy.ndarray
        Per-cell weight (e.g. area in km²) to sum.

    Returns
    -------
    tuple[float, float, float, float]
        ``(tp, fp, fn, tn)`` weighted sums: tp = model wet AND benchmark wet (hit);
        fp = model wet AND NOT benchmark wet (over-prediction); fn = NOT model wet
        AND benchmark wet (under-prediction); tn = NOT model wet AND NOT benchmark wet.
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
    """Compute weighted (tp, fp, fn, tn) sums using continuous benchmark-coverage credit.

    Uses the benchmark coverage FRACTION directly as continuous credit instead of
    thresholding it into a binary wet/dry call first (`wet_mask_from_fraction`): a
    cell 20% covered by the benchmark contributes 0.2 of its weight to the wet side
    and 0.8 to the dry side, regardless of the model's own (still binary) wet/dry
    call. This is the primary scoring path (since 2026-09-30, replacing
    `confusion_counts` + `wet_mask_from_fraction`): the optimal hard wet_fraction
    threshold varied by benchmark, and soft credit avoids having to choose one.

    Parameters
    ----------
    model_wet : numpy.ndarray
        Boolean model wet/dry mask.
    fraction : numpy.ndarray
        Per-cell benchmark coverage fraction (0-1).
    domain_mask : numpy.ndarray
        Boolean mask of cells to include in the sums.
    weight : numpy.ndarray
        Per-cell weight (e.g. area in km²) to sum.

    Returns
    -------
    tuple[float, float, float, float]
        ``(tp, fp, fn, tn)`` weighted sums: tp = model wet, weighted by fraction
        (partial hit); fp = model wet, weighted by (1 - fraction) (partial
        over-prediction); fn = model dry, weighted by fraction (partial miss);
        tn = model dry, weighted by (1 - fraction).
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
    """Compute hard tp/fp/fn/tn with nearby disagreement forgiven as registration offset.

    Diagnostic companion to `confusion_counts_soft`, not a replacement - the primary
    scoring path's HR/FAR/CSI/EB/bias stay untouched. Answers one narrow question:
    how much of the strict (hard, zero-tolerance) disagreement sits within
    `tolerance_cells` pixels of a cell where model and benchmark genuinely agree,
    i.e. looks like the same flood boundary drawn with a small spatial offset rather
    than a real disagreement about whether an area floods at all. A generic
    small-scale-registration check (not specific to any country or to coastlines,
    though it mostly fires along them in practice); written for the permanent-water-
    mask-vs-benchmark-coastline misalignment caveat in methods_04b_MapsValidation.md.

    Operates on a hard benchmark-wet basis (``fraction > 0``, an existence test, not
    a 0.5-style majority threshold). A disagreement cell is forgiven only if a real
    opposite-type cell exists nearby (binary dilation of the actual wet masks, not a
    blanket buffer around the benchmark extent); forgiveness is symmetric (fp against
    nearby benchmark-wet, fn against nearby model-wet). Weight only ever moves from
    fp/fn into fp_forgiven/fn_forgiven, so CSI_tol is always >= the hard CSI at
    tolerance_cells=0. `tolerance_cells` should be set from a real registration-
    uncertainty distance (this pipeline's 30m grid and the permanent-water mask's
    ~25-31m resolution are both close to one cell), not tuned upward until the
    number looks good; callers should report fp_forgiven/fn_forgiven alongside
    CSI_tol so a reader can see how much work the tolerance is doing.

    Parameters
    ----------
    model_wet : numpy.ndarray
        Boolean model wet/dry mask.
    fraction : numpy.ndarray
        Per-cell benchmark coverage fraction; thresholded here at ``> 1e-9`` for a
        hard benchmark-wet basis.
    domain_mask : numpy.ndarray
        Boolean mask of cells to include in the sums.
    weight : numpy.ndarray
        Per-cell weight (e.g. area in km²) to sum.
    tolerance_cells : int
        Dilation radius, in cells, within which a disagreement is forgiven as
        registration offset; 0 disables forgiveness.

    Returns
    -------
    dict[str, float]
        ``tp``, ``fp``, ``fn``, ``tn`` (feed into `metrics_from_counts` for
        CSI_tol/HR_tol/FAR_tol), plus ``fp_forgiven``/``fn_forgiven`` (the weight
        moved out of fp/fn by the tolerance check).
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
    """Compute HR, FAR, CSI, EB, EB_ratio, and bias from (weighted) TP/FP/FN(/TN) counts.

    Zero denominators return NaN, never a silent 0: "no benchmark wet area in this
    domain" must not read as "CSI = 0". `EB` is the over/under-prediction form,
    exactly 0.5 whenever FP == FN (both nonzero); `EB_ratio = FP/FN` is the Wing et
    al. (2017) form, more legible at extremes, kept alongside (not instead of) `EB`.

    Parameters
    ----------
    tp : float
        Weighted true-positive (hit) count.
    fp : float
        Weighted false-positive (over-prediction) count.
    fn : float
        Weighted false-negative (under-prediction) count.
    tn : float, default 0.0
        Weighted true-negative count; unused by any of the returned metrics.

    Returns
    -------
    dict[str, float]
        ``HR``, ``FAR``, ``CSI``, ``EB``, ``EB_ratio``, ``bias``.
    """
    return {
        "HR": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "EB": _safe_div(fp, fp + fn),
        "EB_ratio": _safe_div(fp, fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
