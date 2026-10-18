"""Stateless building blocks for coastal flood-extent validation against hazard maps.

Driven by validation/validate_country.py; design in docs/methods_04b_MapsValidation.md.
Per-benchmark settings live in data_catalog_validation.yml `meta:` blocks.
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
    """Per-benchmark validation settings, parsed from one catalog entry's `meta:` block."""

    key: str
    data_type: str              # only "GeoDataFrame" (vector) is supported
    country_iso: str
    hazard_type: str            # coastal | fluvial | mixed - only "coastal" is comparable
    variable: str = "extent"    # binary wet/dry
    # national: the source surveyed the whole coastline, so outside-polygon = surveyed dry.
    coverage: str = "national"
    attribute_filter: dict | None = None    # {column: value} rows to keep
    # {name: [minx, miny, maxx, maxy]} covering the whole country; no implicit catch-all.
    regions: dict[str, list[float]] | None = None
    # WRI geogunit_107 IDs for sub-national units sharing an ISO code (e.g. Wales 3367,
    # GBR like England/Scotland/N.Ireland 3364/3366/3365).
    geogunit_ids: list[int] | None = None
    # Extra native RPs (years, each needs merged chunks) scored by
    # validation/run_multi_rp_summary.py besides validation.return_period.
    comparison_return_periods: list[int] | None = None


def fix_catalog_meta_encoding(catalog, yml_path: str | Path) -> None:
    """Re-read `yml_path` as UTF-8 and patch each loaded source's `meta` in place.

    hydromt's from_yml uses the OS encoding (cp1252 on Windows), silently
    mojibaking non-ASCII `attribute_filter` values so no rows match.
    """
    with open(yml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "meta" not in entry or key not in catalog.sources:
            continue
        catalog.get_source(key).meta.update(entry["meta"])


def load_benchmark_spec(catalog, key: str) -> BenchmarkSpec:
    """Read one catalog entry's `meta:` block into a BenchmarkSpec."""
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
    """Return the region with the largest lon/lat-degree bbox area, or None if none."""
    if not regions:
        return None

    def _area(bbox: list[float]) -> float:
        minx, miny, maxx, maxy = bbox
        return max(0.0, maxx - minx) * max(0.0, maxy - miny)

    return max(regions, key=lambda name: _area(regions[name]))


def load_benchmark_full(catalog, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Load all of a vector benchmark's geometries (no bbox) and apply its filter."""
    gdf = catalog.get_geodataframe(spec.key)
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return filter_benchmark(gdf, spec)


def filter_benchmark(gdf: gpd.GeoDataFrame, spec: BenchmarkSpec) -> gpd.GeoDataFrame:
    """Keep only the rows matching `spec.attribute_filter` in an already-loaded benchmark."""
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

    Rasterizes at `supersample`x per axis and block-averages: unbiased, unlike
    all_touched=False (loses thin coastal strips) or True (inflates them).
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
    """Threshold a coverage-fraction grid into a binary benchmark-wet mask."""
    return fraction >= wet_fraction


# ── Evaluation domain ────────────────────────────────────────────────────────

def model_domain_mask(depth: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Mark cells where the model computed a value (``depth != nodata``)."""
    return depth != nodata


def permanent_water_mask(land_use: np.ndarray, exclude_codes: tuple[int, ...]) -> np.ndarray:
    """Mark permanent-water cells (excluded from the evaluation domain entirely).

    Codes depend on the source: DeltaDTM mask {1,2,3}; Copernicus land cover {80,200}.
    """
    return np.isin(land_use, np.asarray(exclude_codes))


def _read_categorical_on_grid(
    gfm_catalog, source: str, bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
    dst_crs="EPSG:4326", dtype: str = "float64", variables: list[str] | None = None,
) -> np.ndarray | None:
    """Read a categorical catalog raster, nearest-reprojected onto a target grid.

    `bbox` is the EPSG:4326 read extent. Cells without a source value are -1;
    returns None on read failure or an empty source. `dtype` float32 halves
    memory on very large grids (SF Bay at 2 m).
    """
    read_kwargs = {"bbox": bbox}
    if variables is not None:
        read_kwargs["variables"] = variables
    try:
        da = retry_transient_io(gfm_catalog.get_rasterdataset, source, **read_kwargs).squeeze(drop=True)
    except Exception:
        return None
    if da is None or da.size == 0:
        return None

    dst = np.full(out_shape, -1.0, dtype=dtype)
    reproject(
        source=da.values.astype(dtype), destination=dst,
        src_transform=da.raster.transform, src_crs=da.raster.crs,
        dst_transform=out_transform, dst_crs=dst_crs,
        src_nodata=da.raster.nodata, dst_nodata=-1.0,
        resampling=Resampling.nearest,
    )
    return dst


def read_permanent_water_mask(
    gfm_catalog, permanent_water_source: str, permanent_water_codes,
    bbox: list[float], out_transform: Affine, out_shape: tuple[int, int],
) -> np.ndarray:
    """Read a categorical permanent-water raster, nearest-reprojected to an EPSG:4326 grid.

    Source: DeltaDTM mask (~25-31 m) or Copernicus land use (~100 m), always
    coarser than the model grid. Returns all-False on read failure or empty source.
    """
    dst = _read_categorical_on_grid(gfm_catalog, permanent_water_source, bbox, out_transform, out_shape)
    if dst is None:
        return np.zeros(out_shape, dtype=bool)
    return permanent_water_mask(dst, tuple(permanent_water_codes))


def load_iso_lookup(gfm_catalog, iso_lookup_source: str) -> dict[int, str]:
    """Build a WRI geogunit_107 ID -> ISO-3 lookup from the FLOPROS table (ISO column)."""
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

    Keeps a rectangular bbox from scoring neighbouring countries' flooding as
    false positives. Nearest reprojection; all-False on read failure.
    """
    dst = _read_categorical_on_grid(
        gfm_catalog, geogunit_source, bbox, out_transform, out_shape, variables=["Geogunits"],
    )
    if dst is None:
        return np.zeros(out_shape, dtype=bool)
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
    """Compute hard-threshold weighted (tp, fp, fn, tn) sums within `domain_mask`.

    `weight` sets the unit (km² in production, 1 for cell counts).
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
    """Compute weighted (tp, fp, fn, tn) with benchmark coverage fraction as soft credit.

    Primary scoring path: a cell 20% covered adds 0.2 of its weight to the wet
    side and 0.8 to the dry side; the model call stays binary. Avoids choosing
    a hard wet_fraction, whose optimum varied by benchmark.
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
    """Compute hard tp/fp/fn/tn, forgiving disagreement near agreement as registration offset.

    Diagnostic only (see methods_04b_MapsValidation.md); primary scores are
    untouched. Benchmark wet = fraction > 1e-9 (existence, not majority). An
    fp (fn) cell is forgiven if benchmark-wet (model-wet) lies within
    `tolerance_cells` (8-connected dilation; 0 disables). Forgiven weight is
    returned as fp_forgiven/fn_forgiven, so CSI_tol >= hard CSI. Set the
    tolerance from real registration uncertainty (~1 cell: 30 m grid vs
    ~25-31 m water mask), not tuned upward; report the forgiven weight.
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
    """Compute HR, FAR, CSI, EB, EB_ratio and bias from (weighted) counts; `tn` is unused.

    Zero denominators give NaN, never 0. EB = FP/(FP+FN) (0.5 when FP == FN);
    EB_ratio = FP/FN is the Wing et al. (2017) form.
    """
    return {
        "HR": _safe_div(tp, tp + fn),
        "FAR": _safe_div(fp, tp + fp),
        "CSI": _safe_div(tp, tp + fp + fn),
        "EB": _safe_div(fp, fp + fn),
        "EB_ratio": _safe_div(fp, fn),
        "bias": _safe_div(tp + fp, tp + fn),
    }
