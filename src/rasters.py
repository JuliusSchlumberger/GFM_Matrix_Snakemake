"""Raster extraction and processing functions for a single tile.

Tiles in the tile grid (see `src/connectivity_tiling.py`) are exact rectangles,
so a tile's bounding box is identical to its geometry.
This means raster datasets can simply be clipped with `bbox=...` - no
additional `geometry_mask` step is required.
"""

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import hydromt
import numpy as np
import rasterio
import xarray as xr
from hydromt.exceptions import NoDataException
from rasterio.fill import fillnodata
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from config_utils import retry_transient_io

from merge import AQUEDUCT_NODATA

# ── Integer raster encodings (I/O optimization - see conversation 2026-08-01) ──
#
# DEM: DeltaDTM's valid range is <=30m (even after vertical-datum
# correction), so int16 centimetres gives enormous headroom (+-327.67m).
# The old float land-fill sentinel (9999m, dem_gap_fill.land_fill_value_m)
# would overflow int16 at cm scale by ~30x, so gap-filled cells now get 99m
# instead - comfortably above any real valid value, comfortably inside
# int16's range.
DEM_SCALE = 100  # cm per metre
DEM_NODATA_M = 99.0
DEM_NODATA_INT16 = int(DEM_NODATA_M * DEM_SCALE)  # 9900

# Friction: real land-cover-derived values (Manning's n from
# lu_to_roughness_lookup.csv, /100 - see compute_friction) span roughly
# 0.0001-0.0012. x1_000_000 keeps the source's full real precision
# (Manning's n only has 2 meaningful decimal digits) while leaving ~27x
# headroom below int16's max before any plausible lookup-table value would
# overflow it.
FRICTION_SCALE = 1_000_000

# Water depth output: real flood depths never approach int16's ceiling in
# cm, so the max representable value doubles as an unambiguous "not
# computed" sentinel (the int16 counterpart to merge.AQUEDUCT_NODATA, which
# is sized for float32 and does not fit in int16).
WATERDEPTH_SCALE = 100  # cm per metre
WATERDEPTH_NODATA_INT16 = np.iinfo(np.int16).max  # 32767


def _encode_int16(values: np.ndarray, scale: float, label: str, valid_hint: str) -> np.ndarray:
    """Shared scale-and-cast-to-int16 body for the encoders below - raises on
    out-of-range input rather than silently wrapping (numpy's int16 cast
    does not raise on overflow, it wraps).
    """
    scaled = np.round(values.astype(np.float64) * scale)
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if scaled.min() < lo or scaled.max() > hi:
        raise ValueError(
            f"{label} value out of int16 range after x{scale:g} scaling: "
            f"[{scaled.min() / scale:g}, {scaled.max() / scale:g}] (expected {valid_hint})"
        )
    return scaled.astype(np.int16)


def encode_dem_cm(dem_m: np.ndarray) -> np.ndarray:
    """Encode a float DEM (metres) as int16 centimetres.

    DEM_VALUE_CLIPPED_TO_INT16_RANGE (search this token to find this note
    again): values below DeltaDTM's usual <=30m envelope can still be real
    (confirmed on the Dead Sea and Danakil Depression tiles, both beyond
    int16-cm's +-327.67m range) but are CLIPPED here rather than rejected,
    since an elevation that far below any coastal water level never floods
    either way. `_encode_int16` itself still raises hard on overflow; only
    this DEM-specific caller clips first.

    Parameters
    ----------
    dem_m : np.ndarray
        DEM elevation values in metres.

    Returns
    -------
    np.ndarray
        `dem_m` scaled by `DEM_SCALE` and cast to int16 centimetres, clipped
        to int16's representable range.
    """
    lo_m = np.iinfo(np.int16).min / DEM_SCALE  # -327.68
    hi_m = np.iinfo(np.int16).max / DEM_SCALE  # 327.67
    n_clipped = int(np.sum((dem_m < lo_m) | (dem_m > hi_m)))
    if n_clipped:
        print(
            f"  DEM_VALUE_CLIPPED_TO_INT16_RANGE: {n_clipped} cell(s) outside "
            f"[{lo_m:g}, {hi_m:g}] m (actual range [{float(dem_m.min()):g}, "
            f"{float(dem_m.max()):g}] m) clipped before int16-cm encoding - "
            "see encode_dem_cm's docstring.",
            flush=True,
        )
    dem_m = np.clip(dem_m, lo_m, hi_m)
    return _encode_int16(dem_m, DEM_SCALE, "DEM", f"within DeltaDTM's <=30m validity envelope + the {DEM_NODATA_M}m gap-fill sentinel")


def decode_dem_cm(dem_int16: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre DEM back to float32 metres.

    Parameters
    ----------
    dem_int16 : np.ndarray
        DEM encoded as int16 centimetres (see `encode_dem_cm`).

    Returns
    -------
    np.ndarray
        DEM elevation in metres, as float32.
    """
    return dem_int16.astype(np.float32) / DEM_SCALE


def encode_friction_int16(friction: np.ndarray) -> np.ndarray:
    """Encode a float friction raster as int16 (x `FRICTION_SCALE`).

    Parameters
    ----------
    friction : np.ndarray
        Friction values in the per-metre Manning's-n-derived convention
        (see `compute_friction`).

    Returns
    -------
    np.ndarray
        `friction` scaled by `FRICTION_SCALE` and cast to int16.
    """
    return _encode_int16(friction, FRICTION_SCALE, "Friction", "within the land-cover lookup table's Manning's n / 100 range")


def decode_friction_int16(friction_int16: np.ndarray) -> np.ndarray:
    """Decode an int16 friction raster back to float32.

    Divides by `np.float32(FRICTION_SCALE)`, not the raw Python int: numpy's
    legacy value-based casting silently promotes the result to float64
    once the divisor needs more than 16 bits (`FRICTION_SCALE = 1_000_000`
    does), which previously doubled the peak memory of every flood solve
    since this array's dtype propagates to `solve_eikonal_dense`'s `t`.

    Parameters
    ----------
    friction_int16 : np.ndarray
        Friction encoded as int16 (see `encode_friction_int16`).

    Returns
    -------
    np.ndarray
        Friction values as float32, in the per-metre convention.
    """
    return friction_int16.astype(np.float32) / np.float32(FRICTION_SCALE)


def encode_waterdepth_cm(waterdepth_m: np.ndarray) -> np.ndarray:
    """Encode a float water-depth raster (metres) as int16 centimetres.

    Parameters
    ----------
    waterdepth_m : np.ndarray
        Water depth in metres, `0.0` where dry.

    Returns
    -------
    np.ndarray
        `waterdepth_m` scaled by `WATERDEPTH_SCALE` and cast to int16.

    Raises
    ------
    ValueError
        If any encoded value would collide with `WATERDEPTH_NODATA_INT16`
        (implausible for a real flood depth).
    """
    encoded = _encode_int16(waterdepth_m, WATERDEPTH_SCALE, "Water depth", "well below the int16 'not computed' sentinel")
    if (encoded == WATERDEPTH_NODATA_INT16).any():
        raise ValueError(
            f"Water depth reaches {WATERDEPTH_NODATA_INT16 / WATERDEPTH_SCALE}m, "
            "colliding with the int16 'not computed' sentinel - implausible for a real flood depth."
        )
    return encoded


def decode_waterdepth_cm(waterdepth_int16: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre water-depth raster to float32 metres.

    `WATERDEPTH_NODATA_INT16` cells are mapped to `merge.AQUEDUCT_NODATA` so
    the existing float32 merge/overlap-correction pipeline needs no changes.

    Parameters
    ----------
    waterdepth_int16 : np.ndarray
        Water depth encoded as int16 centimetres (see `encode_waterdepth_cm`).

    Returns
    -------
    np.ndarray
        Water depth in metres, as float32, with `merge.AQUEDUCT_NODATA`
        where not computed.
    """
    decoded = waterdepth_int16.astype(np.float32) / WATERDEPTH_SCALE
    decoded[waterdepth_int16 == WATERDEPTH_NODATA_INT16] = AQUEDUCT_NODATA
    return decoded


# Boundary-forcing (COAST-RP + SLR) water levels stored in boundaries.gpkg
# (see boundaries.save_boundary_points/scripts/extract_boundaries.py): real
# storm-tide + SLR values comfortably fit the same int16-centimetre
# convention as DEM/friction/waterdepth (2026-08) - these are the values
# flood_model._idw_seed_values seeds ocean-boundary cells from and compares
# directly against `dem`, so keeping the same precision as everything else
# it's compared against matters, even though boundaries.gpkg is a small
# vector GeoPackage (not a raster) and gains no I/O-size benefit from it.
WATERLEVEL_SCALE = 100  # cm per metre


def encode_waterlevel_cm(waterlevel_m: np.ndarray) -> np.ndarray:
    """Encode float boundary water levels (metres) as int16 centimetres.

    An empty input (a tile with no findable COAST-RP station - see
    scripts/extract_boundaries.py's empty-placeholder handling) is returned
    as an empty int16 array as-is, bypassing `_encode_int16`'s min/max range
    check, which raises on an empty array.

    Parameters
    ----------
    waterlevel_m : np.ndarray
        Boundary-forcing water levels in metres.

    Returns
    -------
    np.ndarray
        `waterlevel_m` scaled by `WATERLEVEL_SCALE` and cast to int16.
    """
    if waterlevel_m.size == 0:
        return waterlevel_m.astype(np.int16)
    return _encode_int16(
        waterlevel_m, WATERLEVEL_SCALE, "Boundary water level",
        "within COAST-RP storm-tide + SLR fingerprint's realistic range",
    )


def decode_waterlevel_cm(waterlevel_int16: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre boundary water level array to float32 metres.

    Parameters
    ----------
    waterlevel_int16 : np.ndarray
        Boundary water levels encoded as int16 centimetres (see
        `encode_waterlevel_cm`).

    Returns
    -------
    np.ndarray
        Boundary water levels in metres, as float32.
    """
    return waterlevel_int16.astype(np.float32) / WATERLEVEL_SCALE


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as an xarray DataArray with hydromt raster accessor.

    Parameters
    ----------
    path : str or Path
        Path to the single-band GeoTIFF file.

    Returns
    -------
    xr.DataArray
        The raster's single band, with the `band` dimension squeezed out.
    """
    import rioxarray  # noqa: F401 - registers the rasterio xarray backend and spatial metadata
    return retry_transient_io(xr.open_dataarray, path, engine="rasterio").squeeze("band", drop=True)


def get_tile_bbox(tile: gpd.GeoDataFrame) -> list[float]:
    """Return the bounding box of a single-row tile GeoDataFrame.

    Parameters
    ----------
    tile : gpd.GeoDataFrame
        A single-row GeoDataFrame whose geometry is the tile's rectangle.

    Returns
    -------
    list of float
        Bounding box as ``[minx, miny, maxx, maxy]``.
    """
    return tile.total_bounds.tolist()


def compute_model_bbox(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    mask_source: str,
    tile_bbox: list[float],
    buffer_arcsec: float = 1.0,
    elevation_threshold_m: float | None = None,
) -> list[float]:
    """Compute the model domain bbox as the DEM's flood-relevant extent plus a buffer.

    The full tile bbox is much larger than needed: most of the tile is inland
    land without DeltaDTM coverage. Shrinks the domain to the bounding box of
    flood-relevant DeltaDTM cells expanded by `buffer_arcsec` on all four
    sides, then clips the result to `tile_bbox`.

    A cell counts as flood-relevant (kept) unless it is *confirmed* high
    land: ocean/lake/river cells always count, and a land cell only drops out
    when its raw elevation is both present and >= `elevation_threshold_m`
    (land with no elevation reading is conservatively kept, since `extract_dem`
    will interpolate or hard-fill it later anyway). This mirrors the
    obstacle-coupling pre-filter's `dem > max_waterlevel` reasoning, applied
    at domain-selection time; the threshold should match `tile_grid.
    elevation_relevance_threshold_m` (tiles.py) so a tile's recorded
    footprint and its extracted domain agree. If `elevation_threshold_m` is
    `None`, any non-nodata DEM cell counts as valid (land/water split
    ignored). If the DEM has no valid cells in `tile_bbox`, `tile_bbox` is
    returned unchanged.

    Parameters
    ----------
    data_catalog : hydromt.DataCatalog
        HydroMT data catalog.
    dem_source : str
        Catalog key for the DEM (DeltaDTM) RasterDataset.
    mask_source : str
        Catalog key for the DeltaDTM validity mask RasterDataset. Only read
        when `elevation_threshold_m` is given.
    tile_bbox : list of float
        Full tile bounding box as ``[minx, miny, maxx, maxy]``.
    buffer_arcsec : float, default 1.0
        Buffer added to each side of the DEM extent, in arc-seconds. DeltaDTM
        covers right to the shoreline, so even 1 arc-second (~30 m) is enough
        to include a thin seaward strip of ocean cells for flood entry.
    elevation_threshold_m : float, optional
        Elevation (m) at/above which confirmed land is excluded from the
        domain. `None` (default) disables the elevation criterion entirely.

    Returns
    -------
    list of float
        Reduced bounding box ``[minx, miny, maxx, maxy]``, clipped to `tile_bbox`.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=tile_bbox)

    if elevation_threshold_m is None:
        valid = (da.values != da.raster.nodata).squeeze()
    else:
        da_mask = data_catalog.get_rasterdataset(mask_source, bbox=tile_bbox)
        da_mask.raster.set_nodata(255)
        mask_vals = da_mask.raster.reproject_like(da, method="nearest").values.squeeze()
        dem_vals = da.values.squeeze()

        # Land (0) or no mask coverage at all (255) -> land, same convention
        # as extract_dem/extract_dem_mask. Ocean/lake/river always stay valid.
        is_land = (mask_vals == 0) | (mask_vals == 255)
        dem_valid = dem_vals != da.raster.nodata
        confirmed_high_land = is_land & dem_valid & (dem_vals >= elevation_threshold_m)
        valid = ~confirmed_high_land

    if not valid.any():
        return list(tile_bbox)

    y_coords = da.y.values  # descending (north first)
    x_coords = da.x.values  # ascending
    dy = float(abs(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 0.0
    dx = float(abs(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 0.0
    rows, cols = np.where(valid)

    buf = buffer_arcsec / 3600.0
    minx = float(x_coords[cols.min()]) - dx / 2 - buf
    miny = float(y_coords[rows.max()]) - dy / 2 - buf   # rows.max() = southernmost
    maxx = float(x_coords[cols.max()]) + dx / 2 + buf
    maxy = float(y_coords[rows.min()]) + dy / 2 + buf   # rows.min() = northernmost

    # Final clamp to the valid global coordinate range, independent of
    # tile_bbox: a tile whose own bbox already overshoots -180/180 (e.g. an
    # antimeridian-adjacent tile, however slightly, from upstream trim/
    # buffer arithmetic in tiles.py) would otherwise pass that overshoot
    # straight through - `max(minx, tile_bbox[0])` doesn't help when
    # tile_bbox[0] is ITSELF already invalid. A bbox even a fraction of a
    # degree past +/-180 can make downstream hydromt/rioxarray reprojection
    # windows wrap around almost the entire globe instead of clipping
    # cleanly, blowing up into a hundreds-of-GB allocation for what should
    # be a single small tile (see extract_dem's da_mask.raster.reproject_like).
    return [
        max(max(minx, tile_bbox[0]), -180.0),
        max(max(miny, tile_bbox[1]), -90.0),
        min(min(maxx, tile_bbox[2]), 180.0),
        min(min(maxy, tile_bbox[3]), 90.0),
    ]


def _compute_dem_gap_fill(
    dem_vals: np.ndarray,
    nodata: float,
    is_land: np.ndarray,
    min_hard_fill_component_size: int,
    interp_max_search_distance: float,
    interp_smoothing_iterations: int,
    land_fill_value_m: float,
) -> np.ndarray:
    """Pure-array core of `extract_dem`'s gap-filling logic - land/river/lake
    classification, small-vs-large missing-land gap splitting, and
    small-gap interpolation - extracted so it's directly testable without a
    real hydromt.DataCatalog. See `extract_dem`'s own docstring for the full
    land/ocean/lake/river fill-value reasoning; this function is exactly
    that logic, operating on plain arrays.
    """
    missing = dem_vals == nodata
    missing_land = missing & is_land

    # Connected components of missing-land cells only (4-connectivity) -
    # ocean/lake/river nodata cells never participate here and always get
    # 0.0 regardless of gap size.
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labels, _ = ndimage.label(missing_land, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    cell_component_size = component_sizes[labels]
    small_gap = missing_land & (cell_component_size < min_hard_fill_component_size)

    if small_gap.any():
        # Interpolates a candidate value for every originally-missing cell
        # (land and water alike) from real valid DeltaDTM cells only - only
        # the small_gap subset of these candidates is actually used below;
        # large land gaps and water cells get their own fixed fill value
        # regardless of what this computes for them.
        interpolated = fillnodata(
            dem_vals.astype(np.float32).copy(),
            mask=(~missing).astype(np.uint8),
            max_search_distance=interp_max_search_distance,
            smoothing_iterations=interp_smoothing_iterations,
        )
        # fillnodata has nothing to interpolate FROM if the entire read
        # window happens to have zero valid DEM coverage (possible for a
        # small/edge tile right at DeltaDTM's own coverage boundary, even
        # when the validity mask itself claims land/ocean structure there -
        # a genuine source-data mismatch, not a code bug on its own) - such
        # cells are left unchanged (still the raw nodata sentinel) rather
        # than actually filled. Route them through the same hard-fill as a
        # large gap instead of letting a raw nodata value leak into the
        # encoded output (found 2026-08 - a ~3K-pixel tile crashed
        # encode_dem_cm with a stray -9999).
        interpolated = np.where(interpolated == nodata, land_fill_value_m, interpolated)
    else:
        interpolated = dem_vals  # never read below (small_gap is all-False)

    return np.where(
        is_land,
        np.where(small_gap, interpolated, land_fill_value_m),
        0.0,
    ).astype(np.float32)


_GEBCO_BBOX_BUFFER_DEG = 0.01  # ~1.1km - safely > GEBCO's ~0.0042deg (463m) native pixel


def _buffer_bbox_for_coarse_source(bbox: list[float], buffer_deg: float = _GEBCO_BBOX_BUFFER_DEG) -> list[float]:
    """Pad `bbox` before reading a coarse-resolution source like GEBCO.

    Real, confirmed issue (2026-09, tile 1273): `data_catalog.get_rasterdataset
    (bbox=...)` clips to the SOURCE's own native pixel grid, not the exact
    requested bbox - for a coarse source like GEBCO (~463m pixels, vs.
    DeltaDTM's ~30m), this can fall up to half a native GEBCO pixel short of
    a tile's true edge, leaving a thin nodata sliver after
    `reproject_like(dem, ...)` even though the underlying GEBCO VRT has full
    global coverage there (confirmed directly against the raw VRT). A small
    buffer well beyond one native pixel guarantees the read window fully
    covers `bbox` after reprojection; the extra margin costs nothing since
    the read is immediately reprojected onto the DEM's own grid anyway.
    """
    return [
        max(bbox[0] - buffer_deg, -180.0),
        max(bbox[1] - buffer_deg, -90.0),
        min(bbox[2] + buffer_deg, 180.0),
        min(bbox[3] + buffer_deg, 90.0),
    ]


def resolve_offshore_mask_gaps_via_gebco(
    mask_vals: np.ndarray,
    nodata_sentinel: int,
    gebco_vals: np.ndarray,
    gebco_nodata: float,
    ocean_code: int = 1,
    land_code: int = 0,
) -> tuple[np.ndarray, dict]:
    """Reclassify `deltadtm_mask` nodata cells to ocean where GEBCO bathymetry
    and connectivity both agree it's plausible open water.

    `deltadtm_mask` coverage doesn't always extend to a tile's full bbox,
    most commonly for tiles reaching well offshore past DeltaDTM's coastal
    band. The default nodata->land fallback is safe for inland gaps but
    wrong here: a false "land" strip between the true coast and a tile's
    outer edge can starve `hydromt_sfincs`'s `create_boundary()` of almost
    its entire real coastline (it only places forcing on ocean cells
    touching the domain's rectangular perimeter).

    A nodata cell is reclassified to ocean only if GEBCO reads negative
    (plausible bathymetry) there AND it is 4-connected through other
    GEBCO-negative nodata cells to a cell the mask already confirms is real
    ocean. The connectivity requirement protects genuinely isolated
    below-sea-level inland basins (e.g. the Dead Sea, Danakil Depression -
    see `encode_dem_cm`), which also read negative in GEBCO but must stay
    land. This only helps far-offshore gaps: GEBCO's known coastal artifact
    (ocean cells reading up to +16.6m near the DeltaDTM/GEBCO transition)
    means nearshore nodata gaps typically read non-negative and correctly
    stay land.

    Parameters
    ----------
    mask_vals : np.ndarray
        Raw mask array, still containing `nodata_sentinel`.
    nodata_sentinel : int
        Value marking "no mask coverage at all".
    gebco_vals : np.ndarray
        GEBCO elevation, reprojected onto the same grid as `mask_vals`.
    gebco_nodata : float
        GEBCO's own nodata value.
    ocean_code : int, default 1
        Mask value meaning confirmed real ocean.
    land_code : int, default 0
        Mask value to fall back to when a nodata cell isn't resolved to ocean.

    Returns
    -------
    tuple of (np.ndarray, dict)
        The corrected mask array (no `nodata_sentinel` values remain) and a
        diagnostics dict with `n_resolved_ocean` and `n_still_land` counts.
    """
    is_nodata = mask_vals == nodata_sentinel
    is_ocean = mask_vals == ocean_code

    if not is_ocean.any():
        # No confirmed ocean anywhere in this tile's read window at all -
        # nothing to anchor a reclassification to. Leave every nodata cell
        # as the existing conservative land fallback.
        corrected = np.where(is_nodata, land_code, mask_vals)
        n_nodata = int(is_nodata.sum())
        return corrected, {"n_resolved_ocean": 0, "n_still_land": n_nodata}

    gebco_negative = (gebco_vals != gebco_nodata) & (gebco_vals < 0)
    candidate = is_ocean | (is_nodata & gebco_negative)

    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labels, _ = ndimage.label(candidate, structure=structure)
    anchor_labels = np.unique(labels[is_ocean])
    anchor_labels = anchor_labels[anchor_labels != 0]

    resolved_ocean = is_nodata & np.isin(labels, anchor_labels)
    corrected = np.where(resolved_ocean, ocean_code, np.where(is_nodata, land_code, mask_vals))

    return corrected, {
        "n_resolved_ocean": int(resolved_ocean.sum()),
        "n_still_land": int(is_nodata.sum() - resolved_ocean.sum()),
    }


def relabel_small_isolated_ocean_patches(
    mask_vals: np.ndarray,
    ocean_code: int = 1,
    lake_code: int = 2,
    min_cells: int = 30,
) -> tuple[np.ndarray, dict]:
    """Reclassify small ocean-coded connected components to `lake_code`.

    `deltadtm_mask` sometimes mislabels a small inland water body (e.g. a
    coastal lagoon or reservoir) as ocean rather than lake. Left uncorrected,
    `build_sfincs_tile.py::_compute_zsini_array` seeds such a patch with a
    nonzero IDW-interpolated initial water level (restricted to `ocean_code`
    cells) despite it having no hydraulic connection to the sea (found on
    SFINCS validation tile 713, where SFINCS reported it "flooded" from
    timestep zero). Uses 4-connectivity, matching
    `resolve_offshore_mask_gaps_via_gebco`'s convention; every component is
    checked purely by size, with no "except the largest" special case.

    Parameters
    ----------
    mask_vals : np.ndarray
        Mask array with ocean/land/lake/river classification.
    ocean_code : int, default 1
        Mask value meaning ocean.
    lake_code : int, default 2
        Mask value to relabel small ocean components to.
    min_cells : int, default 30
        Minimum connected-component size (cells, 4-connectivity) for an
        ocean-coded component to stay ocean; smaller components are relabeled.

    Returns
    -------
    tuple of (np.ndarray, dict)
        The corrected mask array and a diagnostics dict with `n_relabeled`
        and `n_components_relabeled` counts.
    """
    ocean = mask_vals == ocean_code
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labels, n_components = ndimage.label(ocean, structure=structure)
    if n_components == 0:
        return mask_vals, {"n_relabeled": 0, "n_components_relabeled": 0}

    sizes = ndimage.sum(ocean, labels, index=np.arange(1, n_components + 1))
    small_labels = np.nonzero(sizes < min_cells)[0] + 1
    if len(small_labels) == 0:
        return mask_vals, {"n_relabeled": 0, "n_components_relabeled": 0}

    to_relabel = np.isin(labels, small_labels)
    corrected = np.where(to_relabel, lake_code, mask_vals)
    return corrected, {"n_relabeled": int(to_relabel.sum()), "n_components_relabeled": int(len(small_labels))}


def extract_dem(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    bbox: list[float],
    mask_source: str,
    geoid_offset_raster: str | Path,
    min_hard_fill_component_size: int = 10,
    interp_max_search_distance: float = 100.0,
    interp_smoothing_iterations: int = 0,
    land_fill_value_m: float = DEM_NODATA_M,
    gebco_source: str = "gebco",
) -> xr.DataArray:
    """Clip the DEM to the model domain bbox and fill all missing cells.

    Every output cell has a real value - no nodata remains. The fill value
    is decided from `mask_source` (the DeltaDTM validity mask), reprojected
    onto the DEM's grid, not from a separately sourced land polygon that
    could misalign with DeltaDTM's own coastline:

    - Valid DeltaDTM cells: kept as-is (geoid-corrected, see
      `geoid_offset_raster`).
    - Missing land cells (mask says land, or mask has no coverage at all):
      split by the size of their 4-connected missing-land group. A SMALL
      group (< `min_hard_fill_component_size` cells) is treated as a
      plausible sensor gap and INTERPOLATED from surrounding valid
      elevation; a LARGE group is hard-filled to `land_fill_value_m`
      (99 m - above DeltaDTM's validity envelope, so it reads as real, very
      high land that never floods).
    - Missing ocean/lake/river cells: set to 0.0 regardless of gap size, so
      the flood model can propagate through permanent water rather than
      being blocked by an artificially high elevation.

    Parameters
    ----------
    data_catalog : hydromt.DataCatalog
        HydroMT data catalog containing `dem_source` and `mask_source`.
    dem_source : str
        Name of the DEM RasterDataset in `data_catalog`.
    bbox : list of float
        Model domain bounding box as ``[minx, miny, maxx, maxy]``.
    mask_source : str
        Name of the DeltaDTM validity mask RasterDataset in `data_catalog`,
        used to decide the fill value for missing DEM cells.
    geoid_offset_raster : str or Path
        Path to the cached global EGM2008 -> GOCO06s geoid-offset raster
        (see `vertical_datum.write_geoid_offset_raster`). Resampled onto
        this tile's DEM grid and added to every cell with valid DeltaDTM
        elevation only, never to fill cells.
    min_hard_fill_component_size : int, default 10
        Connected-component size (cells, 4-connectivity) at/above which a
        missing-land gap is hard-filled rather than interpolated.
    interp_max_search_distance : float, default 100.0
        `rasterio.fill.fillnodata`'s search radius (pixels) for small-gap
        interpolation.
    interp_smoothing_iterations : int, default 0
        `rasterio.fill.fillnodata`'s post-fill smoothing pass count for
        small-gap interpolation.
    land_fill_value_m : float, default DEM_NODATA_M
        Elevation (m) written for large missing-land gaps
        (simulation.dem_gap_fill.land_fill_value_m).
    gebco_source : str, default "gebco"
        Name of the GEBCO bathymetry RasterDataset in `data_catalog`, used
        to reclassify far-offshore mask nodata cells to ocean - see
        `resolve_offshore_mask_gaps_via_gebco`.

    Returns
    -------
    xr.DataArray
        The DEM clipped to `bbox`, all missing cells filled, encoded as
        int16 centimetres (see `encode_dem_cm`).
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=bbox)

    # Real, confirmed bug (2026-09, tile 1454, 82.7N): a VRT-mosaic source
    # can carry TWO distinct "missing" representations - a literal nodata
    # sentinel within a real underlying source tile, and genuine NaN
    # wherever the VRT has no underlying source file at all (a true
    # coverage gap - GDAL's own "no source covers this pixel" fill, not a
    # source tile's own choice). Every missing-data check below (this
    # function's own valid_dem/da.where, and _compute_dem_gap_fill's own
    # `== nodata`) only recognised the literal sentinel - a NaN coverage
    # gap silently skipped ALL gap-filling (no interpolation, no
    # land_fill_value_m hard-fill) and passed straight through uncaught
    # even by encode_dem_cm's own overflow-safety check (NaN comparisons
    # are always False in IEEE754, so `scaled.min() < lo`/`scaled.max() >
    # hi` never fires), landing at exactly int16(NaN) == 0 after the final
    # int16-cm cast (confirmed directly on this numpy build) - an
    # invisible, wrong "sea level" reading for what should have been
    # flagged as missing and gap-filled like any other void. Normalized to
    # the literal nodata sentinel immediately after read so every
    # downstream `== nodata` check below handles both cases identically,
    # rather than patching each call site separately.
    if np.isnan(da.values).any():
        da = da.copy(data=np.where(np.isnan(da.values), da.raster.nodata, da.values))

    from vertical_datum import sample_geoid_offset

    valid_dem = (da.values != da.raster.nodata)
    offset = sample_geoid_offset(
        geoid_offset_raster, da.raster.transform, da.raster.crs,
        da.raster.height, da.raster.width,
    ).reshape(da.shape)
    corrected = np.where(valid_dem, da.values + offset, da.values).astype(da.dtype)
    da = da.copy(data=corrected)

    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(255)  # uint8 nodata; mirrors extract_dem_mask
    mask_vals = da_mask.raster.reproject_like(da, method="nearest").values.squeeze()

    da_gebco = data_catalog.get_rasterdataset(gebco_source, bbox=_buffer_bbox_for_coarse_source(bbox))
    gebco_vals = da_gebco.raster.reproject_like(da, method="nearest").values.squeeze()
    mask_vals, _ = resolve_offshore_mask_gaps_via_gebco(
        mask_vals, 255, gebco_vals, da_gebco.raster.nodata,
    )

    # Land (0) -> irrelevant dry land. Ocean (1), lake (2), river (3) ->
    # flood-passable, fill with 0. No 255 remains - resolved above.
    is_land = mask_vals == 0

    dem_vals = da.values.reshape(mask_vals.shape)
    fill = _compute_dem_gap_fill(
        dem_vals, da.raster.nodata, is_land,
        min_hard_fill_component_size, interp_max_search_distance,
        interp_smoothing_iterations, land_fill_value_m,
    ).reshape(da.shape)

    result = da.where(da != da.raster.nodata, da.copy(data=fill))
    # int16 centimetres (well within DeltaDTM's own vertical accuracy, and
    # DeltaDTM's <=30m validity envelope fits int16's +-327.67m range with
    # huge headroom) - both halves the file size vs float32 AND avoids ever
    # writing an incompressible float mantissa (elevation varies continuously
    # almost everywhere), unlike the previous round-to-cm-but-still-float32
    # compromise.
    return result.copy(data=encode_dem_cm(result.values))


def extract_dem_mask(
    data_catalog: hydromt.DataCatalog,
    mask_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    nodata_sentinel: int = 255,
    gebco_source: str = "gebco",
) -> xr.DataArray:
    """Clip the DEM-validity mask and reproject it onto the DEM's grid.

    Reprojected (nearest-neighbour) since the mask source has a different
    native grid/resolution than the DEM. Cells with no mask coverage at all
    default to land (0), consistent with `extract_dem`'s own fill rule,
    UNLESS GEBCO bathymetry and connectivity to confirmed ocean both
    indicate the gap is actually far-offshore water - see
    `resolve_offshore_mask_gaps_via_gebco`. Valid DeltaDTM values (0 = land,
    1 = ocean, 2 = lake, 3 = river) are otherwise kept unchanged, except
    that any small ocean-coded component is relabeled to lake - see
    `relabel_small_isolated_ocean_patches`.

    Parameters
    ----------
    data_catalog : hydromt.DataCatalog
        HydroMT data catalog containing `mask_source`.
    mask_source : str
        Name of the DEM-mask RasterDataset in `data_catalog`.
    bbox : list of float
        Model domain bounding box as ``[minx, miny, maxx, maxy]``.
    dem : xr.DataArray
        The tile's DEM, as returned by `extract_dem`, used as the
        reprojection target grid.
    nodata_sentinel : int, default 255
        Value used by `mask_source` to indicate no data.
    gebco_source : str, default "gebco"
        Name of the GEBCO bathymetry RasterDataset in `data_catalog`, used
        to reclassify far-offshore mask nodata cells to ocean.

    Returns
    -------
    xr.DataArray
        The DEM-validity mask reprojected onto `dem`'s grid with no nodata cells.
    """
    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(nodata_sentinel)
    da_mask_repr = da_mask.raster.reproject_like(dem, method="nearest")

    da_gebco = data_catalog.get_rasterdataset(gebco_source, bbox=_buffer_bbox_for_coarse_source(bbox))
    gebco_vals = da_gebco.raster.reproject_like(dem, method="nearest").values.squeeze()
    mask_vals, _ = resolve_offshore_mask_gaps_via_gebco(
        da_mask_repr.values.squeeze(), nodata_sentinel, gebco_vals, da_gebco.raster.nodata,
    )
    mask_vals, _ = relabel_small_isolated_ocean_patches(mask_vals)
    return da_mask_repr.copy(data=mask_vals.reshape(da_mask_repr.shape))


def compute_friction(
    data_catalog: hydromt.DataCatalog,
    lulc_source: str,
    lookup_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    default_friction: float,
) -> xr.DataArray:
    """Clip land use, reproject it onto the DEM's grid, and convert it to friction.

    Land use classes are reclassified to Manning's n roughness coefficients
    via `lookup_source`, then converted to a per-METRE friction value via
    `roughness / 100`. Cells with no valid land use classification are
    filled with `default_friction` (already in this per-metre convention).

    This raster excludes the grid-resolution correction the eikonal solver
    needs (its discretisation has no distance term, so it needs
    cost-per-GRID-STEP, not cost-per-metre) - that is applied separately at
    runtime via `simulation.flooding.friction_scale_factor`, so this
    expensive land-use raster is computed once and reused unscaled across
    every sweep that varies a different parameter.

    Parameters
    ----------
    data_catalog : hydromt.DataCatalog
        HydroMT data catalog containing `lulc_source` and `lookup_source`.
    lulc_source : str
        Name of the land use RasterDataset in `data_catalog`.
    lookup_source : str
        Name of the land-use-to-roughness lookup DataFrame in `data_catalog`.
        Must have an index named `copernicus_worldcover` and a `manning_n`
        column.
    bbox : list of float
        Tile bounding box as ``[minx, miny, maxx, maxy]``.
    dem : xr.DataArray
        The tile's DEM, as returned by `extract_dem`, used as the
        reprojection target grid.
    default_friction : float
        Friction value written for cells with no valid land use
        classification (e.g. ocean, outside LULC coverage). Should match the
        Aqueduct core default (flooding.default_friction in config).

    Returns
    -------
    xr.DataArray
        A friction raster on `dem`'s grid with no nodata cells, encoded as
        int16 (see `encode_friction_int16`).
    """
    _MISSING = np.float32(-9999.0)  # internal sentinel for unclassified cells

    # Copernicus land-use coverage doesn't reach the poles (cuts off around
    # 80N) - a tile whose bbox extends beyond that has NO real land-use data
    # at all. HydroMT surfaces this two different ways depending on exactly
    # how little coverage remains: either get_rasterdataset itself raises
    # NoDataException (confirmed directly: tile 234, Ellesmere Island,
    # ~80-81N - zero pixels read at all), or it returns successfully but the
    # clipped da_lulc ends up with fewer than 2 cells in one spatial dim
    # (reproject_like then needs >=2 cells in each dim to compute a valid
    # affine transform, raising "Invalid raster: less than 2 cells in y_dim
    # ..." otherwise). Both are the same underlying condition - no valid
    # land use classification anywhere in this tile - so both fall back the
    # same way every individual unclassified cell already does below, using
    # `dem` itself as the coordinate/CRS template since it's already on the
    # exact target grid.
    try:
        da_lulc = data_catalog.get_rasterdataset(lulc_source, bbox=bbox)
    except NoDataException:
        da_lulc = None

    if da_lulc is None or da_lulc.raster.height < 2 or da_lulc.raster.width < 2:
        friction = np.full((dem.raster.height, dem.raster.width), default_friction, dtype=np.float32)
        return dem.copy(data=encode_friction_int16(friction))

    da_lulc_repr = da_lulc.raster.reproject_like(dem, method="mode")
    da_lulc_repr = da_lulc_repr.where(da_lulc_repr != da_lulc_repr.attrs["_FillValue"], _MISSING)

    df_map = data_catalog.get_dataframe(lookup_source)
    df_map = df_map.set_index("copernicus_worldcover")
    df_map = df_map.rename(columns={"manning_n": "N"})
    df_map["N"] = df_map["N"].replace(-999.000, _MISSING)

    max_code = int(df_map.index.max())
    lookup = np.full(max_code + 1, _MISSING, dtype="float32")
    lookup[df_map.index.to_numpy(dtype=int)] = df_map["N"].to_numpy(dtype="float32")

    lulc = da_lulc_repr.values
    valid = np.isfinite(lulc) & (lulc >= 0) & (lulc <= max_code)
    roughness = np.full(lulc.shape, _MISSING, dtype="float32")
    roughness[valid] = lookup[lulc[valid].astype(int)]

    friction = np.where(roughness != _MISSING, roughness / 100, default_friction).astype(np.float32)
    return da_lulc_repr.copy(data=encode_friction_int16(friction))


def _atomic_raster_write(output_path: str | Path, profile: dict[str, Any], write_fn) -> None:
    """Write a raster via a temp file in the same directory + atomic
    `os.replace`, so a process killed mid-write (SIGKILL from a hard
    `scancel`, node failure/preemption, OOM-killer) never leaves a
    truncated/corrupted file sitting at the real `output_path` - see
    conversation 2026-08-11. Either `output_path` is untouched (whatever
    was there before, or nothing) or it's the complete, valid result -
    there is no partially-written state visible at that path, ever.

    `run_aqueduct_cli.py`'s own `_output_already_done` check already
    treats a truncated file as "not done" (it tries to actually read it),
    so a hard kill was already safe in the sense that the work would get
    correctly redone rather than silently accepted as complete - this is
    the complementary fix that avoids ever producing that truncated file
    at the canonical path in the first place, rather than only detecting
    it after the fact.

    The temp filename includes the PID so two processes racing to (re)write
    the SAME output_path concurrently (e.g. an old and a resumed dispatch
    briefly overlapping) never clobber each other's in-progress temp file -
    each gets its own, and only ONE `os.replace` call ultimately wins,
    same as a normal same-content race.

    `write_fn(dst)` performs the actual `dst.write(...)` call(s) - passed in
    so this can wrap save_raster/save_nodata_raster/save_waterdepth_raster
    without duplicating rasterio.open's `with`/exception handling three
    times.
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp{os.getpid()}")
    try:
        with retry_transient_io(rasterio.open, tmp_path, "w", **profile) as dst:
            write_fn(dst)
        retry_transient_io(os.replace, tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_raster(
    da: xr.DataArray,
    output_path: str | Path,
    raster_config: dict[str, Any],
    dtype: str = "float32",
) -> None:
    """Save a DataArray as a raster file using the workflow's raster output settings.

    Writes via a temp file + atomic rename (see `_atomic_raster_write`), and
    creates `output_path`'s parent directory if missing. Tiled explicitly
    (512x512 blocks) regardless of `raster_config["driver"]`, since plain
    GTiff (unlike COG) defaults to striped layout, and downstream code
    (merge.py's per-block reads, tile_split.py's windowed land-pixel count)
    reads small rectangular windows, not full scanlines. For an integer
    `dtype`, the predictor is forced to 1 (none) regardless of
    `raster_config["predictor"]`, since predictor 2 was found empirically to
    be worse than no predictor for this pipeline's integer rasters under zstd.

    Parameters
    ----------
    da : xr.DataArray
        The data array to save. Must have `da.raster.crs`, `da.raster.transform`,
        `da.raster.width` and `da.raster.height` accessors (provided by
        rioxarray via HydroMT's raster accessor).
    output_path : str or Path
        Destination file path.
    raster_config : dict
        The workflow's `raster_format` configuration section, with keys
        `driver`, `compression`, `predictor` and `nodata`.
    dtype : str, default "float32"
        Output data type.
    """
    # raster_config["predictor"] (3) is GDAL's floating-point predictor -
    # correct for the float32 rasters this config was written for, but wrong
    # for integer dtypes. Empirically (see conversation 2026-08-01, real DEM
    # and friction tiles) predictor 1 (none) beats predictor 2 (horizontal
    # differencing) on BOTH file size and read/write speed for this
    # pipeline's actual data - zstd's own entropy coding already captures
    # the redundancy in these rasters better than horizontal differencing
    # does, unlike the conventional "always use predictor 2 for integers"
    # guidance (which is more reliably true for older codecs like LZW).
    predictor = 1 if np.dtype(dtype).kind in "iu" else raster_config["predictor"]
    # Snakemake auto-creates an output:'s parent directory before running a
    # rule's script - but this function is also called from
    # scripts/run_aqueduct_cli.py, a plain standalone CLI (not a Snakemake
    # script:), which gets no such help. Explicit here so every caller is
    # correct regardless of how it's invoked - found live 2026-08-10 (real
    # HPC run: RasterioIOError "No such file or directory" writing
    # waterdepth_*.tif for tiles whose results/ dir had never been created,
    # since only preprocessing's inputs/ dir existed yet).
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    profile = {
        "driver": raster_config["driver"],
        "crs": da.raster.crs,
        "transform": da.raster.transform,
        "dtype": dtype,
        "count": 1,
        "nodata": raster_config["nodata"],
        "compress": raster_config["compression"],
        "predictor": predictor,
        "width": da.raster.width,
        "height": da.raster.height,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    _atomic_raster_write(output_path, profile, lambda dst: dst.write(da.values, indexes=1))


def save_nodata_raster(reference_path: str | Path, output_path: str | Path, raster_config: dict[str, Any]) -> None:
    """Write an all-nodata waterdepth raster on the same grid as `reference_path`.

    Used as a placeholder `waterdepth` output for tiles that are skipped
    instead of being run through Aqueduct (see
    `aqueduct_runner.log_skipped_tile`). Filled with `WATERDEPTH_NODATA_INT16`
    so `merge.merge_tile_rasters_chunk` ignores this tile entirely when
    merging results. Writes via a temp file + atomic rename (see
    `_atomic_raster_write`), and creates `output_path`'s parent directory if
    missing.

    Parameters
    ----------
    reference_path : str or Path
        Path to a raster (e.g. the tile's DEM) whose grid (transform, CRS,
        width, height) the output should match.
    output_path : str or Path
        Destination file path.
    raster_config : dict
        The workflow's `raster_format` configuration section, with keys
        `driver` and `compression` (predictor is fixed at 1).
    """
    with retry_transient_io(rasterio.open, reference_path) as ref:
        profile = ref.profile

    profile.update(
        driver=raster_config["driver"],
        dtype="int16",
        count=1,
        compress=raster_config["compression"],
        predictor=1,
        nodata=WATERDEPTH_NODATA_INT16,
    )
    data = np.full((profile["height"], profile["width"]), WATERDEPTH_NODATA_INT16, dtype="int16")
    # See save_raster's own comment on why mkdir is needed here (called from
    # the standalone run_aqueduct_cli.py, not just Snakemake rules), and
    # _atomic_raster_write's own docstring for why the write itself goes
    # through a temp file + atomic replace.
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    _atomic_raster_write(output_path, profile, lambda dst: dst.write(data, indexes=1))


def save_waterdepth_raster(
    reference_path: str | Path, waterdepth: np.ndarray, output_path: str | Path,
) -> None:
    """Write a genuinely-computed `waterdepth` raster from `flood_model.flood_depth_dense`.

    Encoded as int16 centimetres (see `encode_waterdepth_cm`) - real flood
    depths never approach int16's ceiling, so this is a pure I/O win with no
    meaningful precision loss. `merge.merge_tile_rasters_chunk` decodes this
    format via `merge.decode_waterdepth_array`. Writes via a temp file +
    atomic rename (see `_atomic_raster_write`), and creates `output_path`'s
    parent directory if missing.

    Parameters
    ----------
    reference_path : str or Path
        Path to a raster (e.g. the tile's DEM) whose grid (transform, CRS,
        width, height) the output should match.
    waterdepth : np.ndarray
        The computed water depth array (metres), same shape as the
        reference raster, `0.0` where not flooded.
    output_path : str or Path
        Destination file path.
    """
    with retry_transient_io(rasterio.open, reference_path) as ref:
        transform, crs = ref.transform, ref.crs
        height, width = ref.height, ref.width

    encoded = encode_waterdepth_cm(waterdepth)
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "count": 1,
        "nodata": WATERDEPTH_NODATA_INT16,
        "compress": "zstd",
        "predictor": 1,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    # See save_raster's own comment on why mkdir is needed here (called from
    # the standalone run_aqueduct_cli.py, not just Snakemake rules) - this
    # is the exact call site that hit the real 2026-08-10 HPC failure - and
    # _atomic_raster_write's own docstring for why the write itself goes
    # through a temp file + atomic replace (2026-08-11: protects against a
    # hard `scancel`/node failure leaving a truncated waterdepth_*.tif
    # behind, the most expensive output in this pipeline to silently lose).
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    _atomic_raster_write(output_path, profile, lambda dst: dst.write(encoded, indexes=1))


def average_pool_to_grid(
    numerator: np.ndarray,
    domain: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    numerator_nodata: float,
) -> np.ndarray:
    """Compute the area-weighted fraction of each destination cell belonging to a class.

    Disaggregates from a fine-resolution source grid whose valid domain may
    only partially cover a given destination cell, using the identity
    ``ff = A x B`` where ``A = reproject(numerator, average, nodata
    excluded) = sum(class) / n_in_domain`` and ``B = reproject(domain,
    average) = n_in_domain / n_total``, so ``A x B = sum(class) / n_total``
    and `NaN` propagates correctly for zero-coverage cells. Runs as two
    SEPARATE `reproject()` calls, never combined into one multi-band call:
    `reproject()` shares a single validity mask across bands under
    `Resampling.average`, so a combined call silently corrupts `domain`'s
    average wherever `numerator` is nodata (confirmed on real production
    data: ~360,000 coarse cells differed from the two-call version).

    Parameters
    ----------
    numerator : np.ndarray
        Fine-resolution array, `1.0` where a fine cell belongs to the class
        of interest, `0.0` where in-domain but not in the class, and
        `numerator_nodata` only outside the valid domain.
    domain : np.ndarray
        Fine-resolution array, `1.0` where a fine cell is inside the valid
        domain (regardless of class), `0.0` outside. No nodata.
    src_transform : affine.Affine
        Affine transform of `numerator`/`domain`'s source grid.
    src_crs : rasterio.crs.CRS
        CRS of the source grid.
    dst_transform : affine.Affine
        Affine transform of the destination grid.
    dst_crs : rasterio.crs.CRS
        CRS of the destination grid.
    dst_shape : tuple of int
        Destination array shape as ``(height, width)``.
    numerator_nodata : float
        Nodata value in `numerator`, marking cells outside the valid domain.

    Returns
    -------
    np.ndarray
        `dst_shape` array: `NaN` where a destination cell has zero domain
        coverage, otherwise the fraction of that cell's in-domain fine cells
        belonging to the class (``sum(class) / n_total``).
    """
    out_h, out_w = dst_shape
    dst_kwargs = dict(
        dst_transform=dst_transform, dst_crs=dst_crs,
        dst_nodata=np.nan, resampling=Resampling.average,
    )

    frac_a = np.full((out_h, out_w), np.nan, dtype="float32")
    reproject(
        source=numerator, destination=frac_a,
        src_transform=src_transform, src_crs=src_crs,
        src_nodata=numerator_nodata, **dst_kwargs,
    )

    frac_b = np.full((out_h, out_w), np.nan, dtype="float32")
    reproject(
        source=domain, destination=frac_b,
        src_transform=src_transform, src_crs=src_crs,
        src_nodata=None, **dst_kwargs,
    )

    return frac_a * frac_b
