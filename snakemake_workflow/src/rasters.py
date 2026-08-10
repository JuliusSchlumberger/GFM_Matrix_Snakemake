"""Raster extraction and processing functions for a single tile.

Tiles in the tile grid (see `src/tile_chunking.py`) are exact rectangles,
so a tile's bounding box is identical to its geometry.
This means raster datasets can simply be clipped with `bbox=...` - no
additional `geometry_mask` step is required.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import hydromt
import numpy as np
import rasterio
import xarray as xr
from hydromt.exceptions import NoDataException
from rasterio.fill import fillnodata
from scipy import ndimage
from shapely.geometry import box as shapely_box

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

    DEM_VALUE_CLIPPED_TO_INT16_RANGE (2026-08-10, search this token to find
    this note again): values below DeltaDTM's usual <=30m envelope can
    still be real - confirmed directly against two real preprocessing
    failures: tile 1464 (the Dead Sea, -369.29m) and tile 303 (the Danakil
    Depression / Red Sea rift, -379.08m) - but int16-cm only spans
    +-327.67m, so both genuinely overflow it. CLIPPED here, not rejected:
    this is a deliberate, lossy alteration of the true DEM value for any
    cell this extreme, on the reasoning that once an elevation is this far
    below any plausible coastal water level, the exact number cannot affect
    flood results - "-369m" and "-327.67m" both mean "never floods," so no
    flood-extent difference is expected at either location. Flagged here
    (and printed at runtime below) specifically so this is easy to find
    again if that assumption ever needs revisiting - e.g. if a future
    version of this pipeline computes anything OTHER than coastal flood
    extent from this DEM (elevation-dependent exposure metrics, terrain
    analysis, etc.) where the true (deeper) value would matter.
    _encode_int16 itself is untouched - friction/waterdepth encoding still
    raise a hard error on overflow; only this DEM-specific caller clips.
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
    """Decode an int16-centimetre DEM back to float32 metres."""
    return dem_int16.astype(np.float32) / DEM_SCALE


def encode_friction_int16(friction: np.ndarray) -> np.ndarray:
    """Encode a float friction raster as int16 (x FRICTION_SCALE)."""
    return _encode_int16(friction, FRICTION_SCALE, "Friction", "within the land-cover lookup table's Manning's n / 100 range")


def decode_friction_int16(friction_int16: np.ndarray) -> np.ndarray:
    """Decode an int16 friction raster back to float32.

    Divides by `np.float32(FRICTION_SCALE)`, not the raw Python int - numpy's
    legacy value-based casting promotes `float32_array / python_int` to
    float64 once the int needs more than 16 bits to represent exactly
    (`FRICTION_SCALE = 1_000_000` does; `DEM_SCALE`/`WATERDEPTH_SCALE`/
    `WATERLEVEL_SCALE` at 100 don't, which is why only this one decoder was
    silently affected). This one array (`friction`) sets `solve_eikonal_
    dense`'s `dtype`, which every subsequent large array in a flood solve
    inherits (`t`, in particular, the dominant allocation) - the bug was
    silently doubling the peak memory of every "python" engine flood solve,
    coupling or not, until caught investigating OOM failures during 2026-08
    obstacle-coupling calibration on large real tiles.
    """
    return friction_int16.astype(np.float32) / np.float32(FRICTION_SCALE)


def encode_waterdepth_cm(waterdepth_m: np.ndarray) -> np.ndarray:
    """Encode a float water-depth raster (metres, 0.0 where dry) as int16
    centimetres. Raises if any value would collide with
    WATERDEPTH_NODATA_INT16 (implausible for a real flood depth).
    """
    encoded = _encode_int16(waterdepth_m, WATERDEPTH_SCALE, "Water depth", "well below the int16 'not computed' sentinel")
    if (encoded == WATERDEPTH_NODATA_INT16).any():
        raise ValueError(
            f"Water depth reaches {WATERDEPTH_NODATA_INT16 / WATERDEPTH_SCALE}m, "
            "colliding with the int16 'not computed' sentinel - implausible for a real flood depth."
        )
    return encoded


def decode_waterdepth_cm(waterdepth_int16: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre water-depth raster to float32 metres, with
    WATERDEPTH_NODATA_INT16 mapped to `merge.AQUEDUCT_NODATA` so the
    existing float32 merge/overlap-correction pipeline needs no changes of
    its own.
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

    Boundary GeoPackages can legitimately have zero rows (a tile with no
    findable COAST-RP station - see scripts/extract_boundaries.py's empty-
    placeholder handling) - returned as an empty int16 array as-is, rather
    than going through `_encode_int16`'s min/max range check, which raises
    on an empty array.
    """
    if waterlevel_m.size == 0:
        return waterlevel_m.astype(np.int16)
    return _encode_int16(
        waterlevel_m, WATERLEVEL_SCALE, "Boundary water level",
        "within COAST-RP storm-tide + SLR fingerprint's realistic range",
    )


def decode_waterlevel_cm(waterlevel_int16: np.ndarray) -> np.ndarray:
    """Decode an int16-centimetre boundary water level array to float32 metres."""
    return waterlevel_int16.astype(np.float32) / WATERLEVEL_SCALE


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as an xarray DataArray with hydromt raster accessor."""
    import rioxarray  # noqa: F401 - registers the rasterio xarray backend and spatial metadata
    return retry_transient_io(xr.open_dataarray, path, engine="rasterio").squeeze("band", drop=True)


def get_tile_bbox(tile: gpd.GeoDataFrame) -> list[float]:
    """Return the [minx, miny, maxx, maxy] bounding box of a single-row tile GeoDataFrame."""
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
    land without DeltaDTM coverage.  This function shrinks the domain to the
    bounding box of flood-relevant DeltaDTM cells expanded by ``buffer_arcsec``
    on all four sides, then clips the result to ``tile_bbox``.

    A cell counts as flood-relevant (kept in the domain) unless it is
    *confirmed* high land: ocean/lake/river cells (mask != land) always
    count, regardless of what the raw DEM says there (DeltaDTM routinely has
    no elevation reading at all over water - that's expected, not a gap -
    see rasters.extract_dem's own land/water fill split). A land cell only
    drops out when its raw DeltaDTM elevation is both present and
    ``>= elevation_threshold_m`` - land with no elevation reading (a gap
    extract_dem will later either interpolate or hard-fill, depending on gap
    size) is conservatively kept here rather than trying to replicate that
    gap-size logic a second time. This mirrors the obstacle-coupling static
    pre-filter's own ``dem > max_waterlevel`` reasoning (definitively-dry
    terrain doesn't need to be part of the domain), just applied at
    domain-selection time instead of solve time - and the elevation
    threshold should be the same value the tile-grid construction pipeline
    uses for its own wide-gap/safe-cut detection (``tile_grid.
    elevation_relevance_threshold_m`` - see tiles.py), so a tile's recorded
    footprint and its actually-extracted domain agree on what counts as
    irrelevant.

    If ``elevation_threshold_m`` is ``None``, falls back to the original,
    coarser behaviour: any real (non-nodata) DEM cell counts as valid,
    land/water split not considered at all.

    DeltaDTM covers right to the shoreline, so a buffer of even 1 arc-second
    (~30 m) is enough to include a thin strip of ocean cells on the seaward
    edge, giving the Aqueduct model the coastal entry points it needs for flood
    propagation.  Increase ``buffer_arcsec`` (via ``simulation.model_bbox_buffer_arcsec``
    in config) if coastal tiles show incomplete flood entry.

    If the DEM has no valid cells within ``tile_bbox``, the full ``tile_bbox``
    is returned unchanged.

    Args:
        data_catalog: HydroMT data catalog.
        dem_source: Catalog key for the DEM (DeltaDTM) RasterDataset.
        mask_source: Catalog key for the DeltaDTM validity mask RasterDataset.
            Only read when ``elevation_threshold_m`` is given.
        tile_bbox: Full tile bounding box as ``[minx, miny, maxx, maxy]``.
        buffer_arcsec: Buffer added to each side of the DEM extent, in
            arc-seconds.
        elevation_threshold_m: Elevation (m) at/above which confirmed land is
            excluded from the domain. ``None`` (default) disables the
            elevation criterion entirely (mask/land-vs-water is not
            considered either, matching the pre-2026-08 behaviour).

    Returns:
        Reduced bounding box ``[minx, miny, maxx, maxy]``, clipped to ``tile_bbox``.
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
) -> xr.DataArray:
    """Clip the DEM to the model domain bbox and fill all missing cells.

    Every cell in the output has a real value — no nodata cells remain. The
    fill value is decided entirely from `mask_source` (the DeltaDTM validity
    mask), reprojected onto the DEM's own grid — not from any separately
    sourced land polygon dataset, which can misalign with DeltaDTM's own
    coastline:
    - Cells with valid DeltaDTM elevation: kept as-is (optionally geoid-
      corrected first, see `geoid_offset_raster` below).
    - Missing cells where the mask says land (0), OR where the mask itself
      has no coverage at all (nodata): split by the size of the connected
      (4-connectivity) group of missing-land cells they belong to.
        - SMALL group (< `min_hard_fill_component_size` cells, e.g. an
          isolated pixel or a small cluster): treated as a plausible
          measurement/sensor gap and INTERPOLATED from the surrounding
          valid DeltaDTM elevation (`rasterio.fill.fillnodata`, inverse-
          distance weighted), rather than hard-filled - a lone nodata pixel
          surrounded by real elevation is very unlikely to be exactly
          9999 m in reality.
        - LARGE group (>= `min_hard_fill_component_size` cells): set to
          `land_fill_value_m` (99 m - comfortably above DeltaDTM's <=30m
          validity envelope, so it reads as real, very high elevation and
          never floods, while still fitting the int16-centimetre output
          encoding) - genuinely unmeasured terrain (the common case for land
          nodata), not a small gap worth interpolating over.
      Areas outside the mask's own coverage are treated the same as known
      land throughout this split - irrelevant, definitely-dry terrain if
      the gap is large, an interpolation candidate if it's small.
    - Missing cells where the mask says ocean (1), lake (2), or river (3):
      set to 0.0, regardless of gap size. Julia re-zeros ocean cells anyway
      (``dem[.!landmask] .= 0.0``); lake/river cells are permanent inland
      water bodies where DeltaDTM has no terrain elevation, and filling with
      0 lets the flood model propagate through them rather than blocking it
      with an artificially high land elevation.

    Args:
        data_catalog: HydroMT data catalog containing `dem_source` and `mask_source`.
        dem_source: Name of the DEM RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        mask_source: Name of the DeltaDTM validity mask RasterDataset in
            `data_catalog`, used to decide the fill value for missing DEM cells.
        geoid_offset_raster: Path to the cached global EGM2008 -> GOCO06s
            geoid-offset raster (see vertical_datum.write_geoid_offset_raster
            / rule compute_geoid_offset_raster). Always applied: resampled
            onto this tile's own DEM grid and ADDED to every cell with valid
            DeltaDTM elevation - never to the fill cells above, which are not
            real elevations. See src/vertical_datum.py for the full reasoning.
        min_hard_fill_component_size: Connected-component size (in cells,
            4-connectivity) at/above which a missing-land gap is hard-filled
            to `land_fill_value_m` rather than interpolated. Default 10 (i.e. a
            cell connected to at least 9 other missing-land cells).
        interp_max_search_distance: `rasterio.fill.fillnodata`'s search
            radius (pixels) for small-gap interpolation.
        interp_smoothing_iterations: `rasterio.fill.fillnodata`'s post-fill
            smoothing pass count for small-gap interpolation.
        land_fill_value_m: Elevation (m) written for large missing-land gaps
            (simulation.dem_gap_fill.land_fill_value_m).

    Returns:
        The DEM clipped to `bbox`, all missing cells filled, encoded as
        int16 centimetres (see `encode_dem_cm` - DeltaDTM's <=30m validity
        envelope fits int16-cm with enormous headroom).
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=bbox)

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

    # Land (0) or no mask coverage at all (255) -> irrelevant dry land.
    # Ocean (1), lake (2), river (3) -> flood-passable, fill with 0.
    is_land = (mask_vals == 0) | (mask_vals == 255)

    dem_vals = da.values.reshape(mask_vals.shape)
    missing = dem_vals == da.raster.nodata
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
        interpolated = np.where(interpolated == da.raster.nodata, land_fill_value_m, interpolated)
    else:
        interpolated = dem_vals  # never read below (small_gap is all-False)

    fill = np.where(
        is_land,
        np.where(small_gap, interpolated, land_fill_value_m),
        0.0,
    ).reshape(da.shape).astype(np.float32)

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
) -> xr.DataArray:
    """Clip the DEM-validity mask and reproject it onto the DEM's grid.

    The mask source raster has a different native grid/resolution than the
    DEM, so it is reprojected (nearest-neighbour) onto the DEM's grid to
    ensure matching dimensions, as required by the flood model. Cells with no
    mask coverage at all (value == `nodata_sentinel`) are set to land (0) —
    consistent with `extract_dem`'s own fill rule: areas outside DeltaDTM's
    own coverage are irrelevant, definitely-dry terrain, not an unknown for a
    separately sourced land polygon dataset to arbitrate. Valid DeltaTM
    values (0 = land, 1 = ocean, 2 = lake, 3 = river) are kept unchanged.

    Args:
        data_catalog: HydroMT data catalog containing `mask_source`.
        mask_source: Name of the DEM-mask RasterDataset in `data_catalog`.
        bbox: Model domain bounding box as ``[minx, miny, maxx, maxy]``.
        dem: The tile's DEM, as returned by `extract_dem`, used as the
            reprojection target grid.
        nodata_sentinel: Value used by `mask_source` to indicate no data.

    Returns:
        The DEM-validity mask reprojected onto `dem`'s grid with no nodata cells.
    """
    da_mask = data_catalog.get_rasterdataset(mask_source, bbox=bbox)
    da_mask.raster.set_nodata(nodata_sentinel)
    da_mask_repr = da_mask.raster.reproject_like(dem, method="nearest")
    return da_mask_repr.where(da_mask_repr != nodata_sentinel, 0)


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
    using the `lookup_source` lookup table, then converted to a friction
    value via `roughness / 100` (empirical formula). Cells with no valid
    land use classification are filled with `default_friction`.

    Args:
        data_catalog: HydroMT data catalog containing `lulc_source` and `lookup_source`.
        lulc_source: Name of the land use RasterDataset in `data_catalog`.
        lookup_source: Name of the land-use-to-roughness lookup DataFrame in
            `data_catalog`. Must have an index named `copernicus_worldcover`
            and a `manning_n` column.
        bbox: Tile bounding box as `[minx, miny, maxx, maxy]`.
        dem: The tile's DEM, as returned by `extract_dem`, used as the
            reprojection target grid.
        default_friction: Friction value written for cells with no valid land
            use classification (e.g. ocean, outside LULC coverage). Should
            match the Aqueduct core default (flooding.default_friction in config).

    Returns:
        A friction raster on `dem`'s grid with no nodata cells, encoded as
        int16 (see `encode_friction_int16` - real land-cover-derived values
        comfortably fit int16 x FRICTION_SCALE).
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


def save_raster(
    da: xr.DataArray,
    output_path: str | Path,
    raster_config: dict[str, Any],
    dtype: str = "float32",
) -> None:
    """Save a DataArray as a raster file using the workflow's raster output settings.

    Tiled explicitly (512x512 blocks) regardless of `raster_config["driver"]`, since
    plain GTiff (unlike COG) defaults to striped layout - tiling matters here because
    downstream code (merge.py's per-block reads, tile_split.py's windowed land-pixel
    count) reads small rectangular windows, not full scanlines.

    Args:
        da: The data array to save. Must have `da.raster.crs`, `da.raster.transform`,
            `da.raster.width` and `da.raster.height` accessors (provided by rioxarray
            via HydroMT's raster accessor).
        output_path: Destination file path.
        raster_config: The workflow's `raster_format` configuration section,
            with keys `driver`, `compression`, `predictor` and `nodata`.
        dtype: Output data type.
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
    with retry_transient_io(
        rasterio.open,
        output_path,
        "w",
        driver=raster_config["driver"],
        crs=da.raster.crs,
        transform=da.raster.transform,
        dtype=dtype,
        count=1,
        nodata=raster_config["nodata"],
        compress=raster_config["compression"],
        predictor=predictor,
        width=da.raster.width,
        height=da.raster.height,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(da.values, indexes=1)


def save_nodata_raster(reference_path: str | Path, output_path: str | Path, raster_config: dict[str, Any]) -> None:
    """Write an all-nodata waterdepth raster on the same grid as `reference_path`.

    Used as a placeholder `waterdepth` output for tiles that are skipped
    instead of being run through Aqueduct (see
    `aqueduct_runner.log_skipped_tile`). Filled with `WATERDEPTH_NODATA_INT16`
    (the int16 counterpart of `merge.AQUEDUCT_NODATA` - see
    `decode_waterdepth_cm`, which maps it back to `AQUEDUCT_NODATA` at merge
    time), so `merge.merge_tile_rasters_chunk` ignores this tile entirely
    when merging results.

    Args:
        reference_path: Path to a raster (e.g. the tile's DEM) whose grid
            (transform, CRS, width, height) the output should match.
        output_path: Destination file path.
        raster_config: The workflow's `raster_format` configuration section,
            with keys `driver` and `compression` (predictor is fixed at 1 -
            see `save_raster`'s note on why that beats 2 empirically here).
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
    # See save_raster's own comment on why this is needed here (called from
    # the standalone run_aqueduct_cli.py, not just Snakemake rules).
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    with retry_transient_io(rasterio.open, output_path, "w", **profile) as dst:
        dst.write(data, indexes=1)


def save_waterdepth_raster(
    reference_path: str | Path, waterdepth: np.ndarray, output_path: str | Path,
) -> None:
    """Write a genuinely-computed `waterdepth` raster from `flood_model.flood_depth_dense`.

    int16 centimetres (see `encode_waterdepth_cm`) - real flood depths never
    approach int16's ceiling, so this is a pure I/O win with no meaningful
    precision loss (see conversation 2026-08-01). `merge.
    merge_tile_rasters_chunk` decodes this format via `merge.
    decode_waterdepth_array`.

    Args:
        reference_path: Path to a raster (e.g. the tile's DEM) whose grid
            (transform, CRS, width, height) the output should match.
        waterdepth: The computed water depth array, same shape as the
            reference raster, `0.0` where not flooded.
        output_path: Destination file path.
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
    # See save_raster's own comment on why this is needed here (called from
    # the standalone run_aqueduct_cli.py, not just Snakemake rules) - this
    # is the exact call site that hit the real 2026-08-10 HPC failure.
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    with retry_transient_io(rasterio.open, output_path, "w", **profile) as dst:
        dst.write(encoded, indexes=1)
