"""Extract, encode and write the per-tile rasters (DEM, mask, friction, water depth).

Tiles are exact rectangles, so clipping with `bbox=...` needs no extra geometry mask.
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
from affine import Affine
from rasterio.features import rasterize
from rasterio.fill import fillnodata
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from config_utils import retry_transient_io

from merge import EIKONAL_NODATA

# DeltaDTM is valid up to 30 m, so int16 cm (+-327.67 m) has ample headroom; the
# 99 m gap-fill sentinel sits above any real value and inside int16's range.
DEM_SCALE = 100  # cm per metre
DEM_NODATA_M = 99.0
DEM_NODATA_INT16 = int(DEM_NODATA_M * DEM_SCALE)  # 9900

# Friction (Manning's n / 100) spans ~0.0001-0.0012; x1e6 keeps n's 2 significant
# digits with ~27x headroom below int16's max.
FRICTION_SCALE = 1_000_000

# int16 max doubles as the "not computed" sentinel (int16 twin of merge.EIKONAL_NODATA).
WATERDEPTH_SCALE = 100  # cm per metre
WATERDEPTH_NODATA_INT16 = np.iinfo(np.int16).max  # 32767


def _encode_int16(values: np.ndarray, scale: float, label: str, valid_hint: str) -> np.ndarray:
    # Raise on overflow: numpy's int16 cast wraps silently.
    scaled = np.round(values.astype(np.float64) * scale)
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if scaled.min() < lo or scaled.max() > hi:
        raise ValueError(
            f"{label} value out of int16 range after x{scale:g} scaling: "
            f"[{scaled.min() / scale:g}, {scaled.max() / scale:g}] (expected {valid_hint})"
        )
    return scaled.astype(np.int16)


def encode_dem_cm(dem_m: np.ndarray) -> np.ndarray:
    """Encode a float DEM (metres) as int16 centimetres, clipping to int16's range.

    DEM_VALUE_CLIPPED_TO_INT16_RANGE: real depressions beyond -327.68 m (Dead Sea,
    Danakil Depression) are clipped, not rejected, since they never flood either way.
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
    """Decode an int16-centimetre DEM to float32 metres."""
    return dem_int16.astype(np.float32) / DEM_SCALE


def encode_friction_int16(friction: np.ndarray) -> np.ndarray:
    """Encode per-metre friction (Manning's n / 100) as int16 x `FRICTION_SCALE`."""
    return _encode_int16(friction, FRICTION_SCALE, "Friction", "within the land-cover lookup table's Manning's n / 100 range")


def decode_friction_int16(friction_int16: np.ndarray) -> np.ndarray:
    """Decode an int16 friction raster to float32 (per-metre convention).

    Divides by `np.float32(FRICTION_SCALE)`: a plain int divisor > 16 bits makes
    numpy promote to float64, doubling the solver's peak memory.
    """
    return friction_int16.astype(np.float32) / np.float32(FRICTION_SCALE)


def encode_waterdepth_cm(waterdepth_m: np.ndarray) -> np.ndarray:
    """Encode water depth (metres, 0 where dry) as int16 centimetres.

    Raises ValueError if a value would collide with `WATERDEPTH_NODATA_INT16`.
    """
    encoded = _encode_int16(waterdepth_m, WATERDEPTH_SCALE, "Water depth", "well below the int16 'not computed' sentinel")
    if (encoded == WATERDEPTH_NODATA_INT16).any():
        raise ValueError(
            f"Water depth reaches {WATERDEPTH_NODATA_INT16 / WATERDEPTH_SCALE}m, "
            "colliding with the int16 'not computed' sentinel - implausible for a real flood depth."
        )
    return encoded


def decode_waterdepth_cm(waterdepth_int16: np.ndarray) -> np.ndarray:
    """Decode int16-cm water depth to float32 metres; sentinel -> `merge.EIKONAL_NODATA`."""
    decoded = waterdepth_int16.astype(np.float32) / WATERDEPTH_SCALE
    decoded[waterdepth_int16 == WATERDEPTH_NODATA_INT16] = EIKONAL_NODATA
    return decoded


# Boundary water levels (COAST-RP + SLR) in boundaries.gpkg use the same int16-cm
# precision as the DEM they are compared against when seeding.
WATERLEVEL_SCALE = 100  # cm per metre


def encode_waterlevel_cm(waterlevel_m: np.ndarray) -> np.ndarray:
    """Encode boundary water levels (metres) as int16 centimetres.

    An empty input (tile with no COAST-RP station) is returned as an empty int16 array.
    """
    if waterlevel_m.size == 0:
        return waterlevel_m.astype(np.int16)
    return _encode_int16(
        waterlevel_m, WATERLEVEL_SCALE, "Boundary water level",
        "within COAST-RP storm-tide + SLR fingerprint's realistic range",
    )


def decode_waterlevel_cm(waterlevel_int16: np.ndarray) -> np.ndarray:
    """Decode int16-cm boundary water levels to float32 metres."""
    return waterlevel_int16.astype(np.float32) / WATERLEVEL_SCALE


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as a DataArray with the `band` dim squeezed out."""
    import rioxarray  # noqa: F401 - registers the rasterio xarray backend and spatial metadata
    return retry_transient_io(xr.open_dataarray, path, engine="rasterio").squeeze("band", drop=True)


def get_tile_bbox(tile: gpd.GeoDataFrame) -> list[float]:
    """Return ``[minx, miny, maxx, maxy]`` of a single-row tile GeoDataFrame."""
    return tile.total_bounds.tolist()


def compute_model_bbox(
    data_catalog: hydromt.DataCatalog,
    dem_source: str,
    mask_source: str,
    tile_bbox: list[float],
    buffer_arcsec: float = 1.0,
    elevation_threshold_m: float | None = None,
) -> list[float]:
    """Shrink the tile bbox to the flood-relevant DEM extent plus a buffer.

    A cell is dropped only if it is confirmed high land: mask land (or no mask
    coverage) with a valid elevation >= `elevation_threshold_m` (m). Water cells
    and land without elevation are kept. `None` keeps every non-nodata DEM cell.
    `buffer_arcsec` (~30 m per arcsec) adds a seaward ocean strip for flood entry.
    Returns `tile_bbox` unchanged if no cell is valid; the result is clipped to
    `tile_bbox` and to +/-180/+/-90.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=tile_bbox)

    if elevation_threshold_m is None:
        valid = (da.values != da.raster.nodata).squeeze()
    else:
        da_mask = data_catalog.get_rasterdataset(mask_source, bbox=tile_bbox)
        da_mask.raster.set_nodata(255)
        mask_vals = da_mask.raster.reproject_like(da, method="nearest").values.squeeze()
        dem_vals = da.values.squeeze()

        # No mask coverage (255) counts as land, as in extract_dem.
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

    # Clamp to the global range even if tile_bbox itself overshoots +/-180: a bbox
    # past the antimeridian makes reprojection windows wrap the globe (100s of GB).
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
    missing = dem_vals == nodata
    missing_land = missing & is_land

    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labels, _ = ndimage.label(missing_land, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    cell_component_size = component_sizes[labels]
    small_gap = missing_land & (cell_component_size < min_hard_fill_component_size)

    if small_gap.any():
        interpolated = fillnodata(
            dem_vals.astype(np.float32).copy(),
            mask=(~missing).astype(np.uint8),
            max_search_distance=interp_max_search_distance,
            smoothing_iterations=interp_smoothing_iterations,
        )
        # With no valid DEM in the window, fillnodata leaves the nodata sentinel;
        # hard-fill those so no raw nodata reaches encode_dem_cm.
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
    # get_rasterdataset snaps to the source's native grid, which for GEBCO (~463 m)
    # can fall half a pixel short of the tile edge and leave a nodata sliver.
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
    """Reclassify mask nodata cells to ocean where GEBCO and connectivity agree.

    A nodata cell becomes ocean only if GEBCO < 0 there AND it is 4-connected via
    GEBCO-negative nodata cells to confirmed mask ocean; all other nodata becomes
    `land_code`. Connectivity keeps inland below-sea-level basins (Dead Sea,
    Danakil) as land. A false land strip offshore would starve hydromt_sfincs'
    `create_boundary()`, which only forces ocean cells on the domain perimeter.
    Nearshore gaps usually stay land: GEBCO reads up to +16.6 m on ocean near the
    DeltaDTM/GEBCO transition.

    Returns the corrected mask and ``{"n_resolved_ocean", "n_still_land"}``.
    """
    is_nodata = mask_vals == nodata_sentinel
    is_ocean = mask_vals == ocean_code

    if not is_ocean.any():
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
    """Relabel ocean components smaller than `min_cells` (4-connected) to `lake_code`.

    deltadtm_mask sometimes codes small inland water bodies as ocean, which SFINCS
    then seeds with a nonzero initial water level (`_compute_zsini_array`).
    Returns the mask and ``{"n_relabeled", "n_components_relabeled"}``.
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
    """Clip the DEM to `bbox`, geoid-correct it and fill every missing cell.

    Returns int16 centimetres (`encode_dem_cm`). The EGM2008 -> GOCO06s offset
    (`geoid_offset_raster`) is added to valid DeltaDTM cells only. Fill values follow
    the DeltaDTM mask (mask nodata resolved via `resolve_offshore_mask_gaps_via_gebco`):

    - missing land, 4-connected gap < `min_hard_fill_component_size` cells:
      interpolated (`fillnodata`, radius `interp_max_search_distance` pixels);
    - larger missing-land gaps: `land_fill_value_m` (99 m, never floods);
    - missing ocean/lake/river: 0.0, so flooding propagates through water.
    """
    da = data_catalog.get_rasterdataset(dem_source, bbox=bbox)

    # VRT coverage gaps read as NaN, not the nodata sentinel, and would bypass gap
    # filling and become 0 m after the int16 cast (NaN comparisons are False).
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

    # Mask codes: 0 land, 1 ocean, 2 lake, 3 river.
    is_land = mask_vals == 0

    dem_vals = da.values.reshape(mask_vals.shape)
    fill = _compute_dem_gap_fill(
        dem_vals, da.raster.nodata, is_land,
        min_hard_fill_component_size, interp_max_search_distance,
        interp_smoothing_iterations, land_fill_value_m,
    ).reshape(da.shape)

    result = da.where(da != da.raster.nodata, da.copy(data=fill))
    return result.copy(data=encode_dem_cm(result.values))


def extract_dem_mask(
    data_catalog: hydromt.DataCatalog,
    mask_source: str,
    bbox: list[float],
    dem: xr.DataArray,
    nodata_sentinel: int = 255,
    gebco_source: str = "gebco",
) -> xr.DataArray:
    """Clip the DeltaDTM mask and reproject it (nearest) onto the DEM grid.

    Codes: 0 land, 1 ocean, 2 lake, 3 river. No-coverage cells become land unless
    `resolve_offshore_mask_gaps_via_gebco` resolves them to ocean; small ocean
    patches are relabeled to lake (`relabel_small_isolated_ocean_patches`).
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
    """Map land use onto the DEM grid and convert it to per-metre friction (int16).

    Friction = Manning's n / 100 via `lookup_source` (index `copernicus_worldcover`,
    column `manning_n`); unclassified cells get `default_friction` (same per-metre
    convention, should match `flooding.default_friction`). The per-grid-step
    correction the eikonal solver needs is applied at runtime via
    `simulation.flooding.friction_scale_factor`, not here.
    """
    _MISSING = np.float32(-9999.0)  # internal sentinel for unclassified cells

    # Copernicus land use stops near 80N: HydroMT either raises NoDataException or
    # returns < 2 cells in a dim (reproject_like then fails). Both mean no land use.
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
    # Temp file + os.replace so a killed job never leaves a truncated raster at
    # output_path; the PID suffix keeps concurrent writers off each other's temp file.
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
    """Write `da` atomically with the `raster_format` settings, creating parent dirs.

    Always tiled 512x512 (downstream reads small windows; plain GTiff defaults to
    strips). Integer dtypes force predictor 1, which beat predictor 2 under zstd on
    this pipeline's DEM/friction rasters in both size and speed.
    """
    predictor = 1 if np.dtype(dtype).kind in "iu" else raster_config["predictor"]
    # run_eikonal_cli.py runs outside Snakemake, which would otherwise create the dir.
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
    """Write an all-nodata int16 waterdepth placeholder on `reference_path`'s grid.

    Used for skipped tiles; `merge.merge_tile_rasters_chunk` ignores it entirely.
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
    retry_transient_io(Path(output_path).parent.mkdir, parents=True, exist_ok=True)
    _atomic_raster_write(output_path, profile, lambda dst: dst.write(data, indexes=1))


def save_waterdepth_raster(
    reference_path: str | Path, waterdepth: np.ndarray, output_path: str | Path,
) -> None:
    """Write water depth (metres, 0 where dry) as int16 cm on `reference_path`'s grid."""
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
    """Compute the fraction of each destination cell belonging to a class.

    `numerator` is 1 in-class, 0 in-domain otherwise, `numerator_nodata` outside
    the domain; `domain` is 1/0 with no nodata. Uses ``ff = A x B`` with
    ``A = avg(numerator, nodata excluded) = sum(class) / n_in_domain`` and
    ``B = avg(domain) = n_in_domain / n_total``, so ``ff = sum(class) / n_total``;
    NaN where coverage is zero. Must be two separate `reproject()` calls: a
    multi-band `Resampling.average` call shares one validity mask across bands
    and corrupts `B` wherever `numerator` is nodata.
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


def polygon_domain_mask(
    perimeter_gdf: gpd.GeoDataFrame, dst_transform: Affine, dst_crs, dst_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize study-area perimeter polygons into a boolean domain mask.

    Point-in-polygon (`all_touched=False`), no buffer: the official perimeter is the
    agency's study design. Callers must pre-filter `perimeter_gdf` to relevant zones.
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
