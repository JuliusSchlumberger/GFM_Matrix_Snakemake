"""Vertical datum correction to GOCO06s, for the DeltaDTM DEM and local DEM sources.

DeltaDTM (Pronk et al. 2024) is referenced to EGM2008; the AVISO MDT
(`mdt_cnes_cls22`) used for water levels is relative to GOCO06s. A pure
geoid-to-geoid shift avoids extrapolating MDT across land. EGM2008 is truncated
to GOCO06s's max spherical-harmonic degree; the offset N_EGM2008 - N_GOCO06s (m)
is ADDED to valid DEM pixels. No MDT term is applied here. Method ported from
GCFM_UU 05a_get_elevation.py. The global offset is computed once and cached;
per-tile resampling is valid because the field is smooth at arc-second scale.

A local DEM source referenced to a national datum (not EGM2008) needs its own
bridge to GOCO06s instead of this EGM2008 offset - see `DATUM_BRIDGES` and
`compute_vertical_datum_offset`. GOCO06s (d/o 300, ~67 km half-wavelength)
lacks the short-wavelength signal present in national hybrid geoid models
(OSGM15, GEOID18): decimetre-to-~1 m discrepancies in the resulting bridged
height are an expected property of the target geoid, not a bug.

`navd88_geoid18`'s "hybrid grid" (NAD83(2011)+NAVD88 height -> ITRF2014, GEOID18
plus the mandatory NAD83(2011)->ITRF2014 Helmert shift) is not a downloaded
national grid file like OSGM15 - it is precomputed once via pyproj/PROJ
(`preparation/compute_navd88_itrf2014_shift_raster.py`) and cached as an
ordinary GeoTIFF, after which it is sampled exactly like OSGM15's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.warp import reproject

from config_utils import retry_transient_io


def compute_single_geoid_grid(
    gfc_path: str | Path, lmax: int | None = None,
) -> tuple[np.ndarray, Affine, str]:
    """Compute one geoid's own height above the WGS84 ellipsoid, in metres.

    Input is an ICGEM .gfc file. Returns `(geoid_height, transform, crs)`: a
    north-up float32 grid, lon -180..180. `lmax=None` uses the model's own
    max spherical-harmonic degree; pass another model's `lmax` to truncate
    (pad) this one to the same resolution before differencing two models.
    Raises ImportError without `pyshtools` / `boule`.
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError(
            "pyshtools is required for geoid height computation.\n"
            "Install with: pixi add pyshtools  (or: conda install -c conda-forge pyshtools)"
        )
    try:
        import boule as _boule
    except ImportError:
        raise ImportError(
            "boule is required by pyshtools for ellipsoid definitions.\n"
            "Install with: pixi add boule  (or: conda install -c conda-forge boule)"
        )
    from rasterio.transform import from_origin as _from_origin

    coeffs = retry_transient_io(pysh.SHGravCoeffs.from_file, str(gfc_path), format="icgem")
    if lmax is None:
        lmax = coeffs.lmax
    coeffs = coeffs.pad(lmax)

    # pyshtools 4.12 has no `ellipsoid=` kwarg; pass the WGS84 normal-gravity
    # parameters individually.
    wgs84 = _boule.WGS84
    grid = coeffs.geoid(
        potref=wgs84.reference_normal_gravity_potential,
        a=wgs84.semimajor_axis,
        f=wgs84.flattening,
        omega=wgs84.angular_velocity,
        lmax=lmax,
    )

    da = grid.to_xarray()
    values = da.values.astype(np.float32)

    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())
    lats = da[lat_dim].values
    lons = da[lon_dim].values

    # pyshtools grids run 0..360 deg longitude; DEM tiles use -180..180.
    half = len(lons) // 2
    values = np.roll(values, -half, axis=1)
    lons = np.concatenate([lons[half:] - 360.0, lons[:half]])

    dlat = float(np.abs(lats[0] - lats[1]))
    dlon = float(lons[1] - lons[0])
    transform = _from_origin(
        float(lons[0]) - dlon / 2, float(lats[0]) + dlat / 2, dlon, dlat,
    )
    return values, transform, "EPSG:4326"


def compute_geoid_offset_grid(
    goco06s_gfc: str | Path, egm2008_gfc: str | Path,
) -> tuple[np.ndarray, Affine, str]:
    """Compute the global geoid offset N_EGM2008 - N_GOCO06s in metres.

    Inputs are ICGEM .gfc files. Returns `(offset, transform, crs)`: a north-up
    float32 grid, lon -180..180, at GOCO06s's native resolution.
    """
    grid_goco, transform, crs = compute_single_geoid_grid(goco06s_gfc)
    grid_egm, _, _ = compute_single_geoid_grid(egm2008_gfc, lmax=_lmax_of(goco06s_gfc))
    offset = (grid_egm - grid_goco).astype(np.float32)
    return offset, transform, crs


def _lmax_of(gfc_path: str | Path) -> int:
    """Read a .gfc file's own max spherical-harmonic degree, without computing its grid."""
    import pyshtools as pysh

    return retry_transient_io(pysh.SHGravCoeffs.from_file, str(gfc_path), format="icgem").lmax


def write_grid_raster(
    values: np.ndarray, transform: Affine, crs: str, out_path: str | Path,
) -> None:
    retry_transient_io(Path(out_path).parent.mkdir, parents=True, exist_ok=True)
    with retry_transient_io(
        rasterio.open,
        out_path, "w",
        driver="GTiff", dtype="float32",
        width=values.shape[1], height=values.shape[0],
        count=1, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(values, 1)


def write_geoid_offset_raster(
    goco06s_gfc: str | Path, egm2008_gfc: str | Path, out_path: str | Path,
) -> None:
    """Compute the global geoid offset once and cache it as a GeoTIFF."""
    offset_arr, transform, crs = compute_geoid_offset_grid(goco06s_gfc, egm2008_gfc)
    write_grid_raster(offset_arr, transform, crs, out_path)


def write_single_geoid_raster(gfc_path: str | Path, out_path: str | Path) -> None:
    """Compute one geoid's own absolute height grid once and cache it as a GeoTIFF."""
    values, transform, crs = compute_single_geoid_grid(gfc_path)
    write_grid_raster(values, transform, crs, out_path)


def sample_geoid_offset(
    offset_raster_path: str | Path,
    dst_transform: Affine,
    dst_crs,
    dst_height: int,
    dst_width: int,
) -> np.ndarray:
    """Bilinearly resample the cached geoid offset (metres) onto a destination grid.

    NaN where `offset_raster_path` has no coverage there (e.g. OSGM15/GEOID18's
    onshore-only extent) - without `src_nodata`, GDAL's bilinear kernel would
    blend the raw nodata sentinel into real values right at the nodata boundary,
    exactly where a coastal DEM needs this offset most.
    """
    with retry_transient_io(rasterio.open, offset_raster_path) as src:
        offset_on_dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=offset_on_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return offset_on_dst


# Closed registry of native vertical datums `extract_dem` can bridge to GOCO06s.
# An unrecognized `native_datum` must raise, never silently fall through to EGM2008.
#
# "native": an explicit no-op (zero offset), for a scenario where EVERY numeric
# input sharing a freeboard calculation (DEM and boundary water levels) comes from
# the same un-bridged source and datum, so they are already mutually consistent
# without GOCO06s - e.g. England's local-input DEM (EA LIDAR) and boundary levels
# (EA design sea levels), both natively ODN. Bridging them would add only the
# geoid grids' own bilinear-resampling discretisation noise, confirmed empirically:
# the un-bridged DEM-vs-DeltaDTM and boundary-vs-COAST-RP gaps are each ~1m, but the
# resulting FREEBOARD gap (water level minus DEM, within one run) collapses to
# -0.01..-0.36 m on every tile with enough reliable overlap pixels - the ~1m terms
# are the SAME odn_osgm15 correction applied to both legs, and cancel in freeboard.
# "native" must NEVER be used for a source mixing a local and a global leg (e.g. a
# local DEM paired with the global COAST-RP boundary, or vice versa) - there the
# two legs do not share a native datum, so bridging both to GOCO06s is mandatory,
# not optional. See TODO.md.
#
# "navd88_geoid18": SF Bay's NOAA LIDAR DEM. hybrid_grid_raster is the cached
# NAD83(2011)+NAVD88-height -> ITRF2014 shift (GEOID18 + the mandatory Helmert
# shift, ~0.5-1m in California - see compute_navd88_itrf2014_shift_raster.py),
# not a downloaded national grid - same role as OSGM15, different derivation.
DATUM_BRIDGES = frozenset({"egm2008", "odn_osgm15", "navd88_geoid18", "native"})
# The bridges that also need GOCO06s's absolute geoid (rule compute_goco06s_absolute_raster).
HYBRID_GRID_BRIDGES = frozenset({"odn_osgm15", "navd88_geoid18"})


def _hybrid_minus_goco_grid(
    bridge_name: str, hybrid_grid_raster, goco06s_absolute_raster,
    dst_transform: Affine, dst_crs, dst_height: int, dst_width: int,
) -> np.ndarray:
    """Shared body of the hybrid-grid bridges (odn_osgm15, navd88_geoid18) on a raster grid."""
    if hybrid_grid_raster is None or goco06s_absolute_raster is None:
        raise ValueError(f"{bridge_name} needs hybrid_grid_raster and goco06s_absolute_raster")
    n_hybrid = sample_geoid_offset(hybrid_grid_raster, dst_transform, dst_crs, dst_height, dst_width)
    n_goco = sample_geoid_offset(goco06s_absolute_raster, dst_transform, dst_crs, dst_height, dst_width)
    diff = n_hybrid - n_goco
    # n_hybrid is NaN just outside the hybrid grid's own coverage (e.g. a coastal DEM
    # tile's seaward edge) - fill from the nearest valid value rather than propagate
    # NaN into extract_dem's additive correction.
    missing = np.isnan(diff)
    if missing.any():
        diff = fillnodata(diff, mask=(~missing).astype(np.uint8), max_search_distance=100)
    if np.isnan(diff).any():
        raise ValueError(
            f"{bridge_name} bridge: hybrid_grid_raster has no valid coverage within "
            "fillnodata's search distance of some destination cell(s) - this grid's "
            "own extent likely does not reach this tile at all, not just its "
            "immediate coastal edge."
        )
    return diff


def _hybrid_minus_goco_points(
    bridge_name: str, hybrid_grid_raster, goco06s_absolute_raster, lons: np.ndarray, lats: np.ndarray,
) -> np.ndarray:
    """Shared body of the hybrid-grid bridges (odn_osgm15, navd88_geoid18) at scattered points."""
    if hybrid_grid_raster is None or goco06s_absolute_raster is None:
        raise ValueError(f"{bridge_name} needs hybrid_grid_raster and goco06s_absolute_raster")
    n_hybrid = sample_geoid_offset_at_points(hybrid_grid_raster, lons, lats)
    n_goco = sample_geoid_offset_at_points(goco06s_absolute_raster, lons, lats)
    diff = n_hybrid - n_goco
    if np.isnan(diff).any():
        raise ValueError(
            f"{bridge_name} bridge: hybrid_grid_raster has no coverage at some point(s) - "
            "check for a point outside the hybrid grid's own coverage or a named "
            "local-datum exception that should have been excluded before calling this."
        )
    return diff


def compute_vertical_datum_offset(
    native_datum: str,
    dst_transform: Affine,
    dst_crs,
    dst_height: int,
    dst_width: int,
    *,
    geoid_offset_raster: str | Path,
    hybrid_grid_raster: str | Path | None = None,
    goco06s_absolute_raster: str | Path | None = None,
) -> np.ndarray:
    """Offset (m) to ADD to a DEM's raw elevation to reference it to GOCO06s.

    `native_datum="egm2008"` (DeltaDTM, every production tile) uses the cached
    EGM2008-GOCO06s difference raster (`geoid_offset_raster`).

    `native_datum="odn_osgm15"` (England/Wales lidar DEMs, ODN via OSGM15) needs
    `hybrid_grid_raster` (OSGM15, ETRS89<->ODN) and `goco06s_absolute_raster`
    (GOCO06s's own absolute geoid height, `write_single_geoid_raster`):
    h_ETRS89 = H_ODN + N_OSGM15; H_GOCO = h_ETRS89 - N_GOCO06s, so the offset to
    add to H_ODN is N_OSGM15 - N_GOCO06s.

    `native_datum="navd88_geoid18"` (SF Bay lidar DEM, NAVD88 via GEOID18) needs
    the same two rasters, with `hybrid_grid_raster` instead holding the cached
    NAD83(2011)+NAVD88-height -> ITRF2014 shift (see
    `compute_navd88_itrf2014_shift_raster.py`): same `diff = n_hybrid - n_goco`
    formula as odn_osgm15.

    Any other `native_datum` raises NotImplementedError - never silently treated
    as EGM2008 (that would double/mis-correct a DEM already in a different
    native datum).
    """
    if native_datum == "native":
        return np.zeros((dst_height, dst_width), dtype=np.float32)
    if native_datum == "egm2008":
        return sample_geoid_offset(geoid_offset_raster, dst_transform, dst_crs, dst_height, dst_width)
    if native_datum in HYBRID_GRID_BRIDGES:
        return _hybrid_minus_goco_grid(
            native_datum, hybrid_grid_raster, goco06s_absolute_raster, dst_transform, dst_crs, dst_height, dst_width,
        )
    raise NotImplementedError(
        f"No vertical-datum bridge implemented for native_datum={native_datum!r} "
        f"(known: {sorted(DATUM_BRIDGES)})"
    )


def sample_geoid_offset_at_points(
    offset_raster_path: str | Path, lons: np.ndarray, lats: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample the cached geoid offset (metres) at scattered point coordinates.

    NaN where `offset_raster_path` has no coverage there - see `sample_geoid_offset`.
    """
    from scipy.ndimage import map_coordinates

    with retry_transient_io(rasterio.open, offset_raster_path) as src:
        values = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            values[values == nodata] = np.nan
        a, b, c, d, e, f = src.transform[:6]
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        det = a * e - b * d
        # transform maps (col, row) INTEGERS to a pixel's top-left CORNER; map_coordinates
        # treats integer indices as pixel CENTRES - the -0.5 below converts between them.
        cols = (e * (lons - c) - b * (lats - f)) / det - 0.5
        rows = (a * (lats - f) - d * (lons - c)) / det - 0.5
    return map_coordinates(values, [rows, cols], order=1, mode="constant", cval=np.nan).astype(np.float32)


def compute_vertical_datum_offset_at_points(
    native_datum: str,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    geoid_offset_raster: str | Path,
    hybrid_grid_raster: str | Path | None = None,
    goco06s_absolute_raster: str | Path | None = None,
) -> np.ndarray:
    """Point-geometry counterpart of `compute_vertical_datum_offset` (same formulas).

    Used for the EA coastal boundary stations (points), not a DEM raster. Raises
    ValueError if any point falls outside the native datum's hybrid grid coverage
    beyond what a station location should ever do (e.g. a mis-flagged station) -
    callers must exclude any named local-datum exception (Stornoway, Lerwick, St
    Mary's) before calling this, not rely on it to catch them.
    """
    if native_datum == "native":
        return np.zeros(np.asarray(lons).shape, dtype=np.float32)
    if native_datum == "egm2008":
        return sample_geoid_offset_at_points(geoid_offset_raster, lons, lats)
    if native_datum in HYBRID_GRID_BRIDGES:
        return _hybrid_minus_goco_points(native_datum, hybrid_grid_raster, goco06s_absolute_raster, lons, lats)
    raise NotImplementedError(
        f"No vertical-datum bridge implemented for native_datum={native_datum!r} "
        f"(known: {sorted(DATUM_BRIDGES)})"
    )
