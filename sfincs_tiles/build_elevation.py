"""Build one tile's SFINCS elevation: DeltaDTM on land, MDT-corrected GEBCO at sea.

On dem.tif's native EPSG:4326 grid; the eikonal model reads dem.tif instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, minimum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdt import mdt_lookup_fn  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

LAND_CODE = 0
OCEAN_CODE = 1
LAKE_CODE = 2
RIVER_CODE = 3


# Storm surge never reaches deeper; deeper bathymetry only shrinks the CFL timestep.
MIN_BATHYMETRY_M = -10.0

# GEBCO is MDT-corrected into DeltaDTM's GOCO06s frame, where a flat sea is ~0 m; higher
# values are spikes from GEBCO's ~450 m resolution at the DeltaDTM transition.
MAX_OCEAN_ELEVATION_M = 0.0

# A below-sea-level channel would have to fill before SFINCS overflows onto land,
# which the eikonal's flattened (0 m) lakes/rivers do not model.
LAKE_RIVER_MIN_ELEVATION_M = 0.0

# hydromt_sfincs subgrid_v_table clamps subgrid pixels below zvolmin = -20 m, while depth
# uses the unclamped elevation: unfloored land there floods spuriously. 5 m margin
# (applies to whole dem.tif pixels); all land cells, not just lake/river.
LAND_MIN_ELEVATION_M = -15.0

# fillnodata search radius (px); unreached cells fall back to LAKE_RIVER_MIN_ELEVATION_M.
LAKE_RIVER_INTERP_MAX_SEARCH_DISTANCE_PX = 200

GEBCO_COAST_CLIP_M = 200.0  # GEBCO discarded and interpolated within this of land
GEBCO_COAST_CLIP_MAX_SEARCH_DISTANCE_PX = 50  # margin over the ~7 px (200 m / ~30 m) gap

# 2D minimum filter (not channel-direction-aware), so no bank peak blocks propagation.
LAKE_RIVER_SMOOTH_WINDOW_M = 500.0


def _fill_lake_river_elevation(
    dem_m: np.ndarray, mask: np.ndarray, transform, lat_deg: float,
    interp_max_search_distance_px: float = LAKE_RIVER_INTERP_MAX_SEARCH_DISTANCE_PX,
    smooth_window_m: float = LAKE_RIVER_SMOOTH_WINDOW_M,
) -> np.ndarray:
    # extract_dem.py flat-fills lake/river to 0.0 m: no real DeltaDTM data there
    land = mask == LAND_CODE
    lake_river = (mask == LAKE_CODE) | (mask == RIVER_CODE)
    if not lake_river.any():
        return dem_m

    fill_source = np.where(land, dem_m, np.nan).astype(np.float32)
    valid_mask = (~np.isnan(fill_source)).astype(np.uint8)
    interpolated = fillnodata(
        fill_source, mask=valid_mask, max_search_distance=interp_max_search_distance_px,
    )

    px_w_m = abs(transform.a) * 111320.0 * np.cos(np.radians(lat_deg))
    px_h_m = abs(transform.e) * 110540.0
    win_px = max(1, round(smooth_window_m / max(min(px_w_m, px_h_m), 1e-6)))
    smoothed = minimum_filter(interpolated, size=win_px)

    return np.where(lake_river, smoothed, dem_m)


def _fill_coastal_transition_elevation(
    gebco_corrected: np.ndarray, land_lake_river: np.ndarray, ocean: np.ndarray, transform, lat_deg: float,
    clip_m: float = GEBCO_COAST_CLIP_M,
    interp_max_search_distance_px: float = GEBCO_COAST_CLIP_MAX_SEARCH_DISTANCE_PX,
) -> tuple[np.ndarray, dict]:
    # At ~450 m the nearest GEBCO sample can already be clamped at MIN_BATHYMETRY_M,
    # so bilinear alone gives no gradual approach to shore.
    if not ocean.any():
        return gebco_corrected, {"n_coastal_transition_cells": 0}

    land = ~ocean & ~np.isnan(land_lake_river)  # land_lake_river is finite everywhere land/lake/river
    px_w_m = abs(transform.a) * 111320.0 * np.cos(np.radians(lat_deg))
    px_h_m = abs(transform.e) * 110540.0
    dist_to_land_m = distance_transform_edt(~land, sampling=(px_h_m, px_w_m))

    near_coast_ocean = ocean & (dist_to_land_m <= clip_m)
    if not near_coast_ocean.any():
        return gebco_corrected, {"n_coastal_transition_cells": 0}

    combined_pre = np.where(ocean, gebco_corrected, land_lake_river).astype(np.float32)
    fill_source = np.where(near_coast_ocean, np.nan, combined_pre)
    valid_mask = (~np.isnan(fill_source)).astype(np.uint8)
    interpolated = fillnodata(
        fill_source, mask=valid_mask, max_search_distance=interp_max_search_distance_px,
    )
    gebco_filled = np.where(near_coast_ocean, interpolated, gebco_corrected)
    return gebco_filled, {"n_coastal_transition_cells": int(near_coast_ocean.sum())}


def build_combined_elevation(
    dem_path: Path,
    mask_path: Path,
    gebco_path: Path,
    mdt_path: Path,
    mdt_variable: str = "mdt",
    mdt_fallback_deg: float = 3.0,
    min_bathymetry_m: float = MIN_BATHYMETRY_M,
    max_ocean_elevation_m: float = MAX_OCEAN_ELEVATION_M,
    lake_river_min_elevation_m: float = LAKE_RIVER_MIN_ELEVATION_M,
    land_min_elevation_m: float = LAND_MIN_ELEVATION_M,
) -> tuple[np.ndarray, dict]:
    """Return the combined elevation (m) on dem.tif's grid plus diagnostics.

    dem.tif is int16 cm (DeltaDTM); mask.tif must be pixel-identical. GEBCO gets the
    local MDT added (H_GOCO06s = H_MSL + MDT, see mdt.py) so both halves share DeltaDTM's
    GOCO06s reference, then is clipped to [min_bathymetry_m, max_ocean_elevation_m].
    `mdt_fallback_deg` is the search radius (deg) for a valid MDT near the centroid.
    Raises ValueError on a grid mismatch, no ocean cells, or no valid MDT.
    """
    with retry_transient_io(rasterio.open, dem_path) as src:
        dem_cm = src.read(1)
        dem_nodata = src.nodata
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        bounds = src.bounds
    dem_m = np.where(dem_cm == dem_nodata, np.nan, dem_cm.astype(np.float64) / 100.0)

    with retry_transient_io(rasterio.open, mask_path) as src:
        mask = src.read(1)
        if mask.shape != shape:
            raise ValueError(f"mask.tif shape {mask.shape} != dem.tif shape {shape} - expected pixel-identical grids")

    ocean = mask == OCEAN_CODE
    lake_river = (mask == LAKE_CODE) | (mask == RIVER_CODE)
    if not ocean.any():
        raise ValueError("No ocean cells (mask==1) in this tile - nothing for GEBCO to fill in")

    # Bilinear: GEBCO's 15" (~450 m) under ~30 m pixels would, with nearest, leave a
    # hard artificial cliff against DeltaDTM.
    with retry_transient_io(rasterio.open, gebco_path) as src:
        gebco_arr = np.empty(shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=gebco_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # one centroid MDT for the whole tile: MDT varies smoothly over tens of km
    cx, cy = (bounds.left + bounds.right) / 2.0, (bounds.bottom + bounds.top) / 2.0
    mdt_lookup = mdt_lookup_fn(mdt_path, mdt_variable, mdt_fallback_deg)
    mdt_m = mdt_lookup(cx, cy)
    if np.isnan(mdt_m):
        raise ValueError(f"No valid MDT value found within {mdt_fallback_deg} deg of tile centroid ({cx}, {cy})")
    gebco_shifted = gebco_arr + mdt_m
    gebco_corrected = np.clip(gebco_shifted, min_bathymetry_m, max_ocean_elevation_m)

    lake_river_elevation = _fill_lake_river_elevation(dem_m, mask, transform, lat_deg=cy)

    land = mask == LAND_CODE
    land_floored = np.where(land, np.maximum(dem_m, land_min_elevation_m), dem_m)
    land_lake_river = np.where(
        lake_river, np.maximum(lake_river_elevation, lake_river_min_elevation_m), land_floored,
    )

    gebco_corrected, coastal_transition_info = _fill_coastal_transition_elevation(
        gebco_corrected, land_lake_river, ocean, transform, lat_deg=cy,
    )
    combined = np.where(ocean, gebco_corrected, land_lake_river)

    return combined, {
        "profile": profile, "transform": transform, "crs": crs,
        "mdt_m": mdt_m, "n_ocean_nan": int(np.isnan(gebco_corrected[ocean]).sum()),
        "n_floored": int(np.nansum(gebco_shifted[ocean] < min_bathymetry_m)),
        "n_ceiled": int(np.nansum(gebco_shifted[ocean] > max_ocean_elevation_m)),
        "n_lake_river_cells": int(lake_river.sum()),
        "n_lake_river_floored": int(np.nansum(lake_river_elevation[lake_river] < lake_river_min_elevation_m)) if lake_river.any() else 0,
        "n_land_floored": int(np.nansum(dem_m[land] < land_min_elevation_m)) if land.any() else 0,
        **coastal_transition_info,
    }


def main() -> None:
    """Build and write elevation_combined.tif for one tile."""
    import argparse

    _repo_root = Path(__file__).resolve().parent.parent
    from gfm_config import read_root, resolve_catalog_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out", default=None, help="output GeoTIFF path (default: {base-dir-name}/{tile_id}/sfincs_model/elevation_combined.tif)")
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS", help="output root directory name under paths.root (default: calibration_roughness_SFINCS)")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    catalog_path = _repo_root / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"

    tile_dir = root / args.base_dir_name / args.tile_id / "inputs"
    gebco_path = resolve_catalog_path(catalog_path, root, "gebco")
    mdt_path = resolve_catalog_path(catalog_path, root, "mdt_cnes_cls22")

    combined, info = build_combined_elevation(
        tile_dir / "dem.tif", tile_dir / "mask.tif", gebco_path, mdt_path,
    )

    print(f"MDT applied to GEBCO (ADD): {info['mdt_m']:+.4f} m")
    print(f"Ocean cells with no valid GEBCO value: {info['n_ocean_nan']}")
    print(f"Ocean cells floored at {MIN_BATHYMETRY_M:.0f} m: {info['n_floored']}")
    print(f"Ocean cells ceiled at {MAX_OCEAN_ELEVATION_M:.0f} m: {info['n_ceiled']}")
    print(f"Lake/river cells re-derived from land + smoothed: {info['n_lake_river_cells']} "
          f"(floored at {LAKE_RIVER_MIN_ELEVATION_M:.0f} m: {info['n_lake_river_floored']})")
    print(f"Land cells floored at {LAND_MIN_ELEVATION_M:.0f} m: {info['n_land_floored']}")
    print(f"Ocean cells within {GEBCO_COAST_CLIP_M:.0f} m of the coast: GEBCO discarded and "
          f"interpolated instead: {info['n_coastal_transition_cells']}")
    finite = combined[np.isfinite(combined)]
    print(f"Combined elevation range: {finite.min():.2f} to {finite.max():.2f} m (n={finite.size})")

    out_path = Path(args.out) if args.out else root / args.base_dir_name / args.tile_id / "sfincs_model" / "elevation_combined.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = info["profile"]
    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(combined.astype(np.float32), 1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
