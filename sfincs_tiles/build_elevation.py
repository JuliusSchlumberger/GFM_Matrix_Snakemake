"""Builds the combined elevation surface for one tile's SFINCS model:
DeltaDTM on land, MDT-corrected GEBCO on sea, on dem.tif's own native
EPSG:4326 grid, before any UTM reprojection.

Kept separate from model_outputs/{tile_id}/inputs/dem.tif, which the
eikonal model reads instead.

Lake/river cells (mask==3/2) have no real DeltaDTM elevation data
(upstream extract_dem.py flat-fills them to 0.0m). Their elevation is
re-derived here from nearby valid land data, smoothed with a rolling
minimum along the water body (see LAKE_RIVER_SMOOTH_WINDOW_M) so a single
bank can't locally bridge across a channel, then floored at
LAKE_RIVER_MIN_ELEVATION_M.
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


MIN_BATHYMETRY_M = -10.0  # ocean elevation floor: storm-tide/surge forcing never
# reaches this deep, so deeper bathymetry only widens the solver's elevation range
# and shrinks the CFL-stable timestep.

MAX_OCEAN_ELEVATION_M = 0.0  # ocean elevation ceiling: GEBCO's coarse (~450m)
# resolution can produce spurious dry-land-height spikes near the GEBCO/DeltaDTM
# transition. gebco_corrected is MDT-corrected into DeltaDTM's own GOCO06s frame,
# so a flat sea reads as ~0m there; anything above that is an artifact, clamped at 0m.

LAKE_RIVER_MIN_ELEVATION_M = 0.0  # floor for lake/river elevation: a below-sea-level
# channel would need to physically fill that volume before SFINCS sees it overflow
# onto land, an effect eikonal's own flattened (0m) lakes/rivers don't model.

LAND_MIN_ELEVATION_M = -15.0  # hydromt_sfincs's own subgrid table builder
# (subgrid.py::subgrid_v_table, zvolmin hardcoded to -20.0 in components/grid/
# subgrid.py) clamps any subgrid pixel below -20m to exactly -20m. Since flood
# depth is computed as simulated water level minus the unclamped elevation_combined.tif
# value, an unfloored land cell below that clamp produces a spurious flood. Floored
# 5m above the clamp for margin (this operates on whole dem.tif pixels, coarser than
# the subgrid pixels the clamp itself applies to). Applied to all land cells
# (mask==LAND_CODE), not just lake/river.

LAKE_RIVER_INTERP_MAX_SEARCH_DISTANCE_PX = 200  # rasterio.fill.fillnodata's search
# radius (pixels) for the nearest-valid-land-neighbour fill; generous, since wide
# rivers/large lakes can be far from the nearest bank pixel. Cells it can't reach
# fall back to LAKE_RIVER_MIN_ELEVATION_M via the final floor below.

GEBCO_COAST_CLIP_M = 200.0  # distance (m) from the nearest land cell within which
# GEBCO's own ocean elevation is discarded and interpolated instead - see
# _fill_coastal_transition_elevation's own docstring.

GEBCO_COAST_CLIP_MAX_SEARCH_DISTANCE_PX = 50  # margin over GEBCO_COAST_CLIP_M's own
# ~200m / ~30m-per-pixel = ~7px gap width.

LAKE_RIVER_SMOOTH_WINDOW_M = 500.0  # 2D spatial minimum-filter window (not
# channel-direction-aware): ensures no local bank/rim peak exceeds its own
# neighbourhood's lowest point, so it can't block flood propagation along the channel.


def _fill_lake_river_elevation(
    dem_m: np.ndarray, mask: np.ndarray, transform, lat_deg: float,
    interp_max_search_distance_px: float = LAKE_RIVER_INTERP_MAX_SEARCH_DISTANCE_PX,
    smooth_window_m: float = LAKE_RIVER_SMOOTH_WINDOW_M,
) -> np.ndarray:
    """Re-derives lake/river elevation from nearby valid land cells, then smooths it
    with a rolling minimum along the water body. Returns dem_m unchanged outside
    lake/river cells."""
    land = mask == LAND_CODE
    lake_river = (mask == LAKE_CODE) | (mask == RIVER_CODE)
    if not lake_river.any():
        return dem_m

    fill_source = np.where(land, dem_m, np.nan).astype(np.float32)
    valid_mask = (~np.isnan(fill_source)).astype(np.uint8)
    interpolated = fillnodata(
        fill_source, mask=valid_mask, max_search_distance=interp_max_search_distance_px,
    )

    # per-axis metre resolution at this tile's latitude
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
    """Discards GEBCO's own ocean elevation within `clip_m` of the nearest land
    cell and interpolates across that gap from the land edge and the nearest
    real GEBCO value beyond it. Returns (corrected ocean elevation, diagnostics).

    GEBCO's ~450m native resolution means the nearest real sample to a given
    coastline stretch can already be a clamped-deep value (MIN_BATHYMETRY_M)
    with no real sample between it and the coast, so bilinear reprojection
    alone can't produce a gradual approach to shore.
    """
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
    """Returns (combined_elevation_m, profile) on dem.tif's own grid.

    Land cells use DeltaDTM's dem.tif value, floored at ``land_min_elevation_m``
    (see LAND_MIN_ELEVATION_M). Lake/river cells are re-derived from nearby
    valid land data (see _fill_lake_river_elevation, LAKE_RIVER_SMOOTH_WINDOW_M),
    floored at ``lake_river_min_elevation_m``. Ocean cells use GEBCO's
    bathymetry, reprojected onto this grid and corrected by adding the local
    MDT (H_GOCO06s = H_MSL + MDT, see mdt.py) so both halves of the merged
    surface share DeltaDTM's GOCO06s reference.

    Ocean elevation is then clipped to [``min_bathymetry_m``, ``max_ocean_elevation_m``]
    (default -10 to 0 m) - see MIN_BATHYMETRY_M and MAX_OCEAN_ELEVATION_M.
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

    # Reproject GEBCO onto this tile's exact grid using bilinear, not nearest:
    # GEBCO's 15 arc-sec native resolution (~450m) is far coarser than these ~30m
    # tiles, so nearest-neighbour would replicate a single sample across many
    # pixels, producing a hard artificial cliff where it meets DeltaDTM's finer
    # land detail. Bilinear blends neighbouring GEBCO samples into a gradual
    # approach to the coast instead.
    with retry_transient_io(rasterio.open, gebco_path) as src:
        gebco_arr = np.empty(shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=gebco_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # MDT lookup at the tile's centroid (MDT varies smoothly over tens of km),
    # applied as a uniform additive correction to every ocean cell.
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
    import argparse

    _repo_root = Path(__file__).resolve().parent.parent
    from gfm_config import read_root, resolve_catalog_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out", default=None, help="output GeoTIFF path (default: {base-dir-name}/{tile_id}/sfincs_model/elevation_combined.tif)")
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2", help="output root directory name under paths.root (default: validation_sfincs_v2)")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    catalog_path = _repo_root / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"

    # Reads from the tile's working copy (base_dir_name/inputs/), not
    # model_outputs/ directly.
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
