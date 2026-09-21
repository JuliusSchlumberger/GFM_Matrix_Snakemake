"""Build the SFINCS-only combined elevation surface for one tile: DeltaDTM
on land, MDT-corrected GEBCO on sea - at the tile's own native EPSG:4326
grid (dem.tif's own transform/shape), before any UTM reprojection.

Kept entirely separate from model_outputs/{tile_id}/inputs/dem.tif - the
eikonal model never reads this, never changes.

Rivers (mask==3) are left at DeltaDTM's own dem.tif value (same as land) -
this module doesn't attempt river bathymetry; see sfincs_tiles' own plan
doc note on why test tiles are chosen with ~0% river coverage instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdt import mdt_lookup_fn  # noqa: E402

LAND_CODE = 0
OCEAN_CODE = 1
LAKE_CODE = 2
RIVER_CODE = 3


MIN_BATHYMETRY_M = -10.0  # was -50.0 until 2026-09's A/B test (tile 2335/2335b/2335c):
# real ~1.7x additional speedup (larger CFL-stable timestep - shallower water means a
# lower shallow-water wave celerity sqrt(g*h), so a shallower floor relaxes the
# timestep constraint, not the other way around), confirmed via real SFINCS runs to
# change the actual flood result negligibly (28 vs 29 flooded cells out of 20086,
# identical 2.801 m max depth) ONCE the elevation-reprojection bug that was
# contaminating that comparison got fixed (see build_sfincs_tile.py's own elevation-
# reprojection comment) - the two changes were found and fixed together, don't split
# this value back to -50 without also reverting that fix, the two were validated as a
# pair, not independently.


def build_combined_elevation(
    dem_path: Path,
    mask_path: Path,
    gebco_path: Path,
    mdt_path: Path,
    mdt_variable: str = "mdt",
    mdt_fallback_deg: float = 3.0,
    min_bathymetry_m: float = MIN_BATHYMETRY_M,
) -> tuple[np.ndarray, dict]:
    """Returns (combined_elevation_m, profile) on dem.tif's own grid.

    combined_elevation_m: land/river/lake cells = DeltaDTM's own dem.tif
    value (decoded to metres); ocean cells = GEBCO's own bathymetry,
    reprojected onto this grid and corrected by ADDING the local MDT
    (H_GOCO06s = H_MSL + MDT - see mdt.py's own module docstring) so both
    halves of the merged surface share DeltaDTM's GOCO06s reference.

    Ocean depth is then floored at ``min_bathymetry_m`` (default -50 m):
    the storm-tide/surge signal this model is forced with never reaches
    anywhere near that deep, so real trench/shelf-break bathymetry below
    it (e.g. tile 929's real -640 m near the Norwegian Trench) adds
    nothing physically, just an unnecessarily wide elevation range for the
    solver and for any downstream color scale.
    """
    with rasterio.open(dem_path) as src:
        dem_cm = src.read(1)
        dem_nodata = src.nodata
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        bounds = src.bounds
    dem_m = np.where(dem_cm == dem_nodata, np.nan, dem_cm.astype(np.float64) / 100.0)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        if mask.shape != shape:
            raise ValueError(f"mask.tif shape {mask.shape} != dem.tif shape {shape} - expected pixel-identical grids")

    ocean = mask == OCEAN_CODE
    if not ocean.any():
        raise ValueError("No ocean cells (mask==1) in this tile - nothing for GEBCO to fill in")

    # Reproject GEBCO onto this tile's exact grid (nearest - GEBCO's own 15
    # arc-sec native resolution is coarser than our ~30m tiles at most
    # latitudes, so this is an upsample; nearest keeps real GEBCO values
    # rather than interpolating across what is, locally, a near-flat
    # regional bathymetry gradient anyway).
    with rasterio.open(gebco_path) as src:
        gebco_arr = np.empty(shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=gebco_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    # MDT correction: one lookup per unique-enough location is overkill for
    # a single small tile - look up at the tile's own centroid once (MDT
    # varies smoothly over tens of km, see plan doc's own reasoning) and
    # apply as a uniform ADD to every ocean cell.
    cx, cy = (bounds.left + bounds.right) / 2.0, (bounds.bottom + bounds.top) / 2.0
    mdt_lookup = mdt_lookup_fn(mdt_path, mdt_variable, mdt_fallback_deg)
    mdt_m = mdt_lookup(cx, cy)
    if np.isnan(mdt_m):
        raise ValueError(f"No valid MDT value found within {mdt_fallback_deg} deg of tile centroid ({cx}, {cy})")
    gebco_corrected = np.maximum(gebco_arr + mdt_m, min_bathymetry_m)

    combined = np.where(ocean, gebco_corrected, dem_m)

    return combined, {
        "profile": profile, "transform": transform, "crs": crs,
        "mdt_m": mdt_m, "n_ocean_nan": int(np.isnan(gebco_corrected[ocean]).sum()),
        "n_floored": int(np.nansum((gebco_arr + mdt_m)[ocean] < min_bathymetry_m)),
    }


def main() -> None:
    import argparse

    _repo_root = Path(__file__).resolve().parent.parent
    from gfm_config import read_root, resolve_catalog_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out", default=None, help="output GeoTIFF path (default: validation_sfincs/{tile_id}/sfincs_model/elevation_combined.tif)")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    catalog_path = _repo_root / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"

    tile_dir = root / "model_outputs" / args.tile_id / "inputs"
    gebco_path = resolve_catalog_path(catalog_path, root, "gebco")
    mdt_path = resolve_catalog_path(catalog_path, root, "mdt_cnes_cls22")

    combined, info = build_combined_elevation(
        tile_dir / "dem.tif", tile_dir / "mask.tif", gebco_path, mdt_path,
    )

    print(f"MDT applied to GEBCO (ADD): {info['mdt_m']:+.4f} m")
    print(f"Ocean cells with no valid GEBCO value: {info['n_ocean_nan']}")
    print(f"Ocean cells floored at {MIN_BATHYMETRY_M:.0f} m: {info['n_floored']}")
    finite = combined[np.isfinite(combined)]
    print(f"Combined elevation range: {finite.min():.2f} to {finite.max():.2f} m (n={finite.size})")

    out_path = Path(args.out) if args.out else root / "validation_sfincs" / args.tile_id / "sfincs_model" / "elevation_combined.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = info["profile"]
    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(combined.astype(np.float32), 1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
