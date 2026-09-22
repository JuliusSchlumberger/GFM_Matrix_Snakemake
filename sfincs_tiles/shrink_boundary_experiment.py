"""EXPERIMENTAL (not part of the regular pipeline): shrink an already-built
SFINCS tile's active domain to pull its water-level boundary cells closer to
the real coast, to test whether boundary-to-coast distance is driving
eikonal-vs-SFINCS over-prediction on that tile (see build_sfincs_tile.py's
own _compute_zsini_array docstring / this session's own boundary-distance
investigation - tiles 1751/1760 had boundary cells sitting 12-24km offshore
vs ~1.4km for a well-aligned control tile).

create_boundary() always places boundary cells on the perimeter of the
CURRENT active domain (mask>0) - there's no way to move them closer to
shore without first shrinking the active domain itself. This deactivates
any currently-active, real-ocean cell farther than --max-dist-km from the
tile's own real coastline (native mask.tif land cells), then re-derives
the waterlevel boundary against the smaller domain and rewrites the model.

Does NOT touch the subgrid table or the boundary forcing values (sfincs.bnd/
sfincs.bzs list real station point locations/timeseries, independent of
which grid cells end up marked as boundary - SFINCS's own kernel
interpolates from those points onto whatever cells are mask==2 at runtime,
so the existing forcing stays valid for a new, closer-in boundary).

Usage:
    python shrink_boundary_experiment.py --tile-id 1751 --max-dist-km 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry
import xarray as xr
import yaml
from hydromt_sfincs import SfincsModel
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sfincs_tile import TRUNCATE_WINDOW_HR_DEFAULT, _compute_zsini_array, _ocean_polygon_wgs84  # noqa: E402
from gfm_config import read_root  # noqa: E402


def _far_offshore_by_distance(mog: np.ndarray, mask: np.ndarray, dist_to_land_km: np.ndarray, max_dist_km: float) -> np.ndarray:
    """Real-ocean cells farther than max_dist_km from real land - the original,
    coarse distance-threshold approach."""
    return (mog == 1) & (dist_to_land_km > max_dist_km)


def _far_offshore_by_line(
    mog: np.ndarray, mask: np.ndarray, transform, crs, shape: tuple[int, int], boundary_line_path: Path,
) -> np.ndarray:
    """Real-ocean cells on the far side of a hand-drawn boundary line (an open
    curve - doesn't necessarily reach the domain edges, so a plain polygon
    split doesn't work). Rasterizes the line as a barrier, then flood-fills
    from the tile's ORIGINAL (pre-shrink) boundary cells - which sat well out
    to sea before any of this - to find every ocean cell still reachable
    without crossing the line. That reachable set is "outside" the line and
    gets excluded; ocean on the near/land side of the line, unreachable from
    the original far boundary without crossing it, stays active. Never
    touches real land cells (mog==0), by construction (real_land isn't part
    of the "ocean" passable region at all).
    """
    line_gdf = gpd.read_file(boundary_line_path).to_crs(crs)
    line = line_gdf.geometry.union_all()
    barrier = rasterio.features.rasterize(
        [(line, 1)], out_shape=shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True,
    ).astype(bool)

    ocean = mog == 1
    passable = ocean & ~barrier
    labeled, _ = ndimage.label(passable, structure=np.ones((3, 3), dtype=bool))

    orig_bnd = mask == 2
    seed_labels = set(np.unique(labeled[orig_bnd & (labeled > 0)]).tolist())
    seed_labels.discard(0)
    if not seed_labels:
        raise RuntimeError(
            "boundary line: none of the tile's original (far) boundary cells fall in a barrier-separated "
            "ocean component - can't tell which side of the line is 'outside'. Check the line actually "
            "sits between the coast and the original boundary."
        )
    return np.isin(labeled, list(seed_labels))


def shrink_boundary(
    tile_id: str, root: Path,
    max_dist_km: float | None = None,
    boundary_line_path: Path | None = None,
    truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT,
    sfincs_dir_override: Path | None = None,
) -> None:
    if (max_dist_km is None) == (boundary_line_path is None):
        raise ValueError("pass exactly one of max_dist_km or boundary_line_path")

    tile_dir = root / "model_outputs" / tile_id / "inputs"
    sfincs_dir = sfincs_dir_override or (root / "validation_sfincs" / tile_id / "sfincs_model")

    sf = SfincsModel(root=str(sfincs_dir), mode="r+")
    sf.grid.read()
    ds = sf.grid.data
    mask = ds["mask"].values
    transform = ds.raster.transform
    crs = ds.raster.crs
    shape = mask.shape

    with rasterio.open(tile_dir / "mask.tif") as src:
        mog = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mog,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    real_land = mog == 0
    dist_to_land_km = ndimage.distance_transform_edt(
        ~real_land, sampling=(abs(transform.e), abs(transform.a)),
    ) / 1000.0

    n_bnd_before = int((mask == 2).sum())
    d_bnd_before = dist_to_land_km[mask == 2]
    n_active_before = int((mask >= 1).sum())

    if boundary_line_path is not None:
        far_offshore = _far_offshore_by_line(mog, mask, transform, crs, shape, boundary_line_path)
    else:
        far_offshore = _far_offshore_by_distance(mog, mask, dist_to_land_km, max_dist_km)

    if not far_offshore.any():
        print(f"tile {tile_id}: nothing to shrink")
        return
    assert not (far_offshore & real_land).any(), "far_offshore incorrectly touches real land - refusing to proceed"

    shapes = rasterio.features.shapes(far_offshore.astype(np.uint8), mask=far_offshore, transform=transform)
    geoms = [shapely.geometry.shape(geom) for geom, _ in shapes]
    exclude_gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    sf.mask.create_active(exclude_polygon=exclude_gdf, reset_mask=False)

    # Real, confirmed issue (2026-09, tile 1751 at max_dist_km=1.5): shrinking
    # aggressively enough can fragment the active domain into disconnected
    # pieces (45 components observed - a real coastal blob plus dozens of
    # tiny isolated remnants) - create_boundary() then happily places
    # boundary cells on EVERY fragment's own perimeter, not just the real
    # coastal one, so most of the resulting "boundary" cells end up being
    # spurious artifacts on tiny far-flung islands (400/493 were >10km from
    # land) rather than the intended near-coast boundary. Restrict back down
    # to the single largest connected component before deriving the boundary
    # - the real coastal domain is always overwhelmingly the largest piece.
    active_mask = sf.grid.data["mask"].values >= 1
    labeled, n_components = ndimage.label(active_mask, structure=np.ones((3, 3), dtype=bool))
    if n_components > 1:
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, n_components + 1))
        main_label = int(np.argmax(sizes)) + 1
        n_dropped = int(active_mask.sum() - sizes[main_label - 1])
        print(f"tile {tile_id}: dropping {n_components - 1} disconnected active fragment(s) "
              f"({n_dropped} cells) - keeping only the largest ({int(sizes[main_label - 1])} cells)")
        keep = labeled == main_label
        fragments_gdf_geoms = [
            shapely.geometry.shape(geom)
            for geom, _ in rasterio.features.shapes(
                (~keep).astype(np.uint8), mask=active_mask & ~keep, transform=transform,
            )
        ]
        if fragments_gdf_geoms:
            fragments_gdf = gpd.GeoDataFrame(geometry=fragments_gdf_geoms, crs=crs)
            sf.mask.create_active(exclude_polygon=fragments_gdf, reset_mask=False)

    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=True)

    new_mask = sf.grid.data["mask"].values
    n_bnd_after = int((new_mask == 2).sum())
    n_active_after = int((new_mask >= 1).sum())
    print(f"tile {tile_id}: active cells {n_active_before} -> {n_active_after}")
    print(f"tile {tile_id}: boundary cells {n_bnd_before} -> {n_bnd_after}")
    if n_bnd_before:
        print(f"  boundary-to-land distance BEFORE: median={np.median(d_bnd_before):.2f} km  mean={d_bnd_before.mean():.2f} km")
    if n_bnd_after == 0:
        print("  WARNING: 0 boundary cells after shrink - domain may have been cut off from all ocean, aborting")
        return
    d_bnd_after = dist_to_land_km[new_mask == 2]
    print(f"  boundary-to-land distance AFTER:  median={np.median(d_bnd_after):.2f} km  mean={d_bnd_after.mean():.2f} km")

    # -- recompute zsini against the new (smaller) active domain --
    matched_points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")
    if truncate_window_hr is not None:
        t_start, t_end = truncate_window_hr
        keep = (hydrographs["elapsed_hr"] >= t_start) & (hydrographs["elapsed_hr"] <= t_end)
        hydrographs = hydrographs.loc[keep].reset_index(drop=True)
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]
    locations_utm = matched_points.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)

    grid_coords = sf.grid.data["mask"]
    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    zsini_arr = _compute_zsini_array(
        tile_dir / "mask.tif", station_x, station_y, first_vals,
        grid_coords, main_transform, main_crs, (main_height, main_width),
    )
    zsini_da = xr.DataArray(zsini_arr, dims=grid_coords.dims, coords=grid_coords.coords)
    zsini_da = zsini_da.rio.write_crs(sf.crs)
    zsini_da = zsini_da.rio.write_transform(grid_coords.rio.transform())
    sf.initial_conditions.create(zsini=zsini_da, fill_value=-9999.0, reproj_method="nearest")

    sf.grid.write()
    print(f"tile {tile_id}: grid rewritten (mask/ind/zs), ready to rerun")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max-dist-km", type=float, help="deactivate real-ocean cells farther than this from real land")
    group.add_argument("--boundary-line", help="path to a hand-drawn boundary line (.gpkg) - deactivates real-ocean cells on its far side")
    parser.add_argument("--config", default=None, help="config.yml or a resolved_config.yml; defaults to the main repo config.yml")
    parser.add_argument("--sfincs-dir", default=None, help="override the sfincs_model directory (e.g. an isolated test copy), instead of validation_sfincs/{tile_id}/sfincs_model")
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        root = Path(config["paths"]["root"])
    else:
        _repo_root = Path(__file__).resolve().parent.parent
        root = read_root(_repo_root / "snakemake_workflow" / "config" / "config.yml")

    shrink_boundary(
        args.tile_id, root,
        max_dist_km=args.max_dist_km,
        boundary_line_path=Path(args.boundary_line) if args.boundary_line else None,
        sfincs_dir_override=Path(args.sfincs_dir) if args.sfincs_dir else None,
    )


if __name__ == "__main__":
    main()
