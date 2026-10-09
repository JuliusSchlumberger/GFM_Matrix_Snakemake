"""Builds a runnable SFINCS model for one GFM tile from the prep-stage
outputs (build_elevation.py, build_roughness.py, build_boundary_forcing.py).

Every sfincs_tiles/ script runs under the same hydromt-sfincs-dev env.
src/config_utils.py is not importable there (hydromt 1.4.1 removed
`setuplog` from hydromt.log); gfm_config.py's read_root/resolve_catalog_path
cover what sfincs_tiles/ needs instead:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\hydromt-sfincs-dev\\python.exe build_sfincs_tile.py --tile-id 1907

No weirs, no restart, no discharge. Uses a subgrid: a coarse computational
grid with a finer subgrid table capturing sub-cell terrain detail (see
MAIN_RES_M_DEFAULT/SUBGRID_NR_PIXELS_DEFAULT's own comment below).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
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
from build_boundary_forcing import idw_interpolate_to_grid  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402
from tile_status import write_tile_status  # noqa: E402

_COORD_RE = re.compile(r"at (-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)")
_ANTIMERIDIAN_LON_THRESHOLD = 175.0  # within this many degrees of +-180


def _classify_water_level_create_error(e: Exception, tile_id: str) -> RuntimeError | None:
    """Recognizes hydromt_sfincs's own antimeridian-crossing masking failure
    inside `sf.water_level.create()` and turns it into a clear, actionable
    error. Returns `None` if `e` doesn't match (caller re-raises `e`
    unchanged).

    hydromt's own internal masking does a shapely union_all() in raw
    EPSG:4326 lon/lat: for a tile near +-180 deg longitude, the buffered
    search geometry can self-intersect there, which GEOS rejects regardless
    of buffer size. Not fixable via buffer tuning.

    2026-10-07 fix: matching on exception TEXT alone ("TopologyException"/
    "side location conflict") is too broad - GEOS raises the identical
    message for ANY invalid-geometry self-intersection, antimeridian or
    not (confirmed directly: tile 1262, at 120-122E/71-73N, got this exact
    label with an "Original error" coordinate of 66.96E/-57.11S - South
    Atlantic, nowhere near +-180 - a different, unrelated geometry bug
    mislabeled as antimeridian-crossing). Now also requires the error's own
    embedded coordinate (GEOS always prints "at <lon> <lat>") to actually be
    near +-180 before classifying it this way; anything else falls through
    to `other_error` instead of a confident but wrong diagnosis.
    """
    text = str(e)
    if "TopologyException" not in text and "side location conflict" not in text:
        return None
    match = _COORD_RE.search(text)
    if match is None:
        return None  # can't verify location - don't guess
    lon = float(match.group(1))
    if abs(lon) < _ANTIMERIDIAN_LON_THRESHOLD:
        return None  # real geometry error, just not this one
    return RuntimeError(
        f"tile {tile_id}: antimeridian-crossing geometry error in hydromt_sfincs's own "
        f"water_level.create() masking (tile is near +-180 deg longitude) - not fixable via "
        f"buffer tuning, drop this tile from the batch. Original error: {e}"
    )


def _validate_subgrid_params(resolution_m: float, subgrid_nr_pixels: int) -> None:
    """Validates subgrid params before any grid/reprojection work.
    hydromt_sfincs's own subgrid.create() hard-enforces `nr_subgrid_pixels`
    to be a multiple of 2 (components/grid/subgrid.py ~line 690).
    """
    if subgrid_nr_pixels <= 0 or subgrid_nr_pixels % 2 != 0:
        raise ValueError(
            f"subgrid_nr_pixels must be a positive multiple of 2 (hydromt_sfincs's own "
            f"requirement), got {subgrid_nr_pixels}"
        )
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")


def _reproject_nearest_to_grid(src_path: Path, dst_transform, dst_crs, dst_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour reprojects `src_path` onto the exact destination
    grid (`dst_transform`/`dst_crs`/`dst_shape`). Every output pixel is one
    of the source raster's own real values, never blended/interpolated -
    works around hydromt_sfincs silently ignoring `reproj_method` and always
    forcing bilinear internally (see that step's own comment).

    `src_path`'s own EPSG:4326 (lon/lat) rectangle and the destination UTM
    subgrid's rectangle are rotated relative to each other (UTM axes only
    align with lon/lat near a zone's own central meridian), so the
    destination rectangle's corners can fall just outside the source's real
    coverage even though the tile's bbox/geometry match exactly. Those gaps
    are left NaN by `dst_nodata=np.nan`, then filled via nearest-valid-cell
    inpainting (scipy.ndimage's distance-transform-to-nearest-index) -
    otherwise a NaN-contaminated coarse cell at a water-level boundary would
    corrupt that cell's own subgrid volume table.
    """
    with retry_transient_io(rasterio.open, src_path) as src:
        dst_arr = np.empty(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dst_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    missing = np.isnan(dst_arr)
    if missing.any():
        if missing.all():
            # Zero real-data overlap: nothing to nearest-fill from, so fail
            # loudly rather than hand hydromt_sfincs an all-garbage subgrid source.
            raise ValueError(
                f"{src_path}: reprojected onto the destination subgrid with ZERO valid cells - "
                "no real data to nearest-fill from. Check the source file's own coverage against "
                "this tile's bbox."
            )
        nearest_idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        dst_arr = dst_arr[tuple(nearest_idx)]

    assert not np.isnan(dst_arr).any(), f"{src_path}: NaN survived the nearest-valid-neighbour fill - should be impossible"
    return dst_arr


def _compute_zsini_array(
    native_mask_path: Path, station_x: np.ndarray, station_y: np.ndarray, station_values: np.ndarray,
    grid_coords: xr.DataArray, dst_transform, dst_crs, dst_shape: tuple[int, int],
) -> np.ndarray:
    """IDW-interpolated initial water level, kept only on real ocean cells
    (native mask.tif land=0/lake=2/river=3; any ocean-coded cell counts,
    including isolated blobs). Non-ocean cells get hydromt_sfincs's own
    "no initial water" sentinel (-9999.0), which SFINCS treats as dry at bed
    level - otherwise inland land would start the simulation already
    "flooded" from the interpolated coastal water level.

    Cell centres come from `dst_transform` (rotation-aware), not from
    grid_coords' "x"/"y": on a rotated grid those are plain pixel indices
    (see the forcing step's own comment), which put every cell ~equally far
    from all stations and gave a uniform zsini (the station mean) - fixed
    2026-10-09. `grid_coords` is kept in the signature for callers only.
    """
    rows, cols = np.indices(dst_shape)
    xx, yy = dst_transform * (cols + 0.5, rows + 0.5)
    zsini_arr = idw_interpolate_to_grid(station_x, station_y, station_values, xx, yy).astype(np.float32)

    native_mask_on_grid = _reproject_nearest_to_grid(native_mask_path, dst_transform, dst_crs, dst_shape)
    ocean = native_mask_on_grid == 1  # OCEAN_CODE, native mask.tif convention (0=land,1=ocean,2=lake,3=river)
    return np.where(ocean, zsini_arr, np.float32(-9999.0))


def _ocean_polygon_wgs84(mask_path: Path, ocean_code: int = 1) -> gpd.GeoDataFrame:
    """Vectorize mask.tif's ocean cells into a polygon GeoDataFrame (EPSG:4326)."""
    with retry_transient_io(rasterio.open, mask_path) as src:
        mask = src.read(1)
        transform = src.transform
    ocean = (mask == ocean_code).astype(np.uint8)
    shapes = rasterio.features.shapes(ocean, mask=ocean.astype(bool), transform=transform)
    geoms = [shapely.geometry.shape(geom) for geom, val in shapes]
    if not geoms:
        raise ValueError(f"{mask_path}: no ocean (code={ocean_code}) cells found")
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def _create_mask(sf: SfincsModel, tile_gdf: gpd.GeoDataFrame, ocean_poly: gpd.GeoDataFrame,
                 boundary_lines_gpkg: Path | None, resolution_m: float) -> None:
    """Active cells + waterlevel boundary cells.

    Default: the whole tile active, boundary = active-domain edge cells in the
    ocean. With `boundary_lines_gpkg` (build_station_boundary_lines.py output):
    active = the tile minus the line's inactive (ocean-side) faces, boundary =
    the active-domain edge cells along the boundary line (incl. its tile-edge
    stretches) - the model then ends at the line instead of the open-sea tile edge.
    """
    if boundary_lines_gpkg is None:
        sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
        sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False, all_touched=True)
        return
    tile_4326 = tile_gdf.to_crs("EPSG:4326")
    inactive = retry_transient_io(gpd.read_file, boundary_lines_gpkg, layer="inactive").to_crs("EPSG:4326")
    active = tile_4326.geometry.iloc[0].difference(inactive.union_all()) if len(inactive) else tile_4326.geometry.iloc[0]
    sf.mask.create_active(include_polygon=gpd.GeoDataFrame(geometry=[active], crs="EPSG:4326"), reset_mask=True)
    lines = retry_transient_io(gpd.read_file, boundary_lines_gpkg, layer="boundary_line").to_crs(sf.crs)
    band = gpd.GeoDataFrame(geometry=[lines.union_all().buffer(1.5 * resolution_m)], crs=sf.crs)
    sf.mask.create_boundary(btype="waterlevel", include_polygon=band, reset_bounds=False, all_touched=True)


TRUNCATE_WINDOW_HR_DEFAULT = (40.0, 110.0)  # truncation window (h) around COAST-HG's
# storm peak: every hydrograph in this pipeline shares the same synthetic time axis
# (peak at t=74.5h), so this window is a property of the dataset, not per-tile tuning.

# Subgrid: coarse computational grid at MAIN_RES_M, with a SUBGRID_NR_PIXELS-times-finer
# subgrid table (hypsometric volume/roughness-depth relationships per coarse cell)
# capturing sub-cell terrain detail without running the whole simulation at that finer
# resolution - subgrid table construction is a one-time preprocessing cost only.
# nr_subgrid_pixels must be a multiple of 2 (hydromt_sfincs's own hard-enforced check,
# hydromt_sfincs/components/grid/subgrid.py ~line 690).
#
# 120m / 30m (SUBGRID_NR_PIXELS=4): 30m subgrid pixels match DeltaDTM's own native
# resolution (~30m); finer would let hydromt_sfincs's forced-bilinear interpolation (see
# subgrid.create()'s own comment below) fabricate spurious sub-pixel depressions.
MAIN_RES_M_DEFAULT = 120.0
SUBGRID_NR_PIXELS_DEFAULT = 4  # -> 120/4 = 30m subgrid resolution, DeltaDTM's own native scale
SUBGRID_NR_LEVELS_DEFAULT = 20  # hypsometric bins; memory scales linearly with this
SUBGRID_NRMAX_DEFAULT = 2000  # matches hydromt_sfincs's own default (tile/block size for
# subgrid table construction)



def build_sfincs_tile(
    tile_id: str, root: Path, resolution_m: float = MAIN_RES_M_DEFAULT,
    dtmapout_s: float = 1800.0, tref: datetime | None = None,
    truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT,
    subgrid_nr_pixels: int = SUBGRID_NR_PIXELS_DEFAULT,
    subgrid_nr_levels: int = SUBGRID_NR_LEVELS_DEFAULT,
    subgrid_nrmax: int = SUBGRID_NRMAX_DEFAULT,
    base_dir_name: str = "validation_sfincs_v2",
    rotated: bool = True,
    boundary_lines_gpkg: Path | None = None,
) -> Path:
    _validate_subgrid_params(resolution_m, subgrid_nr_pixels)
    # Reads from the tile's working copy, not model_outputs/ directly.
    tile_dir = root / base_dir_name / tile_id / "inputs"
    sfincs_dir = root / base_dir_name / tile_id / "sfincs_model"
    sfincs_dir.mkdir(parents=True, exist_ok=True)

    tile_gdf = retry_transient_io(gpd.read_file, tile_dir / "tile_geometry.gpkg")

    # -- local data catalog: elevation.create/roughness.create need catalog-keyed
    # sources, not raw DataArrays --
    local_catalog_path = sfincs_dir / "data_catalog_local.yml"
    local_catalog = {
        "meta": {"root": str(sfincs_dir)},
        "local_elevation": {"data_type": "RasterDataset", "uri": "elevation_combined.tif", "driver": "rasterio"},
        "local_roughness": {"data_type": "RasterDataset", "uri": "manning_n.tif", "driver": "rasterio"},
    }
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)

    # -- 1. grid (coarse computational grid, MAIN_RES_M - see subgrid note above) --
    # rotated=True (2026-10, user direction - was left at hydromt_sfincs's own
    # library default of False/axis-aligned UTM until now): a rotated grid
    # aligns its rows/columns with the tile's own coastline/flow direction
    # instead of bare UTM axes, which is what create_from_region's own
    # rotation search actually optimizes for when given a real region
    # geometry (not an arbitrary bbox) - axis-aligned wastes active cells on
    # dry corners for any coastline that isn't already UTM-axis-parallel.
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm", rotated=rotated)
    print(f"[1/8] grid created: {dict(sf.grid.data.sizes)} cells, crs={sf.crs}, rotated={rotated}")

    # -- 2. mask: active cells (whole tile) + waterlevel boundary (ocean edge only) --
    # Built before subgrid: subgrid.create()'s own internals read self.model.grid.mask
    # directly, so it must already exist.
    #
    # all_touched=True: hydromt_sfincs's own create_boundary() defaults to a
    # center-point test, which produces a sparse, broken boundary line on a coastline
    # running diagonally across this tile's rotated UTM grid. all_touched=True
    # includes every cell the ocean polygon touches at all, giving a continuous
    # boundary regardless of grid rotation.
    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")
    _create_mask(sf, tile_gdf, ocean_poly, boundary_lines_gpkg, resolution_m)
    n_active = int((sf.grid.data["mask"] > 0).sum())
    n_bnd = int((sf.grid.data["mask"] == 2).sum())
    print(f"[2/8] mask: {n_active} active cell(s), {n_bnd} waterlevel-boundary cell(s)")
    if n_bnd == 0:
        raise RuntimeError(
            f"tile {tile_id}: 0 waterlevel-boundary cells after create_boundary - "
            "the ocean polygon didn't reach any active-domain edge cell on this grid. "
            "No weir/discharge fallback exists in this pipeline - this tile can't be forced."
        )

    # -- 3. subgrid table: combined elevation (build_elevation.py) and Manning's n
    # (build_roughness.py), pre-reprojected onto the exact fine subgrid grid ourselves
    # (nearest-neighbour) before calling subgrid.create().
    #
    # Nearest, not hydromt_sfincs's own forced-bilinear default: bilinear interpolation
    # below DeltaDTM's own native resolution fabricates artificial depressions with no
    # real source support. Nearest keeps every subgrid pixel traceable to a real
    # DeltaDTM/GEBCO sample, and pre-reprojecting onto the exact destination grid
    # ourselves makes hydromt's own forced-bilinear pass a no-op.
    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    fine_transform = main_transform * main_transform.scale(1.0 / subgrid_nr_pixels)
    fine_height, fine_width = main_height * subgrid_nr_pixels, main_width * subgrid_nr_pixels

    # `_reproject_nearest_to_grid()` guarantees this array has no NaN anywhere
    # (see its own docstring). hydromt_sfincs's own downstream re-read of this
    # file (inside subgrid.create()) can reintroduce NaN, but only in padding
    # cells outside the tile's active domain, which the volume-table
    # construction skips entirely.
    subgrid_sources = {}
    for name, src_uri in [("local_elevation_subgrid", "elevation_combined.tif"), ("local_roughness_subgrid", "manning_n.tif")]:
        out_path = sfincs_dir / f"{Path(src_uri).stem}_subgrid_src.tif"
        fine_arr = _reproject_nearest_to_grid(sfincs_dir / src_uri, fine_transform, main_crs, (fine_height, fine_width))
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": fine_height, "width": fine_width,
            "transform": fine_transform, "crs": main_crs, "nodata": np.nan, "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(fine_arr, 1)
        subgrid_sources[name] = out_path.name

    local_catalog["local_elevation_subgrid"] = {"data_type": "RasterDataset", "uri": subgrid_sources["local_elevation_subgrid"], "driver": "rasterio"}
    local_catalog["local_roughness_subgrid"] = {"data_type": "RasterDataset", "uri": subgrid_sources["local_roughness_subgrid"], "driver": "rasterio"}
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)
    # Reconstruct sf so its DataCatalog re-reads local_catalog_path with the new
    # entries - DataCatalog is parsed once at construction, not re-read live.
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm", rotated=rotated)
    _create_mask(sf, tile_gdf, ocean_poly, boundary_lines_gpkg, resolution_m)

    # write_dep_tif=True writes subgrid/dep_subgrid.tif, the fine-resolution DEM
    # run_sfincs_tile.py's own postprocessing needs for downscale_floodmap().
    sf.subgrid.create(
        elevation_list=[{"elevation": "local_elevation_subgrid"}],
        roughness_list=[{"manning": "local_roughness_subgrid"}],
        nr_subgrid_pixels=subgrid_nr_pixels,
        nr_levels=subgrid_nr_levels,
        nrmax=subgrid_nrmax,
        write_dep_tif=True,
        write_man_tif=True,
    )
    print(f"[3/8] subgrid table created: {subgrid_nr_pixels}x refinement "
          f"({resolution_m:.0f}m main / {resolution_m / subgrid_nr_pixels:.0f}m subgrid), "
          f"{subgrid_nr_levels} levels, nearest-sourced")

    # -- 4. boundary forcing (COAST-HG hydrographs, MDT-corrected by
    # build_boundary_forcing.py) --
    matched_points = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = retry_transient_io(pd.read_csv, sfincs_dir / "corrected_hydrographs.csv")

    # Truncates the full ~148.8h COAST-HG hydrograph to a window around the storm
    # peak (SFINCS's wall-clock cost scales with simulated duration). Every
    # hydrograph in this pipeline shares the same synthetic time axis (peak at
    # t=74.5h, with hour 40/110 near the tidal-only baseline). zsini below uses
    # the truncated series' own first row.
    if truncate_window_hr is not None:
        t_start, t_end = truncate_window_hr
        keep = (hydrographs["elapsed_hr"] >= t_start) & (hydrographs["elapsed_hr"] <= t_end)
        hydrographs = hydrographs.loc[keep].reset_index(drop=True)
        hydrographs["elapsed_hr"] = hydrographs["elapsed_hr"] - hydrographs["elapsed_hr"].iloc[0]

    elapsed_hr = hydrographs["elapsed_hr"].to_numpy()
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]

    tref = tref or datetime(2026, 1, 1)  # arbitrary but fixed reference; only elapsed
    # time within the hydrograph is physically meaningful
    times = pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in elapsed_hr])
    wl_df = hydrographs[station_cols].copy()
    wl_df.index = times
    wl_df.columns = range(len(station_cols))  # water_level.create expects positional columns matching locations' row order

    # matched_points.gpkg and corrected_hydrographs.csv share row order (written
    # together by build_boundary_forcing.py's own CLI). water_level.create()
    # requires an explicit "index" column matching timeseries' column names.
    locations_gdf = matched_points.copy()
    locations_gdf["index"] = range(len(station_cols))

    sf.config.set("tref", tref)
    sf.config.set("tstart", tref)
    tstop = times[-1]
    sf.config.set("tstop", tstop)

    # buffer: computed dynamically from the max distance of any k-nearest-filtered
    # station to the model's own grid extent, since water_level.create() masks every
    # location out (hydromt raises `NoDataException`) if none remain within the buffer.
    # "mask", not "dep": with subgrid, sf.grid.data has no "dep" variable any more
    # (elevation lives only in the subgrid table); "mask" exists from step 2 above,
    # and only its real-world extent is needed here.
    #
    # grid_coords.raster.bounds, NOT grid_coords["x"]/["y"].min()/max() - a REAL
    # bug, found live 2026-10 once rotated=True became the default: for a rotated
    # grid, hydromt_sfincs's own "x"/"y" dim coordinates are plain PIXEL INDICES
    # (e.g. 0..723), not real-world UTM metres - the real per-cell position lives
    # in separate 2D curvilinear "xc"/"yc" arrays instead (confirmed directly:
    # `sf.grid.data.variables` is ['yc','xc','spatial_ref','mask'] when rotated,
    # vs plain ['y','x','spatial_ref','mask'] when not). Comparing those tiny
    # pixel-index numbers against real station UTM coordinates produced a ~7,445
    # km bogus "distance to grid", hence a ~14,914 km buffer_m, which then made
    # water_level.create()'s own internal masking buffer a self-intersecting
    # polygon once reprojected to WGS84 - the actual cause of the
    # "antimeridian-crossing" exception below (mislabeled: GEOS's own
    # TopologyException message doesn't know why the geometry is invalid, and
    # the antimeridian case happens to raise the identical exception type/text,
    # but the real self-intersection coordinates this produced were nowhere near
    # +-180 deg - see the tile-5 troubleshooting session this was found in).
    # `.raster.bounds` is rioxarray's own rotation-aware real-world bounding box -
    # correct (and numerically identical to the old "x"/"y" min/max approach) for
    # a non-rotated grid too, so this is a strict fix, not a rotated-only special case.
    grid_coords = sf.grid.data["mask"]
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = grid_coords.raster.bounds
    locations_utm = locations_gdf.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()
    dx = np.maximum(np.maximum(grid_x_min - station_x, station_x - grid_x_max), 0.0)
    dy = np.maximum(np.maximum(grid_y_min - station_y, station_y - grid_y_max), 0.0)
    dist_to_grid_bbox = float(np.sqrt(dx ** 2 + dy ** 2).max())
    # 2x + a flat 25 km margin: errs safely past the naive bbox-corner distance
    # estimate, since hydromt's own internal masking distance isn't simply that.
    # Only already k-nearest-filtered locations exist to include, so a generous
    # buffer carries no risk of pulling in an unrelated station.
    buffer_m = dist_to_grid_bbox * 2.0 + 25_000.0

    try:
        sf.water_level.create(timeseries=wl_df, locations=locations_gdf, buffer=buffer_m)
    except Exception as e:
        wrapped = _classify_water_level_create_error(e, tile_id)
        if wrapped is not None:
            raise wrapped from e
        raise
    print(f"[4/8] water-level forcing: {len(station_cols)} station(s), {len(times)} timestep(s), "
          f"{tref} -> {tstop} ({elapsed_hr[-1]:.1f} h), buffer={buffer_m / 1000:.1f} km")

    # -- 5. initial conditions (zsini): IDW of the matched stations' own
    # first-timestep corrected value, onto this model's own (coarse) UTM grid -
    # a per-computational-cell initial condition, unaffected by subgrid. grid_coords
    # is the "mask" DataArray from step 2, used purely as a coords/dims/transform
    # template. Kept only on cells the native mask marks as open water - see
    # `_compute_zsini_array`'s own docstring.
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)
    zsini_arr = _compute_zsini_array(
        tile_dir / "mask.tif", station_x, station_y, first_vals,
        grid_coords, main_transform, main_crs, (main_height, main_width),
    )

    zsini_da = xr.DataArray(zsini_arr, dims=grid_coords.dims, coords=grid_coords.coords)
    zsini_da = zsini_da.rio.write_crs(sf.crs)
    zsini_da = zsini_da.rio.write_transform(grid_coords.rio.transform())

    sf.initial_conditions.create(zsini=zsini_da, fill_value=-9999.0, reproj_method="nearest")
    is_ocean_wet = zsini_arr > -9999.0
    n_wet = int(is_ocean_wet.sum())
    print(f"[5/8] zsini set - {n_wet} ocean cell(s) initialized "
          f"{float(zsini_arr[is_ocean_wet].min()):.4f} to {float(zsini_arr[is_ocean_wet].max()):.4f} m "
          f"(interpolated from {len(station_x)} station(s)); land/river/lake left dry (-9999.0, bed-level fallback)"
          if n_wet else "[5/8] zsini set - WARNING: 0 ocean cells found, every cell left dry")

    # -- 6. output config: dtmaxout spans the whole simulation, so the zsmax
    # envelope is one true whole-run maximum (SFINCS default dtmaxout=86400s=1
    # day is shorter than most ~6.2-day COAST-HG events). Only max-inundation/
    # extent is needed, so history/velocity/wet-duration outputs stay off. --
    total_span_s = elapsed_hr[-1] * 3600.0
    sf.config.set("dtmaxout", total_span_s + 3600.0)  # +1h margin, never triggers a reset
    sf.config.set("dtmapout", dtmapout_s)
    sf.config.set("baro", 0)
    print(f"[6/8] output config: dtmaxout={total_span_s + 3600.0:.0f}s (whole-run max), dtmapout={dtmapout_s:.0f}s")

    # -- 7. write --
    sf.write()
    print(f"[7/8] wrote SFINCS model to {sfincs_dir}")
    return sfincs_dir


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--resolution-m", type=float, default=MAIN_RES_M_DEFAULT, help="main computational grid resolution")
    parser.add_argument("--subgrid-nr-pixels", type=int, default=SUBGRID_NR_PIXELS_DEFAULT, help="subgrid refinement factor (must be a multiple of 2) - subgrid res = resolution-m / this")
    parser.add_argument("--subgrid-nr-levels", type=int, default=SUBGRID_NR_LEVELS_DEFAULT)
    parser.add_argument("--subgrid-nrmax", type=int, default=SUBGRID_NRMAX_DEFAULT)
    parser.add_argument(
        "--no-truncate", action="store_true",
        help="use the full ~148.8h COAST-HG hydrograph instead of the default "
             f"{TRUNCATE_WINDOW_HR_DEFAULT} window - for A/B comparison only.",
    )
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2", help="output root directory name under paths.root (default: validation_sfincs_v2)")
    parser.add_argument(
        "--boundary-lines-gpkg", default=None,
        help="build_station_boundary_lines.py output for this tile: mask the ocean side of its "
             "boundary line inactive and force along the line (default: whole tile active, "
             "forced along the open-sea tile edge)",
    )
    parser.add_argument(
        "--no-rotated", dest="rotated", action="store_false",
        help="use an axis-aligned UTM grid instead of the default rotated grid (2026-10 default flip - "
             "was unrotated until now, see build_sfincs_tile's own comment)",
    )
    args = parser.parse_args()

    root = read_root(Path(args.config))
    truncate_window_hr = None if args.no_truncate else TRUNCATE_WINDOW_HR_DEFAULT
    try:
        build_sfincs_tile(
            args.tile_id, root, resolution_m=args.resolution_m, truncate_window_hr=truncate_window_hr,
            subgrid_nr_pixels=args.subgrid_nr_pixels, subgrid_nr_levels=args.subgrid_nr_levels, subgrid_nrmax=args.subgrid_nrmax,
            base_dir_name=args.base_dir_name, rotated=args.rotated,
            boundary_lines_gpkg=Path(args.boundary_lines_gpkg) if args.boundary_lines_gpkg else None,
        )
    except Exception as e:
        text = str(e)
        if "antimeridian-crossing" in text:
            status = "antimeridian"
        elif "0 waterlevel-boundary cells" in text:
            status = "no_boundary_cells"
        else:
            status = "other_error"
        write_tile_status(root, args.base_dir_name, args.tile_id, status=status, stage="build_sfincs_tile.py", message=text[:500])
        raise


if __name__ == "__main__":
    main()
