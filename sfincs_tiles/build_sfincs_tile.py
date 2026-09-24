"""Build a runnable SFINCS model for one GFM tile, from the prep-stage
outputs (build_elevation.py, build_roughness.py, build_boundary_forcing.py).

Every sfincs_tiles/ script, including the prep stage, runs under the SAME
single hydromt-sfincs-dev env - none of them import src/config_utils.py
(confirmed 2026-09: importing it at all fails under this env's hydromt
1.4.1, `ImportError: cannot import name 'setuplog' from 'hydromt.log'` -
removed in hydromt's 1.x rewrite; config_utils.py was written against the
main pipeline's hydromt 0.9.3). gfm_config.py's read_root/resolve_catalog_path
cover the small amount sfincs_tiles/ actually needs from it. No separate
gfm_python_preprocessing env or `conda run` needed for anything in this
folder:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\hydromt-sfincs-dev\\python.exe build_sfincs_tile.py --tile-id 1907

No weirs, no restart, no discharge - see sfincs_tiles' own plan doc for the
full design/reasoning. Uses a subgrid (2026-09, replaces the earlier
single-resolution grid - see MAIN_RES_M_DEFAULT/SUBGRID_NR_PIXELS_DEFAULT's
own comment below for the resolution choice and why).
"""

from __future__ import annotations

import argparse
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


def _classify_water_level_create_error(e: Exception, tile_id: str) -> RuntimeError | None:
    """Recognize hydromt_sfincs's own antimeridian-crossing masking failure
    inside `sf.water_level.create()` and turn it into a clear, actionable
    error - returns `None` if `e` doesn't match (caller should re-raise `e`
    itself unchanged).

    Known, separate limitation (not something the caller's buffer formula
    can fix): hydromt's own internal masking does a shapely union_all() in
    raw EPSG:4326 lon/lat - for a tile near the antimeridian (confirmed
    live: tiles 2029/2077, both in the Chukchi Sea around -179 to -178 deg
    lon), the buffered search geometry can straddle +-180 deg and produce a
    self-intersecting polygon there, which GEOS rejects as an invalid
    topology regardless of how tight or generous the buffer is (confirmed:
    a SMALLER buffer briefly "worked" for tile 2077 only by accident, not
    fixing anything - a marginally bigger one immediately hit the same
    failure). Not worth chasing inside this pipeline (would mean patching
    hydromt_sfincs's own dateline handling) - surfaced as a clear,
    actionable error instead of a raw GEOS traceback, so a batch run's
    failure log says WHY, and this tile can be dropped like any other
    "can't be forced" case.
    """
    if "TopologyException" in str(e) or "side location conflict" in str(e):
        return RuntimeError(
            f"tile {tile_id}: antimeridian-crossing geometry error in hydromt_sfincs's own "
            f"water_level.create() masking (tile is near +-180 deg longitude) - not fixable via "
            f"buffer tuning, drop this tile from the batch. Original error: {e}"
        )
    return None


def _validate_subgrid_params(resolution_m: float, subgrid_nr_pixels: int) -> None:
    """Fail fast, before any grid/reprojection work: hydromt_sfincs's own
    subgrid.create() hard-enforces `nr_subgrid_pixels` to be a multiple of 2
    (components/grid/subgrid.py ~line 690) - checking here gives a clear,
    immediate error instead of discovering it after the main grid, mask,
    AND the (non-trivial, nearest-neighbour) pre-reprojection step have
    already run.
    """
    if subgrid_nr_pixels <= 0 or subgrid_nr_pixels % 2 != 0:
        raise ValueError(
            f"subgrid_nr_pixels must be a positive multiple of 2 (hydromt_sfincs's own "
            f"requirement), got {subgrid_nr_pixels}"
        )
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")


def _reproject_nearest_to_grid(src_path: Path, dst_transform, dst_crs, dst_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour reproject `src_path` onto the EXACT destination
    grid (`dst_transform`/`dst_crs`/`dst_shape`) - the pre-reprojection
    workaround `build_sfincs_tile()`'s own step 3 needs, since
    hydromt_sfincs silently ignores `reproj_method` and always forces
    bilinear internally (see that step's own comment). Every output pixel
    is guaranteed to be one of the source raster's own real values, never a
    blended/interpolated one - that guarantee is the entire point of this
    function, and is what its own test validates.

    Real, confirmed gap (2026-09-24, tile 1736): `src_path`'s own EPSG:4326
    (lon/lat) rectangle and the destination UTM subgrid's rectangle are
    rotated relative to each other (UTM axes only align with lon/lat near a
    zone's own central meridian) - the destination rectangle's CORNERS can
    fall just outside the source's real coverage even though the tile's own
    bbox/geometry match exactly (confirmed: model_bbox.json, tile_geometry.
    gpkg, dem.tif and elevation_combined.tif all share the same extent for
    that tile - this isn't an extent mismatch, purely a rotation effect).
    Left NaN by `dst_nodata=np.nan` above, this measured 1.9% of cells, 84%
    of those within 5 buffer pixels of a tile edge - exactly where SFINCS's
    own water-level boundary cells are placed, so a NaN-contaminated coarse
    cell there corrupts that cell's own subgrid volume table (NaN survives
    `np.maximum(elevation, zvmin)` in hydromt_sfincs's own subgrid_v_table).
    Filled here via nearest-valid-cell inpainting (scipy.ndimage's own
    distance-transform-to-nearest-index trick) rather than by widening the
    read window - this works by construction for ANY rotation severity,
    including the most extreme case in this batch (tile 2084, 82.7 deg N,
    confirmed zero remaining NaN after this fix), without needing to
    re-architect elevation_combined.tif/manning_n.tif's own 1:1 pixel match
    to dem.tif/mask.tif.
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
            # Genuinely zero real-data overlap - the nearest-valid-neighbour
            # fill has nothing to fill FROM, so failing loudly here is safer
            # than silently handing hydromt_sfincs an all-garbage subgrid
            # source (which is exactly the kind of silent boundary-adjacent
            # corruption this whole fix exists to prevent).
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
    """IDW-interpolated initial water level, kept only on real ocean cells.

    IDW is evaluated everywhere (same as before), but any cell that isn't
    ocean-coded in the tile's own native mask.tif (land=0, lake=2, river=3 -
    any isolated/disconnected ocean-coded blob still counts as ocean here,
    kept exactly as interpolated, per 2026-09 review) gets overwritten with
    hydromt_sfincs's own official "no initial water" sentinel (-9999.0 - see
    SfincsInitialConditions.create's own docstring: "For cells with initial
    water levels of -9999.0, the SFINCS kernel will set the initial water
    level to the bed level", i.e. dry, zero depth) instead of the IDW value -
    see build_sfincs_tile()'s own step 5 for why (real, confirmed bug,
    tiles 1702/1798: land far inland was starting the simulation already
    "flooded" from the interpolated coastal water level alone).
    """
    yy, xx = np.meshgrid(grid_coords["y"].values, grid_coords["x"].values, indexing="ij")
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


TRUNCATE_WINDOW_HR_DEFAULT = (40.0, 110.0)  # see build_sfincs_tile()'s own comment at the
# boundary-forcing step for why: every COAST-HG hydrograph in this pipeline shares the exact
# same synthetic time axis (confirmed live across all 4 real tiles tested so far - peak always
# at t=74.5h), so this window isn't a per-tile tuning choice, it's a property of the dataset.

# Subgrid (2026-09, replaces the old single-resolution 30m grid): coarse COMPUTATIONAL
# grid at MAIN_RES_M, with a SUBGRID_NR_PIXELS-times-finer subgrid table (hypsometric
# volume/roughness-depth relationships per coarse cell) capturing real sub-cell terrain
# detail without paying the per-timestep cost of running the whole simulation at that
# finer resolution - subgrid table construction is a one-time PREPROCESSING cost only.
# nr_subgrid_pixels must be a multiple of 2 (hydromt_sfincs's own hard-enforced check,
# hydromt_sfincs/components/grid/subgrid.py ~line 690).
#
# 120m / 30m (SUBGRID_NR_PIXELS=4), REVISED from an initial 90m/15m choice (2026-09, real
# A/B test against the eikonal model on tiles 1573/1907): 15m subgrid pixels are FINER
# than DeltaDTM's own real native resolution (~30m), and hydromt_sfincs's own forced-
# bilinear interpolation (see subgrid.create()'s own comment below) at that over-fine
# scale fabricated small spurious depressions with no real source support - confirmed
# live on tile 1907, which showed 7-9m of "flooding" at 15m that vanished once the
# subgrid resolution was coarsened back toward DeltaDTM's own native ~30m scale. 30m
# subgrid pixels sample real, distinct DeltaDTM cells instead of interpolating below
# them, and matched both the eikonal model's own flood extent and this pipeline's
# earlier pre-subgrid single-resolution SFINCS runs far more closely.
MAIN_RES_M_DEFAULT = 120.0
SUBGRID_NR_PIXELS_DEFAULT = 4  # -> 120/4 = 30m subgrid resolution, DeltaDTM's own native scale
SUBGRID_NR_LEVELS_DEFAULT = 20  # hypsometric bins; memory scales linearly with this, but the
# main grid's own much lower cell count (120m vs the old 30m - 16x fewer cells) more than
# offsets doubling this from hydromt_sfincs's own default of 10.
SUBGRID_NRMAX_DEFAULT = 2000  # matches hydromt_sfincs's own default (tile/block size for
# subgrid table construction) - no reason found to deviate from it.



def build_sfincs_tile(
    tile_id: str, root: Path, resolution_m: float = MAIN_RES_M_DEFAULT,
    dtmapout_s: float = 1800.0, tref: datetime | None = None,
    truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT,
    subgrid_nr_pixels: int = SUBGRID_NR_PIXELS_DEFAULT,
    subgrid_nr_levels: int = SUBGRID_NR_LEVELS_DEFAULT,
    subgrid_nrmax: int = SUBGRID_NRMAX_DEFAULT,
    base_dir_name: str = "validation_sfincs_v2",
) -> Path:
    _validate_subgrid_params(resolution_m, subgrid_nr_pixels)
    # Read from THIS tile's own working copy, not model_outputs/ directly -
    # see build_elevation.py's own note on this same gap.
    tile_dir = root / base_dir_name / tile_id / "inputs"
    sfincs_dir = root / base_dir_name / tile_id / "sfincs_model"
    sfincs_dir.mkdir(parents=True, exist_ok=True)

    tile_gdf = retry_transient_io(gpd.read_file, tile_dir / "tile_geometry.gpkg")

    # -- local data catalog for elevation.create/roughness.create (both need
    # catalog-keyed sources, confirmed via the real hydromt_sfincs API -
    # not a raw-DataArray-accepting signature) --
    local_catalog_path = sfincs_dir / "data_catalog_local.yml"
    local_catalog = {
        "meta": {"root": str(sfincs_dir)},
        "local_elevation": {"data_type": "RasterDataset", "uri": "elevation_combined.tif", "driver": "rasterio"},
        "local_roughness": {"data_type": "RasterDataset", "uri": "manning_n.tif", "driver": "rasterio"},
    }
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)

    # -- 1. grid (coarse computational grid, MAIN_RES_M - see subgrid note above) --
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    print(f"[1/8] grid created: {dict(sf.grid.data.sizes)} cells, crs={sf.crs}")

    # -- 2. mask: active cells (whole tile) + waterlevel boundary (ocean edge only) --
    # Before subgrid, not after - matches hydromt_sfincs's own real reference usage
    # (delta_model/code/step1_build.py: grid -> mask -> ... -> subgrid.create()) and
    # subgrid.create()'s own internals read self.model.grid.mask directly, so it must
    # already exist. Uses tile_gdf/ocean_poly geometry directly, no dependency on
    # elevation - unaffected by not calling elevation.create() separately any more.
    #
    # all_touched=True - real, confirmed gap (2026-09-24, tile 1736): hydromt_sfincs's
    # own create_boundary() defaults to all_touched=False (its own function signature -
    # NOT what its own docstring claims, "True (default)"; a genuine doc/code mismatch),
    # which only includes a cell in the boundary if the OCEAN POLYGON's own geometry
    # happens to cover that cell's CENTER point. For a coastline running diagonally
    # across this tile's own rotated UTM grid (the tile's true rectangular lon/lat
    # geometry becomes a rotated shape in UTM), a center-point test is much stricter
    # than "does the polygon touch this cell at all" and produces a sparse, broken,
    # dotted boundary line instead of a continuous one - confirmed live: whole-tile
    # boundary coverage was a set of short disconnected segments along all four edges,
    # not the continuous line expected given ocean occupies most of three of those
    # edges. all_touched=True includes every cell the polygon touches at all, matching
    # the true coastline far more continuously regardless of grid rotation.
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False, all_touched=True)
    n_active = int((sf.grid.data["mask"] > 0).sum())
    n_bnd = int((sf.grid.data["mask"] == 2).sum())
    print(f"[2/8] mask: {n_active} active cell(s), {n_bnd} waterlevel-boundary cell(s)")
    if n_bnd == 0:
        raise RuntimeError(
            f"tile {tile_id}: 0 waterlevel-boundary cells after create_boundary - "
            "the ocean polygon didn't reach any active-domain edge cell on this grid. "
            "No weir/discharge fallback exists in this pipeline - this tile can't be forced."
        )

    # -- 3. subgrid table: combined DeltaDTM+MDT-corrected-GEBCO elevation (build_elevation.py)
    # + Manning's n (friction.tif decoded by build_roughness.py), pre-reprojected onto the
    # EXACT fine subgrid grid ourselves (nearest-neighbour) before calling subgrid.create() -
    # replaces the old separate elevation.create()/roughness.create() calls entirely
    # (single-resolution grid has no subgrid table).
    #
    # Nearest, not hydromt_sfincs's own forced-bilinear default (2026-09, revised choice):
    # bilinear was deliberately kept THROUGH the first subgrid A/B test for its hypsometric-
    # curve-smoothing benefit, but comparing against the eikonal model exposed a real cost -
    # bilinear interpolation below DeltaDTM's own real native resolution created small
    # artificial depressions that don't exist in the source data (confirmed live: tile 1907
    # showed 7-9m of "flooding" that vanished once the subgrid resolution was coarsened back
    # toward the native ~30m scale). Nearest keeps every subgrid pixel traceable to a REAL
    # DeltaDTM/GEBCO sample - the eikonal model's own dem.tif is itself nearest-sourced at
    # native resolution, so this also minimises the two models' pixel-value divergence, which
    # was the deeper motivation (see conversation on unifying the two models' own grids).
    # Same "pre-reproject to the exact destination grid so hydromt's own forced-bilinear pass
    # becomes a no-op" workaround as the old single-resolution code's elevation step - just at
    # the FINE subgrid resolution this time, not the coarse main-grid resolution.
    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    fine_transform = main_transform * main_transform.scale(1.0 / subgrid_nr_pixels)
    fine_height, fine_width = main_height * subgrid_nr_pixels, main_width * subgrid_nr_pixels

    # `_reproject_nearest_to_grid()` guarantees this array has no NaN
    # anywhere (nearest-valid-neighbour fill for any rotation-induced gap
    # between the source's own lon/lat rectangle and this UTM subgrid's
    # rotated footprint - see that function's own docstring). Investigated
    # 2026-09-24 (tile 1736) whether hydromt_sfincs's own downstream re-read
    # of this file (inside subgrid.create()) could reintroduce NaN despite
    # that - it can, but ONLY in cells outside the tile's own true active
    # domain (confirmed directly, twice, against the real coarse-grid mask
    # block-upsampled with no reprojection involved: 0 of 433,712 active/
    # boundary fine cells were NaN in the real dep_subgrid.tif output; all
    # 8,400 NaN cells were in mask==0 cells outside tile_gdf's own polygon -
    # harmless padding, never read by process_tile_regular's volume-table
    # construction, which skips inactive cells entirely). A buffer-margin
    # workaround was tried and measured to have zero effect (byte-identical
    # NaN count with or without it) - not worth the added complexity/cost.
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
    # Reconstruct sf so its DataCatalog re-reads local_catalog_path with the new entries -
    # confirmed necessary (same as the old single-resolution code's own workaround): a
    # SfincsModel's own DataCatalog is parsed once at construction, not re-read from a
    # mid-session file rewrite.
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False, all_touched=True)

    # write_dep_tif=True writes subgrid/dep_subgrid.tif - the fine-resolution DEM
    # run_sfincs_tile.py's own postprocessing needs for hydromt_sfincs's own
    # downscale_floodmap() utility.
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

    # -- 4. boundary forcing (COAST-HG hydrographs, empirically MDT-corrected -
    # build_boundary_forcing.py's own prep output) --
    matched_points = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = retry_transient_io(pd.read_csv, sfincs_dir / "corrected_hydrographs.csv")

    # Truncate the full ~148.8h COAST-HG hydrograph down to a window around
    # its own storm peak, instead of simulating the whole thing - real,
    # confirmed speedup (2026-09): SFINCS's own wall-clock cost scales with
    # simulated duration at a roughly fixed timestep, so cutting duration
    # from 148.8h to 70h (40-110h) is a ~2.1x reduction on its own, on top
    # of (not instead of) the -50m bathymetry-floor speedup. Every COAST-HG
    # hydrograph in this pipeline shares the SAME synthetic time axis - not
    # a per-station/per-tile-specific timing - confirmed by checking all 4
    # real tiles built so far (2335, 1573, 1907, 929): every one peaks at
    # EXACTLY t=74.5h, and hour 40/hour 110 both sit close to each tile's
    # own tidal-only baseline (well before/after the storm builds up and
    # decays), so this window is a property of the dataset, not something
    # that needs per-tile tuning. zsini (below) now comes from hour 40's
    # own corrected value instead of hour 0's, matching whatever the new
    # truncated series' own first row is - no separate change needed there.
    if truncate_window_hr is not None:
        t_start, t_end = truncate_window_hr
        keep = (hydrographs["elapsed_hr"] >= t_start) & (hydrographs["elapsed_hr"] <= t_end)
        hydrographs = hydrographs.loc[keep].reset_index(drop=True)
        hydrographs["elapsed_hr"] = hydrographs["elapsed_hr"] - hydrographs["elapsed_hr"].iloc[0]

    elapsed_hr = hydrographs["elapsed_hr"].to_numpy()
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]

    tref = tref or datetime(2026, 1, 1)  # arbitrary but fixed reference - only elapsed time
    # within the hydrograph is physically meaningful (see coast_hg's own catalog caveats)
    times = pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in elapsed_hr])
    wl_df = hydrographs[station_cols].copy()
    wl_df.index = times
    wl_df.columns = range(len(station_cols))  # water_level.create expects positional columns matching locations' row order

    # matched_points.gpkg and corrected_hydrographs.csv's columns were written
    # in the same row order by build_boundary_forcing.py's own CLI - no
    # re-matching needed here. water_level.create()'s own docstring requires
    # an explicit "index" column matching timeseries' column names (its
    # internal get_geodataframe/set_index("index") call) - a plain
    # positional/row-order match is not enough.
    locations_gdf = matched_points.copy()
    locations_gdf["index"] = range(len(station_cols))

    sf.config.set("tref", tref)
    sf.config.set("tstart", tref)
    tstop = times[-1]
    sf.config.set("tstop", tstop)

    # buffer: DYNAMIC, not a fixed guess - real bug found and fixed here
    # (2026-09, live on the first 258-tile HPC test batch): a tile with only
    # 1-2 pre-selected boundary points to begin with (build_boundary_forcing.
    # py's own k-nearest filter can't invent stations that don't exist) can
    # have its single real match sit farther than any fixed buffer guess -
    # e.g. tile 1860's only station is 103.3 km from the tile, just past a
    # flat 100 km buffer. water_level.create() then masks EVERY location
    # out and hydromt raises `NoDataException: GeoDataFrame has no data
    # after masking` - a hard crash, not a partial drop, since zero
    # locations remain to build forcing from at all. Computing the real max
    # distance from any of these already-vetted (k-nearest-filtered)
    # stations to the model's own grid extent, instead of guessing a fixed
    # number, means the buffer can never accidentally exclude a point
    # build_boundary_forcing.py already decided belongs in this tile's own
    # forcing set.
    # "mask", not "dep": with subgrid, sf.grid.data has no "dep" variable at all any more
    # (elevation lives only in the subgrid table now) - "mask" exists from step 2 above
    # regardless, and only its x/y coords are needed here anyway.
    grid_coords = sf.grid.data["mask"]
    grid_x_min, grid_x_max = float(grid_coords["x"].min()), float(grid_coords["x"].max())
    grid_y_min, grid_y_max = float(grid_coords["y"].min()), float(grid_coords["y"].max())
    locations_utm = locations_gdf.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()
    dx = np.maximum(np.maximum(grid_x_min - station_x, station_x - grid_x_max), 0.0)
    dy = np.maximum(np.maximum(grid_y_min - station_y, station_y - grid_y_max), 0.0)
    dist_to_grid_bbox = float(np.sqrt(dx ** 2 + dy ** 2).max())
    # 2x + flat margin, not distance + a small flat margin: confirmed live
    # (tile 2084, 79 deg N) that hydromt's own internal masking distance is
    # NOT simply "distance to this grid's own bbox corner" - a +5 km margin
    # on top of the real 106.3 km bbox-corner distance still raised the
    # same NoDataException, and binary-searching the real threshold found
    # it sits somewhere between 110 km and 150 km, i.e. genuinely more like
    # 1.1-1.5x the bbox-corner estimate (plausibly UTM distortion this far
    # north, or masking against the actual waterlevel-boundary-cell
    # geometry rather than the raw grid bbox) - doubling errs safely past
    # that without risk of ever pulling in an unrelated station, since only
    # the already-vetted (k-nearest-filtered) locations exist to include.
    # +25 km flat margin, not +10 km: confirmed live (tile 808, a huge Arctic tile
    # at 69-70 deg N with its own matched stations already INSIDE the grid, i.e.
    # dist_to_grid_bbox == 0) that a +10 km margin alone isn't always enough even
    # when the naive computed distance is already zero - binary-searched the real
    # threshold for that tile between 10 km (fails) and 20 km (works), so +25 km
    # keeps real margin past the confirmed-working value instead of sitting right
    # at it.
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
    # zsini is a per-COMPUTATIONAL-CELL initial condition, unaffected by subgrid
    # (station_x/station_y already computed above for the buffer calc; grid_coords
    # is the "mask" DataArray from step 2, used purely as a coords/dims/transform
    # template here - same grid "dep" used to be, before subgrid removed it) --
    #
    # Real, confirmed bug (2026-09, tiles 1702/1798): the IDW above used to be
    # kept at EVERY cell in the grid, land included, with no check that a cell
    # is actually open water. For a large low-lying tile with real below-sea-
    # level terrain far inland, the interpolated ~0.6-2m coastal water level
    # read as several METRES of "depth" once compared to that cell's own real
    # (very negative) bed elevation - so the cell started the simulation
    # already flooded, before any storm physics ran at all. Confirmed live:
    # 85.7% of tile 1702's real land (8,506 km2, mean "depth" 9.6m) and 28.3%
    # of tile 1798's (4,209 km2) started spuriously wet this way, and 100% of
    # BOTH tiles' own final reported flood extent (zsmax) turned out to be
    # cells that were already wet at t=0 - zero cells were ever newly flooded
    # by the simulated storm. Fixed via `_compute_zsini_array` below.
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

    # -- 6. output config: dtmaxout must span the WHOLE simulation, so the
    # zsmax envelope is one true whole-run maximum, not reset partway
    # through (SFINCS default dtmaxout=86400s=1 day, shorter than most of
    # our ~6.2-day COAST-HG events) - only max-inundation/extent is wanted
    # (see plan doc), so history/velocity/wet-duration outputs stay off. --
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
    args = parser.parse_args()

    root = read_root(Path(args.config))
    truncate_window_hr = None if args.no_truncate else TRUNCATE_WINDOW_HR_DEFAULT
    build_sfincs_tile(
        args.tile_id, root, resolution_m=args.resolution_m, truncate_window_hr=truncate_window_hr,
        subgrid_nr_pixels=args.subgrid_nr_pixels, subgrid_nr_levels=args.subgrid_nr_levels, subgrid_nrmax=args.subgrid_nrmax,
        base_dir_name=args.base_dir_name,
    )


if __name__ == "__main__":
    main()
