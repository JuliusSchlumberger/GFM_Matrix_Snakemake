"""Build a runnable subgrid SFINCS model for one GFM tile (no weirs, restart or discharge).

Inputs: build_elevation.py, build_roughness.py, build_station_boundary_lines.py and
build_line_forcing.py outputs. Runs in hydromt-sfincs-dev:
    C:\\Users\\schlumbe\\AppData\\Local\\miniforge3\\envs\\hydromt-sfincs-dev\\python.exe build_sfincs_tile.py --tile-id 1907
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
import xarray as xr
import yaml
from hydromt_sfincs import SfincsModel
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402



def idw_interpolate_to_grid(
    station_x: np.ndarray, station_y: np.ndarray, station_values: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray, k: int | None = None,
) -> np.ndarray:
    """Interpolate station values onto a grid by inverse distance squared.

    Plain Euclidean distance: stations and grid must share one metric CRS (e.g. UTM).
    `k` nearest stations per grid point; None uses every station.
    """
    k = k if k is not None else len(station_values)
    k = min(k, len(station_values))

    gx = np.asarray(grid_x).ravel()
    gy = np.asarray(grid_y).ravel()
    dist2 = (gx[:, None] - station_x[None, :]) ** 2 + (gy[:, None] - station_y[None, :]) ** 2
    dist2 = np.where(dist2 == 0.0, np.finfo(np.float64).tiny, dist2)

    if k < len(station_values):
        idx = np.argpartition(dist2, k, axis=1)[:, :k]
        dist2 = np.take_along_axis(dist2, idx, axis=1)
        vals = station_values[idx]
    else:
        vals = np.broadcast_to(station_values, (gx.size, len(station_values)))

    weights = 1.0 / dist2
    result = (vals * weights).sum(axis=1) / weights.sum(axis=1)
    return result.reshape(np.asarray(grid_x).shape)


def _validate_subgrid_params(resolution_m: float, subgrid_nr_pixels: int) -> None:
    # hydromt_sfincs subgrid.create() requires nr_subgrid_pixels % 2 == 0 (components/grid/subgrid.py)
    if subgrid_nr_pixels <= 0 or subgrid_nr_pixels % 2 != 0:
        raise ValueError(
            f"subgrid_nr_pixels must be a positive multiple of 2 (hydromt_sfincs's own "
            f"requirement), got {subgrid_nr_pixels}"
        )
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")


def _reproject_nearest_to_grid(src_path: Path, dst_transform, dst_crs, dst_shape: tuple[int, int]) -> np.ndarray:
    # Nearest, because hydromt_sfincs ignores `reproj_method` and always resamples
    # bilinear. The lon/lat source and UTM destination rectangles are rotated against
    # each other, so destination corners can fall outside the source: those NaNs are
    # nearest-valid filled, else they would corrupt a boundary cell's subgrid table.
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
    dst_transform, dst_crs, dst_shape: tuple[int, int],
) -> np.ndarray:
    # Non-ocean cells get -9999.0 (SFINCS: dry at bed level), else inland land would start
    # flooded. Cell centres come from the rotation-aware `dst_transform`, not the grid's
    # x/y coordinates: on a rotated grid those are pixel indices.
    rows, cols = np.indices(dst_shape)
    xx, yy = dst_transform * (cols + 0.5, rows + 0.5)
    zsini_arr = idw_interpolate_to_grid(station_x, station_y, station_values, xx, yy).astype(np.float32)

    native_mask_on_grid = _reproject_nearest_to_grid(native_mask_path, dst_transform, dst_crs, dst_shape)
    ocean = native_mask_on_grid == 1  # native mask.tif: 0=land, 1=ocean, 2=lake, 3=river
    return np.where(ocean, zsini_arr, np.float32(-9999.0))


INTERIOR_CORE_FACTOR = 0.75  # interior forcing point: inactive core radius, x resolution
INTERIOR_RING_FACTOR = 2.0   # ... and boundary-cell ring radius, x resolution
TILE_EDGE_BAND_M = 2000.0    # half-width of the boundary band along tile_edge stretches


def _tile_edge_band(segments: gpd.GeoDataFrame, tile_geom, crs) -> list:
    # Both the line builder's tile edge and hydromt's grid border project the tile by its
    # corners: a UTM chord sagging up to ~1.5 km off a constant-latitude edge, beyond the
    # 1.5-cell line band. Bands go round the stretch as built and moved onto the real
    # lon/lat edge; flat caps stop them running on along the edge.
    edge = gpd.GeoSeries([tile_geom.exterior.segmentize(0.002)], crs="EPSG:4326").to_crs(crs).iloc[0]
    bands = []
    for g in segments.to_crs(crs).geometry:
        pts = [edge.interpolate(edge.project(shapely.Point(c))) for c in g.segmentize(100.0).coords]
        for line in (g, shapely.LineString(pts) if len(pts) >= 2 else None):
            if line is not None and line.length > 0:
                bands.append(line.buffer(TILE_EDGE_BAND_M, cap_style="flat"))
    return bands


def _drop_land_boundary(sf: SfincsModel, new_band: gpd.GeoDataFrame, native_mask_path: Path) -> int:
    # A boundary cell on land would impose the forcing there and flood it artificially.
    da = sf.grid.data["mask"]
    in_band = da.raster.geometry_mask(new_band, all_touched=True).values
    m = da.values.copy()
    cand = (m == 2) & in_band
    if not cand.any():
        return 0
    r, c = np.nonzero(cand)
    # rasterio.transform.xy / rasterio.warp.transform crash this env (DLL delay-load 0xc06d007f)
    from pyproj import Transformer
    x, y = da.raster.transform * (c + 0.5, r + 0.5)
    lon, lat = Transformer.from_crs(sf.crs, "EPSG:4326", always_xy=True).transform(x, y)
    with rasterio.open(native_mask_path) as src:
        nat = src.read(1)
        col, row = ~src.transform * (np.asarray(lon), np.asarray(lat))
    rr = np.clip(np.floor(row).astype(int), 0, nat.shape[0] - 1)
    cc = np.clip(np.floor(col).astype(int), 0, nat.shape[1] - 1)
    land = nat[rr, cc] != OCEAN_CODE_NATIVE
    m[r[land], c[land]] = 1
    sf.grid.set(da.copy(data=m), name="mask")
    return int(land.sum())


OCEAN_CODE_NATIVE = 1


def _create_mask(sf: SfincsModel, tile_gdf: gpd.GeoDataFrame, boundary_lines_gpkg: Path, resolution_m: float,
                 interior: gpd.GeoDataFrame | None = None, native_mask_path: Path | None = None) -> None:
    # Interior (connected_forcing.py) cores are excluded explicitly: create_active's
    # all_touched include would fill them back in. make_valid everywhere: reprojected
    # inactive faces can self-touch and GEOS fails with "side location conflict".
    tile_geom = shapely.make_valid(tile_gdf.to_crs("EPSG:4326").geometry.iloc[0])
    active = tile_geom
    layers = pyogrio.list_layers(boundary_lines_gpkg)[:, 0]
    if "inactive" in layers:
        inactive = retry_transient_io(gpd.read_file, boundary_lines_gpkg, layer="inactive").to_crs("EPSG:4326")
        if len(inactive):
            inactive_geom = shapely.make_valid(shapely.union_all(shapely.make_valid(inactive.geometry.values)))
            active = shapely.make_valid(tile_geom.difference(inactive_geom))
    band_parts = [g.buffer(1.5 * resolution_m) for g in
                  retry_transient_io(gpd.read_file, boundary_lines_gpkg, layer="boundary_line").to_crs(sf.crs).geometry]
    new_parts = []
    if "segments" in layers:
        seg = retry_transient_io(gpd.read_file, boundary_lines_gpkg, layer="segments")
        new_parts += _tile_edge_band(seg[seg["kind"] == "tile_edge"], tile_geom, sf.crs)
    exclude = None
    if interior is not None and len(interior):
        pts = interior.to_crs(sf.crs).geometry
        exclude = gpd.GeoDataFrame(geometry=list(pts.buffer(INTERIOR_CORE_FACTOR * resolution_m)), crs=sf.crs)
        new_parts += list(pts.buffer(INTERIOR_RING_FACTOR * resolution_m))
    sf.mask.create_active(include_polygon=gpd.GeoDataFrame(geometry=[active], crs="EPSG:4326"),
                          exclude_polygon=exclude, reset_mask=True)
    band = gpd.GeoDataFrame(geometry=[shapely.union_all(band_parts + new_parts)], crs=sf.crs)
    sf.mask.create_boundary(btype="waterlevel", include_polygon=band, reset_bounds=False, all_touched=True)
    if new_parts and native_mask_path is not None:
        new_band = gpd.GeoDataFrame(geometry=[shapely.union_all(new_parts).difference(shapely.union_all(band_parts))],
                                    crs=sf.crs)
        n = _drop_land_boundary(sf, new_band, native_mask_path)
        if n:
            print(f"      {n} new boundary cell(s) on land (native mask) set back to active")


# Hours on COAST-HG's shared synthetic time axis (storm peak at t=74.5 h).
TRUNCATE_WINDOW_HR_DEFAULT = (40.0, 110.0)

# 120 m grid / 4 = 30 m subgrid = DeltaDTM's native resolution; finer would let
# hydromt_sfincs's forced bilinear resampling fabricate sub-pixel depressions.
MAIN_RES_M_DEFAULT = 120.0
SUBGRID_NR_PIXELS_DEFAULT = 4
SUBGRID_NR_LEVELS_DEFAULT = 20  # hypsometric bins; memory scales linearly with this
SUBGRID_NRMAX_DEFAULT = 2000  # hydromt_sfincs default block size



def build_sfincs_tile(
    tile_id: str, root: Path, resolution_m: float = MAIN_RES_M_DEFAULT,
    dtmapout_s: float = 1800.0, tref: datetime | None = None,
    truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT,
    subgrid_nr_pixels: int = SUBGRID_NR_PIXELS_DEFAULT,
    subgrid_nr_levels: int = SUBGRID_NR_LEVELS_DEFAULT,
    subgrid_nrmax: int = SUBGRID_NRMAX_DEFAULT,
    base_dir_name: str = "calibration_roughness_SFINCS",
    rotated: bool = True,
    boundary_lines_gpkg: Path | None = None,  # required
) -> Path:
    """Build and write the SFINCS model to `root`/`base_dir_name`/`tile_id`/sfincs_model.

    `truncate_window_hr` is (start, end) h on the COAST-HG time axis; None keeps the
    full ~148.8 h. `tref` defaults to 2026-01-01 (only elapsed time matters).
    Raises RuntimeError if the boundary line yields no waterlevel-boundary cells.
    """
    if boundary_lines_gpkg is None:
        raise ValueError("build_sfincs_tile needs boundary_lines_gpkg (build_station_boundary_lines.py output)")
    _validate_subgrid_params(resolution_m, subgrid_nr_pixels)
    tile_dir = root / base_dir_name / tile_id / "inputs"
    sfincs_dir = root / base_dir_name / tile_id / "sfincs_model"
    sfincs_dir.mkdir(parents=True, exist_ok=True)

    tile_gdf = retry_transient_io(gpd.read_file, tile_dir / "tile_geometry.gpkg")
    matched_all = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    interior = matched_all[matched_all["kind"] == "interior"] if "kind" in matched_all.columns else None

    # hydromt create() methods need catalog-keyed sources, not raw DataArrays
    local_catalog_path = sfincs_dir / "data_catalog_local.yml"
    local_catalog = {
        "meta": {"root": str(sfincs_dir)},
        "local_elevation": {"data_type": "RasterDataset", "uri": "elevation_combined.tif", "driver": "rasterio"},
        "local_roughness": {"data_type": "RasterDataset", "uri": "manning_n.tif", "driver": "rasterio"},
    }
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)

    # rotated: create_from_region fits the grid to the region geometry, so fewer cells
    # are wasted on corners than with an axis-aligned UTM grid
    sf =SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm", rotated=rotated)
    print(f"[1/8] grid created: {dict(sf.grid.data.sizes)} cells, crs={sf.crs}, rotated={rotated}")

    # Mask before subgrid: subgrid.create() reads grid.mask. all_touched=True in
    # _create_mask: the default centre-point test leaves a broken boundary on a
    # line running diagonally across the rotated grid.
    _create_mask(sf, tile_gdf, boundary_lines_gpkg, resolution_m, interior, tile_dir / "mask.tif")
    n_active = int((sf.grid.data["mask"] > 0).sum())
    n_bnd = int((sf.grid.data["mask"] == 2).sum())
    print(f"[2/8] mask: {n_active} active cell(s), {n_bnd} waterlevel-boundary cell(s)")
    if n_bnd == 0:
        raise RuntimeError(
            f"tile {tile_id}: 0 waterlevel-boundary cells after create_boundary - "
            "the ocean polygon didn't reach any active-domain edge cell on this grid. "
            "No weir/discharge fallback exists in this pipeline - this tile can't be forced."
        )

    # Pre-reproject elevation and Manning's n (nearest) onto the exact subgrid so
    # hydromt's forced bilinear pass is a no-op: bilinear below DeltaDTM's native
    # resolution fabricates depressions; nearest keeps every pixel a real sample.
    main_transform = sf.grid.data.raster.transform
    main_crs = sf.grid.data.raster.crs
    main_height, main_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    fine_transform = main_transform * main_transform.scale(1.0 / subgrid_nr_pixels)
    fine_height, fine_width = main_height * subgrid_nr_pixels, main_width * subgrid_nr_pixels

    # subgrid.create()'s re-read can reintroduce NaN, but only in padding outside the active domain
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
    # DataCatalog is parsed once at construction: rebuild sf to pick up the new entries
    sf =SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm", rotated=rotated)
    _create_mask(sf, tile_gdf, boundary_lines_gpkg, resolution_m, interior, tile_dir / "mask.tif")

    # dep_subgrid.tif is needed by run_sfincs_tile.py's downscale_floodmap()
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

    # COAST-HG hydrographs at the line's station nodes, MDT-corrected by build_line_forcing.py
    matched_points = retry_transient_io(gpd.read_file, sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = retry_transient_io(pd.read_csv, sfincs_dir / "corrected_hydrographs.csv")

    # SFINCS cost scales with duration; hours 40/110 are near the tidal-only baseline.
    # zsini uses the truncated series' first row.
    if truncate_window_hr is not None:
        t_start, t_end = truncate_window_hr
        keep = (hydrographs["elapsed_hr"] >= t_start) & (hydrographs["elapsed_hr"] <= t_end)
        hydrographs = hydrographs.loc[keep].reset_index(drop=True)
        hydrographs["elapsed_hr"] = hydrographs["elapsed_hr"] - hydrographs["elapsed_hr"].iloc[0]

    elapsed_hr = hydrographs["elapsed_hr"].to_numpy()
    station_cols = [c for c in hydrographs.columns if c != "elapsed_hr"]

    tref = tref or datetime(2026, 1, 1)
    times = pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in elapsed_hr])
    wl_df = hydrographs[station_cols].copy()
    wl_df.index = times
    wl_df.columns = range(len(station_cols))

    # matched points and hydrograph columns share order (both from build_line_forcing.py);
    # water_level.create() matches the "index" column to the timeseries columns
    locations_gdf = matched_points.copy()
    locations_gdf["index"] = range(len(station_cols))

    sf.config.set("tref", tref)
    sf.config.set("tstart", tref)
    tstop = times[-1]
    sf.config.set("tstop", tstop)

    # water_level.create() raises NoDataException if no location is within `buffer`.
    # "mask", not "dep": with subgrid there is no "dep". Use .raster.bounds: on a
    # rotated grid the x/y coords are pixel indices (real positions are xc/yc).
    grid_coords = sf.grid.data["mask"]
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = grid_coords.raster.bounds
    locations_utm = locations_gdf.to_crs(sf.crs)
    station_x = locations_utm.geometry.x.to_numpy()
    station_y = locations_utm.geometry.y.to_numpy()
    dx = np.maximum(np.maximum(grid_x_min - station_x, station_x - grid_x_max), 0.0)
    dy = np.maximum(np.maximum(grid_y_min - station_y, station_y - grid_y_max), 0.0)
    dist_to_grid_bbox = float(np.sqrt(dx ** 2 + dy ** 2).max())
    # generous: hydromt's masking distance is not the bbox distance, and only
    # pre-filtered stations exist, so nothing unrelated can be pulled in
    buffer_m = dist_to_grid_bbox * 2.0 + 25_000.0

    # Locations in UTM: water_level.create() reprojects boundary-cell polygons to the
    # locations' CRS, and in EPSG:4326 near +-180 deg that gives a GEOS "side location conflict".
    sf.water_level.create(timeseries=wl_df, locations=locations_utm, buffer=buffer_m)
    print(f"[4/8] water-level forcing: {len(station_cols)} station(s), {len(times)} timestep(s), "
          f"{tref} -> {tstop} ({elapsed_hr[-1]:.1f} h), buffer={buffer_m / 1000:.1f} km")

    # zsini on the coarse grid (unaffected by subgrid)
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)
    zsini_arr = _compute_zsini_array(
        tile_dir / "mask.tif", station_x, station_y, first_vals,
        main_transform, main_crs, (main_height, main_width),
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

    # dtmaxout spans the whole run so zsmax is one whole-run maximum (default 86400 s
    # is shorter than the events). +1 h margin so it never resets.
    total_span_s = elapsed_hr[-1] * 3600.0
    sf.config.set("dtmaxout", total_span_s + 3600.0)
    sf.config.set("dtmapout", dtmapout_s)
    sf.config.set("baro", 0)
    print(f"[6/8] output config: dtmaxout={total_span_s + 3600.0:.0f}s (whole-run max), dtmapout={dtmapout_s:.0f}s")

    sf.write()
    print(f"[7/8] wrote SFINCS model to {sfincs_dir}")
    return sfincs_dir


def main() -> None:
    """Build the SFINCS model for one tile from the command line."""
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
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS", help="output root directory name under paths.root (default: calibration_roughness_SFINCS)")
    parser.add_argument(
        "--boundary-lines-gpkg", required=True,
        help="build_station_boundary_lines.py output for this tile: the ocean side of its "
             "boundary line is masked inactive and the model is forced along the line",
    )
    parser.add_argument(
        "--no-rotated", dest="rotated", action="store_false",
        help="use an axis-aligned UTM grid instead of the default rotated grid (2026-10 default flip - "
             "was unrotated until now, see build_sfincs_tile's own comment)",
    )
    args = parser.parse_args()

    root = read_root(Path(args.config))
    truncate_window_hr = None if args.no_truncate else TRUNCATE_WINDOW_HR_DEFAULT
    build_sfincs_tile(
        args.tile_id, root, resolution_m=args.resolution_m, truncate_window_hr=truncate_window_hr,
        subgrid_nr_pixels=args.subgrid_nr_pixels, subgrid_nr_levels=args.subgrid_nr_levels, subgrid_nrmax=args.subgrid_nrmax,
        base_dir_name=args.base_dir_name, rotated=args.rotated,
        boundary_lines_gpkg=Path(args.boundary_lines_gpkg),
    )


if __name__ == "__main__":
    main()
