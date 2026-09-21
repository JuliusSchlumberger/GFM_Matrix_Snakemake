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

No weirs, no restart, no subgrid, no discharge - see sfincs_tiles' own plan
doc for the full design/reasoning.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_boundary_forcing import idw_interpolate_to_grid  # noqa: E402
from gfm_config import read_root  # noqa: E402


def _ocean_polygon_wgs84(mask_path: Path, ocean_code: int = 1) -> gpd.GeoDataFrame:
    """Vectorize mask.tif's ocean cells into a polygon GeoDataFrame (EPSG:4326)."""
    with rasterio.open(mask_path) as src:
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


def build_sfincs_tile(
    tile_id: str, root: Path, resolution_m: float = 30.0,
    dtmapout_s: float = 1800.0, tref: datetime | None = None,
    truncate_window_hr: tuple[float, float] | None = TRUNCATE_WINDOW_HR_DEFAULT,
) -> Path:
    tile_dir = root / "model_outputs" / tile_id / "inputs"
    sfincs_dir = root / "validation_sfincs" / tile_id / "sfincs_model"
    sfincs_dir.mkdir(parents=True, exist_ok=True)

    tile_gdf = gpd.read_file(tile_dir / "tile_geometry.gpkg")

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

    # -- 1. grid --
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")
    print(f"[1/8] grid created: {dict(sf.grid.data.sizes)} cells, crs={sf.crs}")

    # -- 2. elevation (combined DeltaDTM+MDT-corrected-GEBCO, built by build_elevation.py) --
    # Pre-reproject onto the SFINCS UTM grid OURSELVES (nearest-neighbour, via rasterio
    # directly) before handing it to hydromt_sfincs, rather than passing
    # reproj_method="nearest" to elevation.create() and trusting hydromt_sfincs to honour
    # it. Real bug found and fixed here (2026-09, tile 2335 A/B bathymetry-floor test):
    # confirmed passing reproj_method="nearest" has NO EFFECT - hydromt_sfincs's own
    # merge_multi_dataarrays (hydromt_sfincs/workflows/merge.py, ~line 85-100) has
    # `if method is None and da_like is not None: ...resolution-based choice...
    # else: method = "bilinear"` - the else branch (taken whenever a reproj_method IS
    # explicitly given) unconditionally OVERWRITES it with "bilinear" instead of
    # respecting it, a real bug in the vendored library itself, not something fixable
    # from our own elevation_list dict. elevation_combined.tif has a DELIBERATE hard
    # step at the coastline (DeltaDTM land directly abutting a GEBCO+MIN_BATHYMETRY_M-
    # floored ocean value, not a physically continuous surface), and mask.tif
    # (land_mask_path in run_sfincs_tile.py's own compute_max_inundation) is reprojected
    # with nearest-neighbour - so bilinear smoothing across that same cliff produces UTM
    # cells the (nearest) mask still calls "land" but with an interpolated dep tens of
    # metres deep - confirmed live: this alone explained an apparent "-10m floor causes
    # more flooding" result that had nothing to do with wave speed/advection. Workaround:
    # reproject to the EXACT destination grid ourselves first, so hydromt_sfincs's own
    # forced-bilinear pass becomes a no-op (source and destination pixels already
    # coincide exactly - bilinear of an aligned grid returns the same value, no blending
    # possible). No floor-dependent threshold needed downstream once this is fixed at
    # the source.
    grid_transform = sf.grid.data.raster.transform
    grid_crs = sf.grid.data.raster.crs
    grid_height, grid_width = sf.grid.data.sizes["y"], sf.grid.data.sizes["x"]
    elevation_utm_path = sfincs_dir / "elevation_combined_utm.tif"
    with rasterio.open(sfincs_dir / "elevation_combined.tif") as src:
        elevation_utm = np.empty((grid_height, grid_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=elevation_utm,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=grid_transform, dst_crs=grid_crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": grid_height, "width": grid_width,
        "transform": grid_transform, "crs": grid_crs, "nodata": np.nan, "compress": "deflate",
    }
    with rasterio.open(elevation_utm_path, "w", **profile) as dst:
        dst.write(elevation_utm, 1)

    local_catalog["local_elevation_utm"] = {"data_type": "RasterDataset", "uri": "elevation_combined_utm.tif", "driver": "rasterio"}
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)
    sf.data_catalog = None  # force re-read of local_catalog_path with the new entry just added
    sf = SfincsModel(data_libs=[str(local_catalog_path)], root=str(sfincs_dir), mode="w+")
    sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm")

    sf.elevation.create(elevation_list=[{"elevation": "local_elevation_utm"}])
    print(f"[2/8] elevation set - range {float(sf.grid.data['dep'].min()):.2f} to {float(sf.grid.data['dep'].max()):.2f} m")

    # -- 3. mask: active cells (whole tile) + waterlevel boundary (ocean edge only) --
    sf.mask.create_active(include_polygon=tile_gdf, reset_mask=True)
    ocean_poly = _ocean_polygon_wgs84(tile_dir / "mask.tif")
    sf.mask.create_boundary(btype="waterlevel", include_polygon=ocean_poly, reset_bounds=False)
    n_active = int((sf.grid.data["mask"] > 0).sum())
    n_bnd = int((sf.grid.data["mask"] == 2).sum())
    print(f"[3/8] mask: {n_active} active cell(s), {n_bnd} waterlevel-boundary cell(s)")
    if n_bnd == 0:
        raise RuntimeError(
            f"tile {tile_id}: 0 waterlevel-boundary cells after create_boundary - "
            "the ocean polygon didn't reach any active-domain edge cell on this grid. "
            "No weir/subgrid/discharge fallback exists in this pipeline - this tile can't be forced."
        )

    # -- 4. roughness (friction.tif decoded to real Manning's n by build_roughness.py) --
    sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])
    print(f"[4/8] roughness set - range {float(sf.grid.data['manning'].min()):.4f} to {float(sf.grid.data['manning'].max()):.4f}")

    # -- 5. boundary forcing (COAST-HG hydrographs, empirically MDT-corrected -
    # build_boundary_forcing.py's own prep output) --
    matched_points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg")
    hydrographs = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")

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
    dep = sf.grid.data["dep"]
    grid_x_min, grid_x_max = float(dep["x"].min()), float(dep["x"].max())
    grid_y_min, grid_y_max = float(dep["y"].min()), float(dep["y"].max())
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
        # Known, separate limitation (not something this buffer formula can
        # fix): hydromt's own internal masking does a shapely union_all() in
        # raw EPSG:4326 lon/lat - for a tile near the antimeridian (confirmed
        # live: tiles 2029/2077, both in the Chukchi Sea around -179 to -178
        # deg lon), the buffered search geometry can straddle +-180 deg and
        # produce a self-intersecting polygon there, which GEOS rejects as
        # an invalid topology regardless of how tight or generous buffer_m
        # is (confirmed: a SMALLER buffer briefly "worked" for tile 2077
        # only by accident, not fixing anything - a marginally bigger one
        # immediately hit the same failure). Not worth chasing inside this
        # pipeline (would mean patching hydromt_sfincs's own dateline
        # handling) - surfaced as a clear, actionable error instead of a
        # raw GEOS traceback, so a batch run's failure log says WHY, and
        # this tile can be dropped like any other "can't be forced" case.
        if "TopologyException" in str(e) or "side location conflict" in str(e):
            raise RuntimeError(
                f"tile {tile_id}: antimeridian-crossing geometry error in hydromt_sfincs's own "
                f"water_level.create() masking (tile is near +-180 deg longitude) - not fixable via "
                f"buffer tuning, drop this tile from the batch. Original error: {e}"
            ) from e
        raise
    print(f"[5/8] water-level forcing: {len(station_cols)} station(s), {len(times)} timestep(s), "
          f"{tref} -> {tstop} ({elapsed_hr[-1]:.1f} h), buffer={buffer_m / 1000:.1f} km")

    # -- 6. initial conditions (zsini): IDW of the matched stations' own
    # first-timestep corrected value, onto this model's own UTM grid
    # (station_x/station_y already computed above for the buffer calc) --
    first_vals = hydrographs[station_cols].iloc[0].to_numpy(dtype=np.float64)
    yy, xx = np.meshgrid(dep["y"].values, dep["x"].values, indexing="ij")
    zsini_arr = idw_interpolate_to_grid(station_x, station_y, first_vals, xx, yy)
    zsini_da = xr.DataArray(zsini_arr.astype(np.float32), dims=dep.dims, coords=dep.coords)
    zsini_da = zsini_da.rio.write_crs(sf.crs)
    zsini_da = zsini_da.rio.write_transform(dep.rio.transform())

    sf.initial_conditions.create(zsini=zsini_da, fill_value=-9999.0, reproj_method="nearest")
    print(f"[6/8] zsini set - range {float(np.nanmin(zsini_arr)):.4f} to {float(np.nanmax(zsini_arr)):.4f} m "
          f"(interpolated from {len(station_x)} station(s))")

    # -- 7. output config: dtmaxout must span the WHOLE simulation, so the
    # zsmax envelope is one true whole-run maximum, not reset partway
    # through (SFINCS default dtmaxout=86400s=1 day, shorter than most of
    # our ~6.2-day COAST-HG events) - only max-inundation/extent is wanted
    # (see plan doc), so history/velocity/wet-duration outputs stay off. --
    total_span_s = elapsed_hr[-1] * 3600.0
    sf.config.set("dtmaxout", total_span_s + 3600.0)  # +1h margin, never triggers a reset
    sf.config.set("dtmapout", dtmapout_s)
    sf.config.set("baro", 0)
    print(f"[7/8] output config: dtmaxout={total_span_s + 3600.0:.0f}s (whole-run max), dtmapout={dtmapout_s:.0f}s")

    # -- 8. write --
    sf.write()
    print(f"[8/8] wrote SFINCS model to {sfincs_dir}")
    return sfincs_dir


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--resolution-m", type=float, default=30.0)
    parser.add_argument(
        "--no-truncate", action="store_true",
        help="use the full ~148.8h COAST-HG hydrograph instead of the default "
             f"{TRUNCATE_WINDOW_HR_DEFAULT} window - for A/B comparison only.",
    )
    args = parser.parse_args()

    root = read_root(Path(args.config))
    truncate_window_hr = None if args.no_truncate else TRUNCATE_WINDOW_HR_DEFAULT
    build_sfincs_tile(args.tile_id, root, resolution_m=args.resolution_m, truncate_window_hr=truncate_window_hr)


if __name__ == "__main__":
    main()
