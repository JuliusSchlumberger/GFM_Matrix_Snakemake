"""
13_build_sfincs_skeleton.py -- Build the scenario-INDEPENDENT part of a
basin's SFINCS model: grid, elevation, mask (active + waterlevel + outflow
boundary), coastal protection weir, roughness, subgrid table, observation
points. Runs ONCE per basin (no {scenario} wildcard) -- see
13_build_sfincs_skeleton.smk's own module comment for why none of these
sections depend on a scenario's own forcing_mode/design_rp_river_yr/
design_rp_surge_yr.

zsini.tif (sea cells at baseline_m, land dry) is also built here -- a pure
function of zsini_sea_cells_on_grid.tif + baseline_m (both basin-level,
scenario-independent) -- even though whether it's actually USED as the
model's initial condition depends on forcing_mode (see build_sfincs's own
initial-conditions section, 13_build_sfincs.py).

zsini_sea_cells_on_grid.tif is the ONLY sea/land classification zsini is
ever built from, and it is already rasterized directly onto THIS model's
own grid (zero further reprojection here -- see rule modelled_depth_
estimation/empirical_depth_estimation, whichever ran, for where that
raster is produced). There used to be a second, native-resolution
construction path here that read sea_mask_corrected.tif and let HydroMT's
own reproject_like resample it a SECOND time onto this grid -- removed
2026-08-07: two independent nearest-neighbor passes don't invert each
other cleanly even on a confirmed pixel-identical grid (confirmed:
~53% of the corrected coastal-fringe cells were still wet after that
second pass), and maintaining two different construction methods for the
same quantity made the on-disk zsini.tif inconsistent with what actually
ended up in sfincs.ini. Every raster that feeds the model's own grid is
now resampled onto that grid exactly ONCE, at the point it's first
produced as a preprocessing input -- never re-resampled again downstream.

Outputs
-------
sfincs.inp    Grid header (mmax/nmax/dx/dy/x0/y0/epsg) + geometry file
              references (depfile/mskfile/manningfile/sbgfile/weirfile) --
              NOT a directly runnable model on its own (no forcing, no
              tref/tstart/tstop). build_sfincs and run_spinup each load
              this in read mode, redirect writes to their own root
              (sf.root.set(...)), and hand-craft their OWN sfincs.inp that
              forwards these same geometry files via a relative path
              (see either script's own module docstring) plus their own
              forcing/timing on top.
sfincs.weir   Coastal/riverbank protection weir; empty placeholder when
              disabled or not applicable.
zsini.tif     Spatially-varying initial water level (sea = baseline_m,
              land = nodata).

See 13_build_sfincs.py's own module docstring for the forcing_mode
semantics (compound/coastal_only/river_only) -- entirely a per-scenario
concern, not relevant here.
"""

import json
import logging
from pathlib import Path

import rasterio
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import label as _ndimage_label
from shapely.geometry import Polygon
from hydromt_sfincs import SfincsModel

from src.plots import add_land_background_to_geoaxes, label_axes_lonlat
from src.raster import restrict_waterlevel_boundary_to_sea
from src.surge import read_baseline_m

plt.ioff()

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=snakemake.log[0],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.INFO)
    _l.handlers = []
    _l.addHandler(logging.FileHandler(snakemake.log[0]))

# ── paths & params ────────────────────────────────────────────────────────────
domain_path           = Path(snakemake.input.domain_gpkg)
# Native-resolution CONDITIONED elevation (post-monotonicity) -- see this
# rule's own input comment for why this is named elevation_conditioned,
# not elevation_merged (the raw file, rule 05a's own output, unused here).
elevation_conditioned_path = Path(snakemake.input.elevation_conditioned)
# Native resolution -- subgrid table only (see local_roughness_native's own
# comment below). The main regular-grid "manning" field and the weir
# diagnostics section use roughness_on_grid_path/landuse_on_grid_path
# instead (rule grid_align_landuse, 09b) -- already resampled onto this
# model's own grid once, upstream, zero further reprojection here.
roughness_path        = Path(snakemake.input.roughness)
roughness_on_grid_path = Path(snakemake.input.roughness_on_grid)
landuse_on_grid_path  = Path(snakemake.input.landuse_on_grid)
# Grid-aligned land mask (rule grid_align_landuse) -- plot background only.
land_mask_path        = Path(snakemake.input.land_mask_on_grid)
river_network_path    = Path(snakemake.input.river_network)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)
# The ONLY sea/land classification zsini is built from -- already
# rasterized directly onto this model's own grid by whichever of rule
# modelled_depth_estimation/empirical_depth_estimation ran (see this
# script's own module docstring).
zsini_sea_cells_path = Path(snakemake.input.zsini_sea_cells)
surge_forcing_path    = Path(snakemake.input.surge_forcing)
river_forcing_path    = Path(snakemake.input.river_forcing)
river_burned_dem_path = Path(snakemake.input.river_burned_dem)
river_burned_dem_sfincs_grid_path = Path(snakemake.input.river_burned_dem_sfincs_grid)
elevation_conditioned_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
coastal_protection_weir_path = (
    Path(snakemake.input.coastal_protection_weir) if snakemake.input.coastal_protection_weir else None
)

inputs_dir        = Path(snakemake.params.inputs_dir)
skeleton_root     = Path(snakemake.params.skeleton_root)
resolution        = snakemake.params.resolution
include_subgrid   = snakemake.params.include_subgrid
nr_subgrid_pixels = snakemake.params.nr_subgrid_pixels
nr_levels         = snakemake.params.nr_levels
nrmax             = snakemake.params.nrmax
depth_method      = snakemake.params.depth_method
active_mask_enabled = snakemake.params.active_mask_enabled
active_mask_elevation_buffer_m = float(snakemake.params.active_mask_elevation_buffer_m)
outflow_buffer_m   = snakemake.params.outflow_buffer_m
n_top_crossings    = snakemake.params.n_top_crossings
n_per_crossing     = snakemake.params.n_per_crossing
max_downstream_hops = snakemake.params.max_downstream_hops

skeleton_root.mkdir(parents=True, exist_ok=True)

# ── baseline water level from surge forcing ───────────────────────────────────
# baseline_m/coastal_protection_crest_m are basin-level fixed fields (mean
# vertical correction, FLOPROS coastal standard) -- NOT derived from any
# scenario's own design RP. See 07_get_boundary_forcings.py. baseline_m is
# rounded UP to the next 0.1 m (read_baseline_m), like every coastal water
# level the model sees.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _ds:
    baseline_m = read_baseline_m(_ds)
    coastal_protection_crest_m = (
        float(_ds["coastal_protection_crest_m"].values)
        if "coastal_protection_crest_m" in _ds else 0.0
    )
log.info(f"Surge boundary baseline read from surge_forcing.nc: {baseline_m:+.4f} m (rounded up to 0.1 m)")

# zsini.tif (sea cells = baseline_m, land = nodata) is built once, directly
# on this model's own grid, from zsini_sea_cells_on_grid.tif -- see the
# "Initial conditions" section below (needs sf.grid.data["dep"], not yet
# created at this point in the script).
_zsini_out = Path(snakemake.output.zsini)

# ── delta-outline outflow points ──────────────────────────────────────────────
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

# ── data catalog ──────────────────────────────────────────────────────────────
local_catalog_path = skeleton_root / "data_catalog_local.yml"
local_catalog = {
    "meta": {"root": str(inputs_dir)},
    "local_elevation_conditioned": {
        "data_type": "RasterDataset",
        "uri": str(elevation_conditioned_path),
        "driver": "rasterio",
    },
    "local_river_burned": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_dem_path),
        "driver": "rasterio",
    },
    "local_river_burned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_dem_sfincs_grid_path),
        "driver": "rasterio",
    },
    "local_elevation_conditioned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(elevation_conditioned_sfincs_grid_path),
        "driver": "rasterio",
    },
    # Coarse -- already resampled onto this model's own grid (rule
    # grid_align_landuse, 09b), zero further reprojection. Main regular-grid
    # "manning" field only, never subgrid (see local_roughness_native below).
    "local_roughness": {
        "data_type": "RasterDataset",
        "uri": str(roughness_on_grid_path),
        "driver": "rasterio",
    },
    # Native resolution -- subgrid table only, which genuinely needs
    # sub-cell detail for both elevation AND roughness together (same DEM
    # exception the user asked to keep).
    "local_roughness_native": {
        "data_type": "RasterDataset",
        "uri": str(roughness_path),
        "driver": "rasterio",
    },
    **({
        "local_delta_outflow_points": {
            "data_type": "GeoDataFrame",
            "uri": str(delta_outflow_points_path),
            "driver": "pyogrio",
        },
    } if delta_outflow_enabled else {}),
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)
log.info(f"Data catalog written: {local_catalog_path}")

# The burned channel always takes priority over the conditioned background
# wherever it has valid data (elevation_list merges by priority: first
# source wins, later sources fill gaps) -- so ocean/floodplain/gap cells
# transparently fall back to the conditioned DEM.
#
# TWO different burned-channel sources, for two different consumers, both
# falling back to the SAME conditioned (post-monotonicity) DEM -- just at
# each consumer's own native resolution, so neither needs HydroMT to
# reproject its background on the fly:
# - elevation_list_main (sf.elevation.create(), the main "dep" grid) uses
#   'local_river_burned_sfincs_grid' (burned directly at the SFINCS grid's
#   own resolution -- a perfect pixel-for-pixel match, no reprojection),
#   falling back to 'local_elevation_conditioned_sfincs_grid' (rule 09's
#   own SFINCS-grid-resolution output) -- the SAME background rule
#   modelled_depth_estimation's own calibration uses for its round 0.
# - elevation_list_subgrid (sf.subgrid.create()) keeps the NATIVE-resolution
#   'local_river_burned' -- subgrid needs real sub-cell (finer-than-grid-cell)
#   detail, the SFINCS-grid version would just duplicate one flat value
#   across every sub-cell -- falling back to 'local_elevation_conditioned'
#   (native resolution). This is the SAME catalog value (elevation_conditioned.tif)
#   rule modelled_depth_estimation's own subgrid step reads under its own
#   'local_elevation_conditioned' key -- confirmed identical 2026-08-04
#   after finding this rule's own copy was, until then, mislabeled
#   "elevation_merged" (a stale name from before rule
#   enforce_river_monotonicity existed) even though it always pointed at
#   elevation_conditioned.tif, never the actual raw elevation_merged.tif.
elevation_list_main = [
    {"elevation": "local_river_burned_sfincs_grid"}, {"elevation": "local_elevation_conditioned_sfincs_grid"},
]
elevation_list_subgrid = [{"elevation": "local_river_burned"}, {"elevation": "local_elevation_conditioned"}]

# ── load domain boundary ────────────────────────────────────────────────────────
delta_domain = gpd.read_file(domain_path)
log.info(f"Domain boundary: {len(delta_domain)} feature(s), CRS={delta_domain.crs}")

# ── initialise model ──────────────────────────────────────────────────────────
sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(skeleton_root),
    mode="w+",
    write_gis=True,
)
log.info("SfincsModel initialised")

# ── 1. Grid ───────────────────────────────────────────────────────────────────
sf.grid.create_from_region(
    region={"geom": delta_domain},
    res=resolution,
    crs="utm",
    rotated=False,
)
log.info(f"Grid created: {resolution} m, auto-UTM")

# ── 2. Elevation ──────────────────────────────────────────────────────────────
elevation_component = sf.elevation
elevation_component.create(
    elevation_list=elevation_list_main,
)
log.info(f"Elevation set from {[e['elevation'] for e in elevation_list_main]}")

# ── 3. Mask: active cells ─────────────────────────────────────────────────────
active_mask_kwargs = {}
if active_mask_enabled:
    with open(river_elevation_max_path) as f:
        river_elevation_max_m = float(json.load(f)["river_elevation_max_m"])
    clip_elevation_m = river_elevation_max_m + active_mask_elevation_buffer_m
    active_mask_kwargs = {"include_zmax": clip_elevation_m}
mask_component = sf.mask
mask_component.create_active(include_polygon=delta_domain, **active_mask_kwargs)
log.info(
    "Active mask created: cells within delta polygon"
    + (f" AND below {clip_elevation_m:.1f} m (river max {river_elevation_max_m:.1f} m + "
       f"{active_mask_elevation_buffer_m:.1f} m buffer)" if active_mask_kwargs else "")
)

# ── 4. Mask: waterlevel boundary ──────────────────────────────────────────────
# Open sea only, by the model grid's OWN land use (landuse_on_grid != 200 ->
# back to a normal active cell) -- same as rule 10's calibration model; see
# src.raster.restrict_waterlevel_boundary_to_sea.
# reset_bounds=False: create_active just built a fresh 0/1 mask, and hydromt's
# create_boundary with reset_bounds=True and no polygon/elevation filter only
# resets and returns -- it would set no boundary cells at all.
mask_component.create_boundary(btype="waterlevel", reset_bounds=False)
with rasterio.open(landuse_on_grid_path) as _lu_src_bnd:
    _bnd_mask, _n_bnd_on_land = restrict_waterlevel_boundary_to_sea(
        sf.grid.data["mask"].values, _lu_src_bnd.read(1)
    )
if not (_bnd_mask == 2).any():
    raise ValueError(
        "No water-level boundary cell on open sea (landuse_on_grid == 200) at the "
        "active-domain edge -- check the active mask and landuse_on_grid.tif."
    )
sf.grid.data["mask"].values[:] = _bnd_mask
log.info(
    f"Waterlevel boundary set: {int((_bnd_mask == 2).sum())} edge cell(s) on open sea → mask=2 "
    f"({_n_bnd_on_land} edge cell(s) on land use != 200 left as normal active cells)"
)

# ── 4b. Mask: delta-outline outflow boundary ─────────────────────────────────
if delta_outflow_enabled:
    mask_component.create_boundary(
        btype="outflow",
        include_polygon="local_delta_outflow_points",
        include_polygon_buffer=outflow_buffer_m,
        reset_bounds=False,
    )
    log.info(
        f"Outflow boundary set: {len(delta_outflow_gdf)} delta-outline "
        f"crossing(s), {outflow_buffer_m:.0f} m buffer → mask=3"
    )

# ── 4c. Coastal protection weir ───────────────────────────────────────────────
# rivers is loaded here rather than down in the subgrid section since both
# the channel mask below and the subgrid table need the same GeoDataFrame;
# the subgrid section reuses this variable instead of reading it again.
rivers = gpd.read_file(river_network_path)

_domain_wgs84 = delta_domain.to_crs("EPSG:4326")
_domain_union = _domain_wgs84.geometry.union_all()
_domain_poly_4326 = _domain_union if isinstance(_domain_union, Polygon) else _domain_union.convex_hull

rivers_utm = rivers.to_crs(sf.crs)
weir_gdf = gpd.GeoDataFrame({"elevation": [], "par1": []}, geometry=[], crs=sf.crs)
weir_grid = None

if depth_method == "modelled" and coastal_protection_weir_path is not None:
    from src.protection_weir import GridArrays, LANDUSE_SEA
    from src.river_burn import build_channel_mask_regular

    weir_grid = GridArrays.from_regular(sf.grid.data["dep"], sf.grid.data["mask"], sf.crs)
    weir_gdf = gpd.read_file(coastal_protection_weir_path)
    if not weir_gdf.empty:
        sf.weirs.set(weir_gdf, merge=False)
        log.info(f"Coastal protection weir imported directly from rule 9b: {len(weir_gdf)} segment(s)")
    else:
        log.info("Coastal protection weir: 9b's own file has no segments -- sf.weirs left empty")
    with rasterio.open(landuse_on_grid_path) as _lu_src:
        landuse_on_grid = _lu_src.read(1)
    river_channel_mask = build_channel_mask_regular(rivers_utm, "width", weir_grid.shape, weir_grid.transform)
    weir_diagnostics = {
        "applicable": True,
        "ocean_mask": landuse_on_grid == LANDUSE_SEA,
        "river_channel_mask": river_channel_mask,
        "weir_lines": list(weir_gdf.geometry),
        "crest_elevation_m": coastal_protection_crest_m,
    }

else:
    log.info(f"Coastal protection weir: none built -- depth_method={depth_method!r} has no calibrated crest data")
    weir_diagnostics = {"applicable": False}

Path(snakemake.output.weir_gpkg).parent.mkdir(parents=True, exist_ok=True)
if not weir_gdf.empty:
    weir_gdf.to_file(snakemake.output.weir_gpkg, driver="GPKG")
    log.info(f"Weir GeoPackage written: {snakemake.output.weir_gpkg} ({len(weir_gdf)} segment(s))")
else:
    Path(snakemake.output.weir_gpkg).touch()
    log.info("Weir GeoPackage: no segments -- empty sentinel written")

if weir_grid is not None:
    from src.plots import plot_coastal_protection_weir
    plot_coastal_protection_weir(
        weir_grid, weir_diagnostics, _domain_poly_4326,
        str(land_mask_path), str(river_network_path),
        str(snakemake.output.plot_coastal_protection_weir),
        basin_id=skeleton_root.parent.name, weir_gdf=weir_gdf,
    )
else:
    Path(snakemake.output.plot_coastal_protection_weir).touch()

# ── 5. Initial conditions (DEFAULT, non-river_only) ──────────────────────────
# The "ini" variable lives in the SAME grid Dataset as dep/mask/manning
# (SfincsInitialConditions.write() is a no-op -- "the ini file is written
# when all grid files are written", i.e. bundled into sf.grid.write()/
# sf.write() below) -- so this default (real, spatially-varying baseline_m)
# initial condition is built once here and forwarded like any other
# geometry file ("inifile") by build_sfincs's per-scenario forcing step.
# forcing_mode="river_only" is the ONE exception: it overrides this with a
# uniform dry start instead (see 13_build_sfincs.py's own initial-
# conditions section) -- excludes "inifile" from what it forwards, rather
# than anything changing here.
#
# Built directly from zsini_sea_cells_on_grid.tif -- already rasterized onto
# THIS model's own grid by whichever of rule modelled_depth_estimation/
# empirical_depth_estimation ran (zero further reprojection here: passing an
# in-memory DataArray built on sf.grid.data["dep"]'s own coords makes
# .create()'s internal reproject_like a confirmed true no-op, verified
# 0/263907 cells differ from the input array in a real basin's own
# skeleton). There used to be a two-step construction here -- read a
# NATIVE-resolution sea_mask_corrected.tif into HydroMT's own "nearest"
# reproject_like first, then patch the result with this same coarse-direct
# array afterward -- removed 2026-08-07 after confirming two independent
# nearest-neighbor passes over the SAME coarse-to-native-to-coarse round
# trip don't invert each other cleanly even on a pixel-identical grid
# (~53% of the corrected coastal-fringe cells were still wet under the old
# two-step version). One raster, one resampling pass, one construction
# method, matching what actually ends up in sfincs.ini exactly.
initial_conditions_component = sf.initial_conditions
with rasterio.open(zsini_sea_cells_path) as _src:
    _sea_cells_on_grid = _src.read(1)
if _sea_cells_on_grid.shape != sf.grid.data["dep"].shape:
    raise ValueError(
        f"zsini_sea_cells_on_grid.tif shape {_sea_cells_on_grid.shape} does not match "
        f"this model's own grid {sf.grid.data['dep'].shape} -- expected pixel-identical "
        f"grids (same domain_gpkg + grid_resolution.json fed to both rules)."
    )
# Dry cells are real np.nan from the start (not a -9999 sentinel) -- avoids
# relying on xarray operations (e.g. .where()) to preserve rio/nodata attrs
# set on an intermediate object, which isn't guaranteed across xarray
# versions.
_ini_coarse = np.where(_sea_cells_on_grid == np.float32(1.0), np.float32(baseline_m), np.nan)

# ── connected-component dry-out of isolated sea cells ─────────────────────
# When baseline_m is significantly negative (e.g. a large negative MDT
# correction) shallow connections between isolated depressions and the main
# ocean become dry, trapping pockets of water that were previously able to
# drain. The same hazard exists even when baseline_m == 0 (e.g. an inland
# lagoon disconnected from the open ocean by dry land in between), so this
# fix always runs, not just when baseline_m != 0. In "modelled" mode,
# protected_pocket_mask (baked into zsini_sea_cells_on_grid.tif already
# excludes weir-protected cells, but a genuinely isolated, UNPROTECTED
# pocket (not reachable from the open ocean, no weir involved either) can
# still exist and needs the same dry-out treatment. Fix: label connected
# components of sea cells (value=baseline_m), keep only the main open
# ocean, and set isolated wet cells (dep < baseline_m, so water depth > 0)
# to dep = dry start.
_dep_coarse = sf.grid.data["dep"].values.astype(np.float32)
_dep_nd_coarse = np.float32(sf.grid.data["dep"].rio.nodata or -9999.0)
_dep_valid_coarse = np.where(np.isclose(_dep_coarse, _dep_nd_coarse), np.float32(1e6), _dep_coarse)
_sea_cells_coarse = np.isclose(_ini_coarse, np.float32(baseline_m))
_labeled_coarse, _n_coarse = _ndimage_label(_sea_cells_coarse)
# The main open ocean is taken as the SINGLE LARGEST connected sea
# component, not "whichever components touch the raster array's own
# rectangular edge": our domains are a delta-polygon-shaped clip well
# inside their own bounding-box raster (NaN-padded corners), so the
# array's actual edge pixels are almost always outside-domain nodata, not
# real sea -- an edge-touching heuristic would misclassify most of the
# real open ocean as "isolated" here.
if _n_coarse > 0:
    _sizes_coarse = np.bincount(_labeled_coarse.ravel())
    _sizes_coarse[0] = 0  # exclude background (non-sea)
    _main_label_coarse = int(np.argmax(_sizes_coarse))
    _connected_coarse = _labeled_coarse == _main_label_coarse
else:
    _connected_coarse = np.zeros_like(_sea_cells_coarse)
_isolated_wet_coarse = _sea_cells_coarse & ~_connected_coarse & (_dep_valid_coarse < np.float32(baseline_m))
_n_fix_coarse = int(_isolated_wet_coarse.sum())
if _n_fix_coarse > 0:
    # Set isolated wet sea cells to their bed elevation → water depth = 0 (dry).
    _ini_coarse = np.where(_isolated_wet_coarse, _dep_valid_coarse, _ini_coarse)
    log.info(
        f"zsini connectivity fix: {_n_fix_coarse} isolated wet sea-cell(s) set to dep "
        f"(not reachable from domain boundary at baseline_m={baseline_m:+.4f} m)"
    )
else:
    log.info("zsini connectivity fix: no isolated wet sea cells found")

# reproj_method="nearest", NOT the default "average": zsini is a
# near-binary field (baseline_m at sea, NaN/nodata on land), and the
# source/destination grids are already identical here so this is a no-op
# resample regardless -- see 13_build_sfincs.py's own identical comment for
# the full "average" vs "nearest" rationale.
_da_ini_coarse = xr.DataArray(
    _ini_coarse, dims=sf.grid.data["dep"].dims, coords=sf.grid.data["dep"].coords,
)
_da_ini_coarse = _da_ini_coarse.rio.write_crs(sf.grid.data["dep"].rio.crs)
_da_ini_coarse = _da_ini_coarse.rio.write_transform(sf.grid.data["dep"].rio.transform())
_da_ini_coarse.raster.set_nodata(np.nan)

initial_conditions_component.create(ini=_da_ini_coarse, reproj_method="nearest")
_n_wet_coarse = int((sf.grid.data["ini"].values > -9998.0).sum())
log.info(
    f"Default initial conditions set: sea = {baseline_m:+.4f} m, land = -9999 (bed level / dry) "
    f"-- {_n_wet_coarse:,} wet cell(s) at this model's own grid resolution"
)

# zsini.tif (the file, declared as this rule's own output) mirrors
# sf.grid.data["ini"] exactly -- same array, written at this model's own
# grid resolution, so a standalone inspection of the file always matches
# what actually ends up in sfincs.ini.
_zsini_meta_coarse = {
    "driver": "GTiff", "dtype": "float32", "count": 1,
    "height": _ini_coarse.shape[0], "width": _ini_coarse.shape[1],
    "transform": sf.grid.data["dep"].rio.transform(), "crs": sf.grid.data["dep"].rio.crs,
    "nodata": np.float32(-9999.0), "compress": "deflate",
}
with rasterio.open(_zsini_out, "w", **_zsini_meta_coarse) as _dst:
    _dst.write(np.where(np.isnan(_ini_coarse), np.float32(-9999.0), _ini_coarse).astype(np.float32), 1)
log.info(f"zsini written: {_zsini_out} (sea cells = {baseline_m:+.4f} m, this model's own grid resolution)")

# ── 6. Roughness ─────────────────────────────────────────────────────────────
roughness_component = sf.roughness
roughness_component.create(
    roughness_list=[{"manning": "local_roughness"}],
)
log.info("Manning roughness set from 'local_roughness'")

# ── 7. Subgrid table ──────────────────────────────────────────────────────────
rivers["rivwth"] = rivers["width"].fillna(1.0).astype(float)
rivers["rivdph"] = rivers["rivdph"].clip(lower=0.0).astype(float)
river_list = []

subgrid_dir = skeleton_root / "subgrid"
if subgrid_dir.exists():
    for _stale in subgrid_dir.glob("*subgrid*.tif"):
        _stale.unlink()

if include_subgrid:
    log.info(
        f"River network for subgrid: {len(rivers)} reaches, "
        f"rivwth [{rivers['rivwth'].min():.1f}–{rivers['rivwth'].max():.1f} m], "
        f"rivdph [{rivers['rivdph'].min():.2f}–{rivers['rivdph'].max():.2f} m]"
    )
    subgrid_component = sf.subgrid
    subgrid_component.create(
        elevation_list=elevation_list_subgrid,
        roughness_list=[{"manning": "local_roughness_native"}],
        river_list=river_list,
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )
    log.info(
        f"Subgrid table created: {nr_subgrid_pixels} px/cell → "
        f"{resolution / nr_subgrid_pixels:.0f} m effective resolution"
    )
else:
    log.info("Subgrid skipped (include_subgrid=false in config)")

# ── 7. Observation points ────────────────────────────────────────────────────
# For each of the N_TOP_CROSSINGS boundary crossings with the highest bankfull
# discharge, place N_OBS_PER_CROSSING evenly-spaced observation points along
# the downstream river path (up to max_downstream_hops reaches). Uses only
# bankfull_discharge/has_glofas from river_forcing.nc -- basin-level fixed
# fields, not the design discharge at any particular RP.
import re as _re

river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
log.info(f"River forcing: {n_active}/{len(active)} crossings with valid GloFAS data (for observation points)")

N_OBS_PER_CROSSING = n_per_crossing
N_TOP_CROSSINGS    = n_top_crossings


def _norm_rid(x):
    """Normalise a reach ID to a plain int string, or None."""
    s = str(x).strip()
    return None if s.lower() in ("nan", "none", "<na>", "") else (
        str(int(float(s))) if s.replace(".", "").lstrip("-").isdigit() else s or None
    )


obs_points_list = []

if n_active >= 1 and not rivers.empty:
    bankfull_vals = river_ds.bankfull_discharge.values.copy()
    bankfull_vals[~active] = -np.inf   # exclude inactive crossings
    src_lons_all  = river_ds.longitude.values
    src_lats_all  = river_ds.latitude.values

    top_indices = np.argsort(bankfull_vals)[::-1][:N_TOP_CROSSINGS]
    top_indices = [i for i in top_indices if bankfull_vals[i] > 0]

    centroids_utm  = rivers_utm.geometry.centroid
    reach_lookup   = {_norm_rid(r["reach_id"]): r
                      for _, r in rivers_utm.iterrows()
                      if _norm_rid(r.get("reach_id"))}

    obs_id = 1
    for rank, ci in enumerate(top_indices):
        src_pt = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([src_lons_all[ci]], [src_lats_all[ci]]),
            crs="EPSG:4326",
        ).to_crs(sf.crs).geometry.iloc[0]

        nearest_rid = _norm_rid(
            rivers_utm.iloc[centroids_utm.distance(src_pt).idxmin()]["reach_id"]
        )

        dn_rows, current, visited = [], nearest_rid, set()
        for _ in range(max_downstream_hops):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            row = reach_lookup[current]
            dn_rows.append(row)
            dn_raw = str(row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            nxt = next((_norm_rid(t.strip()) for t in _re.split(r"[,\s]+", dn_raw)
                        if _norm_rid(t.strip())), None)
            if not nxt:
                break
            current = nxt

        if not dn_rows:
            log.warning(f"Crossing rank {rank+1}: no downstream reaches — skipping")
            continue

        dn_line   = gpd.GeoDataFrame(dn_rows, crs=sf.crs).geometry.union_all()
        total_len = dn_line.length
        for i in range(N_OBS_PER_CROSSING):
            dist = (i + 0.5) * total_len / N_OBS_PER_CROSSING
            obs_points_list.append({"obs_id": obs_id, "geometry": dn_line.interpolate(dist)})
            obs_id += 1

        log.info(
            f"Crossing rank {rank+1} (Q={bankfull_vals[ci]:.0f} m³/s): "
            f"{N_OBS_PER_CROSSING} obs points along {len(dn_rows)} downstream reaches"
        )

if obs_points_list:
    obs_gdf = gpd.GeoDataFrame(obs_points_list, crs=sf.crs)
    sf.observation_points.create(locations=obs_gdf, merge=False)
    log.info(f"Observation points: {len(obs_points_list)} total")
else:
    log.warning("No observation points placed")

# ── write ─────────────────────────────────────────────────────────────────────
# No forcing component (water_level/discharge_points) was ever .create()'d
# above, so sf.write() naturally skips those (SfincsModel.write() only
# writes populated components) -- this model is deliberately not runnable
# on its own (no tref/tstart/tstop/forcing config at all).
sf.write()
log.info(f"Skeleton written to {skeleton_root}")

subgrid_path = Path(snakemake.output.sfincs_subgrid)
if not subgrid_path.exists():
    subgrid_path.touch()
    log.info("sfincs_subgrid.nc placeholder written (subgrid not built)")

weir_path = Path(snakemake.output.sfincs_weir)
if not weir_path.exists():
    weir_path.touch()
    log.info("sfincs.weir placeholder written (no weir set)")

# ── diagnostic plots ──────────────────────────────────────────────────────────
figs_dir = Path(snakemake.output.plot_grid).parent
figs_dir.mkdir(parents=True, exist_ok=True)


def _save_with_buffer(fig, ax, fname, buffer_frac=0.15):
    """Expand axes extent by buffer_frac, then save."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    dx, dy = (xr - xl) * buffer_frac, (yt - yb) * buffer_frac
    ax.set_xlim(xl - dx, xr + dx)
    ax.set_ylim(yb - dy, yt + dy)
    fig.savefig(figs_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot written: {fname}")


# Background: the grid-aligned land mask (rule grid_align_landuse), not web
# map tiles -- see src.plots' module docstring.
# 1. Grid extent
fig, ax = sf.plot_basemap(variable="grid", plot_region=True, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "01_grid.png")

# 2. Elevation — obs points shown here only (plot_geoms=True is default)
fig, ax = sf.plot_basemap(variable="dep", vmin=-80, vmax=80)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "02_elevation.png")

# 3. Mask
fig, ax = sf.plot_basemap(variable="mask", plot_bounds=False, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "03_mask.png")

# 4. Roughness
fig, ax = sf.plot_basemap(variable="manning", plot_bounds=False, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "04_roughness.png")

# 5. Initial conditions (zsini) -- untracked side-effect plot (not in this
# rule's own output:, just follows figs_dir), same convention build_sfincs
# used before this split.
with rasterio.open(_zsini_out) as _src:
    _zsini_arr = _src.read(1).astype(np.float32)
    _nodata_val = _src.nodata
    _bounds = _src.bounds
    # SFINCS's own grid is y-ASCENDING (row 0 = south, positive row step),
    # unlike the GDAL-standard north-up raster imshow assumes, so this plot
    # came out mirrored vertically until 2026-09-17.
    _row_step = _src.transform.e
    _src_crs_zsini = _src.crs
_origin = "lower" if _row_step > 0 else "upper"

if _nodata_val is not None:
    _zsini_arr = np.where(
        np.isclose(_zsini_arr, np.float32(_nodata_val)), np.nan, _zsini_arr
    )

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    _zsini_arr,
    extent=[_bounds.left, _bounds.right, _bounds.bottom, _bounds.top],
    origin=_origin,
    cmap="Blues",
    aspect="auto",
)
plt.colorbar(im, ax=ax, label="Initial water level (m)")
ax.set_title(
    f"Initial conditions (zsini)\n"
    f"sea = {baseline_m:+.4f} m  |  land = nodata"
)
# Kept on the model's own grid, labelled in lon/lat (see label_axes_lonlat).
label_axes_lonlat(ax, _src_crs_zsini)
_save_with_buffer(fig, ax, "06_zsini.png")
