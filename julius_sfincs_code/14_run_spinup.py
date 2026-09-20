"""
14_run_spinup.py — Run a short, basin-level SFINCS spin-up to produce a
restart file, with the river at a fixed RP=1 and a calm sea, entirely
independent of any scenario's own design RP.

Runs ONCE per basin (no {scenario} wildcard) -- every scenario's own event
run (rule run_event, 16) picks up from the SAME restart file via its own
sfincs.inp's rstfile entry (set in 13_build_sfincs.py). Changing a
scenario's own RP (surge_rp/river_rp in config/scenarios.yml) never
touches this rule's own inputs, so it never needs to re-run for that
reason -- only a real change to the basin's own skeleton (grid/mask/
elevation/etc, or a change to spinup_days/the spin-up forcing itself) does.

Borrows ALL geometry (grid/elevation/mask/roughness/weir/subgrid/
observation points, plus the skeleton's own default spatially-varying
initial condition) from 13_build_sfincs_skeleton.py's own output, via the
SAME technique 13_build_sfincs.py uses for its own scenario forcing: load
the skeleton in read mode, redirect writes to this rule's own directory
(sf.root.set(...)), write ONLY the newly-built forcing components, and
hand-craft this rule's own sfincs.inp forwarding the skeleton's geometry
files via a relative path (src.sfincs_run.forward_geometry_files) rather
than through HydroMT's own sf.write() (which would duplicate the skeleton
into this directory) or sf.config.write() (which silently absolutizes any
file reference outside the model's own root -- see either script's own
module docstring for the full rationale).

Forcing: each crossing's own real RP=1 discharge (`interpolate_discharge_
at_rp` -- an exact table lookup, the river GPD return-value table tabulates
RP=1 directly), held CONSTANT over the whole spinup_days duration, and the
sea at each station's calm-sea level (`calm_sea_levels`, the same level
every event's own boundary lead-in starts at) -- spin-up exists to let the
river network reach a realistic steady background state, not to simulate
an event.

Why a calm sea, not RP=1 (changed 2026-09-11): the event's boundary lead-in
starts at the calm-sea level, so an RP=1 spin-up sea made the boundary drop
0.30 m (basin 2433835) the instant the event started from the restart,
sending a drawdown wave in -- and the spin-up's own rise to RP=1 from the
calm starting sea had already overtopped the coastal dike. A calm spin-up
sea matches both the starting sea (zsini) and the event lead-in.

Outputs (all directly under results/{basin_id}/spin_up/, not nested under
any scenario)
-------
sfincs.inp                          This rule's own hand-crafted config.
sfincs.YYYYMMDD.HHMMSS.rst          SFINCS binary restart file at t = spinup_days.
sfincs_map.nc / sfincs_his.nc       Spin-up's own run output.
validation_spinup.png               Water-level timeseries at observation points.
validation_max_inundation.png       Max inundation depth at spin-up end.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from hydromt_sfincs import SfincsModel

from src.geometry import snap_points_into_region
from src.log import setup_logging
from src.plots import plot_max_inundation_map, plot_water_level_timeseries
from src.postprocessing import compute_max_inundation
from src.river_forcing import interpolate_discharge_at_rp
from src.sfincs_run import forward_geometry_files, parse_sfincs_inp, run_sfincs_subprocess
from src.surge import calm_sea_levels, read_baseline_m

log = setup_logging(snakemake.log[0])

# ── params ────────────────────────────────────────────────────────────────────
skeleton_root       = Path(snakemake.params.skeleton_root)
spin_up_root        = Path(snakemake.params.spin_up_root)
resolution          = snakemake.params.resolution
tref_str            = snakemake.params.tref
spinup_days         = snakemake.params.spinup_days
sfincs_exe          = Path(snakemake.params.sfincs_exe)
rst_fname           = snakemake.params.rst_fname  # e.g. sfincs.20000102.000000.rst
dtmapout_s          = int(snakemake.params.dtmapout_s)
dthisout_s          = int(snakemake.params.dthisout_s)
include_subgrid     = bool(snakemake.params.include_subgrid)
timeout_s           = int(snakemake.params.timeout_s)
waterlevel_buffer_m = snakemake.params.waterlevel_buffer_m
boundary_ramp_hours = float(snakemake.params.boundary_ramp_hours)
land_polygons_path  = Path(snakemake.input.land_mask_on_grid)  # grid-aligned land mask, plot background
sea_mask_path       = Path(snakemake.input.sea_mask)
river_network_path  = Path(snakemake.input.clean_river_network)
domain_gpkg_path    = Path(snakemake.input.domain_gpkg)
surge_forcing_path  = Path(snakemake.input.surge_forcing)
river_forcing_path  = Path(snakemake.input.river_forcing)

spin_up_root.mkdir(parents=True, exist_ok=True)

# Load domain polygon in WGS84 for overlay plots.
_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)

# ── load skeleton, redirect writes to spin_up_root ───────────────────────────
sf = SfincsModel(root=str(skeleton_root), mode="r")
sf.read()
sf.root.set(spin_up_root, mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {spin_up_root}")

skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# ── timing ────────────────────────────────────────────────────────────────────
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")
tstop = tref + timedelta(days=spinup_days)
trstout_sec = int(spinup_days * 86400)
spinup_times = pd.DatetimeIndex([tref, tstop])  # 2-point constant timeseries

log.info(f"Spin-up: {tref} → {tstop} ({spinup_days} days), restart written at t={trstout_sec} s")

# sf.water_level.create()/sf.discharge_points.create() below both slice their
# own timeseries against self.model.get_model_time() (tstart/tstop read
# straight off the in-memory sf.config) -- the skeleton's own sfincs.inp
# deliberately never sets tref/tstart/tstop (it's not meant to be runnable
# on its own), so without this, sf.config still carries hydromt_sfincs's own
# Pydantic defaults (today's date), which never overlaps spinup_times (indexed
# at `tref`, e.g. 2000-01-01) -- NoDataException: "DataFrame has no data
# after time slicing." This has no effect on the actual sfincs.inp written to
# disk below (hand-crafted via plain file I/O, never sf.config.write()) --
# it only fixes what these in-memory .create() calls see.
sf.config.set("tref", tref)
sf.config.set("tstart", tref)
sf.config.set("tstop", tstop)

# ── water-level boundary forcing: calm sea ───────────────────────────────────
# Each station's calm-sea level (calm_sea_levels) -- the level the event's
# own boundary lead-in starts at, so the event continues from the restart
# with no jump at the boundary (see this module's own docstring). The sea
# itself starts at baseline_m (the skeleton's zsini: the ROUNDED basin mean,
# while the station levels are rounded individually -- normally equal, but
# up to 0.1 m apart), so the boundary ramps from that start level to the
# station levels over boundary_ramp_hours rather than stepping: a step at
# the boundary sends a front in that shoals and reflects at the coast (a
# 0.30 m step built up to +1.1 m at the 0.40 m coastal dike of basin
# 2433835).
surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
n_stations = surge_ds.sizes["station"]
stations_gdf = gpd.GeoDataFrame(
    {"index": range(n_stations)},
    geometry=gpd.points_from_xy(surge_ds.longitude.values, surge_ds.latitude.values),
    crs="EPSG:4326",
)
start_level = read_baseline_m(surge_ds)  # the sea's own starting level (skeleton zsini)
calm_levels = calm_sea_levels(surge_ds)  # (n_station,), the event lead-in's own level
ramp_end = tref + timedelta(hours=min(boundary_ramp_hours, spinup_days * 24.0))
wl_times = pd.DatetimeIndex(sorted({tref, ramp_end, tstop}))
wl_values = np.array([np.full(n_stations, start_level) if t == tref else calm_levels for t in wl_times])
wl_df = pd.DataFrame(data=wl_values, index=wl_times, columns=range(n_stations))
sf.water_level.create(timeseries=wl_df, locations=stations_gdf, buffer=waterlevel_buffer_m)
log.info(
    f"Water-level forcing: {n_stations} station(s) at the calm-sea level "
    f"({calm_levels.min():+.2f}..{calm_levels.max():+.2f} m; ramped from the starting sea "
    f"{start_level:+.2f} m over {boundary_ramp_hours:.1f} h), held until {tstop}"
)
sf.water_level.write()

# ── river discharge forcing: RP=1, constant over time ────────────────────────
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
log.info(f"River forcing: {n_active}/{len(active)} crossings with valid GloFAS data")

if n_active == 0:
    log.warning("No active river crossings — discharge forcing skipped")
else:
    table = river_ds["discharge_rp_table"].values[active]
    table_rps = river_ds["return_period"].values
    rp1_discharge = interpolate_discharge_at_rp(table, table_rps, 1.0)  # (n_active,)

    crossings_gdf = gpd.GeoDataFrame(
        {"index": range(n_active)},
        geometry=gpd.points_from_xy(
            river_ds.longitude.values[active], river_ds.latitude.values[active],
        ),
        crs="EPSG:4326",
    )
    inside_reach_ids = (
        river_ds.inside_reach_id.values[active] if "inside_reach_id" in river_ds else [None] * n_active
    )

    dis_df = pd.DataFrame(
        data=np.tile(rp1_discharge, (len(spinup_times), 1)),
        index=spinup_times,
        columns=range(n_active),
    )

    from src.river_burn import build_centerline_cells_regular, snap_points_to_centerline_cells

    rivers_utm = gpd.read_file(river_network_path).to_crs(sf.crs)
    centerline_cells = build_centerline_cells_regular(
        rivers_utm, sf.grid.data["dep"].shape, sf.grid.data["dep"].rio.transform()
    )
    crossings_gdf = snap_points_to_centerline_cells(
        crossings_gdf.to_crs(sf.crs), centerline_cells,
        reach_ids=inside_reach_ids, resolution_m=resolution,
    ).to_crs("EPSG:4326")

    buf_deg = float(resolution) / 111_000.0
    region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
    crossings_filt, in_region = snap_points_into_region(crossings_gdf, region_wgs84, buf_deg)
    n_outside = int((~in_region).sum())
    if n_outside:
        log.warning(f"  {n_outside}/{n_active} discharge crossing(s) outside active SFINCS region — skipped")

    if crossings_filt.empty:
        log.warning("All discharge crossings outside active region — discharge forcing skipped")
    else:
        sf.discharge_points.create(timeseries=dis_df, locations=crossings_filt)
        log.info(f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s) at RP=1, held constant")
sf.discharge_points.write()

# ── hand-craft this rule's own sfincs.inp ────────────────────────────────────
# Forward the skeleton's own grid-header + geometry "*file" entries
# (depfile/mskfile/manningfile/sbgfile/weirfile/inifile) via a relative
# path -- see this script's own module docstring. "inifile" is forwarded
# UNCONDITIONALLY here (unlike build_sfincs's own river_only exclusion):
# spin-up always represents a realistic steady background state, so it
# always starts from the skeleton's own real (baseline_m) initial
# condition, never a uniform dry start.
exclude_keys = frozenset({"rstfile", "bzsfile", "bndfile", "disfile", "srcfile", "netsrcdisfile"})
geometry_lines = forward_geometry_files(skeleton_cfg, skeleton_root, spin_up_root, exclude=exclude_keys)

OWNED_SCALAR_KEYS = frozenset({
    "tref", "tstart", "tstop", "dtmapout", "dtmaxout", "dthisout", "dtrstout", "trstout",
    "storevel", "storevelmax", "storecumprcp", "storemeteo", "storetwet", "baro", "zsini",
})
scalar_lines = [
    f"{k:<20} = {v}" for k, v in skeleton_cfg.items()
    if not k.endswith("file") and k not in OWNED_SCALAR_KEYS
]

lines = list(scalar_lines) + list(geometry_lines) + [
    f"{'tref':<20} = {tref.strftime('%Y%m%d %H%M%S')}",
    f"{'tstart':<20} = {tref.strftime('%Y%m%d %H%M%S')}",
    f"{'tstop':<20} = {tstop.strftime('%Y%m%d %H%M%S')}",
    f"{'trstout':<20} = {trstout_sec}",   # write restart at spinup end
    f"{'dtrstout':<20} = 0",              # no interval rst output, only trstout
    f"{'dthisout':<20} = {dthisout_s}",
    f"{'dtmapout':<20} = {dtmapout_s}",
    f"{'dtmaxout':<20} = {trstout_sec}",  # write max-envelope (zsmax) at spinup end
    f"{'baro':<20} = 0",
    f"{'zsini':<20} = -9999.0",
    # Spin-up needs none of these instantaneous/diagnostic fields.
    f"{'storevel':<20} = 0",
    f"{'storevelmax':<20} = 0",
    f"{'storecumprcp':<20} = 0",
    f"{'storemeteo':<20} = 0",
    f"{'storetwet':<20} = 0",
]
# rstfile is intentionally NOT set: SFINCS would try to READ it at startup,
# but it doesn't exist yet for this cold-start spin-up. Without it, SFINCS
# starts from inifile (forwarded above) instead.

for _key in ("bndfile", "bzsfile", "srcfile", "disfile", "netsrcdisfile"):
    _val = sf.config.get(_key)
    if _val:
        lines.append(f"{_key:<20} = {_val}")

spinup_inp = spin_up_root / "sfincs.inp"
with open(spinup_inp, "w") as fh:
    fh.write("\n".join(lines) + "\n")
log.info(f"Spinup sfincs.inp written: {spinup_inp}")

# ── run SFINCS ────────────────────────────────────────────────────────────────
run_sfincs_subprocess(sfincs_exe, spin_up_root, timeout_s, log, label="SFINCS spin-up", n_threads=snakemake.threads)

rst_path = spin_up_root / rst_fname
if not rst_path.exists():
    written = [f.name for f in spin_up_root.iterdir() if f.is_file()]
    raise FileNotFoundError(
        f"SFINCS ran but expected restart file not found: {rst_path}\n"
        f"Files in spin_up dir: {written}"
    )
log.info(f"Restart file written: {rst_path} ({rst_path.stat().st_size / 1e6:.1f} MB)")

# ── validation plot ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()
plot_path = Path(snakemake.output.plot_spinup)
his_path  = spin_up_root / "sfincs_his.nc"

if not his_path.exists():
    log.warning(f"No history file found at {his_path} — skipping validation plot")
    plot_path.touch()
else:
    try:
        ds = xr.open_dataset(his_path)
        t = ds["time"].values
        if np.issubdtype(t.dtype, np.datetime64):
            t_days = (t - t[0]) / np.timedelta64(1, "D")
        else:
            t_days = np.asarray(t, dtype=float) / 86400.0
    except Exception:
        ds = xr.open_dataset(his_path, decode_times=False)
        t_days = np.asarray(ds["time"].values, dtype=float) / 86400.0

    zs_var = next((v for v in ("point_zs", "zs") if v in ds), None)
    if zs_var is None:
        log.warning(f"No water-level variable found in {his_path}. Available: {list(ds.data_vars)}. Skipping plot.")
        plot_path.touch()
    else:
        zs = ds[zs_var].values
        if zs.ndim == 1:
            zs = zs[:, np.newaxis]
        if zs.shape[0] != len(t_days):
            zs = zs.T

        plot_water_level_timeseries(
            t_days, zs, plot_path,
            day_markers=[(spinup_days, f"Day {spinup_days} (restart written)")],
            basin_id=f"Basin {spin_up_root.parent.name}",
            run_label=f"Spin-up validation | lines should be near-flat at day {spinup_days} if spin-up is sufficient",
            ylabel=f"Water level — {zs_var} (m)",
        )
        log.info(f"Validation plot written: {plot_path}")

# ── max inundation depth map ──────────────────────────────────────────────────
plot_inundation_path = Path(snakemake.output.plot_max_inundation)

da_hmax, _da_dep = compute_max_inundation(
    spin_up_root, skeleton_root, sea_mask_path, hmin=0.0, include_subgrid=include_subgrid,
)
if da_hmax is None:
    log.warning("No max inundation data available — creating empty plot sentinel")
    plot_inundation_path.touch()
else:
    plot_max_inundation_map(
        da_hmax, domain_poly, str(land_polygons_path), str(river_network_path),
        str(plot_inundation_path), basin_id=spin_up_root.parent.name, run_label="spinup",
    )
    log.info(f"Max inundation plot written: {plot_inundation_path}")

