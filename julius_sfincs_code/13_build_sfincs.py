"""
13_build_sfincs.py — Build the scenario-DEPENDENT forcing on top of a
basin's already-built SFINCS skeleton (13_build_sfincs_skeleton.py,
scenario-independent grid/elevation/mask/weir/roughness/subgrid/
observation points, one per basin, not one per scenario).

Loads the skeleton in read mode, redirects subsequent writes to this
scenario's own directory (sf.root.set(...) -- a well-established HydroMT
pattern for "keep the in-memory model, change where writes land", already
used this same way in 10_depth_estimation_modelled.py's own calibration
round loop), builds ONLY the forcing-related components (initial
conditions, water-level boundary, river discharge) that actually depend on
forcing_mode/design_rp_river_yr/design_rp_surge_yr, writes JUST those
(sf.water_level.write()/sf.discharge_points.write() -- independently
callable per-component writers, confirmed via hydromt_sfincs source), and
hand-crafts this scenario's own sfincs.inp: the skeleton's own grid-header
scalars and non-forcing "*file" entries (depfile/mskfile/manningfile/
sbgfile/weirfile/inifile) are forwarded via a relative path back to the
skeleton (computed with os.path.relpath, not hand-derived "../" counting)
using src.sfincs_run.forward_geometry_files -- NEVER re-written/duplicated
into this scenario's own directory. This sidesteps a real HydroMT
behavior: its own config-writing path silently ABSOLUTIZES any file
reference outside the model's current root instead of preserving a
relative "../" string, so a genuinely portable cross-directory reference
has to be written by hand (same technique 14_run_spinup.py already used
for borrowing this rule's own output, before this split).

Splitting the build this way means changing a scenario's own RP
(surge_rp/river_rp in config/scenarios.yml) only re-runs THIS (cheap)
script, not the expensive HydroMT skeleton build -- and, more importantly,
does not force rule run_spinup (now basin-level, RP=1 river + calm sea, entirely
independent of any scenario's own RP) to re-run either.

Forcing mode (derived per-scenario by scenario_params in 00_common.smk,
from the {scenario}'s own river_rp/surge_rp in config/scenarios.yml)
---------------------------------------------------
Controls which forcing(s) actually drive the model, independent of the
input files themselves (always loaded so duration/observation points stay
consistent across modes):
  "compound"     — real surge/tide boundary + real river discharge (default).
                   sfincs.boundary_setup.compound.lag_hr optionally shifts the
                   river discharge timeseries relative to the surge
                   timeseries (positive = river peak arrives later; negative
                   = earlier) — see the "compound lag" section below.
  "coastal_only" — real surge/tide boundary; river discharge forced to 0 at
                   every crossing (discharge points are still created, just
                   with a zeroed timeseries, so the model structure matches
                   the other modes).
  "river_only"   — real river discharge; the coastal water-level boundary is
                   replaced by a flat constant BELOW terrain.gebco_max_depth_m
                   (river_only_flat_level_m, see section below) — guaranteed
                   dry everywhere, so no surge/tide variability and no
                   coastal inflow of any kind confounds the river-discharge
                   contribution being isolated. Initial conditions are also
                   overridden to a uniform dry start (no "inifile" forwarded
                   from the skeleton) for the same reason.

discharge_multiplier (per-scenario, config/scenarios.yml, default 1.0 --
see scenario_params in 00_common.smk) uniformly scales the built discharge
hydrograph at every active river seed/boundary crossing -- applied HERE, at
build time (src.river_forcing.build_design_discharge_matrix), never baked
into river_forcing.nc. Mirrors src.surge's deferred SLR fingerprint: rule 07
and rule 10's weir/depth calibration (which also reads river_forcing.nc, for
its own calibration seed discharge) stay completely independent of this
factor, so changing it only reruns this per-scenario build and its
downstream event run. Being per-scenario (not global) lets e.g.
river_500/river_only_500/compound_500 be amplified past their raw RP lookup
without affecting coast_500's own small RP=2 river_rp.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from hydromt_sfincs import SfincsModel

from src.geometry import snap_points_into_region
from src.river_forcing import build_design_discharge_matrix
from src.sfincs_run import forward_geometry_files, parse_sfincs_inp
from src.surge import build_design_surge_matrix

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
surge_forcing_path = Path(snakemake.input.surge_forcing)
river_forcing_path = Path(snakemake.input.river_forcing)
skeleton_root       = Path(snakemake.params.skeleton_root)
sfincs_root         = Path(snakemake.params.sfincs_root)
spin_up_root        = Path(snakemake.params.spin_up_root)
resolution        = snakemake.params.resolution

tref_str          = snakemake.params.tref
dtmapout          = snakemake.params.dtmapout
dtmaxout          = snakemake.params.dtmaxout
dthisout          = snakemake.params.dthisout
storevelmax       = snakemake.params.storevelmax
storetwet         = snakemake.params.storetwet
forcing_mode      = snakemake.params.forcing_mode
river_only_flat_level_m = float(snakemake.params.river_only_flat_level_m)
# None for a scenario with no design RP on that side (config/scenarios.yml --
# e.g. "coast_100" has river_rp=None -> forcing_mode="coastal_only",
# "river_100" has surge_rp=None -> forcing_mode="river_only") -- NOT forced
# to float unconditionally here, since float(None) raises. design_rp_river_yr
# flows into build_design_discharge_matrix, which already accepts None
# natively (falls back to a constant bankfull hydrograph); design_rp_surge_yr
# is only ever dereferenced inside the `forcing_mode != "river_only"` branch,
# i.e. exactly when it's guaranteed non-None.
design_rp_river_yr = snakemake.params.design_rp_river_yr
design_rp_river_yr = None if design_rp_river_yr is None else float(design_rp_river_yr)
design_rp_surge_yr = snakemake.params.design_rp_surge_yr
design_rp_surge_yr = None if design_rp_surge_yr is None else float(design_rp_surge_yr)
compound_lag_hr   = float(snakemake.params.compound_lag_hr)
# Uniform scaling factor on the built river discharge hydrograph (default
# 1.0 = no-op), applied HERE -- not baked into river_forcing.nc, see this
# script's own module docstring.
discharge_multiplier = float(snakemake.params.discharge_multiplier)
# Target global-mean SLR (m), applied HERE (not baked into surge_forcing.nc,
# see src.surge.apply_slr_fingerprint's own docstring) against each
# station's own slr_fingerprint ratio -- 0.0 whenever SLR is disabled, so
# this scenario's own forcing matches surge_forcing.nc's MDT-only fields
# exactly.
effective_slr_m = float(snakemake.params.slr_m) if snakemake.params.slr_enabled else 0.0
flat_boundary_point_spacing_m = snakemake.params.flat_boundary_point_spacing_m
waterlevel_buffer_m = snakemake.params.waterlevel_buffer_m
include_rstart    = snakemake.params.include_rstart
spinup_days       = snakemake.params.spinup_days
depth_method      = snakemake.params.depth_method
rst_fname         = snakemake.params.rst_fname

sfincs_root.mkdir(parents=True, exist_ok=True)
log.info(f"Forcing mode: {forcing_mode!r}")

# ── load skeleton, redirect writes to this scenario's own directory ─────────
sf = SfincsModel(root=str(skeleton_root), mode="r")
sf.read()
sf.root.set(sfincs_root, mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {sfincs_root}")

skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# ── simulation config ──────────────────────────────────────────────────────────
# tref = tstart: forcing timeseries are in "hours since simulation start" so
# any fixed reference date works — the origin t=0 maps to tref exactly.
# tstop is derived from the actual end of the forcing files so it stays
# consistent even if config parameters (lead_days, period_hr) change.
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")

with xr.open_dataset(surge_forcing_path, decode_times=False) as surge_ds:
    surge_end_hr = float(surge_ds.time.max())
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    river_end_hr = float(river_ds.time.max())

sim_hours = max(surge_end_hr, river_end_hr)
tstop = tref + timedelta(hours=sim_hours)

log.info(
    f"Simulation period: {tref} → {tstop} "
    f"({sim_hours:.0f} h from forcing files)"
)

# sf.water_level.create()/sf.discharge_points.create() below both slice their
# own timeseries against self.model.get_model_time() (tstart/tstop read
# straight off the in-memory sf.config) -- the skeleton's own sfincs.inp
# deliberately never sets tref/tstart/tstop (it's not meant to be runnable
# on its own), so without this, sf.config still carries hydromt_sfincs's own
# Pydantic defaults (today's date), which never overlaps this scenario's
# real forcing timeseries (indexed at `tref`, e.g. 2000-01-01) --
# NoDataException: "DataFrame has no data after time slicing." This has no
# effect on the actual sfincs.inp written to disk below (hand-crafted via
# plain file I/O, never sf.config.write()) -- it only fixes what these
# in-memory .create() calls see.
sf.config.set("tref", tref)
sf.config.set("tstart", tref)
sf.config.set("tstop", tstop)

# ── initial conditions ────────────────────────────────────────────────────────
# forcing_mode="river_only": leave every cell (sea AND land) at the uniform
# zsini=-9999 default -- i.e. every cell starts dry at its own bed level,
# including the ocean. The river_only boundary is a flat constant BELOW
# terrain.gebco_max_depth_m (river_only_flat_level_m, below) -- guaranteed
# dry, so the ocean stays dry (no fill-in from the boundary) for the entire
# run, same as every other cell. Giving the ocean any head start here (as
# compound/coastal_only do, where real surge/tide dynamics need a
# consistent non-transient sea state) would just pre-fill areas right when
# the point of river_only is to isolate the river's own contribution
# against a neutral, dry coast that never contributes water of its own.
#
# Every other mode: the skeleton's own "inifile" (baseline_m spatially-
# varying, built once in 13_build_sfincs_skeleton.py) is forwarded below
# via forward_geometry_files like any other geometry file -- no need to
# rebuild it per scenario.
if forcing_mode == "river_only":
    ini_exclude_keys = frozenset({"inifile", "ncinifile"})
    log.info(
        "Initial conditions: uniform zsini=-9999 (dry at own bed level, "
        "incl. ocean) -- forcing_mode='river_only', boundary stays dry "
        "(below the GEBCO depth clamp) for the entire run"
    )
else:
    ini_exclude_keys = frozenset()
    log.info("Initial conditions: forwarding skeleton's own spatially-varying inifile")

# ── water-level boundary forcing (surge) ──────────────────────────────────────
# sf.water_level.create() with timeseries + locations writes ASCII .bnd/.bzs
# files (into sfincs_root, since root was already redirected above) and
# spatially matches each station to the nearest boundary cell (mask=2,
# already loaded from the skeleton) within the given buffer.
if forcing_mode == "river_only":
    sf.water_level.create_boundary_points_from_mask(bnd_dist=flat_boundary_point_spacing_m)
    sf.water_level.create_timeseries(shape="constant", offset=river_only_flat_level_m)
    log.info(
        f"Water-level forcing: flat constant boundary at {river_only_flat_level_m:+.4f} m "
        f"(forcing_mode='river_only')"
    )
else:
    surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
    surge_times = pd.DatetimeIndex(
        [tref + timedelta(hours=float(h)) for h in surge_ds.time.values]
    )

    n_stations = surge_ds.sizes["station"]
    stations_gdf = gpd.GeoDataFrame(
        {"index": range(n_stations)},
        geometry=gpd.points_from_xy(
            surge_ds.longitude.values,
            surge_ds.latitude.values,
        ),
        crs="EPSG:4326",
    )

    # water_level dims: (station, time) → transpose to (time, station) for DataFrame
    wl_df = pd.DataFrame(
        data=build_design_surge_matrix(
            surge_ds, design_rp_surge_yr, slr_m=effective_slr_m
        ).T,
        index=surge_times,
        columns=range(n_stations),
    )

    sf.water_level.create(
        timeseries=wl_df,
        locations=stations_gdf,
        buffer=waterlevel_buffer_m,
    )
    log.info(
        f"Water-level forcing: {n_stations} stations, {len(surge_times)} time steps "
        f"(SLR: {effective_slr_m:+.3f} m target × per-station fingerprint)"
    )
sf.water_level.write()

# ── river discharge forcing ───────────────────────────────────────────────────
# river_forcing.nc holds one timeseries per boundary crossing. Only crossings
# with has_glofas=1 have a calibrated EVA fit and a meaningful discharge signal;
# crossings without GloFAS data carry zero discharge and are excluded.
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
n_total  = len(active)
log.info(f"River forcing: {n_active}/{n_total} crossings with valid GloFAS data")

if n_active == 0:
    log.warning("No active river crossings — discharge forcing skipped")
else:
    river_times = pd.DatetimeIndex(
        [tref + timedelta(hours=float(h)) for h in river_ds.time.values]
    )

    crossings_gdf = gpd.GeoDataFrame(
        {"index": range(n_active)},
        geometry=gpd.points_from_xy(
            river_ds.longitude.values[active],
            river_ds.latitude.values[active],
        ),
        crs="EPSG:4326",
    )
    inside_reach_ids = (
        river_ds.inside_reach_id.values[active] if "inside_reach_id" in river_ds else [None] * n_active
    )

    # discharge dims: (crossing, time) → transpose to (time, crossing) for DataFrame
    dis_df = pd.DataFrame(
        data=build_design_discharge_matrix(
            river_ds, active, design_rp_river_yr,
            apply_protection_floor=(depth_method == "empirical"),
            discharge_multiplier=discharge_multiplier,
        ).T,
        index=river_times,
        columns=range(n_active),
    )
    log.info(f"Discharge multiplier applied: x{discharge_multiplier:.3f}")

    # ── compound lag: shift river discharge relative to surge ────────────────
    if forcing_mode == "compound" and compound_lag_hr != 0 and len(river_times) > 1:
        dt_hr = float((river_times[1] - river_times[0]).total_seconds() / 3600.0)
        shift_steps = int(round(compound_lag_hr / dt_hr))
        if shift_steps != 0:
            # Scaled the same way build_design_discharge_matrix scaled the
            # rest of dis_df, so the padding value stays consistent with the
            # (already-multiplied) real data being shifted alongside it.
            bankfull_active = river_ds.bankfull_discharge.values[active] * discharge_multiplier
            arr = dis_df.to_numpy()
            shifted = np.empty_like(arr)
            n = min(abs(shift_steps), arr.shape[0])
            if shift_steps > 0:
                shifted[:n, :] = bankfull_active[np.newaxis, :]
                shifted[n:, :] = arr[: arr.shape[0] - n, :]
            else:
                shifted[: arr.shape[0] - n, :] = arr[n:, :]
                shifted[arr.shape[0] - n :, :] = bankfull_active[np.newaxis, :]
            dis_df = pd.DataFrame(shifted, index=dis_df.index, columns=dis_df.columns)
            log.info(
                f"Compound lag applied: river discharge shifted {compound_lag_hr:+.1f} h "
                f"relative to surge ({shift_steps:+d} step(s) at dt={dt_hr:.2f} h); "
                f"{'start' if shift_steps > 0 else 'end'} padded with each "
                f"crossing's bankfull discharge"
            )

    if forcing_mode == "coastal_only":
        # Discharge points are still created below (zeroed) rather than skipped
        # entirely, so the model structure matches the other two modes.
        dis_df.loc[:, :] = 0.0
        log.info("River discharge forced to 0.0 m3/s at all crossings (forcing_mode='coastal_only')")

    # Snap each crossing onto the grid cell its OWN reach's centerline
    # actually passes through -- see src.river_burn.
    # rivers_utm isn't available in this script (river network geometry
    # lives entirely in the skeleton build) -- snapping to centerline cells
    # needs the actual reach geometries, so re-read them directly here.
    from src.river_burn import build_centerline_cells_regular, snap_points_to_centerline_cells

    rivers_utm = gpd.read_file(snakemake.input.river_network).to_crs(sf.crs)
    centerline_cells = build_centerline_cells_regular(
        rivers_utm, sf.grid.data["dep"].shape, sf.grid.data["dep"].rio.transform()
    )
    crossings_gdf = snap_points_to_centerline_cells(
        crossings_gdf.to_crs(sf.crs), centerline_cells,
        reach_ids=inside_reach_ids, resolution_m=resolution,
    ).to_crs("EPSG:4326")

    # Filter crossing points to those within the active SFINCS region, and
    # snap any kept point that falls just outside the exact region back
    # inside -- see src.geometry.snap_points_into_region's docstring.
    buf_deg = float(resolution) / 111_000.0
    region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
    crossings_filt, in_region = snap_points_into_region(crossings_gdf, region_wgs84, buf_deg)
    n_outside = int((~in_region).sum())
    if n_outside:
        log.warning(
            f"  {n_outside}/{n_active} discharge crossing(s) outside active SFINCS "
            f"region — skipped"
        )

    if crossings_filt.empty:
        log.warning("All discharge crossings outside active region — discharge forcing skipped")
    else:
        sf.discharge_points.create(
            timeseries=dis_df,
            locations=crossings_filt,
        )
        log.info(
            f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s), "
            f"{len(river_times)} time steps"
        )
sf.discharge_points.write()

# ── hand-craft this scenario's own sfincs.inp ─────────────────────────────────
# Forward the skeleton's own grid-header + non-forcing "*file" entries via a
# relative path (never re-written/duplicated into this scenario's own
# directory) -- see this script's own module docstring for why this is
# hand-written rather than done via sf.config.write().
exclude_keys = frozenset({"rstfile", "bzsfile", "bndfile", "disfile", "srcfile", "netsrcdisfile"}) | ini_exclude_keys
geometry_lines = forward_geometry_files(skeleton_cfg, skeleton_root, sfincs_root, exclude=exclude_keys)

# Grid-header + physics scalars, forwarded verbatim from the skeleton --
# everything the skeleton's own build set that this script doesn't
# explicitly own (timing/output-storage/zsini, set below).
OWNED_SCALAR_KEYS = frozenset({"tref", "tstart", "tstop", "dtmapout", "dtmaxout", "dthisout",
                                "storevelmax", "storetwet", "baro", "zsini"})
scalar_lines = [
    f"{k:<20} = {v}" for k, v in skeleton_cfg.items()
    if not k.endswith("file") and k not in OWNED_SCALAR_KEYS
]

lines = list(scalar_lines) + list(geometry_lines)

# rstfile: points at run_spinup's own (basin-level, shared, RP=1 river + calm sea) restart
# file -- a SIBLING of this scenario's own sfincs_root (both live under
# results/{basin_id}/, sfincs_root under runs/{scenario}/sfincs/, spin_up
# directly under spin_up/), so the relative path depth depends on the
# actual nesting -- computed, not hand-derived.
if include_rstart:
    tstart_event = tref + timedelta(days=spinup_days)
    rst_rel = os.path.relpath(spin_up_root / rst_fname, start=sfincs_root)
    lines.append(f"{'rstfile':<20} = {rst_rel}")
    log.info(f"tstart updated to {tstart_event} ({spinup_days} days after tref); rstfile = {rst_rel}")
else:
    tstart_event = tref

lines += [
    f"{'tref':<20} = {tref.strftime('%Y%m%d %H%M%S')}",
    f"{'tstart':<20} = {tstart_event.strftime('%Y%m%d %H%M%S')}",
    f"{'tstop':<20} = {tstop.strftime('%Y%m%d %H%M%S')}",
    f"{'dtmapout':<20} = {dtmapout}",
    f"{'dtmaxout':<20} = {dtmaxout}",
    f"{'dthisout':<20} = {dthisout}",
    f"{'storevelmax':<20} = {storevelmax}",
    f"{'storetwet':<20} = {storetwet}",
    f"{'baro':<20} = 0",           # no wind/atmosphere data
    # HydroMT hardcodes zsini=0.0 in sfincs.inp after create(). Override to
    # -9999 so any cell not covered by sfincs.ini falls back to bed level
    # (dry), consistent with how sfincs.ini itself handles land cells --
    # and, for forcing_mode="river_only" (no inifile forwarded above), this
    # IS the initial condition for every cell, sea included.
    f"{'zsini':<20} = -9999.0",
]

# Forcing files this script itself just wrote (bzs/bnd/dis/src) -- bare
# filenames, no relative-path adjustment needed, since they live directly
# in sfincs_root (the current, correct root) and sf.config already
# registered them there via water_level.write()/discharge_points.write().
for _key in ("bndfile", "bzsfile", "srcfile", "disfile", "netsrcdisfile"):
    _val = sf.config.get(_key)
    if _val:
        lines.append(f"{_key:<20} = {_val}")

with open(sfincs_root / "sfincs.inp", "w") as fh:
    fh.write("\n".join(lines) + "\n")
log.info(f"sfincs.inp written: {sfincs_root / 'sfincs.inp'}")

# ── diagnostic plot: forcing timeseries (the only build_sfincs plot that's
# actually forcing_mode/design_rp-dependent -- everything else moved to
# 13_build_sfincs_skeleton.py). Rebuilds the surge/discharge matrices at
# THIS scenario's own design_rp_surge_yr/design_rp_river_yr, exactly like
# the water-level/discharge forcing sections above -- fixed 2026-08-05:
# used to plot surge_forcing.nc's own stored 'water_level' directly, which
# is a basin-level, scenario-independent preview at a fixed diagnostic RP,
# never what this scenario's own sfincs.bzs actually contains. ──────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as _mdates

plt.ioff()
figs_dir = Path(snakemake.output.sfincs_inp).parent.parent / "visuals" / "sfincs_build"
figs_dir.mkdir(parents=True, exist_ok=True)


def _hours_to_dt(hours_arr):
    return [tref + timedelta(hours=float(h)) for h in hours_arr]


with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
    _surge_times = _hours_to_dt(_sds.time.values)
    # Rebuild the ACTUAL forcing this scenario just built, not surge_forcing.nc's
    # own stored 'water_level' -- that field is a basin-level, scenario-
    # independent preview at a fixed diagnostic RP (see 07_boundary_forcings.smk's
    # own comment), never what's actually in this scenario's own sfincs.bzs.
    # Mirrors the water-level boundary section above exactly, RP-for-RP and
    # mode-for-mode, so this plot always shows what was really built.
    if forcing_mode == "river_only":
        _wl = np.full((_sds.sizes["station"], len(_surge_times)), river_only_flat_level_m)
    else:
        _wl = build_design_surge_matrix(_sds, design_rp_surge_yr, slr_m=effective_slr_m)
    _n_stn = _wl.shape[0]

with xr.open_dataset(river_forcing_path, decode_times=False) as _rds:
    _river_times  = _hours_to_dt(_rds.time.values)
    _active_mask  = _rds.has_glofas.values.astype(bool)
    _n_cross      = int(_active_mask.sum())
    _dis_active   = (
        build_design_discharge_matrix(
            _rds, _active_mask, design_rp_river_yr,
            apply_protection_floor=(depth_method == "empirical"),
            discharge_multiplier=discharge_multiplier,
        )
        if _n_cross > 0 else np.zeros((0, len(_river_times)))
    )

_tstart_event = tstart_event if include_rstart else None

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for _i in range(_n_stn):
    ax5a.plot(_surge_times, _wl[_i, :], linewidth=0.8, alpha=0.7)
ax5a.set_ylabel("Water level (m+ref)")
if forcing_mode == "river_only":
    _surge_rp_label = "flat (river_only)"
elif design_rp_surge_yr:
    _surge_rp_label = f"RP{design_rp_surge_yr:g}"
else:
    _surge_rp_label = "flat (no surge)"
ax5a.set_title(f"Surge boundary — water level, {_surge_rp_label} (all stations)")
ax5a.grid(True, alpha=0.3)

for _i in range(_n_cross):
    ax5b.plot(_river_times, _dis_active[_i, :], linewidth=0.8, alpha=0.7)
ax5b.set_ylabel("Discharge (m³/s)")
ax5b.set_title(f"River discharge ({_n_cross} active GloFAS crossing(s))")
ax5b.grid(True, alpha=0.3)

if _tstart_event is not None:
    for _ax in (ax5a, ax5b):
        _ax.axvline(_tstart_event, color="black", linewidth=1.2,
                    linestyle="--", label=f"event start (after {spinup_days}d spinup)")
        _ax.legend(fontsize=8, loc="upper right")

_locator   = _mdates.AutoDateLocator()
_formatter = _mdates.ConciseDateFormatter(_locator)
ax5b.xaxis.set_major_locator(_locator)
ax5b.xaxis.set_major_formatter(_formatter)

fig5.suptitle(
    f"SFINCS forcing — basin {sfincs_root.parent.parent.name}, scenario {sfincs_root.parent.name} "
    f"(full timeseries; dashed = event model tstart)",
    fontsize=10,
)
fig5.tight_layout()
fig5.savefig(figs_dir / "05_forcing.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
log.info("Plot written: 05_forcing.png")
