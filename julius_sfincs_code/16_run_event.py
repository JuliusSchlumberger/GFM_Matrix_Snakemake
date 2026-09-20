"""
16_run_event.py — Run the main flood-event SFINCS simulation and its
sanity-check diagnostics.

The main model's sfincs.inp (written by rule 13, build_sfincs) is already
configured for this exact run: tstart = spin-up end, tstop = end of the full
forcing timeseries, rstfile pointing at rule 14's own basin-level restart
file (results/{basin_id}/spin_up/, shared by every scenario of this basin
-- see 14_run_spinup.py's own module docstring for why spin-up is no
longer scenario-scoped). This script only executes it — cwd = sfincs_root
itself, since that's where sfincs.inp and its own forcing files
(sfincs.bzs/.dis/etc, written by rule 13) live; the geometry files it also
references (dep/msk/manning/subgrid/weir) resolve via relative path back
to the basin's own sfincs_skeleton/ (rule build_sfincs_skeleton).

After the run, produces the same kind of sanity-check diagnostics as rule 15
(sanity_checks) — inundation-ratio map, flood-progression animation — but
for this event run's own sfincs_map.nc, plus a per-timestep
flooded-area / flood-volume CSV.

Flood volume in the CSV is the TRUE peak of the total-volume time series
(sum of instantaneous depth x pixel area at each output timestep, from the
same zs time series used for the flood animation) — not the sum of each
pixel's own envelope maximum (zsmax), which can overstate the true peak
since different pixels may flood at different times. The inundation-ratio
map still uses the zsmax envelope, matching rule 15's convention, since it
answers a different question (the maximum extent ever reached by any pixel,
not the volume present at one instant).

Outputs
-------
sfincs_map.nc                    SFINCS map output for the event run.
event/01_inundation_ratio.png    Envelope (zsmax) flooded/dry map.
event/02_flood_animation.mp4     Instantaneous depth animation.
event/flood_timeseries.csv       Per-timestep flooded_area_km2 / flood_volume_m3.
"""

import gc
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.log import setup_logging
from src.plots import animate_flood_progression, plot_inundation_check
from src.postprocessing import (
    compute_flood_progression,
    compute_flood_timeseries_stats,
    compute_max_inundation,
)
from src.sfincs_run import run_sfincs_subprocess

log = setup_logging(snakemake.log[0])

# ── params ────────────────────────────────────────────────────────────────────
sfincs_root                = Path(snakemake.params.sfincs_root)
# Subgrid reference raster for postprocessing lives here, not physically in
# sfincs_root -- this scenario's own model only references it via a
# relative path in its own sfincs.inp (see 13_build_sfincs.py).
skeleton_root              = Path(snakemake.params.skeleton_root)
sfincs_exe                 = Path(snakemake.params.sfincs_exe)
timeout_s                  = int(snakemake.params.timeout_s)
min_inundation_depth_m     = float(snakemake.params.min_inundation_depth_m)
include_subgrid            = bool(snakemake.params.include_subgrid)
animation_fps               = int(snakemake.params.animation_fps)
land_polygons_path         = Path(snakemake.input.land_mask_on_grid)  # grid-aligned land mask, plot background
sea_mask_path               = Path(snakemake.input.sea_mask)
river_network_path          = Path(snakemake.input.clean_river_network)
domain_gpkg_path            = Path(snakemake.input.domain_gpkg)

event_dir = Path(snakemake.output.plot_inundation_ratio).parent
event_dir.mkdir(parents=True, exist_ok=True)

# Load domain polygon in WGS84 for overlay plots.
_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)
basin_id = sfincs_root.parent.name

# ── run SFINCS ────────────────────────────────────────────────────────────────
# The main sfincs.inp (written by rule 13) already covers exactly this run;
# cwd = sfincs_root so all its relative file references (dep, msk, bnd, ...,
# and rstfile = spinup/<restart file>) resolve correctly.
run_sfincs_subprocess(sfincs_exe, sfincs_root, timeout_s, log, label="SFINCS event run", n_threads=snakemake.threads)

sfincs_map_path = Path(snakemake.output.sfincs_map_nc)
if not sfincs_map_path.exists():
    raise FileNotFoundError(f"SFINCS ran but expected map output not found: {sfincs_map_path}")
log.info(f"Event map output written: {sfincs_map_path} ({sfincs_map_path.stat().st_size / 1e6:.1f} MB)")

# ── Check 1: inundation ratio (envelope, zsmax) ───────────────────────────────
# Mirrors 15_sanity_checks.py's Check 1, sourced from this event run's own
# output (run_dir = sfincs_root) instead of the spin-up's.
plot_ratio_path = Path(snakemake.output.plot_inundation_ratio)

da_hmax, da_dep = compute_max_inundation(
    sfincs_root, skeleton_root, sea_mask_path, hmin=min_inundation_depth_m, include_subgrid=include_subgrid,
)
if da_hmax is None or da_dep is None:
    log.warning("Could not compute max inundation depth (missing 'zsmax' or bed level) — skipping")
    plot_ratio_path.touch()
else:
    n_land    = int(da_dep.notnull().sum().item())
    n_flooded = int(da_hmax.notnull().sum().item())
    frac      = n_flooded / n_land if n_land > 0 else 0.0

    try:
        res = abs(da_dep.rio.resolution()[0] * da_dep.rio.resolution()[1])
    except Exception:
        res = np.nan
    flooded_km2 = n_flooded * res / 1e6
    land_km2    = n_land    * res / 1e6

    log.info(
        f"[Check 1] Inundation ratio  hmin={min_inundation_depth_m} m  "
        f"flooded={n_flooded:,}/{n_land:,} pixels ({frac:.2%})  "
        f"area={flooded_km2:.1f}/{land_km2:.1f} km²"
    )

    plot_inundation_check(
        da_hmax, min_inundation_depth_m, n_flooded, n_land,
        str(land_polygons_path), str(river_network_path),
        str(plot_ratio_path), basin_id=basin_id,
        run_label="event",
    )
    log.info(f"Inundation ratio plot written: {plot_ratio_path}")

# ── flood progression: animation + per-timestep area/volume CSV ──────────────
animation_out_path = Path(snakemake.output.animation_flood_progress)
csv_out_path        = Path(snakemake.output.flood_timeseries_csv)
animation_out_path.parent.mkdir(parents=True, exist_ok=True)

da_h = compute_flood_progression(sfincs_root, sea_mask_path)
if da_h is None:
    log.warning(
        "compute_flood_progression returned None (no 'zs' in sfincs_map.nc) — "
        "skipping flood animation and flood-timeseries CSV"
    )
    animation_out_path.touch()
    pd.DataFrame(columns=["time", "flooded_area_km2", "flood_volume_m3"]).to_csv(
        csv_out_path, index=False
    )
else:
    animate_flood_progression(
        da_h, domain_poly,
        str(land_polygons_path), str(river_network_path),
        str(animation_out_path),
        basin_id=basin_id,
        run_label="event",
        fps=animation_fps,
    )
    log.info(f"Flood animation written: {animation_out_path}")

    # da_h (all frames, mesh resolution) is only needed for the animation
    # above. compute_flood_timeseries_stats below rasterizes one frame at a
    # time onto a fine subgrid raster via hydromt_sfincs's celltree-based
    # rasterize_like, which needs substantial working memory on its own --
    # free da_h first so that call has the headroom it needs.
    del da_h
    gc.collect()

    # Per-timestep flooded area and total flood volume, computed
    # independently from da_h above (which stays at mesh/cell resolution,
    # unmasked, for animation only — see compute_flood_progression's
    # docstring). This uses the same subgrid-downscaled resolution as
    # compute_max_inundation instead, processing one frame at a time (see
    # compute_flood_timeseries_stats) — the TRUE volume present in the
    # domain at each real instant (unlike the zsmax envelope in Check 1
    # above, which sums each pixel's own peak regardless of when it
    # occurred), so the max of flood_volume_m3 here is the physically
    # correct peak total flood volume for the event.
    df = compute_flood_timeseries_stats(
        sfincs_root, skeleton_root, sea_mask_path, min_inundation_depth_m,
        include_subgrid=include_subgrid,
    )
    if df is None:
        log.warning(
            "compute_flood_timeseries_stats returned None (missing zs or "
            "bed level) — writing empty CSV"
        )
        df = pd.DataFrame(columns=["time", "flooded_area_km2", "flood_volume_m3"])
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out_path, index=False)

    if len(df):
        i_peak = int(df["flood_volume_m3"].idxmax())
        log.info(
            f"Flood timeseries written: {csv_out_path} ({len(df)} step(s)). "
            f"Peak total flood volume {df['flood_volume_m3'].iloc[i_peak]:.3e} m³ "
            f"(area {df['flooded_area_km2'].iloc[i_peak]:.2f} km²) at "
            f"t={df['time'].iloc[i_peak]}"
        )
    else:
        log.warning("Flood timeseries is empty — no output timesteps found")

log.info("Done")
