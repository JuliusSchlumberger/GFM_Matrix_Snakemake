#!/usr/bin/env python3
"""Preprocess COAST-RP boundary forcing with MDT correction and SLR scenarios.

Consolidates Boundary_conditions_waterlevels/01-03 notebooks into a single
runnable script.  Processing steps:

  1. Preprocess COAST-RP: drop Antarctic stations (station_y_coordinate < -60°).
  2. Map MDT (AVISO MDT_CNES-CLS22, DOI: 10.24400/527896/a01-2023.003) onto
     each coastal station.  Search strategy per station:
       a. Nearest-neighbour in the full global MDT grid.
       b. If NaN: nearest valid cell within a ±3° bounding box.
       c. If still NaN (enclosed seas / fjords with no nearby ocean MDT cell):
          MDT is set to 0 — the station keeps its raw COAST-RP storm-tide value
          unreferenced to the geoid rather than producing a NaN output.
  3. Compute spatially-varying SLR fingerprints from IPCC AR6 regional
     sea level projections (SSP2-4.5, year 2100, median quantile,
     DOI: 10.5281/zenodo.5914710).  Each target global-mean SLR level
     is scaled from the base fingerprint map.  Stations with no valid SLR
     data within ±1° fall back to the target global-mean value (i.e. spatially
     uniform SLR) so no NaN propagates.
  4. Combine: total_wl = storm_tide(RP) + MDT + SLR_fingerprint(target_slr)
     One NetCDF is written per (return_period, SLR_target) combination with
     the naming convention expected by the Snakemake workflow:
       COAST-RP_EWL_RP{rp}_SLR_{target_slr_mm}.nc

Intermediate files (COAST-RP_preprocessed.nc, MDT_mapped_on_coastal_GTSM_points.nc,
SLR_base_ssp245_2100.nc, SLR_fingerprints_all.nc) are written to
paths.processed_inputs and reused on subsequent runs unless --force is given.

Usage:
    python prepare_boundary_conditions.py [--config PATH] [--force]

    --config  Path to config.yml (default: ../config.yml relative to this file)
    --force   Recompute and overwrite all intermediate and output files

Original notebooks authored by Natalia Aleksandrova (n-aleksandrova), Deltares.
Translated to a standalone script by Julius Schlumberger, 2026-06-17.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import get_preprocessing_catalog  # noqa: E402

logger = logging.getLogger(__name__)

# Root config (choices + output paths)
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yml"
# Workflow config (scenario names + return periods — the single source of truth)
_WORKFLOW_CONFIG = Path(__file__).resolve().parent / "config" / "config.yml"


def _slr_names_to_metres(names: list[str]) -> list[float]:
    """Parse waterlevels.names entries into target SLR levels in metres.

    Converts ``["SLR_0", "SLR_500", "SLR_1000", ...]`` →
    ``[0.0, 0.5, 1.0, ...]`` by stripping the ``SLR_`` prefix and dividing
    the remaining millimetre value by 1000.
    """
    return [int(name.split("_")[1]) / 1000.0 for name in names]


# ---------------------------------------------------------------------------
# SLR data loader (adapted from retrieve_boundary_conditions.py)
# ---------------------------------------------------------------------------

def _load_slr_dataset(
    slr_dir: Path, scenario: str, confidence: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load IPCC AR6 regional SLR dataset; return (gauge, grid) subsets.

    Locations with index < 1e9 are tide-gauge locations; >= 1e9 are gridded.
    """
    path = slr_dir / scenario / f"total_{scenario}_{confidence}_confidence_values.nc"
    ds = xr.open_dataset(path)
    gauge = ds.sel(locations=ds.locations[ds.locations < 1e9])
    grid  = ds.sel(locations=ds.locations[ds.locations >= 1e9])
    return gauge, grid


# ---------------------------------------------------------------------------
# Step 1: Preprocess COAST-RP
# ---------------------------------------------------------------------------

def preprocess_coastrp(raw_path: Path, out_path: Path) -> xr.Dataset:
    """Drop Antarctic stations (lat < -60°) from COAST-RP and save.

    The raw COAST-RP.nc file includes stations along the Antarctic ice
    shelf where MDT and model outputs are unreliable.
    """
    if out_path.exists():
        logger.info("COAST-RP_preprocessed.nc already exists — loading.")
        return xr.open_dataset(out_path)

    logger.info("Step 1: Preprocessing COAST-RP (removing Antarctic stations).")
    ds = xr.open_dataset(raw_path)
    n_raw = int(ds.dims["stations"])
    ds = ds.where(ds.station_y_coordinate > -60, drop=True)
    logger.info("  %d → %d stations (dropped %d Antarctic points) → %s",
                n_raw, int(ds.dims["stations"]), n_raw - int(ds.dims["stations"]), out_path)
    ds.to_netcdf(out_path)
    return xr.open_dataset(out_path)


# ---------------------------------------------------------------------------
# Step 2: Map MDT onto coastal stations
# ---------------------------------------------------------------------------

def _nearest_valid_mdt(ds_sub: xr.Dataset, p_lon: float, p_lat: float) -> float:
    """Return the nearest non-NaN MDT value in a 2-D spatial subset.

    Adapted from find_nearest_valid() in notebook 01 (source:
    https://github.com/pydata/xarray/issues/644).  Returns NaN if the
    subset contains no valid cells.
    """
    mdt_2d = ds_sub["mdt"].squeeze()  # drop time dimension → (latitude, longitude)
    values = mdt_2d.values
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan
    lon_g, lat_g = np.meshgrid(ds_sub.longitude.values, ds_sub.latitude.values)
    dists = np.sqrt((lon_g[valid] - p_lon) ** 2 + (lat_g[valid] - p_lat) ** 2)
    return float(values[valid][np.argmin(dists)])


def map_mdt(ds_coastrp: xr.Dataset, mdt_path: Path, out_path: Path) -> xr.Dataset:
    """Map AVISO MDT_CNES-CLS22 values onto every COAST-RP coastal station.

    For each station the lookup proceeds in three stages:
      1. Nearest-neighbour in the full MDT grid.
      2. If NaN: nearest valid cell within a ±3° bounding box.
      3. If still NaN (e.g. enclosed seas / fjords with no MDT coverage):
         MDT is set to 0 — the station is kept NaN-free at the cost of
         omitting the geoid reference correction.
    """
    if out_path.exists():
        logger.info("MDT mapping already exists — loading.")
        return xr.open_dataset(out_path)

    n_stations = int(ds_coastrp.dims["stations"])
    logger.info("Step 2: Mapping MDT onto %d coastal stations…", n_stations)
    ds_mdt = xr.open_dataset(mdt_path)

    mdt_vals: list[float] = []
    n_zero_fallback = 0

    for ii in range(n_stations):
        p_lon = float(ds_coastrp.station_x_coordinate.values[ii])
        p_lat = float(ds_coastrp.station_y_coordinate.values[ii])

        # Stage 1: exact nearest-neighbour
        val = float(
            ds_mdt.sel(longitude=p_lon, latitude=p_lat, method="nearest")
            .mdt.values.item()
        )

        if np.isnan(val):
            # Stage 2: ±3° bounding-box search for nearest valid cell
            sub = ds_mdt.sel(
                longitude=slice(p_lon - 3, p_lon + 3),
                latitude=slice(p_lat - 3, p_lat + 3),
                drop=True,
            )
            val = _nearest_valid_mdt(sub, p_lon, p_lat)

        if np.isnan(val):
            # Stage 3: no valid MDT nearby — apply no correction (MDT = 0)
            val = 0.0
            n_zero_fallback += 1

        mdt_vals.append(val)
        if ii % 2000 == 0:
            logger.info("  MDT: %d / %d stations processed", ii, n_stations)

    ds_mdt.close()

    if n_zero_fallback:
        logger.warning(
            "%d/%d stations had no valid MDT within ±3° — MDT set to 0 "
            "(storm-tide value used without geoid correction).",
            n_zero_fallback, n_stations,
        )

    ds_out = xr.Dataset(
        {"MDT": (["stations"], np.array(mdt_vals, dtype=np.float64))},
        coords={
            "station_x_coordinate": ds_coastrp.station_x_coordinate,
            "station_y_coordinate": ds_coastrp.station_y_coordinate,
        },
        attrs={
            "title": (
                "AVISO MDT_CNES-CLS22 mapped to COAST-RP coastal stations. "
                "MDT=0 where no valid grid cell was found within ±3°."
            )
        },
    )
    ds_out.to_netcdf(out_path)
    logger.info("  MDT mapping saved → %s", out_path)
    return ds_out


# ---------------------------------------------------------------------------
# Step 3: Compute SLR fingerprints
# ---------------------------------------------------------------------------

def compute_slr_fingerprints(
    ds_coastrp: xr.Dataset,
    slr_dir: Path,
    scenario: str,
    confidence: str,
    target_slr_m: list[float],
    slr_base_path: Path,
    fingerprints_path: Path,
) -> xr.Dataset:
    """Scale IPCC AR6 SLR fingerprints to each target global-mean SLR level.

    The base fingerprint per station is the nearest sea_level_change value
    (mm) from the combined gauge + gridded IPCC AR6 dataset (year 2100,
    median quantile, SSP2-4.5 by default), converted to metres.  The per-
    station fingerprint for a target global SLR T is:

        fingerprint(station) = base(station) / global_mean_2100 * T

    Fallback for stations with no valid SLR data within ±1°: the global-mean
    SLR is used (i.e. no spatial fingerprint, uniform offset = T), so the
    output is always NaN-free.
    """
    if fingerprints_path.exists():
        logger.info("SLR fingerprints already exist — loading.")
        return xr.open_dataset(fingerprints_path)

    n_stations = int(ds_coastrp.dims["stations"])
    logger.info("Step 3: Computing SLR fingerprints for %d stations…", n_stations)

    from shapely.geometry import Point  # used for degree-space proximity

    ds_gauge, ds_grid = _load_slr_dataset(slr_dir, scenario, confidence)
    ds_gauge = ds_gauge.sel(years=2100, quantiles=0.5, drop=True)
    ds_grid  = ds_grid.sel(years=2100, quantiles=0.5, drop=True)
    ds_grid  = ds_grid.where(ds_grid.sea_level_change.notnull(), drop=True)
    ds_combined = ds_gauge.merge(ds_grid)

    # Global-mean SLR in 2100 — denominator for fingerprint scaling
    mean_ori_m = float(ds_grid.sea_level_change.mean()) / 1000.0  # mm → m
    logger.info(
        "  Global mean SLR 2100 (%s, %s confidence): %.3f m",
        scenario, confidence, mean_ori_m,
    )

    slr_base: list[float] = []
    n_uniform_fallback = 0

    for ii in range(n_stations):
        p_lon = float(ds_coastrp.station_x_coordinate.values[ii])
        p_lat = float(ds_coastrp.station_y_coordinate.values[ii])
        pt = Point(p_lon, p_lat)

        # Search within ±1° bounding box; filter out NaN SLR cells
        local = ds_combined.where(
            (ds_combined.lon > p_lon - 1) & (ds_combined.lon < p_lon + 1)
            & (ds_combined.lat > p_lat - 1) & (ds_combined.lat < p_lat + 1),
            drop=True,
        )
        local = local.where(~np.isnan(local.sea_level_change), drop=True)

        if len(local.locations) == 0:
            # No valid SLR data nearby — fall back to spatially uniform global mean
            slr_base.append(mean_ori_m)
            n_uniform_fallback += 1
        else:
            slr_pts = [
                Point(float(local.lon[i]), float(local.lat[i]))
                for i in range(len(local.lon))
            ]
            idx = int(np.argmin([pt.distance(sp) for sp in slr_pts]))
            slr_base.append(float(local.sea_level_change[idx].values) / 1000.0)

        if ii % 2000 == 0:
            logger.info("  SLR fingerprint: %d / %d stations processed", ii, n_stations)

    ds_combined.close()

    if n_uniform_fallback:
        logger.warning(
            "%d/%d stations had no valid SLR data within ±1° — using "
            "global-mean SLR as spatially uniform proxy for these stations.",
            n_uniform_fallback, n_stations,
        )

    # Save per-station base SLR (mm→m already done) for reference / reproducibility
    ds_base = xr.Dataset(
        {"SLR_ssp245_2100": (["stations"], np.array(slr_base, dtype=np.float64))},
        coords={
            "station_x_coordinate": ds_coastrp.station_x_coordinate,
            "station_y_coordinate": ds_coastrp.station_y_coordinate,
        },
    )
    ds_base.to_netcdf(slr_base_path)
    logger.info("  Base SLR per station saved → %s", slr_base_path)

    # Scale to each target global-mean SLR
    slr_arr = np.array(slr_base, dtype=np.float64)
    fp_vars: dict[str, tuple] = {}
    for targ in target_slr_m:
        key = f"SLR_{int(targ * 1000)}mm"
        if mean_ori_m > 0:
            scaled = slr_arr * targ / mean_ori_m
        else:
            scaled = np.zeros(n_stations, dtype=np.float64)
        fp_vars[key] = (["stations"], scaled)

    ds_fp = xr.Dataset(
        fp_vars,
        coords={
            "station_x_coordinate": ds_coastrp.station_x_coordinate,
            "station_y_coordinate": ds_coastrp.station_y_coordinate,
        },
        attrs=ds_base.attrs,
    )
    ds_fp.to_netcdf(fingerprints_path)
    logger.info("  SLR fingerprints saved → %s", fingerprints_path)
    return ds_fp


# ---------------------------------------------------------------------------
# Step 4: Combine into per-scenario water level files
# ---------------------------------------------------------------------------

def combine_scenarios(
    ds_coastrp: xr.Dataset,
    ds_mdt: xr.Dataset,
    ds_slr: xr.Dataset,
    out_dir: Path,
    return_periods: list[int],
    target_slr_m: list[float],
) -> None:
    """Write one COAST-RP_EWL_RP{rp}_SLR_{slr_mm}.nc per (RP, SLR) pair.

    total_wl = storm_tide(RP) + MDT + SLR_fingerprint(target_slr)

    Because MDT and SLR fallbacks ensure no NaN values enter the calculation,
    all output files are guaranteed to be NaN-free.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mdt_arr = ds_mdt["MDT"].values.astype(np.float64)

    # Template with station coordinates only (no data variables)
    coord_template = {
        "station_x_coordinate": ds_mdt.station_x_coordinate,
        "station_y_coordinate": ds_mdt.station_y_coordinate,
    }

    n_written = 0
    for targ in target_slr_m:
        slr_key = f"SLR_{int(targ * 1000)}mm"
        slr_arr = ds_slr[slr_key].values.astype(np.float64)

        for rp in return_periods:
            scen_name = f"COAST-RP_EWL_RP{int(rp)}_SLR_{int(targ * 1000)}"
            out_path = out_dir / f"{scen_name}.nc"
            if out_path.exists():
                logger.debug("  %s.nc already exists — skipping.", scen_name)
                continue

            rp_key = f"storm_tide_rp_{int(rp):04d}"
            storm_tide = ds_coastrp[rp_key].values.astype(np.float64)
            total_wl = storm_tide + mdt_arr + slr_arr

            ds_scen = xr.Dataset(
                {scen_name: (["stations"], total_wl)},
                coords=coord_template,
            )
            ds_scen.to_netcdf(out_path)
            n_written += 1
            logger.info("  Written %s.nc", scen_name)

    logger.info("Step 4 complete: %d new scenario files written to %s", n_written, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare COAST-RP boundary conditions with MDT correction "
            "and IPCC AR6 SLR fingerprint scenarios."
        )
    )
    ap.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"Path to config.yml (default: {_DEFAULT_CONFIG}).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete and recompute all intermediate files and scenario outputs.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    p = config["paths"]
    c = config.get("choices", {})

    # Input data paths come from the preprocessing data catalog
    # (snakemake_workflow/config/preprocessing_data.yml); output paths and
    # scenario choices stay in the root config.yml.
    catalog      = get_preprocessing_catalog()
    coastrp_raw  = Path(catalog.get_source("coast_rp").path)
    mdt_file     = Path(catalog.get_source("mdt_hybrid_cnes_cls22_cmems2020").path)

    slr_scenario = c.get("SLR_scenario", "ssp245")
    confidence   = c.get("confidence_level", "medium")
    # Catalog stores the confidence_output_files root; append the selected
    # confidence sub-directory so _load_slr_dataset can append {scenario}/...
    slr_dir      = Path(catalog.get_source("ipcc_ar6_slr_projections").path) / f"{confidence}_confidence"

    proc_dir     = Path(p["processed_inputs"])
    out_dir      = Path(p["WL_scenarios"])

    # Derive scenario lists from the workflow config so this script and the
    # Snakemake DAG always stay in sync — editing waterlevels.names or
    # waterlevels.return_periods in config/config.yml is the only change needed.
    with open(_WORKFLOW_CONFIG) as fh:
        workflow_cfg = yaml.safe_load(fh)
    wl_cfg = workflow_cfg["boundary_conditions"]
    target_slr_m   = _slr_names_to_metres(wl_cfg["slr_scenarios"])
    return_periods  = wl_cfg["return_periods"]

    proc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_delete(path: Path) -> None:
        if args.force and path.exists():
            path.unlink()
            logger.info("  Deleted %s (--force)", path.name)

    # ------------------------------------------------------------------
    # Step 1: Preprocess COAST-RP (remove Antarctica)
    preprocessed_path = proc_dir / "COAST-RP_preprocessed.nc"
    _maybe_delete(preprocessed_path)
    ds_coastrp = preprocess_coastrp(coastrp_raw, preprocessed_path)

    # ------------------------------------------------------------------
    # Step 2: Map MDT onto coastal stations
    mdt_mapped_path = proc_dir / "MDT_mapped_on_coastal_GTSM_points.nc"
    _maybe_delete(mdt_mapped_path)
    ds_mdt = map_mdt(ds_coastrp, mdt_file, mdt_mapped_path)

    # ------------------------------------------------------------------
    # Step 3: SLR fingerprints
    slr_base_path     = proc_dir / "SLR_base_ssp245_2100.nc"
    fingerprints_path = proc_dir / "SLR_fingerprints_all.nc"
    _maybe_delete(slr_base_path)
    _maybe_delete(fingerprints_path)
    ds_slr = compute_slr_fingerprints(
        ds_coastrp, slr_dir, slr_scenario, confidence,
        target_slr_m, slr_base_path, fingerprints_path,
    )

    # ------------------------------------------------------------------
    # Step 4: Combine into scenario files
    if args.force:
        for f in out_dir.glob("COAST-RP_EWL_*.nc"):
            f.unlink()
            logger.info("  Deleted %s (--force)", f.name)
    combine_scenarios(ds_coastrp, ds_mdt, ds_slr, out_dir, return_periods, target_slr_m)

    logger.info("All done. Run the Snakemake workflow next.")


if __name__ == "__main__":
    main()
