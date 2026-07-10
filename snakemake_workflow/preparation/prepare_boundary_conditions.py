#!/usr/bin/env python3
"""Preprocess COAST-RP boundary forcing with SLR scenarios.

Consolidates Boundary_conditions_waterlevels/01-03 notebooks into a single
runnable script.  Processing steps:

  1. Preprocess COAST-RP: drop Antarctic stations (station_y_coordinate <
     boundary_conditions.coastrp_min_lat in config.yml, -60° by default).
  2. Optionally (boundary_conditions.mdt_correction.enabled): look up the
     AVISO MDT (mdt_cnes_cls22) at each station (nearest
     valid grid cell, falling back to a +/-fallback_search_deg window search
     for stations whose nearest cell is NaN) and record it for step 4 to
     subtract. Disabled by default - see the config comment for when to
     enable it.
  3. Compute spatially-varying SLR fingerprints from IPCC AR6 regional
     sea level projections (SSP2-4.5, year 2100, median quantile,
     DOI: 10.5281/zenodo.5914710).  Each target global-mean SLR level
     is scaled from the base fingerprint map.  Stations with no valid SLR
     data within ±1° fall back to the target global-mean value (i.e. spatially
     uniform SLR) so no NaN propagates.
  4. Combine: total_wl = storm_tide(RP) [- MDT] + SLR_fingerprint(target_slr)
     One NetCDF is written per (return_period, SLR_target) combination,
     named/keyed from boundary_conditions.nc_filename_template/
     nc_variable_template in config.yml (the same templates
     extract_boundaries.py reads with).

MDT (mean dynamic topography) correction is OFF by default: COAST-RP is
referenced to local mean sea level (MSL), and so is the DeltaDTM v1.1 DEM
currently in use (GOCO06s geoid + MDT already subtracted upstream, per
Seeger & Minderhoud 2025 - see the `deltadtm` catalog entry) - the two
already share a vertical reference, so no additional shift is needed. Enable
boundary_conditions.mdt_correction.enabled ONLY once the DEM has also
switched to the pure-geoid GOCO06s reference (see vertical_datum_correction
in config.yml / preparation/merge_tiles.py) - enabling only one side would
leave the forcing and the terrain on different vertical references. When
enabled, MDT is SUBTRACTED (storm_tide - MDT), matching the sign convention
used to re-reference GEBCO/COAST-RP to GOCO06s elsewhere in this project
(GCFM_UU/workflow/src/surge.apply_mdt_correction). See
snakemake_workflow/config/data_catalog_gfm.yml (`coast_rp`, `deltadtm`,
`mdt_cnes_cls22`) for the full reasoning.

Intermediate files (COAST-RP_preprocessed.nc, MDT_mapped_on_coastal_points.nc,
SLR_base_ssp245_2100.nc, SLR_fingerprints_all.nc) are written to
boundary_conditions.processed_inputs_dir and reused on subsequent runs
unless force=True is passed to `run()`.

Not a standalone entry point - exposes `run(config, force=False)`, called
from run_preparation.py (`python run_preparation.py boundary_conditions`).

Original notebooks authored by Natalia Aleksandrova (n-aleksandrova), Deltares.
Translated to a standalone script by Julius Schlumberger, 2026-06-17.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, merged_slr_scenarios  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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

def preprocess_coastrp(raw_path: Path, out_path: Path, min_lat: float) -> xr.Dataset:
    """Drop Antarctic stations (lat < min_lat) from COAST-RP and save.

    The raw COAST-RP.nc file includes stations along the Antarctic ice
    shelf where the underlying storm-tide model outputs are unreliable.
    `min_lat` is boundary_conditions.coastrp_min_lat in config.yml — see
    that config entry for the reasoning behind the threshold value.
    """
    if out_path.exists():
        logger.info("COAST-RP_preprocessed.nc already exists — loading.")
        return xr.open_dataset(out_path)

    logger.info("Step 1: Preprocessing COAST-RP (removing Antarctic stations).")
    ds = xr.open_dataset(raw_path)
    n_raw = int(ds.dims["stations"])
    ds = ds.where(ds.station_y_coordinate > min_lat, drop=True)
    logger.info("  %d → %d stations (dropped %d Antarctic points) → %s",
                n_raw, int(ds.dims["stations"]), n_raw - int(ds.dims["stations"]), out_path)
    ds.to_netcdf(out_path)
    return xr.open_dataset(out_path)


# ---------------------------------------------------------------------------
# Step 2: MDT correction (local MSL -> GOCO06s geoid), optional
# ---------------------------------------------------------------------------

def _load_mdt(mdt_path: Path, mdt_variable: str = "mdt") -> xr.DataArray:
    """Load the AVISO MDT as a 2-D lat/lon DataArray, ascending coordinates.

    Drops any extra dimensions (e.g. 'time') by selecting their first index,
    and remaps longitudes from 0..360 to -180..180 if needed. Ported from
    GCFM_UU/workflow/src/surge.load_mdt.
    """
    with xr.open_dataset(mdt_path) as ds:
        da = ds[mdt_variable].load()

    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())
    for extra in [d for d in da.dims if d not in (lat_dim, lon_dim)]:
        da = da.isel({extra: 0})

    if float(da[lon_dim].max()) > 180:
        da = da.assign_coords(
            {lon_dim: xr.where(da[lon_dim] > 180, da[lon_dim] - 360, da[lon_dim])}
        )
    return da.sortby([lat_dim, lon_dim])


def _nearest_valid_grid(
    da: xr.DataArray, lon_dim: str, lon: float, lat_dim: str, lat: float, fallback_deg: float,
) -> float:
    """Value of a 2-D lat/lon grid nearest (lon, lat), NaN-safe.

    Falls back to the nearest non-NaN cell within +/-fallback_deg if the
    nearest cell itself is NaN. Ports the `find_nearest_valid` helper from
    Boundary_conditions_waterlevels/01_retrieve_MDT_correction.ipynb
    (credited there to https://github.com/pydata/xarray/issues/644) via
    GCFM_UU/workflow/src/surge._nearest_valid_grid. `da` must have ascending
    lat/lon coordinates (see `_load_mdt`).
    """
    val = float(da.sel({lon_dim: lon, lat_dim: lat}, method="nearest").values)
    if not np.isnan(val):
        return val

    window = da.sel({
        lon_dim: slice(lon - fallback_deg, lon + fallback_deg),
        lat_dim: slice(lat - fallback_deg, lat + fallback_deg),
    })
    if window.size == 0:
        return np.nan

    values = window.values
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan

    lons2d, lats2d = np.meshgrid(window[lon_dim].values, window[lat_dim].values)
    dist2 = (lons2d - lon) ** 2 + (lats2d - lat) ** 2
    dist2 = np.where(valid, dist2, np.inf)
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(values[idx])


def compute_mdt_correction(
    ds_coastrp: xr.Dataset, mdt_path: Path, mdt_variable: str, fallback_deg: float, out_path: Path,
) -> xr.Dataset:
    """Look up the AVISO MDT nearest each COAST-RP station.

    Returns (and caches to `out_path`) a Dataset with the same station
    coordinates as `ds_coastrp` and one data variable, `MDT` (m; NaN only
    for stations with no valid MDT cell within +/-fallback_deg of an
    enclosed sea/bay/fjord - see
    Boundary_conditions_waterlevels/01_retrieve_MDT_correction.ipynb's own
    TODO note on this same gap, left unchanged here).
    """
    if out_path.exists():
        logger.info("MDT-mapped station file already exists — loading.")
        return xr.open_dataset(out_path)

    n_stations = int(ds_coastrp.dims["stations"])
    logger.info("Step 2: Looking up MDT for %d stations…", n_stations)

    mdt_da = _load_mdt(mdt_path, mdt_variable)
    lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
    lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())

    mdt_values: list[float] = []
    for ii in range(n_stations):
        p_lon = float(ds_coastrp.station_x_coordinate.values[ii])
        p_lat = float(ds_coastrp.station_y_coordinate.values[ii])
        mdt_values.append(
            _nearest_valid_grid(mdt_da, lon_dim, p_lon, lat_dim, p_lat, fallback_deg)
        )
        if ii % 2000 == 0:
            logger.info("  MDT lookup: %d / %d stations processed", ii, n_stations)

    n_nan = int(np.isnan(mdt_values).sum())
    if n_nan:
        logger.warning(
            "%d/%d stations have no valid MDT within ±%.1f° — MDT correction "
            "will be skipped (i.e. 0) for these stations.",
            n_nan, n_stations, fallback_deg,
        )

    ds_mdt = xr.Dataset(
        {"MDT": (["stations"], np.array(mdt_values, dtype=np.float64))},
        coords={
            "station_x_coordinate": ds_coastrp.station_x_coordinate,
            "station_y_coordinate": ds_coastrp.station_y_coordinate,
        },
        attrs={"title": "MDT values at COAST-RP station locations (local MSL -> GOCO06s)"},
    )
    ds_mdt.to_netcdf(out_path)
    logger.info("  MDT-mapped station file saved → %s", out_path)
    return ds_mdt


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
    #NOTE: check if the ration between station SLR and global mean changes significantly if using different timehorizons or SSP secnarios?
    if fingerprints_path.exists():
        cached = xr.open_dataset(fingerprints_path)
        required_keys = {f"SLR_{int(t * 1000)}mm" for t in target_slr_m}
        if required_keys.issubset(set(cached.data_vars)):
            logger.info("SLR fingerprints already exist — loading.")
            return cached
        cached.close()
        logger.info(
            "SLR fingerprints cache is stale (missing keys for new scenarios) — regenerating."
        )

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
    ds_slr: xr.Dataset,
    out_dir: Path,
    return_periods: list[int],
    target_slr_m: list[float],
    nc_filename_template: str,
    nc_variable_template: str,
    mdt_arr: np.ndarray | None = None,
) -> None:
    """Write one scenario NetCDF per (RP, SLR) pair, named/keyed from
    `nc_filename_template`/`nc_variable_template` (boundary_conditions.* in
    config.yml) - the same templates extract_boundaries.py reads with, so
    changing either config value keeps both sides in sync.

    total_wl = storm_tide(RP) [- MDT] + SLR_fingerprint(target_slr)

    `mdt_arr` (aligned to ds_coastrp's station order, NaN treated as 0 - no
    correction for that station) is only non-None when
    boundary_conditions.mdt_correction.enabled is true; see module docstring
    for when that should be turned on. When None, storm_tide is used
    directly (no MDT correction applied).

    Because the SLR fallback ensures no NaN values enter the calculation,
    all output files are guaranteed to be NaN-free (MDT NaNs, if any, are
    replaced with 0 before use for the same reason).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if mdt_arr is not None:
        mdt_arr = np.nan_to_num(mdt_arr.astype(np.float64), nan=0.0)

    # Template with station coordinates only (no data variables)
    coord_template = {
        "station_x_coordinate": ds_coastrp.station_x_coordinate,
        "station_y_coordinate": ds_coastrp.station_y_coordinate,
    }

    n_written = 0
    for targ in target_slr_m:
        slr_key = f"SLR_{int(targ * 1000)}mm"
        slr_arr = ds_slr[slr_key].values.astype(np.float64)

        for rp in return_periods:
            return_period = f"RP{int(rp)}"
            waterlevel_name = f"SLR_{int(targ * 1000)}"
            filename = nc_filename_template.format(return_period=return_period, waterlevel_name=waterlevel_name)
            variable = nc_variable_template.format(return_period=return_period, waterlevel_name=waterlevel_name)
            out_path = out_dir / filename
            if out_path.exists():
                logger.debug("  %s already exists — skipping.", filename)
                continue

            rp_key = f"storm_tide_rp_{int(rp):04d}"
            storm_tide = ds_coastrp[rp_key].values.astype(np.float64)
            if mdt_arr is not None:
                total_wl = (storm_tide - mdt_arr) + slr_arr
            else:
                total_wl = storm_tide + slr_arr

            ds_scen = xr.Dataset(
                {variable: (["stations"], total_wl)},
                coords=coord_template,
            )
            ds_scen.to_netcdf(out_path)
            n_written += 1
            logger.info("  Written %s", filename)

    logger.info("Step 4 complete: %d new scenario files written to %s", n_written, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: dict, force: bool = False) -> None:
    """Run the full boundary-conditions pipeline (steps 1-4, see module docstring).

    `force`: delete and recompute all intermediate files and scenario
    outputs (otherwise cached files from a previous run are reused).
    """
    wl_cfg = config["boundary_conditions"]
    adapt_cfg = config.get("adaptation", {})
    # Union of base SLR scenarios and adaptation intensities — must match
    # WATERLEVEL_NAMES in Snakefile so every simulated scenario has a file.
    all_slr_names = merged_slr_scenarios(wl_cfg, adapt_cfg)
    target_slr_m   = _slr_names_to_metres(all_slr_names)
    return_periods  = wl_cfg["return_periods"]
    coastrp_min_lat = wl_cfg["coastrp_min_lat"]

    catalog = get_data_catalog(_REPO_ROOT / config["paths"]["hydromt_data_catalog"])
    coastrp_raw  = Path(catalog.get_source("coast_rp").path)

    slr_scenario = wl_cfg.get("slr_scenario", "ssp245")
    confidence   = wl_cfg.get("confidence_level", "medium")
    # Catalog stores the confidence_output_files root; append the selected
    # confidence sub-directory so _load_slr_dataset can append {scenario}/...
    slr_dir      = Path(catalog.get_source("ipcc_ar6_slr_projections").path) / f"{confidence}_confidence"

    proc_dir     = Path(wl_cfg["processed_inputs_dir"])
    out_dir      = Path(wl_cfg["waterlevel_nc_dir"])

    proc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_delete(path: Path) -> None:
        if force and path.exists():
            path.unlink()
            logger.info("  Deleted %s (force=True)", path.name)

    # ------------------------------------------------------------------
    # Step 1: Preprocess COAST-RP (remove Antarctica)
    preprocessed_path = proc_dir / "COAST-RP_preprocessed.nc"
    _maybe_delete(preprocessed_path)
    ds_coastrp = preprocess_coastrp(coastrp_raw, preprocessed_path, coastrp_min_lat)

    # ------------------------------------------------------------------
    # Step 2: MDT correction (optional - see module docstring)
    mdt_cfg = wl_cfg.get("mdt_correction", {})
    mdt_arr = None
    if mdt_cfg.get("enabled", False):
        mdt_path = Path(catalog.get_source("mdt_cnes_cls22").path)
        mdt_mapped_path = proc_dir / "MDT_mapped_on_coastal_points.nc"
        _maybe_delete(mdt_mapped_path)
        ds_mdt = compute_mdt_correction(
            ds_coastrp, mdt_path, mdt_cfg.get("mdt_variable", "mdt"),
            float(mdt_cfg.get("fallback_search_deg", 3.0)), mdt_mapped_path,
        )
        mdt_arr = ds_mdt["MDT"].values
    else:
        logger.info("Step 2: MDT correction disabled (boundary_conditions.mdt_correction.enabled=false).")

    # ------------------------------------------------------------------
    # Step 3: SLR fingerprints
    # Cache filenames bake in scenario/confidence so switching either in
    # config.yml naturally invalidates the cache (new filename -> cache
    # miss -> recompute) instead of silently reusing a stale scenario's
    # fingerprints unless --force is passed.
    slr_base_path     = proc_dir / f"SLR_base_{slr_scenario}_{confidence}_2100.nc"
    fingerprints_path = proc_dir / f"SLR_fingerprints_{slr_scenario}_{confidence}_all.nc"
    _maybe_delete(slr_base_path)
    _maybe_delete(fingerprints_path)
    ds_slr = compute_slr_fingerprints(
        ds_coastrp, slr_dir, slr_scenario, confidence,
        target_slr_m, slr_base_path, fingerprints_path,
    )

    # ------------------------------------------------------------------
    # Step 4: Combine into scenario files
    nc_filename_template = wl_cfg["nc_filename_template"]
    nc_variable_template = wl_cfg["nc_variable_template"]
    if force:
        glob_pattern = nc_filename_template.format(return_period="*", waterlevel_name="*")
        for f in out_dir.glob(glob_pattern):
            f.unlink()
            logger.info("  Deleted %s (force=True)", f.name)
    combine_scenarios(
        ds_coastrp, ds_slr, out_dir, return_periods, target_slr_m,
        nc_filename_template, nc_variable_template, mdt_arr=mdt_arr,
    )

    logger.info("All done. Run the Snakemake workflow next.")


if __name__ == "__main__":
    sys.exit(
        "prepare_boundary_conditions.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py boundary_conditions\n"
        "See run_preparation.py --help for the full list of steps."
    )
