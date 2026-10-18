#!/usr/bin/env python3
"""Build COAST-RP station water levels per return period and SLR scenario.

Entry point: run(config), via `python run_preparation.py boundary_conditions`.
total_wl = storm_tide(RP) + MDT + SLR_fingerprint(target_slr), in metres.
Antarctic stations (lat < coastrp_min_lat) are dropped. SLR fingerprints are
scaled from IPCC AR6 regional projections (DOI: 10.5281/zenodo.5914710).

MDT (AVISO mdt_cnes_cls22) re-references COAST-RP from local MSL to the
GOCO06s geoid, matching the DEM's EGM2008 -> GOCO06s correction. It is ADDED:
H_MSL = H_GOCO06s - MDT (data_catalog_gfm.yml), so H_GOCO06s = H_MSL + MDT.
Subtraction is only correct for the opposite (geoid -> MSL, e.g. GEBCO) direction.

Ported from notebooks by Natalia Aleksandrova (Deltares).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, merged_waterlevel_names, retry_transient_io  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _slr_names_to_metres(names: list[str]) -> list[float]:
    """Convert names like "SLR_500" (mm) to metres (0.5)."""
    return [int(name.split("_")[1]) / 1000.0 for name in names]


def _load_slr_dataset(
    slr_dir: Path, scenario: str, confidence: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load the IPCC AR6 regional SLR dataset; return (gauge, grid) subsets.

    Location index < 1e9 is a tide gauge, >= 1e9 a grid cell.
    """
    path = slr_dir / scenario / f"total_{scenario}_{confidence}_confidence_values.nc"
    ds = retry_transient_io(xr.open_dataset, path)
    gauge = ds.sel(locations=ds.locations[ds.locations < 1e9])
    grid  = ds.sel(locations=ds.locations[ds.locations >= 1e9])
    return gauge, grid


def preprocess_coastrp(raw_path: Path, out_path: Path, min_lat: float) -> xr.Dataset:
    """Drop stations with lat <= min_lat (unreliable Antarctic ice-shelf output) and cache."""
    if out_path.exists():
        logger.info("COAST-RP_preprocessed.nc already exists — loading.")
        return retry_transient_io(xr.open_dataset, out_path)

    logger.info("Step 1: Preprocessing COAST-RP (removing Antarctic stations).")
    ds = retry_transient_io(xr.open_dataset, raw_path)
    n_raw = int(ds.dims["stations"])
    ds = ds.where(ds.station_y_coordinate > min_lat, drop=True)
    logger.info("  %d → %d stations (dropped %d Antarctic points) → %s",
                n_raw, int(ds.dims["stations"]), n_raw - int(ds.dims["stations"]), out_path)
    retry_transient_io(ds.to_netcdf, out_path)
    return retry_transient_io(xr.open_dataset, out_path)


def _load_mdt(mdt_path: Path, mdt_variable: str = "mdt") -> xr.DataArray:
    """Load the AVISO MDT as a 2-D lat/lon DataArray, ascending, lon in -180..180.

    Ported from GCFM_UU/workflow/src/surge.load_mdt.
    """
    with retry_transient_io(xr.open_dataset, mdt_path) as ds:
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
    """Return the grid value nearest (lon, lat), else nearest non-NaN within +/-fallback_deg.

    Requires ascending lat/lon (see _load_mdt). Ported from the
    01_retrieve_MDT_correction notebook's find_nearest_valid
    (https://github.com/pydata/xarray/issues/644).
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
    """Look up the AVISO MDT [m] nearest each COAST-RP station and cache it.

    NaN only for stations with no valid MDT cell within +/-fallback_deg
    (enclosed seas/bays/fjords; known gap from the source notebook).
    """
    if out_path.exists():
        logger.info("MDT-mapped station file already exists — loading.")
        return retry_transient_io(xr.open_dataset, out_path)

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
    retry_transient_io(ds_mdt.to_netcdf, out_path)
    logger.info("  MDT-mapped station file saved → %s", out_path)
    return ds_mdt


def compute_slr_fingerprints(
    ds_coastrp: xr.Dataset,
    slr_dir: Path,
    scenario: str,
    confidence: str,
    target_slr_m: list[float],
    slr_base_path: Path,
    fingerprints_path: Path,
) -> xr.Dataset:
    """Scale IPCC AR6 SLR fingerprints to each target global-mean SLR level [m].

    base(station) = nearest sea_level_change (mm -> m; year 2100, median) from the
    combined gauge + gridded AR6 data; fingerprint = base / global_mean_2100 * T.
    Stations with no valid SLR within +/-1 deg get the uniform global mean, so
    the output is NaN-free.
    """
    #NOTE: check if the ration between station SLR and global mean changes significantly if using different timehorizons or SSP secnarios?
    if fingerprints_path.exists():
        cached = retry_transient_io(xr.open_dataset, fingerprints_path)
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

    from shapely.geometry import Point

    ds_gauge, ds_grid = _load_slr_dataset(slr_dir, scenario, confidence)
    ds_gauge = ds_gauge.sel(years=2100, quantiles=0.5, drop=True)
    ds_grid  = ds_grid.sel(years=2100, quantiles=0.5, drop=True)
    ds_grid  = ds_grid.where(ds_grid.sea_level_change.notnull(), drop=True)
    ds_combined = ds_gauge.merge(ds_grid)

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

        local = ds_combined.where(
            (ds_combined.lon > p_lon - 1) & (ds_combined.lon < p_lon + 1)
            & (ds_combined.lat > p_lat - 1) & (ds_combined.lat < p_lat + 1),
            drop=True,
        )
        local = local.where(~np.isnan(local.sea_level_change), drop=True)

        if len(local.locations) == 0:
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

    ds_base = xr.Dataset(
        {"SLR_ssp245_2100": (["stations"], np.array(slr_base, dtype=np.float64))},
        coords={
            "station_x_coordinate": ds_coastrp.station_x_coordinate,
            "station_y_coordinate": ds_coastrp.station_y_coordinate,
        },
    )
    retry_transient_io(ds_base.to_netcdf, slr_base_path)
    logger.info("  Base SLR per station saved → %s", slr_base_path)

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
    retry_transient_io(ds_fp.to_netcdf, fingerprints_path)
    logger.info("  SLR fingerprints saved → %s", fingerprints_path)
    return ds_fp


def combine_scenarios(
    ds_coastrp: xr.Dataset,
    ds_slr: xr.Dataset,
    out_dir: Path,
    return_periods: list[int],
    target_slr_m: list[float],
    nc_filename_template: str,
    nc_variable_template: str,
    mdt_arr: np.ndarray,
) -> None:
    """Write total_wl = storm_tide(RP) + MDT + SLR_fingerprint per (RP, SLR) [m].

    File and variable names come from boundary_conditions.nc_*_template, shared
    with extract_boundaries.py. MDT NaN -> 0 (no correction); output is NaN-free.
    Existing files are skipped.
    """
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    mdt_arr = np.nan_to_num(mdt_arr.astype(np.float64), nan=0.0)

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
            total_wl = (storm_tide + mdt_arr) + slr_arr

            ds_scen = xr.Dataset(
                {variable: (["stations"], total_wl)},
                coords=coord_template,
            )
            retry_transient_io(ds_scen.to_netcdf, out_path)
            n_written += 1
            logger.info("  Written %s", filename)

    logger.info("Step 4 complete: %d new scenario files written to %s", n_written, out_dir)


def run(config: dict, force: bool = True) -> None:
    """Write per-(return_period, SLR) water-level NetCDFs plus cached intermediates.

    force=True (default) deletes and recomputes all intermediates and outputs;
    the cache-hit path once kept a wrong MDT sign in on-disk output. Pass False
    only to resume an interrupted run.
    """
    wl_cfg = config["boundary_conditions"]
    adapt_cfg = config.get("adaptation", {})
    # Union of base SLR and adaptation scenarios; must match WATERLEVEL_NAMES in Snakefile.
    all_slr_names = merged_waterlevel_names(wl_cfg, adapt_cfg)
    target_slr_m   = _slr_names_to_metres(all_slr_names)
    return_periods  = wl_cfg["return_periods"]
    coastrp_min_lat = wl_cfg["coastrp_min_lat"]

    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    coastrp_raw  = Path(catalog.get_source("coast_rp").path)

    slr_projection = wl_cfg["slr_projection"]
    confidence   = wl_cfg.get("confidence_level", "medium")
    # Catalog path is the confidence_output_files root.
    slr_dir      = Path(catalog.get_source("ipcc_ar6_slr_projections").path) / f"{confidence}_confidence"

    proc_dir     = Path(config["paths"]["processed_inputs_dir"])
    out_dir      = Path(wl_cfg["waterlevel_nc_dir"])

    retry_transient_io(proc_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    def _maybe_delete(path: Path) -> None:
        if force and path.exists():
            path.unlink()
            logger.info("  Deleted %s (force=True)", path.name)

    preprocessed_path = proc_dir / "COAST-RP_preprocessed.nc"
    _maybe_delete(preprocessed_path)
    ds_coastrp = preprocess_coastrp(coastrp_raw, preprocessed_path, coastrp_min_lat)

    mdt_cfg = wl_cfg["mdt_correction"]
    mdt_path = Path(catalog.get_source("mdt_cnes_cls22").path)
    mdt_mapped_path = proc_dir / "MDT_mapped_on_coastal_points.nc"
    _maybe_delete(mdt_mapped_path)
    ds_mdt = compute_mdt_correction(
        ds_coastrp, mdt_path, mdt_cfg.get("mdt_variable", "mdt"),
        float(mdt_cfg.get("fallback_search_deg", 3.0)), mdt_mapped_path,
    )
    mdt_arr = ds_mdt["MDT"].values

    # Scenario/confidence in the cache names, so switching either is a cache miss.
    slr_base_path     = proc_dir / f"SLR_base_{slr_projection}_{confidence}_2100.nc"
    fingerprints_path = proc_dir / f"SLR_fingerprints_{slr_projection}_{confidence}_all.nc"
    _maybe_delete(slr_base_path)
    _maybe_delete(fingerprints_path)
    ds_slr = compute_slr_fingerprints(
        ds_coastrp, slr_dir, slr_projection, confidence,
        target_slr_m, slr_base_path, fingerprints_path,
    )

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
        "prepare_boundary_conditions.py is not a standalone entry point.\n"
        "Run it via: python run_preparation.py boundary_conditions\n"
        "See run_preparation.py --help for the full list of steps."
    )
