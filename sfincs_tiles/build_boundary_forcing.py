"""Match a tile's existing COAST-RP boundary points to COAST-HG hydrograph
stations, apply the empirical MDT offset, and build a spatially-interpolated
zsini (initial water level) from the corrected hydrographs' own first
timestep - see sfincs_tiles' own plan doc for the full reasoning (the
`boundaries_RP100_SLR_0.gpkg` value already carries the real MDT correction;
COAST-HG's own hydrograph shape represents the same RP100 event without one,
so `offset = boundary_value_m - hydrograph_max_m` recovers that MDT
empirically per station, reusing the tile's own already-vetted
ocean-connectivity-filtered boundary points instead of re-selecting COAST-HG
stations from scratch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402

MAX_MATCH_DIST_DEG = 0.5  # generous - COAST-HG's 23,226 stations are dense
# along real coastline; a genuine match should be much closer than this in
# practice (see verification print below) - this is a sanity backstop, not
# a tuned search radius.

K_NEAREST_STATIONS = 20  # default cap on how many of a tile's own already-
# selected boundaries_{RP}_{SLR}.gpkg points get used for SFINCS forcing.
# That gpkg is the EIKONAL model's own ocean-connectivity-filtered boundary
# selection, built with a ~1 deg (~100 km) search radius for ITS purpose
# (cost-distance seeding) - real measured distances for this project's own
# first three SFINCS test tiles showed matched stations up to 65-125 km from
# the tile itself (median 25-93 km). Forcing a small local SFINCS domain via
# IDW blended across stations that far away is not physically appropriate
# (storm-tide phase/amplitude decorrelates well before 100 km) the way it is
# for the eikonal model's own very different use of that same search radius,
# so SFINCS forcing here uses only the k nearest of the already-selected
# points instead of all of them.


def select_k_nearest_boundary_points(
    boundaries_gdf: "gpd.GeoDataFrame", tile_gdf: "gpd.GeoDataFrame", k: int,
) -> tuple["gpd.GeoDataFrame", "pd.Series"]:
    """Keep only the k boundary points nearest to the tile's own geometry
    (real polygon, not just its bbox - distance is 0 for any point already
    inside the tile). Distance is computed in a local UTM CRS (estimated
    from the tile itself via geopandas' own estimate_utm_crs(), the same
    kind of per-tile UTM zone hydromt_sfincs picks for the model grid
    itself) rather than raw EPSG:4326 degrees, so ranking isn't distorted
    by latitude.
    """
    utm_crs = tile_gdf.estimate_utm_crs()
    tile_geom_utm = tile_gdf.to_crs(utm_crs).union_all()
    dist_km = boundaries_gdf.to_crs(utm_crs).geometry.distance(tile_geom_utm) / 1000.0
    keep_idx = dist_km.sort_values().index[:k]
    return boundaries_gdf.loc[keep_idx].reset_index(drop=True), dist_km.loc[keep_idx].reset_index(drop=True)


def match_boundary_points_to_coast_hg(
    boundaries_gdf: gpd.GeoDataFrame,
    value_column: str,
    coast_hg_nc_path: Path,
    hydrograph_variable: str = "hydrograph_average_tide_signal",
    max_match_dist_deg: float = MAX_MATCH_DIST_DEG,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str], dict[int, float]]:
    """For each boundary point, find the nearest COAST-HG station, compute
    the empirical MDT offset, and return the offset-corrected hydrograph
    timeseries for every matched point.

    Returns:
        corrected_df: (time x station) DataFrame, columns = boundary point
            index (0..n-1, matching `locations_gdf`'s own row order),
            values in METRES.
        times: the COAST-HG relative time axis as a DatetimeIndex (still
            needs a real `tref` applied by the caller - these are elapsed
            offsets, not real calendar dates, see coast_hg's own catalog
            known_caveats).
        dropped_reasons: human-readable list of any boundary points dropped
            (no COAST-HG station within max_match_dist_deg) - empty if none.
        offsets_m: {boundary point index: applied mdt_offset (m)} - the
            real applied offset, i.e. boundary_val - raw hg_max (before it
            was added into corrected_df). Recovering this from corrected_df
            alone is impossible: corrected_df's own max is, by
            construction, always == boundary_val, so returned/persisted
            explicitly rather than left for callers to re-derive.
    """
    with retry_transient_io(xr.open_dataset, coast_hg_nc_path) as ds:
        hg_lon = ds.station_x_coordinate.values
        hg_lat = ds.station_y_coordinate.values
        hg_values = ds[hydrograph_variable].values  # (station, time)
        hg_time = ds.time.values

    boundary_vals_m = boundaries_gdf[value_column].to_numpy(dtype=np.float64) / 100.0  # cm -> m
    boundary_lon = boundaries_gdf.geometry.x.to_numpy()
    boundary_lat = boundaries_gdf.geometry.y.to_numpy()

    corrected_cols: dict[int, np.ndarray] = {}
    offsets_m: dict[int, float] = {}
    dropped_reasons: list[str] = []
    match_info: list[str] = []

    for i in range(len(boundaries_gdf)):
        dist2 = (hg_lon - boundary_lon[i]) ** 2 + (hg_lat - boundary_lat[i]) ** 2
        j = int(np.argmin(dist2))
        dist_deg = float(np.sqrt(dist2[j]))
        if dist_deg > max_match_dist_deg:
            dropped_reasons.append(
                f"boundary point {i} (lon={boundary_lon[i]:.3f}, lat={boundary_lat[i]:.3f}): "
                f"nearest COAST-HG station is {dist_deg:.3f} deg away (> {max_match_dist_deg}) - dropped"
            )
            continue

        hg_series = hg_values[j, :].astype(np.float64)
        hg_max = float(np.nanmax(hg_series))
        mdt_offset = boundary_vals_m[i] - hg_max
        corrected_cols[i] = hg_series + mdt_offset
        offsets_m[i] = mdt_offset
        match_info.append(
            f"boundary point {i} -> COAST-HG station {j} ({dist_deg:.4f} deg away): "
            f"boundary={boundary_vals_m[i]:.4f} m, hg_max={hg_max:.4f} m, mdt_offset={mdt_offset:+.4f} m"
        )

    for line in match_info:
        print("  " + line)
    for line in dropped_reasons:
        print("  DROPPED: " + line)

    if not corrected_cols:
        raise ValueError(
            "No boundary point matched a COAST-HG station within the search radius - "
            "this tile cannot be forced with COAST-HG hydrographs (per plan: drop this tile)."
        )

    times = pd.DatetimeIndex(hg_time)  # relative axis, real tref applied by caller
    df = pd.DataFrame(corrected_cols, index=times)
    return df, times, dropped_reasons, offsets_m


def idw_interpolate_to_grid(
    station_x: np.ndarray, station_y: np.ndarray, station_values: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray, k: int | None = None,
) -> np.ndarray:
    """Inverse-distance-squared interpolation of `station_values` (already
    in the SAME planar/metric CRS as station_x/y and grid_x/y - e.g. the
    SFINCS UTM grid) onto every (grid_x, grid_y) point.

    Simpler planar version of flood_model.py::_idw_seed_values/_idw_nearest_k
    (which needs haversine because the eikonal model's own grid is lon/lat) -
    the SFINCS grid is already metric, so plain Euclidean distance applies
    directly, no haversine round-trip needed. `k` defaults to using every
    station (fine for a handful of boundary points on one small tile).
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


if __name__ == "__main__":
    import argparse
    import sys

    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--waterlevel-name", default="SLR_0")
    parser.add_argument("--k-nearest", type=int, default=K_NEAREST_STATIONS)
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2", help="output root directory name under paths.root (default: validation_sfincs_v2)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gfm_config import read_root

    root = read_root(Path(args.config))
    # Read from THIS tile's own working copy, not model_outputs/ directly -
    # see build_elevation.py's own note on this same gap.
    tile_dir = root / args.base_dir_name / args.tile_id / "inputs"
    out_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries_gdf = retry_transient_io(
        gpd.read_file, tile_dir / f"boundaries_{args.return_period}_{args.waterlevel_name}.gpkg"
    )
    if boundaries_gdf.empty:
        raise ValueError(
            f"tile {args.tile_id}: boundaries_{args.return_period}_{args.waterlevel_name}.gpkg is empty "
            "(no COAST-RP station for this tile) - per plan, this tile cannot be forced at all, drop it."
        )

    tile_gdf = retry_transient_io(gpd.read_file, tile_dir / "tile_geometry.gpkg")
    n_before = len(boundaries_gdf)
    boundaries_gdf, dist_km = select_k_nearest_boundary_points(boundaries_gdf, tile_gdf, args.k_nearest)
    print(f"k-nearest station filter: kept {len(boundaries_gdf)} of {n_before} pre-selected boundary points "
          f"(k={args.k_nearest}), distance to tile {dist_km.min():.1f}-{dist_km.max():.1f} km")

    coast_hg_path = root / "inputs" / "COAST_HG" / "COAST-HG_RP100.nc"
    df, times, dropped, offsets_m = match_boundary_points_to_coast_hg(boundaries_gdf, args.waterlevel_name, coast_hg_path)

    matched_idx = list(df.columns)
    matched_points = boundaries_gdf.iloc[matched_idx].reset_index(drop=True)
    matched_points["boundary_idx"] = matched_idx
    matched_points["mdt_offset_m"] = [offsets_m[i] for i in matched_idx]
    matched_points_path = out_dir / "matched_boundary_points.gpkg"
    matched_points.to_file(matched_points_path, driver="GPKG")

    # Elapsed hours since the hydrograph's own start - real tref applied
    # later by build_sfincs_tile.py, not baked in here.
    elapsed_hr = (times - times[0]).total_seconds() / 3600.0
    df_out = df.copy()
    df_out.insert(0, "elapsed_hr", elapsed_hr)
    hydrograph_path = out_dir / "corrected_hydrographs.csv"
    df_out.to_csv(hydrograph_path, index=False)

    print(f"\n{len(matched_idx)}/{len(boundaries_gdf)} boundary point(s) matched and corrected.")
    print(f"Wrote {matched_points_path}")
    print(f"Wrote {hydrograph_path} ({len(df_out)} timesteps, {elapsed_hr[-1]:.1f} h span)")
