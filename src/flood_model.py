"""Coastal flood depth on one tile: seeding, level fields and the eikonal solve.

`flood_depth_dense` is the production entry point.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.transform
from affine import Affine
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from eikonal import solve_eikonal_dense
from event_volume import DurationCorrection, VolumeCap


LAND_CODE = 0


def effective_dem(dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set elevation (m) to 0.0 on non-land cells.

    Mask codes: 0=land, 1=ocean, 2=lake, 3=river. Needed because
    rasters.extract_dem only fills source nodata, so water cells can keep a
    real DeltaDTM elevation.
    """
    return np.where(mask == LAND_CODE, dem, dem.dtype.type(0.0))

EARTH_RADIUS_M = 6_371_000.0
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

# Round-based convergence threshold (m); well below the 0.1 m flood/no-flood
# depth threshold. Config: simulation.flooding.waterlevel_epsilon_m.
WATERLEVEL_EPSILON_M = 0.03

# Prohibitive even in float32 for any path crossing a blocked cell (m/m).
OBSTACLE_BLOCK_FRICTION = 9999.0

# Floor for zero-friction cells (water classes): 1 mm per 30 m, negligible next to any land rate.
MIN_FRICTION_M_PER_M = 0.001 / 30.0


class SolveTrace:
    """Observe flood_depth_dense's eikonal solves; every hook is a no-op here.

    `solve_pass` is "first", or "duration" for the solve with correction B's friction; `outer`
    is 0 without obstacle coupling, else the outer iteration (1-based). `depth_of` maps a
    water level grid to the solve's flood depth (coast-connected, 0 where dry).
    """

    def solve_started(self, solve_pass: str, outer: int, depth_of) -> None:
        pass

    def sweep(self, t: np.ndarray, change: float) -> None:
        pass

    def outer_done(self, outer: int, waterlevel: np.ndarray, blocked: np.ndarray, pct_newly_blocked: float,
                   inner: dict) -> None:
        pass

# Grid step (cells) of the coarse grid the static level is interpolated on before bilinear
# upsampling; the level (MDT + SLR fingerprint) varies over tens of km.
STATIC_LEVEL_COARSEN = 32

# Flood models of flood_depth_dense: a flat-in-time level against the DEM (bathtub), the same
# kept where connected to the sea (bathtub_connected), a uniform attenuation along straight lines
# from the seeds (straight_line), and the attenuation along the cheapest path (eikonal).
FLOOD_MODELS = ("bathtub", "bathtub_connected", "straight_line", "eikonal")


def static_level_grid(
    shape: tuple[int, int],
    transform: Affine,
    stations_lonlat: np.ndarray,
    station_values: np.ndarray,
    k: int,
    idw_power: float,
) -> np.ndarray:
    """Interpolate the static sea level (m, DEM datum) onto the tile grid.

    Inverse-distance k-NN (weights distance^-idw_power) of the station values on a
    STATIC_LEVEL_COARSEN grid, bilinearly upsampled. Station values are MDT + SLR fingerprint
    (`boundaries.static_station_levels`); `transform` must be EPSG:4326.
    """
    rows = np.arange(0, shape[0] + STATIC_LEVEL_COARSEN, STATIC_LEVEL_COARSEN).clip(max=shape[0] - 1)
    cols = np.arange(0, shape[1] + STATIC_LEVEL_COARSEN, STATIC_LEVEL_COARSEN).clip(max=shape[1] - 1)
    rows, cols = np.unique(rows), np.unique(cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = rasterio.transform.xy(transform, rr.ravel(), cc.ravel())
    cell_latlon_rad = np.radians(np.column_stack([ys, xs]))
    coarse = _idw_nearest_k(
        cell_latlon_rad, np.radians(stations_lonlat[:, ::-1]), station_values, min(k, len(station_values)), idw_power,
    ).reshape(rr.shape)
    # Bilinear in index space between the coarse sample rows/cols.
    r_full, c_full = np.arange(shape[0]), np.arange(shape[1])
    r_pos = np.interp(r_full, rows, np.arange(len(rows)))
    c_pos = np.interp(c_full, cols, np.arange(len(cols)))
    return ndimage.map_coordinates(
        coarse, np.meshgrid(r_pos, c_pos, indexing="ij"), order=1, mode="nearest",
    ).astype(np.float64)


def seed_level_idw(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    k: int,
    idw_power: float,
    cell_dx_m: float,
    cell_dy_m: float,
) -> np.ndarray:
    """Interpolate seed water levels (m) over the tile: inverse-distance k-NN in metres.

    Evaluated on a STATIC_LEVEL_COARSEN grid and bilinearly upsampled; the seeds are already
    station IDW values, so the field varies over kilometres.
    """
    r_s = np.unique(np.arange(0, shape[0] + STATIC_LEVEL_COARSEN, STATIC_LEVEL_COARSEN).clip(max=shape[0] - 1))
    c_s = np.unique(np.arange(0, shape[1] + STATIC_LEVEL_COARSEN, STATIC_LEVEL_COARSEN).clip(max=shape[1] - 1))
    rr, cc = np.meshgrid(r_s, c_s, indexing="ij")
    k = min(k, len(values))
    scale = np.array([cell_dy_m, cell_dx_m])
    dist, idx = cKDTree(np.column_stack([rows, cols]) * scale).query(
        np.column_stack([rr.ravel(), cc.ravel()]) * scale, k=k)
    dist, idx = dist.reshape(len(dist), -1), idx.reshape(len(idx), -1)
    weights = np.maximum(dist, 1e-6) ** -idw_power
    coarse = ((np.asarray(values, dtype=np.float64)[idx] * weights).sum(axis=1) / weights.sum(axis=1)).reshape(rr.shape)
    r_pos = np.interp(np.arange(shape[0]), r_s, np.arange(len(r_s)))
    c_pos = np.interp(np.arange(shape[1]), c_s, np.arange(len(c_s)))
    return ndimage.map_coordinates(coarse, np.meshgrid(r_pos, c_pos, indexing="ij"), order=1, mode="nearest")


def straight_line_level(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    attenuation_m_per_m: float,
    cell_dx_m: float,
    cell_dy_m: float,
    bin_m: float = WATERLEVEL_EPSILON_M,
) -> np.ndarray:
    """Return max over seeds of (seed level - attenuation x straight-line distance), m.

    Exact up to `bin_m`: seeds are binned by level, and each bin's Euclidean distance transform
    (cells of cell_dx_m by cell_dy_m) gives its nearest seed.
    """
    values = np.asarray(values, dtype=np.float64)
    bins = np.floor((values - values.min()) / bin_m).astype(np.int64)
    level = np.full(shape, -np.inf)
    for b in np.unique(bins):
        sel = bins == b
        free = np.ones(shape, dtype=bool)
        free[rows[sel], cols[sel]] = False
        dist, (near_r, near_c) = ndimage.distance_transform_edt(free, sampling=(cell_dy_m, cell_dx_m),
                                                             return_indices=True)
        seed_value = np.full(shape, np.nan)
        seed_value[rows[sel], cols[sel]] = values[sel]
        np.maximum(level, seed_value[near_r, near_c] - attenuation_m_per_m * dist, out=level)
    return level


def static_inundation(
    dem: np.ndarray,
    mask: np.ndarray,
    static_level: np.ndarray,
    sources: np.ndarray,
    ocean_code: int = 1,
) -> np.ndarray:
    """Mark land cells below the static level that connect to `sources` through such cells.

    `dem` is the effective DEM (m); `sources` the coastline or hop>=1 seed cells.
    These cells are permanently inundated: a static sea level does not attenuate.
    Rivers and lakes are excluded: their effective DEM is 0 m, so they would carry the
    sea level (and the coastal storm tide seeded on it) up the whole river network.
    """
    below = (dem < static_level) & (mask == LAND_CODE)
    return prune_to_coast_connected(below, sources)


def coastline_mask(mask: np.ndarray, ocean_code: int = 1, river_code: int | None = None) -> np.ndarray:
    """Mark ocean cells of the tile's edge-connected ocean adjacent to land/river.

    Keeps only 8-connected ocean components touching the array edge, so DeltaDTM
    inland "ocean" speckle (ponds, aquaculture, thermokarst) is never seeded.
    `river_code=None` restricts adjacency to land only.
    """
    landlike = mask == 0
    if river_code is not None:
        landlike = landlike | (mask == river_code)
    dilated_landlike = ndimage.binary_dilation(landlike, structure=_STRUCTURE_8)
    return dilated_landlike & open_sea_mask(mask, ocean_code)


def open_sea_mask(mask: np.ndarray, ocean_code: int = 1, seeds: np.ndarray | None = None) -> np.ndarray:
    """Mark ocean cells in 8-connected ocean components touching the array edge or a `seeds` cell."""
    components, _n = ndimage.label(mask == ocean_code, structure=_STRUCTURE_8)
    open_labels = [components[0, :], components[-1, :], components[:, 0], components[:, -1]]
    if seeds is not None:
        open_labels.append(components[seeds])
    open_labels = np.unique(np.concatenate(open_labels))
    return np.isin(components, open_labels[open_labels != 0])


def prune_to_coast_connected(region: np.ndarray, coastline: np.ndarray) -> np.ndarray:
    """Keep only the 8-connected components of `region` touching dilated `coastline`."""
    labels, _ = ndimage.label(region, structure=_STRUCTURE_8)
    if not coastline.any():
        return np.zeros_like(region)
    dilated_coast = ndimage.binary_dilation(coastline, structure=_STRUCTURE_8)
    touching = set(np.unique(labels[dilated_coast])) - {0}
    if not touching:
        return np.zeros_like(region)
    return np.isin(labels, list(touching))


def _idw_seed_values(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: Affine,
    stations_lonlat: np.ndarray,
    station_values: np.ndarray,
    k: int,
    mask: np.ndarray,
    ocean_code: int,
    idw_power: float,
) -> np.ndarray:
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return idw_seed_values_at(np.column_stack([xs, ys]), rows, cols, stations_lonlat, station_values,
                              k, mask, ocean_code, idw_power)


def idw_seed_values_at(
    cell_lonlat: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    stations_lonlat: np.ndarray,
    station_values: np.ndarray,
    k: int,
    mask: np.ndarray,
    ocean_code: int,
    idw_power: float,
) -> np.ndarray:
    """Inverse-distance k-NN (haversine, weights distance^-idw_power) seed values at (lon, lat) cells.

    Positions are (lon, lat) degrees (EPSG:4326; transform projected grids
    first). Each cell uses only stations in its own 8-connected ocean
    component of `mask` (rows/cols locate it), so forcing is not drawn across
    a thin isthmus; a station joins the component of its nearest cell. This is
    tile-local: land barriers mostly outside the tile are invisible. A
    component with no stations falls back to all stations.
    """
    k = min(k, len(station_values))
    cell_latlon_rad = np.radians(np.asarray(cell_lonlat)[:, ::-1])  # (lon,lat) -> (lat,lon)
    station_latlon_rad = np.radians(stations_lonlat[:, ::-1])  # (lon,lat) -> (lat,lon)

    ocean = mask == ocean_code
    components, n_components = ndimage.label(ocean, structure=_STRUCTURE_8)
    cell_components = components[rows, cols]

    if n_components <= 1:
        return _idw_nearest_k(cell_latlon_rad, station_latlon_rad, station_values, k, idw_power)

    seed_tree = BallTree(cell_latlon_rad, metric="haversine")
    _, nearest_seed = seed_tree.query(station_latlon_rad, k=1)
    station_components = cell_components[nearest_seed[:, 0]]

    result = np.empty(len(rows), dtype=np.float64)
    for comp_id in np.unique(cell_components):
        cell_sel = cell_components == comp_id
        station_sel = station_components == comp_id
        if station_sel.any():
            sub_values, sub_latlon = station_values[station_sel], station_latlon_rad[station_sel]
        else:
            sub_values, sub_latlon = station_values, station_latlon_rad
        k_eff = min(k, len(sub_values))
        result[cell_sel] = _idw_nearest_k(cell_latlon_rad[cell_sel], sub_latlon, sub_values, k_eff, idw_power)
    return result


def _idw_nearest_k(
    cell_latlon_rad: np.ndarray,
    station_latlon_rad: np.ndarray,
    station_values: np.ndarray,
    k: int,
    idw_power: float,
) -> np.ndarray:
    tree = BallTree(station_latlon_rad, metric="haversine")
    dist, idx = tree.query(cell_latlon_rad, k=k)
    dist_m = dist * EARTH_RADIUS_M
    # A coincident station would give weight inf; it dominates either way.
    dist_m = np.where(dist_m == 0.0, np.finfo(np.float64).tiny, dist_m)

    weights = dist_m**-idw_power
    values = station_values[idx]
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def apply_volume_cap(
    depth: np.ndarray, dem: np.ndarray, mask: np.ndarray, source_rows: np.ndarray, source_cols: np.ndarray,
    source_levels: np.ndarray, volume_cap: VolumeCap, transform: Affine, cell_dx_m: float, cell_dy_m: float,
    model: str,
) -> tuple[np.ndarray, dict]:
    """Apply correction C to a finished `model` depth (m), as `flood_depth_dense` does last.

    `dem` is the real ground (m) the depth was solved on, before `effective_dem`; the sources
    are the seed cells and their levels (m). Returns the capped depth and its diagnostics.
    """
    if model == "bathtub":
        raise ValueError("apply_volume_cap: the volume cap needs coast-connected patches, not 'bathtub'")
    sources = np.zeros(dem.shape, dtype=bool)
    sources[source_rows, source_cols] = True
    seed_level = np.full(dem.shape, -np.inf)
    np.maximum.at(seed_level, (source_rows, source_cols), np.asarray(source_levels, dtype=np.float64))
    level = effective_dem(dem, mask.astype(np.int8)) + depth
    depth, patches = volume_cap.apply(depth, level, dem, sources, seed_level, transform, cell_dx_m, cell_dy_m,
                                      fill_order="distance" if model == "bathtub_connected" else "level")
    capped = [p for p in patches if p["capped"]]
    return depth, {"n_volume_capped_patches": len(capped),
                   "volume_cap_removed_m3": float(sum(p["model_m3"] - p["kept_m3"] for p in capped))}


def flood_depth_dense(
    dem: np.ndarray,
    mask: np.ndarray,
    friction: np.ndarray,
    transform: Affine,
    *,
    cell_dx_m: float,
    cell_dy_m: float,
    boundaries: gpd.GeoDataFrame | None = None,
    seed_rows: np.ndarray | None = None,
    seed_cols: np.ndarray | None = None,
    seed_values: np.ndarray | None = None,
    k: int = 5,
    idw_power: float | None = None,
    variable: str = "waterlevel",
    ocean_code: int = 1,
    river_code: int | None = None,
    sweep_budget: int | None = None,
    obstacle_coupling: bool = False,
    max_outer_iterations: int | None = None,
    max_rounds: int = 40,
    outer_convergence_pct: float = 0.01,
    waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M,
    static_level: np.ndarray | None = None,
    duration_correction: DurationCorrection | None = None,
    volume_cap: VolumeCap | None = None,
    model: str = "eikonal",
    attenuation_m_per_m: float | None = None,
    trace: SolveTrace | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve flood depth (m) on the full dense tile; 0.0 where not flooded.

    Every cell participates; only the post-solve coast-connectivity filter is applied.
    `dem` (m) must be nodata free. `friction` is the attenuation in m of water level per m
    of path; cells are cell_dx_m (along a row) by cell_dy_m (along a column), the tile
    means from `rasters.grid_spacing_m`.

    Forcing: either `boundaries` (wave-0: stations IDW-seeded onto
    `coastline_mask` over the k nearest, weights distance^-idw_power) or seed_rows/seed_cols/seed_values (hop>=1: absolute
    water levels from a neighbour's wet cells, bypassing coastline/IDW).
    `sweep_budget` (non-coupling only): fixed sweep count instead of converging.

    `obstacle_coupling` addresses the Fast Sweeping/cost-distance leak through
    friction-cheap high terrain (Kasmalkar et al. 2024 "Flow-Tub"): statically
    block cells above the max boundary water level (not ocean), then repeatedly
    solve and block cells with waterlevel <= elevation until the newly blocked
    share < `outer_convergence_pct` percent. Terminates because raising
    friction only lowers eikonal values, so the blocked set only grows.

    `static_level` (m, tile grid, `static_level_grid`) switches on two-component
    propagation: cells below it and connected to the coastline/seeds are permanently
    inundated (`static_inundation`) and become seeds, moving the shoreline inland. Each
    gets max(static level, level of its nearest original seed), so the static part does
    not attenuate, the storm tide stays local to its own coast (a zero-friction new sea
    would carry the highest coastal level everywhere), and only the storm tide on top
    attenuates inland of the new shore.

    `duration_correction` (event_volume.DurationCorrection) scales friction by how long the
    first solve's water stands above the spill level during the storm-tide peak (up where
    briefly, down where long), and solves again with that friction.

    `volume_cap` (event_volume.VolumeCap) then keeps, in each coast-connected patch holding more
    water above the real ground (`dem` before `effective_dem`) than can cross into it during the
    storm-tide hydrograph, only the cells that volume reaches first: highest level first, or for
    "bathtub_connected" (flat level) fewest steps from the coast.

    `model` (FLOOD_MODELS) other than "eikonal" replaces the solve by a level field from the same
    seeds (static seeds included): "bathtub" and "bathtub_connected" take `seed_level_idw`
    (k, idw_power), the latter kept where connected to the coastline; "straight_line" takes
    `straight_line_level` at `attenuation_m_per_m`, connected.
    `friction`, obstacle coupling and the sweep settings are unused there; duration correction
    needs a solve and is refused, as is the volume cap for the unconnected "bathtub".

    `trace` (SolveTrace) observes every eikonal solve: its sweeps and outer iterations.
    """
    if model not in FLOOD_MODELS:
        raise ValueError(f"flood_depth_dense: model {model!r} not in {FLOOD_MODELS}")
    if model != "eikonal" and duration_correction is not None:
        raise ValueError(f"flood_depth_dense: duration correction needs the eikonal solve, not {model!r}")
    if model == "bathtub" and volume_cap is not None:
        raise ValueError("flood_depth_dense: the volume cap needs coast-connected patches, not 'bathtub'")
    if model == "straight_line" and attenuation_m_per_m is None:
        raise ValueError("flood_depth_dense: straight_line needs attenuation_m_per_m")
    using_seeds = seed_rows is not None or seed_cols is not None or seed_values is not None
    if using_seeds and boundaries is not None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values, not both")
    if not using_seeds and boundaries is None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values")
    if using_seeds and (seed_rows is None or seed_cols is None or seed_values is None):
        raise ValueError("flood_depth_dense: seed_rows/seed_cols/seed_values must all be given together")
    # No defaults for the config's idw_power / max_outer_iterations: required where used.
    if idw_power is None and (boundaries is not None or model in ("bathtub", "bathtub_connected")):
        raise ValueError("flood_depth_dense: station IDW and the bathtub models need idw_power")
    if max_outer_iterations is None and obstacle_coupling and model == "eikonal":
        raise ValueError("flood_depth_dense: obstacle_coupling needs max_outer_iterations")

    mask = mask.astype(np.int8)
    ground = dem
    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(MIN_FRICTION_M_PER_M))

    if using_seeds:
        coastline = np.zeros(dem.shape, dtype=bool)
        coastline[seed_rows, seed_cols] = True
        coastline_rows, coastline_cols = seed_rows, seed_cols
        station_values = seed_values
        initial = np.asarray(seed_values, dtype=np.float64)
    else:
        coastline = coastline_mask(mask, ocean_code=ocean_code, river_code=river_code)
        coastline_rows, coastline_cols = np.nonzero(coastline)
        if not coastline.any():
            return np.zeros_like(dem), {"obstacle_coupling": obstacle_coupling}

        stations_lonlat = np.column_stack(
            [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
        )
        station_values = boundaries[variable].to_numpy()
        initial = _idw_seed_values(
            coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
            min(k, len(station_values)), mask, ocean_code, idw_power,
        )

    static_wet = None
    if static_level is not None:
        static_wet = static_inundation(dem, mask, static_level, coastline, ocean_code)
        if static_wet.any():
            s_rows, s_cols = np.nonzero(static_wet)
            nearest = cKDTree(np.column_stack([coastline_rows, coastline_cols])).query(
                np.column_stack([s_rows, s_cols]))[1]
            s_values = np.maximum(static_level[s_rows, s_cols], np.asarray(initial)[nearest])
            coastline_rows = np.concatenate([coastline_rows, s_rows])
            coastline_cols = np.concatenate([coastline_cols, s_cols])
            initial = np.concatenate([np.asarray(initial, dtype=np.float64), s_values])
            station_values = np.concatenate([np.asarray(station_values, dtype=np.float64), s_values])

    def capped(depth: np.ndarray, diagnostics: dict) -> np.ndarray:
        depth, cap_diagnostics = apply_volume_cap(depth, ground, mask, coastline_rows, coastline_cols, initial,
                                                  volume_cap, transform, cell_dx_m, cell_dy_m, model)
        diagnostics.update(cap_diagnostics)
        return depth

    if model != "eikonal":
        rows, cols = np.asarray(coastline_rows), np.asarray(coastline_cols)
        if model == "straight_line":
            level = straight_line_level(dem.shape, rows, cols, initial, attenuation_m_per_m, cell_dx_m, cell_dy_m,
                                        waterlevel_epsilon_m)
        else:
            level = seed_level_idw(dem.shape, rows, cols, initial, k, idw_power, cell_dx_m, cell_dy_m)
        wet = (level > dem) & (mask != ocean_code)
        if model != "bathtub":
            wet = prune_to_coast_connected(wet, coastline)
        depth = np.zeros_like(dem)
        depth[wet] = level[wet] - dem[wet]
        diagnostics = {"model": model}
        if static_wet is not None:
            diagnostics["n_static_wet_cells"] = int(static_wet.sum())
        if volume_cap is not None:
            depth = capped(depth, diagnostics)
        return depth, diagnostics

    on_sweep = trace.sweep if trace is not None else None
    # Seeded components too: a frame of no-data turned land (calibration subgrid) cuts the sea off the edge.
    open_sea = open_sea_mask(mask, ocean_code, seeds=coastline)
    open_sea_coast = open_sea & coastline

    def keep_seeds_local(friction: np.ndarray) -> np.ndarray:
        # At the water class's low friction a basin's highest seed would cross the open sea, or
        # run along the chain of coastline cells, and raise the lower seeds (they are not frozen)
        # and the land behind them. So the open sea beyond the coastline is blocked, and a
        # coastline cell costs as its roughest land/river neighbour.
        land_max = ndimage.maximum_filter(np.where(open_sea, friction.dtype.type(0), friction), size=3)
        out = np.where(open_sea, friction.dtype.type(OBSTACLE_BLOCK_FRICTION), friction)
        out[open_sea_coast] = np.maximum(friction, land_max)[open_sea_coast]
        return out

    def solve(friction: np.ndarray, solve_pass: str) -> tuple[np.ndarray, dict]:
        friction = keep_seeds_local(friction)
        if not obstacle_coupling:
            if trace is not None:
                trace.solve_started(solve_pass, 0, depth_from)
            t = solve_eikonal_dense(
                friction, coastline_rows, coastline_cols, -initial, waterlevel_epsilon_m,
                cell_dx_m=cell_dx_m, cell_dy_m=cell_dy_m,
                max_rounds=max_rounds, sweep_budget=sweep_budget, on_sweep=on_sweep,
            )
            return -t[1:, 1:], {"obstacle_coupling": False}
        dtype = friction.dtype
        n_cells = dem.size
        max_waterlevel = float(station_values.max())
        static_blocked = dem > max_waterlevel

        friction_b = np.where(static_blocked, dtype.type(OBSTACLE_BLOCK_FRICTION), friction)
        prev_blocked = None
        t = None
        inner_diag: dict = {}
        n_outer = 0
        outer_converged = False
        rounds_per_outer, converged_per_outer, newly_blocked_per_outer = [], [], []
        for outer in range(max_outer_iterations):
            n_outer = outer + 1
            if trace is not None:
                trace.solve_started(solve_pass, n_outer, depth_from)
            t, inner_diag = solve_eikonal_dense(
                friction_b, coastline_rows, coastline_cols, -initial, waterlevel_epsilon_m,
                cell_dx_m=cell_dx_m, cell_dy_m=cell_dy_m,
                max_rounds=max_rounds, return_diagnostics=True, on_sweep=on_sweep,
            )
            waterlevel_b = -t[1:, 1:]
            blocked = (waterlevel_b <= dem) | static_blocked
            if prev_blocked is not None:
                # Union keeps the blocked set monotonic; otherwise rerouted flow
                # can unblock a cell and prevent convergence.
                blocked = blocked | prev_blocked
            blocked[coastline_rows, coastline_cols] = False

            n_newly = int(blocked.sum()) if prev_blocked is None else int((blocked & ~prev_blocked).sum())
            pct_newly = 100.0 * n_newly / n_cells
            rounds_per_outer.append(inner_diag["n_rounds_used"])
            converged_per_outer.append(bool(inner_diag["converged"]))
            newly_blocked_per_outer.append(n_newly)
            if trace is not None:
                trace.outer_done(n_outer, waterlevel_b, blocked, pct_newly, inner_diag)
            if prev_blocked is not None and pct_newly < outer_convergence_pct:
                outer_converged = True
                break
            friction_b = np.where(blocked, dtype.type(OBSTACLE_BLOCK_FRICTION), friction)
            prev_blocked = blocked
        return -t[1:, 1:], {
            "obstacle_coupling": True,
            "outer_iterations_used": n_outer,
            "outer_converged": outer_converged,
            "inner_rounds_used": inner_diag.get("n_rounds_used"),
            "inner_converged": inner_diag.get("converged"),
            "final_max_change": inner_diag.get("max_change"),
            # Per outer iteration: the inner solve's rounds (= max_rounds when not converged), and
            # the cells it newly blocked (the first entry counts the static pre-filter too).
            "inner_rounds_per_outer": rounds_per_outer,
            "inner_converged_per_outer": converged_per_outer,
            "newly_blocked_per_outer": newly_blocked_per_outer,
            "last_blocking_outer": max((k for k, n in enumerate(newly_blocked_per_outer, 1) if n), default=0),
        }

    def depth_from(waterlevel: np.ndarray) -> np.ndarray:
        flood = prune_to_coast_connected((waterlevel > dem) & (mask != ocean_code), coastline)
        depth = np.zeros_like(dem)
        depth[flood] = waterlevel[flood] - dem[flood]
        return depth

    waterlevel, diagnostics = solve(friction, "first")
    if duration_correction is not None:
        sources = mask == ocean_code
        sources[coastline_rows, coastline_cols] = True
        multiplier = duration_correction.multiplier(depth_from(waterlevel), dem, sources, transform)
        corrected = multiplier != 1.0
        waterlevel, diagnostics = solve(friction * multiplier, "duration")
        diagnostics["n_duration_corrected_cells"] = int(corrected.sum())
        diagnostics["mean_duration_multiplier"] = float(multiplier[corrected].mean()) if corrected.any() else 1.0

    if static_wet is not None:
        diagnostics["n_static_wet_cells"] = int(static_wet.sum())

    depth = depth_from(waterlevel)
    if volume_cap is not None:
        depth = capped(depth, diagnostics)
    return depth, diagnostics
