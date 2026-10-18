"""Python port of Aqueduct's flood model (`core/src/core.jl`, `flood_depth`).

`flood_depth_dense` is the production entry point. Julia parity in fixed-3-sweep
mode depends on `coastline_mask`'s exact boolean reduction and on the Haversine
axis order (sklearn: (lat, lon) rad; Distances.jl: (lon, lat) deg).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.transform
from affine import Affine
from scipy import ndimage
from sklearn.neighbors import BallTree

from eikonal import solve_eikonal_dense


LAND_CODE = 0


def effective_dem(dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set elevation (m) to 0.0 on non-land cells, as core.jl's `dem[.!landmask] .= 0.0`.

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

# Prohibitive even in float32 for any path crossing a blocked cell.
OBSTACLE_BLOCK_FRICTION = 9999.0


def coastline_mask(mask: np.ndarray, ocean_code: int = 1, river_code: int | None = None) -> np.ndarray:
    """Mark ocean cells of the tile's edge-connected ocean adjacent to land/river.

    Unlike core.jl's `dilate(landmask) & (mask == ocean_code)`, keeps only
    8-connected ocean components touching the array edge, so DeltaDTM inland
    "ocean" speckle (ponds, aquaculture, thermokarst) is never seeded.
    `river_code=None` restricts adjacency to land only.
    """
    ocean = mask == ocean_code
    components, _n = ndimage.label(ocean, structure=_STRUCTURE_8)
    edge_labels = np.unique(np.concatenate(
        [components[0, :], components[-1, :], components[:, 0], components[:, -1]]
    ))
    edge_labels = edge_labels[edge_labels != 0]
    edge_connected_ocean = np.isin(components, edge_labels)

    landlike = mask == 0
    if river_code is not None:
        landlike = landlike | (mask == river_code)
    dilated_landlike = ndimage.binary_dilation(landlike, structure=_STRUCTURE_8)
    return dilated_landlike & edge_connected_ocean


def prune_to_coast_connected(region: np.ndarray, coastline: np.ndarray) -> np.ndarray:
    """Keep only the 8-connected components of `region` touching dilated `coastline`.

    core.jl step 10 (`label_components`), via np.isin on labels to avoid
    core.jl's `component_indices` OOM.
    """
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
) -> np.ndarray:
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return idw_seed_values_at(np.column_stack([xs, ys]), rows, cols, stations_lonlat, station_values,
                              k, mask, ocean_code)


def idw_seed_values_at(
    cell_lonlat: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    stations_lonlat: np.ndarray,
    station_values: np.ndarray,
    k: int,
    mask: np.ndarray,
    ocean_code: int,
) -> np.ndarray:
    """Inverse-distance-squared k-NN (haversine) seed values at (lon, lat) cells.

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
        return _idw_nearest_k(cell_latlon_rad, station_latlon_rad, station_values, k)

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
        result[cell_sel] = _idw_nearest_k(cell_latlon_rad[cell_sel], sub_latlon, sub_values, k_eff)
    return result


def _idw_nearest_k(
    cell_latlon_rad: np.ndarray,
    station_latlon_rad: np.ndarray,
    station_values: np.ndarray,
    k: int,
) -> np.ndarray:
    tree = BallTree(station_latlon_rad, metric="haversine")
    dist, idx = tree.query(cell_latlon_rad, k=k)
    dist_m = dist * EARTH_RADIUS_M
    # dist=0 guard (core.jl propagates Inf); a coincident station dominates either way.
    dist_m = np.where(dist_m == 0.0, np.finfo(np.float64).tiny, dist_m)

    weights = dist_m**-2
    values = station_values[idx]
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def flood_depth_dense(
    dem: np.ndarray,
    mask: np.ndarray,
    friction: np.ndarray,
    transform: Affine,
    *,
    boundaries: gpd.GeoDataFrame | None = None,
    seed_rows: np.ndarray | None = None,
    seed_cols: np.ndarray | None = None,
    seed_values: np.ndarray | None = None,
    resolution: float = 30.0,
    k: int = 5,
    variable: str = "waterlevel",
    ocean_code: int = 1,
    river_code: int | None = None,
    sweep_budget: int | None = None,
    obstacle_coupling: bool = False,
    max_outer_iterations: int = 3,
    max_rounds: int = 40,
    outer_convergence_pct: float = 0.01,
    waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M,
) -> tuple[np.ndarray, dict]:
    """Solve flood depth (m) on the full dense tile; 0.0 where not flooded.

    Every cell participates, as core.jl's full-tile solve; only core.jl's
    post-solve coast-connectivity filter is applied. `dem` (m) must be nodata
    free. `resolution` is pixel size in metres.

    Forcing: either `boundaries` (wave-0: stations IDW-seeded onto
    `coastline_mask`) or seed_rows/seed_cols/seed_values (hop>=1: absolute
    water levels from a neighbour's wet cells, bypassing coastline/IDW).
    `sweep_budget` (non-coupling only): fixed sweep count; 3 = Aqueduct.

    `obstacle_coupling` addresses the Fast Sweeping/cost-distance leak through
    friction-cheap high terrain (Kasmalkar et al. 2024 "Flow-Tub"): statically
    block cells above the max boundary water level (not ocean), then repeatedly
    solve and block cells with waterlevel <= elevation until the newly blocked
    share < `outer_convergence_pct` percent. Terminates because raising
    friction only lowers eikonal values, so the blocked set only grows.
    """
    using_seeds = seed_rows is not None or seed_cols is not None or seed_values is not None
    if using_seeds and boundaries is not None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values, not both")
    if not using_seeds and boundaries is None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values")
    if using_seeds and (seed_rows is None or seed_cols is None or seed_values is None):
        raise ValueError("flood_depth_dense: seed_rows/seed_cols/seed_values must all be given together")

    mask = mask.astype(np.int8)
    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))

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
            min(k, len(station_values)), mask, ocean_code,
        )

    if not obstacle_coupling:
        t = solve_eikonal_dense(
            friction, coastline_rows, coastline_cols, -initial, waterlevel_epsilon_m,
            max_rounds=max_rounds, sweep_budget=sweep_budget,
        )
        waterlevel = -t[1:, 1:]
        diagnostics = {"obstacle_coupling": False}
    else:
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
        for outer in range(max_outer_iterations):
            n_outer = outer + 1
            t, inner_diag = solve_eikonal_dense(
                friction_b, coastline_rows, coastline_cols, -initial, waterlevel_epsilon_m,
                max_rounds=max_rounds, return_diagnostics=True,
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
            if prev_blocked is not None and pct_newly < outer_convergence_pct:
                outer_converged = True
                break
            friction_b = np.where(blocked, dtype.type(OBSTACLE_BLOCK_FRICTION), friction)
            prev_blocked = blocked
        waterlevel = -t[1:, 1:]
        diagnostics = {
            "obstacle_coupling": True,
            "outer_iterations_used": n_outer,
            "outer_converged": outer_converged,
            "inner_rounds_used": inner_diag.get("n_rounds_used"),
            "inner_converged": inner_diag.get("converged"),
            "final_max_change": inner_diag.get("max_change"),
        }

    flood = (waterlevel > dem) & (mask != ocean_code)
    flood = prune_to_coast_connected(flood, coastline)

    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]
    return waterdepth, diagnostics
