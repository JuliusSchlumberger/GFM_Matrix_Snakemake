"""Python port of Aqueduct's flood model (`core/src/core.jl`, `flood_depth`).

`flood_depth_dense` is the validated, production implementation (dense
domain, exactly 3 sweeps) - bit-for-bit identical to real Aqueduct output
across 26 real tiles spanning ~9M-135M cells each (100.000% Jaccard, 0.0m
RMSE/mean-error/90th-percentile/max-diff on every one - see
`docs/python_vs_julia_qa.md`). See the project plan
(`buzzing-enchanting-barto.md`) for the full derivation of every step below,
including the two precision details that would otherwise silently diverge
from Julia: `coastline_mask`'s exact boolean reduction, and the Haversine
axis-order/unit convention required by scikit-learn vs. the
(internally-consistent) convention `core.jl` uses via `Distances.jl`.

An earlier version of this module also had a `flood_depth` (compacted
domain, iterated to full convergence) - removed once `flood_depth_dense`
proved both correct and faster; see git history for reference if ever
needed.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.transform
from affine import Affine
from scipy import ndimage
from sklearn.neighbors import BallTree

from eikonal import solve_eikonal_dense
from flood_extent import effective_dem

EARTH_RADIUS_M = 6_371_000.0
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

# Decision-relevant flood/no-flood depth threshold: below this, a "flooded"
# cell is not considered meaningfully different from dry for reporting/
# comparison purposes (not currently used to change flood_depth_dense's own
# flood/no-flood classification, which stays the exact `waterlevel > dem`
# test it always has - this is for diagnostics and any future consumer that
# needs a "meaningful vs negligible" cut, e.g. the spatial-diagnostics
# tooling's red/green extent-change classification). Expressed in
# centimetres, matching every other quantity's on-disk precision now (see
# rasters.py's int16 encodings) - see conversation 2026-08-01.
MIN_FLOOD_DEPTH_CM = 10

# Default convergence threshold for the eikonal solve's round-based mode
# (used by the default max_rounds-capped path, every obstacle_coupling
# inner solve, and ignored entirely only when an explicit fixed
# `sweep_budget` is passed instead - see solve_eikonal_dense). 0.1m is the
# decision-relevant flood/no-flood threshold - depth precision beyond that
# isn't used - so epsilon is set well below it (not the previous
# friction/resolution-derived formula, which chased ~1e-6 to 1e-7m, far
# finer than anything physically meaningful) - see conversation 2026-07-31.
# Overridable via flood_depth_dense's own `waterlevel_epsilon_m` parameter
# (2026-08 - config-driven via simulation.flooding.waterlevel_epsilon_m in
# config.yml, not just this hardcoded default) - this constant remains the
# fallback for direct calls (tests, calibration scripts) that don't pass one.
WATERLEVEL_EPSILON_M = 0.03

# Friction assigned to cells the obstacle-coupling machinery has determined
# cannot legitimately flood (see flood_depth_dense's `obstacle_coupling`
# docs below) - deliberately far above OBSTACLE_BLOCK_FRICTION's earlier
# exploratory value (100.0) so the cost of any path crossing such a cell is
# unmistakably prohibitive even in float32, at negligible extra cost since
# it is never on a real shortest path once blocked.
OBSTACLE_BLOCK_FRICTION = 9999.0


def coastline_mask(mask: np.ndarray, ocean_code: int = 1, river_code: int | None = None) -> np.ndarray:
    """Ocean cells within 1px of land (or river) that are actually part of
    the tile's real, edge-connected ocean body - 2026-08 fix to `core.jl`'s
    original `coastlinemask` (`.!(dilate(landmask) .!= mask)`, equivalent to
    `dilate(landmask) & (mask == ocean_code)`).

    The original formula treats ANY ocean-coded cell touching land as
    coastline - including isolated inland "ocean" speckle. DeltaDTM
    regularly miscodes ponds/aquaculture/thermokarst as `ocean_code`, the
    same issue already worked around elsewhere in this pipeline (e.g.
    `tile_generation.river_mouth_min_coastal_component_cells`) - such a
    speckle patch would otherwise get IDW-seeded from real ocean stations
    and start flooding from a location with no real path to the sea.

    Fix: label the tile's ocean-coded cells into 8-connected components,
    keep only the component(s) that actually touch the tile's own array
    edge (a genuine land-locked pond can never do this, since the tile
    boundary is where this tile's real ocean connects to the wider world
    ocean outside it), and require adjacency to land OR `river_code` (not
    land only) - river mouths are legitimate forcing entry points too.

    `river_code`: defaults to `None` (old land-only adjacency, for the two
    existing direct callers - `tests/test_obstacle_coupling_calibration.py`,
    `tests/diagnose_large_residual.py` - which still benefit from the
    edge-connectivity fix even without passing it).
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
    """Keep only the 8-connected components of `region` touching `coastline`.

    This is `core.jl`'s own step 10 (`label_components` + keep components
    whose label appears under the dilated coastline). Applied here to two
    different inputs for two different reasons:
      - to the candidate mask (pre-solve): a provably-exact optimization,
        not an approximation - since every genuinely flooded cell satisfies
        `dem < max_waterlevel` (flood subseteq candidate, proven in the plan
        doc), any candidate component disconnected from the coast can never
        contain a flood cell connected to the coast either, so discarding it
        cannot change the final answer.
      - to the flood mask (post-solve): this IS the model's actual required
        step 10, not an optimization at all.
    Implemented via `np.isin` against a label array, never per-component
    index lists - avoiding `component_indices`'s documented OOM cause in
    `core.jl` regardless of domain size.
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
    """Inverse-distance-squared interpolation of `station_values` onto cells,
    restricted to stations that are ocean-connected to each cell.

    Axis order/units are the one place this deliberately diverges from a
    literal transliteration of `core.jl`: `sklearn.neighbors.BallTree` with
    `metric="haversine"` requires (lat, lon) in radians, whereas Julia's
    `Distances.Haversine` (confirmed by reading its source directly) expects
    (lon, lat) in degrees - opposite axis order. Getting this backwards
    silently corrupts the `cos(lat)` scaling term in the distance formula.
    Absolute distance scale (radians vs. meters) does not matter for the IDW
    ratio math or k-NN ranking - only axis order does - but distances are
    still scaled to meters here to keep values physically meaningful for any
    future debugging.

    Connectivity filter: nearest-by-straight-line-distance is not the same
    as nearest-by-water. A coastal cell on a thin isthmus/spit could
    otherwise draw its boundary forcing from a station on the physically
    disconnected far side (e.g. a bay), which is not reachable by any real
    water path. Cells are restricted to stations in the SAME connected
    component of the tile's own ocean mask (8-connectivity, matching
    `coastline_mask`'s dilation). A station is assigned to whichever
    component its nearest coastline seed cell belongs to - stations
    routinely fall outside the tile's own grid (`station_search_buffer_deg`
    deliberately searches beyond the tile bbox), so there is no mask cell to
    look them up in directly; this is a tile-local proxy, not a check
    against the true global ocean topology, so it cannot see a land barrier
    that lies mostly outside this tile - see conversation 2026-08-01.
    Falls back to using every candidate station, ignoring connectivity, only
    for cells in a component that ended up with zero stations assigned to it
    at all (better than NaN/crashing; strictly rarer than the bug being
    fixed, since a component with any real coastline nearly always has its
    own nearby stations).
    """
    k = min(k, len(station_values))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    cell_latlon_rad = np.radians(np.column_stack([ys, xs]))
    station_latlon_rad = np.radians(stations_lonlat[:, ::-1])  # (lon,lat) -> (lat,lon)

    ocean = mask == ocean_code
    components, n_components = ndimage.label(ocean, structure=_STRUCTURE_8)
    cell_components = components[rows, cols]

    if n_components <= 1:
        # Fast, common path: a single connected ocean body means the
        # connectivity filter can never change the result - skip straight to
        # the plain, tile-wide k-NN (identical to the pre-filter behaviour).
        return _idw_nearest_k(cell_latlon_rad, station_latlon_rad, station_values, k)

    # Assign each station to the component of its nearest coastline seed cell.
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
    """Inverse-distance-squared average of the k nearest (haversine) stations
    for each cell - the actual weighting math shared by every connectivity
    branch in `_idw_seed_values`.
    """
    tree = BallTree(station_latlon_rad, metric="haversine")
    dist, idx = tree.query(cell_latlon_rad, k=k)
    dist_m = dist * EARTH_RADIUS_M
    # Guard exact station/cell coincidence (dist=0) without producing NaN;
    # core.jl has no such guard and would propagate Inf there instead, but a
    # coincident station dominating the weighted average either way is the
    # same physical outcome.
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
    max_outer_iterations: int = 5,
    max_rounds: int = 12,
    outer_convergence_pct: float = 0.01,
    waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M,
) -> tuple[np.ndarray, dict]:
    """Production flood-depth solve: dense domain, round-based by default
    (2026-08 - see `sweep_budget`/`max_rounds` below).

    The original fixed `sweep_budget=3` configuration (still available on
    request, see `sweep_budget` below) was validated bit-for-bit identical
    to real Aqueduct output across 26 real tiles spanning ~9M-135M cells
    each (100.000% Jaccard, 0.0m RMSE/mean-error/90th-percentile/max-diff on
    every one - see `docs/python_vs_julia_qa.md`), after two real bugs were
    found and fixed in `eikonal.py`:
      - a float32-precision discriminant bug (this port originally computed
        the eikonal update's discriminant in higher precision than Julia
        actually uses - matching Julia's real, imprecise arithmetic exactly
        was necessary, not "more correct" arithmetic).
      - a sweep-order bug (Julia's real Gray-code sweep order is (1, 4, 3, 2)
        in this module's own orthant numbering, not the numerically-obvious
        (1, 2, 3, 4) - see `eikonal._ORTHANT_ORDER`).

    Follow-up calibration (2026-08, see `tests/sweep_budget_calibration/`
    and `tests/obstacle_coupling_calibration/`, 7 real tiles spanning
    ~4K-207M cells) found that a fixed 3 sweeps under-converges on larger/
    geometrically-complex tiles - e.g. a ~25M-cell tile's flooded-cell count
    was still ~1% below its sweep-64 value at sweep 3, and still slowly
    growing at sweep 64. The round-based mode (default now - `sweep_budget=
    None`, capped at `max_rounds`) fixes this: it costs nothing extra on the
    many tiles that already converge in a handful of sweeps (it just stops
    early via the same epsilon check `WATERLEVEL_EPSILON_M` the
    `obstacle_coupling` inner solve already used), while giving harder tiles
    up to `max_rounds` rounds instead of being hard-capped at 3.

    The dense solver has no candidate/elevation domain restriction at all -
    every cell participates, exactly like Julia's own full-tile solve - so
    there is no pre-solve candidate-mask/connectivity pruning step, only the
    same post-solve connectivity filter Julia itself applies. That's true
    regardless of `obstacle_coupling` - it changes what FRICTION values the
    solver sees, never which cells participate.

    `obstacle_coupling` (default off - see below for why the default
    solve above is unaffected either way): Fast Sweeping shares a known
    structural weakness with cost-distance flood models (Kasmalkar et al.
    2024's "Flow-Tub" critique) - a friction-cheap "shortcut" through terrain
    higher than the locally-attenuated water level can produce an
    illegitimate high potential value at cells beyond it, since Fast
    Sweeping's Gauss-Seidel relaxation has no per-step elevation check the
    way Dijkstra/BFS-based methods do (no monotonic finalization property -
    a cell's value can keep improving across many sweeps, so there's no safe
    intermediate point to check elevation mid-solve). When enabled, this
    runs an outer loop instead of a single solve:
      1. Static pre-filter (free, exact, zero iteration cost): any cell
         whose effective elevation exceeds the highest boundary water level
         anywhere in the tile can never legitimately flood under ANY
         friction/path, since friction only ever attenuates the propagated
         level, never amplifies it. Ocean cells are never blocked by this
         condition (their effective elevation is always 0, below any real
         positive water level) - correctly so, since the sea is how the
         flood signal reaches other coastal points, not something to block.
         These cells get `OBSTACLE_BLOCK_FRICTION` from the start.
      2. Solve (up to `max_rounds` rounds, epsilon=`WATERLEVEL_EPSILON_M`),
         then block any cell where the resulting (locally-attenuated)
         waterlevel <= its elevation - it can't legitimately be on a real
         flood path either - and re-solve. Repeats until the newly-blocked
         cell count drops below `outer_convergence_pct`% of the tile (or
         `max_outer_iterations` is reached).
    Monotonicity (raising friction anywhere can only lower or hold eikonal
    solution values everywhere, never raise them) guarantees the blocked-cell
    set only grows across iterations, so this terminates in finite steps.

    Args:
        dem, mask, friction, transform: same shape/type conventions as
            `core.jl`'s `flood_depth`, except `transform` (an
            `affine.Affine`), needed here to convert pixel coordinates to
            lon/lat for the boundary-station KNN search - `core.jl` gets
            this for free from `GeoArrays.coords`, since GeoArrays rasters
            carry their own transform. Assumes `dem`/`mask`/`friction`
            already have no nodata cells (guaranteed by this pipeline's
            `extract_dem`/`extract_dem_mask`/`compute_friction` - see
            `src/rasters.py`).
        boundaries: real/virtual water-level stations, IDW-seeded onto this
            tile's own `coastline_mask` fringe (2026-08 - the wave-0 path,
            for a tile with its own real ocean edge). Must be non-empty if
            given (the caller, mirroring the existing `run_aqueduct.py`
            skip logic, should not invoke this otherwise).
        seed_rows, seed_cols, seed_values: 2026-08 - the hop>=1 hinterland
            path, an alternative to `boundaries`: DIRECT eikonal seed cells
            and values (already-known absolute water levels, typically
            collected from an already-simulated neighbour tile's own wet
            cells - see `boundaries.collect_neighbor_wave_seeds`), bypassing
            `coastline_mask`/IDW entirely. A hop>=1 tile has no `ocean_code`
            cells of its own by construction, so `coastline_mask` would
            find nothing to seed from regardless of what `boundaries` it
            was given - this is the only way such a tile can flood at all.
            Must be non-empty (same convention as `boundaries`) if given.
            Exactly one of `boundaries` or this triple must be given.
            `obstacle_coupling=True` works with this path too (2026-08,
            validated against a real hop>=1 tile - tile 2494, seeded from
            already-simulated neighbour tile 1482's own real RP1000/SLR_2000
            output, 78,648 real seed cells, values 3.11-4.61m after fixing
            `collect_neighbor_wave_seeds`'s effective_dem bug - see that
            function's own docstring): `station_values` is aliased to
            `seed_values` below, so the static pre-filter's `max_waterlevel
            = station_values.max()` uses the seed values' own max as the
            "highest known water level for this tile" threshold. Coupled
            and uncoupled solves produced IDENTICAL flooded-cell counts
            (89,883 of 15,859,188) and a max per-cell depth difference of
            0.52m / mean 0.07m over flooded-in-either cells (outer loop
            converged in 2 iterations; the final iteration's own inner
            solve did not fully converge within max_rounds=12) - consistent
            with the coupling/non-coupling agreement already established
            for wave-0 tiles, not a new discrepancy specific to the seed path.
        resolution, k, variable, ocean_code, river_code: flood-model
            parameters - resolution in metres, `k` nearest boundary
            stations for IDW, `variable` the boundary water-level column
            name, `ocean_code`/`river_code` the mask values marking open
            ocean/river (only used on the `boundaries` path - see
            `coastline_mask`).
        sweep_budget: if given (an int), forces the OLD fixed-count mode -
            run exactly this many individual directional sweeps
            unconditionally, ignoring `max_rounds`/epsilon entirely (`3`
            reproduces real Aqueduct/Julia's confirmed, bit-exact-validated
            production behaviour - see module docstring history). `None`
            (the default, 2026-08) instead runs the round-based solve capped
            at `max_rounds`, stopping early once converged (round-level
            max change <= `WATERLEVEL_EPSILON_M`) - see module docstring for
            why this replaced the fixed count as the default. Only used when
            `obstacle_coupling` is false (the outer-loop's own inner solve
            below always uses the round-based mode).
        obstacle_coupling: enable the static-pre-filter + outer-loop
            algorithm above instead of the single solve. Off by default.
        max_rounds: caps the round-based solve (4 individual sweeps per
            round - see `sweep_budget` above), each round checked against
            `WATERLEVEL_EPSILON_M` for early exit. Used both by the default
            non-coupling path (when `sweep_budget` is `None`) and by every
            `obstacle_coupling` outer iteration's own inner solve. Default
            12 (up to 48 sweeps) - empirically calibrated 2026-08 across 7
            real tiles spanning ~4K-207M cells (see module docstring).
        max_outer_iterations, outer_convergence_pct: only used when
            `obstacle_coupling` is true - `max_outer_iterations` caps the
            outer loop; `outer_convergence_pct` is the tolerant early-stop
            threshold (percent of the tile's cells newly blocked this outer
            iteration).
        waterlevel_epsilon_m: round-level early-exit convergence threshold
            for the round-based solve (metres - see module docstring's
            `WATERLEVEL_EPSILON_M` note on why 0.03m). Defaults to the
            module constant; exposed as a parameter (2026-08) so it can be
            set from `simulation.flooding.waterlevel_epsilon_m` in
            `config.yml` rather than only ever the hardcoded default.

    Returns:
        `(waterdepth, diagnostics)`. `waterdepth` has the same shape as
        `dem`, `0.0` where not flooded. `diagnostics` always has key
        `obstacle_coupling` (bool); when true it also has
        `outer_iterations_used`, `outer_converged`, `inner_rounds_used` and
        `inner_converged` (the last inner solve's), and `final_max_change`.
    """
    using_seeds = seed_rows is not None or seed_cols is not None or seed_values is not None
    if using_seeds and boundaries is not None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values, not both")
    if not using_seeds and boundaries is None:
        raise ValueError("flood_depth_dense: pass either boundaries or seed_rows/seed_cols/seed_values")
    if using_seeds and (seed_rows is None or seed_cols is None or seed_values is None):
        raise ValueError("flood_depth_dense: seed_rows/seed_cols/seed_values must all be given together")

    # int8, not int64: mask only ever holds a handful of small codes (land=0,
    # ocean/lake/river - see rasters.extract_dem_mask's docstring), and
    # every downstream use is a plain equality/inequality comparison
    # (dtype-agnostic) - int64 was 8x more memory than this array ever
    # needed, for no functional reason (found 2026-08 investigating OOM
    # failures on large real tiles during obstacle-coupling calibration).
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
