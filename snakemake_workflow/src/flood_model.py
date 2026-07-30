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


def coastline_mask(mask: np.ndarray, ocean_code: int = 1) -> np.ndarray:
    """Ocean cells within 1px of land - `core.jl`'s `coastlinemask`.

    `core.jl` computes `.!(dilate(landmask) .!= mask)`. Because
    `dilate(landmask)` is boolean (0/1) and `mask` is integer-valued (0=land,
    1=ocean, 2=lake, 3=river), and every land cell trivially has
    `dilate(landmask) == 1` (dilation always preserves the original set), the
    only way for `dilate(landmask) == mask` to hold is `dilate(landmask)==1
    AND mask==ocean_code` - i.e. this reduces exactly to `dilate(landmask) &
    (mask == ocean_code)`. Lake/river cells adjacent to land are NOT
    coastline under this formula, only ocean cells are.
    """
    landmask = mask == 0
    dilated_land = ndimage.binary_dilation(landmask, structure=_STRUCTURE_8)
    return dilated_land & (mask == ocean_code)


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
) -> np.ndarray:
    """Inverse-distance-squared interpolation of `station_values` onto cells.

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
    """
    k = min(k, len(station_values))
    station_latlon_rad = np.radians(stations_lonlat[:, ::-1])  # (lon,lat) -> (lat,lon)
    tree = BallTree(station_latlon_rad, metric="haversine")

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    cell_latlon_rad = np.radians(np.column_stack([ys, xs]))

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
    boundaries: gpd.GeoDataFrame,
    transform: Affine,
    *,
    resolution: float = 30.0,
    k: int = 5,
    variable: str = "waterlevel",
    ocean_code: int = 1,
    sweep_budget: int | None = 3,
) -> np.ndarray:
    """Production flood-depth solve: dense domain, exactly 3 sweeps.

    Validated bit-for-bit identical to real Aqueduct output across 26 real
    tiles spanning ~9M-135M cells each (100.000% Jaccard, 0.0m
    RMSE/mean-error/90th-percentile/max-diff on every one - see
    `docs/python_vs_julia_qa.md`), after two real bugs were found and fixed
    in `eikonal.py`:
      - a float32-precision discriminant bug (this port originally computed
        the eikonal update's discriminant in higher precision than Julia
        actually uses - matching Julia's real, imprecise arithmetic exactly
        was necessary, not "more correct" arithmetic).
      - a sweep-order bug (Julia's real Gray-code sweep order is (1, 4, 3, 2)
        in this module's own orthant numbering, not the numerically-obvious
        (1, 2, 3, 4) - see `eikonal._ORTHANT_ORDER`).

    The dense solver has no candidate/elevation domain restriction at all -
    every cell participates, exactly like Julia's own full-tile solve - so
    there is no pre-solve candidate-mask/connectivity pruning step, only the
    same post-solve connectivity filter Julia itself applies.

    Args:
        dem, mask, friction, boundaries, transform: same shape/type
            conventions as `core.jl`'s `flood_depth`, except `transform` (an
            `affine.Affine`), needed here to convert pixel coordinates to
            lon/lat for the boundary-station KNN search - `core.jl` gets
            this for free from `GeoArrays.coords`, since GeoArrays rasters
            carry their own transform. Assumes `dem`/`mask`/`friction`
            already have no nodata cells (guaranteed by this pipeline's
            `extract_dem`/`extract_dem_mask`/`compute_friction` - see
            `src/rasters.py`) and that `boundaries` is non-empty (the
            caller, mirroring the existing `run_aqueduct.py` skip logic,
            should not invoke this otherwise).
        resolution, k, variable, ocean_code: flood-model parameters -
            resolution in metres, `k` nearest boundary stations for IDW,
            `variable` the boundary water-level column name, `ocean_code`
            the mask value marking open ocean.
        sweep_budget: number of individual directional sweeps to run before
            stopping (default 3, matching Julia's confirmed real production
            behaviour - see module docstring history). `None` iterates the
            solver to full numerical convergence instead - much slower
            (100x+ on complex tiles) and NOT the validated/matching
            configuration; only useful for diagnostics.

    Returns:
        `waterdepth`, same shape as `dem`, `0.0` where not flooded.
    """
    mask = mask.astype(np.int64)
    dem = effective_dem(dem, mask)
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))

    coastline = coastline_mask(mask, ocean_code=ocean_code)
    coastline_rows, coastline_cols = np.nonzero(coastline)
    if not coastline.any():
        return np.zeros_like(dem)

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    station_values = boundaries[variable].to_numpy()
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(k, len(station_values)),
    )

    # epsilon uses the GLOBAL minimum friction over the whole (unmasked)
    # tile, matching core.jl's `minimum(friction)`.
    epsilon = float(friction.min()) / (resolution * 10.0)

    t = solve_eikonal_dense(
        friction, coastline_rows, coastline_cols, -initial, epsilon,
        sweep_budget=sweep_budget,
    )
    waterlevel = -t[1:, 1:]

    flood = (waterlevel > dem) & (mask != ocean_code)
    flood = prune_to_coast_connected(flood, coastline)

    waterdepth = np.zeros_like(dem)
    waterdepth[flood] = waterlevel[flood] - dem[flood]
    return waterdepth
