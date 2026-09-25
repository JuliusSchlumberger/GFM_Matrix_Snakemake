"""Fast Sweeping Method (Zhao, 2005) for the eikonal equation, dense domain.

Ports the numerical method used by Aqueduct's Julia core (`core/src/core.jl`,
via `Eikonal.jl`'s `FastSweeping`/`sweep!`), operating on the full dense
raster grid - every cell (land, ocean, lake) participates, via plain
array-index arithmetic (no neighbor-index tables). Validated bit-for-bit
identical to real Aqueduct output across 26 real tiles spanning ~9M-135M
cells each (100.000% Jaccard, 0.0m RMSE/mean-error/90th-percentile/max-diff
on every one - see `docs/python_vs_julia_qa.md`) - except for
`solve_eikonal_dense`'s unseeded-cell default, deliberately changed from
Julia's own t=0 to t=+99 (see that function's own comment) to stop unreached
cells from silently reading as "flooded at sea level."

Replicates Eikonal.jl's exact staggered-grid convention (arrival time `t` on
grid vertices, one larger per axis than the `friction`/cell grid, with each
of the 4 sweep directions - "orthants" - using a different single corner
cell to feed a given vertex's update).

Vertex/cell index derivation (0-indexed; vertex grid is (m+1, n+1) for an
(m, n) friction grid), reduced from Eikonal.jl's generic N-D `Orthant`/
`sweep!` to its 4 concrete 2-D cases:

| orthant | row range | col range | t-neighbors used      | cell used   |
|---------|-----------|-----------|-----------------------|-------------|
| 1       | [1, m]    | [1, n]    | (i-1,j), (i,j-1)      | (i-1, j-1)  |
| 2       | [0, m-1]  | [1, n]    | (i+1,j), (i,j-1)      | (i,   j-1)  |
| 3       | [0, m-1]  | [0, n-1]  | (i+1,j), (i,j+1)      | (i,   j)    |
| 4       | [1, m]    | [0, n-1]  | (i-1,j), (i,j+1)      | (i-1, j)    |

The final result at cell (r, c) reads vertex (r+1, c+1) - Julia's
`waterlevel = -t[2:end, 2:end]` drops the vertex grid's first row/col.
Seeding writes directly at vertex (r, c) for coastline cell (r, c) (no
offset) - this asymmetry is Aqueduct's own convention (`core.jl` writes
`solver.t[I] = -initial[I]` using cell-index `I` directly into the vertex
array), not something introduced here.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Julia's `sweep!` visits its 4 Gray-code orthants in this order, in this
# module's own orthant numbering (see module docstring table) - not
# (1, 2, 3, 4).
_ORTHANT_ORDER = (1, 4, 3, 2)


@njit(cache=True)
def _update(t_a: float, t_b: float, v: float, neg_two: float, eight: float, four: float) -> float:
    """Zhao's upwind quadratic update from two orthogonal neighbor times.

    Mirrors Eikonal.jl's `update` for N=2, including its fallback to the
    simpler 1-D characteristic (`t + v`) when the 2-D quadratic solution
    isn't causal (`t <= max(t_a, t_b)`).

    `neg_two`/`eight`/`four` are `-2`/`8`/`4` pre-cast to `t_a`'s own dtype
    by the caller (float32 for real tiles), matching Julia's own arithmetic:
    Julia computes the discriminant in Float32, not Float64, and at real
    friction scales this is a catastrophic cancellation (`b**2` and `4*a*c`
    are O(100-200) while their difference is O(1e-5)) - Julia's result is
    dominated by Float32 rounding noise, not a lightly-rounded true value,
    so matching its output requires reproducing that same noise, computed
    in the same precision with the same literal formula structure.
    """
    b = neg_two * (t_a + t_b)
    c = t_a * t_a + t_b * t_b - v * v
    disc = b * b - eight * c
    fallback = min(t_a + v, t_b + v)
    if disc >= 0:
        cand = (-b + np.sqrt(disc)) / four
        if cand > max(t_a, t_b):
            return min(cand, fallback)
    return fallback


@njit(cache=True)
def _dense_sweep(
    t: np.ndarray, friction: np.ndarray, orthant: int,
    neg_two: float, eight: float, four: float,
) -> float:
    """One dense directional sweep - the 4 concrete cases from the module
    docstring's table, operating directly on `t`'s (m+1, n+1) array via
    index arithmetic (no neighbor-index tables at all).
    """
    m, n = friction.shape
    max_change = 0.0
    if orthant == 1:
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                cand = _update(t[i - 1, j], t[i, j - 1], friction[i - 1, j - 1], neg_two, eight, four)
                if cand < t[i, j]:
                    d = t[i, j] - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    elif orthant == 2:
        for j in range(1, n + 1):
            for i in range(m - 1, -1, -1):
                cand = _update(t[i + 1, j], t[i, j - 1], friction[i, j - 1], neg_two, eight, four)
                if cand < t[i, j]:
                    d = t[i, j] - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    elif orthant == 3:
        for j in range(n - 1, -1, -1):
            for i in range(m - 1, -1, -1):
                cand = _update(t[i + 1, j], t[i, j + 1], friction[i, j], neg_two, eight, four)
                if cand < t[i, j]:
                    d = t[i, j] - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    else:
        for j in range(n - 1, -1, -1):
            for i in range(1, m + 1):
                cand = _update(t[i - 1, j], t[i, j + 1], friction[i - 1, j], neg_two, eight, four)
                if cand < t[i, j]:
                    d = t[i, j] - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    return max_change


def solve_eikonal_dense(
    friction: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_values: np.ndarray,
    epsilon: float,
    max_rounds: int = 10_000,
    sweep_budget: int | None = None,
    verbose: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Solve the eikonal equation on the full dense grid via Fast Sweeping.

    Every cell participates, exactly like Eikonal.jl's own domain (no
    candidate/coastline/ocean restriction at all).

    `verbose`: print each round's `max_change` and elapsed time - diagnostic
    only, for watching convergence rate without waiting for the whole run
    to finish or hit `max_rounds`.

    `return_diagnostics`: if true, also return a dict with `n_rounds_used`,
    `max_change` (the last round's; `None` under `sweep_budget`, which
    tracks no such thing) and `converged` (`max_change <= epsilon`; `None`
    under `sweep_budget`) - used by `flood_model.py`'s obstacle-coupling
    outer loop to log per-tile convergence behaviour.

    Args:
        friction: (m, n) friction values (the full, unmasked tile array).
        seed_rows/seed_cols: (row, col) of each seeded (coastline) CELL -
            seeding writes directly to the same-indexed vertex, per
            Aqueduct's own convention (see module docstring).
        seed_values: Initial `t` values at the seeded vertices (Aqueduct
            seeds with `-waterlevel`). Seeded vertices are NOT frozen after
            initialization - like Julia's `sweep!`, they remain eligible for
            further updates if a neighboring path yields a smaller `t`.
        epsilon: Convergence threshold (max per-round absolute change) -
            mirrors `core.jl`'s `minimum(friction) / (resolution * 10)`.
        max_rounds: Safety cap on rounds of 4 sweeps (only used when
            `sweep_budget` is `None`).
        sweep_budget: If set, run exactly this many individual directional
            sweeps (Julia's real Gray-code order, `_ORTHANT_ORDER`) and
            stop - ignoring `epsilon`/`max_rounds` entirely. `3` reproduces
            Eikonal.jl's own runtime behaviour in Aqueduct's usage
            (bit-for-bit validated, see module docstring). `None` (the
            production default) instead runs the round-based solve below,
            capped at `max_rounds`.

    Returns:
        `t`, shape `(m+1, n+1)` - read cell `(r, c)`'s result at vertex
        `(r+1, c+1)`. If `return_diagnostics`, `(t, diagnostics_dict)`
        instead - see `return_diagnostics` above.
    """
    m, n = friction.shape
    dtype = friction.dtype
    # Unseeded default: t=+99 (waterlevel=-t=-99m), not Julia/Aqueduct's own
    # t=0 (waterlevel=0m) - 0m is a physically real elevation (mean sea
    # level), so a cell the relaxation never reaches would otherwise
    # silently read as "flooded" for any DEM cell below 0m. +99m mirrors
    # rasters.DEM_NODATA_M/land_fill_value_m's own "definitely dry" sentinel
    # (opposite sign), and stays within int16-centimetre range so nothing
    # downstream needs special-case handling for it. The relaxation below
    # only ever decreases t (Gauss-Seidel/Dijkstra-style), so a real
    # candidate from an actual seed always overwrites the sentinel; a cell
    # no seed's influence ever reaches keeps it, correctly reading as
    # "never flooded" rather than "flooded at exactly sea level."
    t = np.full((m + 1, n + 1), 99.0, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values

    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    if sweep_budget is not None:
        for i in range(sweep_budget):
            _dense_sweep(t, friction, _ORTHANT_ORDER[i % 4], neg_two, eight, four)
        if return_diagnostics:
            return t, {"n_rounds_used": None, "max_change": None, "converged": None}
        return t

    if verbose:
        import time
        start = time.perf_counter()

    n_rounds_used = 0
    max_change = 0.0
    for round_idx in range(max_rounds):
        max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            max_change = max(max_change, _dense_sweep(t, friction, orthant, neg_two, eight, four))
        n_rounds_used = round_idx + 1
        if verbose:
            print(f"    round {round_idx + 1}: max_change={max_change:.6g}  "
                  f"epsilon={epsilon:.6g}  elapsed={time.perf_counter() - start:.1f}s",
                  flush=True)
        if max_change <= epsilon:
            break
    if return_diagnostics:
        return t, {"n_rounds_used": n_rounds_used, "max_change": max_change, "converged": max_change <= epsilon}
    return t
