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

# Unseeded/unreached default for `t` - see solve_eikonal_dense's own comment.
UNREACHED_T = 99.0

# Vertex block edge length for `_block_sweep`'s skip-unchanged-blocks
# bookkeeping. Any value gives bit-identical results; 64 measured well on
# real calibration tiles (2026-10: 1.2-5.8x over plain dense sweeps).
SWEEP_BLOCK_SIZE = 64


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

    Loop order is row-outer/column-inner (contiguous for numpy's C-order
    arrays), not Julia's column-outer/row-inner (contiguous for Julia's own
    column-major arrays) - ~2.5-3x faster on real tiles. This is bit-for-bit
    identical to Julia's order, not an approximation: within one orthant
    sweep, vertex (i, j)'s update reads only its two upwind neighbours, and
    both loop orders visit both of those before (i, j) itself, so every
    update sees exactly the same inputs (locked in by
    `tests/eikonal_kernel_validation`).

    Early-out: `_update`'s result is always >= min(t_a, t_b) (the quadratic
    branch requires > max(t_a, t_b); the fallback is min(t_a, t_b) + v with
    v >= 0), so if min(t_a, t_b) >= t[i, j] it can never improve t[i, j] -
    skipping it is exact, and avoids the sqrt for every blocked, unreached
    or already-settled vertex.
    """
    m, n = friction.shape
    max_change = 0.0
    if orthant == 1:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                t_a = t[i - 1, j]
                t_b = t[i, j - 1]
                t_c = t[i, j]
                if min(t_a, t_b) >= t_c:
                    continue
                cand = _update(t_a, t_b, friction[i - 1, j - 1], neg_two, eight, four)
                if cand < t_c:
                    d = t_c - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    elif orthant == 2:
        for i in range(m - 1, -1, -1):
            for j in range(1, n + 1):
                t_a = t[i + 1, j]
                t_b = t[i, j - 1]
                t_c = t[i, j]
                if min(t_a, t_b) >= t_c:
                    continue
                cand = _update(t_a, t_b, friction[i, j - 1], neg_two, eight, four)
                if cand < t_c:
                    d = t_c - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    elif orthant == 3:
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                t_a = t[i + 1, j]
                t_b = t[i, j + 1]
                t_c = t[i, j]
                if min(t_a, t_b) >= t_c:
                    continue
                cand = _update(t_a, t_b, friction[i, j], neg_two, eight, four)
                if cand < t_c:
                    d = t_c - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    else:
        for i in range(1, m + 1):
            for j in range(n - 1, -1, -1):
                t_a = t[i - 1, j]
                t_b = t[i, j + 1]
                t_c = t[i, j]
                if min(t_a, t_b) >= t_c:
                    continue
                cand = _update(t_a, t_b, friction[i - 1, j], neg_two, eight, four)
                if cand < t_c:
                    d = t_c - cand
                    if d > max_change:
                        max_change = d
                    t[i, j] = cand
    return max_change


@njit(cache=True)
def _block_sweep(
    t: np.ndarray, friction: np.ndarray, orthant: int,
    neg_two: float, eight: float, four: float,
    block: int, changed_at: np.ndarray, swept_at: np.ndarray, sweep_idx: int,
) -> tuple[float, int]:
    """`_dense_sweep`, `block x block` vertex block by block, skipping blocks
    that provably cannot change - bit-for-bit identical to `_dense_sweep`
    (same `t`, same returned max_change; locked in by
    `tests/eikonal_kernel_validation`). This is what `solve_eikonal_dense`
    runs; `_dense_sweep` stays as the plain reference kernel.

    Bookkeeping (caller-owned, persistent across one solve's sweeps):
    `changed_at[b]` is the index of the last sweep in which any vertex of
    block `b` changed; `swept_at[o, b]` the index of the last sweep in
    orthant `o` (0-based) that actually processed block `b` (-1 = never).

    Order: blocks in the orthant's own row/column direction, cells within a
    block likewise - still a valid topological order of the sweep's
    dependency graph (each vertex reads only its two upwind neighbours, see
    `_dense_sweep`'s loop-order note), so every update sees the same inputs.

    Skip rule: one orthant sweep is idempotent (re-running it with unchanged
    inputs changes nothing - every vertex is already <= the update from its
    upwind neighbours' final values). A block's inputs in orthant `o` are its
    own vertices plus the halo row/column in its upwind vertical and
    horizontal neighbour blocks (friction is constant within a solve). So if
    none of those three blocks changed since this block was last swept in
    orthant `o`, re-sweeping it is a no-op and is skipped. Changes made
    during that same earlier sweep are already accounted for (upwind blocks
    are processed first), hence `<=`. Later rounds typically only move a
    narrow front through winding channels, so most blocks get skipped.

    Returns `(max_change, n_blocks_swept)`.
    """
    m, n = friction.shape
    if orthant == 1:
        di, dj = -1, -1
    elif orthant == 2:
        di, dj = 1, -1
    elif orthant == 3:
        di, dj = 1, 1
    else:
        di, dj = -1, 1
    fi = -1 if di == -1 else 0
    fj = -1 if dj == -1 else 0
    # Vertex row/col ranges per the module docstring's table.
    row_min, row_end = (1, m + 1) if di == -1 else (0, m)
    col_min, col_end = (1, n + 1) if dj == -1 else (0, n)
    o = orthant - 1
    nbr, nbc = changed_at.shape
    max_change = 0.0
    n_swept = 0
    for bs in range(nbr):
        bi = bs if di == -1 else nbr - 1 - bs
        r_lo = max(bi * block, row_min)
        r_hi = min((bi + 1) * block, row_end)
        if r_lo >= r_hi:
            continue
        for cs in range(nbc):
            bj = cs if dj == -1 else nbc - 1 - cs
            c_lo = max(bj * block, col_min)
            c_hi = min((bj + 1) * block, col_end)
            if c_lo >= c_hi:
                continue
            last = swept_at[o, bi, bj]
            if last >= 0:
                newest = changed_at[bi, bj]
                ui = bi + di
                if 0 <= ui < nbr and changed_at[ui, bj] > newest:
                    newest = changed_at[ui, bj]
                uj = bj + dj
                if 0 <= uj < nbc and changed_at[bi, uj] > newest:
                    newest = changed_at[bi, uj]
                if newest <= last:
                    continue
            swept_at[o, bi, bj] = sweep_idx
            n_swept += 1
            block_changed = False
            for rs in range(r_hi - r_lo):
                i = r_lo + rs if di == -1 else r_hi - 1 - rs
                for cs2 in range(c_hi - c_lo):
                    j = c_lo + cs2 if dj == -1 else c_hi - 1 - cs2
                    t_a = t[i + di, j]
                    t_b = t[i, j + dj]
                    t_c = t[i, j]
                    if min(t_a, t_b) >= t_c:
                        continue
                    cand = _update(t_a, t_b, friction[i + fi, j + fj], neg_two, eight, four)
                    if cand < t_c:
                        d = t_c - cand
                        if d > max_change:
                            max_change = d
                        t[i, j] = cand
                        block_changed = True
            if block_changed:
                changed_at[bi, bj] = sweep_idx
    return max_change, n_swept


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
    candidate/coastline/ocean restriction at all). Sweeps run block-wise
    (`_block_sweep`, `SWEEP_BLOCK_SIZE`), skipping blocks that provably
    cannot change - bit-for-bit identical to plain `_dense_sweep` sweeps.

    `verbose`: print each round's `max_change` and elapsed time - diagnostic
    only, for watching convergence rate without waiting for the whole run
    to finish or hit `max_rounds`.

    `return_diagnostics`: if true, also return a dict with `n_rounds_used`,
    `max_change` (the last round's; `None` under `sweep_budget`, which
    tracks no such thing), `converged` (`max_change <= epsilon`; `None`
    under `sweep_budget`) and `frac_blocks_swept` (share of block visits
    not skipped) - used by `flood_model.py`'s obstacle-coupling
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
    t = np.full((m + 1, n + 1), UNREACHED_T, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values

    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    # _block_sweep's per-block bookkeeping - see its docstring.
    n_block_rows = -(-(m + 1) // SWEEP_BLOCK_SIZE)
    n_block_cols = -(-(n + 1) // SWEEP_BLOCK_SIZE)
    changed_at = np.full((n_block_rows, n_block_cols), -1, dtype=np.int64)
    swept_at = np.full((4, n_block_rows, n_block_cols), -1, dtype=np.int64)
    sweep_count = 0
    blocks_swept = 0

    def sweep(orthant: int) -> float:
        nonlocal sweep_count, blocks_swept
        sweep_count += 1
        change, n_swept = _block_sweep(
            t, friction, orthant, neg_two, eight, four,
            SWEEP_BLOCK_SIZE, changed_at, swept_at, sweep_count,
        )
        blocks_swept += n_swept
        return change

    def frac_blocks_swept() -> float:
        return blocks_swept / max(sweep_count * changed_at.size, 1)

    if sweep_budget is not None:
        for i in range(sweep_budget):
            sweep(_ORTHANT_ORDER[i % 4])
        if return_diagnostics:
            return t, {"n_rounds_used": None, "max_change": None, "converged": None,
                       "frac_blocks_swept": frac_blocks_swept()}
        return t

    if verbose:
        import time
        start = time.perf_counter()

    n_rounds_used = 0
    max_change = 0.0
    for round_idx in range(max_rounds):
        max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            max_change = max(max_change, sweep(orthant))
        n_rounds_used = round_idx + 1
        if verbose:
            print(f"    round {round_idx + 1}: max_change={max_change:.6g}  "
                  f"epsilon={epsilon:.6g}  elapsed={time.perf_counter() - start:.1f}s",
                  flush=True)
        if max_change <= epsilon:
            break
    if return_diagnostics:
        return t, {"n_rounds_used": n_rounds_used, "max_change": max_change, "converged": max_change <= epsilon,
                   "frac_blocks_swept": frac_blocks_swept()}
    return t
