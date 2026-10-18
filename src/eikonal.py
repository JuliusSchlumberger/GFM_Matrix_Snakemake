"""Fast Sweeping Method (Zhao, 2005) for the eikonal equation on a dense grid.

Port of Aqueduct's Julia core (`core/src/core.jl`, Eikonal.jl `sweep!`); every
cell (land, ocean, lake) participates. Deviation from Julia: unseeded t=+99,
not 0. Staggered grid as Eikonal.jl: `t` on vertices, (m+1, n+1) for an (m, n)
friction grid; each orthant feeds a vertex from one corner cell (0-indexed):

| orthant | row range | col range | t-neighbors used      | cell used   |
|---------|-----------|-----------|-----------------------|-------------|
| 1       | [1, m]    | [1, n]    | (i-1,j), (i,j-1)      | (i-1, j-1)  |
| 2       | [0, m-1]  | [1, n]    | (i+1,j), (i,j-1)      | (i,   j-1)  |
| 3       | [0, m-1]  | [0, n-1]  | (i+1,j), (i,j+1)      | (i,   j)    |
| 4       | [1, m]    | [0, n-1]  | (i-1,j), (i,j+1)      | (i-1, j)    |

Cell (r, c) reads vertex (r+1, c+1) (Julia `waterlevel = -t[2:end, 2:end]`),
but seeds write at vertex (r, c): Aqueduct's own asymmetry (`core.jl`
`solver.t[I] = -initial[I]`).
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Julia's Gray-code orthant order, in this module's numbering.
_ORTHANT_ORDER = (1, 4, 3, 2)

UNREACHED_T = 99.0

# Any value is bit-identical; 64 gave 1.2-5.8x over dense sweeps on calibration tiles.
SWEEP_BLOCK_SIZE = 64


@njit(cache=True)
def _update(t_a: float, t_b: float, v: float, neg_two: float, eight: float, four: float) -> float:
    # Eikonal.jl `update` (N=2), falling back to t + v when the quadratic is
    # not causal. Constants are pre-cast to t's dtype: Julia's Float32
    # discriminant is a catastrophic cancellation (O(100) terms, O(1e-5)
    # difference), so matching it needs the same precision and formula.
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
    # Reference kernel. Row-outer order (vs Julia's column-outer) is
    # bit-identical: each vertex reads only its two upwind neighbours, visited
    # first in both orders. Early-out is exact: _update >= min(t_a, t_b).
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
    # _dense_sweep block by block; bit-identical (tests/eikonal_kernel_validation).
    # changed_at[b]: last sweep that changed block b; swept_at[o, b]: last
    # orthant-o sweep that processed b (-1 = never). An orthant sweep is
    # idempotent and a block depends only on itself plus its upwind vertical and
    # horizontal neighbours, so it is skipped if none changed since (<=, since
    # upwind blocks run first).
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

    Returns `t` with shape (m+1, n+1); cell (r, c) is at vertex (r+1, c+1).
    seed_values: initial `t` (Aqueduct seeds -waterlevel); seeds are not frozen.
    epsilon: max per-round change, as `core.jl`'s
    `minimum(friction) / (resolution * 10)`.
    sweep_budget: run exactly this many sweeps in `_ORTHANT_ORDER`, ignoring
    epsilon/max_rounds; 3 reproduces Aqueduct's Eikonal.jl runtime behaviour.
    return_diagnostics: also return n_rounds_used, max_change, converged (None
    under sweep_budget) and frac_blocks_swept.
    """
    m, n = friction.shape
    dtype = friction.dtype
    # Unseeded t=+99 (waterlevel -99 m), not Julia's t=0: 0 m is MSL, so
    # unreached cells below 0 m would read as flooded. Mirrors the +99 m dry
    # sentinel (opposite sign) and fits int16-cm. t only decreases, so any
    # real seed path overwrites it.
    t = np.full((m + 1, n + 1), UNREACHED_T, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values

    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

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
