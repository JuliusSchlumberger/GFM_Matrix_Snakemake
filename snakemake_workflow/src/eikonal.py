"""Fast Sweeping Method (Zhao, 2005) for the eikonal equation, dense domain.

Ports the numerical method used by Aqueduct's Julia core (`core/src/core.jl`,
via `Eikonal.jl`'s `FastSweeping`/`sweep!`), operating on the full dense
raster grid - every cell (land, ocean, lake) participates, exactly like
Julia's own solve, via plain array-index arithmetic (no neighbor-index
tables). Validated bit-for-bit identical to real Aqueduct output across 26
real tiles spanning ~9M-135M cells each (100.000% Jaccard, 0.0m
RMSE/mean-error/90th-percentile/max-diff on every one - see
`docs/python_vs_julia_qa.md`).

An earlier version of this module instead ran on a *compacted* domain
(skipping non-candidate cells via a 1-D index of just the "relevant" ones,
to save memory) - removed once the dense approach proved both simpler and
faster in practice: real coastal tiles don't have much genuinely irrelevant
area to skip, so the compaction bookkeeping (extra `O(N)` integer
neighbor-index arrays) cost more than it saved, and the compacted domain's
real topological gaps (excluded non-candidate land) broke the fixed
3-sweep production shortcut's correctness in a way the dense domain,
having no such gaps, doesn't. See git history for the removed
`CompactGrid`/`VertexDomain`/`build_vertex_domain`/`solve_eikonal` code if
ever needed for reference.

This module replicates Eikonal.jl's exact staggered-grid convention (arrival
time `t` on grid VERTICES, one larger per axis than the `friction`/cell
grid, with each of the 4 sweep directions - "orthants" - using a different
single corner cell to feed a given vertex's update), derived line-by-line
from the installed `Eikonal.jl` source
(`~/.julia/packages/Eikonal/*/src/Eikonal.jl`), not guessed from the paper.

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
# module's own orthant numbering (see module docstring table) - NOT
# (1, 2, 3, 4). Confirmed bit-for-bit exact (100.0000% of 130,225,240 cells,
# all 3 sweeps) against a live Julia run's per-sweep checkpoints on tile
# 2660; the naive (1, 2, 3, 4) order only matched ~14-29% of cells by
# sweep 2/3 on that same tile. See `solve_eikonal_dense`'s use of this
# constant for the full story.
_ORTHANT_ORDER = (1, 4, 3, 2)


@njit(cache=True)
def _update(t_a: float, t_b: float, v: float, neg_two: float, eight: float, four: float) -> float:
    """Zhao's upwind quadratic update from two orthogonal neighbor times.

    Mirrors Eikonal.jl's `update` for N=2, including its fallback to the
    simpler 1-D characteristic (`t + v`) when the 2-D quadratic solution
    isn't causal (`t <= max(t_a, t_b)`).

    `neg_two`/`eight`/`four` are `-2`/`8`/`4` pre-cast to `t_a`'s own dtype
    by the caller (float32 for real tiles) - CONFIRMED, via live
    instrumentation of a genuinely-working Eikonal.jl build (second
    machine, Eikonal v0.1.1), to be necessary for exactly matching Julia's
    real output, correcting an earlier wrong belief recorded in this
    function's history.

    Earlier reasoning (WRONG, now retracted): this function used bare
    Python float literals (`2.0`, `4.0`), reasoning that Julia's own
    `a = N` (a bare `Int`) combined with `Float32` promotes to `Float64`
    under Julia's type-promotion rules "just like numpy's `Int64 *
    Float32 -> Float64`" - so the quadratic formula would be Float64
    internally in both languages, truncating to Float32 only once at the
    final array write. That numpy-derived assumption about Julia's
    promotion rule was simply incorrect: Julia promotes `Integer *
    AbstractFloat` to the FLOAT type's own precision, not up to Float64
    regardless of integer width. A cross-machine live trace proved this
    directly - asked the second machine to run a standalone function
    mirroring Eikonal.jl's literal `update` body with `@printf("%s",
    typeof(...))` on every intermediate, for a specific real (t_a, t_b, v)
    triple pulled from a shared deterministic test case. Julia reported
    `b`, `c`, AND `Δ` all as `Float32` - not Float64. Because `b^2` and
    `4*a*c` are both O(100-200) magnitude while their difference Δ is
    genuinely O(1e-5) at real friction scales (~1e-4), computing Δ in
    native Float32 is a ~7-order-of-magnitude catastrophic cancellation:
    Float32's own rounding error in computing `b^2` alone (~197 *
    2^-24 ≈ 2.3e-5) is LARGER than Δ's true value. Julia's real Δ is
    therefore dominated by Float32 rounding noise at this scale, not a
    lightly-rounded version of the true answer - and reproducing that
    exact noise (not a more "correct"/higher-precision value) is what's
    required to match Julia's actual output. Replicating Julia's EXACT
    literal formula structure (not the algebraically-simplified
    `disc2 = 2v^2-(t_a-t_b)^2` form tried in an earlier, unsuccessful
    attempt at this same fix - mathematically equivalent but rounds
    differently in float32) in native float32 reproduced Julia's live
    trace bit-for-bit: `Δ=1.5258789e-05` (exactly 2^-16, matching Julia's
    reported value exactly), `cand=-3.5119715`, final result
    `-3.5120673` - all matching Julia's reported values to float32
    precision on the shared test cell.

    This also explains the DIRECTION of the small systematic bias found
    between this port and real Aqueduct output (Python running slightly
    deeper/higher than Julia, compounding with path length): when Float32
    noise pushes Julia's computed Δ negative where the true Δ is a tiny
    positive number, Julia spuriously REJECTS the quadratic branch and
    falls back to the cruder, always-shallower 1-D form - while wrongly
    ACCEPTING an invalid quadratic branch near the Δ≈0 threshold barely
    matters (the erroneous candidate is close to the 1-D fallbacks anyway
    and loses the final `min` regardless). That asymmetry turns even
    symmetric Float32 rounding noise in Δ into a systematic tilt toward
    Julia being shallower - i.e. this port, once accurately reproducing
    Julia's real (imprecise) Float32 Δ, should be shallower too, matching
    real Aqueduct output more closely than the higher-precision Float64 Δ
    this function used to compute.
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
) -> np.ndarray:
    """Solve the eikonal equation on the full dense grid via Fast Sweeping.

    Every cell participates, exactly like Eikonal.jl's own domain (no
    candidate/coastline/ocean restriction at all).

    `verbose`: print each round's `max_change` and elapsed time - diagnostic
    only, for watching the convergence RATE on tiles where full convergence
    takes far longer than the usual few rounds (e.g. tile 2660: >150x the
    3-sweep pass's runtime with no end in sight after 30 minutes), without
    waiting for the whole run to finish or hit `max_rounds`.

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
            stop - ignoring `epsilon`/`max_rounds` entirely. `3` matches
            Eikonal.jl's actual runtime behavior in Aqueduct's usage
            (confirmed via live instrumentation on a second machine, on a
            300x300 synthetic, two real 34M/51M-cell tiles, and a full-tile
            130M-cell cross-machine diff) - this is the validated,
            production-matching configuration. `None` iterates to full
            numerical convergence instead - much slower (100x+ on complex
            tiles) and NOT the validated/matching configuration; useful
            only for diagnostics.

    Returns:
        `t`, shape `(m+1, n+1)` - read cell `(r, c)`'s result at vertex
        `(r+1, c+1)`.
    """
    m, n = friction.shape
    dtype = friction.dtype
    t = np.zeros((m + 1, n + 1), dtype=dtype)
    t[seed_rows, seed_cols] = seed_values

    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    if sweep_budget is not None:
        for i in range(sweep_budget):
            _dense_sweep(t, friction, _ORTHANT_ORDER[i % 4], neg_two, eight, four)
        return t

    if verbose:
        import time
        start = time.perf_counter()

    for round_idx in range(max_rounds):
        max_change = 0.0
        for orthant in _ORTHANT_ORDER:
            max_change = max(max_change, _dense_sweep(t, friction, orthant, neg_two, eight, four))
        if verbose:
            print(f"    round {round_idx + 1}: max_change={max_change:.6g}  "
                  f"epsilon={epsilon:.6g}  elapsed={time.perf_counter() - start:.1f}s",
                  flush=True)
        if max_change <= epsilon:
            break
    return t
