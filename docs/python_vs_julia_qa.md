# Python port vs. Julia reference model: three questions for the supervisor update

*Presentation-support notes, not a methods section — written to be read or
adapted into slides.*

---

## 1. How did we get the Python code fast?

Two separate things contributed, and it's worth keeping them separate: the
**algorithmic design of the solver**, and the **execution/deployment model**.
Correctness fixes (Q3) are a *third*, independent axis — they didn't
themselves speed anything up.

### Algorithmic choices

- **Dense array indexing, not a compacted domain.** A tile is a rectangular
  grid of cells; a "dense domain" keeps *every* cell (ocean, lakes, land
  that could never flood — all of it) as one plain 2-D array, where a
  neighbour is just a `±1` array-index step — this is what Julia does. An
  earlier version of the Python port instead used a **compacted domain**:
  discard cells that clearly can't matter and keep only a 1-D list of the
  survivors, meant to save memory — but since that breaks the regular grid
  shape, every kept cell then needs its own explicit up/down/left/right
  neighbour-lookup table instead of simple index arithmetic. In practice,
  real coastal tiles don't have much genuinely irrelevant area to skip, so
  that bookkeeping overhead (extra `O(N)` integer index arrays) cost more
  than it saved. Switching to the plain dense grid removed that overhead
  entirely and turned out to be both simpler and faster.
- **Reusing Julia's own "exactly 3 sweeps" shortcut.** Confirmed via live
  instrumentation of a real, working Julia installation (see Q2): in this
  application's parameter regime, Julia's own production code already
  stops after exactly 3 of the 4 directional sweeps, rather than iterating
  the Fast Sweeping Method to full numerical convergence. Both
  implementations now exploit this — it's not a Python-only shortcut — but
  it's a large part of why the *method as actually deployed* is fast at
  all (full convergence on a large, complex tile can take 100×+ longer,
  as measured directly on tile 2660 during this investigation).
- **Numba JIT-compiled inner loop.** Plain Python is *interpreted* — every
  operation carries generic bookkeeping overhead (type checks, dynamic
  lookups) that's invisible in everyday code but dominates in a tight loop
  run hundreds of millions of times (the sweep visits every cell in the
  grid, up to ~130 million of them, doing one small fixed calculation
  each). Numba compiles a marked function (`@njit`) to real machine code
  the first time it's called, specialized to the exact data types it sees
  — the same class of low-level code a C or Julia compiler would produce
  — so the readable Python loop in `eikonal.py`'s `_dense_sweep` runs at
  roughly compiled-language speed instead of interpreted speed.
  `cache=True` additionally saves that compiled machine code to disk, so
  later runs reuse it instead of recompiling from scratch.

### Execution/deployment model

- **Process lifetime.** Julia's compiled `aqueduct.exe` is launched as a
  fresh operating-system process for every single (tile, return period,
  SLR) scenario, paying a roughly constant ~5s JIT/compilation tax on every
  invocation (confirmed directly by running the same tile twice in one
  session — a small tile's 5.5s cold run drops to 0.1s warm) regardless of
  tile size. For small tiles this constant tax *is* almost the whole story;
  for large tiles it shrinks to a minor share of the total (see below for
  what actually dominates there). The Python path can compile its hot loop
  once and reuse it across many scenarios within one longer-lived
  process/session, so it never pays this repeated tax at all.

### Measured result — corrected, and genuinely mixed

**An earlier version of this section claimed a 4×–31× Python speedup. That
number was wrong** — it compared Python's *sweep-only* time against
Julia's *whole-pipeline* time (an apples-to-pears mistake caught during
review). Once both sides are measured over the same pipeline scope (read
inputs → DEM/mask prep → coastline mask → boundary KNN+IDW → sweep →
connected-component pruning, excluding final raster write on both sides),
the picture is **honestly mixed, not a clean win either way**:

| comparison | Julia | Python | result |
|---|---|---|---|
| Aggregate mean (405 Julia runs/9 tiles/all scenarios vs. 20 Python runs/RP100_SLR_0) | 40.0s | 26.8s | Python ≈1.5× faster |
| 5 tiles present in both datasets, paired | 15–56s | 13–34s | Python faster on 4/5 (1.6×–4.3×), **slower on 1/5** (1.16×) |
| Single-tile (2660), stage-matched, one machine pair | 33.8s | 43.5s | **Julia ≈1.3× faster** |

The reason these disagree: Julia's numbers come from at least two different
machines across this investigation (the historical `Europe_West.log` batch
— likely an HPC node — and a separate live diagnostic session), and
neither matches the machine Python was measured on. I/O-bound stages
especially may reflect disk/filesystem speed rather than language or
algorithm. **The one comparison not confounded this way is the per-stage
breakdown on the single tile where both sides were measured stage-by-stage**
(tile 2660): there, **Julia's sweep is ~2.6× faster than Python's**, and
**Python's connected-component pruning is ~2× faster than Julia's** (this
port deliberately avoids Julia's `component_indices` approach, flagged in
`core.jl`'s own comments as memory-heavy — see Q3). That stage-level split
is the most defensible speed claim available; a single aggregate "Python is
Nx faster/slower" headline should not be presented without this caveat.

See `julia_vs_python_timing.html` and `python_full_pipeline_timing.csv` /
`julia_timing_vs_size.csv` for the full data behind this table.

---

## 2. How did we test that the Python code does the same thing as Julia?

Testing happened in layers, each one only trusted once the previous layer
passed — going from "looks statistically similar" to "is bit-for-bit
identical, and we know exactly why."

1. **Output-level comparison against real production data.** For a growing
   set of real tiles, ran both implementations on identical inputs and
   compared final flood extent (Jaccard index) and depth (RMSE, mean
   signed error, percentile and max differences). This is what first
   surfaced several real, fixable bugs early on (e.g. an incorrect
   cell-vs-vertex grid convention, and excluding open ocean from the solve
   domain when Julia's reference does not) — each fix measurably closed
   the gap across the whole tile set.
2. **Spatial diagnostics to localize *where* mismatches concentrated.**
   Binning mismatched cells by distance-to-coast, depth magnitude, and
   connected-component size repeatedly pointed at specific mechanisms
   (e.g. "extra" cells were concentrated at the immediate coastline, not
   scattered randomly) rather than leaving us guessing.
3. **Two tiles resisted every fix** (2660, 26729 — both large, complex
   coastlines) — up to 96–97% Jaccard rather than the 99.99%+ seen
   elsewhere. Per-round convergence tracing showed *why*: unlike
   well-behaved tiles, which converge smoothly, these two **plateau and
   oscillate** rather than converging — the signature of near-tied
   competing flood paths in the interior, sensitive to any infinitesimal
   arithmetic difference between implementations.
4. **Live cross-machine comparison against a real, working Julia
   installation** (this was the decisive step — source-reading alone had
   already led us to one *wrong* conclusion, see Q3). On a second machine
   with a genuine Aqueduct build, we:
   - traced a single update computation with inputs confirmed
     bit-identical between both implementations, and diffed every
     intermediate value Julia actually computed against Python's — this
     pinpointed a precision bug in one arithmetic step (Q3).
   - later, for the two resistant tiles, requested a **full-tile dump of
     Julia's actual internal solver state** (raw arrays, ~520 MB each)
     after each of the 3 production sweeps on a real 130-million-cell
     tile, and diffed it cell-by-cell against Python's own internal state
     at the same checkpoints — not just the final output. This is what
     found the second, larger bug (Q3): sweep 1 matched **100.000% of 130
     million cells exactly**, but sweep 2 diverged in a way that
     structurally couldn't be a rounding difference — it turned out
     Python was running the wrong *sweep order*.
5. **Full re-validation after both fixes**, across **26 real tiles**
   spanning ~9 million to ~135 million cells each (the original problem
   tiles plus 20 more, randomly sampled): every single tile now shows
   **100.000% Jaccard, zero unique flooded cells on either side, and
   0.0 m RMSE/mean error/90th-percentile/max depth difference** — an exact
   match, not merely a close one.

**In short**: we didn't just compare final outputs statistically — once
statistical comparison found something to explain, we went one level
deeper each time (spatial pattern → convergence behaviour → live
intermediate arithmetic → full internal solver state on a real production
tile) until every remaining discrepancy had a concrete, verified
explanation rather than being written off as "probably just numerical
noise."

---

## 3. Are there crucial differences between the Python and Julia code?

**As currently deployed (dense domain, exactly 3 sweeps): no — verified
bit-for-bit identical on every one of 26 real tiles tested (RP100_SLR_0).**
That's the headline. Two real differences *did* exist and were found and
fixed during this work; it's worth presenting them as resolved history
rather than current risk, plus a couple of caveats on scope.

### Found and fixed (no longer present)

1. **A floating-point precision bug in the core update formula.** The port
   originally computed one internal quantity (the eikonal update's
   discriminant) in a higher precision than Julia actually uses. This
   sounds like it should make Python *more* accurate — but the goal is to
   match Julia's actual (imprecise) arithmetic, not to be more "correct"
   than it, since we're replicating a specific reference model. Root cause
   was independently re-derived from source and then confirmed live
   against Julia's real output before fixing.
2. **A sweep-order bug.** Julia visits its 4 sweep directions in a
   specific order; the port had two of those four swapped. This one was
   subtle enough that reading Julia's source code carefully *twice* still
   missed it — it only surfaced once we diffed real internal solver state,
   cell by cell, against a live Julia run on an actual production tile.

Fixing the first alone measurably helped, but left the two hardest tiles
mostly unresolved (~+0.1 Jaccard points) — for a while, this looked like an
**intrinsic, unfixable property** of replicating an unstable, oscillating
numerical method. That conclusion was wrong: fixing the second bug (which
we only found afterward) resolved those two tiles completely, alongside
everything else.

### Structural differences that remain (by design, not bugs)

- **Process model.** Julia runs as a compiled, standalone executable
  invoked fresh per scenario; Python runs as a JIT-compiled function within
  a longer-lived process. This affects speed (Q1) and deployment
  (portability to HPC, where the Julia toolchain cannot currently be
  installed), not correctness.
- **Both implementations deliberately stop after exactly 3 sweeps** rather
  than fully converging the underlying Fast Sweeping Method, in production.
  This is a shared property of *how the model is actually run*, confirmed
  to be Julia's own real behaviour — not a Python shortcut — but worth
  flagging as a characteristic of the deployed method itself: on very
  complex tiles, the fully-converged mathematical solution could in
  principle still differ somewhat from what either implementation reports
  today.

### Scope caveat

Bit-exact validation has focused on the **RP100/SLR_0** scenario across 26
tiles (plus small synthetic cross-machine test cases covering the full
sweep cycle). We have not yet exhaustively re-run the bit-exact check
across every one of the ~20 return-period/SLR combinations on every tile —
there's no specific reason to expect a difference there (the fixes are in
scenario-independent code), but it hasn't been explicitly re-confirmed at
that scale yet.
