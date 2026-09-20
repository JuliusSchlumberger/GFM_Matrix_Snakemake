# GFM Coastal Flood Model — Calibration & Validation Methodology

**Purpose of this document.** A self-contained, methodology-level description of (a) how the
coastal flood model's independent validation against national hazard maps works, and (b) how
the model's solver parameters are being calibrated against that validation, written for review
by a flood risk domain expert. It intentionally omits software-engineering detail (file paths,
function names, config keys) except where needed to state a method precisely, and points to the
underlying technical documents for anyone who wants that detail:

- `docs/flood_depth_method.md` — the flood model itself (the eikonal-propagation formulation).
- `docs/flood_extent_validation_plan.md` — the validation method's design history and rationale.
- `docs/flood_extent_validation_caveats.md` — every known bias/limitation, per-country and
  pipeline-wide, with mitigations and residual implications.
- `docs/calibration_sweep_plan.md` — the calibration sweep's design history.

This document was produced by a full review of the current implementation (2026-09-18), not
solely from the design documents above — several places where those documents had fallen out of
sync with the actual code were found and corrected as part of producing this review (listed in
§6). Where this document states a fact about "what the pipeline does," that fact was checked
against the running code, not assumed from a design document.

---

## 1. The model being validated, in one paragraph

Coastal flood extent and depth are estimated per raster tile and scenario (a return-period
still-water level, optionally with a sea-level-rise increment) by solving a static
boundary-value problem — the eikonal equation — over a friction (hydraulic resistance) field
derived from land cover, seeded from offshore/coastal water-level forcing points. This is a
steady-state approximation appropriate to a static extreme-value forcing (not a
time-varying storm hydrograph), not a hydrodynamic simulation: it does not resolve
wave run-up/overtopping beyond the prescribed boundary water level, arrival timing, or
structural/hydraulic backwater effects. See `docs/flood_depth_method.md` for the full
mathematical formulation, its justification, and its stated limitations (§2.4 there lists
these explicitly). The Python implementation was validated bit-for-bit identical to the
original reference (Julia) implementation across a real-tile test set (§5 there) for the base
solver; two solver-side extensions beyond that reference implementation (round-based
convergence, and a structural correction described in §4 below) are validated by internal
convergence arguments and limited real-tile comparisons, not by independent-implementation
agreement, since the reference implementation does not have them.

---

## 2. Validation methodology

### 2.1 What is compared, and on what grid

For each country and return period, the model's own continuous water-depth raster is compared
against an independent national flood-hazard dataset — never the model output resampled onto
the benchmark's grid, always the reverse (benchmark data brought onto the model's own grid,
never averaged/interpolated). Rationale: the model carries no real information below its own
resolution (tens of metres); upsampling it to a benchmark's finer resolution (e.g. 5 m) would
manufacture apparent precision without adding real information.

### 2.2 Benchmark data and two structurally different coverage situations

Every benchmark is an independent, third-party national flood-hazard dataset — never something
this pipeline produced. Two structurally different situations occur, and are treated
differently, because treating them the same way would produce a wrong answer in one direction
or the other:

**Partial coverage** (Spain, France, Japan): the benchmark only surveyed specific,
legally-designated or locally-modelled zones — not the whole coastline. A model-wet cell
*outside* every benchmark polygon is genuinely ambiguous: it could be a real false alarm, or
it could be an area nobody ever surveyed. This pipeline's answer is to buffer each benchmark
polygon by a fixed margin (default 2 km) and treat the buffered footprint as the evaluated
domain — cells inside it but outside any polygon count as "benchmark says dry"; cells outside
it entirely are not scored at all. **This buffer is a known, direction-specific source of bias**
(§2.8, item 1) — every false positive inside the buffer but outside a real polygon is an
unresolvable mix of genuine over-prediction and "never surveyed," which inflates the reported
false-alarm rate. Reported alongside every "buffered" result is a second variant,
"model-domain-only" (any cell the model computed at all, no proximity-to-benchmark restriction),
so the sensitivity of the result to this design choice is visible rather than hidden inside one
number.

**National coverage** (Norway, Finland, Denmark): the benchmark's own source assessed the
*entire* coastline — a cell outside every mapped polygon is unambiguously "surveyed, found dry,"
not "unsurveyed." Here a model-wet cell outside the benchmark is a genuine, unambiguous
over-prediction, and the buffering logic above is not just unnecessary but actively wrong to
apply (it would manufacture ambiguity where none exists). These benchmarks are instead compared
directly, per grid cell, over the model's own postprocessing grid, with no buffer at all — plus
an additional country-boundary mask, because a large national benchmark's own rectangular
region of interest can genuinely overlap a neighbouring country's real coastline (confirmed for
both Spain/Portugal and Norway/Sweden/Denmark), and that neighbour's own real flooding must not
be scored as this country's false positive.

Both benchmark **formats** are supported: vector polygon data (every country so far except
Denmark) and continuous-depth raster data (Denmark — see §2.7). Format and coverage type are
independent choices; a raster benchmark could in principle be either partial or national
coverage, though only the national case is implemented so far (§6, item 4).

### 2.3 Evaluation-domain construction (partial-coverage case)

For a partial-coverage benchmark, every one of that benchmark's mapped polygons is buffered
(in real ground distance, not degrees) and overlapping buffers are merged, producing a set of
disjoint "evaluation clusters." Each cluster's own buffered footprint is used directly as the
scored area for whatever model-output tiles/chunks it overlaps. Two exclusions apply within
that footprint before any comparison is scored:

- **Model domain**: only cells the model actually computed something for (its tiles are shaved
  to floodable coastal terrain — inland/high ground is simply absent from the model grid, not
  computed and not zero).
- **Permanent water**: rivers, lakes, and sea cells are excluded entirely — "is this pixel
  flooded" is not a meaningful question over a pixel that is always wet regardless of any storm
  event.

### 2.4 Metrics

Within the evaluation domain, every cell is one of four outcomes (model-wet/benchmark-wet,
model-wet/benchmark-dry, model-dry/benchmark-wet, model-dry/benchmark-dry — the standard
hit/false-alarm/miss/correct-negative confusion categories), weighted by real ground area
(km², accounting for the fact that a raster cell's true ground area varies with latitude), not
by raw cell count. From the area-weighted counts:

| Metric | Formula | Reading |
|---|---|---|
| HR (hit rate) | hits / (hits + misses) | share of benchmark-wet area the model reproduces |
| FAR (false alarm ratio) | false alarms / (hits + false alarms) | share of model-wet area not in the benchmark |
| CSI (critical success index) | hits / (hits + false alarms + misses) | 0 = no skill, 1 = perfect |
| EB (error bias) | false alarms / (false alarms + misses) | > 0.5 over-predicts, < 0.5 under-predicts (this is the requested framing) |
| EB_ratio | false alarms / misses | the equivalent form used in some published literature (Wing et al., 2017) — more legible at extremes; reported alongside EB, not instead of it |
| bias | (hits + false alarms) / (hits + misses) | modelled wet area ÷ benchmark wet area |

A metric with a zero denominator is reported as undefined (not silently zero) — "no benchmark
wet area in this evaluation window" must never be misread as "zero skill." Every metric is also
reported **population-weighted** (weighting each cell by its population rather than its area,
using the same four-outcome classification), since agreement in populated areas is the more
decision-relevant number for an exposure-oriented model — see §2.6.

Every comparison is repeated at four depth thresholds (0.05 m, 0.10 m, 0.25 m, 0.50 m — the
model's water depth must exceed the threshold to count as "wet"), computed in a single pass at
no extra cost since the underlying depth raster is continuous. The primary/headline threshold
is 0.10 m, matching the threshold used elsewhere in the pipeline for population-exposure
accounting.

### 2.5 A required health check, always reported alongside the metrics

Every result row also reports what fraction of the benchmark's own mapped wet area falls
**outside the model's computed domain entirely.** If this is high for a given comparison,
none of that row's other metrics should be quoted as a model result — a high value means the
model and benchmark barely overlap in the first place, for reasons unrelated to model skill
(e.g. the model's tile coverage doesn't extend far enough inland, or a benchmark's survey area
doesn't correspond to modelled terrain at all). Every real result reported so far has this
figure at or near 0%, meaning this has not yet been the limiting factor for any implemented
country — but it must be checked, not assumed, for every new country going forward.

### 2.6 Population-weighted exposure difference

Population data (WorldPop, ~1 km resolution, people-per-cell) is disaggregated onto the model's
much finer grid in proportion to the *area* each fine cell contributes to a given outcome
category, not by any assumption about where within a coarse cell people actually live. This
means a WorldPop cell that is half flooded-per-the-model and half dry has half its population
assigned to each side regardless of true intra-cell settlement pattern — a standard,
unavoidable limitation of area-weighted disaggregation from any coarser population source (not
specific to this pipeline), and the reason population-weighted results should be read as
"population implied by area-proportional disaggregation," not "population confirmed exposed."
From this, the pipeline reports population over-estimated (in false-alarm cells), population
under-estimated (in miss cells), and the net of the two — reporting both sides separately
matters, because a net difference near zero can mean either a genuinely accurate model or a
badly displaced extent with two large errors cancelling out.

### 2.7 Depth-band and continuous-depth comparisons (beyond binary extent)

Where a benchmark carries its own depth information (not just a wet/dry boundary), the model's
continuous depth is compared against it directly, cell by cell, classified as agreeing (model
depth falls within the benchmark's stated band), under-predicting, or over-predicting — the
same over/under framing as EB above, applied to depth rather than extent. Two different benchmark
depth formats have been handled so far, with different technical mechanics but the same
underlying comparison:

- **Categorical depth-class polygons** (France, Japan, Finland): the benchmark assigns each
  surveyed area a discrete depth *class* (e.g. "0.3–0.5 m"); the model's continuous depth is
  checked against that class's own numeric bounds.
- **Continuous depth raster** (Denmark, extent-only so far — see §6, item 4): the benchmark
  itself is a continuous depth value per grid cell, at a resolution finer than the model's own
  grid; classifying "wet" at the benchmark's own native resolution before any resampling is
  necessary, to avoid a coarse destination cell silently missing benchmark flooding that exists
  between the few native-resolution points a naive resample would otherwise sample from.

### 2.8 Known methodological caveats (apply broadly, not just to one country)

The full, current list — including quantitative measurements of each bias's real magnitude
where that has been done — lives in `docs/flood_extent_validation_caveats.md` §1. Summarized:

1. **The evaluation buffer (§2.3) systematically inflates the reported false-alarm rate** for
   every partial-coverage country, because it cannot distinguish "surveyed and dry" from
   "never surveyed." This is not a small-print caveat — it is inherent to why the buffer exists,
   and no buffer width removes it (a narrower buffer trades this bias for more edge-boundary
   disagreement instead). Mitigated by always reporting the buffer-free "model-domain-only"
   variant alongside, so the sensitivity to this choice is visible.
2. Under the current cluster-based implementation, the "model-domain-only" variant is itself
   still implicitly bounded to each benchmark cluster's own local working window, not truly
   "everywhere the model ever computed anything nationally" — a separate, country-wide
   flooded-area/exposed-population figure is reported alongside as a genuine, cluster-independent
   denominator.
3. **Permanent-water exclusion (rivers/lakes/sea) can clip genuine land right at the coastline**,
   because the land/water source used for masking and the benchmark's own coastline were never
   drawn from the same source or vintage. Quantified directly for Norway (§1.3 of the caveats
   doc): switching the water mask to the same terrain-model-derived source the flood model itself
   uses reduced this loss from 45.7% to 30.1% of the near-coast strip, but the remainder — one
   grid cell's width from shore — is a genuine disagreement between two independently-drawn
   coastlines, not fixable by a better mask at any resolution.
4. Benchmark polygons are converted to a wet/dry decision on the model grid via a coverage
   fraction and a 0.5 threshold, which slightly under- and over-counts along polygon edges
   (small relative to item 1).
5. Benchmark geometries are simplified (100 m tolerance) before buffering, for computational
   tractability on some very complex real polygons — small relative to the 2 km buffer width,
   affects only the buffer's own edge, not the underlying benchmark-wet classification.
6. **Every benchmark is itself a model** (its own DEM, its own defences assumptions, its own
   vintage) — "benchmark" means independent reference, not ground truth.
7. **Coastal defence structures (dikes, sea walls, etc.) are handled inconsistently between the
   two sides of the comparison** for most countries: this model applies national flood-protection
   standards only when estimating exposure, never when computing the flood extent itself, while
   several national benchmarks likely reflect real structural defences implicitly. Comparing raw
   extents therefore compares two different protection assumptions wherever real defences exist.
   Finland is the one country where this is no longer a guess: its source data explicitly
   separates "raw depth" from "depth accounting for real declared defences," letting the actual
   size of this effect be measured directly for the areas covered — 0.80% of the raw wet area at
   the primary comparison, in the area checked. Not yet resolved for any other country.
8. **Wave run-up and other momentum-driven effects are not represented** by this model at all
   (§1 above) — Spain's own benchmark methodology explicitly includes wave run-up, so
   under-prediction near exposed, high-energy coastlines is expected there specifically, and is
   not evidence of a model defect.
9. **Return-period definitions are not guaranteed to mean the same event across national
   methodologies** — a "100-year" event in one country's own hazard-mapping convention is not
   necessarily statistically equivalent to this model's own RP100 scenario. This ranges from a
   close, well-defined match (Spain, Denmark) to considerable, currently unresolved uncertainty
   (Japan's "maximum-class" storm-surge scenario type, which may be materially more severe than a
   100-year event; Norway's nearest available native return period, 200 years rather than the
   preferred 100).

### 2.9 Countries implemented so far

All figures below were re-verified directly from the actual output files on 2026-09-18, at the
production configuration's own settings (not the calibration sweep's tile subset — see §3 for
that) — the primary (0.10 m) depth threshold, the country's main/mainland territory where more
than one region is scored (see caveats doc §3.1 for the full per-region breakdown), and the
"buffered" evaluation domain for partial-coverage countries or the only-available "model-only"
domain for national-coverage ones (§2.2).

| Country | Benchmark format | Coverage | Return period used | Comparison type(s) | Real result at primary threshold, most recent production run |
|---|---|---|---|---|---|
| Spain (ESP) | Vector polygons | Partial | RP100 (well-matched) | Extent | HR=0.929, FAR=0.152, CSI=0.796, EB=0.701 (mainland) |
| France (FRA) | Vector polygons | Partial | RP100 (approximate — "at least centennial" class, not an exact value) | Extent + depth bands | HR=0.758, FAR=0.406, CSI=0.499, EB=0.682 (metropole) |
| Norway (NOR) | Vector polygons | National | RP100 in general production use (the same fixed value used for every country) — the calibration sweep specifically overrides this to RP250 for Norway only, as a closer match to its own benchmark's ~200-year class, since the model has no native 200-year scenario (§3.2) | Extent | HR=0.292, FAR=0.256, CSI=0.265, EB=0.124 (mainland, RP100) |
| Japan (JPN) | Vector polygons (19 non-overlapping source files, 12 prefectures) | Partial | RP100 (approximate — benchmark scenario type not confirmed to be a return period at all) | Depth bands only (no separate extent layer needed — depth-band coverage already defines the extent) | pct_agree 3.6–31%, pct_under 67–96% across the 19 areas — a large, one-directional signal, plausibly a return-period/scenario-severity mismatch rather than a model error (not confirmed) |
| Finland (FIN) | Vector polygons (one national-scale source file, chosen over more detailed but redundant local alternatives) | National | RP100 (exact match) | Extent | HR=0.483, FAR=0.466, CSI=0.340, EB=0.450 |
| Denmark (DNK) | Continuous depth raster | National | RP100 (exact match) | Extent only so far | HR=0.818, FAR=0.691, CSI=0.289, EB=0.910 |

Norway's much lower RP100 figures relative to Finland/Denmark (also national-coverage, also
RP100) are plausibly explained, at least in part, by the return-period mismatch itself — Norway's
benchmark represents a rarer (~200-year-class) event than RP100, so a materially lower hit rate
at RP100 is an expected consequence of that mismatch, not necessarily a sign of worse model skill
in Norway specifically. This is a hypothesis consistent with the known caveat (§2.8, item 9), not
independently confirmed by re-running Norway's own comparison at RP250 in production (only the
calibration sweep's own subset-tile run has done that so far — see §2.9's Norway row and §3.2).

Not yet attempted: Germany (existing benchmark data appears largely fluvial, not clearly usable
as a coastal benchmark without further checking; a second dataset is in an unsupported file
format) and Australia (no data downloaded yet).

---

## 3. Calibration methodology

### 3.1 Purpose and design

The flood solver has several internal parameters that were set from judgement/defaults and
never empirically checked against real-world benchmark performance. The calibration effort is a
**one-factor-at-a-time (OFAT) sensitivity screen**: each parameter is varied independently
around its current default, with every other parameter held fixed, and the resulting change (if
any) in the validation metrics above is measured. This is a deliberately cheap first pass
(linear in the number of parameters, not a full combinatorial grid) — appropriate for
identifying which parameters are worth a more careful, joint calibration, not a substitute for
one.

### 3.2 Scope

Re-running the full solver at global scale for every parameter combination is not necessary or
affordable — the sweep is restricted to just the tiles needed to correctly simulate the three
benchmark countries with the most complete/reliable validation data at the time this sweep was
designed: **Spain and France together (at RP100, their benchmarks' native return period) and
Norway separately (at RP250 — the model's own closest available return period to Norway's
preferred ~200-year benchmark class)**. "The tiles needed" is not just the tiles inside these
countries: the solver seeds some tiles from an already-solved neighbouring tile's output rather
than directly from the coast, so a tile-selection procedure computes the full dependency closure
(every tile a benchmark-country tile could possibly depend on, transitively) before any solving
happens. This closure was computed for real, not assumed: **78 tiles for the Spain+France
group, 58 for Norway**, with zero tiles pulled in purely as dependencies beyond the countries'
own real benchmark footprint for either group (i.e. the dependency closure did not turn out to
be materially larger than the country footprint itself, for either group).

### 3.3 Swept parameters and their physical meaning

| Parameter | Physical meaning | Values swept | Note |
|---|---|---|---|
| Friction scale factor | A multiplier on the land-cover-derived hydraulic resistance field — the model's representation of how much vegetation/terrain/built-up land impedes inland flood propagation | 0.5× and 2× the production friction field (relative to 1× baseline) | Directly controls how far/fast floodwater is estimated to propagate inland for a given boundary water level |
| Solver round cap | The maximum number of internal solver iterations allowed before the numerical solution is accepted, whether or not it has fully converged | 4, 8, 20 (baseline: 12) | Controls a precision/cost trade-off in the numerical solve itself, not a physical quantity |
| Solver convergence tolerance | How small the per-iteration change must become before the solver accepts the result as converged (rather than exhausting the round cap above) | 0.01 m and 0.10 m (baseline: 0.03 m) | Also a purely numerical precision/cost trade-off, not physical |
| Obstacle-coupling correction | Whether/how aggressively the structural correction described in `docs/flood_depth_method.md` §4.4a (preventing an illegitimate "shortcut" through high terrain from producing spuriously high water levels beyond it) is applied | disabled entirely; enabled with 1, 3, or 10 correction iterations (baseline: enabled, 5 iterations) | The one solver parameter with a specific, named numerical failure mode motivating it, rather than a general precision knob |
| Exposure threshold | The minimum flood depth for a fine model pixel to count as "exposed" in population/asset accounting | 0.05 m and 0.20 m (baseline: 0.10 m) | **Currently does not affect the validation metrics reported by this sweep at all — see §3.5.** |

The first four are genuine solver-runtime parameters: changing them requires a full new solve of
the eikonal equation per tile. The fifth (exposure threshold) is not read by the solver at all —
it only affects a separate, later step that classifies already-computed depth into
"exposed"/"not exposed" for population and asset accounting.

### 3.4 Isolation and reproducibility mechanism

Every one of the 14 parameter combinations, for each of the 2 country groups (28 combinations
total), writes its own fully isolated copy of every output path the solve/postprocessing/
validation steps produce, so that no combination's output can silently overwrite or blend with
another's or with the production run. The one deliberate exception is the expensive
land-cover/terrain preprocessing step (computing the DEM, land/water mask, and friction
raster for each tile) — this is shared across every parameter combination *within* the same
country group, since none of the five swept parameters change anything that step produces (all
five are read only after preprocessing has already finished); sharing it turns what would be
14× redundant preprocessing work per group into a single run, reused by every combination.

### 3.5 An open methodological issue found during this review

**The exposure-threshold sweep points (0.05 m and 0.20 m) do not currently produce a different
validation result than the baseline run, for a specific, traceable reason: the validation
comparison in §2 reads the model's raw, continuous depth output directly and applies its own
independent set of four depth thresholds (0.05/0.10/0.25/0.50 m, §2.4) — it does not read the
exposure-threshold parameter at all.** That parameter only affects a separate downstream step
(population/asset exposure accounting), which this calibration sweep does not currently run or
compare against any benchmark. Because these two combinations otherwise use identical solver
settings to the baseline, and because preprocessing is shared (§3.4), the only genuinely new
computation these two combinations perform is a repeat of the *entire eikonal solve* for a
parameter that provably cannot change its output — a real, currently-unavoidable inefficiency
under the sweep's present design, not merely a labeling issue.

This is flagged here rather than silently worked around, because it bears directly on how the
calibration results should be read: **if this document is shared alongside a
`calibration_results.csv`/sensitivity figure that includes these two combinations, their
HR/FAR/CSI/EB rows are expected, by construction, to be identical to the baseline row** — that
is not evidence the exposure threshold has no effect on anything, only evidence that this
particular sweep does not currently measure the thing it would affect. Resolving this properly
would mean either (a) removing these two combinations from the extent/depth-band-based
sensitivity comparison entirely, since they add no information there, or (b) extending the
calibration sweep to also re-run the exposure/population-accounting step and compare *that*
against a population-based reference — a different validation question from everything else in
§2, not yet built.

### 3.6 Status at the time of this review

The sweep is run on the HPC cluster sequentially, one full (preprocess → simulate → postprocess
→ validate) chain per combination, and is resumable — a combination already validated is
skipped on a re-run, so an interrupted run (e.g. a dropped remote connection) can be continued
without redoing completed work. As of this review: the friction-scale-factor sweep is the
clearest signal found so far, and in the physically expected direction. At this sweep's own
baseline (on its 78/58-tile subset, not the full production tile set of §2.9's table), both
Spain and France already over-predict relative to their benchmark (EB above 0.5 — ESP 0.717,
FRA 0.652); doubling friction consistently reduces that over-prediction for both (ESP down to
0.489, FRA down to 0.591), and halving friction increases it further (ESP 0.791, FRA 0.717) —
consistent with more friction meaning less-far-reaching predicted flooding, hence fewer false
alarms relative to misses. Norway behaves differently at this same parameter: its baseline
already *under*-predicts slightly (EB 0.335, the opposite direction from Spain/France) and its
EB barely moves with either friction setting (0.328–0.347) — not yet investigated further, but
a genuine difference worth noting rather than assuming Norway would respond the same way. The
solver round cap, convergence tolerance, and obstacle-coupling iteration count show negligible
effect versus baseline for every country wherever they have completed, suggesting the baseline
solver settings were already well-converged for these tiles specifically — not yet confirmed to
generalize beyond them. Several combinations (particularly for Norway) had not yet started or
completed at the time the driving HPC connection was interrupted; the exposure-threshold
combinations (§3.5) had not started at all for any country.

---

## 4. Summary for the reviewer

The validation methodology's single largest, unavoidable source of bias is the survey-coverage
buffer for partial-coverage countries (§2.8, item 1) — it systematically inflates the reported
false-alarm rate, in a way no buffer-width choice removes, and should be kept in mind whenever a
FAR number for Spain, France, or Japan is quoted in isolation. The single largest *unresolved
scientific* question, rather than a known/quantified pipeline limitation, is return-period
comparability (§2.8, item 9) — several countries' benchmarks are being compared against the
model's nearest available return period without a confirmed statistical equivalence, and Japan's
own benchmark scenario type is not confirmed to be a return period at all. The calibration
sweep's clearest finding so far is that solver friction is the dominant lever among the
parameters tested; the sweep also has one confirmed, currently-unresolved design gap (§3.5) that
should be corrected or the affected combinations excluded before the sweep's results are
presented as a complete picture of "what was tested."

---

## 5. Supporting documents (fuller detail, one level down)

- `docs/flood_depth_method.md` — the flood model's mathematical formulation, including the
  Fast Sweeping numerical method and the obstacle-coupling structural correction.
- `docs/flood_extent_validation_plan.md` — the validation method's full design derivation,
  including measured performance/scale numbers and design alternatives that were tried and
  rejected.
- `docs/flood_extent_validation_caveats.md` — the complete, current caveat list, including
  every quantitative measurement of a bias's real magnitude, organized by country plus a
  pipeline-wide section.
- `docs/calibration_sweep_plan.md` — the calibration sweep's design derivation and known risks.

## 6. Corrections made to the supporting documents during this review

The following mismatches between documentation and the actual running code were found and
corrected as part of producing this review, listed here for transparency about what "reviewing
the code" concretely changed:

1. `docs/flood_depth_method.md` did not describe the obstacle-coupling structural correction at
   all, despite it being enabled by default in production and already well-documented in the
   configuration file and solver code. Added as new §4.4a, with a corrected note that this
   correction (and the round-based convergence scheme) were not part of the original
   Python-vs-Julia bit-exact validation, since the reference implementation has neither.
2. The same document's description of the solver's convergence tolerance was stale — it
   described an earlier formula (derived from friction and grid resolution) that was replaced by
   a fixed 0.03 m constant; corrected to describe the current, actual behaviour.
3. Several code comments (in the Snakemake preprocessing rules, the calibration config-builder
   script, the calibration sweep driver script, and one simulation CLI script) stated "12 sweep
   points" / "24 combinations," left over from before the exposure-threshold dimension (§3.3,
   last row) was added; corrected to the current 14/28, and the calibration sweep driver script
   now prints an explicit warning about item 5 below at every run.
4. `docs/calibration_sweep_plan.md` recommended removing four specific dead-code files (which
   imported a module deleted in an earlier rewrite); confirmed those files no longer exist in the
   repository, and updated the note to reflect that this cleanup is already done.
5. The exposure-threshold sweep dimension's disconnect from the validation metrics (§3.5) was
   found during this review — not a stale-documentation issue, but a real gap between what the
   sweep's own design intent and what it currently measures, not previously identified.
