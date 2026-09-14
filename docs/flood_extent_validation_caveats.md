 # Coastal flood-extent validation — caveats, mitigations, and residual bias

**Companion to** `docs/flood_extent_validation_plan.md` (the method/implementation design). This
document is about **trust**, not mechanics: what is uncertain or limited about each benchmark
dataset, what this pipeline does about it, and — critically — what bias that mitigation itself
introduces. Every metric this pipeline reports (HR/FAR/CSI/EB, population-weighted or not) should
be read alongside the relevant caveats below, not as a bare number.

Two kinds of caveat are kept separate throughout:

- **Data caveats** — properties of the benchmark itself we cannot change (coverage gaps, vague
  return-period definitions, methodology differences from our model).
- **Methodological caveats** — properties of *how this pipeline scores agreement*, which we chose
  and could in principle change, but which currently bias the numbers in a specific, describable
  direction. These apply to every country, not just one.

---

## 1. Methodological caveats (apply to every country)

### 1.1 The buffer includes unsurveyed area, not confirmed-dry area — inflates FAR

This is the most important bias in the whole pipeline, and it runs in a specific, textbook
direction. It applies to every `coverage: partial` benchmark (Spain, France) — countries whose
benchmark instead assessed their ENTIRE coastline (`coverage: national`, e.g. Norway, §2.3) don't
have this ambiguity in the first place and use a different comparison path entirely
(`validate_country_national_coverage`) with no buffer at all.

**Mechanism.** The evaluation domain's `buffered` variant scores every model cell within
`buffer_km` (default 2 km) of a benchmark polygon, not just cells inside the polygon. The buffer
exists so that a genuinely flooded coastal cell just outside a survey boundary — where the model
and benchmark plausibly agree but the polygon edge falls a few pixels short — isn't wrongly
counted as disagreement. But the buffer cannot distinguish "surveyed and found dry" from "never
surveyed at all." A benchmark polygon's edge is where the survey **stopped**, not necessarily
where flooding **stops**.

**Consequence.** If the model shows flooding just outside a benchmark polygon, but still inside
the buffer, this pipeline scores it as a false positive (`model_wet & ~benchmark_wet` → FP,
"model over-predicts"). But the true state of that specific cell is *unknown* — it could be a
genuine model error, or it could be a real flood the hazard-map surveyor simply never looked at
because it fell outside the legally-designated study area. Every FP inside the buffer but outside
any actual polygon is a mix of "real over-prediction" and "unsurveyed, unknowable" — and this
pipeline has no way to tell the two apart. **The result: FAR (and EB, and CSI to a lesser degree)
are systematically biased toward making the model look like it over-predicts more than it actually
does.** This is not a small-print caveat — it is baked into the buffer's entire purpose, and there
is no buffer width that removes it (a smaller buffer just trades this bias for more edge-boundary
false disagreement instead).

**What this pipeline already does about it:**
- Reports the `model_only` domain variant (`depth != -9999` minus permanent water, no polygon-shape
  or buffer restriction at all) alongside `buffered`, in every metrics row. Comparing the two shows
  how much FAR moves when the buffer is removed entirely — a large gap between `buffered` FAR and
  `model_only` FAR is itself evidence of how much this bias is doing to the number, per country/run.
- Reports `benchmark_wet_outside_model_domain_pct` as an explicit health check — if most of the
  benchmark itself falls outside the model's own computed domain, no metric in that row should be
  quoted at all (plan doc §3/§5.4).

**What it does *not* yet do, worth considering as a follow-on:**
- No diagnostic currently reports *how much of the buffered domain is actually "padding"* — i.e.
  the fraction of buffered-domain area that is not, and is not near, any real digitized benchmark
  polygon. A high padding fraction would be a direct, per-run measure of how exposed a given
  country's FAR number is to this bias, complementing the buffered-vs-model_only comparison above.
- Buffer width (`buffer_km`) is currently a single global default (2 km) for every country. Since
  this bias scales with buffer width, a country whose benchmark surveys very narrow coastal strips
  (leaving more "unsurveyed" area within a fixed buffer) will be more exposed to it than one with
  broad polygons — worth revisiting per-country once more than one country's real numbers exist to
  compare.

### 1.2 `model_only` is not actually country-wide, under the per-cluster redesign

The evaluation-cluster redesign (plan doc §4.6a) processes each benchmark cluster independently,
reading only that cluster's own local bounding box (the buffered polygon's extent) from the merged
model output. This means `model_only` — intended to represent "the model's domain, with no
benchmark-proximity restriction at all" — is in practice still implicitly bounded to each cluster's
own local working window. It is *not* "everywhere the model ever computed anything in the
country," it is "everywhere the model computed something, within reach of a benchmark polygon
anyway." This narrows the gap between `buffered` and `model_only` somewhat versus what the
original (pre-redesign) per-block design would have shown, and should be kept in mind when
interpreting how large that gap is. It does not reintroduce §1.1's bias, but it does mean
`model_only`'s own absolute area figures understate "the model's true total domain size."

**Mitigation added 2026-09, moved into the general pipeline 2026-09.** Two new CSV columns,
`model_total_wet_km2` and `pop_model_total`, report the model's own flooded area and exposed
population, COUNTRY-WIDE, at the single fixed exposure threshold (`exposure.exceedance_threshold_m`)
— a genuine denominator to compare `model_wet_km2`/`pop_model` (still cluster-window-scoped)
against, independent of any benchmark cluster's local window entirely.

The first version of this computed these two numbers per named `regions:` bbox
(`validate_country.py`'s own `_compute_region_model_totals`, reading merged waterdepth chunks
directly) — but that both duplicated work the general exposure-analysis pipeline already does, and
needed its own fix once a real correctness gap surfaced: a `regions:` bbox is just a rectangle, and
Spain's `mainland` bbox genuinely overlaps Portugal, southern France, and the Morocco/Gibraltar
coast, so without an extra WRI-geogunit-based country mask, those neighbours' flooded
area/population would have silently leaked into Spain's totals.

Rather than carry that extra masking logic in the validation feature, the computation now lives in
**`analysis/compute_flood_totals.py`** (new pipeline step, `analysis.compute_flood_totals` switch,
wired into `run_analysis.py`), which streams every populated chunk once (same architecture as
`compute_exposure_analysis.py`) and aggregates per country via the exact same WRI geogunit-107
raster + FLOPROS ISO lookup already used for per-country EAI aggregation there — exact country
membership, not a bbox approximation, with no extra masking step needed. It writes one CSV per
country (`flood_totals_{ISO}.csv`, one row per modelled RP/SLR, `validation.flood_totals_dir`) that
`validate_country.py` now just reads (`_read_flood_totals`) instead of recomputing.

Two consequences worth knowing: (1) these two columns are now **country-wide, not per-region** —
every region-row of a country's metrics CSV (e.g. both `mainland` and `canary_islands`) carries the
SAME two values, since WRI geogunit aggregation only resolves to country level; (2) they are only
populated at the primary threshold (matching the single fixed `exposure.exceedance_threshold_m` the
underlying pipeline output is built at) — NaN at the metrics CSV's other threshold rows in the
0.05/0.10/0.25/0.50 m sweep, since the number wouldn't apply to them. Both are NaN, not a silent
0.0, until `analysis/run_analysis.py` has actually been (re-)run with `compute_flood_totals` enabled.

### 1.3 Permanent-water masking can clip real coastline at the land/sea boundary — quantified, partially mitigated (2026-09)

Rivers/lakes/sea are excluded from scoring via `validation.permanent_water_source`/
`permanent_water_codes`, reprojected via nearest-neighbour onto the model's much finer grid. Right
at a coastline, a source cell can straddle land that both diagrams reach differently — if that
cell is classified as permanent water, every model cell nearest to it inherits that label too,
even genuine low-lying land cells that both the model and the benchmark might consider
flooded/floodable.

**Was Copernicus Global Land Cover (100 m, codes 80/200) through 2026-09.** Switched to
**DeltaDTM's own native land/ocean/lake/river mask** (codes 1/2/3, ~25-31 m ground cell in
Norway - 4-8x finer in area than Copernicus at every latitude checked, since DeltaDTM is the same
DEM the model itself is built from, so there's no second, independently-registered dataset in the
loop). Zero new code needed - `read_permanent_water_mask` was already fully generic over source;
this is a two-key config change (`permanent_water_source: "deltadtm_mask"`,
`permanent_water_codes: [1, 2, 3]`).

**Now quantified, not just flagged as unmeasured** (real measurement across 5 Norwegian AOIs,
`snakemake_workflow/tests/norway_diagnostics/10_deltadtm_mask_vs_landuse.py` and
`11_permwater_shoreline_profile.py`):

- Coastal-strip masking loss (genuine land wrongly excluded, restricted to cells within 240 m of
  the sea): Copernicus **45.7%** → DeltaDTM **30.1%** — roughly a third of the loss recovered.
- A shoreline-distance breakdown pinpoints why: in the 60-240 m band (the coarse-cell-aliasing
  signature this caveat originally described - one Copernicus cell's inland reach), DeltaDTM
  cuts the loss from 21.9% to 1.9%, an 11x reduction - the switch genuinely eliminates the
  mechanism this caveat is about.
- ~21% of the original "51% of the thin strip" headline figure turned out to be *correct*
  inland-lake masking, not coastline clipping at all - it was never recoverable and shouldn't be
  read as part of this caveat.

**Standing limitation, not fixed by this switch and not currently planned to be:** the dominant
remaining loss sits exactly one grid cell (~30 m) from shore - the median lost cell in the real
measurement was 31 m from the sea, i.e. one validation cell wide. This is a **waterline-definition
disagreement**, not a resolution problem: DeltaDTM's own water classes derive from ESA WorldCover's
~2020 optical imagery, while a benchmark's own coastline comes from its own national
survey/methodology, so the two were never drawn from the same source or vintage in the first
place. No permanent-water mask, at any resolution, can reconcile two independently-drawn
coastlines down to the same pixel - masking only decides what gets excluded from scoring, it
cannot align two *other* geometries (the model's DEM-derived coastline and the benchmark
polygon's own edge) to each other. Concretely, this means: right at the immediate shoreline
(within ~1 cell), expect some irreducible disagreement between model and benchmark that is not a
model error, not a benchmark error, and not something a better permanent-water source can remove -
it is the two datasets' own boundaries not agreeing on sub-cell scale. Two real ways to actually
address it, neither attempted: (i) a "fuzzy"/tolerance-based scoring rule (count a nearby hit
within N metres as agreement instead of requiring exact-cell overlap - already floated as a
possible extension, plan doc §9, not built), or (ii) sourcing a benchmark-consistent coastline
product per country (e.g. Norway's own Kartverket land-cover data, the same agency that made the
storm-surge benchmark, so its coastline would be internally consistent with the benchmark by
construction) - a real data-acquisition task, not a config change.
- One small tradeoff in the wrong direction: sea-masking coverage itself is very slightly lower
  with DeltaDTM (98.06% → 97.91% of the raw benchmark polygon's sea area) - unmasked sea is
  arguably the worse error class for validation (it reads as spurious "agree" in both model and
  benchmark, inflating apparent skill), though the magnitude here is small.
- Combining both sources (union or intersection) was tested and measured worse than either alone
  in both directions - not used.

Real per-AOI numbers and diagnostic GeoTIFFs (`norway_permwater_diff_{aoi}.tif`, QGIS-inspectable)
are in `snakemake_workflow/tests/norway_diagnostics/`.

### 1.4 Benchmark rasterization's 0.5 wet-fraction threshold has a small, edge-only bias

A model cell where the benchmark polygon covers, say, 40% of its area is classified as
benchmark-dry (below the `benchmark_wet_fraction` = 0.5 threshold). This under-counts
benchmark-wet area specifically along polygon edges, symmetric in principle (a cell 60% covered is
over-counted the same way) but not exactly zero-sum in practice, since coastlines are not straight
lines and polygon edges are not randomly oriented relative to the model grid. Minor relative to
§1.1, but part of the same family of edge effects.

### 1.5 Vector simplification (100 m tolerance) shifts benchmark boundaries before buffering

`build_evaluation_clusters` simplifies benchmark geometries (`simplify_tolerance_m` = 100 m,
`preserve_topology=False`) before buffering, for tractability (plan doc §4.6a — this is what makes
buffering a real 397-polygon benchmark take under a second instead of hanging). This can shift a
polygon boundary by up to the tolerance distance. Small relative to `buffer_km` (2000 m default),
and this only affects the *buffer construction*, not the benchmark-wet rasterization itself (which
uses the original, unsimplified geometry via `benchmark_fraction_from_vector`) — but it means the
buffered domain's edge is not pixel-exact relative to the true polygon.

### 1.6 Population disaggregation assumes uniform distribution within a WorldPop cell

`population_by_class`/`average_pool_to_grid` splits a coarse (~1 km) WorldPop population count
across the fine model cells it covers, in proportion to *area* within each classification (TP/FP/
FN/TN), not in proportion to where people actually live within that coarse cell. If a WorldPop
cell is half flooded-per-the-model and half dry, this assigns half its population to each side
regardless of the true intra-cell settlement pattern (e.g. a village clustered on the dry half
would still have half its "people" counted as exposed). This is a standard, unavoidable limitation
of any area-weighted disaggregation from coarser population data, not specific to this pipeline —
but it directly affects `pop_HR`/`pop_FAR`/`pop_CSI`/`pop_diff_net`, which should be read as
"population implied by area-proportional disaggregation," not "population confirmed exposed."

---

## 2. Per-country data caveats

### 2.1 Spain (`ESP`, benchmark: `spain_zi_marina_q100`)

| Caveat | What it means | How this pipeline handles it | Residual implication |
|---|---|---|---|
| **Partial coverage** — the Q100 map covers only surveyed coastal stretches, not the entire coast | A model-wet cell outside every benchmark polygon may be a real false alarm *or* simply unsurveyed | `benchmark_wet_outside_model_domain_pct` health check; `buffered` vs `model_only` domain reporting (§1.1) | FAR is inflated in the direction described in §1.1; magnitude not yet separately quantified for Spain specifically |
| **Includes wave run-up** — Spain's benchmark methodology (IH2VOF at 200 m profile spacing) adds wave run-up on top of still-water level; the GFM model simulates still-water coastal flooding only | Physically asymmetric: the model should show systematically *less* extent than the benchmark near exposed, high-energy coastlines, independent of any real model error | Not corrected for — this is a genuine physical difference, not a bug, and is not currently isolated from other sources of under-prediction (FN) in the metrics | Expect `FN`/under-prediction to skew high near open-ocean-exposed coasts specifically; a national HR/FAR number blends this in with genuine model gaps |
| **Q100 = "probabilidad media" (100-year)** | Reasonably well-defined, single return period | Compared directly against the model's `RP100` scenario | Low residual risk — the best-matched return-period case among the countries investigated so far |
| **Benchmark is itself a model** (own DEM, own defences assumptions, 2014-2016 vintage) | "Benchmark" means independent reference, not ground truth | Not correctable; stated as a standing caveat | Both sides of the comparison carry modelling uncertainty, not just the GFM side |
| **Defences** — Spain's benchmark likely reflects some real structural defences implicitly (through whatever DEM/model produced it); the GFM baseline applies FLOPROS protection only in the *exposure* step, never in the flood extent itself | Comparing raw extents compares two different protection assumptions wherever real defences exist | Not currently addressed — open question, not yet verified whether/how much this matters for Spain's specific coastline | Could show as model over-prediction (FP) in defended areas that isn't a genuine extent error, just a protection-accounting mismatch |
| **Canary Islands scored as their own region** (`regions: {mainland, canary_islands}`, since 2026-09 — previously dropped via `exclude_bbox`) | No longer a scoping exclusion; the Canaries get their own metrics rows, agreement raster, and inset panel in the plot instead of being blended into (or dropped from) the national picture | Each evaluation cluster is tagged by which region's bbox its centroid falls in (`validation.region_for_point`); CSV rows, `_write_agreement_raster` output, and `plot_agreement_map.py`'s inset panels are all per-region | Now symmetric with France's kept-overseas approach (§2.2, §3) — same `regions:` mechanism, same reporting shape |

### 2.2 France (`FRA`, benchmarks: `france_inondable_02moy`/`_reunion_03mcc` (extent),
`france_inondable_ht_02moy`/`_ht_reunion_03mcc` (depth bands) — implemented and verified 2026-09)

| Caveat | What it means | How this pipeline will handle it | Residual implication |
|---|---|---|---|
| **Partial coverage** — only the 34 legally-designated TRI ("Territoires à Risque important d'Inondation") zones are surveyed, not the full coastline (e.g. much of Brittany's north coast, most of the Landes coast are outside any TRI) | Same mechanism as Spain §2.1, same §1.1 bias direction | Same `buffered`/`model_only`/health-check machinery, reused as-is | Same FAR-inflation risk as Spain; scale not yet measured |
| **Vaguer return period** — official methodology describes `02Moy` as "at least centennial," generally cited as a 100-300 year *class*, not an exact value the way Spain's "100 AÑOS" is labelled | The RP100-vs-benchmark comparison carries more definitional slack for France than for Spain | Compared against `RP100` anyway (best available analog), but this caveat should be repeated wherever France's numbers are quoted | A "disagreement" could partly reflect a genuine return-period mismatch (e.g. the true benchmark event being closer to RP150-200) rather than a real model error |
| **Overseas territories included in the same file** — Guadeloupe, Martinique, Guyane, Mayotte all have real TRI coverage in `n_inondable_03_02moy_s`; Réunion's marine-submersion data is a separate, near-empty file (`_ct` suffix, 8-11 polygons for the whole island) | Decision made 2026-09: keep overseas territories in scope, but score each as its own separate region (separate bounding box) rather than blending into one national number | The `regions: {name: bbox}` mechanism (§3) is now implemented and proven on Spain (`mainland`/`canary_islands`) — France's own catalog entry just needs its `regions:` block written (metropole + Guadeloupe + Martinique + Guyane + Mayotte + Réunion) when its benchmark entry is added | Réunion's own regional row, once implemented, will be statistically close to meaningless (near-zero surveyed area) — should be reported but flagged, not treated as a real result |
| **Type-01 edge case** — a small number of fluvial-layer (`typ_inond='01'`) rows are also tagged `cours_deau='submar'` (e.g. Étang de Berre lagoon near Marseille), and are NOT included in the type-03-only benchmark selection | A sliver of real coastal/lagoon flooding has no benchmark coverage under either type-01 (excluded by our filter) or type-03 (doesn't include it) | Not addressed — these cells simply fall into "unsurveyed" like any other gap | Minor; geographically localized to one lagoon |
| **Depth bands available and implemented** (`n_iso_ht_03_02moy_s`, `ht_min`/`ht_max`) | Enables a genuine depth-vs-depth comparison, not just extent-vs-extent | Built 2026-09 (`validate_country_depth_bands`, §3.2) — confirmed the depth-band polygons exactly tile the same extent polygons already validated (union area matches to floating-point noise across every TRI zone checked), so the same clusters/regions/land-use masking apply; sentinel values resolved by a magnitude threshold (`validation.depth_band_open_ended_min_m`, default 50 m), not an enumeration | Guyane has ZERO depth-band rows in this file despite having extent rows — depth-band comparison is not possible for Guyane via this dataset (a real, confirmed data gap, not a bug). Open-ended top-band sentinels are internally consistent per TRI zone but vary across zones (9999 most zones, 999 for Mayotte, 99 for Martinique, NaN — no value recorded — for several Brittany/Normandy zones) |
| **Defences / wave run-up** — not yet independently verified for France's TRI methodology | Assumed to carry similar risk to Spain's equivalent caveats, but not confirmed | Not addressed | Same caution as Spain §2.1, with lower confidence since it hasn't been specifically checked for France |

### 2.3 Norway (`NOR`, benchmark: `norway_stormflo_200ar`) — implemented 2026-09

**Architecturally different from Spain/France, not just a new catalog entry.** Kartverket assessed
Norway's ENTIRE coastline (confirmed via the source dump's own `dekningsomrade` coverage table:
`ikkeRelevant`/inland + `kartlagtUtenFunn`/mapped-nothing-found + `grundigKartlagtMedFunn`/mapped-
with-findings partition the whole country, no unsurveyed gap) — unlike Spain/France, whose
benchmarks only cover isolated designated survey zones (§1.1's whole reason for existing). That
means "outside the benchmark polygon" is unambiguous here (a real dry result, not "never looked
at"), so a model-wet cell there is a genuine over-prediction — there is nothing for
`build_evaluation_clusters`' buffer to protect against. Norway's `meta.coverage: national` therefore
routes it to a **separate driver, `validate_country_national_coverage`**, which processes per
postprocessing chunk directly with NO evaluation-cluster buffer/merge step at all, rather than the
cluster-based `validate_country` Spain/France use (`meta.coverage: partial`, the default). This is
not just a cleaner semantic match — the cluster-based path was tried first and **crashed**: Norway's
long, only-lightly-fragmented coastline made `build_evaluation_clusters`' buffer+merge collapse into
ONE cluster spanning almost the entire country (bbox 27°×13.5°), and reading/rasterizing at that
cluster's own bounding-box size tried to allocate 65.9 GiB. Chunk-sized processing (mirroring
`analysis/compute_flood_totals.py`'s own chunk-streaming architecture) bounds memory per iteration
regardless of how far a national benchmark's footprint spans. See `BenchmarkSpec.coverage`'s and
`validate_country_national_coverage`'s own docstrings for the full reasoning.

**Second real bug found and fixed (2026-09), from a real run's plot.** Chunks for a national-
coverage benchmark are found from its `regions:` bbox directly (there is no cluster geometry to
restrict to, unlike the partial-coverage path) - Norway's `mainland` bbox, generous enough to span
the country's own latitude range, genuinely overlaps real Swedish and Danish coastline. Since GFM
is a global model, it computes real flooding there too, and with no country-membership check, that
real Swedish/Danish flooding was being scored as a Norwegian false positive (visible in the
agreement plot as "coverage" along the Swedish/Danish coast, and inflating FAR) - the exact same
class of bug as §1.2's Spain/Portugal `flood_totals` bbox leakage, just surfacing in the
comparison's own headline metrics this time instead of a secondary diagnostic column. Fixed by
masking to `country_iso`'s own territory (`validation.read_country_mask`, the same WRI-geogunit +
FLOPROS-ISO mechanism already used for per-country exposure aggregation elsewhere in this
pipeline) before scoring - `validation.geogunit_source`/`iso_lookup_source` config keys.

| Caveat | What it means | How this pipeline handles it | Residual implication |
|---|---|---|---|
| **Source format**: Kartverket distributes this as a 26.1 GiB raw PostGIS `pg_dump` (plain-SQL, `COPY`-block, hex-EWKB geometry), not a shapefile/GeoPackage — no shapefile/GeoDataFrame-readable format exists | Not directly readable by geopandas/hydromt the way Spain/France's shapefiles are | One-time conversion script, `preparation/convert_norway_stormflo.py` (reuses a proven dump-parsing approach — direct Python parsing, no PostGIS server needed), streams the `stormflo200ar_klimaarna` table (115,440 rows, real all-rows-converted verified 2026-09) into a real GeoPackage the catalog entry reads normally | Not part of the regular Snakemake DAG — run manually/on-demand (same convention as `analysis/extract_country_population.py`) before this benchmark can be used; if Kartverket re-delivers the dump, re-run it |
| **Return-period mismatch**: Norway has no 100-year layer at all — only 20/200/1000-year (`sikkerhetsklasseflom` F1/F2/F3, TEK17 safety classes) | The RP100-vs-benchmark comparison carries real definitional slack, more than even France's vague "02Moy" | Decision (2026-09): compare F2/~200yr against the model's `RP100` anyway (the standard TEK17 planning class) rather than pursue an exact match at RP1000/F3 (which would need per-country return periods, a real architecture change not taken) | Expect a real, systematic component of disagreement purely from RP mismatch, not a model error — same caution as France's `02Moy` |
| **Benchmark polygons include the sea** — each raw polygon is "everything below water level X", not just newly-flooded land; confirmed 2026-09 that 97.6–99.3% of raw polygon area at sampled AOIs is permanently-wet sea, not land | Naively scoring the raw polygon would produce a hit rate dominated by open water — a near-perfect-looking but meaningless number | NOT stripped at conversion time (no vector-level dissolve/difference) — relies on this pipeline's existing Copernicus land-use permanent-water exclusion instead (§1.3), confirmed 2026-09 to correctly classify 98.06% of the real sea area as permanent water even across Norway's narrow fjords/island coastline (92.7–99.1% at every one of 5 sampled AOIs, including a deliberately narrow fjord and an island maze) | The SAME masking also removes ~51% of the genuine thin land-inundation strip (Norway's real flooded land is only ~0.2–2.1% of the raw polygon to start with, thinner than Copernicus' 100 m resolution can resolve) — not Norway-specific (measured 97.8% for Spain's own most-coastal feature too, see §1.3), but more consequential here since so little of the raw polygon is land to begin with; effective benchmark-wet area ends up roughly ~1% of raw polygon area |
| **Vertical datum**: Kartverket's levels are referenced to NN2000, not the model's GOCO06s geoid | Same class of issue as Martinique's COAST-RP/MDT gap (§2.2 France notes forthcoming) — but a boundary-position question here, not a value-correction one, since this benchmark is already a finished polygon, not a raw water level to combine with the model's own DEM | Measured, not corrected: real NN2000→GOCO06s offset (via Kartverket's own official HREF2018B conversion grid, sign-verified the same way as the Martinique investigation) is −0.196 m mean (−0.334 to −0.006 m across Norway's ~13° of latitude). Norway's coastline is steep enough (≈10 m of horizontal boundary movement per 1 m of water level, measured from Kartverket's own nested hazard layers) that this implies only ~2 m of horizontal boundary shift — 7% of one DEM pixel, 50× below the `vector_simplify_tolerance_m` already accepted in §1.5 | Deliberately not corrected — the correction would be numerically meaningless at this pipeline's own resolution and would add a moving part that can only introduce error. Would matter (~10% of the value) if `vannstandovernn2000` were ever used as a forcing *value* rather than read from a finished polygon — not the case here |
| **Bergen data-quality gap** — Kartverket's own four hazard layers for the Bergen area (mean-high-water/20yr/200yr/1000yr) have inconsistent feature counts (24/34/57/47) and are not cleanly nested | A real inconsistency in the source data itself, large enough to dominate any datum-related signal there | Not addressed — noted as a standing data-quality caveat specific to Bergen | Treat Bergen's own metrics with extra caution relative to the rest of Norway's coastline |

### 2.4 Germany, Japan, Australia — not yet resolved

- **Germany**: Niedersachsen's `Gefahrengebiete_HQ100_Z2` is "largely fluvial" per the original
  inventory (plan doc §2.2) — whether it's coastal enough to use at all is still an open question
  (plan doc §7.2 point 4), not yet answered. Schleswig-Holstein's data is an extension-less WFS GML
  file GDAL won't auto-detect — not yet readable. EPSG:4647 (ETRS89/UTM32N with a 32,000,000 m false
  easting) is a real CRS trap to handle correctly whenever this is picked up (plan doc §2.2).
- **Japan, Australia**: not downloaded yet (only `url.txt` placeholders as of the original
  inventory).

---

## 3. Open questions to resolve before implementing France

### 3.1 Per-region reporting — resolved and implemented (2026-09)

A country's catalog entry now expresses its named sub-regions as `meta.regions: {name: [minx,
miny, maxx, maxy]}` — one explicit bbox per region, covering the *whole* benchmark (there is no
implicit "everything else" catch-all region; every region a country cares about, including its
main/mainland territory, needs its own bbox — see `validation.BenchmarkSpec.regions`'s docstring
for why an implicit catch-all wouldn't give a wide-enough denominator for §1.2's mitigation).
`validate_country.py` tags each evaluation cluster by centroid-in-bbox
(`validation.region_for_point`; a cluster matching no region is labelled `"unclassified"` and
warned about, rather than silently dropped or guessed), and both the CSV accumulation and the
agreement raster (`_write_agreement_raster`, one file per region) are grouped by region in addition
to threshold/domain. `plot_agreement_map.py` discovers every region's raster and draws the largest
as the main map, the rest as small inset panels. Proven on Spain (`mainland` + `canary_islands`,
replacing the old `exclude_bbox` drop). France's own `regions:` block (metropole + 5 overseas
territories) is still just a catalog entry away — no further code changes needed.

### 3.2 Depth-band comparison — resolved and implemented (2026-09)

The framing proposed here turned out to match the real data cleanly, and is now built
(`validate_country_depth_bands` in `validate_country.py`, `rasterize_depth_bands`/
`depth_band_counts`/`depth_band_metrics_from_counts` in `src/validation.py`):

- **agree** = model depth within `[ht_min, ht_max]`; **under** = model depth `< ht_min`
  (model under-predicts); **over** = model depth `> ht_max` (model over-predicts) - only
  evaluated where the benchmark actually assigns a depth band to that cell at all (confirmed
  2026-09: France's depth-band polygons exactly tile the same extent polygons already
  validated, so "has a band" and "is inside the extent polygon" are the same condition, not
  two separate coverage questions).
- **Open-ended top-bucket sentinels** (confirmed 2026-09 against real France data: `9999`,
  `999`, `99`, or `NaN` with no value recorded at all - internally consistent per TRI zone,
  but varies by zone/region, so no single sentinel constant would have worked) are resolved by
  a **magnitude threshold**, `validation.depth_band_open_ended_min_m` (default 50 m -
  comfortably above any real closed band, found to top out at 4 m, comfortably below every
  real sentinel found) rather than an enumeration - any `ht_max` at/above that threshold, or
  missing entirely, is treated as "no upper bound," so a not-yet-seen sentinel value in a
  future country/dataset doesn't need a code change.
- Metrics mirror the extent comparison's own shape (per region/domain, no threshold sweep -
  the model's continuous depth is compared directly, nothing to threshold): `pct_agree`/
  `pct_under`/`pct_over` (a 3-way distribution) and `depth_EB = over/(over+under)`, the same
  over-vs-under framing as the extent comparison's own `EB`, plus population-weighted
  equivalents and an area/population-weighted agreement-category raster
  (`depth_agreement_{country}_{region}_{RP}_{SLR}.tif`, same 0/1/2/3/255 codes and
  priority-reduction as the extent comparison's own raster, plottable via
  `plot_agreement_map.py --metric depth_agreement`).
- Real gap found, not a code issue: France's Guyane TRI zone has extent polygons but **zero**
  depth-band rows in the source shapefile - depth-band comparison silently produces no
  "guyane" region row for France, rather than erroring.
