# Validation against official national flood hazard maps

## Purpose

This validates the GFM eikonal model's own production flood-extent output against official government coastal flood hazard maps, country by country. It is a separate, standalone exercise from `docs/methods_04a_SFINCSvalidation.md` (which validates the eikonal model against a SFINCS hydrodynamic re-simulation on a sample of tiles) — different code (`validation/` + `src/validation.py` at the repo root, not `sfincs_tiles/`), different question (does production match real-world government maps, not does eikonal match a physics reference), different inputs (existing production `merged_results/` output, not a fresh solve).

Not part of the Snakemake DAG — a standalone entry point, run manually after the production pipeline has produced `merged_results/chunks/waterdepth_*.tif`. Also used as the scoring backend for the OFAT calibration sweep (`validate_country.py --run-tag`).

`merge_chunk`'s waterdepth/provenance output (`snakemake_workflow/rules/postprocessing.smk`) always stays on disk, regardless of `postprocessing.plots.enabled` — validation's dependency on `merged_results/chunks/` is a guaranteed contract.

## Benchmark catalog

`data_catalog_validation.yml` (root set by `validation.benchmark_catalog` in `config.yml`) holds one entry per benchmark dataset. `_find_benchmark_keys()` scans every entry for `meta.country_iso`/`meta.hazard_type == "coastal"` — adding a country/dataset is a catalog entry, not a code branch. Each entry's `meta` block (`BenchmarkSpec`, `src/validation.py`):

- `data_type`: `GeoDataFrame` (vector polygons) or `RasterDataset`.
- `variable`: `extent` (default) or `depth`.
- `coverage`: `partial` (default) or `national`.
- `wet_values` / `depth_threshold_m`: raster-only, mutually exclusive (how to classify a raster benchmark as wet).
- `attribute_filter`: `{column: value}` rows to keep (`==` match, ANDed).
- `exclude_bbox`: drops rows whose centroid falls inside the box.
- `regions`: `{name: [minx, miny, maxx, maxy]}` — named sub-areas, reported as separate rows (see "Region blending" below for how these get folded into one country-level row).
- `ht_min_col`/`ht_max_col`: depth-band benchmarks only.
- `geogunit_ids`: explicit WRI geogunit-107 unit ID(s), for a benchmark covering a sub-national unit that shares its `country_iso` with sibling units it does NOT cover (Wales/Scotland both `GBR`, New Brunswick `CAN`) — bypasses `read_country_mask`'s normal ISO-string lookup entirely when set.
- `study_area`: `{source, geometry_type, attribute_filter?}` — **required for every `coverage: "partial"` benchmark**. `source` is a separate catalog key (`category: study_area_source`) holding the benchmark agency's own official study-design geometry. `geometry_type` dispatches which evaluation-domain construction applies — see "Partial-coverage extent" below.
- `comparison_return_periods`: optional list of NATIVE RPs (e.g. Norway/Wales/Scotland's own `[100, 250]` — each must be one of `boundary_conditions.return_periods` in `config.yml`) to score this benchmark at, beyond the single global `validation.return_period` every other entry point uses — see "Multi-RP summary" below. Absent for every other benchmark today.

## The three comparison paths

`validate_country.py::main()` runs all three unconditionally per country; a path with no matching benchmark just produces an empty result and is skipped.

### Partial-coverage extent — `validate_country()`

Spain, France. **Evaluation domain construction is dispatched by `meta.study_area.geometry_type`**:

- **`"perimeter"`** (France's TRI zones, `france_tri_perimeters`): `v.units_with_extent_coverage()` filters the perimeter source (131 TRI zones nationally, including fluvial-only ones) down to only the zones whose `id_tri` actually appears in the coastal extent benchmark being scored (34 real coastal zones) — this is what drops fluvial-only zones, without needing any generic proximity-based filter. **One evaluation unit per surviving zone**; `v.tri_domain_mask()` rasterizes that zone's own polygon exactly as-is (point-in-polygon, `all_touched=False`, no buffer) as the domain.
- **`"segments"`** (Spain's ARPSI coastal seed lines, `spain_arpsi_marina_segments`, filtered to `ORIGEN_INU == "Marina"`): a line is a 1D seed, not an area with its own extent, so there is no "near the line" domain question the way there is for a perimeter polygon. Instead, `v.connectivity_domain_mask()`: two independent 8-connected-component analyses (`scipy.ndimage.label`), one on the model's own raw wet mask and one on the benchmark's own raw wet mask, each keeping only the components that touch a seed cell (seed mask = ARPSI lines rasterized `all_touched=True`, no snap-tolerance buffer) — domain = **union** of both connected results, so a real miss (one map connects to a seed, the other never reaches there) stays detectable rather than being tautologically excluded. Labeling runs on the full unmasked wet mask first; `not_water` is applied only to the final result (rivers/lakes are real flood conduits — masking them before labeling would sever genuine connectivity paths). `build_evaluation_clusters()` (buffer+merge nearby seed segments) still exists here, but purely as a **chunk-windowing optimization** — grouping nearby segments so the same merged chunk isn't re-read once per segment — never as the scoring domain itself.

For each evaluation unit/window: find every production chunk (`merged_results/chunks/waterdepth_{chunk_id}_{RP}_{SLR}.tif`) overlapping its bbox, mosaic them at native resolution/CRS (`rasterio.merge`), rasterize the benchmark onto that grid at 5x supersampling (`benchmark_supersample`) to a 0-1 coverage `fraction`, mask out permanent water (`permanent_water_source: deltadtm_mask`, codes `[1,2,3]`), classify `model_wet` at the single config-driven `primary_threshold_m` (no sweep — see "Metrics" below), and score via `v.confusion_counts_soft(model_wet, fraction, domain, area_km2)` — `fraction` is used directly as continuous credit, never thresholded into a binary benchmark-wet/dry call for scoring. (`wet_mask_from_fraction`/`benchmark_wet_fraction` still exist for two narrow purposes: the agreement-map's own DISPLAY category raster, and the connectivity domain's own binary benchmark-wet input — domain construction, not scoring.)

**Per-unit CSI** (`validate_country.py`'s own `_record_unit`) is also recorded for every evaluation unit — a TRI zone, or (for `"segments"`) each distinct connected component of the final scored domain (`v.connectivity_components()`; not per-original-ARPSI-table-row, which would need a nontrivial nearest-seed attribution step) — written to `units_{benchmark_key}_{RP}_{SLR}.csv`, the input to the CSI-dot map (see "Outputs" below).

### National-coverage extent — `validate_country_national_coverage()`

Norway, Finland, Denmark, Wales, Scotland, New Brunswick. No clustering: the benchmark polygon (or raster) genuinely covers the whole coastline, so "outside it" means "surveyed and found dry," not "never assessed" — there's nothing for a buffer to protect against.

Processes per 5° production chunk overlapping each required `regions` bbox (required here — there's no cluster geometry to derive an extent from otherwise). Domain is always `model_domain & not_water & in_country` — the extra country mask (WRI geogunit-107 raster + FLOPROS ISO lookup, or `geogunit_ids` directly for a sub-national benchmark) exists because a `regions` bbox is a rectangle that can genuinely spill into a neighbouring country/territory. Scoring uses the same `confusion_counts_soft` mechanism as the partial-coverage path, at the same single `primary_threshold_m`.

Denmark's benchmark is the only `RasterDataset` entry: `read_benchmark_raster_fraction()` classifies its native 5m depth GeoTIFF wet/dry at full resolution first, *then* reprojects the already-binary mask with `Resampling.average` — reprojecting the raw depth values first would only sample ~1 of ~36 native sub-pixels per model cell. `Resampling.average` gives a true continuous coverage fraction — the real proportion of wet native sub-pixels per model cell — the raster-grid equivalent of `benchmark_supersample`'s sub-cell coverage estimate for a vector benchmark.

### Depth-band comparison — `validate_country_depth_bands()`

France's `n_iso_ht_*` layers (`variable: depth`) only. `coverage: partial` only; raises `NotImplementedError` on a national-coverage spec (deliberate — no chunk-based depth-band path exists). Uses its own buffer+cluster evaluation domain (`eval_domain.buffer_km`/`also_report_model_domain_only`, "buffered"/"model_only" reporting) — a different, separate domain mechanism from the `study_area`-based one the extent path (above) uses.

Rasterizes the benchmark's own `ht_min`/`ht_max` band bounds directly (categorical, no supersampling needed — bands tile the extent polygons exactly). A closed band with `ht_max` at or above `depth_band_open_ended_min_m` (50m) or `NaN` resolves to `+inf` (real French sentinel values are 9999/999/99/NaN depending on zone). Per cell: `agree = ht_min <= depth <= ht_max`, `under = depth < ht_min`, `over = depth > ht_max`. Metrics (`pct_agree`/`pct_under`/`pct_over`) are the 3-way split over cells *with* a band assigned only. `depth_EB = over / (over + under)`.

## Metrics

`metrics_from_counts()` (`src/validation.py`), run once per accumulated `region` key, never per-unit (a per-unit ratio averaged afterward would let a tiny unit dominate as much as a large one):

```
HR  = tp / (tp + fn)
FAR = fp / (tp + fp)
CSI = tp / (tp + fp + fn)
EB  = fp / (fp + fn)        # >0.5 = over-predicting
EB_ratio = fp / fn          # Wing et al. (2017) form, kept alongside EB
bias = (tp + fp) / (tp + fn)
```

Pure arithmetic on whatever tp/fp/fn/tn it's given — identical formulas whether those counts came from `confusion_counts_soft` (the only production scoring path) or the generic hard-threshold `confusion_counts` (kept for its own unit tests — `tests/flood_extent_validation/test_metrics.py` — but not called anywhere in `validate_country.py`). Zero denominators return NaN, never a silent 0.

**Single config-driven threshold, no sweep**: `model_wet = model_domain & (depth > primary_threshold_m) & not_water`, reading `val_cfg["primary_threshold_m"]` (0.01 — deliberately NOT `exposure.exceedance_threshold_m`, still 0.10, a different purpose: flood-protection/damage accounting, not extent comparison) — this is the MODEL's own depth classification, a separate question from the benchmark's own coverage `fraction`, which is never thresholded for scoring at all. A real sensitivity check (GBR Wales/Scotland, 0.10/0.05/0.01m) found CSI/HR improving smoothly and monotonically all the way to 0.01m with no noise or reversal, despite 0.01m sitting below the eikonal solver's own convergence tolerance (`simulation.flooding.waterlevel_epsilon_m`, 0.03m) — a "pure flood extent, any real depth counts" comparison, closer to what the benchmark maps themselves represent (a binary flood/no-flood extent) than an accounting threshold borrowed from a different pipeline.

**Tolerant-confusion diagnostic, reported alongside CSI/HR/FAR** (`v.confusion_counts_tolerant`): every region row also carries `CSI_tol`/`HR_tol`/`FAR_tol` plus `fp_forgiven_km2`/`fn_forgiven_km2`/`pct_disagreement_forgiven`. This answers one narrow question — how much of the *strict* disagreement (computed on a hard `fraction > 0` benchmark-wet basis, a presence test, not a reintroduced 0.5-style threshold) sits within `coastline_tolerance_cells` pixels of a cell where model and benchmark actually agree, i.e. looks like the same flood boundary drawn with a small spatial offset rather than a real disagreement about whether an area floods at all. It applies uniformly to every benchmark (partial and national coverage alike) — it has no notion of "coastline," it just happens to mostly fire there in practice, since that's where two independently-drawn wet/dry boundaries disagree by a pixel or two. Written for the permanent-water-masking caveat below, but general-purpose.

Three things keep it from being a free CSI inflator: (1) forgiveness is symmetric — a model false positive is only forgiven if a *real* benchmark-wet cell exists nearby, and vice versa for a false negative — so an isolated, unexplained mismatch with no nearby agreement gets zero credit no matter how large `coastline_tolerance_cells` is; (2) it can only move weight out of fp/fn into fp_forgiven/fn_forgiven, never manufacture tp, so `CSI_tol >= CSI` always, by construction; (3) `pct_disagreement_forgiven` is always reported next to `CSI_tol` — a reader sees how much work the tolerance is doing, not just the headline number. `coastline_tolerance_cells` defaults to `1` (one native 30m grid cell), set from the actual measured registration-scale evidence (NN2000↔GOCO06s datum shift ~2m, DeltaDTM mask native resolution ~25-31m — both sub-pixel-to-one-pixel), not tuned upward to make the number look better.

**Region blending is part of the pipeline** (`validate_country.py`'s own `_blend_regions`, called automatically after per-region accumulation). Two region-partition shapes exist in this catalog, auto-detected by bbox containment, no new catalog field needed:

- **Disjoint partition** (Spain `mainland`/`canary_islands`, France `metropole`+5 overseas territories, Norway/Finland's own single region): no region's bbox contains another's — `_blend_regions` sums `tp`/`fp`/`fn`/`tn` across all regions and recomputes HR/FAR/CSI/bias from the pooled sums.
- **Nested whole-territory + sub-area(s)** (Wales `wales`+`severn_estuary`+`menai_strait`, Scotland `scotland`+`firth_of_forth`, New Brunswick `new_brunswick`+`cumberland_basin`+`petitcodiac`): one region's bbox contains every sibling's — that region's own row IS the country total already (its sub-areas are independently re-scored over the *same* ground, so summing them would double-count); `_blend_regions` uses it directly, not a fresh sum.

The blended row is appended with `region == "ALL"` (`_BLENDED_REGION`) whenever a benchmark has more than one region row; a single-region benchmark has nothing to blend, so no "ALL" row appears.

## Outputs

`benchmark_wet_outside_model_domain_km2`/`_pct` — the fraction of the benchmark's *own* wet area falling entirely outside the model's computed domain; high values mean don't trust that row's other metrics.

Written per country to `{validation.output_dir}/{country}/`:
- `metrics_{country}_{RP}_{SLR}.csv` — extent path (partial + national coverage concatenated), one row per region plus one `"ALL"` blended row per benchmark when more than one region exists. Besides the strict `HR`/`FAR`/`CSI`/`EB`/`EB_ratio`/`bias`, every row also carries the tolerant-confusion diagnostic columns (`HR_tol`/`FAR_tol`/`CSI_tol`/`fp_forgiven_km2`/`fn_forgiven_km2`/`pct_disagreement_forgiven` — see "Metrics" above).
- `units_{benchmark_key}_{RP}_{SLR}.csv` — partial-coverage benchmarks only: one row per evaluation unit (TRI zone / connected component), its own `lon`/`lat`/`csi`/`hr`/`far`/`n_cells`.
- `depth_metrics_{country}_{RP}_{SLR}.csv` — depth-band path, separate file (France only).
- `agreement_{country}_{region}_{RP}_{SLR}.tif` / `depth_agreement_{country}_{region}_{RP}_{SLR}.tif` — one 4-category raster per region (`dry=0 < agree=1 < under=2 < over=3`, priority order over>under>agree>dry), at `primary_threshold_m` only. Written at the model's own NATIVE resolution (30m) whenever the region is small enough for that to stay a tractable raster size (`_AGREEMENT_RASTER_MAX_DIM_PX`, 2500px either dimension — every partial-coverage unit and every nested sub-area, e.g. Wales' menai_strait/severn_estuary, Scotland's firth_of_forth, qualifies); only a whole-country-scale region (Norway, Scotland's own mainland) falls back to `plots.resolution_m` (200m) via `Resampling.max`. CSI/HR/FAR are always computed from the native-resolution depth/fraction grids directly, before this raster is ever written — this resolution choice is about the plotted picture, not the scoring.

 A normal run (`--metric agreement`, the default) writes, per country:

Every region raster is first grouped by which BENCHMARK's own `meta.regions` claims it (`_group_region_rasters_by_benchmark`) — a country with more than one benchmark sharing its `country_iso` (GBR: Wales' `wales_nrw_floodzone_seas` + Scotland's `scotland_sepa_coastal_m`, distinguished only by `geogunit_ids`) gets one independent main map + sub-regions grid per benchmark, never one mixed plot that picks a single "main" region across both territories. A country with only one matching benchmark (every country except GBR today) gets exactly one group, and the filenames below have no `_{benchmark_key}` segment.

1. **`{metric}_{country}[_{benchmark_key}]_{RP}_{SLR}.png`** (`plot_agreement_map()`) — within its own benchmark's group, the region with the most pixels as a full-size main map, every other region in that group squeezed into a small corner inset (wrapping past ~4). Every panel's title/label shows that region's own HR/FAR/CSI (`_load_region_metrics`, reads `metrics_{country}_{RP}_{SLR}.csv`, filtered to that benchmark's own rows).
2. **`{metric}_{country}[_{benchmark_key}]_subregions_{RP}_{SLR}.png`** (`plot_subregions_grid()`, written when a benchmark's own group has more than one region) — every NON-main region in that group as its own full-size panel in an auto NxM grid, not a tiny inset. For Wales/Scotland/New Brunswick, this is the "now actually look closely at the sub-area(s)" view paired with (1)'s whole-territory main map.
3. **`tile_coverage_{country}_{benchmark_key}.png`** (`plot_tile_coverage()`, every `GeoDataFrame` extent benchmark — skips Denmark's `RasterDataset` one) — which production tiles intersect the benchmark's real extent, for the single largest region (`v.primary_region_name()`), tile IDs labeled directly (up to ~60 tiles).
4. **`csi_dots_{country}_{benchmark_key}_{RP}_{SLR}.png`** (`plot_unit_csi_dots()`, partial-coverage benchmarks only, skipped if no `units_*.csv` exists) — one dot per evaluation unit at its own centroid, coloured and labeled by its own CSI. This is the per-unit detail a blended region/country number can't show.

## Multi-RP summary — `run_multi_rp_summary.py`

A separate, standalone script (not called by `run_validation.py`, not part of a normal run) for the one question none of the per-RP outputs above answer directly: *across every return period a benchmark should be compared at, how does model skill change with RP?* Only benchmarks with `meta.comparison_return_periods` set (today: Norway, Wales, Scotland — each `[100, 250]`) are in scope — every other benchmark is scored once, at the single global `validation.return_period`, same as every other entry point, and never appears in this script's output at all.

Every listed RP must be NATIVE — a real `merged_results/chunks/waterdepth_*_{RP}_*.tif` already on disk (i.e. one of `boundary_conditions.return_periods` in `config.yml`). For each one, the script re-runs the exact same `validate_country()`/`validate_country_national_coverage()` scoring with `validation.return_period` overridden — no new code path, the identical result a manual `--country NOR --config ...` run at that RP would give, just driven from one script instead of by hand per RP.

Output: one row per `(country, benchmark_key, return_period)` in `{validation.output_dir}/multi_rp_summary.csv` — the country-level blended row (`region == "ALL"`, `_blend_regions`) where one exists, else the benchmark's single region row. Columns: `HR`/`FAR`/`CSI`/`EB`/`EB_ratio`/`bias`, `tp_km2`/`fp_km2`/`fn_km2`, `benchmark_wet_outside_model_domain_pct` — the tolerant-confusion diagnostic columns from "Metrics" above are not included here, since this script's own purpose is the RP trend, not the registration-noise diagnostic.

The script also plots every RP it scores (`plot_agreement_map.plot_country()`, once per distinct `(country, RP)` actually scored, covering every benchmark sharing that country/RP in one call) — the normal `run_validation.py`/`validate_country.py` + `plot_agreement_map.py` pairing only ever plots at the single global `validation.return_period`, so without this, RP250's own agreement maps for Norway/Wales/Scotland would never get produced even though RP250's own category rasters exist on disk (written by the scoring step regardless). Output file names already include the RP (`plot_country`'s own docstring), so RP100 and RP250 never overwrite each other.

## Orchestration and plots

`run_validation.py` discovers every country with a coastal benchmark, then per country subprocess-dispatches `validate_country.py` followed by `plot_agreement_map.py` (module isolation — one country's crash doesn't abort the rest, unless `--fail-fast`; `--skip-plots` to skip the plotting step). After the loop it concatenates every country's `metrics_*.csv` into one `summary_{RP}_{SLR}.csv`.

`validation/plot_agreement_map.py` is pure visualization over the already-written category rasters/CSVs (no recomputation).

## Key config (`config.yml`'s `validation:` block)

| Key | Value | Meaning |
|---|---|---|
| `return_period` / `waterlevel_name` | `RP100` / `SLR_0` | Single scalar, no CLI override — every country validated at RP100/no-SLR-offset only. |
| `primary_threshold_m` | `0.01` | The single model wet-cell threshold scoring uses — no sweep. Deliberately NOT `exposure.exceedance_threshold_m` (still 0.10, a different purpose); see "Metrics" above for the real sensitivity check behind this value. |
| `benchmark_supersample` | `5` | Vector benchmark rasterization supersampling factor. |
| `benchmark_wet_fraction` | `0.5` | DISPLAY-only (agreement-map category raster) and Spain's connectivity domain's own binary benchmark-wet input. Not used for scoring. |
| `coastline_tolerance_cells` | `1` | Dilation radius (native 30m grid cells) for the `CSI_tol`/`HR_tol`/`FAR_tol` diagnostic (`v.confusion_counts_tolerant`) — reported alongside, never instead of, the strict CSI/HR/FAR. |
| `eval_domain.buffer_km` | `2.0` | (1) `validate_country_depth_bands`'s own evaluation domain. (2) `validate_country()`'s `"segments"` branch: chunk-windowing only, not the scoring domain. |
| `eval_domain.also_report_model_domain_only` | `true` | `validate_country_depth_bands` only. |
| `vector_simplify_tolerance_m` | `100.0` | Benchmark/seed polygon simplification before buffering. |
| `permanent_water_source` / `codes` | `deltadtm_mask` / `[1,2,3]` | Ocean/lake/river cells excluded from every domain. |
| `geogunit_source` / `iso_lookup_source` | WRI geogunit-107 / FLOPROS | Country mask for the national-coverage path only. |
| `depth_band_open_ended_min_m` | `50.0` | A depth-band benchmark's `ht_max` at/above this (or NaN) means no upper bound. |
| `plots.resolution_m` | `200` | Agreement-raster output resolution for whole-country-scale regions only — a small region (partial-coverage unit, nested sub-area) is written at the model's own native 30m resolution instead, see "Outputs" above. |

## Current status by country

| Country | `coverage` | Benchmark type | Comparison(s) |
|---|---|---|---|
| Norway | national | GeoDataFrame | extent |
| Denmark | national | RasterDataset | extent |
| Finland | national | GeoDataFrame | extent |
| Wales (GBR, `geogunit_ids`) | national | GeoDataFrame | extent |
| Scotland (GBR, `geogunit_ids`) | national | GeoDataFrame | extent |
| New Brunswick (CAN, `geogunit_ids`) | national | GeoDataFrame | extent |
| Spain | partial | GeoDataFrame | extent (connectivity domain) |
| France | partial | GeoDataFrame | extent (TRI-zone domain) + depth-band (buffer domain) |

Wales/Scotland/New Brunswick all use the `BenchmarkSpec.geogunit_ids` mechanism (bypasses the normal ISO-string country match in `read_country_mask` - their shared ISO code (`GBR`, `CAN`) would otherwise also match sibling territories/provinces the benchmark doesn't cover). Each one's spot-check so far has been a single representative tile/region, not a full country-wide production run - see `data_catalog_validation.yml`'s own per-entry `known_caveats` for what's been verified and what hasn't.

## Caveats

Pipeline-wide mechanisms worth keeping in mind when reading a result:

- **Permanent-water masking has a real, quantified bias in at least one country (Norway)** — switching from Copernicus land_use (~100m) to DeltaDTM's own native mask (~25-31m in Norway) measurably reduced coastal-strip masking loss, but didn't eliminate it; the remainder is a genuine waterline-*definition* disagreement between two independently-drawn coastlines (DeltaDTM/ESA WorldCover's optical water edge vs. a benchmark's own surveyed coastline), not fixable by a better mask at any resolution. The `CSI_tol` diagnostic above (Metrics) is a cheap, already-shipped way to see how much of a country's strict CSI gap this kind of small-scale registration disagreement explains, without sourcing any new data. A real fix — scoring directly against Kartverket's own coastline, defined in their product spec as the "Middelhøyvann" (mean-high-water) contour at the NN2000 datum — is NOT a quick change: that contour is not already present in the benchmark data downloaded for this pipeline (confirmed — the raw delivery's only real table is the hazard-zone polygons themselves), so it would need a new, unconfirmed sourcing/download step from Kartverket plus making `permanent_water_source` overridable per-benchmark (today it is one global value). Separately: Kartverket's own methodology is a pure water-level-above-elevation-threshold method with no river/lake-specific handling and no connectivity-to-sea check documented anywhere in their spec, and the benchmark schema itself has no depth column or water-body-type attribute — so the benchmark does not distinguish coastal flooding from river/lake flooding at all, incidental inclusion of either is expected by design, and nothing in this pipeline can filter one from the other post hoc.
- **Narrow fjords/basins get a documented, systematically oversized Norwegian benchmark** — Kartverket's own product spec flags quality classes 3-4 (narrow constrictions) as copying the open-coast water level across the constriction, which they themselves describe as *overdimensjonerende* (deliberately oversized) for both water level and flooded area — a real, named caveat in their own methodology, not an isolated anomaly.
- **Simplifying benchmark/seed polygons** (100m tolerance) before buffering trades geometric precision for tractability on some genuinely huge (500k+ vertex) real polygons — affects chunk-windowing grouping only (`"segments"` branch), not the scoring domain itself.
- **Every benchmark is itself a model** (its own DEM, its own defences assumptions, its own vintage) — "benchmark" means independent reference, not ground truth.
- **Defences and wave run-up are not represented by this model at all** — several countries' own benchmark methodologies include wave run-up or reflect real structural defences implicitly, while this model simulates still-water coastal flooding only and applies flood-protection standards only in the separate exposure step, never in the extent computation itself.
- **Return-period definitions are not guaranteed to mean the same event across national methodologies** — a "100-year" event in one country's own hazard-mapping convention is not necessarily statistically equivalent to this model's own RP100 scenario (e.g. Norway's nearest native benchmark class is ~200yr, not 100yr).
- Full per-country detail (data provenance, known data-quality gaps, exact caveat magnitudes where measured) lives in each benchmark's own catalog entry (`data_catalog_validation.yml`'s `known_caveats`), not a separate document — this keeps the caveat next to the exact config/filter it describes instead of in a document that can drift out of sync with the catalog.
