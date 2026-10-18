# Validation against official national flood hazard maps

## Purpose

This validates the GFM eikonal model's own production flood-extent output against official government coastal flood hazard maps. It is a separate, standalone exercise from `docs/methods_04a_SFINCSvalidation.md` (which validates the eikonal model against a SFINCS hydrodynamic re-simulation on a sample of tiles) — different code (`validation/` + `src/validation.py` at the repo root, not `sfincs_tiles/`), different question (does production match real-world government maps, not does eikonal match a physics reference), different inputs (existing production `merged_results/` output, not a fresh solve).

Scope: **Wales, Scotland, England (GBR), New Brunswick (CAN) and San Francisco Bay (USA)**, plus their named sub-regions. Every benchmark is a sub-national product covering its whole territory's coastline.

Not part of the Snakemake DAG — standalone entry points, run manually after the production pipeline has produced `merged_results/chunks/waterdepth_*.tif`. `validate_country.py --run-tag` tags every output row, for aggregating several runs.

`merge_chunk`'s waterdepth/provenance output (`snakemake_workflow/rules/postprocessing.smk`) always stays on disk, regardless of `postprocessing.plots.enabled` — validation's dependency on `merged_results/chunks/` is a guaranteed contract.

## Benchmark catalog

`snakemake_workflow/config/data_catalog_validation.yml` (path set by `validation.benchmark_catalog`, data root by `validation.benchmark_root` in `config.yml`) holds one entry per benchmark dataset. `_find_benchmark_keys()` scans every entry for `meta.country_iso`/`meta.hazard_type == "coastal"` — adding a dataset is a catalog entry, not a code branch. Each entry's `meta` block (`BenchmarkSpec`, `src/validation.py`):

- `data_type`: `GeoDataFrame` (vector polygons) — every kept benchmark.
- `variable`: `extent`.
- `coverage`: `national` — the benchmark covers the whole territory, so "outside it" means "assessed and found dry".
- `attribute_filter`: `{column: value}` rows to keep (`==` match, ANDed) — e.g. Wales' `{risk: "Flood Zone 3"}`.
- `regions`: `{name: [minx, miny, maxx, maxy]}` — named sub-areas, each scored and reported as its own row (see "Region blending" below). Required: there is no other geometry to derive a processing extent from.
- `geogunit_ids`: explicit WRI geogunit-107 unit ID(s). Every kept benchmark covers a sub-national unit that shares its `country_iso` with sibling units it does NOT cover (England 3364, Scotland 3366, Wales 3367 under `GBR`; New Brunswick 492 under `CAN`) — bypasses `read_country_mask`'s normal ISO-string lookup entirely when set.
- `comparison_return_periods`: optional list of NATIVE RPs (each must be one of `boundary_conditions.return_periods` in `config.yml`) to score this benchmark at, beyond the single global `validation.return_period` — see "Multi-RP summary" below. Set for Wales/Scotland/England FZ3 (`[100, 250]`, bracketing their ~1-in-200 benchmarks) and England FZ2 (`[1000]`).

San Francisco Bay has no catalog entry — it is a depth raster scored by its own script (see "San Francisco Bay" below).

## Comparison path — `validate_country_national_coverage()`

`validate_country.py::main()` runs this per country (`GBR`, `CAN`) for every matching benchmark. No clustering: the benchmark covers the whole coastline, so there's nothing for a buffer to protect against.

Processes per 5° production chunk overlapping each `regions` bbox: find every production chunk (`merged_results/chunks/waterdepth_{chunk_id}_{RP}_{SLR}.tif`) overlapping the bbox, rasterize the benchmark onto the model grid at 5x supersampling (`benchmark_supersample`) to a 0-1 coverage `fraction`, and score via `v.confusion_counts_soft(model_wet, fraction, domain, area_km2)` — `fraction` is used directly as continuous credit, never thresholded into a binary benchmark-wet/dry call for scoring.

Domain is always `model_domain & not_water & in_country`:

- **Permanent water** (`not_water`): ocean/lake/river cells from DeltaDTM's own native mask (`permanent_water_source: deltadtm_mask`, codes `[1,2,3]`) are excluded from every domain.
- **Country mask** (`in_country`): WRI geogunit-107 raster, selected by `geogunit_ids` (or a FLOPROS ISO lookup when unset) — exists because a `regions` bbox is a rectangle that can genuinely spill into a neighbouring territory.

`model_wet` is classified at the single config-driven `primary_threshold_m` (no sweep — see "Metrics" below). `wet_mask_from_fraction`/`benchmark_wet_fraction` are used only for the agreement-map's own DISPLAY category raster, never for scoring.

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

Pure arithmetic on whatever tp/fp/fn/tn it's given — identical formulas whether those counts came from `confusion_counts_soft` (the production scoring path) or the generic hard-threshold `confusion_counts` (kept for its own unit tests — `tests/flood_extent_validation/test_metrics.py`). Zero denominators return NaN, never a silent 0.

**Single config-driven threshold, no sweep**: `model_wet = model_domain & (depth > primary_threshold_m) & not_water`, reading `val_cfg["primary_threshold_m"]` (0.01 — deliberately NOT `exposure.exceedance_threshold_m`, still 0.10, a different purpose: flood-protection/damage accounting, not extent comparison) — this is the MODEL's own depth classification, a separate question from the benchmark's own coverage `fraction`. A sensitivity check (GBR Wales/Scotland, 0.10/0.05/0.01m) found CSI/HR improving smoothly and monotonically all the way to 0.01m with no noise or reversal, despite 0.01m sitting below the eikonal solver's own convergence tolerance (`simulation.flooding.waterlevel_epsilon_m`, 0.03m) — a "pure flood extent, any real depth counts" comparison, closer to what the benchmark maps themselves represent than an accounting threshold borrowed from a different pipeline.

**Tolerant-confusion diagnostic, reported alongside CSI/HR/FAR** (`v.confusion_counts_tolerant`): every region row also carries `CSI_tol`/`HR_tol`/`FAR_tol` plus `fp_forgiven_km2`/`fn_forgiven_km2`/`pct_disagreement_forgiven`. This answers one narrow question — how much of the *strict* disagreement (computed on a hard `fraction > 0` benchmark-wet basis, a presence test) sits within `coastline_tolerance_cells` pixels of a cell where model and benchmark actually agree, i.e. looks like the same flood boundary drawn with a small spatial offset rather than a real disagreement about whether an area floods at all. It has no notion of "coastline"; it just mostly fires there in practice, since that's where two independently-drawn wet/dry boundaries disagree by a pixel or two.

Three things keep it from being a free CSI inflator: (1) forgiveness is symmetric — a model false positive is only forgiven if a *real* benchmark-wet cell exists nearby, and vice versa for a false negative — so an isolated mismatch with no nearby agreement gets zero credit; (2) it can only move weight out of fp/fn into fp_forgiven/fn_forgiven, never manufacture tp, so `CSI_tol >= CSI` always; (3) `pct_disagreement_forgiven` is always reported next to `CSI_tol`. `coastline_tolerance_cells` defaults to `1` (one native 30m grid cell), set from measured sub-pixel-to-one-pixel registration offsets (vertical-datum shift, DeltaDTM mask resolution), not tuned upward.

**Region blending** (`validate_country.py`'s own `_blend_regions`, called automatically after per-region accumulation). Two region-partition shapes, auto-detected by bbox containment:

- **Nested whole-territory + sub-area(s)** (Wales `wales`+`severn_estuary`+`menai_strait`, Scotland `scotland`+`firth_of_forth`): one region's bbox contains every sibling's — that region's own row IS the territory total (sub-areas are re-scored over the *same* ground, so summing would double-count); `_blend_regions` uses it directly.
- **Disjoint partition** (New Brunswick `cumberland_basin`+`petitcodiac` — no whole-province region, see the catalog entry for why): `_blend_regions` sums `tp`/`fp`/`fn`/`tn` across regions and recomputes the metrics from the pooled sums.

The blended row is appended with `region == "ALL"` (`_BLENDED_REGION`) whenever a benchmark has more than one region row; a single-region benchmark (England) has no "ALL" row.

## Outputs

`benchmark_wet_outside_model_domain_km2`/`_pct` — the fraction of the benchmark's *own* wet area falling entirely outside the model's computed domain; high values mean don't trust that row's other metrics.

Written per country to `{validation.output_dir}/{country}/`:
- `metrics_{country}_{RP}_{SLR}.csv` — one row per region plus one `"ALL"` blended row per multi-region benchmark. Strict `HR`/`FAR`/`CSI`/`EB`/`EB_ratio`/`bias` plus the tolerant-confusion columns.
- `agreement_{country}_{region}_{RP}_{SLR}.tif` — one 4-category raster per region (`dry=0 < agree=1 < under=2 < over=3`, priority over>under>agree>dry), at `primary_threshold_m`. Written at the model's NATIVE resolution (30m) when the region stays under `_AGREEMENT_RASTER_MAX_DIM_PX` (2500px either dimension — every nested sub-area qualifies); a whole-territory region (Scotland, England) falls back to `plots.resolution_m` (200m) via `Resampling.max`. Metrics are always computed from the native-resolution grids first — the resolution choice only affects the picture.

`plot_agreement_map.py` then writes, per country. Every region raster is first grouped by which benchmark's `meta.regions` claims it — GBR has several benchmarks sharing its `country_iso`, distinguished only by `geogunit_ids`, so each gets its own independent plots (filenames carry a `_{benchmark_key}` segment); CAN has one benchmark and no such segment.

The `_{RP}` segment is dropped for a benchmark with several `comparison_return_periods`: its RPs share one figure.

1. **`{metric}_{country}[_{benchmark_key}][_{RP}]_{SLR}.png`** (`plot_agreement_map()`) — the region with the most pixels as the main map, every other region in that group as a small corner inset. Every panel shows its own HR/FAR/CSI (`_load_region_metrics`, reading `metrics_{country}_{RP}_{SLR}.csv`).
2. **`{metric}_{country}[_{benchmark_key}]_subregions[_{RP}]_{SLR}.png`** (`plot_subregions_grid()`, when a group has more than one region) — every non-main region as its own full-size panel. For Wales/Scotland this is the close-up view of the sub-areas paired with (1)'s whole-territory map.
3. **`tile_coverage_{country}_{benchmark_key}.png`** (`plot_tile_coverage()`) — which production tiles intersect the benchmark's extent, for the largest region (`v.primary_region_name()`), tile IDs labeled.

## Multi-RP summary — `run_multi_rp_summary.py`

A separate, standalone script (not called by `run_validation.py`) for the question the per-RP outputs don't answer directly: *across every return period a benchmark should be compared at, how does skill change with RP?* Only benchmarks with `meta.comparison_return_periods` set are in scope.

Every listed RP must be NATIVE (a real `merged_results/chunks/waterdepth_*_{RP}_*.tif` on disk). For each one, the script re-runs the same `validate_country_national_coverage()` scoring with `validation.return_period` overridden — no new code path.

Output: one row per `(country, benchmark_key, return_period)` in `{validation.output_dir}/multi_rp_summary.csv` — the `"ALL"` blended row where one exists, else the single region row. Columns: `HR`/`FAR`/`CSI`/`EB`/`EB_ratio`/`bias`, `tp_km2`/`fp_km2`/`fn_km2`, `benchmark_wet_outside_model_domain_pct` (no tolerant-confusion columns). It also plots every RP it scores (`plot_agreement_map.plot_country()`, once per country after scoring), since the normal run only plots at the global `validation.return_period`; a benchmark's RPs go into one RP-less figure (see "Outputs").

## England — standalone scripts

`fetch_england_flood_zones.py` downloads the EA "Flood Map for Planning: Flood Zones 2 & 3 (Rivers and Sea)" via its OGC API in ~1° tiles (a single national request never returned), keeping modelled-origin features with `flood_source` `sea` or `river and sea` (the estuarine/tidal-river class); plain `river` and `unknown` are excluded. The output is split by zone into the files behind `england_ea_floodzone3_sea` and `england_ea_floodzone2_sea`.

### Flood Zone 3 rescore — `rescore_england_fz3.py`

Re-scores FZ3 at the RP100/RP250 bracket only — the same two numbers `run_multi_rp_summary.py` would give, by a narrower path. It exists because the full GBR multi-RP run failed on memory/runtime once the FZ3 layer became much denser (73,821 features after adding `river and sea`): loading all GBR benchmarks at once, and supersampling a full un-clipped 5° chunk in one array, both exceeded safe memory. The script loops chunk by chunk in fixed 8192-px row blocks (its own constant, not `postprocessing.block_size`) so the supersampled array stays bounded, and scores with the same `confusion_counts_soft`. Writes `{validation.output_dir}/england_fz3_rescore.csv`.

### Flood Zone 2 band — `validate_england_fz2_band.py`

EA Flood Zone 2 (sea) is the INCREMENTAL 0.1%-0.5% AEP band (between the ~1-in-200 Zone 3 outline and the 1-in-1000 outline), not the cumulative 1-in-1000 extent. The plain catalog comparison (`england_ea_floodzone2_sea` against the model's cumulative RP1000) is therefore a benchmark-definition mismatch. This script builds the model's own matching band — wet at RP1000 AND NOT wet at RP100 (RP100 is the nearest native RP below the benchmark's ~RP200 inner edge) — and soft-scores it against the same FZ2 polygons. Standalone rather than a `validate_country.py` code path because subtracting two RPs' wet masks is a one-off need unique to this benchmark. Writes `{validation.output_dir}/GBR/england_fz2_band_result.csv`.

## San Francisco Bay — `validate_sf_bay.py`

Benchmark: USGS CoSMoS flood depth, RP100, existing conditions (no SLR), 9 Bay Area counties. `build_sf_bay_benchmark_mosaic.py` (one-off) mosaics the county rasters (2m, EPSG:26910, int32 cm) into one Bay-wide float32 depth raster in metres.

Standalone because CoSMoS is a continuous DEPTH raster much finer than the model grid, not a vector extent — the soft-fraction machinery doesn't apply. The script scores at two resolutions:

- **A) coarsen** — CoSMoS depth averaged onto the model's native grid (~30m, EPSG:4326).
- **B) refine** — model depth bilinear-resampled onto CoSMoS's 2m grid.

Each direction gives plain confusion counts (HR, FAR, CSI, bias) at `primary_threshold_m` plus RMSE/MAE over the whole domain and over benchmark-flooded pixels only. Evaluation domain is the union of the 9 counties' rectangular raster extents (`sf_bay_county_bbox_union.gpkg`), not the raster's valid-pixel footprint: CoSMoS only carries values inside the mapped flood extent, so its footprint is ~100% wet by construction and would make FAR/bias vacuous. CoSMoS nodata inside the union is treated as "modelled, dry". Permanent water is excluded as everywhere else (CoSMoS depths include the Bay's own bathymetric depth). Results: `validation/sf_bay_results.csv` (with `--out-csv`).

On grid A the script also writes, to `{validation.output_dir}/USA/`, `agreement_USA_sf_bay_RP100_SLR_0.tif` (the country rasters' codes: 0 dry, 1 agree, 2 under, 3 over; 255 outside the domain), `depth_diff_USA_sf_bay_RP100_SLR_0.tif` (model minus CoSMoS, m, NaN outside the domain and where both are dry) and `metrics_USA_sf_bay_RP100_SLR_0.csv`. `plot_sf_bay.py` maps both rasters (same colours, background and metrics box as `plot_agreement_map.py`; depth difference on a symmetric ±`--diff-range-m` scale, default 1 m).

## Reporting tables

- `compare_against_published.py` → `validation/comparison_vs_published.csv`: our CSI per (sub-)region against Bates et al. (2023), Wing et al. (2024) and Chaudhuri et al. (2026), and SF Bay RMSE/MAE against Kasmalkar et al. (2024). Published values are transcribed, not re-derived.
- `build_overview_table.py` → `validation/overview_extent_metrics.csv` (HR/FAR/bias/CSI per region, with published CSI) and `validation/overview_sf_bay_depth.csv` (SF Bay RMSE/MAE, both directions). No new scoring — reshapes the outputs of `validate_country.py`, `rescore_england_fz3.py`, `validate_england_fz2_band.py` and `validate_sf_bay.py`.

## Orchestration and plots

`run_validation.py` discovers every country with a coastal benchmark (or takes `--countries GBR CAN`), then per country subprocess-dispatches `validate_country.py` followed by `plot_agreement_map.py` (module isolation — one country's crash doesn't abort the rest, unless `--fail-fast`; `--skip-plots` to skip plotting). After the loop it concatenates every country's `metrics_*.csv` into one `summary_{RP}_{SLR}.csv`. England FZ2/FZ3 and SF Bay are run via the standalone scripts above.

`validation/plot_agreement_map.py` is pure visualization over the already-written category rasters/CSVs (no recomputation).

## Key config (`config.yml`'s `validation:` block)

| Key | Value | Meaning |
|---|---|---|
| `return_period` / `waterlevel_name` | `RP100` / `SLR_0` | Single scalar, no CLI override — other RPs via `comparison_return_periods` + `run_multi_rp_summary.py`. |
| `primary_threshold_m` | `0.01` | The single model wet-cell threshold scoring uses — no sweep. Deliberately NOT `exposure.exceedance_threshold_m`; see "Metrics" above. |
| `benchmark_supersample` | `5` | Vector benchmark rasterization supersampling factor. |
| `benchmark_wet_fraction` | `0.5` | DISPLAY-only (agreement-map category raster). Not used for scoring. |
| `coastline_tolerance_cells` | `1` | Dilation radius (native 30m cells) for the `CSI_tol`/`HR_tol`/`FAR_tol` diagnostic — reported alongside, never instead of, the strict metrics. |
| `permanent_water_source` / `codes` | `deltadtm_mask` / `[1,2,3]` | Ocean/lake/river cells excluded from every domain. |
| `geogunit_source` / `iso_lookup_source` | WRI geogunit-107 / FLOPROS | Country/territory mask. |
| `plots.resolution_m` | `200` | Agreement-raster resolution for whole-territory regions only; sub-areas are written at native 30m (see "Outputs"). |

## Current status by region

| Region | Benchmark | Model RP(s) scored | Catalog key / script | Sub-regions |
|---|---|---|---|---|
| Wales (GBR) | NRW Flood Zone 3 (sea), 0.5% AEP (~1-in-200) | RP100, RP250 | `wales_nrw_floodzone_seas` | `severn_estuary`, `menai_strait` |
| Scotland (GBR) | SEPA coastal extent, Medium likelihood, present day (~1-in-200) | RP100, RP250 | `scotland_sepa_coastal_m` | `firth_of_forth` |
| England FZ3 (GBR) | EA Flood Zone 3 (sea + river-and-sea), ≥0.5% AEP | RP100, RP250 | `england_ea_floodzone3_sea`; `rescore_england_fz3.py` | — |
| England FZ2 (GBR) | EA Flood Zone 2, incremental 0.1-0.5% AEP band | RP1000 minus RP100 | `england_ea_floodzone2_sea`; `validate_england_fz2_band.py` | — |
| New Brunswick (CAN) | GeoNB present-day 1-in-100 (combined inland + coastal) | RP100 | `new_brunswick_present_day_1in100yr` | `cumberland_basin`, `petitcodiac` |
| San Francisco Bay (USA) | USGS CoSMoS depth, RP100, no SLR | RP100 | no catalog entry; `build_sf_bay_benchmark_mosaic.py` + `validate_sf_bay.py` | — |

Current CSI (from `validation/overview_extent_metrics.csv`; RP250 in brackets where scored):

| Region | CSI | Published CSI |
|---|---|---|
| Wales (whole) | 0.638 (0.663) | Bates et al. (2023) 0.76; Wing et al. (2024) 0.82 |
| Wales — Severn Estuary | 0.811 (0.824) | Chaudhuri et al. (2026) 0.70 |
| Wales — Menai Strait | 0.768 (0.781) | Chaudhuri et al. (2026) 0.49 |
| Scotland (whole) | 0.431 (0.446) | — |
| Scotland — Firth of Forth | 0.477 (0.517) | Chaudhuri et al. (2026) 0.75 |
| England FZ3 | 0.502 (0.498) | Wing et al. (2024) 0.62 |
| England FZ2 band | 0.021 | — |
| New Brunswick — Cumberland Basin | 0.007 | Chaudhuri et al. (2026) 0.87 |
| New Brunswick — Petitcodiac | 0.021 | Chaudhuri et al. (2026) 0.56 |
| San Francisco Bay (A coarsen / B refine) | 0.353 / 0.325 | — |

SF Bay depth errors against Kasmalkar et al. (2024) are in `validation/overview_sf_bay_depth.csv`. The FZ2 band's bias is sane (1.99, vs. 26x for the naive cumulative comparison) but its CSI stays low — plausibly the positional sensitivity of a thin band, not independently confirmed (see the note in `validation/comparison_vs_published.csv`). The near-zero Bay of Fundy CSI is the known forcing limitation below.

## Caveats

- **Permanent-water masking** — DeltaDTM's native mask (~25-31m) is much finer than the earlier Copernicus land_use (~100m), but a waterline-*definition* disagreement remains between two independently-drawn coastlines (DeltaDTM's optical water edge vs. a benchmark's surveyed coastline), not fixable by a better mask. The `CSI_tol` diagnostic shows how much of a region's strict CSI gap this kind of registration disagreement explains.
- **Return-period mismatch** — Wales, Scotland and England FZ3 are ~1-in-200 products with no native model RP; they are bracketed by RP100/RP250, never interpolated. More generally, a "100-year" event in one agency's convention is not necessarily statistically equivalent to this model's RP100.
- **New Brunswick is combined inland + coastal** — no attribute separates them, so inland-only benchmark flooding is an expected mismatch. The Bay of Fundy head (Cumberland Basin/Petitcodiac) shows severe under-prediction, believed to be COAST-RP/GTSM under-resolving Fundy's tidal resonance (boundary water levels cap at ~4.77m there), not an extent-algorithm or benchmark defect — not independently confirmed against tide-gauge extremes.
- **Every benchmark is itself a model** (its own DEM, defences assumptions, vintage) — "benchmark" means independent reference, not ground truth.
- **Defences and wave run-up are not represented by this model** — it simulates still-water coastal flooding only and applies flood-protection standards only in the separate exposure step, while benchmark methodologies may include run-up or reflect defences implicitly.
- Full per-benchmark detail (provenance, data-quality gaps, measured caveat magnitudes) lives in each catalog entry's `known_caveats` in `data_catalog_validation.yml`, next to the config/filter it describes.
