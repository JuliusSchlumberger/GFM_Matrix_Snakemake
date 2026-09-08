# Coastal flood-extent validation against national hazard maps

**Status:** design proposal — no code written yet.
**Written:** 2026-09-08, on the local development machine (which has only a small subset of
the modelled dataset — see §3). Intended to be reviewed on the machine that holds the full
model output before implementation starts.

---

## 1. Purpose and scope

Compare the modelled **RP100 flood extent under present-day conditions (SLR_0)** against
independent, authoritative national flood hazard maps, per country, and report:

- **HR** (hit rate), **FAR** (false alarm ratio), **CSI** (critical success index),
  **EB** (error bias) — the four requested metrics;
- an **exposure difference**: how much population the model over- and under-estimates as a
  result of the extent difference;
- a **map per country** with three colours: agreement, model over-prediction, model
  under-prediction.

Everything below assumes the comparison is **binary wet/dry**, since most benchmark maps are
extent polygons without depth.

---

## 2. What is actually being compared

### 2.1 The modelled side — measured, not assumed

These facts were established by reading the pipeline and opening the real rasters. They are
the things that are *not* obvious from the code alone, and several of them constrain the
design.

| Property | Value | Source |
|---|---|---|
| Product to compare against | `{root}/merged_results/waterdepth_{RP}_{SLR}.vrt`, a GDAL VRT over the per-chunk merged rasters | `rules/postprocessing.smk::build_mosaic_vrt` |
| CRS | EPSG:4326 | measured |
| Pixel size | `0.0004450221727793898°` = **1.6021 arcsec** — constant, not latitude-adaptive | measured on both merged chunks and per-tile outputs |
| Ground size | ~49.5 m N–S everywhere; E–W varies with latitude — **36 m at 43°N (Spain), 23 m at 62°N (Norway)** | derived |
| dtype / nodata | float32, `nodata = -9999` | measured |
| Value semantics | `-9999` = **outside the model domain** (no tile covers this cell); `0.0` = covered and dry; `>0` = flood depth in metres | `src/merge.py::merge_tile_rasters_chunk` (writes `raster_config["nodata"]` wherever `best_tile_id == PROVENANCE_NODATA`) |
| Combination rule across overlapping tiles | per-cell **maximum** depth | same function |
| Flood threshold used elsewhere in the pipeline | `exposure.exceedance_threshold_m = 0.10` m | `config/config.yml` |

**The `-9999` mask is the single most useful thing here.** It gives an exact, free answer to
"where did the model actually compute anything?", which is what makes a defensible evaluation
domain possible (§4.3). The model domain is *not* the whole country: tile footprints are
shaved to floodable coastal terrain (`tile_generation.shave_*` in `config.yml`), so high or
inland ground is simply absent.

> ⚠ **Two things to re-check on the full-data machine.**
>
> 1. **The 1.6021 arcsec grid does not obviously follow from
>    `simulation.flooding.resolution: 30`** (30 m at the equator would be ~0.97 arcsec).
>    Either the model grid is not what that config value implies, or the local outputs
>    predate the current setting. Worth confirming — it changes every area number by ~2.7×.
> 2. **The per-tile files on this machine are float32 with `nodata = 0.0`**, whereas the
>    current `src/rasters.py` writes int16-centimetres with `WATERDEPTH_NODATA_INT16 =
>    32767`. The local outputs are therefore from an older run. The validation reads the
>    *merged* product, not per-tile files, so this should not matter — but confirm the merged
>    chunks on the full machine are still float32 / `-9999`.

### 2.2 The benchmark side — inventory

Metadata-only inventory of `D:\validation_data\` (no geometry loaded):

| Country | Dataset | Format | CRS | Features | Notes |
|---|---|---|---|---|---|
| **Spain** | `Spain/laminas-q100/LAMINAS_Q100_MEDIA_ETRS89.shp` | Shapefile, Polygon | **EPSG:4258** | 397 | *Zonas Inundables de origen marino*, Q100 ("probabilidad media"). **Purely coastal — directly comparable.** Sum of `AREA_KM2` field = **2213 km²** |
| Spain | `Spain/laminas-q500/LAMINAS_Q500_BAJA_ETRS89.shp` | idem | EPSG:4258 | 397 | Q500 — a free second return period for validating RP500 later |
| **France** | `France/tri_2020_sig_di/n_inondable_03_02moy_s.shp` | Shapefile, Polygon | EPSG:4326 | 74 547 | TRI 2020. Layer naming is `n_inondable_{type}_{scenario}_s`: type **`03` = submersion marine**, scenario **`02moy` = moyenne ≈ 100 yr**. Confirm via the `typ_inond` / `scenario` attributes rather than trusting the filename |
| France | `…_03_03mcc_s.shp`, `…_03_01for_s.shp`, `…_03_04fai_s.shp` | idem | EPSG:4326 | 66 051 / 28 064 / 36 686 | medium+climate-change, frequent, rare |
| France | `…_01_*_s.shp`, `…_02_*_s.shp` | idem | EPSG:4326 | large | type 01/02 = **fluvial / runoff — must be excluded** |
| **Germany (NI)** | `HWRM_Deutschland/Niedersachsen/Gefahrengebiete_HQ100_Z2.shp` | Shapefile, Polygon | **EPSG:4647** | 42 | HQ100. **Largely fluvial** — check `FL_RECUR` / `APSFR_NAME` before using as a coastal benchmark |
| Germany (NI) | `…/HWRMRL_Z2_Kuestengebiete/Kuestengebiete_HWRMRL_Z2.shp` | Shapefile, Polygon | EPSG:4647 | 3 | Coastal *designation zones*, not a modelled extent — probably not a valid benchmark |
| **Germany (SH)** | `HWRM_Deutschland/Schleswig-Holstein/WFS_LRP_Karte3_2020` | **WFS 2.0 GML, no file extension** | in-file | ? | GDAL will not auto-detect it without a `.gml` extension or an explicit driver |
| **Japan** | `Japan/` | only `.xml` metadata + `url.txt` | — | — | **data not downloaded yet** |
| **Norway** | `Norway/url.txt` | — | — | — | **data not downloaded yet** |
| **Australia** | `Australia/url.txt` | — | — | — | **data not downloaded yet** |
| Events | `Events/…`, `HWM_…tif` | mixed, incl. a 13.9 MB GeoTIFF | — | — | Observed *event* footprints — a different validation question (single event vs. return period); out of scope here |

**Two CRS traps worth naming.** Spain is EPSG:4258 (ETRS89) — near-identical to WGS84 but
*not* the same authority code, so it must be transformed explicitly rather than assumed.
Germany is EPSG:4647 (ETRS89 / UTM 32N with a **32,000,000 m false easting** that encodes the
zone number in the coordinate); easting values look like `32362181`. pyproj handles it, but
any hand-rolled UTM assumption will not.

---

## 3. The coverage problem — read this before trusting any number

Spain's Q100 map covers **only surveyed coastal stretches**, not the entire coast. That means
a model-wet pixel outside a benchmark polygon may be a genuine false alarm *or* simply
somewhere nobody surveyed. Scoring naively conflates the two and makes FAR meaningless.

A probe on the one chunk where local model output and Spanish benchmark overlap
(`waterdepth_N40W005_RP100_SLR_0.tif`, 21 benchmark polygons) gave:

```
model domain cells : 13 501
benchmark wet      : 49 418   ← of which 45 426 (92%) fall OUTSIDE the model domain
model wet          : 13 336
TP = 3 954   FP = 9 382   FN = 38
HR = 0.990   FAR = 0.704   CSI = 0.296   EB = 0.996   (FP/FN = 247)
```

An HR of 0.99 looks excellent and is **an artefact**: 92% of the benchmark was excluded
before scoring, so HR was computed over the 8% the model happened to cover. This is what a
partial dataset does to these metrics, and it is why §4.3 and the coverage diagnostic in the
CSV exist.

**Local data reality:** this machine has **45 of 2578 tiles** with `RP100_SLR_0` results, and
**1 of 30** tiles overlapping mainland Spain (the 45 present are almost all Norwegian, ~62°N).
A real Spain validation is therefore **not possible here** — hence this document rather than
an implementation.

---

## 4. Method

### 4.1 Common grid: the model grid

All comparison happens on the **model's own grid** — the model raster is never resampled, so
no interpolation artefacts are introduced on the side being evaluated. Benchmark and
population data are brought *onto* that grid.

Rationale beyond simplicity: the model carries no real information below ~50 m. Upsampling it
to Spain's 5 m benchmark resolution would multiply the pixel count ~100× while manufacturing
apparent precision, and every 5 m sub-pixel within a model cell would carry an identical,
duplicated value.

### 4.2 Bringing the benchmark onto the model grid

- **Vector benchmarks** (Spain, France, Germany): rasterise at a **supersampling factor
  `s` (default 5)** within each block, then average the `s × s` sub-pixels to a **coverage
  fraction** per model pixel; wet if `fraction ≥ 0.5`.
  Plain `rasterize(all_touched=False)` systematically loses thin coastal strips, and
  `all_touched=True` systematically inflates them; the fraction is unbiased and costs only
  25× within a block. The retained fractions also give an **exact benchmark area** for the
  coverage diagnostic, independent of the 0.5 threshold.
- **Raster benchmarks finer than the model**: reproject the binary wet mask with
  `Resampling.average` → fraction → same 0.5 rule.
- **Raster benchmarks coarser than the model**: `Resampling.nearest`.

Wet-ness itself comes from the catalog entry (`wet_values`, `depth_threshold_m`, or "any
polygon"), so per-country semantics live in the catalog, not in `if country == …` branches.

### 4.3 Evaluation domain

Scored cells must satisfy **both**:

```
depth != -9999                        # the model actually computed here
AND distance_to_benchmark < buffer    # the benchmark actually surveyed here
```

The buffer (default **2 km**, configurable) is built **per block with
`scipy.ndimage.distance_transform_edt` on the block plus a halo** — *not* by buffering and
dissolving the polygon set globally. A global `buffer().union_all()` over Spain's 397
polygons was still running after 2 minutes on this machine before being killed; the
block-local distance transform is milliseconds.

Because the buffer width is a judgement call, the CSV also carries a
**`domain = model_only`** variant (`depth != -9999` alone). Reporting both makes FAR's
sensitivity to that judgement visible instead of hidden inside one number.

### 4.4 Metrics

Let TP / FP / FN / TN be **area-weighted** (km²) within the evaluation domain.

| Metric | Formula | Reading |
|---|---|---|
| HR | `TP / (TP + FN)` | share of benchmark-wet reproduced |
| FAR | `FP / (TP + FP)` | share of model-wet not in benchmark |
| CSI | `TP / (TP + FP + FN)` | 0 = no match, 1 = perfect |
| **EB** | `FP / (FP + FN)` | **> 0.5 over-predict, < 0.5 under-predict** (the requested form) |
| EB_ratio | `FP / FN` | Wing et al. (2017) form; > 1 = over-predict. Reported alongside because it is the form in the literature and is far more legible at extremes — the probe's EB = 0.996 and FP/FN = 247 are the same fact |
| bias | `(TP + FP) / (TP + FN)` | modelled wet area ÷ benchmark wet area |

**Area-weighted, not cell-counted.** In EPSG:4326 a pixel's ground area varies with latitude
(§2.1: 36 m vs 23 m E–W between Spain and Norway), so counting cells would silently
weight high-latitude coastline differently. Cell area uses the existing cos(lat) formula in
`src/plotting.py::compute_flood_area_km2`. Raw cell counts are reported too, for
reproducibility.

Zero denominators return `NaN`, never a silent `0` — "no benchmark wet area in this domain"
must not read as "CSI = 0".

### 4.5 Exposure difference

Population is `population` in `data_catalog_gfm.yml` (WorldPop 2020, ~30 arcsec ≈ 927 m,
**counts per cell, not a density**).

Requested behaviour is area-weighted disaggregation — splitting a coarse cell across the fine
cells it covers so each gets only its share. That is implemented *without* materialising a
50 m population raster, using the identity:

```
Σ over fine cells of class C of (pop_coarse / n_subcells)
    ≡  Σ over coarse cells of  pop_coarse × (fraction of that coarse cell in class C)
```

The right-hand side is exactly the existing **A×B average-pooling** in
`scripts/compute_flood_fraction_chunk.py`, applied to each class mask. So:

- `pop_over` = population in FP cells (model floods, benchmark does not) — **over-estimate**
- `pop_under` = population in FN cells (benchmark floods, model does not) — **under-estimate**
- `pop_diff_net` = `pop_over − pop_under` — the single net number requested

Reporting the two sides separately as well as the net matters: a net of ~0 can mean a very
good model or a badly displaced extent with cancelling errors.

Population-weighted **HR / FAR / CSI** are computed from the same quantities, i.e. weighting
cells by people rather than area. Since the model exists to estimate exposure, agreement in
*populated* areas is the more decision-relevant number.

> ⚠ `compute_flood_fraction_chunk.py` documents a subtle failure: combining the two
> `reproject()` calls into one two-band call corrupted ~360 000 cells, because `reproject`
> shares one validity mask across bands under `average` resampling. Any reuse **must keep the
> two calls separate.** Recommended approach: extract the A×B core into a shared
> `rasters.average_pool_to_grid()`, refactor the existing script onto it, and verify the
> output is bit-identical on a real chunk before trusting it.

### 4.6 Chunking and memory

Country → chunk grid via the existing `src/chunks.py::build_chunk_grid` (already the
Snakefile's own chunking logic) at a **`validation.chunk_size_deg` of 0.5°** — much smaller
than postprocessing's 5°, since these runs are interactive. Within a chunk, a block loop with
`block_size` (default 2048).

At the model's ~50 m grid this is cheap: a 2048² block is 4.2 M cells (~4 MB per boolean
mask), the buffer halo is `2 km / 50 m = 40` cells, and the accumulators hold only scalars
per (threshold × domain) plus one small plot-resolution grid. Peak working set is well under
1 GB. This machine killed a process with `ENOMEM` during exploration, so keeping
`block_size` in config rather than hardcoded is deliberate.

**All depth thresholds are evaluated in a single pass** — one depth read, four comparisons —
so the threshold-sensitivity sweep is close to free.

---

## 5. Proposed implementation

### 5.1 Files

New:

| Path | Purpose |
|---|---|
| `snakemake_workflow/config/data_catalog_validation.yml` | HydroMT catalog of benchmark maps + metadata; own `root: D:/validation_data` |
| `snakemake_workflow/src/validation.py` | Benchmark loading/dispatch, mask construction, metrics, accumulators |
| `snakemake_workflow/validation/validate_country.py` | Per-country chunked driver → metrics CSV + plot-resolution agreement raster |
| `snakemake_workflow/validation/plot_agreement_map.py` | Three-colour agreement map |
| `snakemake_workflow/validation/run_validation.py` | Entry point over all countries + cross-country summary |

Modified:

| Path | Change |
|---|---|
| `config/config.yml` | new `validation:` section (§5.3) |
| `src/rasters.py` | add `average_pool_to_grid()` (A×B core, §4.5) |
| `scripts/compute_flood_fraction_chunk.py` | refactor onto it — behaviour-preserving, verified bit-identical |

Layout mirrors the existing `snakemake_workflow/analysis/` package (standalone,
config-driven, one subprocess per step, `run_analysis.py`-style entry point) rather than
adding Snakemake rules, so it can be iterated on without the DAG.

**Reused rather than rewritten:** `config_utils.{load_config, get_data_catalog,
retry_transient_io, atomic_write}`, `chunks.build_chunk_grid`, the cos(lat) area formula in
`plotting.compute_flood_area_km2`, the land-polygon plot styling in
`plotting.plot_raster_with_coastlines`, and the A×B pooling in
`compute_flood_fraction_chunk.py`.

### 5.2 `data_catalog_validation.yml`

HydroMT-format so it loads through the existing `config_utils.get_data_catalog()`. It is
loaded as a **second, separate `DataCatalog`** because it needs its own `root` — the GFM
catalog's root override cannot be shared.

```yaml
root: "D:/validation_data"

spain_zi_marina_q100:
  data_type: GeoDataFrame
  driver: vector
  path: Spain/laminas-q100/LAMINAS_Q100_MEDIA_ETRS89.shp
  crs: 4258
  filesystem: local
  meta:
    category: flood_hazard_benchmark
    country: Spain
    country_iso: ESP
    hazard_type: coastal          # coastal | fluvial | mixed — only `coastal` is comparable
    return_period: 100
    variable: extent              # extent | depth
    native_resolution_m: 5        # from PRECISION = "MDT 5x5" (documentation only —
                                  # comparison happens on the model grid, see §4.1)
    # --- transformation rules read by load_benchmark() ---
    attribute_filter: {TIPO_ZONA: "Z.I. PROBABILIDAD MEDIA (100 AÑOS)"}
    exclude_bbox: [-19.0, 27.0, -13.0, 30.0]     # Canary Islands
    # raster sources instead use: wet_values: [1,2] | depth_threshold_m: 0.1 | nodata: 255
    # ---
    description: >
      SNCZI "Zonas Inundables de origen marino" — marine-origin flood zones for the
      Spanish coast, medium probability (100-year return period).
    methodology: >
      Still-water + wave levels from a 60-year C3E hindcast, run-up via IH2VOF
      formulae at profiles spaced 200 m, flooded onto a 5 m LiDAR DTM. 90% band.
    source_url: https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/costas-medio-marino/zi-origen-marino.html
    date_retrieved: "2026-09-08"
    source_license: <to confirm — see the MITECO download page>
    source_version: "2016-04-12"   # DBF_DATE_LAST_UPDATE
    known_caveats: >
      Covers only surveyed coastal stretches, NOT the whole coast — see §3.
      Includes wave run-up, which the GFM model does not simulate.
```

Adding France/Germany is then a catalog entry (plus at most one loader branch for the GML
case), not a new script.

### 5.3 `config.yml` additions

```yaml
validation:
  benchmark_catalog: "snakemake_workflow/config/data_catalog_validation.yml"
  benchmark_root: "D:/validation_data"
  output_dir: "{root}/validation"
  return_period: "RP100"
  waterlevel_name: "SLR_0"
  depth_thresholds_m: [0.05, 0.10, 0.25, 0.50]
  primary_threshold_m: 0.10          # matches exposure.exceedance_threshold_m
  benchmark_supersample: 5           # §4.2
  benchmark_wet_fraction: 0.5
  eval_domain:
    buffer_km: 2.0
    also_report_model_domain_only: true
  chunk_size_deg: 0.5
  block_size: 2048
  population_source: "population"
  write_fullres_agreement_raster: false
  plots:
    resolution_m: 200
    dpi: 300
    figsize: [10, 10]
    agreement_colors:
      agree: "#4daf4a"    # green — both wet
      over:  "#e41a1c"    # red   — model wet only
      under: "#377eb8"    # blue  — benchmark wet only
```

### 5.4 Outputs

**`{root}/validation/{country}/metrics_{country}_RP100_SLR_0.csv`** — one row per
(country × threshold × domain variant):

```
country, iso, benchmark_key, return_period, waterlevel_name, model_resolution_m,
threshold_m, domain,
tp_km2, fp_km2, fn_km2, tn_km2, tp_cells, fp_cells, fn_cells, tn_cells,
benchmark_wet_km2, model_wet_km2,
benchmark_wet_outside_model_domain_km2, benchmark_wet_outside_model_domain_pct,
HR, FAR, CSI, EB, EB_ratio, bias,
pop_tp, pop_fp, pop_fn, pop_benchmark, pop_model,
pop_over, pop_under, pop_diff_net, pop_HR, pop_FAR, pop_CSI
```

`benchmark_wet_outside_model_domain_pct` is the run's health check. At the 92% seen in §3,
nothing else in the row should be quoted as a model result.

**`{root}/validation/{country}/agreement_{country}_RP100_SLR_0.{tif,png}`** — the three-colour
map, written at `plots.resolution_m` (default 200 m). The full-resolution categorical raster
stays **off by default**; it is multi-GB per country and only useful for GIS inspection.

**Plot detail that matters:** each reduced pixel is assigned by **priority — over > under >
agree > dry**, not by `mode` or `average`. Coastal flood extents are thin; averaging a 50 m
result to 200 m for display would erase exactly the disagreement the map exists to show.

Colours are ColorBrewer Set1 (green/red/blue), chosen to stay distinguishable for the most
common colour-vision deficiencies — an orange/red pairing for over/under would not.

---

## 6. Verification plan

Correctness cannot rest on a Spain run here (§3), so:

1. **Metric unit tests** — `metrics_from_counts` against hand-computed contingency tables,
   including every degenerate case (no benchmark wet, no model wet, perfect match). Assert
   `EB == 0.5` exactly when `FP == FN`, and `NaN` (not `0`) where undefined.
2. **Synthetic end-to-end** — a small fake waterdepth GeoTIFF plus a fake benchmark shapefile
   with an analytically known overlap and a known population field. Assert every metric and
   the population over/under split against closed-form values. This is what proves the
   area-weighted population disaggregation and the supersampled rasterisation are exact.
3. **Real partial run** — `--country spain` on the single overlapping chunk (`N40W005`).
   Exercises catalog dispatch, the EPSG:4258 → 4326 transform, chunking, plotting, and
   confirms the coverage diagnostic reports ~92% rather than emitting a confident-looking CSV.
4. **Refactor safety** — run `compute_flood_fraction_chunk.py` before and after the
   `average_pool_to_grid` extraction on a real chunk; assert **bit-identical** output (§4.5).

Per project convention, test scripts, fixtures and outputs go in
`C:\Users\Schlu005\GFM\tests\flood_extent_validation\`; production code lives in
`snakemake_workflow/validation/`.

---

## 7. To confirm on the full-data machine

1. The **1.6021 arcsec grid** vs `simulation.flooding.resolution: 30` (§2.1) — this changes
   every area figure.
2. Merged chunks are still **float32 / `nodata = -9999`** (§2.1).
3. **How much of each benchmark actually falls inside the model domain.** If it is low for a
   country, that country cannot be validated this way, and no metric tuning will fix it.
4. Whether the **2 km buffer** is the right default once real coverage is visible — it is
   currently a guess, and FAR is directly sensitive to it.
5. France: that `typ_inond`/`scenario` attributes confirm `03_02moy` is marine-submersion
   medium-probability, rather than relying on the filename convention.
6. Germany: whether `Gefahrengebiete_HQ100_Z2` is coastal enough to be a valid benchmark at
   all, and how to read the extension-less Schleswig-Holstein GML.

---

## 8. Known methodological caveats

- **Spain's benchmark includes wave run-up** (IH2VOF); the GFM model simulates still-water
  coastal flooding only. This is a genuine physical difference and should show up as
  under-prediction near exposed, high-energy coasts. It should be stated whenever these
  numbers are reported — it is not a model defect.
- **Benchmark maps are themselves models**, with their own DEMs, defences assumptions and
  vintages (Spain's is from 2014–2016). "Benchmark" here means "independent reference", not
  "truth".
- **Defences.** National maps typically account for real dikes; the GFM baseline applies
  FLOPROS protection standards only in the *exposure* step, not in the extent. Comparing raw
  extents therefore compares two different things wherever defences matter — most of the
  Dutch/German/French North Sea coast. Worth deciding explicitly before quoting numbers there.
- **Return-period comparability.** "Q100 marine" and "RP100 storm tide" are not guaranteed to
  be the same event definition across national methodologies.

## 9. Suggested but deferred

- **Fuzzy / tolerance CSI** — count a model-wet cell as a hit if a benchmark-wet cell lies
  within *N* metres. This stops a ~50 m model being penalised for sub-pixel georegistration
  against a 5 m benchmark, and usually separates "wrong place" from "slightly offset".
- **Per-sub-region breakdown** — metrics per Spanish *demarcación hidrográfica*
  (`DEMARCACIO` field) or French TRI (`id_tri`) instead of one national number, which averages
  away large regional differences.
- **Depth comparison** where the benchmark carries depths (France's `n_iso_ht_*` layers have
  `ht_min`/`ht_max` bands) — enabling RMSE/bias on depth, not just extent agreement.
