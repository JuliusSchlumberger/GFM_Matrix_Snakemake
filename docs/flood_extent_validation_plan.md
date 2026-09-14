# Coastal flood-extent validation against national hazard maps

**Status:** design proposal — no code written yet (except the unrelated DEM-VRT resolution fix
in §7.1, applied directly since it's a pipeline-wide correctness issue independent of this
plan).
**Written:** 2026-09-08, on the local development machine (which had only a small subset of
the modelled dataset — see §3).
**Reviewed:** 2026-09-08/09, on the full-data machine — §2.1, §3, §5.2/§5.3, and §7 updated
with real measurements; §3's "not possible here" conclusion no longer applies on this machine.

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
| Product to compare against | `{root}/merged_results/chunks/waterdepth_{chunk_id}_{RP}_{SLR}.tif` — **per-chunk files directly, not a VRT** (see §7.1: no mosaic has been built yet) | measured on the full-data machine |
| CRS | EPSG:4326 | measured |
| Pixel size | **Anisotropic, not the previously-assumed isotropic 1.6021″.** Y (N–S) = exactly 1 arcsec (~30.9 m) everywhere. X (E–W) is latitude-dependent in the *native* DEM tiles (1″ up to ~40°N, 2″ ~60°N, 3″ ~70–76°N, 5″ ~80–83°N — the highest-latitude tile present), but was being flattened into one blended, latitude-blind ~1.678″ value by the merged/simulation grid — see §7.1, now fixed at the source | measured directly on real tiles, root-caused, fixed |
| Ground size | Y ≈ 30.9 m N–S everywhere (not 49.5 m). X varies both by true native latitude-tier *and*, before the §7.1 fix, by an accidental blended value on top of that — do not quote a single "ground size" number; compute it per-tile from that tile's own transform | corrected |
| dtype / nodata | float32, `nodata = -9999` | measured |
| Value semantics | `-9999` = **outside the model domain** (no tile covers this cell); `0.0` = covered and dry; `>0` = flood depth in metres | `src/merge.py::merge_tile_rasters_chunk` (writes `raster_config["nodata"]` wherever `best_tile_id == PROVENANCE_NODATA`) |
| Combination rule across overlapping tiles | per-cell **maximum** depth | same function |
| Flood threshold used elsewhere in the pipeline | `exposure.exceedance_threshold_m = 0.10` m | `config/config.yml` |

**The `-9999` mask is the single most useful thing here.** It gives an exact, free answer to
"where did the model actually compute anything?", which is what makes a defensible evaluation
domain possible (§4.3). The model domain is *not* the whole country: tile footprints are
shaved to floodable coastal terrain (`tile_generation.shave_*` in `config.yml`), so high or
inland ground is simply absent.

> ✅ **Resolved on the full-data machine (2026-09) — see §7.1 for the full trace.** The
> pixel-resolution question above was real, but not what it looked like: the merged product's
> per-cell **area formula is unaffected** (`plotting.compute_flood_area_km2` already reads each
> raster's own transform per-axis and applies `cos(lat)` correctly — confirmed by reading it,
> not assumed), so §4.4's area-weighting can proceed exactly as designed. What *was* wrong has
> been fixed at the DEM-preparation source (`preparation/build_deltadtm_vrt.py`), independent of
> this validation work. The `nodata`/dtype/max-combine claims below were all confirmed exactly
> as stated against real merged chunks.

### 2.2 The benchmark side — inventory

Metadata-only inventory, originally of `D:\validation_data\` on the dev machine. **On the
full-data machine, the real location is `P:\11212688-004-global-floodmaps\modelling\inputs\
validation\`, and the layout differs from what's assumed below**: country folders are
**ISO3 codes** (`ESP\`, not `Spain\`), and there is no `laminas-q100`/`laminas-q500` subfolder
— the shapefile sits directly under the country folder
(`validation\ESP\LAMINAS_Q100_MEDIA_ETRS89.shp`). CRS confirmed EPSG:4258 (ETRS89) directly
from the `.prj` file. All shapefile sidecars (.dbf/.shx/.prj/.sbn/.sbx) plus a `.shp.xml` are
present and the data looks complete, not a stub. **Q500 is not downloaded at all yet** — no
files, not even a placeholder `url.txt` for it specifically — so it is not currently a "free"
anything; treat it as a future addition once it lands.

| Country | Dataset | Format | CRS | Features | Notes |
|---|---|---|---|---|---|
| **Spain** | `ESP/LAMINAS_Q100_MEDIA_ETRS89.shp` (real path — see note above) | Shapefile, Polygon | **EPSG:4258**, confirmed | 397 | *Zonas Inundables de origen marino*, Q100 ("probabilidad media"). **Purely coastal — directly comparable.** Sum of `AREA_KM2` field = **2213 km²** |
| Spain | not yet downloaded | — | — | — | Q500 — would be a second return period for validating RP500 later, once available |
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
CSV exist. This mechanism is real and stays the design driver for §4.3 regardless of the
correction below — but the specific 92%/HR=0.99 numbers above were measured on the one
unrepresentative chunk available on the dev machine and need re-running here before being
quoted or used to justify a buffer width.

> ✅ **Corrected on the full-data machine (2026-09).** This machine has **851 chunks** with
> `RP100_SLR_0` results (not 45), and Spain's full mainland bbox (lon −9.5..3.5, lat 36..44) is
> covered by 8 real chunks (`waterdepth_N35{E000,E005,W005,W010}` /
> `N40{E000,E005,W005,W010}_RP100_SLR_0.tif`), not the single sliver the dev machine had. **A
> real Spain validation is possible here** — the "not possible here, hence a document rather
> than an implementation" conclusion below no longer applies; it was a dev-machine data
> limitation, not a property of the method.

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

> ✅ **Redesigned 2026-09** (see §4.6a) — the description below (raster distance-transform)
> was the ORIGINAL design and is superseded. Kept here for the record; the live
> implementation buffers the benchmark's own polygons directly instead.

Scored cells must satisfy **both**:

```
depth != -9999                        # the model actually computed here
AND permanent_water == False          # not a river/lake/sea pixel (§4.6a)
AND cell inside the buffered benchmark-cluster polygon   # the benchmark actually surveyed here
```

~~The buffer (default **2 km**, configurable) is built **per block with
`scipy.ndimage.distance_transform_edt` on the block plus a halo** — *not* by buffering and
dissolving the polygon set globally. A global `buffer().union_all()` over Spain's 397
polygons was still running after 2 minutes on this machine before being killed; the
block-local distance transform is milliseconds.~~ Superseded — see §4.6a: the "2-minute
hang" turned out to be one degenerate 596,191-vertex record, not an inherent limit: with
`simplify(100m, preserve_topology=False)` + `buffer(quad_segs=2)` + `union_all()`, the exact
buffer+dissolve this paragraph rejected runs on the real 397-polygon Spain benchmark in
well under a second, and is now the live implementation, used directly as each evaluation
cluster's own domain (no distance transform, no halo, no approximation).

Because the buffer width is a judgement call, the CSV also carries a
**`domain = model_only`** variant (`depth != -9999` alone, still land-use-masked). Reporting
both makes FAR's sensitivity to that judgement visible instead of hidden inside one number.

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
and, per §2.1/§7.1, its E–W size also varies by which native DeltaDTM latitude-tier a given
tile came from — so counting cells would silently weight coastline differently by both
latitude and tile provenance. Cell area uses the existing cos(lat) formula in
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

> ✅ **Superseded 2026-09 by §4.6a** — this section (uniform 0.5° grid of blocks) was the
> ORIGINAL design, implemented, and then replaced after real profiling showed it was both
> slow and structurally wasteful. Kept for the record.

Country → chunk grid via the existing `src/chunks.py::build_chunk_grid` (already the
Snakefile's own chunking logic) at a **`validation.chunk_size_deg` of 0.5°** — much smaller
than postprocessing's 5°, since these runs are interactive. Within a chunk, a block loop with
`block_size` (default 2048).

At the model's ~30–50 m grid (§2.1/§7.1 — varies by axis and tile latitude-tier, treat "50 m"
here as an order-of-magnitude planning figure, not a precise constant) this is cheap: a 2048²
block is 4.2 M cells (~4 MB per boolean mask), the buffer halo is roughly `2 km / 50 m ≈ 40`
cells, and the accumulators hold only scalars
per (threshold × domain) plus one small plot-resolution grid. Peak working set is well under
1 GB. This machine killed a process with `ENOMEM` during exploration, so keeping
`block_size` in config rather than hardcoded is deliberate.

**All depth thresholds are evaluated in a single pass** — one depth read, four comparisons —
so the threshold-sensitivity sweep is close to free.

### 4.6a Redesign: benchmark-geometry clusters, not a uniform grid (2026-09)

**Why.** The 0.5° grid did real, wasted work: most of a benchmark's bounding box has no
benchmark data at all (§3 — only surveyed coastal stretches), so most blocks paid the full
cost (window read, per-block distance transform, ~64 `reproject()` calls per block for
population disaggregation across 4 thresholds × 2 domains × 4 classes) for near-empty
content, and a single contiguous benchmark polygon routinely got sliced across several
blocks, each independently recomputing an overlapping slice of the same buffer geometry. A
real run against this design was still processing 504 blocks for Spain with no end in sight.

**What changed.** Evaluation units now come directly from the benchmark's own geometry:

1. Load the country's full benchmark (all rows, filtered by `attribute_filter`/`exclude_bbox`).
2. `src/validation.py::build_evaluation_clusters`: reproject to an estimated local UTM zone
   (`gdf.estimate_utm_crs()`), `simplify(tolerance=100m, preserve_topology=False)`,
   `buffer(buffer_km, quad_segs=2)`, `union_all()`. The result's individual polygons ARE the
   clusters — geometries that overlap after buffering merge into one polygon automatically;
   isolated ones stay separate. **This is exactly the buffer+dissolve §4.3 rejected as a
   2-minute hang** — the hang turned out to be one degenerate 596,191-vertex/27,189-hole
   record (a likely raster-to-vector artifact), not an inherent limit; with
   `preserve_topology=False` simplification (geopandas' own default topology-preserving
   simplify is itself pathologically slow on that same record) and a coarse buffer curve,
   the whole 397-polygon Spain benchmark clusters into 127-182 polygons (182 pre-land-use-
   filtering; 127 after `attribute_filter`/`exclude_bbox`) in under a second.
3. Each cluster's own bounding box is its working extent — no halo-padding bookkeeping,
   since the buffer is already baked into the geometry. `rasterio.merge.merge()` mosaics
   whichever postprocessing chunk(s) the bbox touches (a cluster is no longer guaranteed to
   fit inside one 5° chunk the way a 0.5° block was).
4. The cluster polygon is rasterized directly onto the local grid as the evaluation domain
   (`rasterize(cluster_geom, ...)`) — no distance transform, no approximation.
5. **New: permanent-water exclusion.** `validation.permanent_water_mask` + the `land_use`
   catalog entry (Copernicus Global Land Cover, confirmed codes `80 = Permanent water
   bodies`, `200 = Open sea` against `inputs/Copernicus/lu_to_roughness_lookup.csv`) excludes
   river/lake/sea pixels from the domain entirely, on both the `buffered` and `model_only`
   variants — "is this pixel flooded" is meaningless over water that's always wet regardless
   of any storm event. Added after real QGIS inspection of the benchmark vs. model rasters
   surfaced this gap during design review; confirmed load-bearing on real data (one spot-
   checked cluster: 36 open-sea pixels correctly excluded that would otherwise have scored as
   false-positive "model floods the sea").

**Verified on real production data (Spain, RP100/SLR_0, 2026-09):** 397 shapefile rows → 396
pass `attribute_filter` → 258 after excluding the Canaries → **127 clusters, all with real
model data**. Whole-country run: **1m44s** (vs. the old design, which had not finished after
substantially longer on the same machine). `benchmark_wet_km2` = 1772 km² (excludes the
Canaries, consistent with the shapefile's national ~2213 km² total). Primary threshold
(0.10 m, buffered domain): HR=0.928, FAR=0.153, CSI=0.795, EB=0.700, bias=1.096 — valid,
in-range, and moving the expected direction across the threshold sweep.

Config changes: `validation.chunk_size_deg`/`block_size` removed (no longer meaningful);
added `vector_simplify_tolerance_m` (100.0), `buffer_quad_segs` (2), `landuse_source`
(`"land_use"`), `permanent_water_codes` (`[80, 200]`) - **superseded 2026-09**: renamed to
`permanent_water_source`, default switched to `"deltadtm_mask"`/`[1, 2, 3]` - see caveats
doc §1.3 for why.

---

## 5. Proposed implementation

### 5.1 Files

New:

| Path | Purpose |
|---|---|
| `snakemake_workflow/config/data_catalog_validation.yml` | HydroMT catalog of benchmark maps + metadata; own `root` (§5.2 — `{root}/inputs/validation` on this machine) |
| `snakemake_workflow/src/validation.py` | Benchmark loading/dispatch, mask construction, metrics, accumulators |
| `snakemake_workflow/validation/validate_country.py` | Per-country driver (evaluation-cluster based, §4.6a) → metrics CSV + plot-resolution agreement raster |
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
root: "P:/11212688-004-global-floodmaps/modelling/inputs/validation"

spain_zi_marina_q100:
  data_type: GeoDataFrame
  driver: vector
  path: ESP/LAMINAS_Q100_MEDIA_ETRS89.shp
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
    # Named sub-regions covering the whole benchmark (2026-09 - see the caveats doc's
    # §3.1): every evaluation cluster is tagged by which region's bbox its centroid
    # falls in, and metrics/plots are reported per region instead of one blended
    # national row. Previously the Canaries were dropped entirely via exclude_bbox;
    # now scored as their own region instead.
    regions:
      mainland: [-9.5, 35.0, 4.5, 44.0]          # continental Spain + Balearic Islands
      canary_islands: [-19.0, 27.0, -13.0, 30.0]
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

**Folder convention for multi-dataset countries.** Since Q500 isn't downloaded yet, this is a
good moment to fix a layout before more data lands: `{benchmark_root}/{ISO3}/{dataset_slug}/`
(e.g. `ESP/zi_marina_q100/LAMINAS_Q100_MEDIA_ETRS89.shp`, `ESP/zi_marina_q500/...` once
downloaded) rather than dumping every dataset flat under the country folder — avoids
reshuffling the catalog again when Q500/France/Germany land.

### 5.3 `config.yml` additions

```yaml
validation:
  benchmark_catalog: "snakemake_workflow/config/data_catalog_validation.yml"
  benchmark_root: "{root}/inputs/validation"  # NOT the GFM catalog's own {root} - see §5.2
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
  vector_simplify_tolerance_m: 100.0 # §4.6a - simplify(preserve_topology=False) tolerance before buffering
  buffer_quad_segs: 2                # §4.6a - coarse buffer curve resolution
  population_source: "population"
  permanent_water_source: "deltadtm_mask"  # §4.6a - permanent-water exclusion; renamed from
  # `landuse_source` and switched from "land_use"/[80,200] to DeltaDTM's own native mask 2026-09
  # - see caveats doc §1.3
  permanent_water_codes: [1, 2, 3]
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
(country × region × threshold × domain variant). `region` (2026-09, see caveats doc §3.1) is
whichever named bbox in the benchmark's `meta.regions` a cluster's centroid falls in (e.g.
Spain's `mainland`/`canary_islands`), or the country's own ISO code if `regions` isn't
configured for it yet:

```
country, iso, benchmark_key, region, return_period, waterlevel_name, model_resolution_m,
threshold_m, domain,
tp_km2, fp_km2, fn_km2, tn_km2, tp_cells, fp_cells, fn_cells, tn_cells,
benchmark_wet_km2, model_wet_km2,
benchmark_wet_outside_model_domain_km2, benchmark_wet_outside_model_domain_pct,
model_total_wet_km2, pop_model_total,
HR, FAR, CSI, EB, EB_ratio, bias,
pop_tp, pop_fp, pop_fn, pop_benchmark, pop_model,
pop_over, pop_under, pop_diff_net, pop_HR, pop_FAR, pop_CSI
```

`model_total_wet_km2`/`pop_model_total` (2026-09) are the model's own flooded area/exposed
population, COUNTRY-WIDE (not per-region - same value on every region-row of a given country),
at the single fixed exposure threshold (`exposure.exceedance_threshold_m`) - a genuine
denominator to compare the cluster-window-scoped `model_wet_km2`/`pop_model` against (mitigates
§1.2 of the caveats doc). Computed by the general analysis pipeline's own
`analysis/compute_flood_totals.py` (streams every populated chunk once, aggregates per country
via the same WRI geogunit + FLOPROS ISO lookup `compute_exposure_analysis.py` already uses -
exact country membership, no bbox approximation), written to
`validation.flood_totals_dir/flood_totals_{ISO}.csv`, and just READ here - not recomputed per
validation run. NaN at every threshold row except the primary one (the underlying pipeline
output only exists at that one fixed threshold), and NaN entirely until
`analysis/run_analysis.py` has been run with `analysis.compute_flood_totals` enabled.

`benchmark_wet_outside_model_domain_pct` is the run's health check. At the ~92% seen in §3's
single-chunk dev-machine probe (stale — re-measure per §7.2 point 1), nothing else in the row
should be quoted as a model result; the same rule applies whatever the real full-coverage
number turns out to be.

**`{root}/validation/{country}/agreement_{country}_{region}_RP100_SLR_0.tif`** — one
three-colour category raster PER REGION (2026-09), written at `plots.resolution_m` (default
200 m) only. A full-resolution categorical raster was considered (multi-GB per country, only
useful for GIS inspection) but was never implemented - `_write_agreement_raster` always
downsamples; there is no config toggle for it. One raster per region rather than one combined
per country because regions can be geographically far apart (mainland Spain vs. the Canary
Islands) - a combined mosaic would be huge and mostly empty.

**`{root}/validation/{country}/agreement_{country}_RP100_SLR_0.png`** — the combined plot:
`plot_agreement_map.py` draws the largest region's raster as the main map and every other
region as a small inset panel in the same figure, rather than one PNG per region.

**Plot detail that matters:** each reduced pixel is assigned by **priority — over > under >
agree > dry**, not by `mode` or `average`. Coastal flood extents are thin; averaging a 50 m
result to 200 m for display would erase exactly the disagreement the map exists to show.

Colours are ColorBrewer Set1 (green/red/blue), chosen to stay distinguishable for the most
common colour-vision deficiencies — an orange/red pairing for over/under would not.

---

## 6. Verification plan

Correctness shouldn't rest on a Spain run alone even though one is now genuinely possible
(§3), so:

1. **Metric unit tests** — `metrics_from_counts` against hand-computed contingency tables,
   including every degenerate case (no benchmark wet, no model wet, perfect match). Assert
   `EB == 0.5` exactly when `FP == FN`, and `NaN` (not `0`) where undefined.
2. **Synthetic end-to-end** — a small fake waterdepth GeoTIFF plus a fake benchmark shapefile
   with an analytically known overlap and a known population field. Assert every metric and
   the population over/under split against closed-form values. This is what proves the
   area-weighted population disaggregation and the supersampled rasterisation are exact.
3. **Real run** — `--country spain`, now genuinely coverable by the real 8-chunk mainland bbox
   (§3), not just one sliver. Exercises catalog dispatch, the EPSG:4258 → 4326 transform,
   chunking, and plotting end-to-end, and reports the real (not dev-machine-stale)
   benchmark-outside-model-domain percentage rather than emitting a confident-looking CSV
   without checking it.
4. **Refactor safety** — run `compute_flood_fraction_chunk.py` before and after the
   `average_pool_to_grid` extraction on a real chunk; assert **bit-identical** output (§4.5).

Per project convention, test scripts, fixtures and outputs go in
`{code_root}/tests/flood_extent_validation/` (machine-specific `code_root` — this repo's own
checkout, not a hardcoded path; on the dev machine that was `C:\Users\Schlu005\GFM`, on the
full-data machine it's wherever this repo is checked out); production code lives in
`snakemake_workflow/validation/`.

---

## 7. To confirm on the full-data machine

### 7.1 Pixel resolution — resolved (2026-09)

The 1.6021″-isotropic assumption was wrong, but so was assuming it was a benign local
peculiarity. Full trace:

- **Real native DeltaDTM tile resolution is latitude-tiered, not isotropic**: Y (N–S) is a
  constant 1 arcsec everywhere; X (E–W) coarsens 1″ (≤~40°N) → 2″ (~60°N) → 3″ (~70–76°N) → 5″
  (~80–83°N, the highest-latitude tile present) to compensate longitude convergence near the
  poles. Confirmed directly on real `.tif` tiles at 7 latitudes, and DEM/mask tiles match each
  other exactly at every one — no cross-product misalignment risk.
- **Root cause of the blended 1.678″ value**: `preparation/build_deltadtm_vrt.py` builds both
  `deltadtm.vrt` and `deltadtm_mask.vrt` via `gdal.BuildVRT()` without setting `resolution=`,
  so GDAL defaulted to `resolution="average"` — one blended X/Y pair, averaged across the
  *entire* global tileset, baked into each VRT's single `GeoTransform` and applied uniformly to
  every read regardless of the query's actual latitude. Confirmed by reading both live VRTs'
  `GeoTransform` directly (identical to ~15 significant digits) and by their `<ComplexSource
  resampling=...>` / fractional `DstRect` structure, which is the fingerprint of a locally
  `gdal.BuildVRT`-built mosaic. (A now-superseded `deltadtm.vrt.bak_before_relative_fix`
  confirms `deltadtm.vrt` *was* once a downloaded, path-patched 4TU mosaic historically — but
  that file was already replaced by a local rebuild before this fix, so both VRTs share the
  same current bug for the same reason, not different origins.)
- **Fix applied**: `_build_vrt()` in `build_deltadtm_vrt.py` now passes `resolution="highest"`,
  so both VRTs' shared grid matches the finest native resolution (1″) instead of the blended
  average — the vast majority of reads (everywhere outside the arctic/antarctic fringe) get
  their true native data with zero resampling loss; only the already-coarser high-latitude
  tiles get upsampled onto it. Regenerate both VRTs with:
  `python snakemake_workflow/preparation/run_preparation.py build_deltadtm_vrt`. This changes
  what *new* reads see; it does not retroactively fix already-computed simulation output —
  whether/when to re-run preprocessing+simulation to benefit from it is a separate decision,
  out of scope for this validation effort.
- **Consequence for this plan**: none needed. `plotting.compute_flood_area_km2` already derives
  per-cell area from each raster's own transform, per axis, with `cos(lat)` applied correctly —
  confirmed by reading it — so §4.4's area-weighting works whether or not the VRT fix has been
  applied yet, and regardless of which pixel grid a given merged chunk was produced under.

### 7.2 Still open

1. **How much of each benchmark actually falls inside the model domain**, re-measured on this
   machine's real 851-chunk coverage (§3's 92% figure is stale — dev-machine-only). If it is
   low for a country, that country cannot be validated this way, and no metric tuning will fix
   it.
2. Whether the **2 km buffer** is the right default once real coverage is visible — it is
   currently a guess, and FAR is directly sensitive to it.
3. France: that `typ_inond`/`scenario` attributes confirm `03_02moy` is marine-submersion
   medium-probability, rather than relying on the filename convention.
4. Germany: whether `Gefahrengebiete_HQ100_Z2` is coastal enough to be a valid benchmark at
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
- **Finer-grained sub-region breakdown** — metrics per Spanish *demarcación hidrográfica*
  (`DEMARCACIO` field) or French TRI (`id_tri`) instead of one national number, which averages
  away large regional differences. Distinct from (and finer than) the `regions:` mechanism
  implemented 2026-09 (§3.1 of the caveats doc), which only separates a country's main
  territory from its geographically-distant outlying ones (e.g. mainland Spain vs. Canary
  Islands) — this idea is about breaking down the *mainland* number further still.
- ~~**Depth comparison** where the benchmark carries depths (France's `n_iso_ht_*` layers have
  `ht_min`/`ht_max` bands) — enabling RMSE/bias on depth, not just extent agreement.~~
  Implemented 2026-09 as an agree/under/over-predict comparison against the benchmark's own
  bands (not RMSE, since the benchmark itself is banded, not a point depth) -
  `validate_country_depth_bands` in `validate_country.py`; see caveats doc §3.2 for the full
  design (band-membership framing, open-ended-sentinel handling, real gaps found).
