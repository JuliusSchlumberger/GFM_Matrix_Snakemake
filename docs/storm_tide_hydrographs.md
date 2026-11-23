# Storm-tide boundary forcing: return levels and per-RP hydrographs

Review brief for the coastal boundary conditions of the Global Flood Model (GFM). It covers how
the production storm-tide product is built from FES2022, FES2014b, GESLA-4.1, GTSM-ERA5 and
COAST-RP, how it is used by the flood models, and the code that implements it. Section 7 lists
the points we would most like reviewed.

## 1. Summary

For every COAST-RP station (23,226 points along the global coast; 22,670 north of 60°S are
used) and every return period RP ∈ {2, 5, 10, 25, 50, 100, 250, 500, 1000} years, the product
gives:

1. a **storm-tide return level** (m above local mean sea level), and
2. a **storm-tide hydrograph**: a 10-min time series around the peak whose maximum equals that
   return level.

The levels come from a joint-probability (skew surge ⊗ tide) analysis: GTSM-ERA5 skew surges,
corrected towards tide gauges, convolved with the FES astronomical tide; the tropical-cyclone
(TC) contribution is added as an exceedance rate from COAST-RP's TC component. The hydrographs
are synthetic: an FES tide window around a representative high water plus a normalised surge
shape taken from the station's own GTSM annual-maximum events of matching empirical RP, scaled
so that the combined peak equals the return level.

| Role in the flood models | What is used |
|---|---|
| Eikonal solver: coastline seed levels and static sea level | return levels (stations + coastline nodes), + MDT + 2020 offset + SLR fingerprint |
| Eikonal duration correction ("correction B") | per-RP hydrograph of the nearest station, only its shape near the peak |
| SFINCS (calibration/validation of the eikonal solver) | RP100 hydrograph, shifted so its peak equals the tile boundary's RP100 level |

## 2. Data sources and their roles

| Dataset | Version / period | Used for |
|---|---|---|
| **GTSM-ERA5** storm-surge residual (Copernicus CDS, hourly, ~43,000 output points) | v3 reanalysis, 1950-2024 | surge time series at each station (nearest output point, all within 0.8 km): skew surges for the return levels; annual-maximum events for the hydrograph shapes. GTSM's own tide is **not** used (it is 7-8% too high on average, §5). |
| **FES2022b** ocean tide (extrapolated grids) | - | default astronomical tide at stations, coastline nodes and gauges (34 constituents via pyTMD; Sa/Ssa removed, they belong to the surge). |
| **FES2014b** ocean tide (extrapolated grids) | - | replaces FES2022 where FES2022 has a known artefact (version rule, §3.2): 66 stations. |
| **GESLA-4.1** tide gauges | coastal, "no obvious issues", deduplicated at 1 km | (a) surge ratio r (gauge vs GTSM skew surge); (b) amplitude correction of the FES tide; (c) TC surge shapes; (d) veto on the FES version rule; (e) reference RP100 levels for the TC basin shift; (f) validation. |
| **COAST-RP** (Dullaart et al., 2021) total, `_ETC`, `_TC` | - | station locations, order and IDs (the product keeps COAST-RP's layout); TC component as an exceedance rate; fallback levels at 15 stations with a broken GTSM series; classification of gauges as TC-dominated. |
| **IBTrACS** | v04r01, all tracks | cyclone mask on the ETC fit; cyclone basin of each TC station. |
| **COAST-HG** (Dullaart et al., 2023) | RP100 | only in the fallback source `coast_rp` (§3.8), not in production. |

## 3. Conceptual method

All levels are relative to local (model) mean sea level. MDT, the 2020 sea-level offset and SLR
are added downstream (§3.7).

### 3.1 Tide

1. **Atlas choice.** FES2022 everywhere, except where FES2022's MHWS exceeds FES2014's by both a
   factor 1.5 and 1.25 m. The flag then grows to neighbours within 25 km that pass the factor and
   a 0.25 m difference (so that stations a few km apart on the same FES artefact do not take
   different atlases), but never within 50 km of a gauge where FES2022 matches the gauge MHWS
   within 0.25 m. Where gauges exist, the rule picks the atlas that matches the gauge (Juneau,
   Skagway, Port Alberni; one wrong flag at the Saguenay mouth).
2. **Seasonal constituents.** Sa and Ssa are zeroed: in FES2022 they come from altimetry and
   represent seasonal sea level, which the GTSM surge residual already carries.
3. **Gauge correction of the tide amplitude.** Each coastal GESLA record gets a short harmonic
   fit (first two years; M2, S2, N2, K1, O1, M4). The tide amplitude is the sum of the M2, S2, N2,
   K1 and O1 amplitudes. A station or node takes factor = exp(median log(gauge / FES amplitude))
   over the usable gauges within 25 km (leave-one-out best of 10-200 km), else 1. The factor
   scales all FES constants (amplitude only, phases kept), and MHWS with them. Records that dry
   at low water, have < 70% coverage, or a range > 20 m (feet) are not used.
4. **Prediction.** Tide is predicted as a matrix product of a cached (time × constituent) basis
   with pyTMD nodal corrections and inferred minors (checked against pyTMD to 1e-8 m): hourly over
   the 1950-2024 record (aligned with GTSM) and at 10 min over one nodal cycle (18.61 y from 1996)
   for the convolution and the hydrograph tide windows.

### 3.2 Surge

- The hourly GTSM-ERA5 non-tidal residual (tide+surge run minus tide-only run) at the station's
  nearest output point, 1950-2024 (75 years).
- **Despiking**: hours more than 2 m from their 5-h running median are dropped. A station with
  ≥ 5 such jumps (15 stations, Arafura coast, jumps up to 13 m) keeps COAST-RP's levels and gets a
  Gaussian surge shape (`level_source = coast_rp`).

### 3.3 Return levels

The hydrograph peak *is* the return level, so the level method is part of the hydrograph.

1. **Skew surges.** Per tidal cycle: highest total level (GTSM surge + FES tide) minus the
   predicted high water (Williams et al., 2016), cycles bounded halfway between predicted high
   waters. Low-water bumps of shallow-water tides (a maximum in the lowest quarter of the 25-h
   range with prominence < 10% of the range) are not high waters; without this rule the next real
   high water reads as a skew surge of metres (Ungava Bay 4.5 → 0.8 m). Years with < 80% of cycles
   observed are dropped.
2. **Cyclone mask** (stations with a COAST-RP TC part only): cycles within ±24 h of an IBTrACS
   storm (≥ 34 kt, or any track without wind on record) passing within 300 km leave the ETC fit,
   so cyclones are not counted twice (ERA5 record and COAST-RP_TC).
3. **Tail model.** Empirical distribution below the 99.5% quantile; GPD above it, on cluster
   maxima (clusters separated by > 3 cycles; extremal index θ = clusters / exceedances). The GPD
   **shape is pooled**: each station takes the shape that maximises the summed profile
   likelihood of all stations within 300 km (scale fitted per station). 300 km halves the change
   of a station's shape between 1979-2024 and 1950-2024 relative to a per-station fit; gauge
   scores are insensitive to the choice.
4. **Gauge surge ratio r.** At GESLA gauges with ≥ 20 complete years since 1979, a GTSM point
   ≤ 10 km away, ≥ 15 common years, and not TC-dominated: r = gauge RP10 skew surge / GTSM-on-FES
   RP10 skew surge, both on the same 1979-2024 years. The gauge skew surges are measured against
   the gauge's own harmonic fit (FES2022's constituent set, fitted to the gauge record of
   1996-2014, ≥ 3 years, Sa/Ssa zeroed), after each calendar year's mean is removed from the
   record, so sea-level rise and land motion at the gauge do not enter them. The GTSM skew
   surges are measured against the FES tide at the gauge. A station
   takes the inverse-distance mean of r over gauges within 500 km (≥ 3 gauges); with fewer, a
   power law fitted over all calibration gauges, r = exp(b) · S10^a with S10 the station's own GTSM
   RP10 skew surge (fitted: r ≈ 1.03 · S10^-0.32). r is capped at 2.0. Only positive skew surges
   are scaled (S → r·S for S > 0). Median station r is 1.18; ~13,600 stations use the power law.
5. **Convolution** (skew surge assumed independent of the tide):
   λ_ETC(z) = N_cycles/yr · θ · Σ_k w_k · P(S_corrected > z − t_k),
   with t_k the FES high-water heights over a nodal cycle binned to 1 cm and w_k their frequencies.
6. **TC component.** COAST-RP_TC levels, raised by a shift fitted per IBTrACS basin, become an
   annual exceedance rate λ_TC(z) (log-linear in the rate between COAST-RP's RPs, extrapolated
   beyond). The shift is fitted in the build at TC-dominated gauges: each gauge's station is
   rebuilt (mask, pooled shape, r without the gauge) at trial shifts 0-0.4 m, and the basin takes
   the shift that zeroes the median RP100 error of the *other* basins' gauges (leave-one-basin-out).
   Expected values 0.18-0.25 m.
7. **Return level**: solve λ_ETC(z) + λ_TC(z) = −ln(1 − 1/RP) by bisection. This is the same
   combination COAST-RP uses for its own ETC and TC parts.
8. **Flags** in the levels file: `flag_tail_uncertain` (pooled shape > 0.2), `flag_tc_coast` (a
   COAST-RP TC part), `flag_fallback_r` (r from the power law). RP250-1000 extrapolate a 75-year
   record and the r correction is validated to RP100 only.

### 3.4 Hydrograph generator

Per station and RP (`station_product`, `rp_hydrograph`). Validated as the "generator, smoothed
surge" variant of `boundary_test/INPUT_ANALYSIS.md` §8.3.

1. **Events.** Annual maxima of the hourly total level (despiked GTSM surge + station FES tide),
   1950-2024; a year counts with ≥ 80% data, its maximum with a ≥ 90% complete window. Empirical
   RP = (n + 1) / rank. At least 10 events are needed, else the station has no hydrograph.
2. **Event selection per RP.** Events with empirical RP in (RP/2, 2·RP]; for an RP above the
   record's largest empirical RP (n + 1), the 2 largest events (their shape is used and only the
   amplitude follows the RP). Fewer than 2 selected: all events. On a 75-year record RP50 still
   has its own window; RP100-1000 take the 2 largest events.
3. **Window length.** ±72 h, or ±120 h when ≥ 25% of the annual maxima stay above MSL over a whole
   72-h half (censored events: Baltic, Mediterranean and other surge-dominated microtidal coasts;
   2.7% of stations). The output grid is always ±120 h (1,441 steps), NaN outside the window.
4. **Tide window.** The tide level during extremes is the median FES tide at the peaks of the
   selected events (≈ 1.2 × MHWS in the median). From the 10-min nodal-cycle tide, the high water
   whose height is closest to that level is the window centre. If that window already exceeds the
   RP level anywhere, only high waters whose whole window stays below the RP level qualify.
   Using the tide level of observed extremes rather than an MHWS spring tide is the main
   difference to HGRAPHER / COAST-HG, whose neighbouring high waters are 0.1-0.15 of the peak too
   low (§5).
5. **Surge shape (extratropical).** The GTSM surge of each selected event over the window, smoothed
   by a 13-h running mean over the available hours (one semidiurnal period, removing the
   tide-surge interaction ripple of the residual; missing hours and the window ends leave the
   mean rather than counting as zero), each event shifted so its surge maximum within ±12 h of the water-level peak sits
   at t = 0, normalised to 1 there, median over events, interpolated to 10 min.
6. **Surge shape (tropical cyclone).** Where the TC rate exceeds the ETC rate at the RP level (and,
   below RP10, also at RP10, because COAST-RP_TC is flat between RP2 and RP10): the median surge
   shape of the nearest GESLA "donor" gauge within 50 km (gauges with ≥ 2 annual maxima whose
   surge is ≥ 50% of the level), else a Gaussian with half width at half maximum 8 h (the median
   surge duration at TC-dominated gauges).
7. **Other fallbacks.** GTSM-QA stations (§3.2): Gaussian. No selected event with a positive surge
   at its peak: Gaussian.
8. **Scaling.** Hydrograph = tide window + a · shape, with a found by bisection so that the maximum
   of the sum equals the RP level exactly (all 209,034 peaks within 5e-7 m). The surge peak is
   placed on the high water (no tide-surge phase). The maximum can sit on a neighbouring high
   water (4.1% of hydrographs, median 1.6 cm above the centre), which leaves the peak value intact.

The shape is independent of SLR: the same hydrograph serves every SLR scenario of an RP.

### 3.5 Coastline nodes

To carry tide gradients between stations (estuaries, macrotidal bays), the levels file also holds
one node per FES2022 grid cell on the DeltaDTM coastline. A node blends the corrected skew-surge
models of its 5 nearest product stations (inverse distance, power 3; quantiles and GPD parameters
averaged) and convolves them with its **own** gauge-corrected FES tide, using the atlas of its
nearest station. TC levels are blended the same way if the nearest station has a TC part. Nodes
have **no hydrograph**; correction B uses the nearest station's.

### 3.6 Station-to-tile interpolation

Coastline seeds and the static level of each tile are the inverse-distance mean (k = 5, power 3)
over stations and nodes in the same ocean component. Leave-one-out at 22,670 stations: RP100 MAE
5.7 cm, p90 14.3 cm; the error grows with station spacing (2 cm within 5 km, ~20 cm beyond 50 km).

### 3.7 Downstream use

- **Scenario levels** (`prepare_boundary_conditions.py`):
  total(RP, SLR) = storm tide(RP) + MDT + 2020 offset + SLR fingerprint, where MDT (AVISO
  CNES-CLS22) re-references local MSL to the GOCO06s geoid of the DEM, the 2020 offset is the AR6
  global-mean rise from the MDT epoch to 2020 (0.049 m, regionalised) and the fingerprint is the
  AR6 regional pattern scaled to the target global-mean SLR.
- **Correction B** (eikonal): after a first solve, each wet cell's head above its spill level
  (ground, or the sill behind a ridge) gives T = time the nearest station's hydrograph at that RP
  stays within that head of its peak. Friction is multiplied by (12 h / T)^1, capped at 10, and the
  tile is solved again. Only time near the peak matters, so datum and SLR do not.
- **SFINCS**: forcing points every ≤ 2 km along the boundary line, each with the IDW blend (k = 5,
  power 3, ≤ 55 km) of its nearest stations' RP100 hydrograph shapes, cut to COAST-HG's 894-step
  axis (peak at 74.5 h) and shifted so its maximum equals the eikonal seed level at the nearest
  coastline cell (stations + coastline nodes, RP100 SLR_0 incl. MDT and 2020 offset). SFINCS
  interpolates between the points.

### 3.8 Fallback source

`boundary_conditions.storm_tide_source: coast_rp` uses COAST-RP levels and the COAST-HG RP100
hydrograph for every RP. A marker file next to the scenario NetCDFs records the source they hold;
a config asking for the other source stops instead of mixing them.

## 4. Code implementation

### 4.1 Entry point and data flow

```
python preparation/run_preparation.py storm_tide_product   # then: boundary_conditions
```

[prepare_storm_tide_product.run](../preparation/prepare_storm_tide_product.py#L291) runs, in order:

| Step | Function | Output |
|---|---|---|
| FES2022 decompressed, tide bases (hourly 1979-2024 for gauges, hourly 1950-2024 for stations, 10-min nodal cycle) | `tide_prediction.ensure_local_fes`, `_bases` | cache |
| GESLA records, gauge-GTSM pairs, gauge harmonic fits, gauge/FES MHWS, ratios r, TC donors | [gauge_table](../preparation/prepare_storm_tide_product.py#L96) | `gauges/gauges.parquet`, `tc_donors.parquet`, `tc_donor_shapes.npy` |
| COAST-RP stations matched to GTSM points | `stp.match_stations` | `stations.csv` |
| GTSM series at stations (1979-2024 and 1950-1978 stores) | `gtsm_series.extract_series` | cache (int16 mm, station-major) |
| Tide-correction gauges | [tide_gauge_table](../preparation/prepare_storm_tide_product.py#L139), [tide_correction.gauge_table](../src/tide_correction.py#L103) | `gauges/tide_gauges.parquet` |
| Atlas choice and tide factor per station | [station_atlas](../preparation/prepare_storm_tide_product.py#L151) | cache (`fes_constants.npz`, `fes_mhws.npz`) |
| Station r, power-law fallback, TC levels, TC donor shapes, IBTrACS mask and basin | `stp.station_ratios`, `ratio_covariate`, `tc_hours`, `station_basins` | in memory |
| GPD profile likelihoods, pooled shape | `stp.build_profiles`, [pooled_shapes](../src/storm_tide_product.py#L461) | `shape_profiles.parquet` |
| TC basin shift fit | [tc_shift_fit](../preparation/prepare_storm_tide_product.py#L249), [loo_basin_fit](../src/storm_tide_product.py#L407) | `tc_shifts.csv`, `tc_shift_gauges.parquet` |
| Levels and hydrographs per station | `stp.build_stations` → [station_product](../src/storm_tide_product.py#L644) | `levels.parquet`, `hydrographs.npy`, `hydrograph_info.parquet`, `surge_models.npy` |
| Coastline nodes | [node_levels](../preparation/prepare_storm_tide_product.py#L182), `stp.build_nodes` | `node_levels.parquet` |
| Product files | [export_levels](../src/storm_tide_product.py#L805), [export_hydrographs](../src/storm_tide_product.py#L856) | `GFM_storm_tide_levels.nc`, `GFM_storm_tide_hydrographs.nc` |

Stations are processed in chunks of 25 in a process pool (`storm_tide_product.workers`). The
cache (~30-45 GB) holds derived copies of raw inputs only and is reused across runs; `force=True`
recomputes everything else. Run time is dominated by the GTSM extraction (900 monthly files read
in full) and two passes over the stations.

### 4.2 Module map

| Module | Content |
|---|---|
| [src/storm_tide_product.py](../src/storm_tide_product.py) | constants (L55-101), gauge ratios, TC mask/basins/shift, pooled shape, [rp_levels](../src/storm_tide_product.py#L622), [station_surge_model](../src/storm_tide_product.py#L596), [station_product](../src/storm_tide_product.py#L644), nodes, export |
| [src/skew_surge.py](../src/skew_surge.py) | high waters, skew surges, declustering, GPD fit and profile likelihood, r scaling, exceedance, convolution |
| [src/storm_tide_hydrograph.py](../src/storm_tide_hydrograph.py) | [annual_maximum_events](../src/storm_tide_hydrograph.py#L175), [rp_event_selection](../src/storm_tide_hydrograph.py#L199), [median_surge_shape](../src/storm_tide_hydrograph.py#L210), [high_water_window](../src/storm_tide_hydrograph.py#L152), [rp_hydrograph](../src/storm_tide_hydrograph.py#L276) / [scale_to_peak](../src/storm_tide_hydrograph.py#L287). Also holds the HGRAPHER port (`cell_hydrograph`), used only in the validation scripts. |
| [src/tide_prediction.py](../src/tide_prediction.py) | FES models via pyTMD, constant extraction (20 km extrapolation), basis prediction, harmonic fit, MHWS, [version rule](../src/tide_prediction.py#L259) |
| [src/tide_correction.py](../src/tide_correction.py) | GESLA short harmonic fits, [correction_factors](../src/tide_correction.py#L116) |
| [src/gesla.py](../src/gesla.py), [src/gtsm_series.py](../src/gtsm_series.py) | GESLA metadata, deduplication, record extraction (UTC, QC 1-2); GTSM station-major store |
| [src/boundaries.py](../src/boundaries.py#L159) | product file names, `storm_tide_levels_path`, [storm_tide_hydrograph_path](../src/boundaries.py#L193), static level |
| [preparation/prepare_boundary_conditions.py](../preparation/prepare_boundary_conditions.py) | MDT, 2020 offset, SLR fingerprints → scenario NetCDFs |
| [src/event_volume.py](../src/event_volume.py) | [duration_friction_multiplier](../src/event_volume.py#L161), `DurationCorrection`, [load_coast_hg](../src/event_volume.py#L264) (reads both COAST-HG and the product) |
| [src/eikonal_runner.py](../src/eikonal_runner.py#L200) | `build_duration_correction`: hydrograph at the scenario RP |
| [sfincs_tiles/build_line_forcing.py](../sfincs_tiles/build_line_forcing.py#L40) | RP100 hydrograph on COAST-HG's axis, shifted to the boundary level |

### 4.3 Configuration

`snakemake_workflow/config/config.yml`:
- `boundary_conditions.storm_tide_source` (`gfm_product` | `coast_rp`), `storm_tide_product_dir`;
- `storm_tide_product`: `gesla_dir`, `gtsm_dir`, `fes2022_dir`, `ibtracs_path`, `cache_dir`,
  `workers`; COAST-RP and FES2014 come from the data catalog (`coast_rp`, `fes2014_ocean_tide`);
- `simulation.flooding.knn` / `idw_power` (station interpolation) and
  `simulation.flooding.duration_correction` (`alpha`, `t_ref_h`, `max_multiplier`).

Method constants are module-level in the source, not config, so that the product is defined by
the code version:

| Constant | Value | Meaning |
|---|---|---|
| `skew_surge.THRESHOLD_Q` / `RUN_CYCLES` | 0.995 / 3 | GPD threshold quantile; declustering run length (cycles) |
| `skew_surge.HW_PROMINENCE` | 0.1 | low-water-bump rule (fraction of the 25-h range) |
| `skew_surge.SPIKE_M` / `storm_tide_product.GTSM_QA_JUMPS` | 2.0 m / 5 | despike threshold; jumps that send a station to COAST-RP |
| `POOL_KM` / `TAIL_UNCERTAIN_SHAPE` | 300 km / 0.2 | GPD shape pooling radius; tail flag |
| `GAUGE_GOOD_YEARS`, `GAUGE_GTSM_KM`, `MIN_YEARS`, `RATIO_RP` | 20, 10 km, 15, 10 | ratio-gauge criteria |
| `IDW_RADIUS_KM`, `MIN_NEIGHBOURS`, `R_CAP` | 500 km, 3, 2.0 | station r |
| `TC_MASK_KM`, `TC_MASK_HR`, `TC_MIN_WIND_KT` | 300 km, 24 h, 34 kt | cyclone mask |
| `TC_TRIAL_SHIFTS_M`, `REFERENCE_RP`, `GAUGE_STATION_KM` | 0-0.4 m, 100, 25 km | TC basin shift fit |
| `BASE_HR` / `EXTENDED_HR` / `CENSORED_SHARE` | 72 / 120 h / 0.25 | hydrograph window |
| `SMOOTH_HR`, `MIN_EVENTS` | 13 h, 10 | surge smoothing; minimum annual maxima |
| `TC_GAUGE_KM`, `TC_SURGE_SHARE`, `TC_HALF_WIDTH_HR`, `TC_SHAPE_MIN_RP` | 50 km, 0.5, 8 h, 10 | TC shape |
| `storm_tide_hydrograph.RP_WINDOW_FACTOR`, `LARGEST_EVENTS` | 2, 2 | event selection per RP |
| `tide_correction.CORRECTION_KM` | 25 km | tide correction radius |
| `tide_prediction.VERSION_*`, `FES2022_OK_M` | 1.5, 1.25 m, 25 km, 0.25 m, 50 km; 0.25 m | FES version rule |

### 4.4 Output format

- `GFM_storm_tide_levels.nc`: COAST-RP layout (`stations`, `station_id`, `station_x/y_coordinate`,
  `storm_tide_rp_XXXX`, m above local MSL), plus `level_source` (`product` | `coast_rp` |
  `failed` | `coastline_node`; `failed` stations raised during the build, carry COAST-RP's levels
  and no hydrograph, and are counted in the run log), `r`, `atlas`, `gpd_shape` and the three flags; coastline nodes appended as
  `node_<n>`.
- `GFM_storm_tide_hydrographs.nc`: `storm_tide_hydrograph(stations, return_period, time)`, 10-min
  steps over ±120 h around the peak at time 0, NaN outside each station's window, m above local MSL.

## 5. Validation evidence

Main sources: `boundary_test/INPUT_ANALYSIS.md` §3-8 and
`input_analysis/boundary_forcing/boundary_forcing_input_analysis.md` (figures fig01-fig31).

**Tide.** GTSM's tide is 7-8% too high against gauges; FES matches better. Gauge correction of the
FES amplitude (2,373 usable gauges, leave-one-out): amplitude MAE 0.185 m (FES at the gauge) →
0.133 m (locally corrected).

**Return levels** at 618 gauges with a station within 25 km; reference = the gauge's own skew-surge
convolution; r rebuilt without the gauge. Station − gauge, median / MAE (m):

| | RP10 COAST-RP | RP10 product | RP100 COAST-RP | RP100 product |
|---|---|---|---|---|
| ETC-dominated (451) | −0.017 / 0.179 | −0.005 / 0.106 | −0.068 / 0.244 | −0.013 / 0.163 |
| TC-dominated (167) | −0.192 / 0.239 | −0.011 / 0.113 | −0.247 / 0.443 | −0.027 / 0.361 |

Large regional departures from COAST-RP are on the gauges' side: southern North Sea 0.6-1.0 m
lower (COAST-RP exceeds the gauge GEV by 0.9-1.1 m), western Baltic 0.3-0.9 m higher, macrotidal
basins 2-4 m higher (GTSM's tide too small).

**Hydrographs** at 698 quality GESLA sites, RP 2-100, against the gauge's own annual maxima of the
same RP window; cells = 5 tidal-range classes × 4 RPs:

| Variant | Cells within ±15% on T(0.25), T(0.5), nb24 | Median deviation T(0.25) / T(0.5) / nb24 |
|---|---|---|
| generator, smoothed surge (production) | 14 of 20 | 8% / 5% / 3% |
| HGRAPHER port | 0 of 20 | 46% / 17% / 15% |
| COAST-HG (RP100 only) | 0 of 4 | 53% / 26% / 8% |

T(x) = hours above x × peak; nb24 = level of the next-day high water relative to the peak. Flood
impact (SFINCS 1-D sill experiments, 2,910 runs): the generator's flood volume is within 10% of the
median of real gauge events at RP2 and RP10, inside their own 16-22% event-to-event spread. Median
T(0.25) at RP100 is 12 h against COAST-HG's 6 h.

**Caveat.** The level scores above come from a build without the 1950-1978 record, the cyclone
mask, the pooled GPD shape, the TC basin shift and the tide-amplitude correction. Each of those was
scored separately in `boundary_test/open_guesses.py` (results in TODO.md, "external review of
decisions 1-3"); the combined build still has to be rescored.

## 6. Assumptions

- Skew surge is independent of the tide (convolution); the hydrograph places the surge peak on a
  high water.
- r is multiplicative and applies to positive skew surges only; it changes the peak level, not
  the normalised shape, which comes from uncorrected GTSM.
- The ETC hydrograph shape at RP ≥ 50 is that of the largest observed events; its width falls with
  RP only through the growing surge share of the peak.
- Tide and hydrograph shape are unchanged under SLR.
- The TC component is a gap-filler: COAST-RP_TC (synthetic STORM tracks) plus a basin shift, with
  a donor-gauge or Gaussian shape.
- Wave setup and runup are not included; the levels are still-water storm tide.
- The GTSM residual is barotropic: it carries the wind- and pressure-driven seasonal cycle but not
  the steric one, and Sa/Ssa are removed from the tide.

## 7. Points for review

Conceptual:
1. **Joint probability and TC handling**: masked ETC fit + COAST-RP_TC shifted per basin, rates
   added. Without the mask, the gauges score better (TC-dominated RP100 median −0.03 vs −0.17 m
   with mask and no shift) because two low sources double count. Is the shift a defensible
   correction of COAST-RP_TC, or should the TC part wait for STORM-forced GTSM?
2. **GPD shape pooling at 300 km** (TC and non-TC stations pooled together) and RP250-1000 from a
   75-year record.
3. **Surge ratio r**: scaling the positive skew surges by a gauge ratio; power-law fallback for
   ~60% of stations (sparse gauges), which overshoots on calm coasts where GTSM is right (East
   Africa r ≈ 1.6 vs gauges 0.8-1.1).
4. **Hydrograph construction**: tide level of observed extremes, 13-h smoothing, median shape,
   surge peak on high water. Known misses: RP2 T(0.25) ~30% short on micro- and macrotidal coasts
   (1-9% flood volume); overflow above 0.9 × peak 20-40% short on > 2 m tidal ranges (the mild
   direction of error for flood volume).
5. **TC hydrographs**: 8-h HWHM Gaussian at ~2,500 stations at RP100; donor gauge shapes at ~270.
6. **Coastline nodes without hydrograph**: at estuary heads the nearest station's tide range can be
   smaller than the node's, so T near the peak is too long there.
7. **SLR invariance** of the hydrograph and tide.
8. **SFINCS forced at RP100 only** for the calibration of the eikonal friction; correction B uses
   the per-RP hydrographs at all RPs.

Implementation:
9. **Generator validation period.** The hydrograph generator was validated on 1979-2024;
   production builds its annual maxima from 1950-2024.
10. **Ratio years vs fit years.** r is fitted on 1979-2024 (gauge store and hourly basis), the
    station GPD on 1950-2024.

## 8. References

- Dullaart, J. C. M., et al. (2021). Accounting for tropical cyclones more than doubles the global
  population exposed to low-probability coastal flooding. *Commun. Earth Environ.* 2, 135 (COAST-RP).
- Dullaart, J. C. M., et al. (2023). Enabling dynamic modelling of global coastal flooding by
  defining storm tide hydrographs. *NHESS* 23, 1847-1862 (HGRAPHER, COAST-HG).
- Haigh, I. D., et al. (2020). The tides they are a-changin'. *Rev. Geophys.* 58, e2018RG000636.
- Haigh, I. D., et al. (2023). GESLA version 3. *Geosci. Data J.* 10, 293-314 (GESLA-4.1 follows the
  same format).
- Horsburgh, K. J., and Wilson, C. (2007). Tide-surge interaction and its role in the distribution
  of surge residuals in the North Sea. *J. Geophys. Res.* 112, C08003.
- Lyard, F. H., et al. (2021). FES2014 global ocean tide atlas. *Ocean Sci.* 17, 615-649; FES2022b
  (AVISO+).
- Muis, S., et al. (2020). A high-resolution global dataset of extreme sea levels, tides, and storm
  surges, including future projections. *Front. Mar. Sci.* 7, 263 (GTSM).
- Knapp, K. R., et al. (2010). The International Best Track Archive for Climate Stewardship
  (IBTrACS). *Bull. Amer. Meteor. Soc.* 91, 363-376.
- Williams, J., et al. (2016). Tidal modulation of storm surges. *Geophys. Res. Lett.* 43, 9277-9283
  (skew-surge independence).
