# Boundary forcing inputs: datasets and how they compare

This note assesses the coastal water-level sources that can force the eikonal and SFINCS models.
It covers what each dataset is, the common reference frame they are compared in, and where and
why they differ. It answers three questions:

1. Do the sources share a vertical reference and a mean sea level?
2. How well do the tidal levels and extremes of the global models match tide gauges?
3. What shape do storm-tide hydrographs have, and what does that mean for the duration
   correction of the eikonal friction?

Scope: global, at the GESLA tide gauges. Model comparisons (eikonal vs SFINCS on real tiles) are
outside this note. Scripts live in `boundary_test/`; outputs, including all figures named below,
in `.cleanup_baseline/level_sources/` (local, not in git).

## 1. Datasets

| Dataset | What it is | Coverage | Vertical reference | Role in the pipeline |
|---|---|---|---|---|
| **GESLA 4.1** | observed total water level at tide gauges (hourly or finer) | 1,154 sites with >= 20 y record | each gauge's own datum (chart datum, national datum, often unspecified) | reference ("truth") for this note |
| **GTSM v3 tide** (Copernicus CDS) | modelled tide: astronomic tide, self-attraction and loading, radiational tides and a rising mean sea level | global model stations, 10 min, 1996-2014 | model MSL (apparently 1986-2005), with a built-in trend (median 2.6 mm/yr) | tide component of COAST-RP |
| **GTSM v3 surge** (CDS) | modelled water level minus tide, forced by ERA5 | hourly, 1979-2024 | relative to the tide run, no trend | surge component of COAST-RP (extratropical) |
| **FES2022** | ocean-tide atlas, 34 harmonic constituents, assimilates altimetry and gauges | global grid | zero-mean by construction | candidate replacement for the GTSM tide ("H1") |
| **COAST-RP** | storm-tide return levels (RP 1-1000 y) per GTSM station: GTSM surge statistics (ETC) and STORM synthetic cyclones (TC), combined with the tide | 23,226 stations; separate ETC-only and TC-only files | GTSM model MSL, no MDT | boundary level in production (+ MDT + SLR) |
| **COAST-HG** | one RP100 storm-tide hydrograph per COAST-RP station, 149 h at 10 min, average- and spring-tide variants | as COAST-RP | as COAST-RP; its peak equals COAST-RP RP100 exactly | hydrograph shape for the duration correction and for SFINCS calibration forcing |
| **MDT CNES-CLS22** | mean dynamic topography: mean sea surface above the geoid | global ocean, 1/8 deg | GOCO06s geoid, mean of 1993-2012 | converts COAST-RP from local MSL to the DEM's geoid datum |

Definitions used throughout:
- **MSL:** mean sea level. **MHWS / MLWS:** mean high / low water springs, the average spring-tide
  high / low water. **Spring range:** MHWS - MLWS. **HAT / LAT:** highest / lowest astronomical tide.
- **RP10 / RP100:** the level reached on average once in 10 / 100 years.
- **GEV fit:** generalised extreme value distribution fitted to annual maxima. 90% intervals come
  from 100 bootstrap refits.
- **ETC / TC:** extratropical / tropical-cyclone storm surges.

## 2. Common reference frame

The sources do not share an MSL or a datum, so every comparison removes them first.

- **Local MSL per year.** Each calendar year's mean is removed from GESLA and from the GTSM tide.
  Gauge datum, sea-level rise, land motion and interannual MSL drop out; FES is already zero-mean.
  All levels below are therefore heights above local MSL, never absolute heights. Absolute MSL +
  MDT against gauges with a geodetic datum is not tested here.
- **One harmonic method.** GESLA and the GTSM tide are fitted (least squares) with the 34 FES
  constituents, including nodal corrections and inferred minor constituents. FES uses its atlas
  constants. All three are then predicted over the same 18.61-year epoch at 10 min, and the same
  statistics code (`src/tide_statistics.py`) derives MHWS, MHW, MLW, MLWS, HAT and LAT.
  Differences in method, epoch or sampling therefore cannot explain what remains.
- **Pairing.** Each gauge is paired with its nearest GTSM / COAST-RP / COAST-HG station (median
  distance 1.9 km). FES is evaluated both at the gauge and at the GTSM station.
- **Quality sites** (698 of 1,154): fit over >= 3 years, explaining >= 50% of the variance, GTSM
  station within 10 km.

## 3. Tidal levels

Figures: `analysis/levels_scatter.png` (GTSM vs gauge, FES vs gauge, GTSM vs FES),
`analysis/levels_ratio_maps.png` (the same as maps), `analysis/levels_constituents.png`.

| Comparison (MHWS) | Median ratio | MAE | Within 0.1 m |
|---|---|---|---|
| GTSM vs gauge | 1.08 | 0.16 m | 53% |
| FES2022 at the gauge vs gauge | 0.99 | 0.05 m | 88% |
| GTSM vs FES2022 at the GTSM station | 1.08 | 0.16 m | 52% |
| FES2022 at the gauge vs at the GTSM station | 1.00 | 0.01 m | 98% |

- **GTSM's tide is 7-8% too large; FES2022 matches the gauges.** GTSM is too large in the
  Mediterranean (1.18), Africa / Indian Ocean (1.19-1.20), NW Europe (1.08-1.09) and S America
  (1.10). It is too small on the N American Atlantic and Gulf coast (0.85) and right on the
  N American Pacific coast (1.01).
- **Not a location or method effect:** FES is the same at the gauge and at the GTSM station, the
  GTSM bias does not grow with gauge-station distance, and the raw GTSM series and its harmonic
  fit agree.
- **Which constituents:** amplitude ratios GTSM / gauge are S2 1.20, M2 1.06, O1 1.10 and
  MS4 1.28, against 0.99-1.00 for FES. The spring range is roughly M2 + S2, so the S2 and M2
  excess produces the MHWS excess. A plausible but unverified cause is GTSM's treatment of the
  radiational part of S2; FES assimilates data that constrain S2 directly.
- **FES fails locally, at fjord and estuary heads.** 20 of 698 sites miss MHWS by > 0.25 m
  (GTSM: 124). Examples:
  - too high at Juneau (5.95 vs 2.72 m) and Fulford Harbour (1.99 vs 0.89 m), and at the Port
    Phillip entrance;
  - too low in the upper St Lawrence (Saint-François 1.27 vs 2.87 m).

  Inside Minas Basin, the upper Gulf of California and the Strait of Magellan, FES gives MHWS of
  14.5-16.3 m, about twice published values. GESLA has no gauge there to check against. FES's
  own native / extrapolated mask does not flag these points.
- **Bay of Fundy and SF Bay** (at the gauges): FES matches; GTSM is 25% too low in Fundy
  (Saint John 2.64 vs 3.54 m) and 10-35% too low in SF Bay.

## 4. Extremes

Compared: COAST-RP return levels against GEV fits to GESLA annual maxima (record median 30 y)
and to 46 years of GTSM (surge + harmonic tide).

| Comparison | RP10 median diff | RP10 MAE | RP100 median diff | RP100 MAE |
|---|---|---|---|---|
| COAST-RP - gauge GEV | -0.06 m | 0.22 m | -0.06 m | 0.37 m |
| GTSM GEV - gauge GEV | -0.06 m | 0.20 m | -0.11 m | 0.37 m |
| COAST-RP - GTSM GEV (same model, different statistics) | -0.01 m | 0.07 m | +0.02 m | 0.15 m |

- **Tide and surge errors offset each other.** COAST-RP's RP10 error tracks GTSM's tide excess
  (correlation 0.88). With the tide excess removed, COAST-RP is 0.11-0.12 m too low: its surge
  part is too low, and the too-large tide hides that. Only 18% (RP10) / 44% (RP100) of COAST-RP
  levels fall inside the gauge's 90% interval.
- **Tropical cyclones.** A station is called TC-dominated when its TC-only RP100 exceeds its
  ETC-only RP100 (from the COAST-RP_TC / _ETC files).

  | Station type | Sites | COAST-RP - gauge GEV, RP10 / RP100 |
  |---|---|---|
  | TC-dominated | 194 | -0.19 / -0.24 m |
  | ETC-dominated, TC value present | 200 | -0.04 / -0.08 m |
  | no TC station | 254 | +0.04 / +0.02 m |

  The TC part adds 0.17 m (median) to RP100 at TC-dominated stations and is still too low there:
  Japan / E Asia -0.25 m, US Atlantic and Gulf -0.24 m. GTSM alone is lower still (-0.29 m), as
  expected, since ERA5 does not resolve tropical cyclones. A 30-40-year gauge record can also
  miss rare cyclones, so the gauge GEV is itself uncertain at these sites.
- **Fundy and SF Bay:** COAST-RP RP100 is 1.2-1.4 m below the gauge GEV in the outer Bay of Fundy
  (Saint John 3.45 vs 4.66 m) and 0.3-0.5 m below in SF Bay. No source overestimates there; the
  overestimates are FES's, at basin heads without gauges.

## 5. Hydrograph shapes

Events are the annual maxima at each gauge, with ±72 h of record around each peak: 21,100 GESLA
and 32,100 GTSM events. COAST-HG gives one RP100 event per station. Each event's empirical return
period is its rank among the site's annual maxima.

Duration measures, in hours, for levels relative to local MSL:
- **Contiguous time above MSL:** the unbroken stretch above MSL that contains the peak.
- **Cumulative time above MSL:** all hours above MSL in the window.
- **T(f):** the total time, across all tidal cycles in the window, that the level is within
  f × (peak - MSL) of the peak. This is the T of the eikonal duration correction for a cell whose
  water depth is that fraction of the storm tide (`event_volume.time_within_peak`).

Figures: `analysis/shapes_profiles.png`, `analysis/shapes_profiles_by_rp.png`,
`analysis/shapes_durations.png`.

| Source | Contiguous above MSL | T(0.25) | T(0.5) |
|---|---|---|---|
| GESLA annual maxima | 9 h (34% > 12 h) | 15 h | 39 h |
| GTSM annual maxima | 8 h (27% > 12 h) | 15 h | 38 h |
| COAST-HG, spring tide | 9 h | 5 h | 24 h |

- **Longer than 12 h above MSL** depends on the tidal regime: the median is 20 h where the spring
  range is below 1 m (41 h for events with RP >= 10 y), and 7-8 h where it exceeds 1 m. Where
  the tide is large, the level falls below MSL every cycle, whatever the surge does.
- **GTSM reproduces observed shapes:** per site, Spearman rho 0.87 and median ratio 1.00 for
  time above MSL. T(0.25) gives rho 0.77, ratio 1.00.
- **COAST-HG's central peak has the observed width, but its neighbouring high waters are too
  low:** about 0.72 of the peak at ±12/24 h, against 0.75-0.88 observed in meso- and macrotidal
  regimes. Annual maxima occur at spring tide, so several consecutive high waters are nearly as
  high as the peak. Because T counts every hour near the peak, those neighbours drop out of
  COAST-HG's T (5 h vs 15 h). In microtidal regimes COAST-HG also drops below MSL between high
  waters, which observed events do not.
- **Shape vs return period** (gauges, RP < 2 y → RP >= 10 y):
  - Rarer events are more surge-dominated: the surge share of the peak rises from 0.36 to 0.63
    (range < 1 m), 0.15 to 0.35 (1-2 m) and 0.05 to 0.13 (> 4 m).
  - Their tidal neighbours are therefore relatively lower. At ±24 h they fall from 0.81 to 0.58
    of the peak (< 1 m) and from 0.86 to 0.75 (1-2 m), and are unchanged above 2 m.
  - The surge also keeps the level above MSL longer.

## 6. Implications for the duration correction

The eikonal friction is multiplied by m = clip((t_ref / T)^α, 1, 10), with t_ref = 12 h. A longer
T gives a smaller m, less friction and a longer inland reach, which is the physically expected
direction. The forms of m were compared on 540 synthetic 1-D SFINCS floodplains
(`duration_experiment.py`, `duration_refit.py`), with fsf refitted per form:

- In SFINCS, absolute duration sets the inland reach where terrain allows a long reach. Under
  steady forcing there is no level loss.
- With fsf calibrated, a constant t_ref is absorbed into fsf; "12 h" only sets where the lower
  clip at 1 starts. Removing that clip fits slightly better (log-extent RMSE 0.140 vs 0.148).
- Taking t_ref as the hydrograph's own longest exposure, or as the calibration hydrograph's T at
  the same depth, fits worse than either. The first gives a 2 h and a 48 h event of the same
  shape the same friction; the second adds a depth-dependent friction unrelated to the event.
- T should come from hydrographs with realistic neighbouring high waters. GTSM-based shapes
  match the gauges; COAST-HG's give T about 3x too short.

## 7. Datum epoch of the zero-SLR scenario

`total_wl = storm_tide(RP) + MDT + SLR_fingerprint` (`preparation/prepare_boundary_conditions.py`).
COAST-RP is relative to model MSL, so the level of SLR_0 is the MSL of the MDT epoch, 1993-2012
(centre about 2003). The analysis instead reads the SLR axis against AR6 global-mean trajectories
rebased to 2020 (`analysis/extract_slr_trajectories.py`), and population is a 2020 baseline.

AR6 puts 2020 at +0.049 m (median, all SSPs) above its 1995-2014 baseline, so SLR_0 lies about
5 cm (global mean) below 2020 sea level. Consequences:
- Present-day hazard is evaluated about 5 cm too low.
- Every year in which an SLR level is reached comes out late by 5 cm / rate, roughly 7-10 years
  at mid-century rates.

The SLR fingerprints are unaffected: they are ratios of local to global change. A fix can be made
in the analysis alone, by referencing the trajectories to the MDT / AR6 baseline instead of 2020,
with no model reruns. Regional 2003-2020 rise and land motion can differ from the global 5 cm.

## 8. Open assessment questions

GTSM, COAST-RP and COAST-HG (same model, different statistics and shapes):
1. How do GTSM's peaks differ from COAST-RP's per RP and per station type (ETC / TC)? This note
   has the median and MAE only; the spread, the regional pattern and the sign at TC stations are
   still to do.
2. Do hydrographs built from GTSM's largest events at a station match COAST-HG's shape there,
   event by event, and how do both compare with the gauge's largest events?
3. Does the T computed from GTSM events vary with RP enough to matter, given that COAST-HG
   offers only RP100 and the pipeline uses its shape for every RP?
4. At TC-dominated stations, what hydrograph should set T, when neither ERA5-driven GTSM nor the
   gauges sample enough cyclones? A synthetic TC shape is one option.

Tide and surge:
5. If COAST-RP is rebuilt with the FES tide (re-combining the GTSM surge distribution with the
   FES tide, as COAST-RP combines surge and tide), how close does it come to the gauge GEV
   compared with COAST-RP? A level-only swap (subtracting GTSM MHWS, adding FES MHWS) would
   remove the tide excess but expose the surge deficit.
6. Is the surge deficit (about 0.1 m) regional and stable enough for a correction calibrated
   at gauges?
7. Which rule replaces FES at basin heads where it is implausible and no gauge exists? For
   example, fall back to GTSM where FES/GTSM MHWS > 1.5.
8. How large is the absolute MSL + MDT error at gauges with a geodetic datum tie?

## 9. Reproducing

Run in this order, each with `--out-dir .cleanup_baseline/level_sources` in the `gfm` env (see
each script's docstring for further arguments):
1. `levels_pairs.py`
2. `extract_gtsm_series.py --stations-csv`
3. `levels_gesla_extract.py`
4. `levels_tides.py`
5. `levels_analysis.py`
6. `levels_extremes.py`
7. `shapes_events.py`
8. `shapes_analysis.py`
9. `duration_experiment.py` (local SFINCS)
10. `duration_analysis.py`
11. `duration_refit.py`

Inputs on `<data_root>` are read only.
