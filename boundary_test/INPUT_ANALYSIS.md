# Boundary forcing inputs: datasets and how they compare

This note assesses the coastal water-level sources that can force the eikonal and SFINCS models:
what each dataset is, the common reference frame they are compared in, and where and why they
differ. It answers three questions:

1. Do the sources share a vertical reference and a mean sea level?
2. How well do the tidal levels and extremes of the global models match tide gauges?
3. What shape do storm-tide hydrographs have, and what does that mean for the duration
   correction of the eikonal friction?

Scope: global, at the GESLA tide gauges. Model comparisons (eikonal vs SFINCS on real tiles) are
outside this note.
- **Scripts:** `boundary_test/`.
- **Per-site data:** `boundary_test/input_analysis_data/` (§10), so the numbers here can be
  checked and re-cut.
- **Figures:** `.cleanup_baseline/level_sources/analysis/` (local, not in git).

## 1. Datasets

| Dataset | What it is | Coverage | Vertical reference | Role in the pipeline |
|---|---|---|---|---|
| **GESLA 4.1** | observed total water level at tide gauges (hourly or finer) | 1,154 sites with >= 20 y record | each gauge's own datum (chart datum, national datum, often unspecified) | reference ("truth") for this note |
| **GTSM v3 tide** (Copernicus CDS) | modelled tide: astronomic tide, self-attraction and loading, radiational tides and a rising mean sea level | global model stations, 10 min, 1996-2014 | model MSL (apparently 1986-2005), with a built-in trend (median 2.6 mm/yr) | tide component of COAST-RP |
| **GTSM v3 surge** (CDS) | modelled water level minus tide, forced by ERA5 | hourly, 1979-2024 | relative to the tide run, no trend | surge component of COAST-RP (extratropical) |
| **FES2022** | ocean-tide atlas, 34 harmonic constituents, assimilates altimetry and gauges | global grid, 1/30 deg | zero-mean by construction | candidate replacement for the GTSM tide |
| **COAST-RP** | storm-tide return levels (RP 1-1000 y) per GTSM station: GTSM surge statistics (extratropical, ETC) and STORM synthetic cyclones (tropical, TC), combined with the tide | 23,226 stations; separate ETC-only and TC-only (9,609 stations) files | GTSM model MSL, no MDT | boundary level in production (+ MDT + 2020 offset + SLR) |
| **COAST-HG** | one RP100 storm-tide hydrograph per COAST-RP station, 149 h at 10 min, average- and spring-tide variants | as COAST-RP | as COAST-RP; its peak equals COAST-RP RP100 exactly | hydrograph shape for the duration correction and for SFINCS calibration forcing |
| **MDT CNES-CLS22** | mean dynamic topography: mean sea surface above the geoid | global ocean, 1/8 deg | GOCO06s geoid, mean of 1993-2012 | converts COAST-RP from local MSL to the DEM's geoid datum |
| **IPCC AR6 sea-level projections** | regional and global-mean SLR relative to 1995-2014 | gauges + 1 deg grid | relative to 1995-2014 | SLR fingerprints, the 2020 offset (§7) and the analysis SLR axis |

Definitions used throughout:
- **MSL:** mean sea level.
- **MHWS / MLWS:** mean high / low water springs, the average spring-tide high / low water,
  here as heights above local MSL.
- **Spring range:** MHWS - MLWS, about twice MHWS.
- **HAT / LAT:** highest / lowest astronomical tide.
- **RP10 / RP100:** the level reached on average once in 10 / 100 years.
- **GEV fit:** generalised extreme value distribution fitted to annual maxima; 90% intervals
  from 100 bootstrap refits.
- **TC-dominated station:** one whose COAST-RP TC-only RP100 exceeds its ETC-only RP100 (3,050
  of 23,226 stations).
- **Regions:** longitude-latitude boxes (`levels_analysis.REGIONS`, first match wins). They are
  coarse: "Baltic" includes central Norway, "Mediterranean" includes the Bay of Biscay south of
  46 N, and "Africa+Indian Ocean" includes the Canary Islands.

## 2. Common reference frame

The sources do not share an MSL or a datum, so every comparison removes them first.

- **Local MSL per year.** Each calendar year's mean is removed from GESLA and from the GTSM tide.
  Gauge datum, sea-level rise, land motion and interannual MSL drop out; FES is already zero-mean.
  All levels below are heights above local MSL, never absolute heights. Absolute MSL + MDT
  against gauges with a geodetic datum is not tested.
- **One harmonic method.** GESLA and the GTSM tide are fitted (least squares) with the 34 FES
  constituents, including nodal corrections and inferred minor constituents. FES uses its atlas
  constants. All three are predicted over the same 18.61-year epoch at 10 min, and the same
  statistics code (`src/tide_statistics.py`) derives MHWS, MHW, MLW, MLWS, HAT and LAT.
  Differences in method, epoch or sampling therefore cannot explain what remains.
- **Pairing.** Each gauge is paired with its nearest GTSM / COAST-RP / COAST-HG station (median
  distance 1.9 km). FES is evaluated both at the gauge and at the GTSM station.
- **Quality sites** (698 of 1,154): fit over >= 3 years, explaining >= 50% of the variance, GTSM
  station within 10 km.

## 3. Tidal levels

Figures: `levels_scatter.png` (GTSM vs gauge, FES vs gauge, GTSM vs FES; MHWS), `levels_ratio_maps.png`
(the same ratios as maps), `levels_constituents.png`. Data: `sites_levels.csv`.

| Comparison (MHWS) | Median ratio | MAE | Within 0.1 m |
|---|---|---|---|
| GTSM vs gauge | 1.08 | 0.16 m | 53% |
| FES2022 at the gauge vs gauge | 0.99 | 0.05 m | 88% |
| GTSM vs FES2022 at the GTSM station | 1.08 | 0.16 m | 52% |
| FES2022 at the gauge vs at the GTSM station | 1.00 | 0.01 m | 98% |

Median MHWS ratio per region (the colour scale of the maps):

| region | n | GTSM / gauge | FES / gauge | GTSM / FES | gauge spring range (m) |
|---|---|---|---|---|---|
| Africa+Indian Ocean | 53 | 1.20 | 1.00 | 1.19 | 1.72 |
| Australia+NZ | 100 | 1.05 | 0.99 | 1.06 | 1.44 |
| Baltic | 4 | 1.10 | 1.00 | 1.08 | 2.24 |
| Japan+E Asia | 146 | 1.07 | 0.99 | 1.08 | 1.30 |
| Mediterranean | 23 | 1.18 | 1.00 | 1.21 | 0.88 |
| N America Atlantic+Gulf | 109 | 0.85 | 1.01 | 0.88 | 1.03 |
| N America Pacific | 79 | 1.01 | 1.00 | 1.01 | 1.78 |
| NW Europe | 93 | 1.09 | 0.99 | 1.10 | 4.24 |
| S America | 38 | 1.10 | 0.99 | 1.13 | 1.28 |
| other | 53 | 1.09 | 1.00 | 1.10 | 1.20 |

Stratified sample of `sites_levels.csv` (3 random sites per region, seed 0; MHWS in m):

| site | country | region | lon | lat | spring range | MHWS gauge | MHWS GTSM | MHWS FES | GTSM/gauge | FES/gauge | GTSM/FES |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Durban | ZAF | Africa+Indian Ocean | 31.00 | -29.88 | 1.81 | 0.91 | 1.09 | 0.89 | 1.21 | 0.98 | 1.23 |
| FuerteventuraTG | ESP | Africa+Indian Ocean | -13.85 | 28.50 | 2.33 | 1.18 | 1.39 | 1.17 | 1.19 | 1.00 | 1.19 |
| Las_Palmas | ESP | Africa+Indian Ocean | -15.42 | 28.13 | 2.15 | 1.09 | 1.27 | 1.09 | 1.17 | 1.00 | 1.17 |
| Cooktown | AUS | Australia+NZ | 145.25 | -15.45 | 1.85 | 0.92 | 0.87 | 0.89 | 0.94 | 0.96 | 0.97 |
| Lord_Howe_Island | AUS | Australia+NZ | 159.06 | -31.52 | 1.64 | 0.82 | 0.86 | 0.82 | 1.05 | 0.99 | 1.06 |
| Port_Adelaide_Outer_Harbor | AUS | Australia+NZ | 138.48 | -34.78 | 1.94 | 0.97 | 1.11 | 0.92 | 1.15 | 0.94 | 1.22 |
| Rorvik | NOR | Baltic | 11.25 | 64.87 | 2.25 | 1.14 | 1.23 | 1.14 | 1.08 | 1.00 | 1.08 |
| TrondheimTG | NOR | Baltic | 10.39 | 63.44 | 2.60 | 1.37 | 1.54 | 1.52 | 1.12 | 1.11 | 1.01 |
| Kainan | JPN | Japan+E Asia | 135.19 | 34.14 | 1.45 | 0.72 | 0.94 | 0.65 | 1.30 | 0.90 | 1.45 |
| Maizuru | JPN | Japan+E Asia | 135.32 | 35.45 | 0.19 | 0.10 | 0.11 | 0.10 | 1.09 | 1.07 | 1.02 |
| Naze | JPN | Japan+E Asia | 129.50 | 28.40 | 1.69 | 0.84 | 0.81 | 0.80 | 0.97 | 0.95 | 1.02 |
| Gibraltar | GBR | Mediterranean | -5.35 | 36.13 | 0.93 | 0.47 | 0.53 | 0.42 | 1.13 | 0.90 | 1.24 |
| MotrilTG | ESP | Mediterranean | -3.52 | 36.72 | 0.47 | 0.24 | 0.29 | 0.24 | 1.23 | 1.01 | 1.21 |
| Venezia | ITA | Mediterranean | 12.43 | 45.42 | 0.81 | 0.39 | 0.47 | 0.40 | 1.22 | 1.03 | 1.19 |
| Bedford_Institute | CAN | N America Atlantic+Gulf | -63.62 | 44.68 | 1.65 | 0.84 | 0.69 | 0.80 | 0.82 | 0.95 | 0.86 |
| Pensacola | USA | N America Atlantic+Gulf | -87.21 | 30.40 | 0.54 | 0.27 | 0.30 | 0.30 | 1.09 | 1.11 | 0.98 |
| Tadoussac | CAN | N America Atlantic+Gulf | -69.71 | 48.14 | 4.31 | 2.18 | 3.37 | 2.14 | 1.55 | 0.98 | 1.57 |
| Campbell_River_BC | CAN | N America Pacific | -125.25 | 50.04 | 2.06 | 1.09 | 1.48 | 1.26 | 1.36 | 1.15 | 1.23 |
| Port_Orford | USA | N America Pacific | -124.50 | 42.74 | 2.03 | 0.98 | 0.99 | 0.99 | 1.01 | 1.01 | 1.00 |
| Queen_Charlotte_City | CAN | N America Pacific | -132.07 | 53.25 | 5.63 | 2.73 | 2.89 | 2.50 | 1.06 | 0.92 | 1.15 |
| Holyhead | GBR | NW Europe | -4.62 | 53.31 | 4.91 | 2.38 | 2.51 | 2.28 | 1.06 | 0.96 | 1.11 |
| Newlyn | GBR | NW Europe | -5.54 | 50.10 | 4.72 | 2.33 | 2.62 | 2.29 | 1.12 | 0.98 | 1.15 |
| Saint_Malo | FRA | NW Europe | -2.03 | 48.64 | 10.74 | 5.44 | 6.04 | 5.62 | 1.11 | 1.03 | 1.07 |
| Esmeraldas | ECU | S America | -79.65 | 0.99 | 2.98 | 1.49 | 1.63 | 1.48 | 1.09 | 0.99 | 1.11 |
| Matarani | PER | S America | -72.11 | -17.00 | 0.87 | 0.44 | 0.56 | 0.44 | 1.25 | 0.99 | 1.26 |
| San_Felix | CHL | S America | -80.11 | -26.29 | 0.74 | 0.38 | 0.46 | 0.37 | 1.21 | 0.99 | 1.22 |
| Atka | USA | other | -174.17 | 52.23 | 1.25 | 0.58 | 0.57 | 0.57 | 0.98 | 0.99 | 0.99 |
| Flores_Santa_Cruz | PRT | other | -31.12 | 39.45 | 1.10 | 0.56 | 0.61 | 0.56 | 1.10 | 1.01 | 1.09 |
| Rarotonga_Avarua_Harbour | COK | other | -159.78 | -21.20 | 0.72 | 0.36 | 0.38 | 0.35 | 1.07 | 0.97 | 1.10 |

- **GTSM's tide is 7-8% too large; FES2022 matches the gauges.** GTSM is too large almost
  everywhere and too small on the N American Atlantic and Gulf coast. The GTSM/FES map reproduces
  the GTSM/gauge map, so FES can stand in for gauges where none exist (except in the basins below).
- **Not a location or method effect:** FES is the same at the gauge and at the GTSM station, the
  GTSM bias does not grow with gauge-station distance, and the raw GTSM series and its harmonic
  fit agree.
- **Which constituents:** amplitude ratios GTSM / gauge are S2 1.20, M2 1.06, O1 1.10 and
  MS4 1.28, against 0.99-1.00 for FES. The spring range is roughly M2 + S2, so the S2 and M2
  excess produces the MHWS excess. A plausible but unverified cause is GTSM's treatment of the
  radiational part of S2; FES assimilates data that constrain S2 directly.
- **Where FES misses gauges** (|FES - gauge MHWS| > 0.25 m: 20 of 698 sites; GTSM: 124):
  estuaries, fjords and narrow entrances, in both directions.

  | site | region | spring range | MHWS gauge | MHWS GTSM | MHWS FES | FES/gauge |
  |---|---|---|---|---|---|---|
  | Saint_Francois_IO (upper St Lawrence) | N America Atlantic | 5.29 | 2.87 | 1.58 | 1.27 | 0.44 |
  | Cape_Hatteras_Fishing_Pier | N America Atlantic | 1.11 | 0.56 | 0.40 | 0.27 | 0.48 |
  | Haikou (Qiongzhou Strait) | Japan+E Asia | 1.81 | 0.84 | 0.62 | 0.55 | 0.66 |
  | Delaware_City | N America Atlantic | 1.85 | 0.94 | 0.93 | 0.69 | 0.72 |
  | Madeira (Brazil) | S America | 5.39 | 2.62 | 2.80 | 2.19 | 0.83 |
  | Anchorage | N America Pacific | 9.21 | 4.36 | 4.27 | 3.82 | 0.88 |
  | Newport (Severn) | NW Europe | 11.50 | 6.02 | 6.79 | 5.75 | 0.96 |
  | Arcachon_Eyrac | NW Europe (Biscay) | 3.74 | 1.81 | 1.87 | 2.11 | 1.17 |
  | Bridgeport / Port_Jefferson / Kings_Point / Willets_Point (Long Island Sound) | N America Atlantic | 2.3-2.7 | 1.16-1.35 | 0.76-1.10 | 1.42-1.66 | 1.21-1.26 |
  | Bamfield_BC | N America Pacific | 2.58 | 1.26 | 1.28 | 1.77 | 1.40 |
  | Weymouth | NW Europe | 1.74 | 0.91 | 1.01 | 1.33 | 1.47 |
  | Point_Lonsdale / Queenscliff (Port Phillip entrance) | Australia | 0.70-1.23 | 0.33-0.64 | 1.44-1.45 | 0.83-0.99 | 1.55-2.53 |
  | Patricia_Bay_BC / Fulford_Harbour_Bc (Salish Sea) | N America Pacific | 1.8 | 0.89-0.92 | 1.36 | 1.42-1.99 | 1.55-2.25 |
  | Ocean_City_Inlet | N America Atlantic | 0.78 | 0.37 | 0.41 | 0.62 | 1.69 |
  | Juneau | N America Pacific | 5.60 | 2.72 | 4.74 | 5.95 | 2.19 |

- **Macrotidal basin heads without gauges** (Bay of Fundy, Strait of Magellan, upper Gulf of
  California). These basins do have extreme tides, and FES gets most of them right:
  - **Bay of Fundy, where gauges exist:** FES matches the gauges (Saint John MHWS 3.50 vs 3.54 m),
    while GTSM is 25% too low.
  - **Minas Basin:** FES's spring range is plausible at Parrsboro and Cape Split (14-15 m). At
    Burntcoat Head it is about 16 m, against a published mean spring range of 14.5 m (Guinness
    World Records) and a 16.3 m record range.
  - **Upper Gulf of California:** FES's range at San Felipe and Puerto Peñasco (7.0-7.6 m over
    two years) agrees with the ~7 m spring range reported for the uppermost Gulf.
  - **Strait of Magellan:** FES MHWS of 4.0-4.6 m at the first narrows agrees with the reported
    9 m spring range on the Atlantic side (Medeiros and Kjerfve, 1988).

  What fails is a small set of FES nodes at the very heads:

  | FES nodes | FES MHWS above MSL | FES spring range | Reference |
  |---|---|---|---|
  | Cobequid Bay head, 63.4-63.5 W 45.37 N (23 valid nodes in the Minas box, median 14.45 m) | 14.5 m | ~29 m | Minas spring range 14.5 m (MHWS ~7.3 m) |
  | Avon estuary (Hantsport) | M2 alone 8.6 m | ~19 m | as above |
  | Bahía Lomas, 69.1-69.2 W 52.67 S | 15.5-16.3 m | ~32 m | Strait spring range 9 m on the Atlantic side; FES at Río Gallegos 27.6 m vs ~9-11 m in tide tables |
  | Colorado delta, 114.93 W 31.90 N (one node) | 15.4 m | ~31 m | neighbouring nodes ~3.0 m; spring range 7-10 m at the river mouth |

  At these nodes FES's MHWS above MSL is as large as the full observed spring range. These are
  the values that flood the eikonal tiles: the solver seeds each boundary cell with its own
  level, so on calibration tile 891 (Minas Basin) CSI against SFINCS fell from 0.57 to 0.27.
  FES's native / extrapolated mask does not flag these nodes.

## 4. Extremes

Compared: COAST-RP return levels against GEV fits to GESLA annual maxima (record median 30 y)
and to 46 years of GTSM (surge + harmonic tide). Data: `sites_extremes.csv`.

| Comparison | RP10 median diff | RP10 MAE | RP100 median diff | RP100 MAE |
|---|---|---|---|---|
| COAST-RP - gauge GEV | -0.06 m | 0.22 m | -0.06 m | 0.37 m |
| GTSM GEV - gauge GEV | -0.06 m | 0.20 m | -0.11 m | 0.37 m |
| COAST-RP - GTSM GEV (same model, different statistics) | -0.01 m | 0.07 m | +0.02 m | 0.15 m |

- **Tide and surge errors offset each other.** COAST-RP's RP10 error tracks GTSM's tide excess
  (correlation 0.88). With the tide excess removed, COAST-RP is 0.11-0.12 m too low: its surge
  part is too low, and the too-large tide hides that. Only 18% (RP10) / 44% (RP100) of COAST-RP
  levels fall inside the gauge's 90% interval.
- **COAST-RP against the gauges, by station class:**

  | Station class | Sites | COAST-RP - gauge GEV, RP10 / RP100 |
  |---|---|---|
  | TC-dominated | 194 | -0.19 / -0.24 m |
  | ETC-dominated, TC value present | 200 | -0.04 / -0.08 m |
  | no TC station | 254 | +0.04 / +0.02 m |

  The TC part adds 0.17 m (median) to RP100 at TC-dominated stations and is still too low
  (Japan / E Asia -0.25 m, US Atlantic and Gulf -0.24 m). A 30-40-year gauge record can also
  miss rare cyclones, so the gauge GEV is itself uncertain there.
- **GTSM's own extremes against COAST-RP** (`compare_peaks.csv`; GTSM GEV - COAST-RP):

  | Station class | Sites | RP10 median (10-90%) | RP10 MAE | RP100 median (10-90%) | RP100 MAE |
  |---|---|---|---|---|---|
  | all | 698 | +0.01 (-0.14..+0.07) | 0.07 | -0.02 (-0.27..+0.09) | 0.15 |
  | ETC-dominated | 229 | +0.01 (-0.07..+0.05) | 0.04 | 0.00 (-0.11..+0.07) | 0.09 |
  | no TC value | 267 | -0.03 (-0.24..+0.04) | 0.09 | -0.04 (-0.28..+0.05) | 0.13 |
  | TC-dominated | 202 | +0.03 (-0.06..+0.15) | 0.06 | -0.04 (-0.45..+0.27) | 0.25 |

  Where ETC surges dominate, GTSM and COAST-RP are the same model to within a few cm; the
  remaining difference is the statistics (GEV on 46 annual maxima vs COAST-RP's method). At TC
  stations the RP100 spread is wide because COAST-RP's TC part replaces GTSM there, in both
  directions.
- **Fundy and SF Bay:** COAST-RP RP100 is 1.2-1.4 m below the gauge GEV in the outer Bay of Fundy
  (Saint John 3.45 vs 4.66 m) and 0.3-0.5 m below in SF Bay. No source overestimates there.

## 5. Hydrograph shapes

Events are the annual maxima at each gauge, with ±72 h of record around each peak: 21,100 GESLA
and 32,100 GTSM events. COAST-HG gives one RP100 event per station. Each event's empirical return
period is its rank among the site's annual maxima.

Duration measures, in hours, for levels relative to local MSL:
- **Contiguous time above MSL:** the unbroken stretch above MSL that contains the peak.
- **T(f):** the total time, across all tidal cycles in the window, that the level is within
  f × (peak - MSL) of the peak, i.e. at or above (1 - f) of the peak. This is the T of the eikonal
  duration correction for a cell whose water depth is that fraction of the storm tide
  (`event_volume.time_within_peak`).
- **Neighbouring high water (nb12, nb24):** the highest level / peak 9-16 h and 21-28 h from the
  peak, averaged over both sides.

Figures: `shapes_profiles.png`, `shapes_profiles_by_rp.png`, `shapes_durations.png`.
Data: `sites_shapes.csv`.

### 5.1 All annual maxima

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

### 5.2 The largest events per site: GTSM and COAST-HG against the gauge

The median normalised profile of the 3 largest annual maxima per site, for GESLA and GTSM,
against the COAST-HG RP100 profiles at the paired station (`compare_top_events.csv`, medians over
sites). RMSE is the profile difference from GESLA over ±48 h, in units of the peak.

| Spring range | Source | nb12 | nb24 | T(0.25) h | T(0.5) h | RMSE vs GESLA |
|---|---|---|---|---|---|---|
| < 1 m | GESLA | 0.52 | 0.64 | 8 | 29 | - |
| | GTSM | 0.46 | 0.70 | 8 | 27 | 0.22 |
| | COAST-HG spring | 0.49 | 0.61 | 6 | 19 | 0.25 |
| | COAST-HG average | 0.66 | 0.50 | 7 | 24 | 0.27 |
| 1-2 m | GESLA | 0.61 | 0.82 | 11 | 32 | - |
| | GTSM | 0.69 | 0.86 | 12 | 36 | 0.15 |
| | COAST-HG spring | 0.63 | 0.73 | 5 | 23 | 0.17 |
| | COAST-HG average | 0.71 | 0.57 | 5 | 19.5 | 0.22 |
| 2-4 m | GESLA | 0.75 | 0.86 | 13 | 40 | - |
| | GTSM | 0.77 | 0.89 | 14 | 39 | 0.13 |
| | COAST-HG spring | 0.68 | 0.75 | 6 | 26 | 0.14 |
| | COAST-HG average | 0.72 | 0.61 | 5 | 21 | 0.22 |
| > 4 m | GESLA | 0.84 | 0.88 | 17 | 41 | - |
| | GTSM | 0.85 | 0.89 | 17 | 41 | 0.10 |
| | COAST-HG spring | 0.76 | 0.73 | 6 | 35 | 0.12 |
| | COAST-HG average | 0.70 | 0.58 | 3 | 23 | 0.22 |

- GTSM's largest events match the gauge's in every regime: same T, same neighbouring high
  waters, and the smallest profile error.
- COAST-HG's central peak is right, but its high water one day later (nb24) is 0.1-0.15 of the
  peak too low. The average-tide variant is lower still. Its T(0.25) is 5-6 h against 11-17 h for
  observed and GTSM extremes in meso- and macrotidal regimes; with B's m ∝ 1/T, that is 2-3x too
  much friction.
- The GTSM and gauge sets are each source's own 3 largest events, not necessarily the same storms.

### 5.3 Shape against return period

Rarer events are more surge-dominated. From RP < 2 y to RP >= 10 y at the gauges, the surge share
of the peak rises from 0.36 to 0.63 (range < 1 m), 0.15 to 0.35 (1-2 m) and 0.05 to 0.13
(> 4 m). Their tidal neighbours are therefore relatively lower, and T falls (`compare_t_by_rp.csv`):

| Spring range | GESLA T(0.25), RP < 2 y → RP >= 10 y | ratio | GTSM, same | ratio |
|---|---|---|---|---|
| < 1 m | 14 → 9 h | 1.56 | 13 → 10.5 h | 1.24 |
| 1-2 m | 16 → 11 h | 1.45 | 16 → 13 h | 1.23 |
| 2-4 m | 17 → 14 h | 1.21 | 17 → 15 h | 1.13 |
| > 4 m | 22 → 17 h | 1.29 | 22 → 19 h | 1.16 |

T(0.5) changes less (1.0-1.3), since it spans several tidal cycles either way. The pipeline uses
COAST-HG's RP100 shape for every RP. A shape taken from rare events therefore gives frequent
events (RP 2-5) a T that is 15-55% too short: as much again in friction at α = 1, on top of
COAST-HG's own short T.

### 5.4 Tropical-cyclone stations

Gauge coverage (`compare_tc_coverage.csv`): share of COAST-RP stations with a quality gauge within

| Station class | Stations | 25 km | 50 km | 100 km | 250 km |
|---|---|---|---|---|---|
| TC-dominated | 3,050 | 19% | 30% | 43% | 64% |
| ETC-dominated | 6,524 | 10% | 18% | 29% | 52% |
| no TC value | 13,652 | 11% | 21% | 32% | 46% |

The 3 largest gauge events per site (`compare_tc_events.csv`, medians over sites; the scale
factors bring them to COAST-RP RP100):

| Station class | Surge duration above half its maximum | Surge share of peak | Whole-event scale | Surge-only scale (10-90%) |
|---|---|---|---|---|
| TC-dominated (194) | 15 h | 0.54 | 0.90 | 0.97 (0.59-1.93) |
| ETC-dominated (200) | 18 h | 0.21 | 0.97 | 1.03 |
| no TC value (254) | 18 h | 0.25 | 1.04 | 1.29 |

- At TC-dominated gauges the largest observed events are surge-dominated and somewhat shorter
  than elsewhere (15 vs 18 h). Their T(0.25) of 6 h is close to COAST-HG's (5 h) and to GTSM's
  (7 h; `compare_top_events.csv` by station class), so COAST-HG is closest to observed shapes at TC stations and furthest
  elsewhere.
- In half of the TC-dominated sites the largest observed event already exceeds COAST-RP RP100
  (whole-event scale < 1), consistent with COAST-RP being low there (§4).
- **Using gauge shapes at TC stations** is supported where a gauge is close. Scale only the surge
  part (observed minus harmonic tide) to the target peak and keep the local tide: scaling the
  whole hydrograph would also scale the tide. Limits: only 30% of TC stations have a gauge within
  50 km; the scale factor spreads widely (0.6-1.9); and hourly gauge data under-resolve the
  shortest TC peaks.

## 6. Implications for the duration correction

The eikonal friction is multiplied by m = clip((t_ref / T)^α, 1, 10), with t_ref = 12 h. A longer
T gives a smaller m, less friction and a longer inland reach, which is the physically expected
direction. The forms of m were compared on 540 synthetic 1-D SFINCS floodplains
(`duration_experiment.py`, `duration_refit.py`), with fsf refitted per form:

- In SFINCS, absolute duration sets the inland reach where terrain allows a long reach. Under
  steady forcing there is no level loss.
- With fsf calibrated, a constant t_ref is absorbed into fsf; "12 h" only sets where the lower
  clip at 1 starts. Removing the clip fits slightly better (log-extent RMSE 0.140 vs 0.148).
  α = 0.5 and 1 perform alike.
- A reference taken from the hydrograph's own longest exposure, or from the calibration
  hydrograph's T at the same depth, fits worse than either.
- T should come from hydrographs with realistic neighbouring high waters and a shape that fits the
  RP (§5.2-5.3). GTSM events satisfy both outside TC regions.

## 7. Datum epoch of the zero-SLR scenario

`total_wl = storm_tide(RP) + MDT + baseline_offset + SLR_fingerprint` (`preparation/prepare_boundary_conditions.py`).

- COAST-RP is relative to model MSL and MDT is a 1993-2012 mean, so storm tide + MDT is the sea
  level of about 2003.
- The analysis reads the SLR axis against AR6 global-mean trajectories rebased to 2020
  (`analysis/extract_slr_trajectories.py`), and population is a 2020 baseline.
- `baseline_offset` closes that gap. It is AR6's global-mean rise from 1995-2014 to 2020 (0.049 m,
  median, equal across SSPs; `boundary_conditions.slr_baseline_offset_m`), spread regionally with
  the same fingerprint scaling as the SLR scenarios. SLR_0 is therefore the 2020 sea level.
- The two-component static level adds the offset back from the MDT station file, because it
  cancels in total(SLR) - total(SLR_0) (`boundaries.static_station_levels`).
- The residual mismatch is the two years between the MDT epoch and the AR6 baseline (~6 mm).

## 8. Open questions

1. **Tide-corrected COAST-RP:** recombine the GTSM surge distribution with the FES tide, as COAST-RP
   combines surge and tide, and test it against the gauge GEV. A level-only swap (subtract GTSM
   MHWS, add FES MHWS) removes the tide excess but exposes the surge deficit.
2. Is the surge deficit (about 0.1 m outside TC regions) regional and stable enough for a
   correction calibrated at gauges?
3. **FES basin-head rule:** detect the head nodes of §3 automatically. A neighbour test does not
   work in Cobequid Bay, where 23 nodes share the bad value. Options: compare against GTSM where a
   station exists, or bound MHWS by a regional reference spring range; review the flagged nodes by
   hand.
4. **TC hydrographs:** a Gaussian surge at TC-dominated RPs (§8.1). Gauge shapes, with
   surge-only scaling, are an option only where a gauge is within ~50 km (30% of TC stations).
5. Absolute MSL + MDT error at gauges with a geodetic datum tie.

### 8.1 Next step: a per-RP storm-tide hydrograph generator, validated at the gauges

**Goal.** One hydrograph per COAST-RP station and return period that forces SFINCS and sets the T
of the eikonal duration correction, replacing COAST-HG in both roles.

**Why not re-use the method as is.** COAST-HG is HGRAPHER output (Dullaart et al. 2023): a GTSM
surge shape scaled to COAST-RP RP100 and laid on a tide. `src/storm_tide_hydrograph.py` ports the
same method with per-RP shape selection and a spring-tide window, so it may inherit COAST-HG's
errors (§5.2): next-day high water 0.1-0.15 of the peak too low, and T(0.25) 5-6 h against
11-17 h observed. The cause has to be found before the generator is adopted.

**Generator.** Per station and RP, with levels relative to local MSL:
- **Tide:** FES2022 at the station, 10 min, ±72 h around a spring high water (FES matches the
  gauges, §3).
- **Surge shape:** selected for the RP from GTSM surge events 1979-2024 (HGRAPHER-RP event
  selection). The shape class follows the component that dominates COAST-RP at that RP: ETC from
  GTSM; a Gaussian at TC-dominated RPs, half width 8 h, matching the 15 h median surge duration of
  the largest events at TC-dominated gauges (§5.4).
- **Peak:** the surge is scaled so the maximum of tide + surge equals the peak level. The peak
  level is an input (COAST-RP for now), so a tide-corrected peak (question 1) can replace it
  without changing the shape code. With an FES tide, the surge amplitude COAST-RP - MHWS_FES
  absorbs GTSM's tide excess until then.
- MDT, the 2020 offset and SLR are added downstream; the shape is the same for every SLR scenario.

**Diagnosis of COAST-HG's error.** Swap one component at a time and score each variant at the 698
quality sites:

| Component | Variant 1 | Variant 2 |
|---|---|---|
| Tide window | average tide (COAST-HG) | spring-tide window (FES) |
| Surge shape | HGRAPHER level-by-level mean of limb durations | empirical mean of the normalised surge residuals of the selected events, centred on their peaks |
| Surge-tide phasing | surge peak on the spring high water | the phase of the selected GTSM events relative to high water |

**Acceptance test.** Build hydrographs at RP2 and RP10-100 and compare their normalised shapes
with the gauge events of the matching empirical-RP class (< 2 y; >= 10 y), per tidal-range class:
- the median T(0.25), T(0.5) and nb24 within ±15% of the gauge medians;
- the profile RMSE against the gauge no larger than that of GTSM's own events (0.10-0.22, §5.2);
- the T(0.25) ratio between RP classes reproduced (1.2-1.6, §5.3).

**Outputs.**
- Per station and RP: a 10-min hydrograph over ±72 h.
- Per station, RP and water depth: a T lookup (`event_volume.time_within_peak`).
- The test scores per variant, regime and RP class.

Inputs are the GTSM series, FES constants and GESLA events already extracted for this note; no
HPC runs are needed. Production use (the global product, SFINCS forcing, fsf recalibration with
α = 1 and no lower clip on the multiplier) follows only once a variant passes.

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
9. `sources_compare.py`
10. `duration_experiment.py` (local SFINCS)
11. `duration_analysis.py`
12. `duration_refit.py`
13. `export_input_tables.py`

Inputs on `<data_root>` are read only.

## 10. Data files

`boundary_test/input_analysis_data/`, written by `export_input_tables.py`. There is one row per
quality site (698), keyed by the GESLA file name; levels are in m above local MSL.

- **`sites_levels.csv`**: the data of `levels_scatter.png` and `levels_ratio_maps.png`.
  - Site information: site, country, region, lon, lat, `gtsm_dist_km`, `fit_years`.
  - Levels: `spring_range_gesla_m`, and MHWS of GESLA, GTSM, FES at the gauge and FES at the
    GTSM station.
  - The three ratios.
- **`sites_extremes.csv`**:
  - GESLA GEV RP10 / RP100 with 5% / 95% bounds (`n_annual_maxima_gesla` years);
  - GTSM GEV RP10 / RP100;
  - COAST-RP total, ETC-only and TC-only RP10 / RP100;
  - `station_class` (tc / etc / no_tc).
- **`sites_shapes.csv`**: one row per site and source (gesla, gtsm, coast_hg_spring,
  coast_hg_average), for the 3 largest events: nb12, nb24, T(0.1 / 0.25 / 0.5), contiguous hours
  above MSL, `rmse_vs_gesla`, `range_class`, `station_class`.
- **`tables.md`**: the regional summary, the stratified sample and the FES-miss table above, as
  generated.

Sources for the reference tidal ranges in §3: Guinness World Records (greatest mean spring range,
Burntcoat Head, 14.5 m); Medeiros, C. and Kjerfve, B. (1988), Tidal characteristics of the Strait
of Magellan; the Wikipedia article on San Felipe, Baja California
(~7 m spring tides) and its Gulf of California references.
