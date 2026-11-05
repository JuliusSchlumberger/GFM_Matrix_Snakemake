# Boundary forcing inputs: datasets and how they compare

This note assesses the coastal water-level sources that can force the eikonal and SFINCS models:
what each dataset is, the common reference frame they are compared in, where and why they
differ, and how the findings sit against the published literature and against the boundary
conditions used by other global coastal flood models. It answers four questions:

1. Do the sources share a vertical reference and a mean sea level?
2. How well do the tidal levels and extremes of the global models match tide gauges?
3. What shape do storm-tide hydrographs have, and what does that mean for the duration
   correction of the eikonal friction?
4. What should be tested, decided and refined first before the inputs go into production (§8),
   and how does this input chain compare with Fathom's RFA (Collings et al., 2024; Wing et al.,
   2024) and the skew-surge + HAT seeding of Chaudhuri et al. (2027) (§9)?

Scope: global, at the GESLA tide gauges. Model comparisons (eikonal vs SFINCS on real tiles) are
outside this note.
- **Scripts:** `boundary_test/`.
- **Per-site data:** `boundary_test/input_analysis_data/` (§11), so the numbers here can be
  checked and re-cut.
- **Figures:** `.cleanup_baseline/level_sources/analysis/` (local, not in git).
- **References:** §12.

## 1. Datasets

| Dataset | What it is | Coverage | Vertical reference | Role in the pipeline |
|---|---|---|---|---|
| **GESLA 4.1** | observed total water level at tide gauges (hourly or finer); successor of GESLA-3 (Haigh et al., 2023) | 1,154 sites with >= 20 y record | each gauge's own datum (chart datum, national datum, often unspecified) | reference ("truth") for this note |
| **GTSM v3 tide** (Copernicus CDS) | modelled tide from GTSMv3.0 (Muis et al., 2020): astronomic tide, self-attraction and loading, radiational tides and a rising mean sea level. Forward model with uncalibrated bathymetry and bottom friction; the calibrated successor is GTSMv4.1 (Wang et al., 2022) | global model stations, 10 min, 1996-2014 | model MSL (1986-2005; Collings et al., 2024), with a built-in trend (median 2.6 mm/yr) | tide component of COAST-RP |
| **GTSM v3 surge** (CDS) | modelled water level minus tide, forced by ERA5 | hourly, 1979-2024 | relative to the tide run, no trend | surge component of COAST-RP (extratropical) |
| **FES2022** | ocean-tide atlas, 34 harmonic constituents, assimilates altimetry and gauges; includes an atmospheric (radiational) S2 forcing, as FES2014 did (Lyard et al., 2021) | global grid, 1/30 deg | zero-mean by construction | candidate replacement for the GTSM tide |
| **COAST-RP** | storm-tide return levels (RP 1-1000 y) per GTSM station (Dullaart et al., 2021): GTSM surge statistics (extratropical, ETC) and STORM synthetic cyclones (tropical, TC; Bloemendaal et al., 2020), combined with the tide | 23,226 stations; separate ETC-only and TC-only (9,609 stations) files | GTSM model MSL, no MDT | boundary level in production (+ MDT + 2020 offset + SLR) |
| **COAST-HG** | one RP100 storm-tide hydrograph per COAST-RP station (HGRAPHER; Dullaart et al., 2023), 149 h at 10 min, average- and spring-tide variants | as COAST-RP | as COAST-RP; its peak equals COAST-RP RP100 exactly | hydrograph shape for the duration correction and for SFINCS calibration forcing |
| **MDT CNES-CLS22** | mean dynamic topography: mean sea surface above the geoid (successor of CNES-CLS18; Mulet et al., 2021) | global ocean, 1/8 deg | GOCO06s geoid, mean of 1993-2012 | converts COAST-RP from local MSL to the DEM's geoid datum |
| **IPCC AR6 sea-level projections** | regional and global-mean SLR relative to 1995-2014 (Fox-Kemper et al., 2021; Garner et al., 2021) | gauges + 1 deg grid | relative to 1995-2014 | SLR fingerprints, the 2020 offset (§7) and the analysis SLR axis |

Definitions used throughout:
- **MSL:** mean sea level.
- **MHWS / MLWS:** mean high / low water springs, the average spring-tide high / low water,
  here as heights above local MSL.
- **Spring range:** MHWS - MLWS, about twice MHWS.
- **HAT / LAT:** highest / lowest astronomical tide.
- **Skew surge:** the difference between the observed high water and the nearest predicted
  tidal high water, regardless of timing (Williams et al., 2016). Not yet used in this note;
  proposed for the revised extremes comparison (§8).
- **RP10 / RP100:** the level reached on average once in 10 / 100 years.
- **GEV fit:** generalised extreme value distribution fitted to annual maxima; 90% intervals
  from 100 bootstrap refits.
- **TC-dominated station:** one whose COAST-RP TC-only RP100 exceeds its ETC-only RP100 (3,050
  of 23,226 stations).
- **Regions:** country rules, then longitude-latitude boxes (`levels_analysis.region_of`, first
  match wins). "Norway+Nordic Seas" holds Norway, Iceland, the Faroes and Svalbard;
  "Baltic+Kattegat" holds the Baltic-coast countries east of 9.4 E; "Mediterranean+Black Sea"
  excludes the Atlantic coast north of 40 N (Biscay belongs to NW Europe); "NE Atlantic islands"
  holds the Canaries, Madeira and the Azores. The tide set has no Baltic site at all: every
  Baltic gauge fails the harmonic-fit filter (§2).

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

Two properties of this frame were tested (§8.1, tests 1 and 3):

- **The site set is tide-selected.** "Harmonic fit explains >= 50% of the variance" is the right
  filter for the tide comparison and the wrong one for extremes and shapes: it removes 117
  sites, 78 of them in the Baltic and 20 in the Mediterranean, i.e. the surge-dominated
  microtidal coasts. Test 1 therefore scores §4-§5 also on an **extremes set**: every site with
  >= 20 complete years since 1979 and a GTSM station within 10 km, whatever its tide (626 sites;
  552 shared with the tide set, 74 only in the extremes set, 42 of them Baltic). Results for both
  sets are in §4 and §5: the overall numbers barely move, the Baltic and Mediterranean ones do.
- **Removing annual means could remove real hazard.** Interannual MSL variability (ENSO, NAO,
  steric) is part of the flood level, and COAST-RP (barotropic, fixed MSL) cannot represent it.
  Test 3 refits the gauge GEV with a linear trend removed instead of annual means: the RP10 level
  rises by a median 0.010 m (interquartile 0.003-0.019 m), the RP100 level by 0.007 m (within
  the fit noise, -0.03..+0.03 m). The largest regional RP10 effects are S America (+0.023 m),
  N America Pacific (+0.018 m) and the Baltic (+0.014 m). The seasonal cycle stays in both fits.
  The term is an order of magnitude below the surge deficit, so the production level needs no
  MSL-anomaly term (`msl_sensitivity.png`).

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
| Africa+Indian Ocean | 45 | 1.21 | 1.00 | 1.21 | 1.65 |
| Australia+NZ | 100 | 1.05 | 0.99 | 1.06 | 1.44 |
| Japan+E Asia | 146 | 1.07 | 0.99 | 1.08 | 1.30 |
| Mediterranean+Black Sea | 16 | 1.22 | 1.00 | 1.22 | 0.52 |
| N America Atlantic+Gulf | 109 | 0.85 | 1.01 | 0.88 | 1.03 |
| N America Pacific | 79 | 1.01 | 1.00 | 1.01 | 1.78 |
| NE Atlantic islands | 11 | 1.17 | 1.00 | 1.18 | 2.04 |
| NW Europe | 87 | 1.10 | 0.99 | 1.10 | 4.38 |
| Norway+Nordic Seas | 22 | 1.08 | 0.99 | 1.10 | 2.20 |
| S America | 38 | 1.10 | 0.99 | 1.13 | 1.28 |
| other | 45 | 1.09 | 1.00 | 1.10 | 1.07 |

Stratified sample of `sites_levels.csv` (3 random sites per region, seed 0; MHWS in m):

| site | country | region | lon | lat | spring range | MHWS gauge | MHWS GTSM | MHWS FES | GTSM/gauge | FES/gauge | GTSM/FES |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dakar | SEN | Africa+Indian Ocean | -17.42 | 14.68 | 1.31 | 0.65 | 0.61 | 0.64 | 0.94 | 0.99 | 0.95 |
| East_London | ZAF | Africa+Indian Ocean | 27.92 | -33.00 | 1.62 | 0.81 | 1.00 | 0.81 | 1.23 | 1.00 | 1.23 |
| Hiron_Point | BGD | Africa+Indian Ocean | 89.47 | 21.78 | 2.41 | 1.18 | 1.44 | 1.28 | 1.22 | 1.09 | 1.13 |
| Cooktown | AUS | Australia+NZ | 145.25 | -15.45 | 1.85 | 0.92 | 0.87 | 0.89 | 0.94 | 0.96 | 0.97 |
| Lord_Howe_Island | AUS | Australia+NZ | 159.06 | -31.52 | 1.64 | 0.82 | 0.86 | 0.82 | 1.05 | 0.99 | 1.06 |
| Port_Adelaide_Outer_Harbor | AUS | Australia+NZ | 138.48 | -34.78 | 1.94 | 0.97 | 1.11 | 0.92 | 1.15 | 0.94 | 1.22 |
| Kainan | JPN | Japan+E Asia | 135.19 | 34.14 | 1.45 | 0.72 | 0.94 | 0.65 | 1.30 | 0.90 | 1.45 |
| Maizuru | JPN | Japan+E Asia | 135.32 | 35.45 | 0.19 | 0.10 | 0.11 | 0.10 | 1.09 | 1.07 | 1.02 |
| Naze | JPN | Japan+E Asia | 129.50 | 28.40 | 1.69 | 0.84 | 0.81 | 0.80 | 0.97 | 0.95 | 1.02 |
| Ajaccio_Aspretto | FRA | Mediterranean+Black Sea | 8.76 | 41.92 | 0.24 | 0.12 | 0.14 | 0.11 | 1.18 | 0.91 | 1.29 |
| Ceuta | ESP | Mediterranean+Black Sea | -5.32 | 35.90 | 0.88 | 0.45 | 0.52 | 0.44 | 1.17 | 0.98 | 1.22 |
| Venezia | ITA | Mediterranean+Black Sea | 12.43 | 45.42 | 0.81 | 0.39 | 0.47 | 0.40 | 1.22 | 1.03 | 1.19 |
| Bedford_Institute | CAN | N America Atlantic+Gulf | -63.62 | 44.68 | 1.65 | 0.84 | 0.69 | 0.80 | 0.82 | 0.95 | 0.86 |
| Pensacola | USA | N America Atlantic+Gulf | -87.21 | 30.40 | 0.54 | 0.27 | 0.30 | 0.30 | 1.09 | 1.11 | 0.98 |
| Tadoussac | CAN | N America Atlantic+Gulf | -69.71 | 48.14 | 4.31 | 2.18 | 3.37 | 2.14 | 1.55 | 0.98 | 1.57 |
| Campbell_River_BC | CAN | N America Pacific | -125.25 | 50.04 | 2.06 | 1.09 | 1.48 | 1.26 | 1.36 | 1.15 | 1.23 |
| Port_Orford | USA | N America Pacific | -124.50 | 42.74 | 2.03 | 0.98 | 0.99 | 0.99 | 1.01 | 1.01 | 1.00 |
| Queen_Charlotte_City | CAN | N America Pacific | -132.07 | 53.25 | 5.63 | 2.73 | 2.89 | 2.50 | 1.06 | 0.92 | 1.15 |
| FuerteventuraTG | ESP | NE Atlantic islands | -13.85 | 28.50 | 2.33 | 1.18 | 1.39 | 1.17 | 1.19 | 1.00 | 1.19 |
| Funchal | PRT | NE Atlantic islands | -16.91 | 32.64 | 2.04 | 1.03 | 1.21 | 1.04 | 1.17 | 1.01 | 1.16 |
| Pluz | ESP | NE Atlantic islands | -15.41 | 28.15 | 2.15 | 1.09 | 1.27 | 1.09 | 1.17 | 1.00 | 1.17 |
| Lowestoft | GBR | NW Europe | 1.75 | 52.47 | 1.85 | 0.84 | 0.88 | 0.82 | 1.05 | 0.97 | 1.07 |
| Newlyn | GBR | NW Europe | -5.54 | 50.10 | 4.72 | 2.33 | 2.62 | 2.29 | 1.12 | 0.98 | 1.15 |
| St_Helier_Jersey | GBR | NW Europe | -2.12 | 49.18 | 9.71 | 4.97 | 5.40 | 4.87 | 1.09 | 0.98 | 1.11 |
| AndenesTG | NOR | Norway+Nordic Seas | 16.14 | 69.33 | 1.85 | 0.92 | 0.99 | 0.91 | 1.07 | 0.99 | 1.08 |
| Ny_alesund | NOR | Norway+Nordic Seas | 11.94 | 78.93 | 1.30 | 0.65 | 0.71 | 0.63 | 1.11 | 0.97 | 1.14 |
| Vardo | NOR | Norway+Nordic Seas | 31.10 | 70.38 | 2.73 | 1.34 | 1.24 | 1.36 | 0.92 | 1.01 | 0.91 |
| Esmeraldas | ECU | S America | -79.65 | 0.99 | 2.98 | 1.49 | 1.63 | 1.48 | 1.09 | 0.99 | 1.11 |
| Matarani | PER | S America | -72.11 | -17.00 | 0.87 | 0.44 | 0.56 | 0.44 | 1.25 | 0.99 | 1.26 |
| San_Felix | CHL | S America | -80.11 | -26.29 | 0.74 | 0.38 | 0.46 | 0.37 | 1.21 | 0.99 | 1.22 |
| Kerguelen | FRA | other | 70.22 | -49.34 | 1.56 | 0.78 | 0.82 | 0.77 | 1.05 | 0.98 | 1.07 |
| Lombrum | PNG | other | 147.38 | -2.04 | 0.69 | 0.30 | 0.29 | 0.32 | 0.97 | 1.05 | 0.93 |
| Nuku_Hiva | FRA | other | -140.08 | -8.93 | 1.20 | 0.61 | 0.68 | 0.60 | 1.12 | 1.00 | 1.13 |

- **GTSM's tide is 7-8% too large; FES2022 matches the gauges.** GTSM is too large almost
  everywhere and too small on the N American Atlantic and Gulf coast. The GTSM/FES map reproduces
  the GTSM/gauge map, so FES can stand in for gauges where none exist (except in the basins below).
- **Not a location or method effect:** FES is the same at the gauge and at the GTSM station, the
  GTSM bias does not grow with gauge-station distance, and the raw GTSM series and its harmonic
  fit agree.
- **Which constituents:** amplitude ratios GTSM / gauge are S2 1.20, M2 1.06, O1 1.10 and
  MS4 1.28, against 0.99-1.00 for FES. The spring range is roughly M2 + S2, so the S2 and M2
  excess produces the MHWS excess. The likely cause is the uncalibrated bathymetry and bottom
  friction of GTSMv3.0 rather than its treatment of radiational S2: FES already carries an
  atmospheric S2 forcing (Lyard et al., 2021), and Wang et al. (2022) show that a significant
  part of GTSM's remaining tide error is parameter uncertainty, reduced in GTSMv4.1 by a joint
  bathymetry-friction estimation that used FES2014 as the reference in most areas. One check
  remains: whether the CDS tide-only run was forced with ERA5 pressure. If not, radiational S2
  sits in the CDS "surge" component, and recombining GTSM surge with FES tide (which has its own
  radiational S2) double-counts a few cm; small, but it should be stated.
- **Literature context.** The direction and size of the bias are consistent with Muis et al.
  (2020), who note that GTSM biases increase in regions with a high tidal range and that GTSM may
  be outperformed by FES there, and with Collings et al. (2024), who for the same reason isolate
  the GTSM-ERA5 surge with the tide-only run and lay it on an FES2014 tide. Both corroborate
  replacing the GTSM tide by FES; the new result here is the size (7-8% in MHWS, 20% in S2) and
  its global map.
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
    Burntcoat Head it is about 16 m, against a mean spring range of about 14.5 m and a record
    range of 16.3 m (Desplanque and Mossman, 2001; Archer, 2013).
  - **Upper Gulf of California:** FES's range at San Felipe and Puerto Peñasco (7.0-7.6 m over
    two years) agrees with the ~7 m spring range reported for the uppermost Gulf (Archer, 2013).
  - **Strait of Magellan:** FES MHWS of 4.0-4.6 m at the first narrows agrees with the reported
    9 m spring range on the Atlantic side (Medeiros and Kjerfve, 1988; Archer, 2013).

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
  FES's native / extrapolated mask does not flag these nodes. Two other global pipelines read
  FES at the coast without such a filter: Collings et al. (2024) scale their index flood by the
  FES2014 tidal range at every coastline point, and Chaudhuri et al. (2027) join each coastal
  seed to the nearest FES2022b HAT vertex within 20 km. Whether FES2014 shares the artefact is
  one of the checks in §8.1 (test 6).

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
  levels fall inside the gauge's 90% interval. (The interval statistic overstates the
  disagreement: the gauge interval at RP10 is a few cm wide after 30 years, the stations sit
  1.9 km from the gauge in a 2.5 km model, and harbour and river effects at the gauge are not
  resolved. The median and MAE are the numbers to carry forward.)
- **Statistics vs model.** COAST-RP's return periods are empirical, computed after a
  probabilistic combination of surge and tide (Dullaart et al., 2021); they are not a POT / GPD
  fit. The "same model, different statistics" row (GEV on 46 GTSM annual maxima vs COAST-RP)
  accounts for about a third (RP10, 0.07 of 0.22 m) to two fifths (RP100, 0.15 of 0.37 m) of the
  COAST-RP - gauge MAE. A clean separation needs one estimator for all three combinations of
  surge and tide (§8.1, tests 2 and 5).
- **Extremes set (test 1).** On the record-based set (623 sites with a GEV fit) COAST-RP - gauge
  is -0.061 / -0.072 m (RP10 / RP100), MAE 0.21 / 0.35 m, against -0.059 / -0.064 m on the tide
  set. The Baltic, absent from the tide set, is -0.04 m at RP10 and -0.17 m at RP100 (42 sites,
  MAE 0.49 m); the Mediterranean MAE at RP100 falls from 0.56 m (15 sites) to 0.11 m (29 sites)
  once its microtidal gauges are in. Figure `sets_extremes.png`, data `sets_extremes.csv`.
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
- **Literature context.** A surge deficit of this order is expected and documented. ERA5 resolves
  storm wind maxima incompletely (Dullaart et al., 2020; Hodges et al., 2017); Muis et al. (2020)
  report a mean bias of -0.04 m in GTSM-ERA5 annual maxima against gauges, and Dullaart et al.
  (2021) find COAST-RP's TC component low against IBTrACS-forced runs at a third of TC-prone
  stations. Observation-based spatial models go further: Morim et al. (2025) find that their
  Bayesian skew-surge reanalysis (BAYEX; Calafat and Marcos, 2020) exceeds COAST-RP RP100 on 76%
  of the US coastline by 50% on average, and site-by-site CoDEC estimates on 87% by 65%, with the
  largest differences on the TC-exposed Gulf and East coasts. The -0.24 m found here at
  TC-dominated gauges is therefore a lower bound on the TC deficit, since 30-40-year gauge
  records under-sample cyclones too. Fathom's RFA sits on the other side: it is biased high
  against single-site gauge fits by +2 cm at RP2 and +21 cm at RP100 by construction (Collings
  et al., 2024, Sect. 4.3), because pooling spreads the rarest event of a region to every site.
- **Why the comparison should move to skew surge.** Comparing total-water-level extremes mixes a
  tide bias with a surge bias, which is why the correlation argument above was needed. The
  skew-surge joint probability method (Pugh and Vassie, 1980; Batstone et al., 2013) separates
  them: compare the tidal-peak distribution (FES / GTSM vs gauge) and the skew-surge
  distribution (GTSM vs gauge) directly, then convolve. This is the method Alexandre and
  Chaudhuri (2025) apply globally and Morim et al. (2025) use for the US. Skew surge and tidal
  peak are near-independent at most sites (Williams et al., 2016), but not everywhere: Arns et
  al. (2020) find significant dependence for the skew surges that produce the highest 1% of total
  water levels, Santamaria-Aguilar and Vafeidis (2018) in mixed semidiurnal regimes, and
  Alexandre and Chaudhuri (2025) reject independence at 17% of their GTSM coastal points. A tidal
  covariate on the skew-surge tail (D'Arcy et al., 2023) or at least a per-site flag is needed
  on macrotidal shelves.

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
- **Extremes set (test 1).** With the surge-dominated coasts in, the microtidal (< 1 m range)
  median contiguous time above MSL rises from 20 h to 45 h (80% of events > 12 h). The Baltic
  median is 122 h and the Mediterranean 90 h; both run into the ±72 h event window, so the true
  durations are longer. T(0.25) hardly changes (14 h), and classes above 1 m are unchanged
  (`sets_durations.png`, `sets_durations.csv`). The top-event comparison of §5.2 gives the same
  ranking on the extremes set (`sets_top_events.csv`): GTSM closest to the gauge, COAST-HG's
  nb24 and T(0.25) too low.

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
- **Literature context.** Dullaart et al. (2023) tested HGRAPHER's sensitivity to two
  assumptions only (normal vs spring tide, and coincidence of the surge and tide maxima); the
  shapes were not scored against observed events, so this is the first quantitative test of
  COAST-HG against gauges. The methods HGRAPHER was built from treat the surge differently: the
  UK design hydrographs (Environment Agency, 2018) are skew-surge hydrographs, built from the
  duration of each percentile level across the largest events, which avoids the tide-surge
  interaction oscillations that a non-tidal residual carries; Wahl et al. (2011) and MacPherson
  et al. (2019) fit parametric surge shapes stochastically for the German North Sea and Baltic
  coasts; Xu and Huang (2014) give an empirical TC surge hydrograph for the Gulf of Mexico. The
  nb24 deficit found here points at either a surge that decays too fast or a tide window that is
  not the spring tide of the gauge; which one it is decides the generator design (§8.3).

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
- The α and clip conclusions come from 1-D profiles, which have no lateral storage; in deltas it
  is storage that converts duration into extent. They are confirmed on real 2-D tiles, including
  a surge-dominated microtidal one, before production (§8.1, test 8). The static
  friction-attenuation engine of Chaudhuri et al. (2027) is the closest published relative of the
  eikonal solver; it has no duration term at all, which is the main thing this correction adds.

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
- **Geoid consistency.** CNES-CLS22 is referenced to GOCO06s; DeltaDTM (Pronk et al., 2024) is
  delivered on EGM2008 and converted to GOCO06s before use (`src/vertical_datum.py`: EGM2008
  truncated to the GOCO06s degree, the offset added to the DEM), so forcing and DEM share one
  geoid. Collings et al. (2024) work on EGM2008 instead.
- **Literature context.** Adding an MDT to a model-MSL level is the treatment used by Collings
  et al. (2024) (CNES-CLS18 to EGM2008) and Wing et al. (2024). Chaudhuri et al. (2027) skip it,
  arguing that EGM2008 approximates MSL globally; the MDT reaches ±1 m (western Pacific, western
  boundary currents, Southern Ocean), so that assumption is a regional bias of that size.
  Coastal MDT is itself extrapolated from altimetry over the last 10-20 km (Mulet et al., 2021,
  quote errors up to 0.1 m), and the absolute datum step is the one part of the chain not yet
  tested here. GNSS stations co-located with gauges (SONEL; Woodworth et al., 2012) give the
  ellipsoidal height needed to test it.

## 8. What to test, decide and refine first

Ordered by how much the result could change the production boundary levels, cheapest first
within each tier. Items 1-4 use only data already extracted for this note; none needs HPC.

### 8.1 Tests

1. **Done** (§2, §4, §5; `site_sets.py`). **Split the site sets.** Keep the harmonic-quality set (698) for §3. Define a second set for
   §4-§5 on record completeness and length only (>= 20 complete years, GESLA flag "no obvious
   issues", no datum jumps), so that surge-dominated microtidal coasts are represented. Redraw
   the region boxes at the same time ("Baltic" → Baltic proper plus a separate "Norway / Nordic
   Seas"; split "Mediterranean" from Biscay). Re-run §4 and §5 on the new set and report both.
   GESLA's quality flag is already "no obvious issues" at all 1,154 sites; datum jumps are not
   screened.
2. **Match the estimator; merged with test 5.** COAST-RP is not a POT / GPD fit (§4), so its
   estimator cannot be matched at a gauge directly. Instead one skew-surge convolution estimator
   is applied three ways at the gauge stations: gauge skew surge + gauge tide; GTSM skew surge +
   GTSM tide (should reproduce COAST-RP's ETC levels, which checks the estimator); GTSM skew surge
   + FES tide. Gauge vs GTSM-GTSM isolates the surge error, GTSM-GTSM vs GTSM-FES the tide error.
   Report median, MAE and the 10-90% range per region and station class; drop the "fraction
   inside the 90% interval" statistic.
3. **Done** (§2; `levels_extremes.py --msl linear`, `msl_sensitivity.py`). **Detrending sensitivity.** Repeat the extremes comparison with a linear trend removed instead
   of annual means. The difference per region is the size of the interannual-MSL term that the
   production level does not contain; largest expected in the Baltic and NE Pacific. Result:
   about 1 cm at RP10, within the fit noise at RP100.
4. **Close the absolute-datum test.** For gauges with a SONEL co-located GNSS station and a known
   gauge datum: GNSS ellipsoidal height - geoid height - gauge MSL should equal the MDT at the
   station to ~0.1 m. The DEM-geoid part is settled: DeltaDTM is converted to GOCO06s (§7).
5. **Tide-corrected level by skew-surge convolution** (replaces the earlier "recombine like
   COAST-RP" idea; at the gauges it is the third combination of test 2). Per GTSM station: skew surge from the GTSM-ERA5 series (1979-2024; or
   1950-2024 from GTSM-ERA5-E, Muis et al., 2025), GPD on its tail with a tidal covariate where
   the independence test fails, convolved with the FES2022 tidal-peak distribution over a nodal
   cycle (Batstone et al., 2013). Compare against the gauge convolution from test 2. This
   exposes the surge deficit cleanly, and the peak-level input of the hydrograph generator (§8.3)
   can take it without code change. Collings et al. (2024) did the same superposition (GTSM
   surge on FES2014 tide) before their RFA, which is a useful precedent.
6. **FES basin-head rule.** Cross-check the flagged nodes (§3) against FES2014, TPXO10-atlas and
   EOT20 (catalog entry `eot20_ocean_tide`; EOT20 is a residual model on FES2014b, so where
   altimetry is sparse it largely tests FES2014 too). If FES2014 lacks the artefact it is an FES2022 extrapolation issue and a version mask
   suffices; if all atlases fail, bound MHWS by the GTSM value where a station exists (GTSM is low
   in Fundy, so conservative in the right direction). Automate: flag any node whose MHWS exceeds
   the maximum of the second atlas within 25 km by more than 50%, then review by hand.
7. **Decompose COAST-HG's error.** Separate COAST-HG into its tide and surge parts and score each
   against the gauge events: is the nb24 deficit a surge that decays too fast, or a tide window
   that is not the gauge's spring tide (daily inequality, neap-spring phase, average-tide
   variant)? The answer fixes the generator design before it is built (§8.3).
8. **Duration correction on real tiles.** Confirm "α = 1, no lower clip" on 5-10 real 2-D tiles
   spanning tidal regimes, including a surge-dominated microtidal case, before adopting it.
9. **Surge-deficit correction.** Once tests 2 and 5 are in, test whether the deficit outside TC
   regions (~0.1 m) is regional and stable enough for a gauge-calibrated correction: either
   regional pooling of gauge-minus-model skew-surge bias (as Collings et al., 2024, correct their
   RP1 bias), or the bias correction of Alexandre and Chaudhuri (2025). Leave-one-out across
   gauges is the acceptance test.
10. **TC hydrographs and TC tails.** Replace the Gaussian placeholder with surge shapes from the
    STORM-forced GTSM runs behind COAST-RP's TC component (VU / Deltares), which also lets the TC
    deficit be attributed to the wind model or the surge model. Gauge-based surge-only scaling
    stays the option where a gauge is within ~50 km.

### 8.2 Decisions

- **Wave setup.** The pipeline has none; Fathom adds 20% of ERA5 Hs on coastline points that are
  exposed over a 50 km fetch (Collings et al., 2024; after Dean and Walton, 2010; Vousdoukas et
  al., 2016), and Vousdoukas et al. (2018) and Marcos et al. (2019) show that neglecting waves
  underestimates extreme sea levels on open coasts by up to a factor of 2 along 30% of the
  coastline. Gauges in harbours do not see setup, so the validation here cannot detect its
  absence. Either add a setup term with an explicit exposure mask (sheltered deltas and estuaries,
  where most exposure sits, get none), or state in the product scope that open-coast hazard
  excludes setup. Not deciding is the worst option.
- **Level product.** Three candidates: (i) COAST-RP as is; (ii) the tide-corrected skew-surge
  convolution of test 5 with a gauge-calibrated surge correction (test 9); (iii) the Fathom RFA of
  Collings et al. (2024), available on request for academic non-commercial use. See §9 for why
  (iii) is a benchmark rather than a replacement. The default is (ii), with (i) kept as the
  fallback and (iii) requested as an independent validation layer.
- **Reference epoch and MSL anomaly.** Keep 2020 (§7). Test 3 puts the interannual-MSL term at
  about 1 cm in the RP levels, so no MSL-anomaly term is added.
- **Hydrograph source.** COAST-HG is retired for both roles once a generator variant passes the
  acceptance test in §8.3. Until then, T from GTSM events (not COAST-HG) where no TC dominates.
- **Open-data posture.** If the global product is to be open and reproducible, every level input
  must be regenerable from open sources (GESLA, CDS GTSM, FES, COAST-RP, MDT). A request-only
  third-party level dataset cannot be redistributed with it.

### 8.3 Next step: a per-RP storm-tide hydrograph generator, validated at the gauges

**Goal.** One hydrograph per COAST-RP station and return period that forces SFINCS and sets the T
of the eikonal duration correction, replacing COAST-HG in both roles.

**Why not re-use the method as is.** COAST-HG is HGRAPHER output (Dullaart et al., 2023): a GTSM
surge shape scaled to COAST-RP RP100 and laid on a tide. `src/storm_tide_hydrograph.py` ports the
same method with per-RP shape selection and a spring-tide window, so it may inherit COAST-HG's
errors (§5.2): next-day high water 0.1-0.15 of the peak too low, and T(0.25) 5-6 h against
11-17 h observed. The cause has to be found before the generator is adopted (§8.1, test 7).

**Generator.** Per station and RP, with levels relative to local MSL:
- **Tide:** FES2022 at the station, 10 min, ±72 h around a spring high water (FES matches the
  gauges, §3), after the basin-head filter (test 6).
- **Surge shape:** selected for the RP from GTSM surge events 1979-2024 (HGRAPHER-RP event
  selection), built on the skew surge rather than the non-tidal residual where the tidal range
  exceeds ~1 m (Environment Agency, 2018). The shape class follows the component that dominates
  COAST-RP at that RP: ETC from GTSM; at TC-dominated RPs a STORM-forced GTSM shape (test 10),
  with a Gaussian of half width 8 h (matching the 15 h median surge duration at TC-dominated
  gauges, §5.4) as the placeholder until those runs are available.
- **Peak:** the surge is scaled so the maximum of tide + surge equals the peak level. The peak
  level is an input (COAST-RP for now), so the tide-corrected peak of test 5 can replace it
  without changing the shape code. With an FES tide, the surge amplitude COAST-RP - MHWS_FES
  absorbs GTSM's tide excess until then.
- **Phasing:** surge peak on the spring high water is the conservative default; tide-surge
  interaction shifts the largest surges away from high water on shallow shelves (Horsburgh and
  Wilson, 2007), so the observed phase distribution of the selected GTSM events is the second
  variant.
- MDT, the 2020 offset and SLR are added downstream; the shape is the same for every SLR scenario
  (tides themselves change with SLR, Haigh et al., 2020, which is a known simplification).

**Diagnosis of COAST-HG's error.** Swap one component at a time and score each variant at the
quality sites:

| Component | Variant 1 | Variant 2 |
|---|---|---|
| Tide window | average tide (COAST-HG) | spring-tide window (FES) |
| Surge shape | HGRAPHER level-by-level mean of limb durations | empirical mean of the normalised surge residuals of the selected events, centred on their peaks |
| Surge definition | non-tidal residual | skew surge |
| Surge-tide phasing | surge peak on the spring high water | the phase of the selected GTSM events relative to high water |

**Acceptance test.** Build hydrographs at RP2 and RP10-100 and compare their normalised shapes
with the gauge events of the matching empirical-RP class (< 2 y; >= 10 y), per tidal-range class
and on the extremes site set of test 1:
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

## 9. Comparison with other global coastal boundary conditions

| | This pipeline (GCFM) | Fathom RFA (Collings et al., 2024; Wing et al., 2024) | Chaudhuri et al. (2027) |
|---|---|---|---|
| Still-water extremes | COAST-RP: GTSM-ERA5 POT (ETC) + STORM synthetic TCs | RFA after Sweet et al. (2020, 2022): GPD on pooled, normalised daily maxima; gauges where >= 3 within 400 km (851 of 5,975 1° cells), otherwise GTSM-ERA5 (4,555 cells) or single-site GTSM-ERA5 (569 cells); shape capped at 0.35 | skew-surge return levels from a global SSJPM on GTSM-ERA5, bias-corrected at gauges (Alexandre and Chaudhuri, 2025) |
| Tide | GTSM tide inside COAST-RP (7-8% high); FES2022 proposed | FES2014: MHHW/MLLW scale the index flood; GTSM surge laid on FES2014 tide | HAT from FES2022b added to the skew-surge RP level |
| Vertical reference | model MSL + CNES-CLS22 MDT (GOCO06s) + 2020 offset | 2011 epoch MSL + CNES-CLS18 MDT to EGM2008; RP1 bias-corrected to single-site fits (mean 8 cm) | none: EGM2008 taken as MSL |
| Waves | none | 20% ERA5 Hs on exposed coasts (50 km fetch test) | none in the published description |
| Time dimension | hydrograph → T → friction multiplier (eikonal); full hydrograph (SFINCS) | none for return-period maps | none (static friction attenuation on DeltaDTM) |
| Resolution | COAST-RP stations (23,226) | ~1 km coastline points (3.4 million) | 0.1° ESL points joined to FES vertices within 20 km |
| Validation reported | this note: COAST-RP -0.06 m median at RP10/RP100 vs gauge GEV (698 sites) | +2 cm (RP2) to +21 cm (RP100) mean bias vs single-site gauge GPD; RP10 5-95% -8..+27 cm | not in the excerpt reviewed |
| Access | open inputs | output on request, academic non-commercial only | seed database and outputs proprietary (Geosapiens) |

**What the Fathom RFA does better.** It uses gauges directly where they are dense, so it has no
model tide bias there; pooling stabilises the tail and spreads rare TC events along a coast (the
Cyclone Yasi example); it carries wave setup; and it is delivered at 1 km. **What it does not
fix.** 85% of its coastal cells rest on GTSM-ERA5, with the same ERA5 surge deficit found here
and without synthetic TCs, so its TC tails are limited by the reanalysis (the authors name the Bay
of Bengal and NW India as under-represented); the index flood scaled by tidal range is a weak
model for surge-dominated microtidal coasts; a fixed shape-parameter cap of 0.35 was applied to
34 cells, mostly Gulf of Mexico and Japan; and the RFA inflates levels against single-site fits
by construction (+21 cm at RP100), which the authors treat as a feature and Morim et al. (2025)
treat as still too low on the US Gulf and East coasts. Its tide model is FES2014, which §3 shows
FES2022 improves on at the gauges. For this pipeline it is an independent benchmark, not a
replacement (§8.2).

**What Chaudhuri et al. do differently, and why the seed level is not an RP100 level.** The seed
is the RP100 skew surge plus HAT. Skew surge and tidal peak are close to independent, so their
sum has a return period far beyond 100 years: HAT is reached a few times per nodal cycle, and the
joint event is rarer still. Morim et al. (2025) compute the same alternative and describe the HAT
variant as very conservative, exceeding the convolution everywhere and most on macrotidal coasts;
the excess is roughly HAT - MHWS plus the interaction term, i.e. 0.3-0.5 m in NW Europe and ~1 m
in Fundy. The irony is that Alexandre and Chaudhuri (2025) built a global SSJPM product precisely
to avoid this, and the engine then discards the convolution. "100-year inundation envelope" is
the accurate description; it is not comparable to an RP100 map. Two further differences: no MDT
(§7), and no duration (§6). The friction-attenuation engine itself is the closest published
relative of the eikonal solver, so the comparison belongs in the model paper, with the T-based
multiplier as the differentiator.

**Scale of the biases.** DeltaDTM's vertical MAE is 0.45 m (Pronk et al., 2024). Most biases
quantified here (0.05-0.25 m) are below the DEM error at a point, but they are sign-consistent
over whole regions, whereas DEM error is largely random; that is the argument for correcting
them. The FES basin-head failures (several metres over tens of kilometres) are the exception and
must be filtered regardless.

## 10. Reproducing

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
10. `site_sets.py`
11. `levels_extremes.py --msl linear`, then `msl_sensitivity.py`
12. `duration_experiment.py` (local SFINCS)
13. `duration_analysis.py`
14. `duration_refit.py`
15. `export_input_tables.py`

Inputs on `<data_root>` are read only.

## 11. Data files

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

## 12. References

Alexandre, D. A. and Chaudhuri, C. (2025). A multistep statistical model to derive extreme sea
levels for the global coastlines. Coastal Engineering, 204, 104909.
https://doi.org/10.1016/j.coastaleng.2025.104909

Archer, A. W. (2013). World's highest tides: Hypertidal coastal systems in North America, South
America and Europe. Sedimentary Geology, 284-285, 1-25.
https://doi.org/10.1016/j.sedgeo.2012.12.007

Arns, A., Wahl, T., Wolff, C., Vafeidis, A. T., Haigh, I. D., Woodworth, P., Niehüser, S. and
Jensen, J. (2020). Non-linear interaction modulates global extreme sea levels, coastal flood
exposure, and impacts. Nature Communications, 11, 1918.
https://doi.org/10.1038/s41467-020-15752-5

AVISO+ (2024). FES2022 global ocean tide atlas (product description and data).
https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html

AVISO+ (2023). Mean dynamic topography CNES-CLS22 (product description and data).
https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mdt.html

Batstone, C., Lawless, M., Tawn, J., Horsburgh, K., Blackman, D., McMillan, A., Worth, D.,
Laeger, S. and Hunt, T. (2013). A UK best-practice approach for extreme sea-level analysis along
complex topographic coastlines. Ocean Engineering, 71, 28-39.
https://doi.org/10.1016/j.oceaneng.2013.02.003

Bloemendaal, N., Haigh, I. D., de Moel, H., Muis, S., Haarsma, R. J. and Aerts, J. C. J. H.
(2020). Generation of a global synthetic tropical cyclone hazard dataset using STORM. Scientific
Data, 7, 40. https://doi.org/10.1038/s41597-020-0381-2

Calafat, F. M. and Marcos, M. (2020). Probabilistic reanalysis of storm surge extremes in
Europe. Proceedings of the National Academy of Sciences, 117, 1877-1883.
https://doi.org/10.1073/pnas.1913049117

Chaudhuri, C., Alexandre, D. A. and Gill-Fortin, J. (2027). A global coastal flood depth model
using friction-attenuated water surface elevation propagation on high-resolution terrain.
Coastal Engineering, 213, 105164.
https://www.sciencedirect.com/science/article/pii/S0378383926002188

Collings, T. P., Quinn, N. D., Haigh, I. D., Green, J., Probyn, I., Wilkinson, H., Muis, S.,
Sweet, W. V. and Bates, P. D. (2024). Global application of a regional frequency analysis to
extreme sea levels. Natural Hazards and Earth System Sciences, 24, 2403-2423.
https://doi.org/10.5194/nhess-24-2403-2024

D'Arcy, E., Tawn, J. A., Joly, A. and Sifnioti, D. E. (2023). Accounting for seasonality in
extreme sea-level estimation. Annals of Applied Statistics, 17(4). Preprint: arXiv:2207.09870.

Dean, R. G. and Walton, T. L. (2010). Wave setup. In: Kim, Y. C. (ed.), Handbook of Coastal and
Ocean Engineering, World Scientific, 1-24. https://doi.org/10.1142/10353

Desplanque, C. and Mossman, D. J. (2001). Bay of Fundy tides. Geoscience Canada, 28(1), 1-11.

Dullaart, J. C. M., Muis, S., Bloemendaal, N. and Aerts, J. C. J. H. (2020). Advancing global
storm surge modelling using the new ERA5 climate reanalysis. Climate Dynamics, 54, 1007-1021.
https://doi.org/10.1007/s00382-019-05044-0

Dullaart, J. C. M., Muis, S., Bloemendaal, N., Chertova, M. V., Couasnon, A. and Aerts,
J. C. J. H. (2021). Accounting for tropical cyclones more than doubles the global population
exposed to low-probability coastal flooding. Communications Earth & Environment, 2, 135.
https://doi.org/10.1038/s43247-021-00204-9

Dullaart, J., Muis, S., Bloemendaal, N., Chertova, M., Couasnon, A. and Aerts, J. C. J. H.
(2022). COAST-RP: A global COastal dAtaset of Storm Tide Return Periods (Version 2).
4TU.ResearchData. https://doi.org/10.4121/13392314.v2

Dullaart, J. C. M., Muis, S., de Moel, H., Ward, P. J., Eilander, D. and Aerts, J. C. J. H.
(2023). Enabling dynamic modelling of coastal flooding by defining storm tide hydrographs.
Natural Hazards and Earth System Sciences, 23, 1847-1862.
https://doi.org/10.5194/nhess-23-1847-2023

Dullaart, J., Muis, S., Eilander, D., Aerts, J. C. J. H., de Moel, H. and Ward, P. J. (2023).
COAST-HG: A global COastal dAtaset of Storm Tide HydroGraphs (Version 2). 4TU.ResearchData.
https://doi.org/10.4121/21270948.v2

Environment Agency (2018). Coastal flood boundary conditions for the UK: 2018 update. Technical
report, 116 pp. https://www.gov.uk/government/publications/coastal-flood-boundary-conditions-for-uk-mainland-and-islands-design-sea-levels

Fox-Kemper, B., Hewitt, H. T., Xiao, C., et al. (2021). Ocean, cryosphere and sea level change.
In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth
Assessment Report of the IPCC, Chapter 9. Cambridge University Press.
https://doi.org/10.1017/9781009157896.011

Garner, G. G., Hermans, T., Kopp, R. E., et al. (2021). IPCC AR6 sea level projections (Version
20210809). Zenodo. https://doi.org/10.5281/zenodo.5914709

Haigh, I. D., Pickering, M. D., Green, J. A. M., Arbic, B. K., Arns, A., Dangendorf, S., et al.
(2020). The tides they are a-changin': A comprehensive review of past and future nonastronomical
changes in tides, their driving mechanisms, and future implications. Reviews of Geophysics, 58,
e2018RG000636. https://doi.org/10.1029/2018RG000636

Haigh, I. D., Marcos, M., Talke, S. A., Woodworth, P. L., Hunter, J. R., Hague, B. S., Arns,
A., Bradshaw, E. and Thompson, P. (2023). GESLA Version 3: A major update to the global
higher-frequency sea-level dataset. Geoscience Data Journal, 10, 293-314.
https://doi.org/10.1002/gdj3.174

Hodges, K., Cobb, A. and Vidale, P. L. (2017). How well are tropical cyclones represented in
reanalysis datasets? Journal of Climate, 30, 5243-5264.
https://doi.org/10.1175/JCLI-D-16-0557.1

Horsburgh, K. J. and Wilson, C. (2007). Tide-surge interaction and its role in the distribution
of surge residuals in the North Sea. Journal of Geophysical Research, 112, C08003.
https://doi.org/10.1029/2006JC004033

Idier, D., Bertin, X., Thompson, P. and Pickering, M. D. (2019). Interactions between mean sea
level, tide, surge, waves and flooding: Mechanisms and contributions to sea level variations at
the coast. Surveys in Geophysics, 40, 1603-1630. https://doi.org/10.1007/s10712-019-09549-5

Lyard, F. H., Allain, D. J., Cancet, M., Carrère, L. and Picot, N. (2021). FES2014 global ocean
tide atlas: design and performance. Ocean Science, 17, 615-649.
https://doi.org/10.5194/os-17-615-2021

MacPherson, L. R., Arns, A., Dangendorf, S., Vafeidis, A. T. and Jensen, J. (2019). A
stochastic extreme sea level model for the German Baltic Sea coast. Journal of Geophysical
Research: Oceans, 124, 2054-2071. https://doi.org/10.1029/2018JC014718

Marcos, M., Rohmer, J., Vousdoukas, M. I., Mentaschi, L., Le Cozannet, G. and Amores, A.
(2019). Increased extreme coastal water levels due to the combined action of storm surges and
wind waves. Geophysical Research Letters, 46, 4356-4364. https://doi.org/10.1029/2019GL082599

Medeiros, C. and Kjerfve, B. (1988). Tidal characteristics of the Strait of Magellan.
Continental Shelf Research, 8, 947-960. https://doi.org/10.1016/0278-4343(88)90079-1

Menéndez, M. and Woodworth, P. L. (2010). Changes in extreme high water levels based on a
quasi-global tide-gauge data set. Journal of Geophysical Research, 115, C10011.
https://doi.org/10.1029/2009JC005997

Morim, J., Rasmussen, D. J., Wahl, T., Calafat, F. M., Kopp, R. E., Oppenheimer, M. and
Dangendorf, S. (2025). US-CoastEX: Observation-based probabilistic reanalysis of storm surge and
sea level extremes for the United States. Scientific Data, 12, 1395.
https://doi.org/10.1038/s41597-025-05730-1

Muis, S., Apecechea, M. I., Dullaart, J., de Lima Rego, J., Madsen, K. S., Su, J., Yan, K. and
Verlaan, M. (2020). A high-resolution global dataset of extreme sea levels, tides, and storm
surges, including future projections. Frontiers in Marine Science, 7, 263.
https://doi.org/10.3389/fmars.2020.00263

Muis, S., et al. (2025). Global dataset of storm surges and extreme sea levels for 1950-2024
based on the ERA5 climate reanalysis (GTSM-ERA5-E). Earth System Science Data Discussions,
preprint (discussion started 3 September 2025).

Mulet, S., Rio, M.-H., Etienne, H., Artana, C., Cancet, M., Dibarboure, G., Feng, H., Husson,
R., Picot, N., Provost, C. and Strub, P. T. (2021). The new CNES-CLS18 global mean dynamic
topography. Ocean Science, 17, 789-808. https://doi.org/10.5194/os-17-789-2021

Pronk, M., Hooijer, A., Eilander, D., Haag, A., de Jong, T., Vousdoukas, M., Vernimmen, R.,
Ledoux, H. and Eleveld, M. (2024). DeltaDTM: A global coastal digital terrain model. Scientific
Data, 11, 273. https://doi.org/10.1038/s41597-024-03091-9

Pugh, D. T. and Vassie, J. M. (1980). Applications of the joint probability method for extreme
sea level computations. Proceedings of the Institution of Civil Engineers, 69(4), 959-975.
https://doi.org/10.1680/iicep.1980.2179

Pugh, D. and Woodworth, P. (2014). Sea-Level Science: Understanding Tides, Surges, Tsunamis and
Mean Sea-Level Changes. Cambridge University Press. https://doi.org/10.1017/CBO9781139235778

Santamaria-Aguilar, S. and Vafeidis, A. T. (2018). Are extreme skew surges independent of high
water levels in a mixed semidiurnal tidal regime? Journal of Geophysical Research: Oceans, 123.
https://doi.org/10.1029/2018JC014282

SONEL (Système d'Observation du Niveau des Eaux Littorales). GNSS at tide gauges and ellipsoidal
heights. https://www.sonel.org

Stammer, D., Ray, R. D., Andersen, O. B., et al. (2014). Accuracy assessment of global
barotropic ocean tide models. Reviews of Geophysics, 52, 243-282.
https://doi.org/10.1002/2014RG000450

Sweet, W. V., Genz, A. S., Obeysekera, J. and Marra, J. J. (2020). A regional frequency analysis
of tide gauges to assess Pacific coast flood risk. Frontiers in Marine Science, 7, 581769.
https://doi.org/10.3389/fmars.2020.581769

Sweet, W. V., Hamlington, B. D., Kopp, R. E., et al. (2022). Global and regional sea level rise
scenarios for the United States: Updated mean projections and extreme water level probabilities
along U.S. coastlines. NOAA Technical Report NOS 01, 111 pp.

Vousdoukas, M. I., Voukouvalas, E., Mentaschi, L., Dottori, F., Giardino, A., Bouziotas, D.,
Bianchi, A., Salamon, P. and Feyen, L. (2016). Developments in large-scale coastal flood hazard
mapping. Natural Hazards and Earth System Sciences, 16, 1841-1853.
https://doi.org/10.5194/nhess-16-1841-2016

Vousdoukas, M. I., Mentaschi, L., Voukouvalas, E., Verlaan, M. and Feyen, L. (2017). Extreme
sea levels on the rise along Europe's coasts. Earth's Future, 5, 304-323.
https://doi.org/10.1002/2016EF000505

Vousdoukas, M. I., Mentaschi, L., Voukouvalas, E., Verlaan, M., Jevrejeva, S., Jackson, L. P.
and Feyen, L. (2018). Global probabilistic projections of extreme sea levels show
intensification of coastal flood hazard. Nature Communications, 9, 2360.
https://doi.org/10.1038/s41467-018-04692-w

Wahl, T., Mudersbach, C. and Jensen, J. (2011). Assessing the hydrodynamic boundary conditions
for risk analyses in coastal areas: a stochastic storm surge model. Natural Hazards and Earth
System Sciences, 11, 2925-2939. https://doi.org/10.5194/nhess-11-2925-2011

Wang, X., Verlaan, M., Veenstra, J. and Lin, H. X. (2022). Data-assimilation-based parameter
estimation of bathymetry and bottom friction coefficient to improve coastal accuracy in a global
tide model. Ocean Science, 18, 881-904. https://doi.org/10.5194/os-18-881-2022

Williams, J., Horsburgh, K. J., Williams, J. A. and Proctor, R. N. F. (2016). Tide and skew
surge independence: New insights for flood risk. Geophysical Research Letters, 43, 6410-6417.
https://doi.org/10.1002/2016GL069522

Wing, O. E. J., Bates, P. D., Quinn, N. D., Savage, J. T. S., Uhe, P. F., Cooper, A., et al.
(2024). A 30 m global flood inundation model for any climate scenario. Water Resources Research,
60, e2023WR036460. https://doi.org/10.1029/2023WR036460

Woodworth, P. L., Hughes, C. W., Bingham, R. J. and Gruber, T. (2012). Towards worldwide height
system unification using ocean information. Journal of Geodetic Science, 2(4), 302-318.
https://doi.org/10.2478/v10156-012-0004-8

Xu, S. and Huang, W. (2014). An improved empirical equation for storm surge hydrographs in the
Gulf of Mexico, USA. Ocean Engineering, 75, 174-179.
https://doi.org/10.1016/j.oceaneng.2013.11.004
