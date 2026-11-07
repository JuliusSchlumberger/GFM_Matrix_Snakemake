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
- **Figures:** `.cleanup_baseline/level_sources/analysis/` (local, not in git); a selection with a
  summary of the findings is in `input_analysis/boundary_forcing/`.
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
- **Regions:** the IPCC AR6 reference regions (Iturbide et al., 2020; `regions.ar6_regions`). A
  site takes the nearest AR6 land region within 200 km, otherwise the AR6 ocean region that
  contains it (remote islands, e.g. Hawaii in the North Pacific Ocean region). The Baltic and
  Norway are part of N.Europe (NEU); the Atlantic coasts of France and Iberia of
  West&Central-Europe (WCE).

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
  the fit noise, -0.03..+0.03 m). The largest regional RP10 effects are C.Australia (+0.044 m,
  5 sites), the Equatorial Pacific (+0.026 m) and N.W.South-America (+0.024 m). The seasonal cycle stays in both fits.
  The term is an order of magnitude below the surge deficit, so the production level needs no
  MSL-anomaly term (`msl_sensitivity.png`).

## 3. Tidal levels

Figures: summary fig01 (GTSM vs gauge, FES vs gauge, GTSM vs FES; MHWS), fig02 (the same ratios as
maps), fig03 (constituents). Data: `sites_levels.csv`.

| Comparison (MHWS) | Median ratio | MAE | Within 0.1 m |
|---|---|---|---|
| GTSM vs gauge | 1.08 | 0.16 m | 53% |
| FES2022 at the gauge vs gauge | 0.99 | 0.05 m | 88% |
| GTSM vs FES2022 at the GTSM station | 1.08 | 0.16 m | 52% |
| FES2022 at the gauge vs at the GTSM station | 1.00 | 0.01 m | 98% |

Median MHWS ratio per AR6 region with >= 5 quality sites (the colour scale of the maps; the 15
smaller regions and a stratified sample of sites are in `input_analysis_data/tables.md`):

| region | n | gtsm_gesla | fes_gesla | gtsm_fes | spring_range_m |
|---|---|---|---|---|---|
| C.Australia (CAU) | 5 | 1.22 | 1.00 | 1.23 | 0.95 |
| C.North-America (CNA) | 8 | 1.16 | 1.16 | 1.01 | 0.49 |
| Caribbean (CAR) | 15 | 0.96 | 1.01 | 0.95 | 0.32 |
| E.Asia (EAS) | 108 | 1.07 | 0.99 | 1.08 | 1.28 |
| E.Australia (EAU) | 24 | 1.03 | 0.98 | 1.05 | 1.38 |
| E.North-America (ENA) | 76 | 0.79 | 1.00 | 0.80 | 1.37 |
| E.Southern-Africa (ESAF) | 6 | 1.22 | 0.99 | 1.23 | 1.74 |
| Equatorial.Indian-Ocean (EIO) | 6 | 1.12 | 1.00 | 1.14 | 0.88 |
| Equatorial.Pacific-Ocean (EPO) | 22 | 1.08 | 1.00 | 1.07 | 1.10 |
| Mediterranean (MED) | 27 | 1.17 | 1.00 | 1.19 | 1.20 |
| N.Atlantic-Ocean (NAO) | 5 | 1.10 | 0.99 | 1.09 | 1.10 |
| N.Australia (NAU) | 14 | 0.98 | 1.01 | 0.97 | 2.13 |
| N.Central-America (NCA) | 14 | 1.04 | 0.99 | 1.02 | 0.96 |
| N.E.North-America (NEN) | 5 | 0.91 | 1.00 | 0.78 | 2.15 |
| N.Europe (NEU) | 78 | 1.08 | 0.99 | 1.10 | 3.37 |
| N.Pacific-Ocean (NPO) | 24 | 0.99 | 0.99 | 0.99 | 0.60 |
| N.W.North-America (NWN) | 20 | 1.07 | 0.99 | 1.08 | 3.42 |
| N.W.South-America (NWS) | 12 | 1.11 | 1.00 | 1.10 | 1.55 |
| New-Zealand (NZ) | 16 | 1.07 | 0.99 | 1.09 | 1.89 |
| S.Australia (SAU) | 30 | 1.11 | 1.00 | 1.14 | 0.70 |
| S.Central-America (SCA) | 7 | 1.10 | 1.01 | 1.11 | 1.20 |
| S.E.Asia (SEA) | 32 | 1.14 | 1.01 | 1.10 | 1.60 |
| S.Indian-Ocean (SIO) | 5 | 1.37 | 1.00 | 1.38 | 0.57 |
| S.Pacific-Ocean (SPO) | 20 | 1.07 | 0.98 | 1.20 | 1.03 |
| S.W.South-America (SWS) | 9 | 1.21 | 0.99 | 1.20 | 0.94 |
| Sahara (SAH) | 8 | 1.19 | 1.00 | 1.19 | 2.10 |
| W.North-America (WNA) | 36 | 0.99 | 1.01 | 0.99 | 1.92 |
| W.Southern-Africa (WSAF) | 11 | 1.26 | 1.00 | 1.26 | 1.56 |
| West&Central-Europe (WCE) | 23 | 1.12 | 1.00 | 1.11 | 5.20 |

- **GTSM's tide is 7-8% too large; FES2022 matches the gauges.** GTSM is too large almost
  everywhere and too small on the east coast of North America (ENA 0.79). The GTSM/FES map reproduces
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
  seed to the nearest FES2022b HAT vertex within 20 km. FES2014 does not share the artefact
  (§8.1, test 6): at the same nodes its proxy is 3.6-6.0 m, as EOT20's, so the failure is
  FES2022's extrapolation into the basin heads. FES2014 is therefore the fallback at such nodes
  (§8.2, tide atlas).

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
  set. The regions that gain the surge-dominated gauges change most: the Mediterranean (MED)
  RP100 MAE falls from 0.41 m (26 sites) to 0.14 m (39 sites) once its microtidal gauges are in,
  and N.Europe (NEU, with the Baltic) moves from +0.12 to +0.04 m at RP100 (73 to 101 sites).
  By region: summary fig07 (`input_analysis/boundary_forcing/`), data `sets_extremes.csv`.
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

Figures: `shapes_profiles.png`, summary fig08 (profiles by RP), `shapes_durations.png`.
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
  median contiguous time above MSL rises from 20 h to 45 h (80% of events > 12 h). In
  N.Europe (with the Baltic) and the Mediterranean the median rises from 8 to 23 h and from 7 to
  21.5 h, with upper quartiles of 107 and 125 h that run into the ±72 h event window, so the true
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
4. **Deferred.** **Close the absolute-datum test.** For gauges with a known tie between the gauge
   datum and the ellipsoid: ellipsoidal height of gauge MSL - geoid height should equal the MDT at
   the station to ~0.1 m. The DEM-geoid part is settled: DeltaDTM is converted to GOCO06s (§7).
   Two routes:
   - SONEL GNSS at gauges: needs the ULR ellipsoidal height and the antenna-to-gauge-zero tie,
     and a GESLA series on that same zero; the usable sample is small and assembled by hand.
   - National datums: NOAA's datum API gives MSL (1983-2001 epoch) relative to NAVD88 at the US
     gauges; NAVD88 → ellipsoid with NGS GEOID18, then → GOCO06s, correcting the 2-3 cm of
     sea-level rise between the NOAA and MDT epochs. ODN, NAP and NHN follow the same way with
     their national geoid models.
5. **Tide-corrected level by skew-surge convolution** (replaces the earlier "recombine like
   COAST-RP" idea; at the gauges it is the third combination of test 2). Per GTSM station: skew surge from the GTSM-ERA5 series (1979-2024; or
   1950-2024 from GTSM-ERA5-E, Muis et al., 2025), GPD on its tail with a tidal covariate where
   the independence test fails, convolved with the FES2022 tidal-peak distribution over a nodal
   cycle (Batstone et al., 2013). Compare against the gauge convolution from test 2. This
   exposes the surge deficit cleanly, and the peak-level input of the hydrograph generator (§8.3)
   can take it without code change. Collings et al. (2024) did the same superposition (GTSM
   surge on FES2014 tide) before their RFA, which is a useful precedent.
6. **FES basin-head rule.** Cross-check the flagged nodes (§3) against FES2014, TPXO10-atlas and
   EOT20 (a residual model on FES2014b, so where
   altimetry is sparse it largely tests FES2014 too). If FES2014 lacks the artefact it is an FES2022 extrapolation issue and a version mask
   suffices; if all atlases fail, bound MHWS by the GTSM value where a station exists (GTSM is low
   in Fundy, so conservative in the right direction). Automate: flag any node whose MHWS exceeds
   the maximum of the second atlas within 25 km by more than 50%, then review by hand.

   **EOT20 and FES2014 done** (`atlas_check.py`, which keeps the FES2014 part; the EOT20 screen is
   in its git history up to commit 35f697a; summary fig04-fig06; TPXO10 not obtained). FES2014 is the AVISO
   `fes2014b_elevations_extrapolated` product (34 constituents, 1/16 deg), read like FES2022.
   - *At the gauges* (tide set, same 18.61-y prediction and statistics as §3): EOT20 MHWS median
     ratio 1.00, MAE 0.113 m, 84% within 0.1 m, against FES 0.99 / 0.053 m / 88%. EOT20 is not
     defined at 13 gauges (Juneau, Delaware City, the Fraser and Vancouver harbours, among others).
     Its error is dominated by solar-constituent artefacts in the St Lawrence estuary: at Lauzon,
     Saint-François and Saint-Joseph-de-la-Rive the atlas files hold S2 of 14-31 m, T2 up to 23 m
     and S1 up to 18 m, against an M2 of 1.75 m (MHWS 20-41 m). Without those sites the EOT20 MAE
     is 0.059 m, close to FES. Constituent ratios to the gauges (M2, S2, K1, O1) centre on 1.00
     for both atlases; EOT20's spread is wider.
   - *At the 89,628 FES boundary nodes* of the test tiles, screened with the proxy
     |M2|+|S2|+|K1|+|O1| for both atlases: 3,876 nodes have FES > 1.5 x the largest EOT20 within
     25 km (1,343 at 2x, 446 at 3x), and 1,595 have no EOT20 value within 25 km. The three
     basin-head clusters of §3 are caught: Cobequid Bay (FES proxy 12.4 m vs EOT20 6.1 m, 18 of 26
     nodes), Bahía Lomas (15.7 vs 5.0 m, 44 of 148 nodes). The Colorado-delta node has no EOT20
     value nearby. Further gross clusters at 2x: Glacier Bay / Lynn Canal (FES MHWS up to 12.2 m vs
     EOT20 2.3 m; the Skagway and Juneau gauges show FES 7.5 / 6.0 m against 2.8 / 2.7 m observed),
     Penzhina Bay, Frobisher Bay, the Río Gallegos estuary, central Gulf of California. About half
     of the flagged nodes lie north of 66 N, where EOT20 (altimetry under sea ice) is the weaker
     atlas.
   - *Can a second atlas referee FES?* Not for moderate errors. At the 14 gauges where FES is too
     high by > 0.25 m and > 25%, EOT20 or GTSM is usually wrong in the same direction (Point
     Lonsdale, Queenscliff and Long Island Sound with EOT20; Juneau and Skagway with GTSM). The
     rule FES/EOT20 > 1.5 flags 19 gauges, of which 2 are real FES errors; FES > 1.3 x max(GTSM,
     EOT20) flags 18 and catches 5 of the 14; requiring both references to disagree flags 4 and
     catches none. Most flags are places where EOT20 itself fails (Tacoma 0.12 m vs 1.46 m
     observed, Trondheim, Seattle).
   - *FES2014 at the gauges:* median ratio 0.99, MAE 0.067 m, 84% within 0.1 m (n=695), so
     FES2022 is the better atlas overall (0.053 m), as Lyard et al. (2021) and §3 lead one to
     expect. At the 20 sites where FES2022 misses by > 0.25 m, FES2014 is better at Juneau (3.33
     vs 5.95 m, gauge 2.72 m), Bamfield, Willets Point, Arcachon and Delaware City, and worse at
     Patricia Bay. Its own large miss is Trondheim (0.44 m vs 1.37 m observed), shared with
     EOT20, which is built on FES2014.
   - *FES2014 at the nodes:* both FES versions are extrapolated to the coast, so they are compared
     at the node itself. FES2022 / FES2014 has median 1.00 at the EOT20-ok nodes (p90 1.15; 1.7%
     above 1.5) and 1.84 at the EOT20-flagged nodes (93% above 1.5). In all four basins FES2014
     matches EOT20 and only FES2022 has the hot spots: Cobequid Bay proxy max FES2022 12.4 m,
     FES2014 6.0 m, EOT20 6.1 m; Bahía Lomas 15.7 / 4.9 / 5.0 m; the Colorado-delta node 17.1 /
     3.6 m; Glacier Bay / Lynn Canal 11.7 / 2.3 / 2.3 m (fig05). The artefacts are
     therefore FES2022-specific, which answers the version question of this test. In the Arctic
     FES2014 sides with EOT20 as well, but there the two are not independent.
   - *Version rule, on MHWS* (`atlas_node_mhws.py`, `atlas_rule_sensitivity.py`). FES2022 and
     FES2014 are predicted at all 103,228 boundary nodes (102,935 with both values) over the same
     18.61-y cycle and reduced to MHWS as at the gauges, so the rule works in the quantity that
     forces the model. FES2022 is replaced by FES2014 at a node where MHWS_FES2022 > R x
     MHWS_FES2014 **and** MHWS_FES2022 - MHWS_FES2014 > D, with **R = 1.5 and D = 1.25 m**. The
     ratio alone flags microtidal coasts where both MHWS are centimetres; the difference alone
     would switch macrotidal coasts where FES2022 is only 20-40% higher. At the adopted pair 1,287
     nodes (1.25%) are replaced. At the 905 gauges with all three MHWS the rule fires 3 times, all
     fixes: Skagway (7.46 -> 2.76 m, gauge 2.78 m), Juneau (5.95 -> 3.33 m, gauge 2.72 m) and Port
     Alberni (2.79 -> 1.36 m, gauge 1.28 m).
   - *Threshold sensitivity* (R in 1-3, D in 0-1.5 m; `atlas_rule_grid.csv`,
     `atlas_rule_bands.csv`; summary fig06 in `input_analysis/boundary_forcing/`):
     - *Which atlas is closer to the gauge, by MHWS difference FES2022 - FES2014.* In every band
       below 0.5 m FES2014 is closer at only 31-50% of the gauges and switching gains nothing or
       loses (median gain of |FES2022 - gauge| - |FES2014 - gauge| from -0.34 to +0.02 m); the
       three sites at 0.5-0.75 m all lose (median -0.64 m). Above 1 m FES2014 is closer at 4 of 5 gauges, median
       gain +1.43 m; the fifth is Trondheim, where FES2014 (and EOT20) is 0.9 m too low.
     - *Gauge outcome over the grid.* No pair with D < 1.25 m avoids harm: Trondheim's difference
       is 1.08 m, and below 0.75 m the microtidal sites add more (40 harms at D = 0, R = 1.1).
       From D = 1.25 m no pair harms a gauge, and every R <= 1.75 fires at the same 3 sites. At
       D = 1.5 m Port Alberni (difference 1.43 m) drops out.
     - *Known artefacts,* defined without FES2014: the 178 nodes in the four basin boxes whose
       FES2022 MHWS exceeds 1.25 x the largest observed MHWS there (Minas 8.15 m, Magellan 5.5 m,
       Colorado 5.0 m, Skagway 2.8 m). Every R <= 1.5 replaces all 178 at every D up to 1.5 m;
       R = 1.75 replaces 172, R = 2 151, R = 3 63.
     - *Size of the change.* At D = 1.25 m the number of replaced nodes falls from 1,464 (R = 1.1)
       to 1,287 (R = 1.5) and 1,066 (R = 1.75).

     D is therefore bounded to 1.08-1.43 m by two gauges (Trondheim below, Port Alberni above),
     and 1.25 m sits in the middle. R is not pinned down by the data: everything from R = 1 (the
     ratio condition switched off, D alone) to R = 1.5 gives the same gauge result and catches all
     known artefacts, and R >= 1.75 misses some. Within that range R only changes how many other
     nodes are switched (1,464 at R = 1, 1,287 at R = 1.5, for D = 1.25 m): the 177 nodes in
     between lie on macrotidal coasts where FES2022 is up to 50% and more than 1.25 m higher and
     no gauge decides. R = 1.5 is the cautious choice that keeps FES2022 there. The gauge evidence rests on few sites (5 with a difference above 1 m),
     so the known-artefact set carries the ratio bound.
   - *Decision.* FES2014 is the single reference and fallback atlas; EOT20 is not used further.
     EOT20 is a residual model on FES2014b, so where altimetry is sparse it repeats FES2014, and
     where it departs from it it brings its own failures (St Lawrence S2 artefacts, Puget Sound,
     no values in narrow inlets). FES2022 stays the tide source; the version rule with R = 1.5,
     D = 1.25 m (MHWS) replaces it by FES2014 at basin-head nodes, and a gauge overrides both where
     one exists. Moderate fjord and inlet errors (0.25-1 m) are not caught by any atlas.
7. **Done** (`coast_hg_decompose.py`; data `coast_hg_vs_events.csv`; summary fig09).
   **Decompose COAST-HG's error.** Separate COAST-HG into its tide and surge parts and score each
   against the gauge events: is the nb24 deficit a surge that decays too fast, or a tide window
   that is not the gauge's spring tide? COAST-HG holds the total only; its tide is recovered as
   the stretch of the station's GTSM harmonic tide (hourly, 1979-2024) that best matches the
   hydrograph where the surge has died out (|t| >= 48 h; spring variant), or jointly with a
   multiple of the spring variant's surge shape over the whole window (average variant). The
   spring fit matches to 4-8% of MHWS, so COAST-HG's spring tide is a real GTSM tide stretch.
   Per quality site, medians, against the gauge's annual maxima of empirical RP >= 10 y (their
   tide = level - non-tidal residual):

   | Spring range | Source | nb24 | nb24, tide part | nb24, surge part | tide at peak / MHWS | surge above half max |
   |---|---|---|---|---|---|---|
   | < 1 m | COAST-HG spring | 0.61 | 0.39 | 0.20 | 1.14 | 35 h |
   | | gauge | 0.59 | 0.37 | 0.18 | 1.26 | 18.5 h |
   | 1-2 m | COAST-HG spring | 0.73 | 0.59 | 0.12 | 1.05 | 22 h |
   | | gauge | 0.81 | 0.66 | 0.12 | 1.28 | 19 h |
   | 2-4 m | COAST-HG spring | 0.75 | 0.64 | 0.10 | 1.07 | 20 h |
   | | gauge | 0.87 | 0.77 | 0.09 | 1.25 | 15 h |
   | > 4 m | COAST-HG spring | 0.73 | 0.70 | 0.03 | 1.03 | 10 h |
   | | gauge | 0.88 | 0.83 | 0.04 | 1.18 | 9.5 h |

   - **The deficit is in the tide window, not the surge.** Above a 1 m range the surge parts of
     the next-day high water agree, and COAST-HG's surge lasts as long as the observed one or
     longer; the whole nb24 deficit sits in the tide part. Observed extremes coincide with tides
     of 1.18-1.28 x MHWS (the largest springs of the nodal cycle); COAST-HG's tide window reaches
     1.03-1.07 x MHWS and assigns the rest of the RP100 level to the surge, so its neighbouring
     tides are relatively lower. The average-tide variant is lower still (tide 0.56-0.78 x MHWS).
   - **The surge tail is the second, smaller difference.** Observed surges still hold 0.2-0.35 of
     their maximum 30-48 h from the peak; COAST-HG's surge is back to zero by about ±35 h. On
     microtidal coasts that tail is what keeps the observed troughs near MSL.
   - The surge core itself barely changes with RP (median duration above half the maximum 15-18 h
     at the gauges, 16-21 h in GTSM, for every RP class from 0-2 to 20-50 y).
8. **Duration correction on real tiles.** Confirm "α = 1, no lower clip" on 5-10 real 2-D tiles
   spanning tidal regimes, including a surge-dominated microtidal case, before adopting it.
9. **Done, with tests 2 and 5 at the gauges** (`surge_correction.py`; `.cleanup_baseline/
   level_sources/surge_correction/`). **Surge-deficit correction.** Test whether the deficit
   outside TC regions (~0.1 m) is regional and stable enough for a gauge-calibrated correction,
   with leave-one-out across gauges as the acceptance test.
   - *Estimator (tests 2, 5).* Skew surge per tidal cycle (highest level minus the predicted high
     water), GPD on its declustered exceedances above the 99.5% quantile, convolved with the
     predicted high waters assuming independence. 618 gauges of the extremes set with >= 15 years
     (median 34.5) and a harmonic fit, gauge and GTSM on the same years. It reproduces the gauge
     GEV (convolution - GEV, median +0.006 m RP10, +0.03 m RP100; MAE 0.05 / 0.19 m, the RP100
     MAE being GEV sampling noise) and COAST-RP's ETC levels from GTSM (+0.03 / +0.04 m; at
     ETC and no-TC stations +0.00-0.04 m).
   - *Error decomposition* (median, all 618 gauges, RP10 / RP100): surge (GTSM surge + gauge
     tide - gauge) -0.08 / -0.13 m, -0.15 / -0.28 m at TC-dominated gauges; GTSM tide +0.04 m
     (the 7-8% tide excess of §3); FES2022 tide -0.002 m (MAE 0.06 m); the production
     combination GTSM surge + FES tide -0.09 / -0.14 m (MAE 0.15 / 0.27 m). So the FES tide
     removes the tide error and leaves the surge deficit uncovered.
   - *Deficit as a ratio* r = gauge / GTSM skew-surge level at RP10: median 1.19 at the 451
     non-TC-dominated gauges (10-90%: 0.99-2.12), 1.39 at TC-dominated ones. Regional medians
     1.08 (N and W Europe) to 1.4-1.5 (W North America, E Asia) and > 2 where surges are small
     (equatorial Pacific, NW South America). Stable in time: r from the first and second half of
     each record correlates at 0.85 (median difference 0.10). Regional: r correlates at 0.69 with
     the nearest other gauge (median 84 km) and ~0.5 out to ~280 km.
   - *Leave-one-out correction*, r applied to GTSM's positive skew surges, production levels (FES
     tide) - gauge, 451 non-TC-dominated gauges, median / MAE:

     | Correction | RP10 | RP100 |
     |---|---|---|
     | none | -0.069 / 0.140 m | -0.113 / 0.212 m |
     | global median r | -0.006 / 0.120 m | -0.016 / 0.180 m |
     | inverse-distance r within 250 km | -0.005 / 0.115 m | -0.027 / 0.173 m |
     | inverse-distance r within 500 km | -0.004 / 0.115 m | -0.020 / 0.173 m |
     | inverse-distance r within 1000 km | +0.008 / 0.115 m | -0.003 / 0.169 m |

     With the gauge tide (surge error only) the MAE falls from 0.104 to 0.076 m (RP10) and 0.179
     to 0.137 m (RP100) at 500 km. At TC-dominated gauges the same correction halves the RP10 bias
     (-0.149 to -0.044 m) but leaves RP100 at -0.10 m (MAE 0.32 m): the TC deficit is not a
     ratio on ERA5 surges and stays with test 10.
   - *Result.* The correction passes: it removes the bias and cuts the MAE by 18% (production) to
     27% (surge only); the radius between 250 and 1000 km hardly matters, the regional pooling
     gains ~5% MAE over a global constant. Outside 500 km of a gauge the global median (1.19)
     applies. Ratios on small-surge coasts (> 2) rest on skew surges of a few decimetres; any cap
     from 1.5 to 2.5 leaves the leave-one-out MAE unchanged, and production caps at 2.0 (§8.4).
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
- **Tide atlas.** FES2022, with FES2014 replacing it where the FES2022 / FES2014 MHWS ratio
  exceeds 1.5 and the difference exceeds 1.25 m, and gauges overriding both (§8.1, test 6). EOT20
  is not used further.
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
same method with per-RP shape selection and a spring-tide window, so it would inherit
COAST-HG's main error (§5.2): next-day high water 0.1-0.15 of the peak too low, and T(0.25) 5-6 h
against 11-17 h observed. Test 7 traces it to the tide window: observed extremes ride on tides of
1.18-1.28 x MHWS, an MHWS-level spring window gives 1.03-1.07.

**Generator.** Per station and RP, with levels relative to local MSL. The design choices below
are decided; where a variant is listed, the acceptance test picks it.
- **Tide:** FES2022 at the station (FES2014 where the version rule of test 6 fires), 10 min,
  around a high water whose height is the station's own tide level during extremes: the median
  of tide-at-peak / MHWS over the station's GTSM annual maxima of the same RP class, applied to
  the FES MHWS. As a ratio, GTSM's tide excess cancels. Test 7 found 1.18-1.28 x MHWS by tidal
  regime against COAST-HG's ~1.05; a plain MHWS-level spring window reproduces COAST-HG's
  deficit.
- **Window:** ±72 h by default, extended (to ±120 h) at stations whose GTSM events are censored
  by the ±72 h window, i.e. still above MSL at its edges (surge-dominated microtidal coasts such
  as the Baltic and Mediterranean, §5.1), so that T and the SFINCS forcing see the full duration.
- **Surge shape, extratropical:** the median normalised GTSM surge of the station's events of
  the matching empirical RP class (HGRAPHER-RP event selection, 1979-2024). For RP >= 50, beyond
  the 46-year record, the shape of the station's largest events (RP 20-45) is used and only its
  amplitude follows the RP; the further fall of T with RP is then carried by the surge's growing
  share of the peak alone (an assumption, §5.3). Surge definition: the non-tidal residual of GTSM
  (tide + surge run minus tide-only run), smoothed by a 13-h running mean before the shape is
  built; the smoothing removes the tidal-period ripple that tide-surge interaction leaves in the
  residual (the role the skew surge has in Environment Agency, 2018; the skew surge itself is not
  yet tested).
- **Surge shape, tropical cyclones** (RPs where COAST-RP's TC part dominates): the surge part
  (observed minus predicted tide) of TC-driven gauge events where a quality gauge lies within
  50 km, scaled to the target; otherwise a Gaussian of half width 8 h (the 15 h median surge
  duration at TC-dominated gauges, §5.4). STORM-forced GTSM shapes replace the Gaussian once
  available (test 10). Limits: about 30% of TC stations have a gauge within 50 km, the scale
  factors spread from 0.6 to 1.9, and hourly gauge data under-resolve the sharpest peaks.
- **Peak:** the surge is scaled so the maximum of tide + surge equals the peak level. The peak
  level is an input (COAST-RP for now), so the tide-corrected peak of test 5 can replace it
  without changing the shape code. With an FES tide, the surge amplitude COAST-RP - tide
  absorbs GTSM's tide excess until then.
- **Phasing:** surge peak on the selected high water is the default; the observed phase of the
  selected GTSM events relative to high water (Horsburgh and Wilson, 2007) is the second variant.
- MDT, the 2020 offset and SLR are added downstream; the shape is the same for every SLR scenario
  (tides themselves change with SLR, Haigh et al., 2020, which is a known simplification).

**Diagnosis of COAST-HG's error.** Swap one component at a time and score each variant at the
quality sites:

| Component | Variant 1 | Variant 2 |
|---|---|---|
| Tide window | MHWS-level spring window (as COAST-HG; test 7: the cause of its deficit) | the tide level of the observed extremes (about 1.2-1.3 x MHWS) |
| Surge shape | HGRAPHER level-by-level mean of limb durations | empirical mean of the normalised surge residuals of the selected events, centred on their peaks |
| Surge definition | non-tidal residual | skew surge |
| Surge-tide phasing | surge peak on the spring high water | the phase of the selected GTSM events relative to high water |

**Acceptance test.** Build hydrographs at RP2 and RP10-100 and compare their normalised shapes
with the gauge events of the matching empirical-RP class (< 2 y; >= 10 y), per tidal-range class
and on the extremes site set of test 1:
- the median T(0.25), T(0.5) and nb24 within ±15% of the gauge medians;
- the profile RMSE against the gauge (peak-normalised profiles, +-48 h, as a fraction of the peak) no
  larger than that of GTSM's own events (0.10-0.22, §5.2);
- the T(0.25) ratio between RP classes reproduced (1.2-1.6, §5.3).

**Result** (`boundary_test/generator_validation.py`, `src/storm_tide_hydrograph.rp_hydrograph`;
`generator_acceptance.csv`, `generator_scores.parquet`; summary fig10, fig11). All 698 quality
sites, RP 2-100, gauge reference = the gauge's own annual maxima of the matching RP window (RP 50
and 100: its RP 20-45 events). Cells are tidal class x RP (20); a cell passes when T(0.25), T(0.5)
and nb24 all lie within ±15% of the gauge medians.

| Variant | Cells passing | Median deviation from the gauge, T(0.25) / T(0.5) / nb24 |
|---|---|---|
| generator, smoothed surge (adopted) | 14 of 20 | 8% / 5% / 3% |
| generator, raw residual | 10 of 20 | 9% / 8% / 3% |
| generator, observed surge-tide phasing | 10 of 20 | 12% / 7% / 4% |
| generator, shape from mean limb durations | 1 of 20 | 21% / 11% / 5% |
| HGRAPHER port | 0 of 20 | 46% / 17% / 15% |
| COAST-HG (RP100 only) | 0 of 4 | 53% / 26% / 8% |

- **COAST-HG's main error is fixed.** The next-day high water matches the gauges in every class
  (within 6% except microtidal RP 25), and T(0.25) rises from 5-6 h (COAST-HG, HGRAPHER) to within
  about 10% of the gauges, falling with RP as observed. The T(0.25) ratio RP 2 / RP 10 is 1.0-1.25
  against the gauges' 1.13-1.19.
- **Remaining misses:** T(0.25) at RP 2 is about 30% too short on microtidal (10 vs 14.5 h) and
  macrotidal coasts (16 vs 22 h); microtidal RP 25 is 18-20% too long; the RP 100 point of the
  1-2 m class is ~15% short, partly because it is compared with RP 20-45 events. Scaling the
  generator to the gauge events' own peaks instead of COAST-RP leaves the RP 2 deficit unchanged,
  so it lies in the shape, not in COAST-RP. On microtidal coasts all synthetic hydrographs drain
  lower between high waters than the gauges (generator about -0.15 of the peak, COAST-HG -0.3):
  the observed surge tail of test 7 is missing.
- **Profile RMSE criterion:** passed in 4 of 20 cells for every variant. The benchmark, GTSM's own
  events, is a median over many events and therefore smoother than any single hydrograph; the
  generator exceeds it by a median 0.02 of the peak. The criterion as written is too strict for a
  single synthetic event and should be restated (e.g. no more than 0.03 of the peak above GTSM's).
- **Profile RMSE in metres** (x COAST-RP peak, per site, median): generator 10-19 cm on 0-1 m
  coasts, 16-26 cm on 1-2 m, 25-36 cm on 2-4 m, 36-56 cm above 4 m; above GTSM's own events by
  0-4 cm except RP 2 on 2-4 m coasts (8 cm) and above 4 m (9-13 cm). The RMSE over +-48 h weighs
  low water as much as the peak, so on macrotidal coasts it is dominated by tidal phase; it says
  little about the part of the hydrograph that floods.
- **Surge tail and RP 2 (test 7b, `boundary_test/surge_tail_check.py`; `surge_tail_check.csv`).**
  Same sites and years, surge residual normalised as the generator does. GTSM's surge tail is not
  shorter than the gauges': at +-36 h it keeps 0.31-0.46 of the surge peak against the gauges'
  0.28-0.31 (raw; RP >= 10: 0.33-0.42 vs 0.19-0.30). No GESLA tail correction is needed. The
  microtidal levels between high waters are too low already in GTSM's own events (-0.38 vs -0.20
  of the peak; RP >= 10: -0.27 vs -0.09) because GTSM's surge is too small (RP >= 10 surge maximum
  0.37 vs 0.54 m, 0.55 vs 0.68 of the peak): with less surge in the peak the tide dominates the
  normalised profile. That is the surge-deficit correction of test 9, not a shape issue; scaling
  to the gauge peaks recovers part of it (RP 10: -0.28 vs -0.38). GTSM's own RP 2 events match the
  gauges' T(0.25) (13 vs 14.5 h on 0-1 m, 22.5 vs 22 h above 4 m), so the generator's RP 2 deficit
  comes from its construction (tide window or median shape), not from GTSM.
- **Where the RP 2 deficit arises** (`generator_validation.py --sample 150 --tag _swap`, ETC RPs;
  `generator_scores_swap.parquet`). Each station's own GTSM maxima of the RP window are rebuilt
  one generator step at a time, median T(0.25) at RP 2 (0-1 / 1-2 / 2-4 / >4 m): GTSM event as is
  13.0 / 15.8 / 17.5 / 21.5 h (gauges 14.2 / 15.2 / 16.5 / 23.0); surge scaled to COAST-RP
  12.0 / 14.5 / 15.0 / 19.5; surge smoothed 12.5 / 16.0 / 16.5 / 21.5; surge maximum moved onto
  the high water 11.0 / 15.5 / 15.0 / 21.0; GTSM tide scaled to MHWS_FES 11.0 / 13.5 / 15.5 /
  19.5; the generator's FES tide window with the event's own surge 9.8 / 11.8 / 17.0 / 19.0;
  generator 9.5 / 11.5 / 18.0 / 20.0. The loss is spread over steps; the largest are on the tide
  side (the lower FES tide and the FES high water picked by height alone, ~2 h each on 1-2 m
  coasts) and, on microtidal coasts, placing the surge maximum on the high water (~1.5 h). The
  median surge shape instead of the event's own costs under 1 h.
- **Flood impact of hydrograph errors (test 11, `boundary_test/sill_experiment.py`,
  `sill_analysis.py`; `.cleanup_baseline/level_sources/sill_experiment/`).** SFINCS 1-D strips:
  10 ocean cells, a 3-cell sill with its crest a head of 0.1, 0.25 or 0.5 x peak below the peak,
  then a closed basin of 2 or 5 km or a 2e-4 slope over 20 km, all at MSL; Manning 0.03 / 0.06.
  Part A (3,240 runs): per tidal class and RP 2 / 10 / 100, the gauge event closest to the class
  medians of T(0.25), T(0.5) and nb24, scaled to the class median COAST-RP, then stretched in time
  (x 0.7-1.3) or with the levels away from the peak shifted by +-0.05 / 0.1 of the peak. Part B
  (2,910 runs, basin 5 km and slope, n = 0.03): at 6 random sites per class and RP, the generator
  against up to 6 of the gauge's own events of the RP window, all scaled to the same COAST-RP.
  - Duration: flood volume changes by 0.6-0.9 x the relative change in duration (stretch 0.9:
    -6 to -9%; 0.8: -13 to -18%; 0.7: -20 to -28%), extent on the slope by about half that
    (0.8: -11%), maximum depth by about a quarter (0.8: -5 to -7%). The closed basins fill wall to
    wall at heads >= 0.25, so their extent does not change. Sensitivity is largest for low sills
    (head 0.1: volume elasticity ~1) and small for high ones (head 0.5: the hinterland fills to
    near the peak whatever the duration; stretch 0.7: -6%). The tidal class hardly matters.
  - Levels away from the peak act one-sidedly: -0.1 of the peak costs -10% volume (median), +0.1
    adds +46% (head 0.1: +218%), because the neighbouring high waters then overtop low sills too.
    Too high neighbouring high waters are the costly generator error, too short peaks the mild one.
  - Predictors: the overflow volume (time integral of the level above the crest) explains the
    flood volume change (R^2 of log ratios 0.68 part A, 0.57 part B); T(0.25) does not (0.20 both),
    since it counts whole hours and ignores how far above the threshold the level goes.
  - Generator against real events (part B, flood volume as a ratio to the median of the gauge's
    events): RP 2 0.91 / 0.99 / 0.99 / 0.98 and RP 10 1.03 / 0.95 / 0.95 / 0.93 (0-1 / 1-2 / 2-4
    / >4 m); extent 1.00 and maximum depth 0.97-1.01 in every class. The spread between the
    gauge's own events (IQR / median) is 0.16-0.22 in volume at RP 2. The generator lies within
    the range of the real events in 64-86% of the site-domain cases at RP 2 and 10, about what a
    randomly drawn event of 6 would score (71%). Low sills carry most of the bias (head 0.1:
    0.83-0.86 against an event spread of 0.31-0.37). The 24-33% short RP 2 T(0.25) on 0-1 and
    1-2 m coasts thus costs ~9% and ~1% of flood volume, well inside the event-to-event spread.
    At RP >= 25 the gauges have only 1-2 events per RP window, too few for this comparison.
- **Acceptance criteria derived from test 11** (replacing the +-15% on T(0.25), T(0.5) and nb24
  and the +-48 h profile RMSE): per tidal class x RP, the median of the generator's overflow
  volume above 0.9 and 0.75 x peak within +-10% of the gauge events' median (about +-7-10% in
  flood volume), with the deviation above the gauges no larger than that below (levels away from
  the peak that are too high are the costly error). Where gauge events suffice (RP 2, 10), the
  sill-test flood volume ratio within +-10% as a direct check: met in all 8 cells.
- **Overflow scores, all sites** (`generator_validation.py`, ex090 / ex075 = time integral of the
  peak-normalised level above 0.9 / 0.75, ratio of the medians to the gauge events'). The adopted
  generator passes 6 of 20 cells (the +-15% shape test: 14). It never exceeds the gauges by more
  than 10%, so the costly error is absent; it falls short instead: ex090 0.93-1.10 on 0-1 m coasts,
  0.80-0.89 on 1-2 m, 0.74-0.79 on 2-4 m, 0.61-0.94 above 4 m (lowest at RP 2-10); ex075 0.74-1.08.
  GTSM's own events pass 12 of 20: at RP >= 25 the
  gauges have 1-2 events per window and GTSM differs by +16-20% on 0-1 m and >4 m coasts, so
  +-10% is below the noise of the reference there. Rebuilding GTSM's events step by step (ex090,
  1-2 / 2-4 / >4 m, RP 10): scaled 1.07 / 0.97 / 1.01, smoothed 1.24 / 1.19 / 1.12, recentred
  1.12 / 1.09 / 0.99, tide scaled to MHWS_FES 1.01 / 0.94 / 0.81, median surge shape 0.95 / 0.83 /
  0.73, FES tide window (the generator) 0.89 / 0.78 / 0.68. The shortfall is spread over the last
  three steps: the lower FES tide, the median surge shape (largest on macrotidal coasts) and the
  FES high-water window, each -5 to -18%.
- **Decision: the generator (smoothed surge) is accepted as is.** Its overflow shortfall is the
  mild direction of error, and its flood volume behind the test 11 sills lies within 10% of the
  median of real events (RP 2, 10), inside their 16-22% spread. The overflow and shape scores stay
  in `generator_acceptance.csv` as diagnostics; no tolerance is tuned to the result.
- **Rejected variants:** keeping the observed surge-tide phasing adds nothing; averaging limb
  durations per level (the HGRAPHER construction) narrows the shape further. Smoothing the residual
  is what helps, by removing the ripple whose highest bump otherwise sets the normalisation.
- Tropical-cyclone RPs (13% of the site-RP cases) use a neighbour gauge's surge shape within 50 km,
  otherwise the Gaussian. Only 2.7% of stations needed the ±120 h window. FES2014 was not applied.

**Outputs.**
- Per station and RP: a 10-min hydrograph over ±72 h.
- Per station, RP and water depth: a T lookup (`event_volume.time_within_peak`).
- The test scores per variant, regime and RP class.

Inputs are the GTSM series, FES constants and GESLA events already extracted for this note; no
HPC runs are needed. Production use (the global product, SFINCS forcing, fsf recalibration with
α = 1 and no lower clip on the multiplier) uses the accepted generator.

### 8.4 Level product (ii) at all COAST-RP stations: the storm-tide product

The production step `run_preparation.py storm_tide_product` (`src/storm_tide_product.py`,
`preparation/prepare_storm_tide_product.py`) builds level product (ii) and the accepted
generator's hydrographs at the 23,226 COAST-RP stations from the raw inputs (GESLA, GTSM-ERA5,
FES2022, FES2014, COAST-RP, IBTrACS), into `boundary_conditions.storm_tide_product_dir`:
- `levels.parquet`: one row per station, RPs 2-1000;
- `hydrographs.npy`: (station, RP, 1441), 10-min steps over ±120 h, NaN outside each station's
  window;
- `hydrograph_info.parquet`: window, tide level and shape source per station and RP;
- `gauges/gauges.parquet`: the gauge ratios r, station class and tidal MHWS of the GESLA gauges;
- the two production files of §8.5.

**Construction.**
- *Stations.* Each COAST-RP station takes the nearest GTSM-ERA5 surge series (all within 0.8 km;
  1979-2024, hourly) and the FES2022 constants at its position.
  - FES2014 replaces FES2022 at 66 stations: the 56 that the version rule of §8.2 flags, plus 10
    neighbours on the same artefact (§8.5).
  - A station on an FES land cell (IBIS_GL_TS_MO_62198, Le Havre) takes the constants of the
    nearest station with values.
- *ETC levels.* GTSM skew surges on the hourly FES tide are fitted with a GPD, scaled by r (test
  9) and convolved with the 10-min FES high waters over a nodal cycle.
- *Surge ratio r.* r = `ratio_fes`, i.e. gauge / GTSM skew-surge RP10 with the FES tide under
  both. At the gauges it is equivalent to test 9's `ratio`: median 1.18 vs 1.19, correlation
  0.97. Each station takes the inverse-distance mean of r at calibration gauges within 500 km
  (at least 3), otherwise the gauge median, and r is capped at 2.0.
  - The leave-one-out MAE is the same for any cap between 1.5 and 2.5; 2.0 is about the gauge
    p90.
  - Station r: median 1.18. 203 stations are capped; 13,583 take the median (no 3 gauges within
    500 km).
- *TC part.* COAST-RP_TC enters as an annual exceedance rate added to the ETC rate, which is
  COAST-RP's own combination. TC is a gap-filler until test 10.
- *Hydrographs.* The accepted generator of §8.3. Where the TC rate exceeds the ETC rate at the
  RP level (and, below RP10, also at RP10), the station uses the TC shape: a neighbouring gauge's
  surge-dominated shape within 50 km, else the Gaussian.

**QA.** A first build gave GPD shapes up to 1.9 and RP100 levels up to 22 m. Four failure modes
caused them (fig17):

| failure mode | where | fix | stations affected |
|---|---|---|---|
| A second low-water bump of the shallow-water tide counts as a high water; the next real high water then reads as a skew surge of several metres | Ungava Bay, Hudson Strait, Amazon mouth | a maximum in the lowest quarter of the 25-h tidal range with a prominence < 10% of that range is not a high water (`skew_surge.HW_PROMINENCE`); weak lesser high waters of mixed tides stay | largest skew surge at Ungava 4.5 → 0.8 m |
| GTSM residual spikes of up to 12 m within an hour (a model artefact) | Arafura coast, New Guinea (137-139°E, 5-8°S) | hours more than 2 m from their 5-h median are dropped; a station with ≥ 5 hourly jumps > 2 m takes COAST-RP's levels and a Gaussian shape (`level_source = coast_rp`) | 15 |
| Tropical cyclones in the ERA5 record count twice: once in the ETC fit and once in COAST-RP_TC | Mississippi delta (Katrina), every TC coast | at stations with a TC part, cycles within ±24 h of an IBTrACS storm (≥ 34 kt, since 1980) passing within 300 km leave the ETC fit | 9,247 stations; median 0.3% of cycles removed, at most 4.7% |
| A heavy GPD tail fitted to a few large events | scattered | GPD shape capped at 0.35 and the scale refitted (Fathom RFA's cap; about the gauges' p95) | 149 |

Two further QA steps:
- *Peak overshoot.* Where a neighbouring high water in the window already exceeded the RP level,
  the window now moves to the nearest high water whose window stays below it. All 209,034 peaks
  equal the RP level to within 5e-7 m.
- *Spurious TC dominance at low RPs.* COAST-RP_TC is flat from RP2 to RP10 (the tide floor of
  its synthetic tracks). Below RP10, the TC shape therefore requires TC dominance at RP10 too.

**Against COAST-RP** (new - COAST-RP, median and 10-90%, m; number of stations with |difference|
> 0.5 m in brackets):

| RP | all 23,226 | no TC part (13,979) | TC part (9,247) |
|---|---|---|---|
| 2 | -0.009 (-0.25..+0.15) [1,180] | -0.019 (-0.33..+0.11) | +0.005 (-0.20..+0.30) |
| 10 | +0.018 (-0.23..+0.20) [1,230] | +0.014 (-0.28..+0.16) | +0.023 (-0.21..+0.34) |
| 100 | +0.026 (-0.22..+0.23) [1,316] | +0.039 (-0.27..+0.22) | +0.012 (-0.18..+0.29) |
| 1000 | +0.028 (-0.17..+0.33) [1,816] | +0.073 (-0.22..+0.35) | +0.000 (-0.13..+0.28) |

How to read the table (fig12, fig13):
- *Two effects of opposite sign.* With r = 1, the ETC level is 0.05-0.08 m below COAST-RP's ETC
  at stations without a TC part. That is the FES tide replacing GTSM's tide, which is 7-8% too high
  (§3). r lifts the level by 0.06-0.09 m. The net change is -0.02 to +0.07 m, which is test 9's
  gauge deficit with the tide error removed.
- *TC stations.* With the TC cycles masked, the ETC part moves as at the other stations. The
  total hardly changes at RP100 and above, because COAST-RP's TC rate dominates there.

**At the gauges** (fig14). There are 618 test 9 gauges with a station within 25 km. The reference
is the gauge's own skew-surge convolution. The new product's r is rebuilt without the gauge.
Station - gauge, median / MAE in m:

| | RP10 COAST-RP | RP10 new | RP100 COAST-RP | RP100 new |
|---|---|---|---|---|
| ETC-dominated gauges (451) | -0.017 / 0.179 | -0.005 / 0.107 | -0.068 / 0.244 | -0.021 / 0.164 |
| TC-dominated gauges (167) | -0.192 / 0.239 | -0.088 / 0.137 | -0.247 / 0.443 | -0.165 / 0.391 |
| stations with a TC part (323) | -0.128 / 0.216 | -0.036 / 0.119 | -0.181 / 0.342 | -0.072 / 0.269 |

The new product's MAE is 33-40% below COAST-RP's at ETC gauges. Its median bias is a third to
a half of COAST-RP's everywhere. Even the TC-dominated gauges improve: the ETC part alone, with r = 1, is 0.47 m low at RP100;
with COAST-RP_TC and r it is 0.17 m low.

So r does not overcorrect at TC coasts; they remain too low, and test 10 stays the route for TC
levels.

Where the new product departs strongly from COAST-RP, the gauges side with the new product:
- *Southern North Sea and Dover Strait* (Thames, Flanders, the Netherlands, Boulogne): the new
  levels are 0.6-1.0 m lower, all of it from the tide. At the test 9 gauges COAST-RP ETC RP100
  exceeds the gauge GEV by 0.9-1.1 m (Dunkerque 4.94 vs 4.06 m, Boulogne 6.06 vs 4.96 m).
- *Western Baltic* (Danish straits, Kiel Bight): the new levels are 0.3-0.9 m higher (fig15). At
  the 20 gauges there, COAST-RP ETC RP100 is 0.3-2.3 m below the gauge GEV, and r = 1.06-1.37.
- *Macrotidal basins* (Ungava Bay, Hudson Strait, King Sound, the Amazon mouth, Hangzhou Bay):
  the new levels are 2-4 m higher. GTSM's tide is too small there; Leaf Basin's MHWS, for example,
  is 8.2 m in FES and gauges.
- *Skagway* (Lynn Canal): FES2014, picked by the version rule, gives 4.01 m against COAST-RP's
  6.78 m.

**Hydrographs** (fig15).
- *Shape source* at RP100: GTSM events at 19,288 stations, TC Gaussian at 3,354, TC gauge shape at
  568, the GTSM-QA Gaussian at 15, the no-surge Gaussian at 1. The TC share grows from 5% at RP2
  to 22% at RP1000. It is larger than before the TC mask, because the masked ETC part is lower.
- *Window and tide.* Median window ±72 h. The median tide at the annual-maximum peaks is 1.2 x
  MHWS, the same as the k of §8.3.

All levels are relative to model MSL. MDT, the 2020 offset and SLR are added downstream as for
COAST-RP (§7). RPs above 250 y extrapolate ~46 years of skew surges and are flagged as such
(TODO.md).

### 8.5 From stations to tiles: interpolation, FES consistency and the production switch

**Interpolation** (`station_interpolation.py`, fig16). Production seeds each tile's coastline
cells by inverse-distance weighting over the k nearest stations of the same ocean component
(`flood_model.idw_seed_values_at`; k = 15 and power 2 until this test). A leave-one-out test over the 22,670 stations
north of 60 S shows how well a coastline cell between stations is predicted. It is global, i.e.
without the ocean-component restriction.

| RP100 | MAE (cm) | p90 \|error\| (cm) | stations > 0.5 m |
|---|---|---|---|
| nearest station | 7.0 | 17.8 | 556 |
| k = 15, power 2 (former production) | 7.0 | 18.5 | 470 |
| k = 5, power 2 | 5.9 | 15.1 | 379 |
| k = 3, power 3 | 5.6 | 13.8 | 380 |

| k = 5, power 3 (production) | 5.7 | 14.3 | 377 |

- The gain from fewer neighbours is about 1.5 cm.
- The spacing of the stations sets the error: 2 cm within 5 km, 3-4 cm at 5-10 km, 8-12 cm at
  10-50 km and about 20 cm beyond 50 km.
- Production uses k = 5 and power 3 (`simulation.flooding.knn`, `idw_power`) for both sources,
  for the coastline seeds and the static level. k = 5 is barely worse than k = 2-3 and has the
  fewest stations off by more than 0.5 m; it depends less on a single bad station.
- The test predicts a station from neighbours that are farther away than a coastline cell's,
  so it overstates the error. The ranking should hold.
- `src/along_water.py` (IDW over water-path distances) would address the errors across
  headlands and at estuary heads. It is built and unit-tested but not yet scored (TODO.md).

**FES consistency** (the basin-head QA of the plan). The 73 stations whose level differs by more
than 1 m from the k = 5 prediction are mostly real gradients:
- estuary and fjord heads;
- stations on opposite sides of a headland;
- sparsely sampled macrotidal basins (Ungava Bay, Hudson Strait, King Sound, Shelikhov Gulf).

A few were neighbours that the version rule split. The rule flags FES2022 where it exceeds FES2014
by a ratio of 1.5 and by 1.25 m. Two stations 3 km apart in Suriname had FES2022 MHWS of 2.54 and
2.37 m against FES2014's 1.20 m. Only the first passed the 1.25 m difference, so the RP100 levels
of the two were 2.17 and 3.28 m.

`consistent_version_flag` grows the flag:
- A station within 25 km of a flagged one is flagged too when it passes the ratio and a difference
  of 0.25 m. This repeats until no station is added.
- No station within 50 km of a gauge where FES2022 matches GESLA within 0.25 m is added. This veto
  keeps FES2022 in the St Lawrence near Tadoussac (gauge 2.18 m, FES2022 2.14 m, FES2014 0.94 m).

The rule adds 10 stations: Varangerfjord, Labrador, Suriname, Maranhão, Sonmiani Bay, the Aldabra
group and the Falklands. One of them is doubtful: in São Marcos Bay (Maranhão) FES2022's MHWS of
3.2 m matches the published spring range better than FES2014's 2.1 m (TODO.md).

**Production switch.** `boundary_conditions.storm_tide_source` (`gfm_product`, the default, or
`coast_rp`) selects the source for every model. The product step writes two files:
- `GFM_storm_tide_levels.nc`: COAST-RP's stations, order, coordinates and `storm_tide_rp_XXXX`
  variables, plus `level_source`, r, atlas, `gpd_shape` and `tc_masked_share`.
  `prepare_boundary_conditions` reads it in place of COAST-RP and keeps the same 22,670
  stations north of 60 S; MDT, the 2020 offset and SLR are added as before.
- `GFM_storm_tide_hydrographs.nc`: `storm_tide_hydrograph` (station, return period, time).
  Correction B reads it at the scenario's RP (`event_volume.load_coast_hg`). The SFINCS forcing
  takes its RP100 hydrograph, cut to COAST-HG's time axis, and shifts it to the tile boundary's
  level as before (`sfincs_tiles/build_line_forcing.py`).

A marker next to the scenario NetCDFs records which source they hold. A config asking for the
other source stops at the station cache instead of reading them. What is left to run (the product
on `<data_root>`, the SFINCS forcing, the fsf recalibration, test 8, the tile rerun) is listed in
TODO.md.

**Durations** (`product_durations.py`, fig18). Production with COAST-HG uses one RP100
hydrograph per station for every RP. The product's hydrographs are wider over nearly the whole
coast: median T(0.25) 12 h against 6 h at RP100, and T(0.5) 35 h against 23 h. T falls with the
RP, as the gauges show (fig10): T(0.25) is 13 h at RP2 and 11 h at RP1000. The product is
narrower than COAST-HG only where the TC Gaussian replaces the shape (Gulf of Mexico, the
Caribbean, Pacific islands). Correction B reads T from these hydrographs, so the longer T lowers
its friction multiplier. That is why fsf must be recalibrated with the product.

### 8.6 The product's estimated rules, tested

`open_guesses.py` (fig19-fig23) varies each rule that was set by estimate. Where it can, it
scores the variant at the 618 ratio gauges with r left one out. The reference is test 9: the
convolution of the gauge's own skew surges with its own tide.

| rule | variants | at the gauges (RP100 station - gauge, median / MAE) | outcome |
|---|---|---|---|
| GPD shape cap 0.35 (fig20) | none, 0.5, 0.35, 0.25, 0.2 | all gauges: MAE 0.224-0.227 m for every cap | keep; it only bounds RP1000 |
| despike 2 m, >= 5 jumps (fig19) | 1-3 m, 1-20 jumps | not testable at gauges | keep |
| TC mask 300 km, +-24 h, 34 kt (fig21) | none; 150 / 500 km; +-12 / 48 h; 64 kt | TC-dominated: none -0.03 / 0.36 m, production -0.17 / 0.39 m | open: the gauges favour no mask |
| FES version rule and its growth (fig22) | at gauges and stations near gauges | FES2014 matches the gauge where the rule fires | keep; one wrong flag |
| hydrograph maximum off time 0 (fig23) | diagnosis | not a level question | keep; cosmetic |

- *Shape cap.* The cap binds at 149 stations. At the gauges no cap and a cap of 0.2 differ by
  3 mm of MAE: RP100 is still close to the data and the gauges rarely sit where GTSM's tail is
  heavy. It matters at RP1000. Without it, the heavy-tailed stations rise by up to 9.7 m, and
  by up to 4.1 m with 0.5; a cap of 0.2 lowers them by up to 3.1 m. The gauges' own shapes
  (median 0.01, standard deviation 0.25) are no narrower than GTSM's (-0.03, 0.19), so they do
  not fix the value. 0.35 stays as a bound on the extremes.
- *Despike.* The median station's largest hourly change in the GTSM residual is 0.17 m.
  - 22 stations have a jump above 2 m. 15 of them have 3 or more, and the same 15 are picked
    for any count from 3 to 5. They are the Arafura cluster, with jumps up to 13 m.
  - A 1 m threshold would also catch real fast surges: outside Arafura, the stations with
    20 or more jumps above 1 m are in the southern North Sea and the German Bight.
- *TC mask.* The mask makes the TC coasts worse at the gauges.
  - TC-dominated gauges at RP100: median -0.03 m and MAE 0.36 m without the mask, against
    -0.17 m and 0.39 m with production's. Stations with a TC part: -0.02 m and 0.25 m against
    -0.07 m and 0.27 m. Non-TC gauges are unchanged.
  - A 150 km radius lies between the two; the time window and the wind threshold hardly
    matter.
  - Without the mask a cyclone counts twice: once in the ERA5 record and again in COAST-RP's
    TC part. It still matches the gauges better, because both sources are too low. GTSM-ERA5
    resolves cyclones coarsely, and COAST-RP's TC part is 0.2-0.25 m below the gauges (§8.1).
    The double count makes up for that.
  - The mask is right in principle and wrong on the gauges until test 10 (STORM-forced GTSM)
    replaces the TC part. Whether to drop it is in TODO.md.
- *FES version rule.* Where gauges exist, the rule is right.
  - It fires at 3 of 912 gauges (Juneau, Skagway, Port Alberni). There FES2022 is 1.5-4.7 m
    too high and FES2014 within 0.1 m (0.6 m at Juneau).
  - Flagged stations near a gauge sit at the gauge's level with FES2014: Varangerfjord against
    Vardø (1.35 m), Le Crouesty, Sonmiani against Karachi, Kukup, and the Falklands against
    Port Stanley.
  - In São Marcos Bay, the Ponta da Madeira gauge inside the bay reads 2.62 m. The outer
    stations' FES2014 (2.1 m) is therefore plausible and FES2022 (3.2-3.8 m) is not; the doubt
    of §8.5 is resolved in favour of the rule.
  - One original flag is wrong. At the Saguenay mouth the Tadoussac gauge (2.18 m, 30 km)
    sides with FES2022 (2.65 m) over FES2014 (1.38 m). A gauge veto on the rule itself, not
    only on its growth, would catch it.
  - 41 flagged stations have no gauge within 150 km. Most are FES2022 artefacts of known
    basins (Magellan, Gulf of California, Ungava). Some lie in real macrotidal gulfs (Khambhat,
    Incheon), where neither atlas is checked.
- *Hydrograph maxima off time 0.* 4.1% of all hydrographs and 3.2% at RP100 peak away from the
  centre high water, most often at RP2 in semidiurnal tides.
  - Where the maximum falls instead: 64% a day away (±24.8 h), 19% one tidal cycle away
    (±12.4 h), 8% within the hour.
  - The centre is the median tide at the annual maxima. A neighbouring high water (diurnal
    inequality, spring-neap cycle) is higher, and the slowly varying surge lifts it above the
    centre.
  - Half exceed the time-0 level by less than 1.6 cm and 99% by less than 21 cm. The maximum
    is the RP level in every case, which is what SFINCS and correction B use.

## 9. Comparison with other global coastal boundary conditions

| | This pipeline (GCFM) | Fathom RFA (Collings et al., 2024; Wing et al., 2024) | Chaudhuri et al. (2027) |
|---|---|---|---|
| Still-water extremes | COAST-RP: GTSM-ERA5 surge (ETC) + STORM synthetic TCs, empirical RPs after a probabilistic surge-tide combination | RFA after Sweet et al. (2020, 2022): GPD on pooled, normalised daily maxima; gauges where >= 3 within 400 km (851 of 5,975 1° cells), otherwise GTSM-ERA5 (4,555 cells) or single-site GTSM-ERA5 (569 cells); shape capped at 0.35 | skew-surge return levels from a global SSJPM on GTSM-ERA5, bias-corrected at gauges (Alexandre and Chaudhuri, 2025) |
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
12. `atlas_check.py` (FES2014 on `<data_root>`), `atlas_node_mhws.py`, then
    `atlas_rule_sensitivity.py`
13. `coast_hg_decompose.py` (after `shapes_events.py`)
14. `duration_experiment.py` (local SFINCS)
15. `duration_analysis.py`
16. `duration_refit.py`
17. `generator_validation.py` (then `--sample 150 --tag _swap` for the step diagnostics),
    `surge_tail_check.py`
18. `sill_experiment.py` (local SFINCS, after `generator_validation.py`), then `sill_analysis.py`
19. `surge_correction.py`
20. `export_input_tables.py`
21. `input_analysis/make_boundary_forcing_figures.py` (the summary figures and their descriptions)
22. `preparation/run_preparation.py storm_tide_product` (§8.4; ~6 h on 4 workers, into
    `boundary_conditions.storm_tide_product_dir`)
23. `level_product_gauges.py`, `level_product_qa.py` (`--product-dir`), `station_interpolation.py`,
    `product_durations.py` (`--out-dir` = that directory), `open_guesses.py cap despike tc_mask
    fes_rule peaks` (`--product-dir`, §8.6), then the figure script with `--product-dir` for
    fig12-fig23

Inputs on `<data_root>` are read only.

## 11. Data files

`boundary_test/input_analysis_data/`, written by `export_input_tables.py`. There is one row per
quality site (698), keyed by the GESLA file name; levels are in m above local MSL.

- **`sites_levels.csv`**: the data of fig01 and fig02.
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

Iturbide, M., Gutiérrez, J. M., Alves, L. M., Bedia, J., et al. (2020). An update of IPCC climate
reference regions for subcontinental analysis of climate model data: definition and aggregated
datasets. Earth System Science Data, 12, 2959-2970. https://doi.org/10.5194/essd-12-2959-2020

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
