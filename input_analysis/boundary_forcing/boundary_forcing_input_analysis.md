# Boundary forcing: input analysis summary

The coastal boundary of the GFM pipeline needs two things per station and return period (RP):
a **peak water level** and a **hydrograph**. The hydrograph forces SFINCS and sets the duration T
of the eikonal friction correction. This note summarises which global datasets can supply those
two things, how they compare with tide gauges, and what was decided. It condenses the full
analysis in `boundary_test/INPUT_ANALYSIS.md`; section references (§) and test numbers point there.

Everything is evaluated at GESLA 4.1 tide gauges, which serve as the reference. Model comparisons
(eikonal vs SFINCS on real tiles) are outside this note. Every figure in `figures/` has a `.txt`
file of the same name that explains what it shows and how to read it; they are drawn by
`input_analysis/make_boundary_forcing_figures.py`.

## Datasets and terms

| Dataset | What it is | Role |
|---|---|---|
| **GESLA 4.1** | observed water levels at 1,154 tide gauges with >= 20 y of record | reference |
| **GTSM v3** (CDS) | global hydrodynamic model: tide-only run (1996-2014) and ERA5-forced surge (1979-2024) | tide and surge behind COAST-RP |
| **FES2022** | ocean tide atlas, 34 constituents, 1/30 deg, assimilates altimetry and gauges | tide source |
| **FES2014** | the previous FES version (`fes2014b_elevations_extrapolated`, 1/16 deg) | fallback where FES2022 fails |
| **COAST-RP** | storm-tide return levels RP 1-1000 y at 23,226 GTSM stations; extratropical (ETC) surge from GTSM-ERA5 plus tropical-cyclone (TC) surge from STORM synthetic tracks, combined with the GTSM tide | production peak level |
| **COAST-HG** | one RP100 storm-tide hydrograph per COAST-RP station (HGRAPHER method), average- and spring-tide variants; total level only | current hydrograph shape |

EOT20, an altimetry tide model built as a residual on FES2014b, was used as a second check and is
not used further (§2).

- **MHWS:** mean high water springs, as a height above local mean sea level (MSL). The main tide
  metric here; the spring range is about twice MHWS.
- **T(f):** total time, over all tidal cycles in a ±72 h event window, that the water level stays
  within a fraction f of the peak (above MSL). T(0.25) is the T of the eikonal correction for a
  cell whose water depth is a quarter of the storm tide.
- **nb24:** the next-day high water as a fraction of the peak; it measures how much the
  neighbouring tides of a storm still flood.
- **Empirical RP** of an observed event: (n + 1) / rank among the site's n annual maxima. With 30-46
  years of record the largest is about 45 y.
- **Regions:** IPCC AR6 reference regions (Iturbide et al., 2020): the nearest land region within
  200 km of a gauge, otherwise the ocean region containing it.

How the comparison is made fair: every source is reduced to heights above its own local MSL
(annual means removed), fitted with the same 34-constituent harmonic method, predicted over the
same 18.61-year nodal cycle at 10 min, and reduced to MHWS with the same statistics code. Each
gauge is paired with its nearest GTSM station (median 1.9 km). The tide comparison uses 698
quality sites, where the harmonic fit explains >= 50% of the variance. The extremes comparison
uses a record-based set of 623 sites that includes the surge-dominated Baltic and Mediterranean
gauges (test 1).

## 1. Which source represents the tide best?

**FES2022** (fig01-fig03). The comparison is made on tidal levels (MHWS) and on the harmonic
constituents behind them, each derived from the source's own series or atlas constants; whole
water-level series enter in §4.

| MHWS against the gauge (698 sites) | Median ratio | Mean absolute error | Within 0.1 m |
|---|---|---|---|
| GTSM tide | 1.08 | 0.16 m | 53% |
| FES2022 | 0.99 | 0.05 m | 88% |
| FES2014 (695 sites) | 0.99 | 0.07 m | 84% |

- **GTSM's tide is 7-8% too large** almost everywhere (about 20% in the Mediterranean and on
  African coasts) and too small on the east coast of North America (fig01, fig02). FES2022 is
  close to the gauges in every region.
- **The cause is in the constituents** (fig03): GTSM's S2 is 20% and M2 6% too large, against
  0.99-1.00 for FES. The spring range is roughly M2 + S2, so that is the MHWS excess. The likely
  source is GTSM v3's uncalibrated bathymetry and bottom friction (Wang et al., 2022).
- **Not a location effect:** FES gives the same MHWS at the gauge and at the GTSM station (MAE
  0.01 m), so the bias is GTSM's, and FES can stand in for gauges where there are none.
- **Where FES still misses** (by more than 0.25 m at 20 of 698 sites; GTSM at 124): estuaries,
  fjords and narrow entrances (upper St Lawrence, Long Island Sound, Salish Sea, Port Phillip
  entrance, Juneau).

The same tide excess sits inside COAST-RP, which is built on the GTSM tide (§3).

## 2. Where FES2022 is badly wrong, and the version rule

At a few basin and fjord heads FES2022's MHWS is as large as the whole observed spring range.
These values flood the eikonal tiles, because the solver seeds every boundary cell with its own
level.

**How they were found.** Not by comparison with GTSM, which is itself biased (25% too low in the
Bay of Fundy). The chain was: the eikonal CSI against SFINCS on calibration tile 891 (Minas
Basin) fell from 0.57 to 0.27 with the FES forcing, which led to FES nodes with MHWS of 14.5 m;
published spring ranges (Minas 14.5 m mean, 16.3 m record; Strait of Magellan ~9 m; Colorado mouth
7-10 m) showed FES right over most of each basin and wrong only at the heads; the Skagway and Juneau
gauges exposed a fourth cluster (FES2022 7.5 / 6.0 m against 2.8 / 2.7 m observed); finally the
second atlases (test 6).

**FES2014 does not have them** (fig04, fig05). FES2022 and FES2014 were both predicted at the
102,935 boundary nodes of the test tiles. They agree almost everywhere; where FES2022 is far
higher, FES2014 stays within the observed ranges:

| Basin head | FES2022 MHWS max | FES2014 MHWS max | Observed |
|---|---|---|---|
| Minas Basin / Cobequid Bay | 14.5 m | 7.2 m | MHWS ~7.3 m (spring range 14.5 m) |
| Strait of Magellan / Bahía Lomas | 16.3 m | 4.6 m | spring range ~9 m |
| Upper Gulf of California | 15.4 m | 3.0 m | spring range 7-10 m |
| Glacier Bay / Lynn Canal | 12.2 m | 1.8 m | Skagway 2.78 m, Juneau 2.72 m |

At the gauges FES2014 is slightly worse overall than FES2022 (§1), so FES2022 stays the primary
source. EOT20 is dropped: it repeats FES2014 where altimetry is sparse and adds its own failures
elsewhere (solar constituents of 14-31 m in the St Lawrence files, far too low in Puget Sound and at
Trondheim, no values in narrow inlets).

**The rule.** At a boundary node FES2014 replaces FES2022 when both hold:
MHWS_FES2022 > **1.5** × MHWS_FES2014 and MHWS_FES2022 - MHWS_FES2014 > **1.25 m**. A gauge
overrides both atlases where one exists. The ratio alone would fire on microtidal coasts where both
MHWS are centimetres; the difference alone would switch macrotidal coasts where FES2022 is only
moderately higher. The rule replaces 1,287 nodes (1.25%), concentrated at the basin heads.

**Why 1.5 and 1.25 m** (fig06). The rule was evaluated for R from 1.1
to 3 and D from 0 to 1.5 m against two kinds of evidence:

1. **The 905 gauges with all three MHWS.** Below a difference of 0.5 m FES2014 is not the better
   atlas (closer to the gauge at only 31-50% of sites), so switching there harms. Above 1 m it is
   closer at 4 of 5 gauges by a median of 1.43 m. D must sit above the largest difference at which
   FES2014 is wrong (Trondheim, 1.08 m) and below the smallest real FES2022 error that must be
   caught (Port Alberni, 1.43 m). With D = 1.25 m the rule fires at 3 gauges, all fixes: Skagway
   7.46 → 2.76 m (gauge 2.78), Juneau 5.95 → 3.33 m (2.72), Port Alberni 2.79 → 1.36 m (1.28).
   The MHWS RMSE over the 905 gauges falls from 0.233 m (FES2022 alone) to 0.129 m, the lowest
   of all pairs.
2. **178 known artefacts**: basin-head nodes whose FES2022 MHWS exceeds 1.25 × the largest observed
   MHWS of their basin, a set defined without FES2014. Every R up to 1.5 replaces all of them;
   R = 1.75 misses 6, R = 2 misses 27.

The data do not pin R down: from R = 1 (ratio condition off, D alone) to R = 1.5 the result is
the same on both counts. R only changes how many other nodes switch (1,464 at R = 1, 1,287 at
R = 1.5): the 177 nodes in between lie on macrotidal coasts where FES2022 is up to 50% higher and
no gauge decides, and R = 1.5 is the cautious choice that keeps FES2022 there. The gauge evidence rests on
few sites (5 with a difference above 1 m).

**No atlas catches the moderate errors.** Where FES2022 is too high by 0.25-1 m (fjords, inlets),
FES2014 and GTSM are usually wrong in the same direction; only gauges can correct these.

## 3. Peak levels: COAST-RP against the gauges

COAST-RP return levels are compared with generalised extreme value (GEV) fits to the gauge annual
maxima, per AR6 region (fig07; numbers in `figures/fig07_coast_rp_vs_gauges_data.csv`).

| 623 record-based sites | RP10 median difference | RP10 MAE | RP100 median difference | RP100 MAE |
|---|---|---|---|---|
| COAST-RP - gauge | -0.06 m | 0.21 m | -0.07 m | 0.35 m |

- **Two errors cancel.** COAST-RP's error tracks GTSM's tide excess (correlation 0.88). With the
  tide excess removed, COAST-RP is 0.11-0.12 m too low: its surge part is too low and the
  too-large tide hides it. Replacing the tide with FES alone would therefore lower the levels and
  expose the surge deficit.
- **The deficit is largest in tropical-cyclone regions:** E.North-America -0.36 / -0.41 m (RP10 /
  RP100), E.Asia -0.15 / -0.22 m, the Caribbean, N.Australia and the Pacific regions -0.07 to
  -0.18 m. Gauge records of 30-40 years under-sample cyclones, so these are lower bounds. In
  West&Central-Europe COAST-RP is too high (+0.27 / +0.32 m), where the tide excess dominates.
- **Same model, different statistics.** Where ETC surges dominate, a GEV on 46 years of GTSM and
  COAST-RP agree within a few cm; the statistics explain a third to two fifths of the MAE.
  Interannual MSL variability adds only about 1 cm (test 3), so it needs no extra term.

The peak level stays COAST-RP for now. The planned replacement is a skew-surge convolution of GTSM
surge with the FES tide and a gauge-calibrated surge correction (tests 2, 5 and 9).

## 4. Hydrograph shapes: COAST-HG, GTSM and the gauges

Events are the annual maxima at each quality gauge with ±72 h around the peak, normalised by the
peak (fig08). GTSM's own events reproduce the gauges' shapes in every tidal regime (same T, same
neighbouring high waters, the smallest profile error; per site rho 0.87 for the time above MSL).
COAST-HG's central peak is right, but on coasts with a spring range above 1 m its neighbouring
high waters are 0.1-0.15 of the peak too low, and its T(0.25) of 5-6 h against 11-17 h observed
means 2-3× too much eikonal friction.

**Why COAST-HG's neighbours are too low (test 7, fig09).** COAST-HG holds only the total level.
Its tide was recovered as the stretch of the station's GTSM tide that matches the hydrograph where
the surge has died out (to 4-8% of MHWS), and the surge as the remainder. On coasts with a spring
range above 1 m:

| | next-day high water / peak | its tide part | its surge part | tide at peak / MHWS |
|---|---|---|---|---|
| COAST-HG spring tide variant | 0.73-0.75 | 0.59-0.70 | 0.03-0.12 | 1.03-1.07 |
| gauge events, RP >= 10 y | 0.81-0.88 | 0.66-0.83 | 0.04-0.12 | 1.18-1.28 |

The surge parts agree, and COAST-HG's surge lasts as long as the observed one. **The deficit is
in the tide window:** observed extremes coincide with tides of 1.18-1.28 × MHWS, the largest
springs of the nodal cycle, while COAST-HG's tide reaches about 1.05 × MHWS and the rest of the
RP100 level is assigned to surge. A second, smaller difference is the surge tail: observed surges
still hold 0.2-0.35 of their maximum 30-48 h after the peak, COAST-HG's is gone by about ±35 h.

### Shape against return period

Rarer events are more surge-dominated, so their tidal neighbours are relatively lower and T falls
(fig08). T(0.25) at the gauges, by empirical RP class:

| Spring range | RP 0-2 | 2-5 | 5-10 | 10-20 | 20-50 y |
|---|---|---|---|---|---|
| < 1 m | 14 h | 14 h | 13 h | 9.5 h | 8 h |
| 1-2 m | 16 h | 15 h | 13 h | 12 h | 9 h |
| 2-4 m | 17 h | 16 h | 14 h | 15 h | 13 h |
| > 4 m | 22 h | 21 h | 20 h | 18 h | 15 h |

- **No convergence within the observable range.** T(0.25) is still falling from RP 10-20 to RP
  20-50 in every regime, at the gauges and in GTSM. T(0.5), which spans several tidal cycles,
  hardly changes above a 2 m range (38-42 h). Above RP ~45 the records cannot say whether the
  shape settles, so the hydrograph has to be built per RP rather than assume it.
- **The surge itself does not change with RP** (fig09): its core lasts a median of 15-18 h above
  half its maximum at the gauges (16-21 h in GTSM) in every RP class. The change of the total
  shape with RP comes from the surge's growing share of the peak, not from longer surges.
- **Consequence today:** using COAST-HG's RP100 shape for every RP gives frequent events (RP 2-5)
  a T that is 15-55% too short, on top of COAST-HG's own short T.

### Tropical-cyclone stations

GTSM-ERA5 does not resolve tropical cyclones, so its events cannot supply TC shapes. At
TC-dominated gauges the largest events are surge-dominated (surge share 0.54) and shorter (15 h
against 18 h above half the surge maximum). Gauge shapes could be used there, scaling only the
surge part, but only 30% of TC stations have a gauge within 50 km and the scale factors spread from
0.6 to 1.9, so this is a local option.

## 5. Conclusion on the hydrograph: build it per RP from GTSM

Retire COAST-HG for both roles and generate one hydrograph per station and RP (§8.3, after
HGRAPHER-RP in GCFM_UUdev):

- **Tide:** FES2022 at the station (FES2014 where the version rule fires), around a high water
  at the station's own tide level during extremes: the median tide-at-peak / MHWS of its GTSM
  annual maxima of the same RP class (1.18-1.28 × MHWS by regime, test 7), applied to the FES MHWS.
- **Window:** ±72 h, extended to ±120 h where the station's events outlast it (surge-dominated
  microtidal coasts).
- **Surge shape:** the median normalised GTSM surge of the station's events of that RP class; for
  RP >= 50, beyond the 46-year record, the shape of its largest events (RP 20-45), with only the
  amplitude following the RP. The GTSM residual is smoothed by a 13-h running mean first, which
  removes the tidal-period ripple of tide-surge interaction.
- **Tropical cyclones:** the scaled surge part of TC-driven gauge events where a gauge lies within
  50 km, otherwise a Gaussian of 8 h half width, until STORM-forced GTSM shapes are available
  (test 10).
- **Peak:** the surge is scaled so that tide + surge reaches the RP peak level (COAST-RP for now).

**Acceptance test** (fig10, fig11; all 698 quality sites, RP 2-100). A tidal class and RP passes
when T(0.25), T(0.5) and the next-day high water are all within ±15% of the gauges' own events of
that RP window. The generator passes 14 of 20 cases, the HGRAPHER port 0 of 20 and COAST-HG 0 of 4;
its median deviation from the gauges is 8% for T(0.25) against 46-53% for those two. The next-day
high water now matches the gauges in every class, and T falls with RP as observed. Still missed:
T(0.25) at RP 2 on microtidal and macrotidal coasts (about 30% too short), and on microtidal coasts
the levels between high waters (the observed surge tail is missing). Scaling to the gauges' own
peaks leaves the RP 2 deficit unchanged, so it lies in the shape, not in COAST-RP. Rejected along
the way: keeping the observed surge-tide phasing (no gain) and building the shape from mean limb
durations as HGRAPHER does (narrower still).

Neither GESLA nor regional pooling supplies the shapes: GESLA is the yardstick the generator is
scored against but covers too few stations, and pooling was not analysed for shapes (it appears
only for the surge-deficit correction, test 9). **Acceptance test before production:** at RP2 and
RP10-100 the generator's median T(0.25), T(0.5) and nb24 lie within ±15% of the gauge medians of
the matching RP class and tidal regime, its profile error is no larger than GTSM's own events', and
it reproduces the fall of T(0.25) with RP.

## 6. What this means for the eikonal duration correction

The friction multiplier is m = (t_ref / T)^α, capped at 10 with no lower clip: a longer T gives a
smaller m, less friction and a longer reach. On 540 synthetic 1-D SFINCS floodplains, with the friction factor fsf
refitted per form, m ∝ T^-α without a lower clip fits best (log-extent RMSE 0.140 against 0.148
for the production form); α = 0.5 and 1 perform alike, and a constant t_ref is absorbed into fsf.
The agreed setting is α = 1 with fsf calibrated on SFINCS, with T from the per-RP hydrographs of §5.
Confirmation on 5-10 real 2-D tiles follows once the generator exists (test 8).

## 7. Decisions and open items

**Decided:**
- Tide: FES2022, FES2014 where the version rule fires (MHWS ratio > 1.5 and difference > 1.25 m),
  gauges overriding both. EOT20 not used.
- Hydrograph: per-RP generator (§5, built and tested: 14 of 20 cases pass) from GTSM surge shapes on the FES tide at each station's tide
  level during extremes, scaled to the RP peak; largest-event shapes for RP >= 50; window
  extended where events outlast ±72 h; gauge shapes, else a Gaussian, at TC stations. COAST-HG is
  retired once a variant passes the acceptance test.
- Duration correction: α = 1, fsf calibrated on SFINCS.
- Zero-SLR scenario at the 2020 sea level: the AR6 rise from 1995-2014 to 2020 (0.049 m,
  regionalised with the SLR fingerprints) is added to the boundary level.

**Open:**
- The FES version rule is in the level product (§8), not yet in production; how far a gauge
  override reaches is undefined.
- Tests not yet run: 2+5 (skew-surge convolution, one estimator three ways), 9 (surge-deficit
  correction), 10 (STORM-GTSM TC shapes; needs the STORM-forced runs), 8 (real tiles, after the
  generator). Deferred: test 4 (absolute MSL + MDT against geodetic gauge datums); wave setup.

## 8. The level product at all COAST-RP stations

The decisions above are built into one product at the 23,226 COAST-RP stations
(`boundary_test/build_level_product.py`; details in `boundary_test/INPUT_ANALYSIS.md` §8.4-8.5):
GTSM-ERA5 skew surges on the FES tide, scaled by the gauge ratio r and convolved with the FES high
waters, plus COAST-RP's TC part as a rate; per-RP generator hydrographs at every station.

- **QA** (fig17): four failure modes of a first build: double low waters taken as high waters,
  GTSM residual spikes, tropical cyclones counted twice, and heavy GPD tails. They were removed by
  prominent high waters only, despiking (15 Arafura stations fall back to COAST-RP), an IBTrACS
  mask at stations with a TC part, and a GPD shape cap of 0.35.
- **Against COAST-RP** (fig12, fig13): median +0.02 m at RP100 (10-90%: -0.23 to +0.22 m). The FES
  tide lowers the levels by 0.05-0.08 m and r raises them by 0.06-0.09 m. Large departures are
  regional: lower in the southern North Sea, higher in the western Baltic and the macrotidal
  basins where GTSM's tide is too small.
- **Against the gauges** (fig14; 618 gauges, r left one out): RP100 MAE 0.17 m against
  COAST-RP's 0.25 m at ETC gauges; the median bias is halved, also at TC gauges, which stay
  0.17 m low.
- **Hydrographs** (fig15): every peak equals the RP level.
- **Interpolation** (fig16): leave-one-out RP100 MAE 7.0 cm with production's 15 neighbours,
  5.6 cm with 3; the error grows with station spacing (2 cm < 5 km, ~20 cm > 50 km).
- Neighbours on one FES artefact are made consistent: the version rule is grown by 25 km, which
  adds 10 stations.

Production can read both export files unchanged; switching the catalog and config, regenerating
the SFINCS forcing and recalibrating fsf are listed in TODO.md.

## Figures

| Figure | Shows |
|---|---|
| `fig01_tide_mhws_scatter` | MHWS of GTSM, FES2022 and FES2014 against the gauges |
| `fig02_tide_mhws_ratio_maps` | the GTSM / gauge and FES2022 / gauge MHWS ratios as maps |
| `fig03_tide_constituent_ratios` | constituent amplitude ratios, GTSM and FES2022 against the gauges |
| `fig04_fes2022_vs_fes2014` | FES2022 / FES2014 MHWS at all boundary nodes, map and scatter with the rule |
| `fig05_fes_basin_heads` | FES2022 and FES2014 MHWS in the four basins, replaced nodes and gauges |
| `fig06_version_rule_sensitivity` | rule thresholds against the gauges, the known artefacts and the number of nodes changed |
| `fig07_coast_rp_vs_gauges` | COAST-RP minus gauge GEV by AR6 region, RP10 and RP100 |
| `fig08_profiles_by_rp` | storm-tide profiles by RP class, gauges and GTSM, with COAST-HG |
| `fig09_surge_by_rp` | surge-only profiles and surge durations by RP class, with COAST-HG's surge |
| `fig10_generator_acceptance` | acceptance test: T(0.25) and next-day high water by RP and tidal class |
| `fig11_generator_profiles` | median generated hydrographs against the gauges, HGRAPHER and COAST-HG |
| `fig12_level_product_maps` | RP100 level of the product minus COAST-RP, and the surge ratio r, as maps |
| `fig13_level_product_by_rp` | product minus COAST-RP by RP: total, FES tide alone, with r |
| `fig14_level_product_gauges` | COAST-RP and the product (r left one out) against the gauges, RP10 and RP100 |
| `fig15_level_product_hydrographs` | product hydrographs at Dunkerque, Kiel, Galveston and Hiron Point |
| `fig16_station_interpolation` | leave-one-out interpolation error by neighbours, power and station spacing |
| `fig17_level_product_qa` | the QA failure modes at three stations, skew surges before and after |

## References

- Archer, A. W. (2013). World's highest tides: Hypertidal coastal systems in North America, South
  America and Europe. Sedimentary Geology, 284-285, 1-25. https://doi.org/10.1016/j.sedgeo.2012.12.007
- Desplanque, C. and Mossman, D. J. (2001). Bay of Fundy tides. Geoscience Canada, 28(1), 1-11.
- Dullaart, J. C. M., Muis, S., Bloemendaal, N., Chertova, M. V., Couasnon, A. and Aerts,
  J. C. J. H. (2021). Accounting for tropical cyclones more than doubles the global population
  exposed to low-probability coastal flooding. Communications Earth & Environment, 2, 135.
  https://doi.org/10.1038/s43247-021-00204-9 (COAST-RP)
- Dullaart, J. C. M., Muis, S., de Moel, H., Ward, P. J., Eilander, D. and Aerts, J. C. J. H.
  (2023). Enabling dynamic modelling of coastal flooding by defining storm tide hydrographs.
  Natural Hazards and Earth System Sciences, 23, 1847-1862. https://doi.org/10.5194/nhess-23-1847-2023
  (COAST-HG, HGRAPHER)
- Hart-Davis, M. G., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M. and Seitz, F. (2021).
  EOT20: a global ocean tide model from multi-mission satellite altimetry. Earth System Science
  Data, 13, 3869-3884. https://doi.org/10.5194/essd-13-3869-2021
- Iturbide, M., Gutiérrez, J. M., Alves, L. M., Bedia, J., et al. (2020). An update of IPCC climate
  reference regions for subcontinental analysis of climate model data: definition and aggregated
  datasets. Earth System Science Data, 12, 2959-2970. https://doi.org/10.5194/essd-12-2959-2020
- Lyard, F. H., Allain, D. J., Cancet, M., Carrère, L. and Picot, N. (2021). FES2014 global ocean
  tide atlas: design and performance. Ocean Science, 17, 615-649. https://doi.org/10.5194/os-17-615-2021
- Medeiros, C. and Kjerfve, B. (1988). Tidal characteristics of the Strait of Magellan.
  Continental Shelf Research, 8, 947-960. https://doi.org/10.1016/0278-4343(88)90079-1
- Muis, S., Apecechea, M. I., Dullaart, J., de Lima Rego, J., Madsen, K. S., Su, J., Yan, K. and
  Verlaan, M. (2020). A high-resolution global dataset of extreme sea levels, tides, and storm
  surges, including future projections. Frontiers in Marine Science, 7, 263.
  https://doi.org/10.3389/fmars.2020.00263
- Wang, X., Verlaan, M., Veenstra, J. and Lin, H. X. (2022). Data-assimilation-based parameter
  estimation of bathymetry and bottom friction coefficient to improve coastal accuracy in a global
  tide model. Ocean Science, 18, 881-904. https://doi.org/10.5194/os-18-881-2022

The full reference list is in `INPUT_ANALYSIS.md` §12.
