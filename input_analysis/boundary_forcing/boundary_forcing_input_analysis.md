# Boundary forcing: input analysis summary

The coastal boundary of the GFM pipeline needs two things per station and return period (RP):
a **peak water level** and a **hydrograph**. The hydrograph forces SFINCS and sets the duration T
of the eikonal friction correction. This note summarises which global datasets can supply those
two things, how they compare with tide gauges, and what was decided. It condenses the full
analysis in `boundary_test/INPUT_ANALYSIS.md`; section references (§) and test numbers point there.

Everything is evaluated at GESLA 4.1 tide gauges, which serve as the reference. Model comparisons
(eikonal vs SFINCS on real tiles) are outside this note.

## Datasets and terms

| Dataset | What it is | Role |
|---|---|---|
| **GESLA 4.1** | observed water levels at 1,154 tide gauges with >= 20 y of record | reference |
| **GTSM v3** (CDS) | global hydrodynamic model: tide-only run (1996-2014) and ERA5-forced surge (1979-2024) | tide and surge behind COAST-RP |
| **FES2022** | ocean tide atlas, 34 constituents, 1/30 deg, assimilates altimetry and gauges | proposed tide source |
| **FES2014** | the previous FES version (`fes2014b_elevations_extrapolated`, 1/16 deg) | fallback where FES2022 fails |
| **EOT20** | empirical altimetry tide model, built as a residual on FES2014b | second-atlas check only; not used further |
| **COAST-RP** | storm-tide return levels RP 1-1000 y at 23,226 GTSM stations; extratropical (ETC) surge from GTSM-ERA5 plus tropical-cyclone (TC) surge from STORM synthetic tracks, combined with the GTSM tide | production peak level |
| **COAST-HG** | one RP100 storm-tide hydrograph per COAST-RP station (HGRAPHER method), average- and spring-tide variants | current hydrograph shape |

- **MHWS:** mean high water springs, as a height above local mean sea level (MSL). The main tide
  metric here; the spring range is about twice MHWS.
- **Proxy:** |M2|+|S2|+|K1|+|O1|, the summed amplitude of the four main constituents. It
  approximates MHWS on semidiurnal coasts and is cheap enough to evaluate at every boundary node.
- **T(f):** total time, over all tidal cycles in a ±72 h event window, that the water level stays
  within a fraction f of the peak (above MSL). T(0.25) is the T of the eikonal correction for a
  cell whose water depth is a quarter of the storm tide.
- **nb24:** the next-day high water as a fraction of the peak; it measures how much the
  neighbouring tides of a storm still flood.
- **TC-dominated station:** one whose COAST-RP TC-only RP100 exceeds its ETC-only RP100 (3,050 of
  23,226 stations).

How the comparison is made fair: every source is reduced to heights above its own local MSL
(annual means removed), fitted with the same 34-constituent harmonic method, predicted over the
same 18.61-year nodal cycle at 10 min, and reduced to MHWS with the same statistics code. Each
gauge is paired with its nearest GTSM station (median 1.9 km). The tide comparison uses 698
quality sites, where the harmonic fit explains >= 50% of the variance. The extremes and shape
comparisons also use a record-based set of 626 sites that includes the surge-dominated Baltic
and Mediterranean (test 1).

## 1. Which source represents the tide best?

**FES2022.** The comparison was made on the tidal levels (MHWS and the other datums), and on the
harmonic constituents behind them, all derived from each source's own time series or atlas
constants. It was not a comparison of raw water-level time series: those enter in §4, where
whole storm events are compared.

| MHWS against the gauge (698 sites) | Median ratio | Mean absolute error | Within 0.1 m |
|---|---|---|---|
| GTSM tide | 1.08 | 0.16 m | 53% |
| FES2022 | 0.99 | 0.05 m | 88% |
| FES2014 | 0.99 | 0.07 m | 84% |
| EOT20 | 1.00 | 0.11 m | 84% |

- **GTSM's tide is 7-8% too large** almost everywhere (Africa and the Mediterranean about 20%),
  and too small on the North American Atlantic and Gulf coast (Fig. 1, Fig. 2). FES2022 matches
  the gauges in every region.
- **The cause is in the constituents** (Fig. 3): GTSM's S2 is 20% and M2 6% too large, against
  0.99-1.00 for FES. Since the spring range is roughly M2 + S2, that excess is the MHWS excess.
  The likely source is GTSMv3's uncalibrated bathymetry and bottom friction (Wang et al., 2022).
- **It is not a location effect:** FES gives the same MHWS at the gauge and at the GTSM station,
  and GTSM's bias does not grow with the gauge-station distance. So the GTSM/FES ratio map
  reproduces the GTSM/gauge map, and FES can stand in for gauges where there are none.
- **Where FES still misses** (by more than 0.25 m at 20 of 698 sites; GTSM at 124): estuaries,
  fjords and narrow entrances, in both directions (upper St Lawrence, Long Island Sound, Salish
  Sea, Port Phillip entrance, Juneau).

The same tide excess sits inside COAST-RP, because COAST-RP is built on the GTSM tide (§3 here).

## 2. Where FES2022 is badly wrong, and how that was found

FES2022 is right in most macrotidal basins, but at a small number of basin-head nodes its MHWS
above MSL is as large as the whole observed spring range. Those values flood the eikonal tiles,
because the solver seeds every boundary cell with its own level.

**How the artefacts were found.** Not by comparison with GTSM, which is itself biased (25% too
low in the Bay of Fundy). The chain was:

1. **Model behaviour.** On calibration tile 891 (Minas Basin) the eikonal CSI against SFINCS fell
   from 0.57 to 0.27 with the FES forcing. Tracing it back led to FES nodes with MHWS of 14.5 m.
2. **Published tidal ranges.** These were checked against the literature: the Minas Basin mean
   spring range of 14.5 m and record of 16.3 m (Desplanque and Mossman, 2001; Archer, 2013), the
   9 m spring range of the Strait of Magellan (Medeiros and Kjerfve, 1988), and 7-10 m at the
   Colorado mouth. FES agrees with these over most of each basin and fails only at the heads.
3. **Gauges, where they exist.** Skagway and Juneau show FES2022 at 7.5 / 6.0 m against
   2.8 / 2.7 m observed, which exposed a fourth cluster in Glacier Bay and Lynn Canal.
4. **Second atlases** (test 6), as the systematic check below.

| Basin head | FES2022 proxy | FES2014 proxy | EOT20 proxy | Observed reference |
|---|---|---|---|---|
| Cobequid Bay (Minas Basin) | 12.4 m | 6.0 m | 6.1 m | spring range 14.5 m, MHWS ~7.3 m |
| Bahía Lomas (Strait of Magellan) | 15.7 m | 4.9 m | 5.0 m | spring range ~9 m |
| Colorado delta (upper Gulf of California) | 17.1 m | 3.6 m | none | spring range 7-10 m |
| Glacier Bay / Lynn Canal | 11.7 m | 2.3 m | 2.3 m | Skagway MHWS 2.78 m |

**The artefacts are FES2022's alone** (Fig. 5). FES2014 and EOT20 agree with each other and with
the observed ranges in all four basins, so the failure is FES2022's extension of the tide into
the basin heads, not a problem shared by the FES family. At the gauges FES2014 is slightly worse
than FES2022 overall (Fig. 4), so FES2022 stays the primary source.

**EOT20 is dropped.** It is a residual model on FES2014b, so where altimetry is sparse it repeats
FES2014, and where it departs from FES2014 it adds its own failures: solar constituents of 14-31
m in the St Lawrence files, Puget Sound and Trondheim far too low, no values in narrow inlets.

**No atlas catches the moderate errors.** At the 14 gauges where FES2022 is too high by more than
0.25 m and 25%, FES2014, EOT20 or GTSM is usually wrong in the same direction. Errors of 0.25-1 m
in fjords and inlets can only be corrected with gauges.

## 3. The FES version rule and its thresholds

**Rule.** At a boundary node, FES2014 replaces FES2022 when both hold on the proxy:

- FES2022 > R × FES2014, with **R = 1.5**, and
- FES2022 - FES2014 > D, with **D = 1 m**.

A gauge overrides both atlases where one exists. The ratio alone would fire on microtidal coasts
where both amplitudes are centimetres (Sea of Azov 5 vs 2 cm); the difference alone would switch
macrotidal coasts where FES2022 is only 20-40% higher. With both, 1,911 of the 89,628 boundary
nodes of the test tiles (2%, in 62 clusters of 2°) switch to FES2014 (Fig. 6).

**Sensitivity** (Fig. 7; R from 1.1 to 3, D from 0 to 2 m), against three kinds of evidence:

1. **Gauges, binned by how much FES2022 exceeds FES2014.** Below a difference of about 0.75 m,
   FES2014 is closer to the gauge at only 33-44% of sites in every bin: switching would make
   things worse, because FES2022 is the better atlas there. Above 1 m FES2014 is closer at all 5
   gauges, by a median of 1.43 m. This sets where D has to lie.
2. **Gauge outcome for every (R, D) pair.** No pair with D >= 1 m and R <= 2 makes any gauge
   worse (by more than 5 cm). Below 1 m harm starts with Trondheim, where FES2014 is 0.9 m too low
   and the proxy difference is 0.93 m. At the adopted values the rule fires at 4 gauges and fixes
   3: Skagway 7.46 → 2.76 m (gauge 2.78), Juneau 5.95 → 3.33 m (2.72), Port Alberni 2.79 → 1.36 m
   (1.28). The fourth, Little River, barely changes.
3. **Known artefacts.** These are the 169 basin-head nodes whose FES2022 MHWS exceeds 1.25 × the
   largest observed MHWS in their basin. The set is defined without FES2014, so the test is not
   circular. R <= 1.4 replaces all of them, R = 1.5 replaces 167, R = 1.75 93% and R = 2 81%;
   detection also drops once D >= 1.5 m.

**Why these values.** D is the threshold that matters, and the evidence bounds it to about
0.95-1.25 m: below, gauges are harmed; above, known artefacts start to be missed. 1 m is inside
that window, 7 cm above Trondheim. R only guards large-tide coasts: anything from 1.1 to 1.5
gives the same gauge result within one site and changes the number of replaced nodes by 10%.
1.5 is the cautious end of that range. It keeps FES2022 at 210 macrotidal nodes (Hecate Strait,
Frobisher Bay, Hudson Strait, the Amazon mouth) where FES2022 is 30-45% higher and no gauge
decides between the two. The cost is one gauge (Wyndham, where FES2014 would be 0.22 m better)
and 2 marginal Bahía Lomas nodes.

**Limits.** The gauge evidence for D rests on few sites (5 above 1 m). The upper bound therefore
leans on the known-artefact set.

## 4. Peak levels: COAST-RP against the gauges

COAST-RP return levels are compared with generalised extreme value (GEV) fits to the gauge annual
maxima (Fig. 8).

| | RP10 median difference | RP10 MAE | RP100 median difference | RP100 MAE |
|---|---|---|---|---|
| COAST-RP - gauge | -0.06 m | 0.22 m | -0.06 m | 0.37 m |

- **Two errors cancel.** COAST-RP's error tracks GTSM's tide excess (correlation 0.88). With the
  tide excess removed, COAST-RP is 0.11-0.12 m too low: its surge part is too low, and the
  too-large tide hides it. Replacing the tide with FES alone would therefore lower the levels and
  expose the surge deficit.
- **TC regions are low.** At TC-dominated stations COAST-RP is -0.19 / -0.24 m (RP10 / RP100)
  below the gauges, at ETC-dominated stations -0.04 / -0.08 m, and without a TC value +0.04 /
  +0.02 m. Gauge records of 30-40 years under-sample cyclones, so the TC deficit is a lower bound.
- **Same model, different statistics.** Where ETC surges dominate, a GEV on 46 years of GTSM and
  COAST-RP agree within a few cm. The statistics explain a third to two fifths of COAST-RP's MAE.
- **Regional:** the Baltic is -0.17 m at RP100 (MAE 0.49 m); the outer Bay of Fundy is 1.2-1.4 m
  too low. Interannual MSL variability adds only about 1 cm (test 3), so it needs no extra term.

The peak level stays COAST-RP for now. The planned replacement is a skew-surge convolution of
GTSM surge with the FES tide, with a gauge-calibrated surge correction (tests 2, 5 and 9).

## 5. Hydrograph shapes: COAST-HG, GTSM and the gauges

Events are the annual maxima at each gauge with ±72 h around the peak (21,100 GESLA and 32,100
GTSM events), normalised by their peak. Fig. 9 shows the median profiles per tidal range class.
The table compares the 3 largest events per site with COAST-HG at the paired station.

| Spring range | Source | nb24 | T(0.25) | RMSE vs gauge |
|---|---|---|---|---|
| < 1 m | gauge / GTSM / COAST-HG spring | 0.64 / 0.70 / 0.61 | 8 / 8 / 6 h | - / 0.22 / 0.25 |
| 1-2 m | | 0.82 / 0.86 / 0.73 | 11 / 12 / 5 h | - / 0.15 / 0.17 |
| 2-4 m | | 0.86 / 0.89 / 0.75 | 13 / 14 / 6 h | - / 0.13 / 0.14 |
| > 4 m | | 0.88 / 0.89 / 0.73 | 17 / 17 / 6 h | - / 0.10 / 0.12 |

- **GTSM reproduces the observed shapes** in every regime: same T, same neighbouring high waters,
  the smallest profile error. Per site, the time above MSL correlates with the gauge at Spearman
  rho 0.87 with a median ratio of 1.00.
- **COAST-HG's central peak is right, but its neighbouring high waters are too low** (nb24 0.1-0.15
  of the peak lower), and the average-tide variant is lower still. Its T(0.25) of 5-6 h against
  11-17 h observed in meso- and macrotidal regimes means 2-3× too much eikonal friction.
- **Durations depend on the tidal regime** (Fig. 11). Where the spring range is above 1 m, the
  level drops below MSL every tidal cycle whatever the surge does (7-8 h contiguous above MSL).
  Where it is below 1 m, events last much longer: 45 h median on the record-based set, 122 h in
  the Baltic and 90 h in the Mediterranean, both cut off by the ±72 h window.

### Shape against return period

Rarer events are more surge-dominated: from RP < 2 y to RP >= 10 y the surge share of the peak
rises from 0.36 to 0.63 (range < 1 m) and from 0.05 to 0.13 (> 4 m). Their tidal neighbours are
relatively lower, so T falls (Fig. 10; empirical RP from the rank among each site's annual
maxima):

| Spring range | GESLA T(0.25) at RP <2 / 2-5 / 5-10 / >=10 y | GTSM, same |
|---|---|---|
| < 1 m | 14 / 14 / 13 / 9 h | 13 / 13 / 12 / 10.5 h |
| 1-2 m | 16 / 15 / 13 / 11 h | 16 / 15 / 14 / 13 h |
| 2-4 m | 17 / 16 / 14 / 14 h | 17 / 16 / 16 / 15 h |
| > 4 m | 22 / 21 / 20 / 17 h | 22 / 22 / 21 / 19 h |

- **The shape does not settle within the observable range.** T(0.25) is still falling into the
  top class (RP >= 10 y) in three of four regimes; only the gauges of the 2-4 m regime are flat
  from RP 5-10 on.
  T(0.5), which spans several tidal cycles, changes little above a 2 m range (39-42 h throughout).
- **The records cannot answer the question above RP ~30.** The top class is open-ended and, with
  30-46 years of record, holds events of roughly RP 10-45. Whether shapes converge above that is
  open, and a per-RP shape selection avoids having to assume it.
- **Consequence for the pipeline today:** using COAST-HG's RP100 shape for every RP gives
  frequent events (RP 2-5) a T that is 15-55% too short, on top of COAST-HG's own short T.

### Tropical-cyclone stations

GTSM-ERA5 does not resolve tropical cyclones, so its events cannot supply TC shapes. At
TC-dominated gauges the largest observed events are surge-dominated (surge share 0.54) and
shorter (surge above half its maximum for 15 h against 18 h elsewhere). Their T(0.25) of 6 h is
close to COAST-HG's 5 h, so COAST-HG is closest to observations at TC stations and furthest
elsewhere. Gauge shapes could be used at TC stations, scaling only the surge part (observed minus
predicted tide) to the target peak. Two limits make that a local option, not a global one: only
30% of TC stations have a gauge within 50 km, and the scale factors spread from 0.6 to 1.9.

## 6. Conclusion on the hydrograph: build it from GTSM per RP

Retire COAST-HG for both roles and generate one hydrograph per station and RP (§8.3, the "Stage
A" generator, after HGRAPHER-RP in GCFM_UUdev):

- **Tide:** FES2022 at the station, ±72 h around a spring high water, with the FES2014 version
  rule at basin heads.
- **Surge shape:** selected for that RP from the GTSM surge events of 1979-2024 at the station,
  built on the skew surge where the tidal range exceeds ~1 m. At RPs where COAST-RP's TC part
  dominates, a STORM-forced GTSM shape (test 10), with a Gaussian of 8 h half width as the
  placeholder until those runs are available. Gauge surge shapes are a local option near gauges.
- **Peak:** the surge is scaled so that tide + surge reaches the RP peak level (COAST-RP for now;
  the input can later take the tide-corrected level of test 5 without code changes).

Neither GESLA nor regional pooling supplies the shapes. GESLA is the reference the generator is
scored against, but it covers too few stations: in particular only 30% of TC stations have a gauge
within 50 km. Regional pooling was not analysed for shapes; it appears only as an option for
the surge-deficit correction of the peak level (test 9).

**Acceptance test before production:** at RP2 and RP10-100, the generator's median T(0.25), T(0.5)
and nb24 must lie within ±15% of the gauge medians of the matching RP class and tidal regime, its
profile error must not exceed that of GTSM's own events, and it must reproduce the 1.2-1.6
T(0.25) ratio between RP classes. Test 7 (splitting COAST-HG's error into its tide and surge
parts) comes first, because it decides which parts of the HGRAPHER method the generator can keep.

## 7. What this means for the eikonal duration correction

The friction multiplier is m = clip((t_ref / T)^α, 1, 10). A longer T gives a smaller m, less
friction and a longer reach. On 540 synthetic 1-D SFINCS floodplains, with the friction factor
fsf refitted per form, m ∝ T^-α without a lower clip fits best (log-extent RMSE 0.140, against
0.148 for the production form); α = 0.5 and 1 perform alike, and a constant t_ref is absorbed into
fsf. The agreed setting is α = 1 with fsf calibrated on SFINCS. T must come from hydrographs with
realistic neighbouring high waters and an RP-appropriate shape, which is what §6 provides.
Confirmation on 5-10 real 2-D tiles follows once the generator exists (test 8).

## 8. Decisions and open items

**Decided:**
- Tide: FES2022, FES2014 where the version rule fires, gauges overriding both. EOT20 not used.
- Hydrograph: per-RP generator from GTSM surge shapes on the FES tide, scaled to the RP peak;
  COAST-HG retired once a variant passes the acceptance test.
- Duration correction: α = 1, fsf calibrated on SFINCS.
- Zero-SLR scenario at the 2020 sea level: the AR6 rise from 1995-2014 to 2020 (0.049 m,
  regionalised with the SLR fingerprints) is added to the boundary level.

**Open:**
- **The FES version rule is not implemented in the boundary preparation yet.** That needs
  FES2014 unpacked on the project drive and a data catalog entry.
- **How a gauge overrides the atlases** spatially (over which distance, how blended) is not
  defined.
- **Tests not yet run:** test 7 (COAST-HG error split), tests 2+5 (skew-surge convolution, one
  estimator three ways), test 9 (surge-deficit correction), test 10 (STORM-GTSM TC shapes; needs
  the STORM-forced runs), test 8 (real tiles, after the generator).
- **Deferred:** test 4 (absolute MSL + MDT against geodetic gauge datums); wave setup.

## Figures

All in `figures/`, copied from the analysis output (`.cleanup_baseline/level_sources/analysis/`).
The script in brackets regenerates each one.

| Figure | Shows | Source |
|---|---|---|
| `fig01_tide_mhws_scatter.png` | MHWS: GTSM vs gauge, FES2022 vs gauge, GTSM vs FES2022 | `levels_scatter.png` (`levels_analysis.py`) |
| `fig02_tide_mhws_ratio_maps.png` | the same three MHWS ratios as maps | `levels_ratio_maps.png` (`levels_analysis.py`) |
| `fig03_tide_constituent_ratios.png` | constituent amplitude ratios GTSM and FES vs gauge | `levels_constituents.png` (`levels_analysis.py`) |
| `fig04_atlases_at_gauges.png` | FES2022, EOT20, FES2014 MHWS vs gauge, and per constituent | `atlas_gauges.png` (`atlas_check.py`) |
| `fig05_fes_basin_heads.png` | proxy at the boundary nodes in four basins, FES2022 / EOT20 / FES2014 | `atlas_basins.png` (`atlas_check.py`) |
| `fig06_fes2022_vs_fes2014.png` | FES2022 / FES2014 ratio map and per-node scatter with the rule boundary | `atlas_versions.png` (`atlas_check.py`) |
| `fig07_version_rule_sensitivity.png` | R × D sensitivity: gauge bins, gauge outcome, known artefacts, replaced nodes | `atlas_rule_sensitivity.png` (`atlas_rule_sensitivity.py`) |
| `fig08_coast_rp_vs_gauges.png` | COAST-RP minus gauge GEV by region, tide set and record-based set | `sets_extremes.png` (`site_sets.py`) |
| `fig09_hydrograph_profiles.png` | median normalised event profiles: gauge, GTSM, COAST-HG spring and average | `shapes_profiles.png` (`shapes_analysis.py`) |
| `fig10_profiles_by_rp.png` | median profiles by empirical RP class, gauge and GTSM, with COAST-HG | `shapes_profiles_by_rp.png` (`shapes_analysis.py`) |
| `fig11_event_durations.png` | distributions of time above MSL and T per event | `shapes_durations.png` (`shapes_analysis.py`) |

Scripts are in `boundary_test/` and run in the order of `INPUT_ANALYSIS.md` §10. The per-site
data behind the tables are in `boundary_test/input_analysis_data/`.

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
