# Boundary test: implementation notes (step 0)

Plan: `handoff_tide_corrected_duration_boundary.md`. Outputs: `.cleanup_baseline/boundary_test/`
(local; P: stays read-only). Nothing here changes the production DAG.

## 0. Analysis plan: sources first, then shapes, then model comparisons

No eikonal-vs-SFINCS comparison is designed until steps 1 and 2 are done; the runs in §2.20 are
not used for decisions. Scope: global, at the GESLA 4.1 gauges (1,154 sites with >= 20 y record,
`gesla_sites.parquet`). Outputs: `.cleanup_baseline/level_sources/`.

1. **Water-level sources: GESLA, GTSM, FES2022, COAST-RP, COAST-HG (+ MDT).**
   - 1a Reference level of each source: what 0 m is, whether MSL is zero, whether MDT or SLR
     is in it; checked against the data, not only the metadata.
   - 1b Tidal levels on one basis: MSL, MHWS, MHW, MLW, MLWS, HAT, LAT, range. Same harmonic
     constituents, same 18.61-y prediction epoch and the same `tide_statistics` code for
     GESLA (harmonic fit), GTSM tide (fit and raw) and FES2022 (atlas constants).
   - 1c Extremes: COAST-RP RP levels vs GEV fits to GESLA and GTSM annual maxima.
   - 1d Where the differences are (regime, coast type, distance to the model point) and which
     explanation each pattern supports.
2. **Hydrograph shapes: GESLA vs GTSM vs COAST-HG.**
   - 2a Events per gauge from observed and modelled total water level; duration above MSL
     (MDT-consistent local MSL), above MHW/MHWS and within h of the peak.
   - 2b Does the shape depend on event size (annual maxima ranks, RP)?
   - 2c Does the water stay above MSL longer than 12 h? If so, B's fixed t_ref = 12 h is not
     a meaningful reference, and a reference tied to the hydrograph itself (or to the
     calibration hydrographs) is derived and tested on synthetic cases.
3. Model comparisons, designed from 1 and 2 (not started).

Scripts, in run order: `levels_pairs.py`, `extract_gtsm_series.py --stations-csv`,
`levels_gesla_extract.py`, `levels_tides.py`, `levels_analysis.py`, `levels_extremes.py`,
`shapes_events.py`, `shapes_analysis.py`, `duration_experiment.py`, `duration_analysis.py`.
Quality sites (698 of 1,154): harmonic fit over >= 3 y explaining >= 50% of the variance, GTSM
station within 10 km.

### 0.1 Reference levels (1a)

| Source | Content | 0 m is | MSL in the data | MDT | SLR / trend |
|---|---|---|---|---|---|
| GESLA 4.1 | observed total water level | gauge datum (chart datum, national datum; 311 sites "Unspecified") | arbitrary offset | no | relative SLR incl. land motion, median 3.0 mm/yr (1996-2014, 708 sites, 5-95%: -2.8..8.6) |
| GTSM v3 tide (CDS, 10 min, 1996-2014) | "astronomic tide, self-attraction and loading, radiational tides and mean sea-level" (file attribute) | model MSL, apparently 1986-2005 (to confirm from the dataset documentation) | mean +0.02 m over 1996-2014 (5-95%: 0.00..0.05) | no (barotropic) | yes: median 2.6 mm/yr (1.2..4.4) imposed in the tide run |
| GTSM v3 surge (CDS, hourly, 1979-2024) | water level - tide | relative to the tide run | mean -0.04..+0.07 m per station | no | none (0.1 mm/yr) |
| FES2022b ocean tide | harmonic constants, 34 constituents | zero-mean by construction | 0 | no | no. Sa/Ssa present but ~0 (0.000-0.004 m at sample gauges). Ocean tide, not geocentric: what a gauge measures |
| COAST-RP v1 | storm tide RP levels (GTSM ERA5 1979-2018 + STORM TC) | GTSM model MSL (no datum attribute in the file) | - | no (pipeline adds MDT) | not documented; at most a few cm over 1979-2018 |
| COAST-HG RP100 | storm-tide hydrograph, 149 h at 10 min | same as COAST-RP: the peak equals COAST-RP RP100 exactly at all 23,226 stations | tails +0.47 m (average tide) / +0.60 m (spring) = HGRAPHER high water | no | no |
| MDT CNES-CLS22 hybrid | mean dynamic topography | geoid | epoch 1993-2012 | - | - |

Consequences: the sources do not share an MSL. Every comparison here removes each calendar
year's mean from GESLA and from the GTSM tide (local MSL of that year), so datum, SLR trend and
interannual MSL drop out; FES is zero-mean. In production the pipeline adds MDT (1993-2012
epoch) to COAST-RP (GTSM model MSL) and compares with DeltaDTM on GOCO06s: the epochs differ by
roughly a decade of SLR (order 5 cm). Not checked here: absolute MSL against MDT at gauges with
geodetic datums (needs each national datum's offset to GOCO06s).

### 0.2 Tidal levels (1b, 1d), 698 quality sites

| Comparison | MHWS median ratio | MHWS MAE | within 0.1 m | spring range ratio |
|---|---|---|---|---|
| GTSM (raw series) vs GESLA | 1.07 | 0.16 m | 53% | 1.08 |
| GTSM (harmonic fit) vs GESLA | 1.08 | 0.16 m | 53% | 1.08 |
| FES2022 at the gauge vs GESLA | 0.99 | 0.05 m | 88% | 0.99 |
| FES2022 vs GTSM at the GTSM station | 0.92 | 0.16 m | 52% | 0.92 |
| FES2022 at the gauge vs at the station | 1.00 | 0.01 m | 98% | 1.00 |

- **GTSM's tide is ~7-8% too large; FES2022 matches the gauges.** The earlier "FES lower than
  GTSM" (§2.16) is GTSM being high, not FES being low. HAT behaves the same (GTSM 1.03-1.05).
- **Where:** GTSM is high almost everywhere (Mediterranean 1.18, Africa/Indian Ocean 1.19-1.20,
  NW Europe 1.08-1.09, Baltic 1.10, S America 1.10, Japan 1.06), low on the N American Atlantic
  and Gulf coast (0.85), right on the N American Pacific coast (1.01). Independent of the tidal
  range class (1.06-1.09) and of the gauge-station distance (1.03-1.09 for 0-1 km up to >10 km).
- **Not location:** FES at the gauge equals FES at the GTSM station (ratio 1.000), and GTSM's
  bias does not grow with distance. **Not method or sampling:** raw GTSM and its harmonic fit
  agree (1.07 vs 1.08), and GESLA, GTSM and FES go through the same fit/prediction/statistics.
- **Which constituents:** median amplitude ratio GTSM / GESLA: M2 1.06, S2 1.20, N2 1.03,
  K1 0.97, O1 1.10, M4 1.02, MS4 1.28. FES / GESLA: all 0.99-1.00 except M4 0.94, MS4 1.04.
  The spring range is ~M2 + S2, so S2 20% and M2 6% too large give the 7-8% in MHWS. A plausible
  cause is GTSM's radiational/SAL treatment of S2 (the file lists radiational tides); FES2022
  assimilates altimetry and gauges, which constrain S2 directly. Not verified beyond this.
- **FES's failures are local:** 20 of 698 sites miss MHWS by > 0.25 m (GTSM: 124), all at fjord
  or estuary heads and narrow entrances: Juneau 5.95 vs 2.72 m, Fulford Harbour 1.99 vs 0.89,
  Queenscliff/Point Lonsdale (Port Phillip entrance), and too low in the upper St Lawrence
  (Saint-Francois 1.27 vs 2.87). This is the basin-head failure of §2.19, in both directions.
  The FES native/extrapolated mask does not flag these points.
- Implication for H1: replacing GTSM's tide by FES's is supported by the gauges on open and
  most estuarine coasts (FES within 5 cm MAE); at fjord/estuary heads neither source is reliable
  and a gauge- or plausibility-based rule is needed.

### 0.25 Extremes (1c): COAST-RP vs GEV fits to annual maxima, 648 quality sites

GESLA GEV: annual maxima of the record with yearly means removed (median 30 years); GTSM GEV:
46 years of GTSM surge + GTSM harmonic tide; 90% interval from 100 bootstrap fits.

| Comparison | RP10 median diff | RP10 MAE | RP100 median diff | RP100 MAE |
|---|---|---|---|---|
| COAST-RP - GESLA GEV | -0.06 m | 0.22 m | -0.06 m | 0.37 m |
| GTSM GEV - GESLA GEV | -0.06 m | 0.20 m | -0.11 m | 0.37 m |
| COAST-RP - GTSM GEV (same model, different statistics) | -0.01 m | 0.07 m | +0.02 m | 0.15 m |

- COAST-RP's error at RP10 follows GTSM's tide error (correlation 0.88 with the MHWS excess of
  §0.2). Without the tide excess, COAST-RP is 0.11 m (RP10) / 0.12 m (RP100) too low: the surge
  part is too low on most coasts and the too-large tide hides it. Only 18% (RP10) / 44% (RP100)
  of COAST-RP levels lie inside the GESLA 90% interval.
- By region (RP100): NW Europe +0.19 m (tide excess dominates), N America Atlantic+Gulf -0.29 m,
  Japan/E Asia -0.17 m (tropical cyclones with 30-46 y records on both sides), others within 0.1 m.
- Implication for H1: replacing only the tide removes the excess but keeps the surge deficit,
  so on most coasts H1 lowers levels that are already ~0.1 m too low against gauges. A tide
  correction needs the surge part assessed with it.

### 0.3 Event shapes and durations (2a, 2b), 21,100 GESLA and 32,100 GTSM annual maxima at 698 sites

Durations (hours; local MSL = 0; B's T = time within f x (peak - MSL) of the peak over +-72 h):

| Source | above MSL, contiguous | above MHW | above MHWS | B's T, f = 0.25 | f = 0.5 | surge above half peak |
|---|---|---|---|---|---|---|
| GESLA annual maxima | 9 (34% > 12 h) | 6 | 5 | 15 (68% > 12 h) | 39 (97%) | 14 |
| GTSM annual maxima | 8 (27%) | 5 | 4 | 15 (69%) | 38 (98%) | 15 |
| COAST-HG RP100, average tide | 33 (57%) | 7 | 5 | 5 (3%) | 21 (85%) | - |
| COAST-HG RP100, spring tide | 9 (35%) | 6 | 5 | 5 (1%) | 24 (90%) | - |

- **Above MSL longer than 12 h:** depends on the tidal regime. Contiguous time above MSL,
  median: spring range < 1 m 20 h (41 h for events with empirical RP >= 10 y), 1-2 m 8 h, 2-4 m
  7 h, > 4 m 7 h. Where the tide is large, the level drops below MSL every tidal cycle; the
  contiguous time above MSL is bounded by the tide, not the surge. Where it is small, the surge
  holds the level above MSL for one to several days.
- **GTSM reproduces the observed shapes** per site (Spearman rho, median ratio GTSM/GESLA:
  time above MSL 0.87 / 1.00, B's T at f = 0.25 0.77 / 1.00, surge duration 0.63 / 1.14).
- **COAST-HG does not:** its peak is ~3x narrower than observed (B's T at f = 0.25: 5 h vs 15 h),
  and its neighbouring high waters are lower (average-tide variant 0.5-0.6 of the peak at
  +-12/24 h, observed 0.75-0.85 in macrotidal regimes: annual maxima happen at spring tides,
  so several consecutive high waters are nearly as high). The difference is not the RP: within
  a site, B's T at f = 0.25 falls only from 16 to 11 h between RP < 2 y and RP >= 10 y events.
- **Event size:** larger events stay above MSL longer (rho 0.14-0.25) and their surge lasts
  longer (surge dur50 12 -> 18 h at 1-2 m range), but their peak is relatively narrower (B's T at
  f = 0.25 falls with RP, rho -0.1 to -0.24). Figures: `analysis/shapes_profiles.png`,
  `analysis/shapes_durations.png`.

### 0.4 What duration does to inland levels (2c), synthetic SFINCS floodplains

`duration_experiment.py`: 1-D strips, shore crest at MSL + 0.5 m, slopes 0 / 1e-4 / 3e-4 /
1e-3, Manning 0.03 / 0.06 / 0.12, peak 2.0 m; hydrographs: steady, 20 synthetic (tide amplitude
0-1.5 m x surge duration 2-48 h) and 24 observed GESLA events; SFINCS v2.3.0, 540 runs.

- **SFINCS: duration matters only where the terrain allows a long reach.** On 1e-3 (1.5 km to
  the 2 m contour) every hydrograph floods to the contour with no level loss. On 1e-4 (15 km)
  the reach is 4.1 km for a 2 h surge and 12.6 km for a 48 h surge; with 1.5 m tide the 48 h
  surge reaches 7.5 km. Steady forcing fills to the peak level everywhere: in SFINCS the level
  loss with distance is entirely a duration effect.
- **Best single predictor of the SFINCS reach** (relative to the steady reach) is the
  contiguous time above the shore crest (Spearman 0.58-0.65 on the slopes where duration
  matters), better than B's cumulative T or the excess volume (0.27-0.48).
- **Eikonal:** its level falls linearly with distance at friction x fsf for every hydrograph,
  including steady forcing (extent 0.50 of SFINCS's at fsf 12, 0.65 at 7.5). B, with its
  multiplier clipped to [1, 10], can only add loss: it sharpens the fall for short peaks but
  cannot remove the loss for long events. Median eikonal/SFINCS extent ratio over all runs:
  no B (fsf 12) 0.69, B t_ref 12 h 0.65-0.77, B t_ref = time above the shore crest ("longest
  exposure") 0.58-0.60, B t_ref = T(head) of the median COAST-HG hydrograph ("relative to the
  calibration hydrograph") 0.64-0.82, the best of the four; level RMSE ranks the same.
- **Answer on t_ref:** a fixed 12 h is not a physical reference: T is window-dependent
  (cumulative over the hydrograph) and the water stays above MSL > 12 h in a third of observed
  events, so the absolute value only fixes where the clip at 1 starts. Scaling by the
  hydrograph's own longest exposure is worse: it makes every hydrograph's correction depend only
  on its shape, so a 2 h and a 48 h event of the same shape get the same friction, while SFINCS
  shows the absolute duration sets the reach. The reference that does best is the T of the
  calibration hydrograph at the same head, because the friction factor was fitted on SFINCS runs
  forced with COAST-HG (70 h runs): m = (T_calib(head) / T(head))^alpha. It reduces to 1 for the
  calibration forcing and, unclipped below 1, would also lower friction for events longer than
  the calibration ones; the strips suggest that is needed (SFINCS has no loss for long events).
  Caveats: uniform 1-D strips, uniform friction, no channels or sills; only the ranking and the
  mechanisms transfer, not the ratios.

## 1. What the branch does today (duration correction B)

- **Form:** friction x m, with m = clip((t_ref / T)^alpha, 1, 10), alpha = 1, t_ref = 12 h,
  production `friction_scale_factor` 7.5. Config: `simulation.flooding.duration_correction`.
  Code: `src/event_volume.py` `DurationCorrection`, `duration_friction_multiplier`.
- **Input is a duration per wet cell, not a factor.** First solve, then per wet cell
  head = local eikonal level - spill level (priority flood from sea + seeds, Barnes 2014).
  T = time the forcing hydrograph stays within `head` of its peak
  (`event_volume.time_within_peak`). Then a second solve with friction x m.
- **Hydrograph source:** COAST-HG RP100 `hydrograph_average_tide_signal` (10-min steps). Each
  cell uses its nearest station (haversine, on a 32-cell coarse grid). The same shape is used
  for every RP and SLR, and the datum is irrelevant because only the shape near the peak is used.
- **Tests:** `tests/event_volume_validation/validate_event_volume.py` (synthetic sill basin:
  B dries it under a short peak, identity for long events).
- **Calibration:** 498 random SFINCS RP100 tiles. Pooled CSI 0.742 without B (fsf 12),
  0.763 with B (fsf 7.5). Median tile unchanged; the gain comes from overflooded basins.
- **Interface for H2:** `DurationCorrection(station_lonlat, series, dt_s, ...)` takes any set of
  (location, hydrograph). The per-boundary-cell hydrographs of plan §5.E plug in as "stations"
  without a solver change. The along-water nearest lookup (plan §5.E) would replace
  `station_index`.

## 2. Where the plan is wrong or underspecified

1. **T(z_cell) (plan §5.E) is not what B uses.** B uses the time within (local level - spill
   level) of the peak. That accounts for attenuation and for sills: behind a sill the
   controlling elevation is the crest, not the cell ground. A T(z) lookup on absolute ground
   elevation would ignore both. Decided: keep B's definition; the T(z) lookup files are not
   written, B evaluates the stored hydrographs. A check against T(z_cell) is in TODO.md.
2. **H1 needs new-forcing SFINCS runs.** Against the existing SFINCS runs, which use COAST-HG
   shifted to COAST-RP, H1 changes the eikonal forcing but not SFINCS, so macrotidal tiles
   disagree by construction; those scores are labelled stand-in. New-forcing runs are made
   locally (`boundary_test/sfincs_local.py`, SFINCS v2.3.0 Windows build in
   `C:/Users/schlumbe/hydromt_sfincs/delta_model/software/`): the existing tile model with only
   sfincs.bzs shifted by L_RP100 - production level of the nearest boundary seed, the same shift
   as the eikonal H1 seeds. The local build with the original bzs reproduces the HPC v2.4.0 hmax
   exactly on tile 3758 (same 1,784 wet cells, max depth difference 5e-7 m), so the HPC run is
   the control. ~15 min per tile at 2 threads: 50 tiles locally (`sfincs_tiles_local.csv`).
3. **FES2022 data:** the plan names `ocean_tide_extrapolated` (34 constituents). On P: there
   are only `load_tide` (8 constituents, not needed: solid-earth loading) and the unstructured
   mesh `FES2022b_OceanTide_NSgrid.nc` (downloading). pyTMD reads the gridded files.
4. **GTSM hourly:** only `storm_surge_residual` (variable `surge`, 1950-2024) is on P:. There is
   no `waterlevel`, so "tide = waterlevel - surge" (§5.B) cannot be computed. The CDS
   `tidal_elevation` variable (tide-only run; historical / 10-min / v1, <= 2014) gives the tide
   directly; `preparation/sync_gtsm_water_levels.py` now requests it for 1996-2014.
   `total_water_level` under experiment=historical is GCM-forced, not ERA5, so it must not be
   combined with the reanalysis surge.
5. **MHWS definition:** "within +-36 h of each monthly spring maximum" is ambiguous, since springs
   come twice per lunation. Implemented: per half-lunation (14.77 d) block
   (`src/tide_statistics.py`). For diurnal tides the block maximum is the tropic maximum
   (13.66 d cycle); a block needs >= 10 high waters (a diurnal block has ~15).
   High-water values are parabola-refined crests. Tide series must be 10-min, not hourly:
   at a 4.3 m MHWS GTSM station (NW Australia) hourly sampling with parabolic or 5-7 point
   refinement still misses crests by 2-8 cm (shallow-water constituents) and picks the wrong
   spring high water, giving MHWS 7-10 cm low. FES is therefore predicted at 10 min too.
6. **TC identification (decision 1) is settled by the data:** `COAST-RP_TC.nc` / `COAST-RP_ETC.nc`
   exist. Station is TC-dominated when TC RP100 > ETC RP100 (3,050 of 23,226). Open: the
   Gaussian width, implemented as half width at half maximum = `tc_half_width_hr`
   (dur50 = 2 x 8 h = 16 h).
7. **GESLA metadata mis-parse:** the header lacks a comma between the last two names, so
   default pandas puts longitudes under LATITUDE. Read with explicit names
   (`boundary_test/gesla_inventory.py`). `NUMBER OF YEARS` is the span, not flag-1 years.
8. **Fundy:** COAST-RP has no station east of 64.76 W (no Minas Basin, no Chignecto head). The
   GTSM spring high water at Saint John is 2.4 m vs about 4 m real, so H1's
   premise holds well before the head. Calibration tiles 891 (into Minas Basin, CSI 0.54) and 575
   have SFINCS runs; the automatic strata put 891 in mesotidal (median station on the Atlantic
   side), so both were added to the estuary stratum by hand.
9. **Held-out regions** must be excluded from the test tiles explicitly: an unfiltered draw
   picked SF Bay (728), Scotland (2450) and the Wash (2490).
10. Plan §0.2 says "§7" for the open decisions; they are in §8.
11. **Not yet on disk:** Daigle 2020 PDF, CAN-EWLAT bundle, UK CFB 2018
    (`validation/{NewBrunswick,England}/new_inputs/boundary_forcing` do not exist). GCFM_UUdev is
    not cloned locally; read-only from its public GitHub copy.

12. **F6a result (decision 3).** 1,085 GTSM stations around the 20 test tiles, 1950-2024,
    in `.cleanup_baseline/boundary_test/F6a_surge_duration_vs_peak.png` and
    `surge_duration_summary.csv`:
    - dur50 rises with the surge peak only for small events. Above about 0.2-0.4 m it levels
      off at 10-60 h, and at several stations it falls for the largest events.
    - A log-linear tau(peak) fit over all events is driven by the small, short events and
      extrapolates to absurd values (tau(RP100) 100-23,000 h). The pooled shape with a fitted
      log-linear time scale (plan §5.E.2) is therefore not viable as specified.
    - HGRAPHER-RP's per-RP selection, with the 10 largest events above the record, follows the
      plateau (median dur50 of the top 10 events: 6-61 h).
    - Recommendation: keep per-RP selection. If a pooled shape is wanted, use a constant tau
      from the upper-tail events, not a fit over all events.
13. **Annual MSL in the surge residual (decision 4): essentially absent.** The spread of the
    annual means is 5-40 mm, with trends within +-0.3 mm/yr. Removing the 1-yr running mean
    stays as a harmless safeguard.
14. **The COAST-HG spring-tide proxy is not a usable MHWS for S_RP.** At macrotidal tiles
    3583, 805 and 1423 the RP1-2 surge target (COAST-RP - proxy) already lies above every
    recorded GTSM surge. The proxy, the highest tide in the hydrograph tails, underestimates
    GTSM's MHWS. S_RP needs MHWS from the GTSM tidal_elevation series (§5.B); the proxy is used
    only for stratification.
15. **Test tiles: representative set, not only the 20 stratified picks.** 20 tiles show the
    direction per stratum but give no pooled estimate. Eikonal runs use the 500 random tiles of
    the representative calibration (R1/R2) outside the held-out regions plus the stratified
    picks: 496 tiles (212 meso, 125 micro, 82 macrotidal estuary, 56 TC, 11 macrotidal open,
    10 macrotidal other), `test_tiles_representative.csv`. The 20 picks stay the first 20 of the
    50 local SFINCS tiles; the other 30 are drawn round-robin over strata.
    GTSM series for ~10,800 stations are stored as int16 mm on disk (22 GB tide, 9 GB surge
    1979-2024), see `extract_gtsm_series.py`.
16. **MHWS_FES vs MHWS_GTSM (F4, first 50 tiles, 1,483 stations; against gauges see §0.2: GTSM
    is the one that is off): FES is lower almost
    everywhere.** Median MHWS_FES - MHWS_GTSM: -0.07 m below 1 m MHWS, -0.13 m at 1-1.5 m,
    -0.22 m at 1.5-3 m, -1.2 m above 3 m (median ratio 0.87-0.90). Not a datum effect (GTSM tide
    mean +0.02 m) and not sampling (both 10-min, same `tide_statistics`). GTSM is 2-3x FES in the
    western Baltic and on the Siberian/Alaskan Arctic shelf, and lower in the Gulf of Mexico and
    Bohai. Against approximate gauge MHWS (chart datum to MSL, +-0.2 m), FES is within 0.1-0.2 m
    at Brest, Dover, Avonmouth, Cuxhaven and Kiel (0.07 vs GTSM 0.21 m at Kiel); at Saint John it
    is 0.4 m low (GTSM 1.5 m low). So H1 lowers the boundary level on most coasts (median shift
    -0.1 to -0.25 m on the macrotidal tiles, -0.05 to -0.1 m microtidal) and raises it where
    GTSM under-resolves the amplification (tiles 1800, 1517, 104, 575: +0.1 to +0.9 m).
    `fes_tide_stats.py` predicts 94% of the boundary-cell nodes; the rest are inland nodes
    > CUTOFF_KM from a valid FES cell and keep the production level.
17. **Tile 1624** fails `calibration_experiments.load_tile` (its SFINCS grid does not nest into
    its subgrid); it is skipped everywhere.
18. **TC stations:** at tile 2146 (Japan) the RP10-1000 surge targets are 0.6-2.2 m, against
    ERA5-driven GTSM peaks of at most 0.5 m. This confirms the plan's point that ERA5 events do
    not contain the storms that set TC return levels.
19. **FES2022 is up to 2x too high at the heads of some macrotidal basins.** The native grids
    (mask 0, not extrapolated) give M2 amplitudes of 10.7 m in Minas Basin, 11.5 m in Bahia Lomas
    (Strait of Magellan) and 13.7 m in the upper Gulf of California. The real values are about
    5-6 m, 4-5 m and 2.5-3 m. Resulting MHWS: 14.5 m, 16.3 m, 15.4 m. Gauges on open macrotidal
    coasts are fine (Brest M2 2.19 m, Avonmouth MHWS 6.35 m). The native/extrapolated mask does
    not separate these nodes. 29 of 494 tiles have a boundary cell whose H1 level is more than
    2 m above production. These cells are 0.9% of all boundary cells; 2.5% of cells rise by more
    than 1 m. The analysis reports every score with and without these tiles
    (`analyse_boundary_test.py`, `SUSPECT_SHIFT_M`). A QA rule for MHWS_FES is open (§4).
20. **First results (206 of 496 eikonal tiles, 39 of 50 local SFINCS H1 tiles; batches stopped
    early).** These are pooled CSI and bias per stratum without the FES-suspect tiles, from
    `.cleanup_baseline/boundary_test/analysis/scores_*.csv`.
    - Against the existing SFINCS runs (stand-in, old levels), 190 tiles: the baseline is the
      current calibration (CSI 0.71-0.90, bias 0.90-1.19). H1 lowers CSI in mesotidal
      (0.71 -> 0.67) and macrotidal estuary (0.77 -> 0.71) tiles. This is expected, because
      the stand-in keeps the old levels, so it is not a test of H1.
    - H2 (synthetic hydrographs, production levels) does not beat B from COAST-HG at any
      exponent:
      - At alpha 1, CSI is within -0.01 of baseline in every stratum.
      - Higher alpha lowers the bias, but at alpha 2 macrotidal-open CSI drops by 0.04.
      - The stand-in SFINCS was forced with COAST-HG shapes, so B from COAST-HG matches it by
        construction. A fair H2 test needs SFINCS forced with the synthetic hydrographs (§5.F).
      - Macrotidal-estuary bias is already 1.01 at baseline in this sample, so the H2
        acceptance threshold (<= 1.3) does not discriminate here.
    - Against the local SFINCS with H1 forcing, 35 tiles: eikonal H1 beats the eikonal baseline
      in every stratum (CSI +0.01 to +0.10; macrotidal estuary bias 1.29 -> 1.08). H1H2 at
      alpha 1 is within 0.02 of H1.
    - H4 (shared product >= current calibration), 18 tiles with both scores: median CSI of
      H1-vs-SFINCS-H1 is >= baseline-vs-stand-in in four strata. Microtidal is the exception
      (3 tiles, -0.015). Two tiles lose more than 0.02 (805 -0.05, 870 -0.05).
    - At the FES-suspect tiles, eikonal H1 is much worse even against SFINCS with the same
      shift (891 Minas Basin: CSI 0.57 -> 0.27, bias 3.25; 2325: 0.68 -> 0.45). The eikonal seeds
      every boundary cell with its own level, while SFINCS takes the shift only at its offshore
      boundary points. Extreme cell levels at the basin head therefore flood the eikonal and
      not SFINCS.

## 3. File-level plan

New (this repo only; GCFM_UUdev untouched; `sfincs_tiles/` untouched):

| File | Content | State |
|---|---|---|
| `src/tide_statistics.py` | MHWS, MHW, HW99, HAT, high waters | done, tested |
| `src/storm_tide_hydrograph.py` | HGRAPHER-RP port; spring-tide window; per_rp / pooled tau(peak) / gaussian shapes; `scale_to_peak`; `time_above` | done, tested |
| `src/along_water.py` | water-path distance (Dijkstra, 8-conn.) + IDW with haversine fallback | done, tested |
| `boundary_test/station_inventory.py` | COAST-RP + TC/ETC + GTSM match + COAST-HG spring HW proxy | done |
| `boundary_test/gesla_inventory.py` | deduplicated GESLA sites | done |
| `boundary_test/select_test_tiles.py` | strata and provisional picks | done (proxy MHWS) |
| `boundary_test/extract_gtsm_series.py` | surge / tide series at test stations | done |
| `boundary_test/surge_duration_diagnostic.py` | F6a: dur50 vs surge peak, above_record onset | done, see §2.12 |
| `boundary_test/fes_tide_stats.py` | FES prediction (pyTMD) at boundary cells -> MHWS etc. | waits for FES |
| `boundary_test/build_boundary_product.py` | S_RP, L_RP, hydrographs per boundary cell | waits for FES |
| `boundary_test/run_boundary_test.py` | eikonal runs 1-4 on test tiles, scoring (`--against-local-h1`) | after product |
| `boundary_test/sfincs_local.py` | SFINCS with H1 forcing (shifted bzs), local exe | done (39 tiles) |
| `boundary_test/analyse_boundary_test.py` | per-stratum scores, F2, F4, F9, H1 shift map | done |
| `boundary_test/daigle_extract.py` | §4a | waits for PDF |

Config: none in `config.yml`; the driver takes its settings from a `boundary_test:` block in its
own YAML once the product exists.

## 4. Decisions still open

- Tile list review: `.cleanup_baseline/boundary_test/test_tiles_representative.csv` (strata
  from the proxy MHWS; re-stratify for reporting with FES MHWS).
- TC Gaussian width: 8 h HWHM fixed, or scaled by the TC/ETC RP100 ratio.
- Per-RP vs pooled shape: F6a favours per-RP selection (§2.12); confirm.
- CHS chart datum -> CGVD2013 at Saint John; source of N_CGG2013.
- QA for MHWS_FES at basin heads (§2.19): cap by the nearest gauge, by a ratio to the
  interpolated MHWS_GTSM, or keep the production level where the two differ by more than X m.
- Finish or rescope the stopped batches: 290 eikonal tiles and 11 local SFINCS tiles remain;
  1627, 2657 were interrupted mid-run.
- F3, F6, F7, F8 and the depth metrics are not produced yet (the runs keep no depth maps).
