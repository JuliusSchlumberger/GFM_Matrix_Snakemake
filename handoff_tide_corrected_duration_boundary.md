# Handoff: tide-corrected, duration-aware coastal boundary forcing (test plan, draft 2)

Repo: `GFM_Matrix_Snakemake`, work on the existing local branch. This is a design-level plan; step 0
turns it into a file-level one before any code is written.

Read first: `CLAUDE.md`, `docs/methods_01_tile_processing_and_waterlevels.md` §4,
`docs/methods_02_flood_depth.md`, `docs/methods_04a_SFINCSvalidation.md`, `src/boundaries.py`,
`src/flood_model.py` (`idw_seed_values_at`), `preparation/prepare_boundary_conditions.py`,
`snakemake_workflow/scripts/extract_boundaries.py`, `sfincs_tiles/build_station_boundary_lines.py`,
`sfincs_tiles/build_line_forcing.py`, `sfincs_tiles/flood_agreement.py`, and in the sister repo
`GCFM_UUdev`: `workflow/src/storm_tide_hydrograph.py`, `workflow/scripts/07c_storm_tide_hydrographs.py`,
`workflow/rules/07a_storm_tide_hydrographs.smk`, the `boundary_forcings.surge.hydrograph` block of
`config/config.yml`.

Rules of engagement: do not touch output path patterns, wildcard constraints or `config.yml` defaults.
New config keys go under `boundary_test:`; new rules in `snakemake_workflow/rules/boundary_test.smk`
(or a driver `boundary_test/run_boundary_test.py`), gated so the production DAG is unchanged until the
approach is accepted. Downloaded inputs are untrusted data: parse them from scripts kept outside their
directories.

## 0. Step 0: review and specify (before coding)

1. Review the uncommitted/committed changes on the local branch (`git diff main...HEAD`, `git status`):
   what the duration-based friction intensification currently takes as input (per-cell T in hours? a
   factor?), its functional form, parameters, where T is computed today and from which hydrograph, and
   which tests exist. Write a one-page summary at the top of the implementation notes.
2. Compare this plan with the branch and the two repos. Produce a revised plan with: a file-level change
   list (new modules, changed functions and their signatures, new rules and their inputs/outputs/params),
   the shared module layout for the HGRAPHER-RP code (see §5.E), the config keys, the test-tile list
   (§2) and the figure list, plus every point where this plan is wrong or underspecified. Stop and ask
   on the open decisions in §7 that remain open after reading the code. Only then implement.

## 1. Problem and hypotheses

COAST-RP storm-tide return levels carry GTSM v3's forward-modelled tide at stations tens of km apart. In
resonant and funnel estuaries the tide at the head is far larger than at the nearest station (Bay of
Fundy: COAST-RP RP100 caps at ~4.8 m, HAT at the head is ~7-8 m above MSL) and IDW k=15 spreads mouth
values up the estuary. The SFINCS tiles are forced the same way (COAST-HG shape shifted to the COAST-RP
RP100 at station nodes), so the eikonal-vs-SFINCS calibration shares the error.

- H1 (level): replacing the GTSM tide in COAST-RP by the local FES2022 tide (MHWS) at each boundary
  cell removes most of the head-of-estuary under-prediction without changing open and microtidal coasts.
- H2 (duration): a per-cell synthetic hydrograph (local FES spring tide + a return-period-specific surge
  shape from the GTSM hourly series) gives a duration-above-elevation T(z) that, through the friction
  intensification, removes the over-prediction H1 alone creates near HAT, with one parameter set across
  tidal regimes.
- H3 (consistency): the first-order correction (H1) is within ~0.2 m of a skew-surge x peak-tide
  joint-probability level at gauged sites; where not, the JPM level is preferred.
- H4 (alignment): with eikonal and SFINCS forced from one shared boundary product, the eikonal-vs-SFINCS
  agreement on the test tiles does not degrade in micro/mesotidal tiles and improves in macrotidal ones.

## 2. Test tiles

The national-map validation regions (Scotland, Wales, England, California/SF Bay) are held out; they are
used only once at the end (§6). The only validation region used during testing is the upper Bay of Fundy,
because it is the case that motivates the work and it has no SFINCS run.

Testing is done against SFINCS, which is the instrument the model is already calibrated against. Pick
tiles from the existing SFINCS calibration set (`calibration_500_tiles` / the 1000-tile friction sweep;
tiles with a finished SFINCS run and a "suitable" boundary line), stratified by tidal regime using the
FES2022 MHWS at the tile's boundary cells:

| Stratum | MHWS above MSL | n tiles | purpose |
|---|---|---|---|
| microtidal | < 0.5 m | 4 | must not change |
| mesotidal | 0.5-1.5 m | 4 | small change expected |
| macrotidal, open coast | > 1.5 m, no estuary | 3 | tide term large, no amplification |
| macrotidal, estuary/funnel | > 1.5 m, estuary in tile | 3-4 | where H1 and H2 act |
| TC-dominated | COAST-RP TC share > 50 % at RP100 (if the file carries the split) | 2-3 | surge-shape switch |

Plus the two Fundy GeoNB regions (`cumberland_basin`, `petitcodiac`), eikonal-only, scored against GeoNB
and the Daigle 2020 levels (§4a). Selection is a script (`boundary_test/select_test_tiles.py`) that writes
the tile list with its stratum and the FES statistics used; the choice is reviewed once, then frozen.

## 3. Datasets (locations given by the user; write catalog entries)

Paths are as given; register each as a catalog entry in the existing `meta:` style (new
`data_catalog_boundary_test.yml` loaded alongside `data_catalog_gfm.yml`), including vertical datum and
epoch for every entry.

| # | Dataset | Location | Needed for | Catalog key | Datum / epoch to record |
|---|---|---|---|---|---|
| 1 | FES2022b tidal atlas (`ocean_tide_extrapolated`, 34 constituents) | `P:\11212688-004-global-floodmaps\modelling\inputs\fes2022b` | tide term (A), hydrograph tide (E) | `fes2022_ocean_tide` | relative to local MSL by construction |
| 2 | GTSM storm-tide hourly reanalysis (`waterlevel`, `surge`, tide) | `P:\11212688-004-global-floodmaps\modelling\inputs\GTSM_storm_tide_hourly` | HW_GTSM at stations (B), surge shapes (E), skew surge (D) | `gtsm_hourly` | GTSM MSL; check whether annual MSL is already removed or must be (1-yr running mean as in GCFM_UUdev) |
| 3 | COAST-RP | catalogued (`coast_rp`) | levels; check for TC/ETC split variables | `coast_rp` | local MSL of the GTSM run |
| 4 | COAST-HG RP100 | catalogued (`coast_hg`) | only as a cross-check of the new shapes (F6) | `coast_hg` | local MSL |
| 5 | GESLA 4.1 (`GESLA4.1_ALL` + metadata CSV) | `P:\11212688-004-global-floodmaps\modelling\inputs\GESLA4.1_ALL` | gauge checks at all test tiles with a gauge (D, F10, §4a) | `gesla_4_1`, `gesla_4_1_metadata` | per record: station datum, see metadata |
| 6 | Daigle (2020) NB report PDF + CAN-EWLAT CSV bundle | `validation/NewBrunswick/new_inputs/boundary_forcing` | Fundy reference levels and HHWLT (§4a) | `nb_daigle2020_report`, `nb_daigle2020_levels`, `can_ewlat_datum_levels` | CGVD2013 (CGG2013 geoid); CAN-EWLAT: CGVD28/CGG2013 at epoch 2010 |
| 7 | UK Coastal Flood Boundary 2018 | `validation/England/new_inputs/boundary_forcing` | end-of-test validation only (§6), not for testing | `uk_cfb_2018` | ODN (Newlyn) |
| 8 | MDT CNES-CLS22 | catalogued | datum chain for 5-7 | `mdt_cnes_cls22` | 1993-2012 mean |

NOAA extreme water levels: not downloaded, SF Bay is held out anyway; dropped.

Datum rule for the whole test: every level that enters a figure or a metric is carried to GOCO06s with
the offset used recorded in the output table (`datum_src`, `offset_m`, `offset_source`). Chains:
local-MSL products + MDT (existing); CGVD2013 + (N_CGG2013 - N_GOCO06s); ODN via the station MSL-ODN
offsets in the CFB dataset itself (then + MDT) or via a GOCO06s/OSGM15 comparison, whichever is available;
GESLA via station datum -> MSL (metadata) -> + MDT. A level whose chain is unknown is plotted hollow and
excluded from metrics.

Python: `pyTMD` (FES prediction with nodal corrections), `pyextremes`, `utide` (phase 2 only). Add to
`environment.yml` behind a comment; do not change pinned versions.

## 4. Pre-checks

1. GESLA metadata parses with aligned lat/lon columns (known pandas mis-parse); deduplicated station list
   near the test tiles (1 km grouping, research-quality/national contributors first, >= 20 y of flag-1).
2. COAST-RP and GTSM hourly stations within each test tile; verify the GTSM hourly station set is the
   COAST-RP station set (GCFM_UUdev matches them at 0.1 km); plot with the DeltaDTM coastline (F1).
3. §4a Daigle extraction and verification.
4. SFINCS: list which test tiles have a finished SFINCS run and its boundary line; record the
   `build_line_forcing` settings used (average-tide COAST-HG, RP100, MDT-corrected COAST-RP).

### 4a. Daigle (2020) extraction and verification

Read the PDF with `pdfplumber` from a script under `boundary_test/`. Expected per coastal section:
HHWLT, storm-surge return level(s) (need 1:100; note others), SLR per horizon, and the summed "flooding
estimate" per horizon (present day, 2100, maybe 2050); confirm the datum (expected CGVD2013) and whether
wave run-up or freeboard is included. Write `processed_inputs/boundary_test/nb_daigle2020_levels.csv`:
`section_id, section_name, region, lon, lat, hhwlt_m, surge_rp100_m, surge_rp_other (json), slr_<year>_m,
twl_rp100_<year>_m, datum, includes_runup, pdf_page, notes`. Georeference each Fundy section (midpoint or
named place) so it matches boundary cells, COAST-RP stations and the GeoNB regions.

Checks (rows in `nb_daigle2020_checks.csv`, panels in F0):
1. Internal consistency: TWL = HHWLT + surge (+ SLR) within rounding; flag sections where it fails.
2. Saint John (GESLA, CHS, ~100 y): detrend to the report's epoch; AMAX/GEV and POT/GPD RP100 vs the
   section's present-day TWL. Chart datum -> CGVD2013 offset from CHS (user supplies if not in metadata).
   > 0.3 m gap = datum problem before statistics problem.
3. Tide: `utide` HAT at Saint John vs HHWLT vs FES2022 HAT at the nearest node; HHWLT vs FES2022 HAT at
   the CAN-EWLAT Fundy stations (no gauge needed) - first evidence on FES2022 in the bay.
4. Surge: skew surge at Saint John (observed minus predicted high waters), POT/GPD RP100 vs the report's
   surge term; a mismatch is informative, not a failure.
5. Map-edge check: elevation (DEM in CGVD2013; NB LiDAR if available, else DeltaDTM) along the boundary
   of each GeoNB wet extent; its mode should sit at the section TWL within DEM error. This tells which
   column drove the map and whether run-up was added.
6. COAST-RP gap: nearest COAST-RP RP100 (+MDT) vs section TWL per Fundy section.

## 5. Method

Conventions throughout: high-water statistic = MHWS (mean of the high waters within +-36 h of each
monthly spring maximum, over one nodal cycle); every level in GOCO06s; "cell" = a boundary cell as seeded
by `extract_boundaries`; "node" = FES2022 node; "station" = COAST-RP/GTSM station.

### A. Tide statistics at boundary cells (FES2022)
- Map each boundary cell to the nearest ocean-connected FES2022 node (reuse the ocean-connectivity logic),
  max 20 km. Predict hourly over 18.61 y at each unique node (pyTMD, nodal corrections). Derive MHWS, MHW,
  HW99, HAT, and the empirical high-water distribution. For now test tiles only; the code must run for
  any tile list so the global pass later is a config change.
- Output `processed_inputs/boundary_test/tide_stats_fes2022.parquet` + per-tile cell->node map.

### B. Same statistics from the GTSM hourly tide at stations
- tide = waterlevel - surge after removing the 1-yr running-mean MSL (as GCFM_UUdev
  `remove_mean_sea_level`); MHWS/MHW/HW99/HAT at every station in the test tiles.
- F2: HW_FES(nearest node) vs HW_GTSM(station), scatter + map, estuary stations highlighted.

### C. Tide-replacement correction (H1)
- Per station: `S_RP = COASTRP_RP - MHWS_GTSM` (surge-like residual; the same quantity GCFM_UUdev uses as
  the surge target when `match_on: surge`).
- Interpolate S onto boundary cells by along-water distance: nearest stations by shortest path through
  ocean-connected water (Dijkstra/fast-marching on the coarse connectivity raster already built at
  `connectivity_resolution_m`), IDW over the k nearest by that distance; keep haversine IDW as a fallback
  for tiles where the coarse raster is disconnected, and log which was used.
- `L_RP(cell) = S_RP(cell) + MHWS_FES(cell) + MDT(cell) + SLR(cell)`.
- Write in the same NetCDF/GeoPackage shape as the current scenario files so `extract_boundaries` and the
  eikonal read it unchanged (new filename template under `boundary_test:`).
- F3: along-estuary profiles (distance from mouth) for the estuary test tiles and Fundy: COAST-RP raw,
  L_RP, HAT, reference level where one exists (Daigle 2020; GESLA POT RP100 at gauges).
- F4: map of `MHWS_FES - MHWS_GTSM` at all stations in the test tiles.

### D. Skew-surge joint probability at gauged sites (H3, phase 2)
- Skew surge per tidal cycle from the GTSM hourly series; POT/GPD at stations within 25 km of a GESLA
  gauge; convolve with the FES high-water distribution from A to the RP level.
- F5: JPM vs C vs GESLA POT RP100 per gauge, with CIs.

### E. Synthetic hydrograph per boundary cell (H2)
Port the GCFM_UUdev HGRAPHER-RP code (`storm_tide_hydrograph.py`) into a shared module
(`src/storm_tide_hydrograph.py` here, or a small package both repos depend on; decide in step 0), with
these changes, applied in both repos:
1. Tide from FES2022 at the cell (a 6-day window centred on a spring high water, MHWS-consistent), not
   the GTSM average tide cycle. The surge target per RP is `S_RP` from C (so H1 and H2 use one HW
   statistic; GCFM_UUdev's `match_on: surge` target `level - average high water` becomes
   `level - MHWS`).
2. Shape per RP: keep the event-selection machinery, but add a diagnostic first: per station, the time
   above half-peak (`dur50_hr`) of every event against its surge peak, with the RP targets marked.
   Report per station the RP from which `shape_mode == above_record` (10 largest events reused). If the
   duration-vs-peak relation is flat or noisy, replace per-RP event selection by one pooled shape with a
   fitted time scale `tau(peak)` (log-linear fit of dur50 on surge peak over all events, extrapolated
   to the RP targets); this uses all events and gives a monotonic RP dependence. Decide per the
   diagnostic (F6a), not a priori.
3. TC stations: where the COAST-RP RP level is TC-dominated (TC/ETC split if present; else stations
   inside IBTrACS-basin boxes with COAST-RP RP100 >> GTSM-ERA5 largest event), the ERA5-driven events
   do not contain the storms that set the RP level. Use a separate narrow shape there: Gaussian with
   half-width `tc_half_width_hr` (default 8 h, config key), flagged in the output. Longer term: shapes
   from the synthetic-TC GTSM runs behind COAST-RP, if those series can be obtained.
4. Surge peak on tidal high water (as COAST-HG / GCFM_UUdev); record this assumption; the cross-station
   lag question stays open.
- Scale the surge so that `max(tide + a*shape) = L_RP(cell) - MDT - SLR`, then add MDT + SLR.
- T(z): hours above z over the window, z from -2 m to the peak in 0.05 m steps; lookup per boundary cell
  (`T_lookup_{tile}_{RP}_{SLR}.npz`). Land cells take T(z_cell) from the nearest boundary cell by
  along-water distance (fallback haversine), then the branch's friction intensification.
- F6a: dur50 vs surge peak per station (diagnostic above). F6: tide / surge / combined hydrograph and T(z)
  for one cell per stratum; COAST-HG RP100 overlaid as the cross-check; TC shape overlaid for TC tiles.
- F7: map of T at z = L_RP - 1 m and of the friction factor, one macrotidal estuary tile and Fundy.

### F. One boundary product for eikonal and SFINCS (H4)
- Define `boundary_cells_{tile}_{RP}_{SLR}` (GeoPackage/NetCDF): cell id, lon/lat, node id, station ids
  and weights, `L_RP`, components (S, MHWS_FES, MDT, SLR), hydrograph (time_hr, level) and the T(z)
  lookup reference. `extract_boundaries` reads `L_RP`; a new `sfincs_tiles/build_line_forcing.py` mode
  (`--forcing boundary_product`) reads the hydrograph at the station nodes of the existing boundary line
  (nearest boundary cell to each node, by along-water distance), replacing the COAST-HG lookup and the
  `boundary_value - hydrograph_max` shift. Nothing else in the SFINCS build changes.
- SFINCS runs on the test tiles with the new forcing (HPC; the user runs them). Until they exist, the old
  SFINCS runs serve as a stand-in and are labelled as such in every figure.

### G. Runs and evaluation
Per test tile, RP100, SLR_0 (RP10 and RP1000 second pass):
1. baseline eikonal (production forcing) vs baseline SFINCS (old forcing)  - the current calibration
2. H1 eikonal vs SFINCS(new forcing)
3. H1+H2 eikonal vs SFINCS(new forcing)
4. H1+H2 with the intensification exponent swept over 3-5 values, calibrated on the macrotidal estuary
   stratum and Fundy, checked on the others.
Metrics: the existing eikonal-vs-SFINCS extent and depth metrics (`flood_agreement.py`: HR/FAR/CSI,
bias, depth r/RMSE), per stratum. Fundy: existing soft-scored metrics against GeoNB.
- F8: agreement maps baseline / H1 / H1+H2 for one tile per stratum and Fundy.
- F9: CSI and bias vs exponent, one line per stratum; the chosen exponent must not lower any stratum's
  CSI by > 0.02 relative to its best.
- F10: at gauged test tiles, L_RP (C), JPM (D) and COAST-RP vs GESLA POT RP100 (MAE/RMSE, not ME).
- Table: per stratum and tile, flooded area, CSI, bias, depth RMSE for runs 1-3.

## 6. Held-out validation (after acceptance)
Rerun the existing `validation/` scope (Scotland, Wales, England, California) once with the accepted
configuration; add the UK CFB 2018 levels as the water-level check there (F3-type profiles for the Severn
and Forth). No tuning on these.

## 7. Acceptance
- H1: Fundy-head CSI from ~0.002 to > 0.3; micro/mesotidal strata unchanged within 0.02 CSI vs SFINCS.
- H2: one exponent brings the macrotidal-estuary stratum bias to <= 1.3 without failing the F9 rule.
- H3: |C - JPM| <= 0.2 m at >= 80 % of gauged sites, else JPM goes to production.
- H4: eikonal-vs-SFINCS CSI with the shared product >= the current calibration's in every stratum.

## 8. Decisions
Settled: MHWS as the high-water statistic; along-water-distance interpolation (fallback haversine);
test regions only for the FES pass now, code global-ready; strict datum bookkeeping (§3).
Open, to settle in step 0:
1. TC shape: fixed 8 h Gaussian vs a width scaled by the COAST-RP TC/ETC ratio, and how TC stations are
   identified if the file has no split.
2. Where the shared HGRAPHER-RP module lives (copy into both repos vs one package).
3. Per-RP event selection vs pooled shape with fitted time scale (decide from F6a).
4. GTSM hourly: whether annual MSL is removed in the files as delivered.
5. The CHS chart-datum -> CGVD2013 offset for Saint John and the N_CGG2013 source.
