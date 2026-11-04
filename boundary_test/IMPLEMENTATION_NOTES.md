# Boundary test: implementation notes (step 0)

Plan: `handoff_tide_corrected_duration_boundary.md`. Outputs: `.cleanup_baseline/boundary_test/`
(local; P: stays read-only). Nothing here changes the production DAG.

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
16. **MHWS_FES vs MHWS_GTSM (F4, first 50 tiles, 1,483 stations): FES is lower almost
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
| `boundary_test/sfincs_local.py` | SFINCS with H1 forcing (shifted bzs), local exe | after product |
| `boundary_test/daigle_extract.py` | §4a | waits for PDF |

Config: none in `config.yml`; the driver takes its settings from a `boundary_test:` block in its
own YAML once the product exists.

## 4. Decisions still open

- Tile list review: `.cleanup_baseline/boundary_test/test_tiles_representative.csv` (strata
  from the proxy MHWS; re-stratify for reporting with FES MHWS).
- TC Gaussian width: 8 h HWHM fixed, or scaled by the TC/ETC RP100 ratio.
- Per-RP vs pooled shape: F6a favours per-RP selection (§2.12); confirm.
- CHS chart datum -> CGVD2013 at Saint John; source of N_CGG2013.
