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
   elevation would ignore both. Recommendation: keep B's definition. The T(z) lookup files are
   then unnecessary; store the hydrographs and let B evaluate them.
2. **H4 and runs G2/G3 need new-forcing SFINCS runs (HPC).** Against the existing SFINCS runs,
   which use COAST-HG shifted to COAST-RP, H1 changes the eikonal forcing but not SFINCS. So
   macrotidal tiles will disagree by construction. Until new SFINCS runs exist, only Fundy
   (GeoNB), the GESLA gauges and Daigle test H1; vs-SFINCS numbers test H2 only, at unchanged
   levels.
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
   (`src/tide_statistics.py`).
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
15. **TC stations:** at tile 2146 (Japan) the RP10-1000 surge targets are 0.6-2.2 m, against
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
| `boundary_test/run_boundary_test.py` | eikonal runs 1-4 on test tiles, scoring | after product |
| `boundary_test/daigle_extract.py` | §4a | waits for PDF |

Config: none in `config.yml`; the driver takes its settings from a `boundary_test:` block in its
own YAML once the product exists.

## 4. Decisions still open

- Tile list review (provisional, proxy MHWS): `.cleanup_baseline/boundary_test/test_tiles_provisional.csv`.
  Re-stratify with FES MHWS once available.
- T definition: keep B's head-above-spill (recommended) or adopt T(z_cell).
- TC Gaussian width: 8 h HWHM fixed, or scaled by the TC/ETC RP100 ratio.
- Per-RP vs pooled shape: F6a favours per-RP selection (§2.12); confirm.
- CHS chart datum -> CGVD2013 at Saint John; source of N_CGG2013.
