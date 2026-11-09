# TODO

Open items after the cleanup on branch `cleanup` (not merged into `main`). Everything in
the code is done; what remains needs you, the HPC cluster or `<data_root>`.

## Next steps

1. **Merge the latest `cleanup` commits into `main`** (the earlier merge is done; `git log origin/main..cleanup`).
2. **On the cluster:** back up the `*_materialized.yml` files in `<code_root>/snakemake_workflow/config/`,
   `git pull` on `main`, create the gitignored site files
   (`cp sfincs_tiles/site_env.example.sh sfincs_tiles/site_env.sh` and
   `cp snakemake_workflow/config/config_hpc.example.yml snakemake_workflow/config/config_hpc.yml`;
   the examples hold the example site's values), then `python snakemake_workflow/scripts/materialize_configs.py` (the cluster
   checkout is not under `<data_root>`, so it cannot be done from the local machine). Configs with the old keys
   `slr_scenario` / `slr_scenarios` fail with KeyError.
3. **On the cluster:** `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part models` once, to check the
   shell drivers (only tested locally with stubbed sbatch/python). Paths live in
   `sfincs_tiles/site_env.sh`; keep its roots in sync with `config_hpc.yml`.
4. **Cluster `gfm` env:** `mamba install -n gfm pytest=8.4.2 ruff=0.12.0`.
5. **`<data_root>` cleanup:** `sfincs_smoke_test/` is no longer used; delete it if you like. When
   calibration stage 1 runs again, pass `--overwrite` so screen / pre-flight error records cached by
   earlier runs are recomputed.
Global deltas are flood-hazard only; their CSV and figures were regenerated on 2026-10-21 from the
existing ssp126/ssp245 simulations (`flooded_area_km2` = the former unprotected value; old files in
`.cleanup_baseline/deltas_before_hazard_only/`).

Done on 2026-10-18 (outputs regenerated with the new code, values identical to the previous files;
old copies in `.cleanup_baseline/p_backup_regen/`): flood totals (cluster job), Bangkok case exposure,
delta flood extent + SLR-year column, delta design selection and `deltas_design_materialized.yml`
(design runs since removed, item 6),
England FZ3 / FZ2 metrics, calibration_500_tiles batch jobs, the validation summary tables, and the
materialized configs on this machine. The 2026-10-15 production `resolved_config.yml` was moved to
`model_outputs/hpc_jobs/resolved_config_2026-10-15.yml`.

Snakemake: the config-key rename re-triggers `cache_waterlevel_stations`, `extract_boundaries` and
`run_eikonal` the next time a scenario runs (+25 jobs for bangkok_chao_phraya). Accepted.

Renamed CLI flags (old spellings now rejected): `--outdir` -> `--out-dir`, `--figdir` -> `--fig-dir`,
`--output` / `--out-csv` -> `--out`, `--dir` -> `--timings-dir`, `--threads` -> `--workers`,
`--max-outer` -> `--max-outer-iterations`, `--inner-max-rounds` -> `--max-rounds`,
`--tiles` -> `--mask-files`, `--skip-done` / `--no-force` -> `--skip-existing`,
`--n-tiles` -> `--n-panel-tiles` (plots), `--cpus` -> `--cpus-per-task`,
`--flood-totals-cpus` -> `--flood-totals-cpus-per-task`; positional arguments of the calibration
study scripts and `build_tile_subset_by_id.py` are options now (`--out-dir`, `--n-candidates`,
`--fig-dir`, `--tile-ids`, `--tile-ids-file`, `--out`). Notebooks or scripts outside the repo that
import renamed modules (pipeline_plots, analysis_figures, exposure_grid, ...) need the same update.

## Reviewer critiques (branch `review-critiques`)

Evidence: `review_critiques/` scripts; results, figures and the summary page source in
`.cleanup_baseline/review_critiques/`.

Decided and implemented:
- **SLR propagation (critique 1a):** `simulation.flooding.slr_propagation: two_component` is the
  default. The static level S = MDT + SLR fingerprint spreads by coast connectivity without
  attenuation and acts as sea; only the storm tide on top attenuates. Bangkok tiles: 1.8-2.7x the
  attenuated flooded area from +0.5 m SLR, identical at SLR 0.
- **Duration correction B (critique 3):** `simulation.flooding.duration_correction` on, alpha 1,
  t_ref 12 h, COAST-HG shape of the nearest station per cell, second solve (~2x runtime). On SFINCS
  RP100 tiles: pooled CSI 0.742 -> 0.763 on 498 random tiles (median tile unchanged; the gain comes
  from a minority of overflooded basins), false alarms down; t_ref 6-24 h changes CSI by < 0.01. The
  weir volume cap (A) changed nothing and stays in `src/event_volume.py` only for the experiments.
  These numbers are for the clipped form clip((t_ref/T)^alpha, 1, 10). The multiplier is now
  (t_ref/T)^alpha with no lower clip (1-D refit, INPUT_ANALYSIS.md §6), which changes results:
  recalibrate friction_scale_factor with B on SFINCS (HPC), confirm on 5-10 real tiles (test 8),
  then rerun the tile simulations. Snakemake does not see the change (config values unchanged),
  so existing eikonal outputs must be removed or rerun explicitly.
- **Land-cover friction (critique 2):** kept, described as a land-cover-indexed attenuation rate
  (docs/methods_02 §2.2, code comments; config keys and file names unchanged). Land-cover and
  uniform attenuation are within 0.005 pooled CSI on 498 random tiles, with and without B.
- **friction_scale_factor with B:** 7.5 (498 random RP100 tiles, pooled CSI 0.763, bias 1.08;
  9.0 gives 0.757, bias 0.98). Without B the optimum stays 12.0 (CSI 0.742). Results in
  `.cleanup_baseline/review_critiques/calibration/representative/`, figures R1/R2.
- **Straight-line baseline** (level = nearest seed level - rate x Euclidean distance, no solver;
  `--stage distance`): best pooled CSI 0.713 coast-connected (0.10 m/km), 0.668 without
  connectivity, against 0.739 eikonal uniform and 0.763 eikonal + B. Median tile within 0.006;
  worse by > 0.2 CSI on 3-4% of tiles (sills, basins, ridges). Figure R3.
- **Outputs:** new results go to `{root}/model_outputs_v2`, `merged_results_v2`, `figures_v2`,
  `validation_v2`; per-tile inputs stay in `{root}/model_outputs` (`preprocessing_inputs_dir`).
  Every scenario follows config.yml's model settings (no pins: the whole pipeline is rerun,
  user 2026-11-09) and sets `preprocessing_inputs_dir: null` (own inputs).
- **Protection rule:** `protection.rule: flood_fraction | water_level` (implemented, default
  flood_fraction; `minimum_rp` 2 applies to the water-level rule only).
- **Permanent inundation:** to be counted in EAI by anchoring the exceedance curve at p = 1 with
  the permanently inundated exposure, and reported as a separate metric (not implemented yet, see
  Open).

Open:
- **Consistency audit (2026-11-09), open after the pins removal, own deltas station caches and
  the shared pixel-size lookup.** Before the full rerun:
  - Obstacle coupling blocks cells the inner solve never reached (t = 99, `flood_model.py` outer
    loop), unlike the obstacle-coupling study (`reached`). Kept as is (user, 2026-11-09).
  - England/Wales local input (and the hop test) keep DEM and EA levels in ODN; two_component
    still adds the static level MDT + 2020 offset in the GOCO06s frame
    (`boundaries.static_station_levels`). Kept (user, 2026-11-09): if these scenarios
    overestimate systematically, check this offset first.
  - Every documented run path passes `--rerun-triggers mtime` (run_eikonal_dag.py, HPC
    generators, run_bangkok_case_study.bat), so config changes never rerun outputs there; the
    rerun needs clean output directories. CLAUDE.md / docs/hpc.md describe the profile only.
  - Not declared as inputs although they change results: the hydrograph NetCDF (duration
    correction), the MDT station file and SLR_0 NetCDF (static level), the tile-grid contents, the
    DeltaDTM/land-cover sources (fix_ocean_mask edits in place). `boundary_conditions` is passed
    whole as a param (any edit reruns everything when params trigger).
  - `generate_hpc_preprocess_job.py` hard-codes the global station cache, ignoring
    `stations_cache_dir`; `coastline_nodes.parquet` is not cleared by `force` and ignores the mask
    codes; `compute_goco06s_absolute_raster` is unreachable (extract_dem takes it as a param).
  - After the reruns (user, 2026-11-09), docs behind the code: methods_02 §4.1 (knn 5 / power 3, edge-connected ocean, IDW per ocean
    component), §2.4 and config.yml:263 (per-RP product hydrographs), §4.4 and
    flood_model.py:37 (no 0.10 m wet threshold; validation at 0.01 m), §4.5 (lakes and rivers
    are not excluded); methods_01 §4 (hop seeds from the whole overlap; coastline nodes; the
    2020 baseline offset); methods_04a (SFINCS harness without a static level; old fsf units in
    "Structural limitations"); README outputs (`*_v2`); eikonal.py:211 epsilon; the 0.25 friction
    provenance (fitted with knn 15 / power 2, no static level) until the recalibration.
  - Hygiene: code defaults differing from config (idw_power 2.0, max_outer_iterations 3,
    rasters.extract_dem fill sizes, sync_deltadtm delete_zips .get False); study constants
    (epsilon, 40 rounds, CELL_M 30); hard-coded P:/ paths in review_critiques/ and boundary_test/;
    duplicate decode_waterdepth (rasters / merge); dead flood_agreement.prune_to_ocean_connected;
    dangling references (config.yml:158, event_volume.py:203, flood_agreement.py:10, boundaries.py:27).
- **Flood-model comparison and calibration (branch `model-variants`).** Seven variants share
  seeds, coastline, static level and wet rule (`src/flood_variants.py`, `flood_model.FLOOD_MODELS`):
  bathtub, connected bathtub (IDW of the seed levels), straight-line attenuation (exact max over
  seeds of level - rate x distance), eikonal with a uniform or a land-cover rate (obstacle
  coupling), and both eikonals with the duration correction. Built and tested; the steps below
  run with the new storm-tide forcing (user, 2026-11-09):
  1. Hydrax: product rebuild with coastline nodes, SFINCS forcing and runs, then
     `calibration_split.py` (30 % calibration / 70 % validation by tile-id hash; check
     `calibration_split_balance.csv`).
  2. Solver studies per friction source (`generate_calibration_batch_jobs.py sweep_budget |
     obstacle_coupling --friction-source landcover` at friction_scale_factor 0.25, and
     `--friction-source uniform --uniform-attenuation-m-per-km 0.2`); set `max_rounds` and
     `obstacle_coupling.max_outer_iterations` per friction source in the variant configs.
  3. `RESET_RUNS=1 PHASE=sweep bash sfincs_tiles/rerun_calibration.sh --part models`: friction
     factor and uniform rate on the calibration tiles; post job writes variant_selection.json.
  4. `PHASE=duration ...`: alpha of both duration variants at their base variant's value.
  5. `PHASE=selected ...`: every variant at its selected run on all tiles; validation-group
     scores in variant_selection.json and `plot_validation_results.py --group validation`.
  6. Best variant (pooled CSI on the calibration group) -> config.yml `simulation.flooding`
     (model, friction_source, parameter, alpha); production run on Hydrax.
  7. Local cases: `materialize_configs.py <scenario> --variants all --selection
     <variant_selection.json>` for england/wales/scotland/sf_bay local and global input; run and
     score each `<scenario>__<variant>_materialized.yml` with the scenario's usual commands.
  Open: the grids in `sfincs_tiles/calibration_paths.py` (10 friction factors 0.1-0.8, 8 rates
  0.05-0.45 m/km, alphas 0.25-1.5) widen if an optimum lands on an edge; the SFINCS harness has
  no static (two-component) level, as the SFINCS runs have no separate SLR part; New Brunswick
  has no local-input scenario yet; `straight_line` treats lon/lat cells as square, as the
  eikonal step does.
- **Boundary sources: a rule for FES2022 at fjord and estuary heads.** Partly answered by the
  version rule (FES2014 where FES2022 is implausible) and its neighbour growth in the level
  product (§8.5); the cases below where neither atlas fits stay open. FES matches GESLA within
  0.05 m MAE on MHWS at 698 gauges, but misses by > 0.25 m at 20, all at fjord/estuary heads or
  narrow entrances (too high in Juneau, the Salish Sea, Port Phillip; too low in the upper
  St Lawrence). Without gauges, a few head nodes are implausible: Cobequid Bay (MHWS 14.5 m above
  MSL, about the observed spring range), Bahia Lomas (15.5-16.3 m) and one Colorado-delta node
  (15.4 m); FES elsewhere in Minas Basin, the Strait of Magellan and the upper Gulf of California
  matches published ranges. Decide on a detection rule before FES levels go into forcing
  (`boundary_test/INPUT_ANALYSIS.md` §3, §8).
- **Boundary sources: why GTSM's tide is 7-8% too large.** S2 is +20% and M2 +6% against
  GESLA (FES: ~0%). Check against the GTSM v3 validation and the dataset documentation (radiational
  S2, SAL). Confirm the GTSM tide run's MSL reference epoch (data suggest 1986-2005 + an imposed
  ~2.6 mm/yr trend). COAST-RP inherits the tide excess, but its surge part is about 0.1 m too low
  against GESLA GEV fits, so the two partly cancel (§0.25). Replacing only the tide (H1) would
  lower levels that are already about 0.1 m low. Decide how the surge part is treated before H1.
- **FES extraction cache bug (fixed in `fes_tide_stats.extract_constants`):** pyTMD altered the
  cached atlas grid in place, so later extraction groups came back NaN (never wrong values). It
  cost 15 gauges / 6 GTSM stations in `levels_tides.py` (2 tide-set sites) and an unknown share of
  the ~13,600 "invalid" FES nodes in `.cleanup_baseline/boundary_test/fes_nodes.parquet`. Rerun
  `levels_tides.py` and `fes_tide_stats.py` before those outputs are used again.
- **Storm-tide product: run it on `<data_root>` and switch production** (implemented:
  `boundary_conditions.storm_tide_source: gfm_product` is the default; `src/storm_tide_product.py`,
  `preparation/prepare_storm_tide_product.py`; method and gauge validation in
  `boundary_test/INPUT_ANALYSIS.md` §8.4-8.5). The source drives the scenario levels
  (`prepare_boundary_conditions`), the eikonal seeds and static level, correction B's hydrographs
  (`boundaries.storm_tide_hydrograph_path`) and the SFINCS forcing shapes (`sfincs_tiles/gfm_config`).
  Done on `<data_root>` (2026-11-07): `run_preparation.py storm_tide_product
  boundary_conditions` without the TC mask, cache (~29 GB) under
  `processed_inputs/storm_tide_product/cache` for both machines; `WL_scenarios` holds the
  product's levels (marker `storm_tide_source.txt`; level + MDT + 2020 offset checked at all
  22,670 stations; the former files predate the 2020 offset, fig24); scenario configs
  re-materialized here. Left:
  1. re-materialize the scenario configs on Hydrax;
  2. Snakemake: the station caches re-run (their scenario NetCDFs are inputs) and with them every
     tile; then the SFINCS forcing, the fsf recalibration on the 498 tiles
     (`sfincs_tiles/generate_calibration_jobs.py stage1|stage2` on Hydrax), test 8 and the tile rerun.
  One processed_inputs tree serves every config of a data root; the deltas scenarios build their
  own WL_scenarios and station caches (`boundary_conditions.waterlevel_nc_dir`,
  `stations_cache_dir`) with `run_preparation.py boundary_conditions` and their config.
  Re-materialize the scenario configs on both machines (`materialize_configs.py`).
- **Storm-tide product: external review of decisions 1-3 (review brief, 2026-11-08).** Order,
  each step gated on the one before; the production rerun waits for steps 1-4.
  1. GPD shape cap: parametric bootstrap of the station shapes (`open_guesses.py shape_noise`).
     Gate: if a light tail reproduces the heavy shapes, replace the cap by a regionally pooled
     shape (~500 km, TC and non-TC pooled separately, station threshold and scale); score capped
     vs regional at gauges with > 60 y. Add a per-station 5-95% band above RP100 and a
     tail-uncertain flag (shape > 0.2 or cap binding).
  2. Cyclone mask: masked ETC fit + COAST-RP_TC scaled by a basin gauge ratio r_TC (N Atlantic,
     NW Pacific, N Indian, Australia/S Pacific; leave one basin out) against the double count, on
     median, MAE, 10-90% and the RP100 jump between neighbouring stations with and without an
     ERA5 landfall. Adopt if MAE within 0.02 m and spread and jumps smaller; else keep with a
     TC-coast flag. US coast: rescore against US-CoastEX (Morim et al., 2025).
  3. Gauge ratio r: open-coast calibration set (drop estuary/harbour/tidal-river gauges),
     multiplicative vs additive, median vs covariate fallback (shelf width, latitude, GTSM cell
     size) or 1 with a flag; leave-one-out on the 451 calibration and the 98 sparse gauges.
  Results (2026-11-08, `open_guesses.py shape_noise | tc_scale | tc_jumps | r_variants`, review brief):
  - 1: not noise. Stations with a TC part: 753 MLE shapes > 0.35 against 6 expected from a light
    tail at their storm counts (91 separate 5-deg cells); without a TC part 70 against 15. The cap
    question is decision 2.
  - 2: masked fit + COAST-RP_TC shifted by 0.24 m (fitted per basin, basin left out; 0.18 m WP):
    TC-dominated RP100 MAE 0.363 vs 0.361 m, 10-90% -0.64..+0.42 vs -0.59..+0.51; shapes > 0.35
    753 -> 69; RP1000 - COAST-RP p99 2.06 -> 1.16 m, max 8.4 -> 3.2 m; neighbour jumps unchanged;
    RP2 MAE (all gauges) 0.088 -> 0.104 m because the shift also lifts COAST-RP_TC's flat tide
    floor. Scaling instead of shifting is worse (MAE 0.413). No tapered shift or other device that
    exists only to improve the fit (user, 2026-11-08). Decided (user, 2026-11-08): mask on, no shift
    and no scale. Masked without a shift: TC-dominated RP100 median -0.165 m, MAE 0.391 m (double
    count -0.027 / 0.361 m); non-TC-dominated unchanged (0.164 vs 0.163 m).
  - 3: covariate fallback r = 1.03 x S10^-0.32 (GTSM RP10 skew surge, m; cap 2) where < 3 gauges
    within 500 km: sparse gauges RP100 MAE 0.226 -> 0.194 m, bias -0.11 -> -0.03 m; all 451
    0.163 -> 0.157 m. Fallback 1 worse; dropping gauges with S10 < 0.2 m no effect; additive
    offset +3 cm RP10 bias. The high ratios are open-ocean islands and Peru-Chile (S10 ~0.1 m),
    not estuaries. Decided (user, 2026-11-08): covariate fallback, fitted on all calibration
    gauges in `storm_tide_product.ratio_covariate`; the station's own GTSM RP10 skew surge sets it.
    Leave-one-out r MAE at the 197 sparse gauges 23% -> 15% (fig30c). It overshoots on calm coasts
    where GTSM is right: East Africa (Zanzibar r 0.79, Lamu 1.08, S10 ~0.24 m) gets r ~1.6, so the
    surge-signal uncertainty there is 0.59 m (fig29). Check the regional residuals of the power law
    after the rebuild.
  - 1950-2024 tail check (`open_guesses.py long_record`; all 23,211 stations rebuilt on 75 years,
    method unchanged; 1950-1978 store and FES basis in `.cleanup_baseline/storm_tide_product/
    open_guesses/long_record`): the cyclone tail stays. TC-part shapes > 0.35: 753 -> 737, the
    75-year median at those stations still 0.40; 245 drop out and 229 enter, so single storms set
    the shape (Hangzhou Bay, Pilbara RP1000 up to +3.2 m from 1950-78 storms). Without a TC part
    70 -> 33 (ETC heavy shapes mostly noise). Gauge RP100 MAE 0.195 (46 yr) vs 0.202 m (75 yr),
    TC-dominated 0.361 vs 0.364 m. Early/late skew RP10 median 1.00 (TC part 1.04, no TC part
    0.985): no pre-1979 storm deficit in ERA5. A longer record does not replace the cap at TC
    stations; justified options are the COAST-RP_TC tail (masked, decision 2) or a regionally
    pooled GPD shape. Decided (user, 2026-11-08): use the 1950-2024 record in production (store,
    FES basis and IBTrACS mask from 1950; IBTrACS before the satellite era is less complete).
    `storm_tide_product.ibtracs_path` is the full IBTrACS (`ibtracs.ALL.list.v04r01.csv`, 1842 on); the
    `*_materialized.yml` still name `ibtracs.since1980` until they are re-materialized.
  4. Tile interpolation: interpolate skew-surge levels and convolve with the local FES tide at
     the tile boundary, vs total-level IDW, on the stations whose current error exceeds 0.2 m.
     Coastline seeds are total-level IDW of the stations now, so an estuary never exceeds its
     nearest stations (Fundy: Cape Enrage MHWS 5.0 m forces Shepody Bay and Cumberland Basin).
     FES2022 (extrapolated grid) at coastline points: Hopewell Cape MHWS 7.6 m, Joggins 6.2 m
     (plausible), but Petitcodiac / Amherst Point 12.8 / 11.5 m (spring range 22-23 m, above
     Fundy's ~16 m maximum) and Rosyth 1.55 m below Leith 2.51 m: extrapolated cells in narrow
     estuaries are not usable as they are. Native cells do not help: `inputs/fes2022b/mask_fes2022B.nc`
     (0 native, 1 extrapolated) marks Amherst Point M2 8.4-9.3 m and Rosyth 1.15 m (Leith 1.80 m) as
     native. At 916 gauges, FES MHWS at the gauge vs IDW of the 5 nearest stations' FES MHWS: MAE
     0.061 vs 0.062 m, local closer at 51%; the 16 gauges where they differ by > 0.2 m split evenly
     (fjords: Juneau, Skagway FES 6-7 m against 2.7 m). Gauges sit near stations, so they do not
     test the estuary heads. GESLA's short records do: upper Fundy gauges agree with FES in open
    water and at Burntcoat Head (observed 0.1-99.9% range 15.2 m, FES spring range 16.1 m), while
    FES is too high in the tidal rivers (Hantsport 13.8 m observed vs 19.3 m; Moncton 8.3 m vs
    21 m). Some records are in feet (Five Islands, Diligent River), sparse (Burntcoat) or dry at
    low water (Moncton, Joggins). No GESLA gauge in the inner Forth. Next: the same check on all
    GESLA records, then a gauge-based correction of FES like r, and test 4 on top.
    All GESLA (first 2 years per record, M2 by least squares; 2,717 gauges after coverage >= 70%,
    no drying floor; `open_guesses/fes_check/fes_vs_gesla_all.parquet`): FES M2 MAE 0.12 m, 385
    gauges off by > 25%, FES at the gauge and IDW of the stations equally good (local closer 53%).
    Large errors in both directions: St Lawrence (Quebec, Ile aux Coudres FES 0.8 vs 1.8-2.0 m),
    Bangladesh (Khepupara 0.06 vs 0.87 m), Alaska/BC fjords, Ungava, Fundy tidal rivers. Gauge
    correction (median log gauge/FES M2 ratio of the other gauges within R, gauge left out):
    MAE 0.121 -> 0.099 / 0.089 / 0.095 / 0.103 / 0.116 m for R 10 / 25 / 50 / 100 / 200 km; at
    gauges with neighbours within 25 km 0.152 -> 0.100 m, 91 gauges worse by > 0.1 m. Feet
    records (old MEDS) still need a filter.
    Built: `src/tide_correction.py` in the product (25 km; `levels.parquet` tide_factor /
    tide_gauges), stage `open_guesses.py tide_correction` (2,373 usable sites: amplitude MAE FES at
    the gauge 0.185, current forcing 0.157, stations corrected 0.139, local corrected 0.133 m;
    feet filtered by a 20 m range) and `tile_interpolation` (test 4 at the 618 ratio gauges: the
    decomposition alone scores as the current IDW, RP100 0.195 vs 0.194 m; where the tide
    correction exceeds 10% (72 gauges) RP100 0.377 -> 0.332 m with corrected station tides, 0.350 m
    with the corrected local tide). fig25-fig28, input analysis section 9. The ratio gauges lie
    <= 10 km from a station; estuary heads are tested for the tide only.
    Decided (user, 2026-11-08): build the decomposition into the tiles. Built: coastline nodes
    (`src/coastline_nodes.py`, one per FES2022 grid cell of DeltaDTM coastline) appended to the
    product levels file; each node blends its knn nearest stations' corrected skew-surge models
    and convolves them with its own gauge-corrected FES tide; the seeds take IDW over stations and
    nodes together. Scored by `open_guesses.py coastline_nodes`. Open:
    - Nodes have no hydrograph: the duration correction takes the nearest station's, whose tide
      range can be smaller (estuary heads), so the time near the peak is too long there.
    - A node blends stations by distance only, across an isthmus too (Chignecto: Fundy and
      Northumberland Strait stations); the tide is the node's own.
    - The SFINCS boundary lines leave the nodes out (`boundary_lines_geometry._tile_stations`,
      matched by position against the product's level file), so SFINCS makes the estuary tide
      itself. Rebuild the lines after the product rebuild.
  5. Global rerun with flags (tail-uncertain, TC-coast, fallback-r); test 10 afterwards revises
     the flagged stations. The 1950-1978 surge months are on `<data_root>`
     (`inputs/GTSM_storm_tide_hourly/storm_surge_residual`, all 900 months).
  6. Regionally pooled GPD shape (`open_guesses.py pooled_shape`): on 1950-2024 with the mask, each
     station's shape maximises the summed profile likelihood of the stations within 100 / 300 /
     1000 km (scale per station); stability 1979-2024 vs 1950-2024 and gauge scores against the
     cap. Result (mask also covers IBTrACS storms without any wind on record, i.e. most pre-1979
     SI/SP/NI tracks; the wind-only run is kept in `open_guesses/pooled_shape_wind_only`):
     shapes > 0.35 at stations with a TC part 753 (production) -> 6 (75 yr, masked, per station),
     0 pooled at 100-1000 km; without a TC part 32 -> 3 (100-300 km). Change of a station's shape
     from 1979-2024 to 1950-2024 (|full - late| p50 / p90): 0.06 / 0.16 alone, 0.04 / 0.11 at 100 km,
     0.02 / 0.08 at 300 km, 0.02 / 0.05 at 1000 km. Cap and no cap score the same (<= 3 mm); pooling
     does not change the gauge scores (RP10-100 MAE within 3 mm), so the radius rests on stability.
     Masked without a shift, TC-dominated gauges: RP100 median -0.16 m, MAE 0.389 m (production
     -0.03 / 0.361 m). Decided (user, 2026-11-08): pool at 300 km, drop the cap, and shift the
     COAST-RP_TC levels per cyclone basin (test 2's basin fit, 0.18-0.25 m) with the mask on.
- **Storm-tide product: decisions taken as estimates (open questions; tested in
  `boundary_test/INPUT_ANALYSIS.md` §8.6, fig19-fig23).**
  - No TC mask (decided): ERA5 cyclones stay in the ETC fit and count again in COAST-RP's TC part,
    which matches the gauges best because both sources are too low (§8.6). Retest the mask
    (`boundary_test/open_guesses.py tc_mask`) when test 10 replaces the TC part.
  - Interpolation: `knn` 5, `idw_power` 3 (config default now; leave-one-out RP100 MAE 5.7 cm
    against 7.0 cm for k 15 / power 2, fewest stations off by > 0.5 m). The test predicts a station
    from neighbours farther away than a coastline cell's, so it overstates the error; the ranking
    should hold.
  - Along-water interpolation (`src/along_water.py`: water-path distance, built and unit-tested,
    never wired or scored): most of the 73 stations > 1 m off the k 5 prediction sit across
    headlands, at estuary heads or in sparse macrotidal basins, where a water-path distance would
    help. Score it with the same leave-one-out test in those regions before wiring it.
  - High waters (`skew_surge.high_waters`): a maximum in the lowest quarter of the 25-h range with
    a prominence < 10% of it is a low-water bump, not a high water (Ungava: largest skew surge
    4.5 -> 0.8 m). Test 9 rerun: dropping every weak maximum instead scores 5 mm better at RP100
    (MAE 0.168 vs 0.174 m, 451 calibration gauges) but also drops real lesser high waters of
    mixed tides; the low-band rule changes 26 gauge ratios by > 0.05 (the all-maxima rule 77).
    Weak maxima in mixed / diurnal tides (Sandakan, Kapingamarangi) carry part of GTSM's largest
    skew surges: GTSM's RP10 skew surge at Sandakan halves with either rule, the gauge's does not.
  - Gauge ratios without GTSM's own tide: production fits r on gauge vs GTSM-on-FES skew surges
    and takes the common years of those two series; test 9 also required GTSM's harmonic tide
    (`gtsm_harm`) and its years. The port's gauge set may differ by a few sites.
  - GPD shape cap 0.35 (`SHAPE_CAP`, Fathom RFA's cap): kept, and without the TC mask it sets the
    RP250-RP1000 tail at 832 stations, mostly hurricane/typhoon coasts (Hangzhou Bay RP1000
    12.6 m against COAST-RP's 4.8 m). The gauges cannot tell caps apart (RP100 MAE
    0.215-0.217 m from none to 0.2); at RP1000 no cap adds up to 12.9 m, 0.2 removes up to
    3.9 m. Above RP100 nothing checks it; test 10 (STORM-forced GTSM) is the check for the
    cyclone tail.
  - GTSM despike (|residual - 5-h median| > 2 m; >= 5 jumps > 2 m -> COAST-RP levels, 15
    Arafura stations): kept; 3-5 jumps give the same 15, and 1 m would catch real North Sea
    surges.
  - TC shape below RP10 requires TC dominance at RP10 (`TC_SHAPE_MIN_RP`).
  - FES version rule and its growth (`tide_prediction.consistent_version_flag`): right wherever
    a gauge checks it (Juneau, Skagway, Port Alberni, Vardo, Le Crouesty, Karachi, Kukup, Port
    Stanley; Sao Marcos resolved: Ponta da Madeira 2.62 m inside the bay supports FES2014 at the
    outer stations). One wrong flag: id_coast_glob_11746 at the Saguenay mouth (Tadoussac gauge
    2.18 m, FES2022 2.65 m, FES2014 1.38 m). Option: apply the gauge veto to the rule itself, not
    only to its growth. 41 flagged stations have no gauge within 150 km; Khambhat and Incheon are
    real macrotidal gulfs where neither atlas is checked.
  - No gauge override of the levels; r carries the gauges' information only through the surge.
  - The hydrographs of the 15 COAST-RP-fallback stations are Gaussian on the FES tide.
  - Hydrograph maxima off time 0 (4.1% overall, 3.2% at RP100; 64% one day away, 19% one
    tidal cycle): a neighbouring high water of the diurnal inequality or spring-neap cycle is
    higher than the centre one; median excess 1.6 cm, 99% < 21 cm. Kept: the maximum is the RP
    level, which is what SFINCS and B use; recentring the window on the maximum would only be
    cosmetic.
- **Per-RP hydrograph generator** (`src/storm_tide_hydrograph.rp_hydrograph`, generator_smooth;
  INPUT_ANALYSIS.md §8.3): its flood volume behind the test 11 sills is within 10% of real events
  at RP 2 and 10, although it falls short on overflow above 0.9 x peak (-20 to -40% on 2-4 m and
  >4 m coasts, never over). Known, not pursued: RP 2 T(0.25) ~30% short (costs 1-9% flood
  volume); the low microtidal troughs belong to GTSM's surge deficit (test 9). Against COAST-HG
  (fig18) its peaks are 2-4x wider (median T(0.25) 12 vs 6 h at RP100), which lowers B's
  multiplier.
- **FES version rule** (decided, INPUT_ANALYSIS.md test 6, §8.2): FES2014 replaces FES2022 where
  the two atlases' MHWS differ by the ratio and minimum difference of
  `tide_prediction.version_flag`; it reaches production through the product's levels. How far a
  gauge value reaches is undefined. Moderate inlet errors (0.25-1 m) stay uncaught.
- **Rerun after the 2020 baseline offset** (`boundary_conditions.slr_baseline_offset_m`, 0.049 m
  regionalised): `run_preparation.py boundary_conditions`, then every tile simulation and the
  analysis. Until then `boundaries.static_station_levels` raises on the old MDT station file.
  The SFINCS calibration runs predate the offset (forcing ~5 cm lower); recalibrating fsf should
  use regenerated forcing (`boundary_test/INPUT_ANALYSIS.md` §7).
- **Return periods above 250 y: tail of the storm-tide product.** COAST-RP grows by 0.11 m per decade
  of RP up to RP100 and 0.083 m from RP500 to RP1000 (median of 22,670 stations); RP1000 - RP100 is
  0.79 x RP100 - RP10 (0.72 above 30 deg latitude, 0.94 below; < 0.5 at 16% of stations). The ETC
  tail extrapolates ~40 y of ERA5 by 25x, and the surge correction is validated to RP100 only.
  Keep RP250-1000 (EAI tail term, FLOPROS standards above 250 y) but flag them as extrapolated;
  extend `surge_correction.RPS` to 1000 and compare the convolution's GPD tail against COAST-RP's.
- **Open input-assessment questions** (FES-tide recombination of COAST-RP, surge deficit, FES
  basin-head rule, TC hydrograph source; absolute MSL + MDT deferred): `boundary_test/INPUT_ANALYSIS.md` §8.
- **Duration correction B: reference duration and one-sided clip.** The water stays above MSL
  for more than 12 h in a third of observed annual maxima (microtidal median 20 h, macrotidal 7 h).
  B's T is cumulative over the hydrograph window, and SFINCS shows that the absolute duration,
  not the shape, sets the reach. On synthetic SFINCS floodplains with fsf refitted per form,
  m proportional to T^-alpha without the lower clip fits best; the production form (t_ref 12 h,
  clip >= 1) is close; "the hydrograph's longest exposure" and "T of the calibration hydrograph
  at the same head" are worse. A constant t_ref is absorbed into fsf once the clip goes.
  Decided: lower clip dropped (in code), alpha 1, hydrograph source the per-RP generator (COAST-HG's
  neighbouring high waters are too low: B's T 5 h vs 15 h observed). Needs the recalibration of
  fsf on the 498 SFINCS tiles (§0.3-0.4).
- **COAST-RP at TC-dominated stations** is 0.2-0.25 m below GESLA GEV even with its TC
  component (Japan/E Asia, US Atlantic+Gulf); GESLA records may also undersample TCs.
  Test 9 answers it: the skew-surge ratio correction halves the RP10 bias there but leaves RP100
  at -0.10 m (MAE 0.32 m), so TC stations need the STORM-based route of test 10.
- **fsf grid between 7.5 and 9:** the optimum may lie between the tested values (bias crosses 1
  near 8.7). Refine only if the bias matters more than the last 0.005 CSI.
- **HPC calibration tooling uses neither the static level nor B:** `sfincs_tiles/run_calibration_models.py` calls
  `flood_depth_dense` without the static level or B, so stage-1/2 calibration results describe the
  old physics. Wire it when `sfincs_tiles/` may be touched.
- **`run_eikonal_dag.py` OOM tile split:** `tile_split.split_tile` reads the mask from
  `model_outputs/<tile>/inputs`; with `preprocessing_inputs_dir` set it raises FileNotFoundError
  instead of splitting. Pass the inputs dir if splits are needed in production.
- **Permanent inundation p = 1 anchor:** implement with the static-flood extent of the
  two-component solve (flood_depth_dense diagnostic `n_static_wet_cells` only counts it now).
- **Tide in the static level:** decide later whether S = MDT + SLR (current choice) should also
  include a tidal high water (MHHW). Needs a tide dataset (GTSM tide statistics or FES2014); COAST-RP
  cannot separate tide from surge.
- **`protection.default_rp: 5`** for geogunits without FLOPROS values vs the RP2 minimum: keep 5?
- **Flood-fraction rule never exposes saturated cells:** where the design event already floods the
  whole cell (ff = 1), no larger event or SLR can exceed the design flood fraction, so the cell stays
  protected forever. Bangkok case: 33% of the population in the design floodplain sits in such cells.
  Production uses this rule; the water-level rule does not have the flaw.
- **Water-level rule + binary breach is harsh under SLR:** over all COAST-RP stations the median RP100
  standard drops to "fails every year" by ~0.25 m SLR (RP1000 by ~0.4 m), and a defence then counts
  as fully breached. Bangkok: protect at the +1 m design equals no adaptation from +1.5 m. Freeboard
  or an overtopping volume would soften this before the water-level rule is adopted.
- **SLR fingerprints scale vertical land motion with global SLR (production issue, left as is for
  now):** `prepare_boundary_conditions.compute_slr_fingerprints` takes the nearest AR6 point within
  1 deg (tide gauges included), whose 2100 total relative SLR contains VLM, and scales it by T / GMSL.
  Upper Gulf of Thailand stations (13.46 N, 100.2-100.7 E) get 5.16 m at T = 1.5 m; the global
  0.1-99.9 percentiles are -3.9 / +4.3 m. Affects both propagation variants.
- **Materialized configs on the cluster:** re-materialize after pulling (new keys `protection.rule`,
  `protection.minimum_rp`, `flooding.slr_propagation`, `flooding.duration_correction`, new paths).
- **Cluster runs (not planned for now):** SFINCS at +1 m / +2 m SLR (benchmark for 1a) and at
  RP10 / RP1000 on ~100 tiles (transferability, critique 1b, now with B on; RP100 COAST-HG shape
  shifted to each RP peak).

## Known behaviour

- `friction_scale_factor` (config.yml) is a bare per-metre rate, rescaled by a tile's own
  `pixel_size_m` (eikonal_runner.py) - production 0.25 x 30 (DeltaDTM) = 7.5 effective, with
  correction B. Outputs kept as computed at effective 9.0 (deltas, bangkok_chao_phraya; pinned as
  0.3) and effective 30.0 (calibration_500_tiles; pinned as 1.0) predate this rescale and are
  unaffected by it (always 30 m tiles); each config says so.
- A hop>=1 triple whose neighbour output is missing fails without writing output. Local multi-hop
  `snakemake simulate` / `run_eikonal_dag.py` can therefore fail when Snakemake schedules a hop-1
  tile before its neighbours (the rule does not declare neighbour inputs); rerun, or run wave by
  wave as `run_delta_floodmaps.py` does.

## Local-input validation (branch `local_input_validation`)

Plan: run the eikonal model with local high-res inputs (England first) against the same
tile grid as the global-input run, to separate model skill from input-data skill. Sections
1-2 done; Section 3's core mechanism done; Section 4 done (preprocess verified end-to-end
for all 15 fine tiles, both configs); Section 5 (England, boundary-forcing bridge through
scoring) done for the Isles of Scilly + Wash hop test. Section 6 (Wales, Severn Estuary
sample) done. User's stated replication order: Wales, then Scotland, then SF Bay/New
Brunswick - Section 7 (Scotland) started then PAUSED before any simulation (boundary-forcing
approach superseded, see Section 7); New Brunswick skipped for the same reason (user,
2026-11-06: "New Brunswick is actually the same case" - also a boundary-only local swap).
Section 8 (SF Bay, Alameda County sample) scored, with a real methodological caveat on the
depth-error numbers - see Section 8.

**Section 1 `[ASSERTED]` items in `data_catalog_local_input.yml` - NEEDS VERIFICATION:**
- `ea_coastal_design_sea_levels`: the "Copyright Associated British Ports 2010-2014" access
  constraint on the dataset page may restrict redistribution of derived outputs - not checked.
- `wales_lidar_dem`: RESOLVED (2026-11-06) - read `LiDAR_Guidance_2018.pdf` directly (the
  Read tool handles PDFs natively; no poppler-utils needed). Confirms ODN ("elevations
  recorded above Ordnance Datum Newlyn") and OGL ("we publish the data under the terms of
  the Open Government Licence") - both now set `native_datum: native`/`source_license` in
  the catalog entry, `[ASSERTED]`/NEEDS VERIFICATION markers removed.
- `sf_bay_lidar_dem`: source_license asserted as US public domain, not confirmed against this
  product's own terms page.
- `new_brunswick_sea_level_rise_reports`: vertical_reference asserted as CGVD2013 (matching
  the existing benchmark entry) and source_license, both unconfirmed against the two source
  PDFs; source_version needs each PDF's own publication date; the extraction script itself
  does not exist yet (re-check `review-critiques`'s `boundary_test/` for reusable PDF/tide
  tooling before writing one).

**Section 3 (vertical datum bridge to GOCO06s) - done:**
`src/vertical_datum.py`'s `compute_vertical_datum_offset`/`_at_points` (closed `DATUM_BRIDGES`
registry: `egm2008`, `odn_osgm15`; anything else raises `NotImplementedError`), wired into
`extract_dem`'s new `native_datum` parameter. `geoid_goco06s_absolute.tif` cached in
`inputs/ICGM/` (rule `compute_goco06s_absolute_raster`). Sign convention for `odn_osgm15`
confirmed against PROJ's own registered `EPSG:4937->5701` transform (3 points: London,
Liverpool, Portsmouth - offsets -1.12/-0.94/-1.45 m, N_OSGM15 and N_GOCO06s both ~45-53 m,
physically plausible). Regression tests: `tests/vertical_datum_validation/`.

**Section 3 - a real bug found and fixed (2026-11-04):** the EA WCS service's "no survey
coverage here" response (e.g. a 5km square entirely over open sea) is an all-zero raster
with NO nodata tag at all, not a properly nodata-flagged empty raster -
`preparation/ea_coastal_dtm.py`'s own comment wrongly assumed uniform nodata tagging across
every downloaded tile. This affected not just any future conversion script but the
production `dtm_1m_2022_coastal.vrt` itself (confirmed by inspecting its XML: GDAL's
per-source `<NODATA>` override only masks pixels equal to the real tiles' sentinel, not a
degenerate tile's literal 0.0) - so `england_lidar_dem` was silently exposing ~35% of
England's coastal tiles as a fake near-zero "underwater plain" to any reader. Fixed:
`mosaic()` now drops any tile whose band has no nodata value set before building the VRT;
regenerated `dtm_1m_2022_coastal.vrt` on `<data_root>` (3661 real tiles kept, 1944 dropped).
Source `.tif` tiles themselves untouched.

**Section 3 - output products done (2026-11-05):**
`preparation/convert_england_dtm_to_goco06s.py` (opt-in `run_preparation.py` step
`england_dtm_goco06s`) converts the corrected VRT's 3660 real tiles one EA 5km tile at a
time: full-res (1m) GOCO06s-referenced DTM + mosaic VRT (`lidar_dem_goco06s/dtm/`,
`england_dtm_goco06s.vrt`, 207 GB on `<data_root>`), and a 100m block-mean-coarsened diff
raster + mosaic VRT (`lidar_dem_goco06s/diff/`, `england_dtm_goco06s_diff.vrt`, 49 MB -
the offset is smooth at GOCO06s's own ~67km half-wavelength, so a full-res diff would be
pure storage cost with no extra signal) plus a `README.md` recording the formula and grid
versions. Diff range over all valid cells: -1.781 to -0.373 m, mean -1.027 m - smooth,
decimetre-to-~1m order, no step-edge artefacts, consistent with the 3-point spot-check.

**Section 3 - still open:**
- `navd88_geoid18` (SF Bay) deliberately not implemented - needs the mandatory NAD83(2011) ->
  ITRF2014 Helmert shift (~0.5-1m in California) the plan's doc flags as non-negligible.
- The EA boundary-shapefile point bridge has its sampling primitive
  (`compute_vertical_datum_offset_at_points`) but the actual script reading the shapefile and
  excluding Stornoway/Lerwick/St Mary's is Section 5's `cache_local_waterlevel_stations.py`,
  not written yet.

**Section 4 (tile subsetting + per-tile dem_source) - done (2026-11-05):**
`preparation/build_england_validation_tiles.py` selects production tiles intersecting
England's dissolved `coastal_tiles.gpkg` footprint, closes over hop dependencies
(`select_country_tiles._hop_distance_closure`, unmodified) -> 28 tiles, all hop 0, 0 pulled
in - and writes `processed_inputs/mask/validation/england_local_tiles.gpkg` (shared by both
`england_local_input.yml`/`england_global_input.yml`, so the comparison stays
apples-to-apples per D5) plus `england_dem_source_manifest.csv` (28 `england_lidar_dem`, 0
`deltadtm` - no hop-dependency tile was pulled in here). Snakefile loads the manifest once
(`_dem_source_for_tile`/`_dem_source_meta_for_tile`, reading `native_datum`/`pixel_size_m`/
`hybrid_grid_raster_path` from the resolved dem_source's own catalog meta - never three
independent values that can drift apart); wired into `extract_dem`'s and `run_eikonal`'s
`params:` in `preprocessing.smk`/`simulation.smk`. `extract_dem_mask`/`compute_friction`
unchanged (mask stays global `deltadtm_mask` always, Section 4b - no new
mask-reconciliation code needed, extract_dem's existing nodata-aware gap-fill already does
the right thing cell-by-cell). `get_data_catalog`'s list-of-paths support (Section 1) needed
one more fix: the Snakefile's own `_data_catalog` build did `os.path.join(workflow.basedir,
<string>)`, which raises on a list - now maps over the list first. Verified: both
`england_local_input`/`england_global_input` dry-run to the identical 1401-job plan (28
tiles) on the identical `tile_grid.path`; `england_lidar_dem`'s catalog meta resolves
correctly through the merged two-catalog list (`native_datum=odn_osgm15`, `pixel_size_m=1`,
`hybrid_grid_raster_path` formats to the real path); `deltadtm` falls back to
`egm2008`/`30.0` with no such meta key.

**Section 4 - investigated, not a bug (2026-11-05):** `coastal_tiles.gpkg` lists 6290
candidate 5km tiles but only 5604 `.tif` files exist on disk (3660 real + 1944 degenerate);
686 candidates were never downloaded at all (604 of them inside the 28 selected production
tiles). Live-tested the EA WCS service directly: 18/18 sampled "missing" tiles return an
identical, reproducible HTTP 500 "Internal server error" right now, while known-good tiles
and `GetCapabilities` succeed normally interleaved with them - ruling out a transient
network blip or an old crashed run. Mapping the 686 against the candidate footprint shows
they trace a thin ring along the buffer polygon's own outer/inner perimeter (`coast.boundary
.buffer(30000)` is a ~60km-wide band; these sit at its far edge), 15-30km from the true
coastline in every case, not along the coast itself - consistent with EA's WCS mosaic having
no granule to serve at that liminal edge (500, not even a blank raster like the degenerate
tiles get). Not critical to the England comparison; no fix made.

**Section 4 - D2/D5 revised, a new fine tile grid replaces the production-tile-subset
approach (2026-11-05):** the first real (non-dry) `extract_dem` run (`england_local_input`,
tile 1750) crashed with `ArrayMemoryError: Unable to allocate 68.7 GiB` - checked and this
was systemic, not a one-off: EVERY one of the 28 selected production tiles is ~50-190GB at
England's native 1m resolution (production's `tile_grid` is sized for DeltaDTM's 30m global
data, ~25M cells/tile; the same bbox at 1m is a 900x pixel-count blow-up). Worse than a
memory bug: even a memory-safe windowed read would leave the eikonal solver needing
~1-1.6 BILLION cells per tile at full 1m resolution - computationally infeasible regardless
of memory. Explicit decision (user): revise D2/D5 rather than downsample England's LIDAR or
only-partially-solve-at-1m - reuse the production tile grid's SIZE PHILOSOPHY (budget_cells,
~25-50M cells/tile) but not its GEOMETRY, via a new, purpose-built, much simpler tile
generator for this one already-known-connected coastal area:

`preparation/build_england_fine_tiles.py` (`--production-tile-id <id>`) tiles one selected
production tile's REAL (non-degenerate) EA 5km cores directly (no new grid-alignment scheme
needed - EA's own existing 5km squares from `coastal_tiles.gpkg` are the cores), pads each by
1km on every side (~7km domains, ~49M cells at 1m - lands almost exactly on the user's ~50M
target: sqrt(50M)=7071m, 7071-2*1000=5071m core, i.e. EA's 5km unit almost exactly), and
computes hop_distance by reusing `connectivity_tiling.compute_hop_distances` **UNMODIFIED**
against the existing global DeltaDTM mask (per the user's explicit request to review/reuse
that algorithm rather than invent a new one). Deliberate simplifications vs. production's
full pipeline, documented in the script's own docstring: no Phase 1/2 connectivity graph
(every tile gets the same `component_id`, since they all come from one already-connected
production tile - `compute_hop_distances`'s own adjacency/BFS still runs for real); no Phase
3/4 budget-splitting/merging (cores are EA's own squares, not recursively derived); no Phase
6 hop=0-only pad (every tile is padded uniformly instead, per the user's overlap request).
Writes `processed_inputs/mask/validation/england_fine_tiles.gpkg` +
`england_fine_dem_source_manifest.csv` (all `england_lidar_dem`, by construction);
`england_local_input.yml`/`england_global_input.yml` repointed at these (still the exact
same file for both configs, preserving D5's apples-to-apples intent at the new scale).

First real run, production tile 3263 (Isles of Scilly, smallest/cheapest of the 28): 15 real
EA tiles -> 15 fine tiles, **all hop=0** (every one independently touches ocean - this
specific footprint never exercised hop>=1 propagation, though the mechanism ran for real,
including the unreachable-domain-drop path). Real `extract_dem` for fine tile_id 0 under
both configs succeeded in ~10s each, no memory issue. Result is a clean illustration of
exactly what the framework is for: DeltaDTM (global, 30m, EPSG:4326) sees this small island
as essentially flat open water (0.0-1.11m, mean 0.002m); the local LIDAR DEM (1m, EPSG:27700,
GOCO06s-corrected) shows real terrain (-6.47 to 18.89m, mean -3.49m, 0% nodata, no spurious
99m hard-fill cells).

**Section 4 - real bugs found running the full `preprocess` chain for all 15 fine tiles
(2026-11-05):** `extract_dem_mask.py`/`compute_friction.py` both built their HydroMT `bbox`
as `list(dem.raster.bounds)` - the DEM's own native-CRS bounds, unreprojected. HydroMT's
`DataCatalog.get_rasterdataset(bbox=...)` always interprets `bbox` as EPSG:4326 regardless of
the source's CRS, so this was a silent no-op for every `deltadtm`-sourced tile (already
EPSG:4326) but wrong for England's EPSG:27700 LIDAR DEM: `extract_dem_mask.py` raised
`hydromt.exceptions.NoDataException: No data was read from source: deltadtm_mask` outright;
`compute_friction.py` had a pre-existing `except NoDataException` fallback that silently
caught the SAME bug and produced uniform default-friction output instead of raising - would
have shipped wrong (but not obviously wrong) friction rasters for every local-input tile.
Fixed with a new `dem_bounds_4326()` helper in `src/rasters.py` (reprojects
`dem.raster.bounds` via `rasterio.warp.transform_bounds` to EPSG:4326) and updated both call
sites to use it instead of `list(dem.raster.bounds)`. `sfincs_tiles/regenerate_dem_mask.py:49`
has the identical pattern but is untouched (always deltadtm/EPSG:4326 there, so inert, and
`sfincs_tiles/` is "don't touch without asking").

**Section 4 - all 15 fine tiles verified, full `preprocess` target (2026-11-05):** ran
`snakemake preprocess` (not just `extract_dem`) for all 15 fine tiles under both
`england_local_input` (743/743 jobs) and `england_global_input` (748/748 jobs), 0 errors,
after the two CRS-bbox fixes above (found by this exact run - no prior test had reached
`extract_dem_mask`/`compute_friction` with a non-EPSG:4326 DEM source). Cross-tile sanity
check: friction now varies correctly by land cover on 11/15 tiles (2-9 distinct values; the
remaining 4 are small near-all-water tiles with little land-cover diversity, not a bug) -
confirms the fix, not just a lack-of-crash. DEM comparison across all 15 shows two distinct,
consistent failure modes of the global DEM: tiles 0-4, DeltaDTM reads as flat open water
(mean ~0.00-0.02m) where local LIDAR shows real terrain up to 19-27m; tiles 5-10/13-14,
DeltaDTM caps at exactly 30.27-30.28m (its own documented `elev_threshold_m` validity
ceiling) where local LIDAR correctly shows true summit heights of 43-49m above it; tiles
11-12 are genuinely near-all-water in both (not a discrepancy). A stale `model_outputs/1750/`
directory under `england_local_input` (partial leftover from the pre-route-3 giant-tile
attempt, abandoned before any `dem.tif` was written there) was found and deleted.

**Section 4 - still open:**
- All 15 fine tiles came back hop=0 for production tile 3263 - hop>=1 propagation has not
  actually been exercised yet. If that matters, add a second `--production-tile-id` run over
  one of the estuary-heavy tiles (e.g. 2459/2460/2461, Humber/Wash) where an inland segment
  is more likely to need multi-hop seeding.
- Adding `dem_source`/`native_datum`/`pixel_size_m`/`hybrid_grid_raster`/
  `goco06s_absolute_raster` as new `params:` on `extract_dem` (and `pixel_size_m` on
  `run_eikonal`) is a real `params:` structure change - it will re-trigger
  extract_dem/extract_dem_mask/compute_friction/run_eikonal for every existing tile in every
  existing scenario the next time each one runs, with output content unchanged for every
  `deltadtm` tile (same precedent as the config-key rename note above - accepted).
- `preparation/build_england_validation_tiles.py` and its outputs (`england_local_tiles.gpkg`,
  `england_dem_source_manifest.csv`) are now superseded by `build_england_fine_tiles.py` for
  the actual local-input solve - left in place (still a valid "subset of production's own
  tile grid" mechanism, just not usable at England's native 1m resolution); not deleted since
  it is not clearly dead (a future coarser-resolution local-input source could still use it).

**Section 5 - boundary-forcing bridge, done (2026-11-05):** fixed `validate_country.py`'s
`_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"]` (pathlib `/` on a list raises TypeError;
same bug as the Snakefile's, same fix: join each entry individually). Added
`boundary_conditions.stations_cache_dir` as an optional override in `preprocessing.smk`
(defaults to the existing shared `{processed_inputs_dir}/WL_scenarios_cache`, same precedent
as `simulation.preprocessing_inputs_dir`) - needed because the global COAST-RP `SLR_0` cache
is shared across every scenario by design; without an override, writing England's own
EA-derived `SLR_0` stations to the default path would have corrupted that shared cache for
every other scenario (bangkok, production, `england_global_input` itself). `england_local_input.yml`
now sets this to an isolated `{root}/england_local_input/processed_inputs/WL_scenarios_cache`,
plus `boundary_conditions.waterlevel_names: ["SLR_0"]` and `adaptation.slr_intensities: []`
(D4 - both are needed: `WATERLEVEL_NAMES` in the Snakefile is their union, so leaving
`adaptation.slr_intensities` at its production default of `["SLR_500","SLR_1000"]` would have
pulled those back in with no local cache for them).

`preparation/cache_local_waterlevel_stations.py` (new, one-off, not Snakemake-tracked) reads
`ea_coastal_design_sea_levels` (3873 points, `t{RP}` columns = return-period levels in ODN,
confirmed to match `config.yml`'s own `return_periods` list exactly, no probability->RP
mapping needed), excludes the 3 named local-chart-datum groups (`location` in
`{"OUTER HEBRIDES", "Shetland", "St. Mary's (Scilly)"}` - Stornoway/Lerwick are those two
regions' own reference tide gauges, St Mary's is named directly; 442/3873 points excluded),
applies the `odn_osgm15` point-bridge (`vertical_datum.compute_vertical_datum_offset_at_points`,
offset range [-1.86, +0.17] m, mean -1.05 m across the whole GB-wide station set - the England
spot-check's -1.1 to -1.45 m range sits well inside this, the wider range here is expected
since GB-wide spans much further geodetically than England alone), and writes
`stations_{RETURN_PERIOD}_SLR_0.gpkg` (note: the Snakemake wildcard is the `RP<n>` form, e.g.
`RP100`, not the plain-int config-list form - caught by a first dry-run showing the real
rule's expected filename) for all 9 configured return periods under the isolated cache dir.
Ran for real (user permission granted) - 3431 stations/file after exclusion.

Verified: dry-run of `england_local_input`'s `preprocess` target now shows exactly 135
`extract_boundaries` jobs (15 tiles x 9 RPs), `waterlevel_name=SLR_0` only, zero
`cache_waterlevel_stations` jobs (correctly bypassed - the pre-seeded cache satisfies it).
Ran the real (non-dry) RP100+RP250/SLR_0 subset (30 jobs, per the plan's own "RP100 first"
step and the user's request to limit scope for the model-vs-model comparison) - all
succeeded. `england_global_input`'s own RP100/RP250/SLR_0 boundaries already existed from
the earlier full-preprocess run (unaffected - it still reads the real, un-isolated global
COAST-RP cache).

**Section 5 - a real, substantial, England-wide RP100 SLR_0 discrepancy found and mapped
(2026-11-05):** `preparation/compare_boundary_forcing_local_vs_global.py` (new) nearest-matches
every EA design-level point (post-exclusion, England-footprint-scoped: 1293 points) against
the real global COAST-RP SLR_0/RP100 cache, both bridged to GOCO06s. 1250/1293 matched within
~11km. Local is LOWER than global **everywhere** in England (diff range -3.01 to -0.30 m,
median -1.13 m, IQR -1.38 to -0.92 m) - never once higher. Figure (2-panel: England map
colored by the gap's magnitude, local-vs-global scatter with 1:1 line) and the matched-pairs
CSV written to `england_local_input/figures/boundary_forcing_local_vs_global_RP100_SLR_0.{png,csv}`.
The gap has real spatial structure, not noise: smallest (~-0.3 m) on the Solway/Cumbria coast
(NW England), largest (-3.0 m) at the Thames Estuary, broadly mid-sized (~-1 to -1.5 m) along
most of the rest of the coastline - consistent with a systematic datum/methodology difference
(onto which an estuary-specific effect stacks at the Thames) rather than a spatially random
bug.

**Section 5 - gap explained: ruled out MDT, confirmed it's the SAME odn_osgm15 term on both
legs, and it cancels in freeboard (2026-11-05):** sampled AVISO MDT (`mdt_cnes_cls22`, the
exact correction COAST-RP's own pipeline adds) at all 1250 matched England points: only
0.03-0.11 m (median 0.064 m), moderately anti-correlated with the gap (r=-0.56) - far too
small to explain a ~1.1 m median gap, so MDT is not the driver (adding it would help by only
a few cm). Resampled the local LIDAR DEM onto DeltaDTM's native 30m grid for all 15 fine
tiles and compared "fair-overlap" pixels (real, non-degenerate terrain on both sides, i.e.
excluding DeltaDTM's near-zero/no-coverage cells and its 30.27/30.28 m cap): on the 6 tiles
with enough reliable samples (tiles 5-10, thousands of pixels each), local LIDAR elevation is
ALSO systematically lower than DeltaDTM, by -0.6 to -1.2 m - the same sign and size as the
boundary-level gap (the small-sample tiles 0-4/13 are noise, pulled from DeltaDTM's
near-degenerate edge, not evidence of anything). This is exactly expected: the DEM's
`odn_osgm15` bridge and the boundary's `odn_osgm15` bridge apply the IDENTICAL correction
term (mean -1.027 m for the DEM diff raster, Section 3; -1.046 m for the England boundary
points here) - both legs of England's local dataset carry the same shift relative to the
global system.

The decisive check: FREEBOARD (water level minus DEM elevation - the quantity that actually
drives flooding) on the same 6 robust tiles: tile 5 -0.23 m, tile 6 -0.29 m, tile 7 -0.36 m,
tile 8 +0.08 m, tile 9 -0.01 m, tile 10 -0.25 m - an order of magnitude smaller than either
individual ~1 m bias, i.e. the shared `odn_osgm15` term very largely cancels once DEM and
boundary are differenced. The Thames Estuary's -3.0 m boundary outlier was not re-tested this
way (no DEM fair-overlap pixels there) and likely still carries a real, separate
estuary-specific effect (surge funnelling and/or a poor nearest-station match in a narrow,
fast-varying channel) on top of the baseline.

**Decision: England's local-input DEM and boundary levels are no longer bridged to GOCO06s -
both stay in their native ODN datum.** Since both legs come from the EA/ODN family already,
bridging only adds the geoid grids' own bilinear-resampling discretisation noise to freeboard
without adding correctness - confirmed by the near-cancellation above. Implemented via a new,
explicit `"native"` entry in `vertical_datum.DATUM_BRIDGES` (a documented no-op returning a
zero offset, never a silent fallthrough) rather than skipping the bridge machinery ad hoc.
`data_catalog_local_input.yml`'s `england_lidar_dem.meta.native_datum` changed from
`odn_osgm15` to `native`; `cache_local_waterlevel_stations.py` now reads this same catalog
meta (rather than hardcoding `odn_osgm15`) so the DEM and boundary legs can never drift
apart again. Snakemake's `params`-change rerun-trigger picked up the DEM change automatically
(`extract_dem`'s `native_datum` param); the boundary cache was regenerated by hand (it isn't
Snakemake-tracked) and `extract_boundaries` picked up the new mtimes. Re-ran both for the
15 tiles / RP100+RP250 / SLR_0 (user permission granted).

**This decision is England-specific and must NOT be generalized without checking each
region's own DEM/boundary source pairing:**
- **SF Bay**: local DEM (NOAA LIDAR, NAVD88/GEOID18) + GLOBAL boundary forcing (COAST-RP,
  stays global per the plan) - a genuinely MIXED pairing. The two legs do NOT share a native
  datum, so bridging the DEM to GOCO06s (the planned `navd88_geoid18` bridge, including its
  mandatory NAD83(2011)->ITRF2014 Helmert shift) remains mandatory, not optional, when this
  region is implemented - review this specifically, don't assume England's shortcut applies.
- **Scotland**: GLOBAL DEM (DeltaDTM, stays global per the plan) + local boundary forcing
  (reuses `ea_coastal_design_sea_levels`, native ODN) - also a MIXED pairing, opposite
  combination from SF Bay. Bridging the boundary points to GOCO06s remains mandatory there
  too (the DEM and boundary do not share a native datum) - review before wiring Scotland's
  boundary-forcing script, don't assume `native_datum=native` applies.
- The general rule going forward: `native_datum="native"` is only valid when EVERY numeric
  input sharing one freeboard calculation (DEM and boundary) comes from the same native,
  un-bridged source; any local/global mix needs the real bridge on whichever leg doesn't
  already match the other's datum.

**Section 5 - comparison script removed (2026-11-05):** `compare_boundary_forcing_local_vs_global.py`
(the one-off diagnostic that produced the gap/freeboard findings above) is deleted now that the
investigation it was for is resolved - its job was done. Its CSV/figure under
`england_local_input/figures/boundary_forcing_local_vs_global_RP100_SLR_0.{png,csv}` are stale
leftovers (computed against the old, now-superseded `odn_osgm15`-bridged local values) - not
deleted yet, flag for cleanup.

**Section 5 - eikonal solve run for real, both configs, RP100+RP250/SLR_0 (2026-11-05):** ran
`run_eikonal` for all 15 fine tiles under both `england_local_input` (60 jobs - 15
`extract_dem_mask`/`compute_friction` re-derived first, picked up by `dem.tif`'s new mtime
from the native-datum change above, cheap and output-content-unchanged; 30 `run_eikonal`) and
`england_global_input` (30 `run_eikonal`, no re-derivation needed - unaffected by the datum
change). All succeeded; depths are physically sensible (e.g. local tile 9 RP100: 4.1% wet,
0.01-8.33 m).

**Section 5 - tile-geometry sanity check: fine tiles vs. production's own tiles, do they
agree? Yes (2026-11-05):** before trusting `england_global_input`'s fine-tile solve as "the"
global baseline, checked whether using a small (~7km) fine tile instead of production's own
much larger tile (sized for 25-50M cells) changes the global-DeltaDTM flood result at all for
this geography - D5 only requires local/global to share a tile grid, not that the grid match
production's. Clipped production's own merged, continuous chunk rasters
(`merged_results/chunks/waterdepth_{N45W010,N50W010}_RP{100,250}_SLR_0.tif`, float32 m,
mosaicked across the N45/N50 chunk boundary the fine tiles happen to straddle) to each fine
tile's bbox and compared against the fresh fine-tile solve. Agreement is close on every tile:
wet-cell counts match within a few percent (tile 5 is the one outlier, +9% wet cells/-0.09 m
mean depth in the fine-tile version; every other tile is within ~1%), mean depths agree to
1-2 cm; tiles 11/12 are zero-flood in both. Conclusion: for Scilly's small, naturally
self-contained islands, tile geometry doesn't materially affect the global-input result - the
fine-tile mechanism (Section 4) is validated against production's own real output, not just
internally self-consistent. Not guaranteed to generalize to a larger or more
hydraulically-connected geography (e.g. an estuary) - re-check if/when a second
`--production-tile-id` test is run (Section 4's still-open hop>=1 item).

**Section 5 - the actual deliverable is scoring against the national benchmark, not a
model-vs-model diff (correction, 2026-11-05):** the comparison this framework is for is
local-input vs. global-input, EACH scored against England's real FZ3-sea flood map
(HR/FAR/CSI/bias, the same metric `rescore_england_fz3.py` already uses for production), not
a direct raster diff between the two eikonal runs.

**Section 5 - a real bug in `merge_chunk`/`src/merge.py`, found running it on England's local
tiles for the first time (2026-11-05):** `merge_tile_rasters_chunk`'s whole design
(module docstring) assumes every tile natively shares one EPSG:4326 pixel grid with the
chunk grid itself - true of every deltadtm-sourced tile, never checked. England's local
tiles are natively EPSG:27700 (BNG, 1m) - `chunk_bounds` (always degrees) got silently
treated as BNG metres, producing a degenerate 5x5-pixel output with zero real data (caught by
the scorer reporting "0/2 chunks had real model data", not by any error). Fixed:
`merge_tile_rasters_chunk` now checks every tile's real CRS and raises clearly (rather than
silently producing garbage) if any tile isn't EPSG:4326 - production and `england_global_input`
are completely unaffected, same code path as always.

First fix attempt (reverted) reprojected the merged BNG raster onto the EPSG:4326 chunk grid -
technically correct but reprojecting a ~479M-pixel array (the 15 tiles' union footprint at 1m)
via `rasterio.warp.reproject` in one shot, x4 concurrent jobs, was heavy enough to likely be
swapping rather than computing (decelerating, non-stalled growth over many minutes). Replaced
with the user's better alternative: **never reproject the big raster at all** - the national
benchmark is a vector, reprojecting a few polygon vertices into the tiles' own native CRS is
trivial by comparison. New `merge_tile_rasters_native` (src/merge.py) merges tiles sharing one
native CRS at full native resolution, no grid/CRS change, finishing in seconds (verified:
EPSG:27700, 1m, 22402x21390 px, 411M valid / 5.18M wet px, depths 0.01-8.77m). Threaded an
optional `dst_crs` through `validation.read_permanent_water_mask`/`read_country_mask`/
`window_scoring_domain` (default unchanged: "EPSG:4326", zero behaviour change for every
existing caller) so a native-CRS window can still correctly query the global, always-4326
permanent-water/geogunit catalog rasters (`bbox` stays EPSG:4326 - the catalog read extent -
the window's own `out_transform`/`dst_crs` is what differs).

`preparation/merge_england_local_tiles.py` (new) merges england_local_input's 15 tiles via
`merge_tile_rasters_native`, one raster per RP, under `merged_results/local_native/` - not
Snakemake-tracked, rerun manually after a new eikonal solve. `validation/rescore_england_fz3_local.py`
(new, companion to `rescore_england_fz3.py`) scores that native-CRS raster directly against
`england_ea_floodzone3_sea`, reprojecting only the benchmark vector (and block bbox corners,
for the catalog lookups) into EPSG:27700 - mirrors the original's block-wise (8192px) design
for the same memory reasons. Also fixed the same `pathlib /`-on-list bug in
`rescore_england_fz3.py` (needed to run it against `england_local_input`'s list-valued catalog).

**Section 5 - first real local-vs-global national-benchmark comparison, Isles of Scilly
(2026-11-05):** ran both scorers (native for local, the original unmodified one for global -
its EPSG:4326 merged chunks from Section 5's earlier tile-geometry check are unaffected by any
of today's changes).

| | local (EA LIDAR + EA boundary) | global (DeltaDTM + COAST-RP) |
|---|---|---|
| RP100 HR / FAR / CSI / bias | 0.884 / 0.386 / 0.568 / 1.439 | 0.721 / 0.422 / 0.473 / 1.246 |
| RP250 HR / FAR / CSI / bias | 0.899 / 0.389 / 0.572 / 1.471 | 0.734 / 0.420 / 0.479 / 1.265 |

The two scorers' printed `benchmark_wet_km2` differ a lot (3.3 vs 13.2 km²) but this is a
diagnostic-only artefact: the ORIGINAL scorer's `benchmark_wet_km2`/`model_wet_km2` accumulate
over a whole 8192px block (~250km wide at 1 arcsec) whenever the block has ANY real model
data, unmasked by `model_domain` - harmless at production's continental scale, but inflates the
diagnostic (not the score) when almost all of a block is nodata around one tiny island cluster.
HR/FAR/CSI/bias themselves ARE correctly restricted to `domain_mask` (`model_domain & in_scope`)
in both scorers.

**Section 5 - methodology correction: the standard scoring uses a 1km false-positive buffer,
the above doesn't (2026-11-06).** The user flagged the global numbers above as off versus
previously-reported England figures (HR .70-.71, FAR .36-.37, bias 1.09-1.12, CSI ~.5).
Cause: `rescore_england_fz3.py`/`rescore_england_fz3_local.py` count a false positive anywhere
in the model domain; the project's actual standard (Bates et al. 2021, following Wing et al.
2017, already implemented in the existing `validation/rescore_1km_fp_buffer.py` - not applied
by mistake in the runs above) restricts false positives to within 1 km of the benchmark's own
flood extent, which HR doesn't depend on but FAR/CSI/bias do. Ran `rescore_1km_fp_buffer.py`
for `england_global_input` - buffered RP100: HR=0.721 (unchanged, FP-independent),
FAR 0.422->0.386, CSI 0.473->0.496; RP250: FAR 0.420->0.385, CSI 0.479->0.503 - now matching
the remembered reference closely. Wrote `validation/rescore_1km_fp_buffer_local.py` (new,
native-CRS counterpart, same relationship to `rescore_england_fz3_local.py` as the original
has to `rescore_england_fz3.py` - simpler buffer-mask step since a projected CRS's pixel size
is already in metres, no per-block latitude correction needed) and ran it for
`england_local_input`. Corrected, apples-to-apples comparison (bias derived algebraically for
the global side, `bias = HR / (1 - FAR)`, since `rescore_1km_fp_buffer.py` doesn't print it -
verified exactly against the local side's own printed bias first):

| | local, buffered | global, buffered |
|---|---|---|
| RP100 HR / FAR / CSI / bias | 0.884 / 0.347 / 0.602 / 1.353 | 0.721 / 0.386 / 0.496 / 1.174 |
| RP250 HR / FAR / CSI / bias | 0.899 / 0.350 / 0.606 / 1.385 | 0.734 / 0.385 / 0.503 / 1.192 |

Local input now scores better on all three FP-dependent metrics too (not just comparable on
FAR as the unbuffered table suggested) - CSI 0.60 vs 0.50, FAR 0.35 vs 0.39 (lower is better),
alongside the already-clear HR advantage (0.88-0.90 vs 0.72-0.73). Local still over-predicts
more (bias 1.35-1.39 vs 1.17-1.19) - consistent with a model that captures more of the true
extent (higher HR) at the cost of some extra false-positive area, not a sign it's worse.

**Section 5 - still open:**
- Fix item above (list-valued `hydromt_data_catalog`) resolved `validate_country.py`/
  `rescore_england_fz3.py` only; `validate_england_fz2_band.py`, `rescore_1km_fp_buffer.py`,
  `validate_sf_bay.py`, `plot_agreement_map.py`, `plot_sf_bay.py` (and ~14 preparation/analysis
  scripts) have the same pattern, not yet hit/fixed - none has been called with a list-valued
  catalog yet (`rescore_1km_fp_buffer.py` ran fine above because it was only run against
  `england_global_input`, whose catalog is a plain string).
- Only RP100/RP250 scored so far (D5's own "RP100 first" step, kept narrow deliberately) - the
  full RP x SLR_0 grid (9 RPs) would need the same merge+score run repeated, plus (if ever
  wanted) a true FZ2 band score (needs RP1000, not yet solved).
- The benchmark's own `england` region bbox (`[-6.37, 49.88, 1.77, 55.82]`) cuts off the
  westernmost sliver of Scilly (our tiles reach -6.48°) - a small amount of local-input's own
  solved area is excluded from scoring by the benchmark's own extent, not a bug to fix here.
- Wales/SF Bay/Scotland/New Brunswick replication not started.
- Check whether `exposure.exceedance_threshold_m` (0.10 m, the damage/exposure-accounting
  threshold - distinct from `validation.primary_threshold_m`, 0.01 m, used for HR/FAR/CSI/bias
  scoring) is actually appropriate for England and Wales. Motivation: England's bias is quite
  strong (1.17-1.39 depending on scenario/buffering, this session's Section 5 results) and
  `primary_threshold_m`'s own config comment already notes CSI/HR improved monotonically down
  to 0.01 m in earlier Wales/Scotland testing of 0.10/0.05/0.01 m - i.e. a meaningful share of
  the signal sits in shallow depths between 0.01 and 0.10 m. Worth checking whether exposure/
  damage accounting at the coarser 0.10 m threshold is missing or mis-weighting that same
  shallow-flooding signal, not just the extent-scoring metrics.

**Section 4/5 - extending beyond the Isles of Scilly test to all of England (2026-11-06).**
User's framing: the single-production-tile test was always meant to validate the mechanism,
not the final scope - the same method should apply across all of England, which both gives a
real hop>=1 test for free and avoids a deduplication step.

Investigated first (not yet decided to build): naively looping `build_england_fine_tiles.py`
once per one of the 28 selected production tiles (the superseded per-production-tile
closure, `build_england_validation_tiles.py`) would NOT give one fine tile per EA core.
Checked geometrically: production's own tiles overlap each other substantially by design
(real shared area, not edge-slivers - e.g. up to ~14,000 km² between some pairs, matching
`merge_chunk`'s own max-combine philosophy). Of 6290 candidate EA cores, only 1522 intersect
exactly one of the 28 production tiles; **4630 (74%) intersect 2 or more** (max 6) - looping
per production tile would solve most of England 2-3x over.

**Decision: made `--production-tile-id` optional in `build_england_fine_tiles.py` - omitting
it runs every real EA core across all of England in ONE pass**, not per-production-tile.
This sidesteps the deduplication problem entirely (each core gets exactly one fine tile by
construction) and lets hop_distance come out whatever it really is, rather than assuming
every tile is hop=0 as the single-tile Scilly test did. Single-tile mode keeps writing the
original filenames (`england_fine_tiles.gpkg` etc., still what `england_local_input.yml`/
`england_global_input.yml` point at - unaffected); all-England mode writes distinctly-named
outputs (`england_fine_tiles_all_england.gpkg`) instead of silently superseding the
already-validated Scilly artifact - repointing the configs at it is a deliberate, separate
step once the full tile set is reviewed.

**First all-England build - real findings, not just confirmation:** 3660 real EA cores (every
real tile, matching the earlier VRT-regeneration count exactly - confirms no duplication),
**0 dropped as unreachable**, but hop_distance goes all the way to **hop=20** (not just 0/1):
982 hop=0, 488 hop=1, 435 hop=2, ... down to a long tail (654 tiles at hop 6-19, 1 at hop=20).
Most England coastal LIDAR squares sit slightly inland of where the coarse (500m) connectivity
mask resolves "ocean", so they chain through several neighbours before reaching the real coast
- this is physically plausible, not obviously a bug, but far more hop-chaining than the Scilly
test (all hop=0) suggested would be needed.

**Mechanical consequence, checked before trusting any real run:** `run_eikonal.py`'s own
docstring says the rule does NOT declare neighbour outputs as inputs, so Snakemake does not
auto-sequence hop levels - a naive full run would just fail every hop>=1 job whose neighbour
isn't done yet. The fix already exists in this codebase ("run multi-hop grids wave by wave, as
run_delta_floodmaps.py and the HPC wave barriers do") - not yet read/adopted for England;
needed before any hop>=1 England tile can actually be solved. The neighbour-seeding mechanism
itself (`collect_neighbor_wave_seeds`, reading a lower-hop neighbour's wet cells) is
dem_source-agnostic by design and should work unchanged for `england_lidar_dem` tiles, but has
never actually been exercised for one - the Scilly test never reached this code path (all
hop=0). Needs a real small-scale test (a few tiles spanning hop 0->1->2) before trusting it at
the full 3660-tile scale.

**Three refinements investigated, two implemented, one dropped, following the user's own
review of the first all-England build:**

1. *Dropped*: using the LIDAR DEM's own `nodata` as a free ocean-touch signal (confirmed
   empirically: real EA tiles have nodata over open sea, e.g. `SV81nw` 95% nodata, with real
   -often negative, intertidal- elevation elsewhere; DeltaDTM has no equivalent - it needs a
   separate mask file precisely because its own raster is NOT nodata-flagged over water).
   Not pursued: plain nodata also means "no bare-earth return" over inland lakes/wide rivers,
   which the DeltaDTM mask's semantic ocean/lake/river classes can distinguish and plain
   nodata cannot - England's 30km buffer plausibly contains some genuinely inland water, so
   this would risk misclassifying a lake-adjacent tile as ocean-connected.

2. *Implemented*: `compute_hop_distances`'s own ocean-adjacency check was the actual
   bottleneck (736s for 3660 domains) - not because of inherent algorithmic cost, but because
   `_mosaic_nearest_coarse` re-opened the same underlying 1deg mask tile from scratch on every
   single domain call, with zero reuse across nearby domains sharing one tile (real cost on
   the networked data_root - `retry_transient_io`'s own docstring already flags SMB blips on
   it). Fixed in `src/connectivity_tiling.py`: `_mosaic_nearest_coarse`/`_domain_touches_ocean`
   take an optional `open_cache` dict (default `None` keeps every existing caller, including
   production's own `build_tile_manifest.py` Phase 1 `has_floodable`, bit-identical - the
   read+resample call itself is unchanged, only file-open overhead is avoided); `compute_hop_distances`
   now opens one cache for its whole loop and closes it (`close_open_cache`) when done.

3. *Implemented*: a new max-waterlevel elevation pre-filter in `build_england_fine_tiles.py`
   (`_MAX_WATERLEVEL_M = 10.0`). Max design sea level across England's `ea_coastal_design_sea_levels`
   (native ODN, 3 local-datum-exception regions excluded) **at RP250 specifically - this
   validation framework only ever runs RP100/RP250, never RP1000, so the filter is sized
   against what actually gets solved, not a return period this framework doesn't use**: t250
   max 9.12 m, 95% CI upper bound (c2_t250) 9.54 m, rounded up to 10.0 m for a clean margin.
   Any EA core whose own minimum elevation sits above this can never flood under any scenario
   this framework runs, and is dropped before tiling - cheaper than, and done before, the
   ocean-connectivity check (no point hop-checking a tile that can't flood regardless of ocean
   adjacency). Uses GDAL's EXACT (non-approximate) `ComputeStatistics(0)` for the minimum, not
   the default approximate stats - confirmed empirically these differ meaningfully (-4.17 m
   approx vs -5.25 m exact on one real tile) and approximate stats can under-sample a real but
   spatially small low point, which is exactly the wrong direction for a filter whose failure
   mode must never be "wrongly excluded a tile that could actually flood". Costs ~0.5-0.7s/tile
   (~30-45 min one-time for 3660 tiles) - accepted as worthwhile given it runs once per rebuild
   and should materially shrink the final tile count.

**Section 5 - re-ran `build_england_fine_tiles.py` (all-England mode) with both fixes, done
(2026-11-06):** 3660 -> 1837 real EA cores (1824 dropped by the waterlevel filter), 1 further
dropped as unreachable -> **1836 final tiles**. hop=0 share nearly doubled (982/3660=27% ->
967/1836=53%) and max hop dropped from 20 to **13** - removing never-flooding inland tiles also
shortened the inland hop-chains they were part of. Hop-check time: 736s -> **141s** (fewer
domains and ~2.6x faster per domain from the caching fix alone).

**Section 5 - `collect_neighbor_wave_seeds` validated end-to-end for `england_lidar_dem`
tiles, done (2026-11-06):** read `run_delta_floodmaps.py`'s `run_simulate_stage` (groups tiles
by `hop_distance`, runs each wave's targets as a blocking `snakemake` call, stops immediately
if a wave fails - the same pattern `generate_hpc_simulation_jobs.py`'s wave tiers use for the
HPC path) and adapted it for a small, real, self-contained test cluster: 51 tiles near The Wash
estuary (Lincolnshire), hop 0-5, selected by bbox from the all-England build and hop_distance
**recomputed fresh for just this subset** (came back numerically identical - 13/10/11/6/6/5 -
confirming the box cuts no external dependency). New isolated scenario config
`england_hop_test.yml` (same pattern as `england_local_input.yml`, local-only - no global
counterpart needed since this tests the hop mechanism, not a local-vs-global comparison).

Ran `preprocess` (715 jobs; one transient `P:\` network disconnect mid-run killed the first
attempt at 339/715 - unrelated to hop logic, `--rerun-incomplete` resumed and finished cleanly)
then the wave-by-wave `simulate` for RP100/SLR_0: **all 6 waves (hop 0 through 5) completed
successfully**. Spot-checked every hop>=1 tile's actual output: 13 tiles got real, physically
varied flooding propagated from their lower-hop neighbour (wet fraction 0.06%-96%, depths
0.01-7.9 m) and 7 tiles came back correctly dry with an explicit, logged reason in
`skipped_tiles/` ("no non-zero water level found in any available overlapping earlier-wave
tile") - not a silent zero or a crash. Both branches of `eikonal_runner.run_tile_scenario`'s
hop>=1 logic (real seeds found vs. none found) are now confirmed working on real
`england_lidar_dem` data, for the first time - the Scilly test never exercised this at all.

**Section 5 - local-vs-HPC runtime estimate for the full all-England run, done (2026-11-06):**
Extrapolated from the Wash hop-test's real wall-clock (cores=4, same machine, no HPC): preprocess
took ~49 min for 51 tiles/714 input files; simulate (wave-by-wave, RP100/SLR_0 only) took ~15.7
min for 51 tiles over 6 waves. Scaling linearly by tile count to the full 1836-tile/13-wave grid:
preprocess ~**29 h**, simulate ~**9.4 h per RP**. For RP100+RP250 only (the pair the RP250
max-boundary-forcing filter actually needs): ~29 h + ~19 h = **~2 days** of continuous local
run; all 9 RPs would be ~**4-5 days**. Caveats: linear extrapolation from one small, hop-0-light
sample (13/51 tiles were hop=0 vs. 53% for the full grid - likely an overestimate, since more
tiles per wave doesn't slow a throughput-bound 4-core run down); real per-tile variance could
differ (the Humber/Wash estuary tiles were already flagged as slower, TODO.md:236); the small
test already hit one transient `P:\` network disconnect mid-run, and a 2-day unattended local
run has zero margin for that recurring. HPC would cut this to hours via
`generate_hpc_simulation_jobs.py`'s wave-tier mechanism (many tiles per wave run in parallel
across nodes instead of 4 at a time) - but the user may be fine accepting the ~2-day local
runtime instead while away on holiday soon.

**Section 5 - still open:**
- Full all-England scale (1836 tiles after the waterlevel filter, up to hop=13 waves): local vs.
  HPC execution path not yet decided (see runtime estimate above) - no HPC job generation
  attempted yet for this scenario, and no local run of the full grid started either.
- The Wash hop-test tiles/config/outputs (`england_hop_test*`) are a one-off validation
  artifact, not wired into any scoring - fine to leave as-is or clean up later, not blocking.

**Section 6 (Wales, Severn Estuary sample) - done (2026-11-06):** replicated England's
local-vs-global framework for Wales, scoped deliberately to one small sample (same staging
discipline as England's Scilly/Wash tests) - not the full-Wales grid.

`preparation/build_england_validation_tiles.py` deleted: dead code, superseded by
`build_england_fine_tiles.py` (its only other caller of `select_country_tiles
._hop_distance_closure` besides that function's own `main()`, which stays). New
`preparation/build_wales_fine_tiles.py` replaces the tile-selection mechanism entirely for
Wales, since Wales (one 45GB COG) has no natural tile index the way England's
WCS-downloaded tiles did: builds a uniform 5km candidate grid directly over a named
`wales_nrw_floodzone_seas` benchmark region's buffered bbox (`severn_estuary`/
`menai_strait`/`wales`), windowed-reads each candidate's exact min elevation from the COG
via `gdal`, drops cells above the region's own RP250 design-sea-level ceiling (confirmed to
reuse England's own `_MAX_WATERLEVEL_M=10.0` - the same `ea_coastal_design_sea_levels`
"UK MAINLAND" stations set both), then feeds survivors through the unmodified
`connectivity_tiling.py` primitives (`Domain`/`ConnectivityConfig`/`compute_hop_distances`/
`assign_tile_id`). 5km (not 7-10km) because the elevation filter keys off each cell's MIN,
not mean - a coarser cell spanning both floodable and never-flooding terrain still has a
low MIN and is kept whole, and nothing downstream re-applies the filter per sub-piece.

Severn Estuary build: 21 real tiles, hop distribution 0:10/1:7/2:3/3:1 - genuine multi-hop
structure (unlike England's Scilly test, which came back all hop=0), a real exercise of the
hop-propagation mechanism beyond what the Wash test alone proved.

Parameterized for multi-region reuse (defaults preserve England's exact prior behaviour,
each verified via a bare no-arg call): `cache_local_waterlevel_stations.py` gained
`--local-dem-key` (default `england_lidar_dem`); `rescore_england_fz3_local.py` renamed to
`rescore_floodzone3_local.py` and, with `rescore_1km_fp_buffer_local.py` and
`plot_local_vs_global_agreement.py`, gained `--benchmark-key`/`--country-iso`/`--region`
(the last fixing a latent bug: `next(iter(spec.regions.items()))` silently picks whichever
region sorts first in a multi-region benchmark's `regions` dict - harmless for England's
single-region benchmark, wrong for Wales's 3-region one, which would have defaulted to the
full-country `wales` bbox instead of `severn_estuary`). `validate_country.py` gained
`--rp`/`--slr` (overriding `validation.return_period`/`waterlevel_name` - it otherwise only
scores the single RP/SLR pair named in config, no multi-RP loop of its own).

New `preparation/run_simulate_waves.py`: thin CLI wrapper promoting the previously
scratchpad-only wave-by-wave simulate runner
(`run_delta_floodmaps.run_simulate_stage`, already fully generic) to a reusable script for
any materialized config, not just the named `deltas_ssp126/ssp245` scenarios.

**A real near-miss, caught by the user, not by this session's own checks:** the first
`wales_global_input` preprocess invocation ran unrestricted (945 `extract_boundaries` jobs -
21 tiles x 9 RPs x 5 SLR, the production default, matching `england_global_input`'s own
precedent of a full preprocess) - cheap, finished fine. But `run_simulate_stage` reads RPs/
SLR straight from `boundary_conditions`/`adaptation` with no restriction of its own, and
England's actual eikonal solve had only ever been limited to RP100+RP250/SLR_0 by passing
explicit snakemake targets by hand, never inside the (then-new) `run_simulate_waves.py`
wrapper. Running that wrapper as originally written against `wales_global_input` would have
tried to eikonal-solve all 9 RPs x 5 SLR x 21 tiles. Fixed: `run_simulate_stage` and
`run_simulate_waves.py` both gained explicit `return_periods`/`waterlevel_names` override
params/flags (default `None` = config's own full list, preserving
`run_delta_floodmaps.py`'s own ssp126/ssp245 callers unchanged). A second instance of the
same trap: the blanket `snakemake postprocess` target also depends on the full RP x SLR
matrix (`_POSTPROCESS_OUTPUTS`), so running it directly tried to re-trigger `run_eikonal`
for every RP/SLR combo this sample never simulated, failing loudly (hop>=1 neighbour-file
check, not a silent corruption) once it hit a tile needing a never-simulated neighbour - no
wasted compute (checked: `waterdepth_*.tif` count stayed at 42 = 21 tiles x 2 RPs,
unaffected). Fixed by targeting `merge_chunk`'s own output files directly
(`{merged_outputs}/chunks/waterdepth_{chunk_id}_RP{100,250}_SLR_0.tif`, `chunk_id` from
`build_chunk_grid` - one chunk, `N50W005`, for this sample) instead of the blanket rule name.

Ran for real, both scenarios, RP100+RP250/SLR_0 only: preprocess (945+1051 jobs, 0 errors),
simulate (wave-by-wave through hop=3, 42 `waterdepth_*.tif` each, 0 errors) - hop-distance
propagation confirmed working for `wales_lidar_dem` the same way as England's Wash test:
tile 20 (hop=3, the furthest tile) shows real, physically sensible flooding (0.52% wet,
1.95-4.08m depths) correctly seeded from its lower-hop neighbours, not degenerate/all-dry.
Merge: `merge_england_local_tiles.py` (unmodified) for local-native, `merge_chunk`
(targeted, see above) for global - both produced physically sensible depths (global
`N50W005` chunk: 27% wet, 0.02-11.8m).

Scoring (severn_estuary region, 100% benchmark coverage confirmed via
`benchmark_wet_outside_model_domain_pct=0.0` - the England/whole-Wales rows in the same
`validate_country.py` run are correctly flagged mostly-outside-domain and not meaningful
here, since this sample's tiles don't cover them):

| | RP100 HR/FAR/CSI/bias | RP250 HR/FAR/CSI/bias |
|---|---|---|
| Global (`validate_country.py`) | 0.820 / 0.012 / 0.811 / 0.830 | 0.834 / 0.014 / 0.825 / 0.846 |
| Local (`rescore_floodzone3_local.py`) | 0.775 / 0.001 / 0.774 / 0.776 | 0.801 / 0.001 / 0.800 / 0.802 |

1km FP-buffer rescoring changed nothing for either scenario (baseline FAR was already
near-zero at this scale, confirmed via `rescore_1km_fp_buffer.py`/`rescore_1km_fp_buffer_local.py`).
Local has a lower hit rate than global here but near-zero false positives, vs. global's
higher hit rate with a small but real false-positive rate - a genuine local-vs-global skill
contrast, not an anomaly; not yet investigated further (see still-open below). Agreement
maps written: `wales_local_input/agreement_local_vs_global_RP{100,250}_SLR_0.png`.

**Bug found from the agreement map itself, fixed same session:** the user spotted a
perfectly axis-aligned gray cross in the local panel only (lon ~-2.97, lat ~51.53) -
dry-looking but also not scored as agreement. Root cause: `window_scoring_domain`'s
per-block catalog reads (permanent-water + WRI geogunit masks) in the `_BLOCK_SIZE=8192px`
chunked native-CRS classification loop (`_category_and_metrics_local` in
`plot_local_vs_global_agreement.py`, and the identical pattern in
`rescore_floodzone3_local.py`/`rescore_1km_fp_buffer_local.py`) read each block's own tight
bbox with no margin. Nearest-neighbour GDAL reprojection of the coarse sources (geogunit
~30 arcsec, DeltaDTM mask ~30m) onto the fine 1m local grid then has no source pixel to
sample right at each block's edge, falls back to fill, and that fill never matches
"in Wales"/"not water" - so every `_BLOCK_SIZE` boundary (every 8.192km) got a false
excluded strip, worst one 100+ rows deep. Only visible locally because the global path's
merged-chunk grid (~90m) never spans more than one block, so the same code path never
seams there - confirmed: global numbers above are unaffected, re-ran unchanged.
Fixed with one shared-primitive change: `src/validation.py::_read_categorical_on_grid`
now pads its catalog-read bbox by `_READ_PAD_DEG=0.02` deg (>=2 geogunit pixels) before
reprojecting, while the destination grid/transform stay exactly as given. Verified
before/after via a row/col nodata-spike scan of the category raster (clean spikes at
rows 8192/16384 and col 8192 pre-fix, gone after) and by eye on the regenerated
`agreement_local_vs_global_RP{100,250}_SLR_0.png` (cross gone, metrics shifted ~+0.004
on HR/CSI/bias for both RPs, FAR unchanged - table above already corrected). England's
earlier Scilly/Wash local-input numbers used the same un-padded code path and likely
carry the same small bias; not re-run this session (see still-open below).

**Section 6 - still open:**
- Per the explicit scope boundary for this pass: the full-Wales fine-tile grid was not
  built, no full-country Wales simulation was run, and no local-vs-HPC call was made - all
  deferred to a follow-up pass, mirroring England's own still-open full-1836-tile status.
- The local-vs-global HR/FAR contrast above (local misses more of the benchmark but is
  more precise where it does flag) is only a one-region, one-sample observation - not
  compared against England's own Scilly/Wash numbers yet. `friction_scale_factor x
  pixel_size_m` ruled out as a cause (user, 2026-11-06): attenuation per grid cell scales
  linearly with pixel size, so the cost per metre of travel is identical at 1m and 30m -
  not a resolution-dependent bias. Leading (still untested) hypothesis: the only inputs
  that actually differ between `wales_local_input`/`wales_global_input` are the DEM (1m
  LIDAR vs DeltaDTM, ~25-31m at this latitude) and the boundary (`ea_coastal_design_sea_levels`
  vs COAST-RP); local's FAR dropping ~10x (0.012->0.001) alongside its HR dropping
  (0.820->0.775) - i.e. local isn't leaking flood into new area, it's failing to reach some
  benchmark-wet cells it would reach at coarser resolution - is consistent with 1m LIDAR
  relief resolving small real barriers (verges, minor raised ground, hedge lines) that block
  propagation to cells NRW's generalized FZ3 extent still counts as flooded; NRW's
  known_caveats note "defences and wave run-up treatment not reviewed" and give no DEM
  resolution of their own, so the assumption that the benchmark was itself made at LIDAR
  resolution is unconfirmed. Not tested: whether the missed cells cluster near known
  small-scale terrain features, or near the boundary-condition source difference instead.
- The two agreement PNGs look visually similar despite the 1m vs ~28m resolution gap for a
  separate, confirmed (not hypothesized) reason: `plot_local_vs_global_agreement.py` renders
  both panels into one `figsize=[10,10]*2, dpi=300` figure (~3000x3000 px/panel). Local's
  native category array for severn_estuary is 21439x14864 px (diag2_after_fix.log) - about
  7x downsampled onto the panel - while global's native array (~30m over the ~14.5x23km
  region) is only a few hundred px across and gets upsampled onto the same panel. Both
  therefore render at a similar effective ~5-7 m/displayed-pixel, independent of the real
  ~28x native-resolution difference between the two rasters; the resolution difference is
  real in the underlying data and the metrics, just not visible in the published image at
  these figsize/dpi settings.
- Scotland and SF Bay/New Brunswick replication (user's stated order) not started.
- England's earlier Scilly/Wash local-input HR/FAR/CSI numbers were computed before the
  block-boundary domain-mask padding fix above and likely carry the same small (~+0.004-ish)
  upward bias on HR/CSI/bias; not re-verified this session, re-run if those numbers matter
  for a future comparison.

**Section 7 (Scotland) - PAUSED before any simulation, boundary-forcing approach superseded
(2026-11-06):** started replicating the Wales pattern for Scotland (mixed pairing: global
DeltaDTM DEM unchanged, only `ea_coastal_design_sea_levels` swapped in for the boundary,
bridged ODN->GOCO06s via `odn_osgm15` - TODO.md's own prior note that this bridge is
mandatory here, unlike England/Wales's native-ODN no-op). Built and ran, read-only/cheap
steps only:
- `preparation/select_country_tiles.py` gained a `--benchmark-key`/`--region` single-region
  mode (bypassing the whole-`GROUPS` closure) - general-purpose, kept regardless of
  Scotland's status. Regression-checked against a scratch rerun of the no-arg GROUPS path;
  found the *existing* `gbr_wales_scotland_tiles.gpkg`/`new_brunswick_tiles.gpkg` under
  `processed_inputs/mask/calibration/` are themselves stale (44 tiles vs 56 on a fresh
  rebuild) - built 2026-10-07, before `wales_nrw_floodzone_seas`'s regions were reshaped
  2026-10-18/19. Not rebuilt (confirmed via grep this file has no live consumer besides this
  script's own `main()` - `build_england_validation_tiles.py`, its only other caller, is
  already deleted) - harmless staleness, flagged here only for completeness.
- Built `scotland_sepa_coastal_m_firth_of_forth_tiles.gpkg` (new single-region mode): only 4
  production tiles, all hop_distance 0 - unlike Wales's 21-tile/hop 0-3 sample, this sample
  as scoped would exercise NO hop-distance propagation at all (production tiles are much
  larger than Wales's fine tiles).
- `preparation/cache_local_waterlevel_stations.py` gained `--boundary-native-datum` +
  `--hybrid-grid-raster-path`, bypassing `--local-dem-key` entirely (Scotland has no local
  DEM to key off) - general-purpose, kept. Ran it for real with `odn_osgm15` +
  `uk_os_OSGM15_GB.tif` - worked, offsets GB-wide [-1.86, 0.17] m (mean -1.05 m, consistent
  with the magnitude already seen in England's own odn_osgm15 investigation, Section 5).

**Before running preprocess/simulate, checked how close the real boundary stations are to
the Firth of Forth test region - found a real gap, not a bug:** nearest
`ea_coastal_design_sea_levels` station is ~0.49 deg (~55 km) from the region's centre.
Scotland overall has good density (1925/3431 GB stations fall in a broad Scotland bbox), but
the ~91 stations nearest the Firth of Forth all sit along the open coast outside the firth's
mouth (roughly -2.2 to -3.25 deg lon, Angus/Berwickshire coast), not inside the estuary
itself (-3.85 to -3.64 deg lon, where the benchmark region actually is) - a real coverage gap
inside this specific tidal estuary, not a filtering mistake.

**User decision (2026-11-06): do not use local boundary data for Scotland right now.** The
boundary-forcing derivation itself is being reworked elsewhere (moving to FES-based tides,
closer to how a separate reference study does it) - use global boundary data (COAST-RP) for
Scotland instead of `ea_coastal_design_sea_levels`/`odn_osgm15` until that rework lands.
Consequence: with the DEM already global (decided earlier) and now the boundary global too,
Scotland's "local-input" scenario has nothing left to differentiate it from the global-input
baseline - **paused, not abandoned**: no local-input case exists to build or run for Scotland
until the FES-based boundary work defines what "local" means here. No preprocess/simulate
was run. `snakemake_workflow/config/scotland_local_input.yml` (+ its `_materialized.yml`) and
the cached `odn_osgm15`-bridged boundary station `.gpkg`s under
`{root}/scotland_local_input/processed_inputs/WL_scenarios_cache/` are left in place
un-deleted (small, isolated, harmless) but should NOT be used until this is revisited -
they're premised on the now-superseded boundary approach.

**Section 7 - still open:** Scotland local-input validation resumes once the FES-based
boundary-forcing rework defines a real local source for Scotland (or a decision to drop
Scotland's local-input case entirely, if no suitable local source ever materializes). New
Brunswick skipped for the identical reason (user, 2026-11-06).

**Section 8 (SF Bay, Alameda County sample) - IN PROGRESS (2026-11-06):** SF Bay is
structurally like England/Wales, not Scotland - it swaps the DEM (NOAA 2025 SF Bay-Delta
LIDAR, NAVD88/GEOID18, EPSG:26910, 1m) while the boundary stays global COAST-RP
(`California/new_inputs/boundary_forcing/datasource.txt` = "CoastRP", already decided) - not
blocked by the boundary rework. User picked Alameda County for the first sample (a
per-county CoSMoS depth raster was already staged at
`California/alameda_flooding_slr000.zip`).

New `navd88_geoid18` vertical-datum bridge implemented in `src/vertical_datum.py`
(`DATUM_BRIDGES`, plus a new shared `_hybrid_minus_goco_grid`/`_hybrid_minus_goco_points`
helper the `odn_osgm15` branch now also uses) - via `pyproj`'s registered
`EPSG:6318+5703 -> EPSG:7912` transform (NAD83(2011)+NAVD88 height -> ITRF2014: GEOID18 +
the mandatory Helmert shift in one call), NOT a downloaded NOAA grid file as originally
assumed. Real correctness hazard found and designed around: pyproj's network defaults off,
under which PROJ silently substitutes a "ballpark" operation that omits GEOID18 entirely
(confirmed: -32.90 m with network on at a SF point vs -0.52 m with it off, only a
`UserWarning`) - the new one-off `preparation/compute_navd88_itrf2014_shift_raster.py`
hard-enables the network and hard-raises unless the real grid-based (accuracy=0.015m)
operation was selected. Ran for real over the Bay AOI: shift range [-34.98, -31.57] m (matches
the expected ~-32 to -34 m GEOID18 magnitude); combined with the existing GOCO06s-absolute
raster at a SF test point gives ~-1.7 m to add to H_NAVD88 - plausible (Helmert ~0.5-1m +
~1m geoid-model discrepancy, both already expected per the catalog's own notes). Cached to
`California/new_inputs/lidar_dem/navd88_itrf2014_shift.tif`; `sf_bay_lidar_dem`'s catalog
entry now sets `native_datum: navd88_geoid18` + `hybrid_grid_raster_path` accordingly.
`tests/vertical_datum_validation/validate_vertical_datum.py` updated (navd88_geoid18 no
longer asserted to raise; gained its own `test_navd88_geoid18_combines_hybrid_and_goco_grids_correctly`,
same formula shape as the odn_osgm15 test) - `pytest -m "not needs_data"`: 76 passed.

New `preparation/build_sf_bay_fine_tiles.py` (mirrors `build_wales_fine_tiles.py`; no
catalog `regions` dict exists for SF Bay, so Alameda's bbox comes from the staged CoSMoS
raster's own bounds, read directly via rasterio's `zip://` VFS - no extraction needed).
`_MAX_WATERLEVEL_M` derived from COAST-RP directly (not a local design-levels product, since
SF Bay's boundary stays global): max `storm_tide_rp_0250` among stations in the Bay Area
bbox is 1.725 m (queried this session) vs England/Wales's ~9.1-9.5 m - used a looser 5.0 m
ceiling rather than reusing 10.0 m by rote. Ran for real: 187 candidate 5km cells, 138
dropped (DEM's own AOI is a narrow flood-extent-clipped strip, not the whole county), 0
dropped on elevation, **49 tiles kept, hop 0: 36 / hop>=1: 13, 0 unreachable**.

`validation/validate_sf_bay.py` parameterized (`bbox`/`benchmark_path` optional, default
unchanged = full 9-county window; new `--native-merged-path` to read a local scenario's
native-CRS merged file directly, same "never reproject the big depth raster" pattern as
Wales's plotting script) - regression-safe, no behaviour change with no new flags.

New scenario config `sf_bay_local_input.yml` only (added to `materialize_configs.py`) -
**no `sf_bay_global_input.yml` and no separate global-input preprocess/run**: caught this
twice in the same session (first for Scotland, then started repeating the mistake here
before the user corrected it again) - production's own already-computed merged chunks
(global DeltaDTM + COAST-RP) are the global baseline as-is, read live by pointing
`validate_sf_bay.py` at `config.yml` with the Alameda bbox/benchmark overrides, never
rerun. A brief `sf_bay_global_input` preprocess was started, caught, and killed before any
real compute happened (`TaskStop`) - its few partial per-tile scratch files under
`{root}/sf_bay_global_input/` were left in place (small, harmless) after a recursive delete
of that tree was blocked by the sandbox; safe to remove by hand if it matters.

Preprocess (49 tiles, 2451 jobs), wave-by-wave simulate (RP100/SLR_0 only - no usable CoSMoS
RP250-equivalent, so this is the only RP in scope), and the native-CRS merge via
`merge_england_local_tiles.py` all completed cleanly (exit 0, no errors). One portability fix
made along the way: `src/step_runner.py`'s banner print uses a Unicode box-drawing
character, which crashes with `UnicodeEncodeError` under Windows' default `cp1252` console
encoding whenever stdout is piped (e.g. `| tail`) - worked around per-invocation with
`PYTHONUTF8=1 PYTHONIOENCODING=utf-8` rather than touching that shared file.

**`navd88_geoid18` bridge spot-checked against the real extracted tile** (not just the ad
hoc cross-check above): comparing one Alameda tile's `dem.tif` output against the raw local
LIDAR source gives a tightly-clustered applied offset of -1.57 m (range -1.64 to -1.51 m,
std 0.027 m) - consistent in sign and magnitude with the independent manual check, and
nowhere near the ~32 m the silent ballpark-fallback bug would have produced. Bridge is
correct.

**Four real bugs found and fixed while scoring (all in `validate_sf_bay.py`, none in the
model or the bridge) - in the order hit:**

1. **Production file overwrite.** The first scoring call (`--config config.yml` +
   custom `--bbox`/`--benchmark-path`, no `--out-dir`) silently wrote its Alameda-only
   results over `{validation.output_dir}/USA/agreement_USA_sf_bay_RP100_SLR_0.tif` and
   `depth_diff_..._.tif` - real full-9-county production outputs from 2026-10-18, since
   `OUTPUT_TAG` and the implicit default `out_dir` don't vary with `bbox`/`benchmark_path`.
   Caught via file mtimes (the `.png` renders stayed at their Oct-18 mtime while the `.tif`
   sources jumped to today). **Fixed forward**: re-ran `validate_sf_bay.py --config
   config.yml` with no overrides to regenerate the correct full-9-county files (the
   underlying merged-chunk data was never touched, so this is an exact, lossless redo) -
   confirmed restored (HT/FAR/CSI/bias printed matched the expected full-domain shape).
   **Fixed at the root**: `--out-dir` no longer has an implicit default at all - every
   invocation (production or sample) must now pass it explicitly, so a custom-scoped run can
   never again silently collide with production's path.
2. **OOM in the new `--native-merged-path` branch.** A 1m LIDAR merge over Alameda's
   49-tile footprint is a ~57000x47000 array - one `float64` copy of it alone is ~21 GiB,
   on a 32 GB machine, and the original implementation held several such arrays at once.
   **Fixed**: rewrote the native-merge scoring path (both grid directions) to iterate
   2048x2048-pixel windows and accumulate confusion-matrix counts + squared/absolute-error
   sums (`_Accumulator` - summing disjoint windows this way is exact, not approximate) -
   confirmed via live RSS (~6-7 GB, not 20+) across both runs. The plain chunks-based
   default path (DeltaDTM's own ~30m grid) was never touched; it was already small.
3. **Window-clamp bug** in the new windowed code: `_model_native_window`'s read window was
   sized from the *requested* bbox (Alameda's full CoSMoS benchmark extent, ~80km wide)
   without clamping to the merge file's *actual* extent (~47km wide - only the 49 simulated
   tiles' footprint). Sub-window reads near that edge came back narrower than the loop
   expected, raising a shape-mismatch `ValueError`. Fixed by intersecting with
   `Window(0, 0, src.width, src.height)`, the same pattern `validate_country.py` already
   uses for exactly this reason.
4. **Benchmark units bug.** USGS CoSMoS's per-county extract files (the Alameda zip) store
   depth as **int32 centimetres** (nodata -9999; sampled values ~133, 119, 134…); the full
   9-county production mosaic stores depth as **float32 metres** (same locations read
   ~1.33, 1.19, 1.34…) - confirmed by reading both at the same coordinates. Using the
   Alameda file as `--benchmark-path` without converting inflated every RMSE/MAE by ~100x
   (first symptom: `RMSE_flooded_m` ~87-98 m for *both* local and global at this bbox, vs
   ~0.35-0.8 m for the full 9-county baseline). Fixed with an explicit `--benchmark-scale-m`
   CLI flag/parameter (default 1.0, correct for the mosaic; 0.01 for a CoSMoS per-county
   file) - deliberately not auto-detected from dtype, so a future region's benchmark file
   can't silently misfire the same way.

**Final corrected metrics (Alameda bbox, `--benchmark-scale-m 0.01`, `primary_threshold_m` =
0.01 m), vs. the full-9-county production baseline for context:**

| | HT | FAR | CSI | bias | RMSE_flooded (m) | MAE_flooded (m) |
|---|---|---|---|---|---|---|
| Local (A, model-native 1m grid) | 1.000 | 0.934 | 0.066 | 15.07 | 11.80 | 11.79 |
| Local (B, CoSMoS 2m grid) | 1.000 | 0.932 | 0.068 | 14.64 | 11.78 | 11.77 |
| Global-at-Alameda-bbox (A) | 0.842 | 0.738 | 0.250 | 3.21 | 0.63 | 0.51 |
| Global-at-Alameda-bbox (B) | 0.921 | 0.774 | 0.222 | 4.07 | 0.59 | 0.47 |
| Global, full 9-county (A) | 0.685 | 0.579 | 0.353 | 1.62 | 0.77 | 0.64 |
| Global, full 9-county (B) | 0.721 | 0.628 | 0.325 | 1.94 | 0.78 | 0.65 |

**Real, understood methodological finding - not a further bug, and NOT yet a clean
local-vs-global comparison:** both local and global score far worse at the Alameda bbox than
the full 9-county baseline (FAR/bias both elevated), and the *local* run's RMSE/MAE stay an
order of magnitude worse than global's even after the units fix. Diagnosed by sampling
individual flooded cells directly (not guessed, and corrected once - see below): in the worst
windows, model depth is a near-uniform ~13 m while CoSMoS shows ~1-3 m at the same cells.

First hypothesis (**wrong, caught by the user**): that ~13 m was mis-read as real bed
bathymetry in a narrow tidal channel the global water mask fails to resolve. The user
objected on physical grounds - airborne topographic LIDAR cannot measure through water, so a
LIDAR-derived DEM showing "bathymetry" at a channel is implausible - and the objection was
right. Checked directly instead of re-asserting: the *raw, unbridged* local LIDAR source
value at the worst cell is -9.947 m, repeating almost identically across a wide block of
neighbouring pixels (-9.94685, -9.94685, -9.946846, ...) - the signature of **hydro-flattening
applied to a water-body polygon** (one representative elevation assigned across a detected
water surface), not a measured bed profile. The `deltadtm_mask` value at that exact pixel is
`0` (land), while a text-rendered map of the surrounding mask shows a small `2` (lake) patch
immediately next to a much larger area sitting at the same flat elevation and classified
`0`/`1` (land/ocean) by the global mask. So this is a **diked, below-sea-level pond or
basin** - visible to LIDAR either as dry ground behind a levee or as a shallow managed pond
whose flat surface was hydro-flattened - where the global-resolution mask only catches a
small core of it as "lake" and misclassifies the rest as ordinary land at the 1 m local
scale. The eikonal model then (correctly, given that classification) floods the misclassified
area to the full ~13-14 m implied by its elevation versus the RP100 boundary level, while
CoSMoS's benchmark does not treat it the same way there. The underlying issue - a global
water-classification product too coarse for a local-resolution comparison - is the same as
first suspected; the specific feature is a diked pond/basin, not channel bathymetry.

**Confirmed further, not just inferred:** the Alameda zip also ships a `waterelev` raster
(CoSMoS's internal water-surface elevation, alongside the `flddepth` raster scored above)
and the `sfb_fldhazd`/`sfb_lowfld` hazard-extent polygons. At the exact worst pixel,
`waterelev` is **valid (2.89 m)**, not nodata, and the point sits inside `sfb_fldhazd`
(DN=1) - so CoSMoS's own water level does reach this location; `flddepth` is nodata there
only because `depth = waterelev - ground` came out <=0, i.e. CoSMoS's own ground elevation
at this pixel is at/above 2.89 m. That is a ~12 m disagreement with our local LIDAR's -9.9 m
hydro-flattened reading at the same spot - not a benchmark-coverage gap. CoSMoS's terrain
model does not see this diked basin as a depression (consistent with a DEM that treats
leveed interior ponds/basins as filled/disconnected rather than open water), while our
global water mask fails to flag it as closed water, so the eikonal solver fills it to its
true floor elevation once the (wrongly "land"-classified) cells let the flood in. This is
real supporting evidence for the already-deferred "bathtub"/hydraulic-connectivity check
below, not a new, separate issue.

This hits the **local** run much harder than global because the local sample's domain is, by
construction, restricted to the 49 "likely-to-flood" near-shore tiles
(`build_sf_bay_fine_tiles.py`'s elevation filter) - i.e. the domain is concentrated on exactly
the shoreline zone where these diked/depressed features sit - while the global comparison's
domain spans the full bbox rectangle (ordinary dry land included), diluting the same
underlying issue to a much smaller aggregate effect.

**Net assessment:** the HT/FAR/CSI/bias and RMSE/MAE numbers above should **not** be read as
"the 1m local DEM is dramatically worse than the 30m global DEM at SF Bay" - they mostly
reflect a handful of misclassified diked/depressed water features near the shoreline, not a
DEM-quality difference. A genuine local-vs-global comparison here needs either a finer local
permanent-water mask (the NOAA LIDAR DEM is already hydro-flattened, so its own flat-water
signature could plausibly drive one - exactly the signature that exposed this issue) or a
domain/threshold choice that excludes implausibly-deep "flooded" cells before scoring. **Not
building either now** - out of scope for this sample; logged here as the next concrete step
if SF Bay is revisited. Per the usual discipline: **stopping here** - no full-9-county local
build, no local-vs-HPC call, and the cross-cutting bathtub-check/threshold-sweep work below
stays deferred until this caveat is resolved (scoring a flawed domain at three more
thresholds would just triple the noise, not add information).

**Follow-up (2026-11-08): local-input tiles now derive their own mask from the LIDAR itself,
not the global mask - implemented, confirmed correct at the original pixel, but made the
Alameda sample's aggregate score materially worse, not better.** User pushed back (correctly)
on two framings in a row before landing on the real design - see
`src/rasters.derive_local_water_mask`/`extract_local_dem_mask` (`extract_dem.py`/
`extract_dem_mask.py` branch on `dem_source`, gated by the already-existing per-tile
`_dem_source_for_tile` mechanism; global dem_source tiles are untouched):

1. Within a local dem_source tile, every valid (non-nodata) LIDAR cell is now land at its own
   true elevation, never zeroed by `effective_dem` regardless of what the global mask says -
   fixes the originally-diagnosed mechanism exactly: the small lake-coded corner that used to
   bridge the connected-component check into the misclassified "land" cells is gone (confirmed:
   the new `mask.tif` is uniformly land around the pond, elevation -11.72 m preserved).
2. An edge-touching nodata component (water the LIDAR did not survey) is classified ocean/
   lake/river by majority vote against the global mask - a coarse classification question,
   never a per-pixel boundary one - so a lake/river tile edge is never seeded with ocean
   forcing. Verified empirically before building: all 36 hop=0 Alameda tiles' edge-touching
   nodata samples 100%/92% ocean-coded; interior nodata negligible everywhere checked.
3. Unit tests added (`tests/rasters_encoding_validation/validate_rasters_encoding.py`,
   4 new cases); `ruff check` clean; full `pytest -m "not needs_data"` suite passes (80/80).

**But the specific worst-case pixel's own depth did not change** (~13.67 m before and after) -
traced directly: the straight-line DEM profile from that pixel to the nearest open-water cell
(110 m away) is flat at -11.72 m the entire way, no ridge/levee anywhere along it. The mask fix
removed the *mask-driven* bridge, but at this pixel that was never the limiting mechanism - the
real 1 m terrain itself shows no barrier to block, so the solver's physics (friction + the
production `obstacle_coupling` elevation barrier, confirmed `enabled: true` for this scenario)
floods it regardless, exactly as input. Global production shows 1.49 m at the same pixel, but
its coarse DEM almost certainly doesn't resolve the true -11.72 m floor at all, so that number
is not necessarily the correct one either.

**Re-scored the full 49-tile Alameda sample anyway (user's call) - aggregate result is worse:**

| | HT | FAR | CSI | bias | RMSE_flooded (m) | MAE_flooded (m) | model flooded px (tp+fp) |
|---|---|---|---|---|---|---|---|
| Local (A), before this fix | 0.9998 | 0.9337 | 0.0663 | 15.07 | 11.80 | 11.79 | 357,630,408 |
| Local (A), after this fix | 0.999999 | 0.9626 | 0.0374 | 26.75 | 11.73 | 11.73 | 634,736,009 |
| Local (B), before this fix | 1.000 | 0.9317 | 0.0683 | 14.64 | 11.78 | 11.77 | (not recomputed) |
| Local (B), after this fix | 1.000 | 0.9614 | 0.0386 | 25.92 | 11.73 | 11.73 | (not recomputed) |

The model's own flooded-cell count nearly **doubled** (+77%). Of the 881.8M wet cells in the
full 49-tile native merge, 542.2M (61.5%) exceed 10 m depth. RMSE/MAE *among already-flooded*
cells barely moved (~11.7 m both before and after) - this is not a small regression at the
margins, it is the same implausible-depth mechanism now firing across a much larger area.

**Leading hypothesis, not yet confirmed:** the South Bay margin historically held an extensive
network of diked salt ponds (real, physically below sea level, actively managed/pumped dry in
reality - some have since been deliberately breached for tidal-marsh restoration). The global
mask happened to code much of this complex as water, so `effective_dem` zeroed a large share of
it regardless of any one cell's true elevation - incidentally suppressing most of these basins'
depth even though the *reason* (treating them as tidally zeroed) was never correct either. This
fix makes every such cell trust its own (frequently very negative, apparently almost always
barrier-free within ~100 m at 1 m resolution) elevation instead, so the same no-resolvable-
barrier mechanism found at the one originally-diagnosed pixel appears to fire across most of
the historic salt-pond footprint, not just the one small mask-conflation case it was built to
fix. Whether these basins are genuinely barrier-free now (restored to tidal action) or the real
levees simply are not resolved/preserved by this LIDAR dataset's processing is unresolved.

**Not reverted, not taken further yet - open decision for the user:** keep the fix (it is
correct for the mechanism it targets and the small-conflation case), but the Alameda sample's
aggregate numbers above should not be read as "the fix works" until this bigger question is
resolved. Candidate next steps, none started: (a) check a real levee/imagery source for a
handful of the newly-deep-flooding basins to see whether a real crest is being lost; (b) check
whether South Bay Salt Pond Restoration Project breach records explain some of this directly;
(c) build the deferred bathtub/connectivity check above specifically to separate "no resolvable
barrier within the LIDAR" from "genuinely eikonal-connected" as a distinct diagnostic layer;
(d) accept this as a real, deeper DEM-resolution limitation and exclude diked-basin-like cells
from scoring explicitly rather than from the model. **Stopping here per usual discipline** - no
further code changes without the user's direction on which of these (if any) to pursue.

**Cross-cutting, deferred until SF Bay's local-input sample actually works (user,
2026-11-06, confirmed as specified):** add an upper-bound "bathtub" sanity check across
every local-input calibration area (Wales severn_estuary, England Scilly/Wash, SF Bay
Alameda, and Scotland Firth of Forth if/when it resumes) - NOT a naive global elevation
threshold, but an ocean-seeded, hydraulically-**connected** flood fill: a cell floods only
if its elevation is below the boundary water level AND it has a path of such cells back to
the sea (same connectivity principle `src/connectivity_tiling.py`/`compute_hop_distances`
already uses elsewhere in this pipeline, just applied as the flood criterion itself, not a
tiling decision). This is a genuine upper bound on inundation extent - it has no friction,
so it can only flood *more* than the eikonal solver, never less - making it a useful second
reference point alongside each benchmark: compare it against (a) each region's own local
benchmark and (b) the eikonal solver's own result, using the existing `validation.py`
confusion-matrix machinery (HR/FAR/CSI/bias), to see how much of the local-vs-global skill
gap is really coming from the eikonal solver's physics as opposed to just "is this cell
connected to the coast at all."

Also apply to **both** the local and the global analysis (user, 2026-11-06): score at
multiple critical depth thresholds - **0.05 m, 0.1 m, and 0.2 m** - not just the single
`validation.primary_threshold_m` (currently 0.01 m) production uses today. I.e. for every
region, report HR/FAR/CSI/bias at each of the three thresholds, for all three comparisons
(local-eikonal vs benchmark, global-eikonal vs benchmark, bathtub vs benchmark) plus
bathtub-vs-eikonal (both local and global) - a 3-threshold x multiple-comparison grid per
region, not a single number. Note: config.yml's own comment on `primary_threshold_m`
already records an earlier Wales/Scotland sweep at 0.10/0.05/0.01 m (CSI/HR improved
monotonically down to 0.01 m there) - this new sweep is for the bathtub/local/global
three-way comparison specifically, not a re-litigation of production's own default
threshold.

Not scoped or started - revisit once SF Bay's sample is scored.

**Section 8 follow-up (2026-11-08): DEM source corrected (NOAA 1m -> USGS 3DEP 10m), but the
nodata-edge mask mechanism does not transfer - stopped before any simulate/score run.** User
learned `sf_bay_lidar_dem` was pointed at the wrong source entirely - swapped in **USGS 3D
Elevation Program 1/3 arc-second (10 m) DEM**, distributed by OpenTopography (DOI
10.5069/G98K778D, US Government Public Domain), full replace of the NOAA 2025 1 m
hydro-flattened LIDAR. Done:
- `preparation/sf_bay_usgs10m_dem.py` (new, replaces `noaa_sfbay_lidar.py` in
  `run_preparation.py`'s opt-in step list) reprojects the hand-downloaded
  `output_USGS10m.tif` (NAD83 geographic, nodata -999999.0) to EPSG:26910 @ 10 m ->
  `sfbay_dem_usgs10m.tif`. Ran for real.
- `data_catalog_local_input.yml`'s `sf_bay_lidar_dem` entry updated in place (same key):
  new path, `pixel_size_m: 10`, `nodata: -999999`, citation/license as given by the user
  (US Government Public Domain, not `[ASSERTED]`), `native_datum: navd88_geoid18` kept
  unchanged - this product's 2021 vintage and the independently-downloaded
  `us_noaa_g2018u0.tif` (NGS GEOID18 CONUS grid, also in the same directory) both corroborate
  GEOID18, so this was not reopened as a question.
- `preparation/compute_navd88_itrf2014_shift_raster.py` rerun over the new DEM's actual
  (wider) extent (`-122.75 37.19 -121.70 38.31`) - the old cached `navd88_itrf2014_shift.tif`
  didn't reach the new DEM's southern/eastern edges. Shift range [-36.08, -31.46] m, same
  GEOID18 grid-based operation verified (accuracy 0.015 m) as before.
- `preparation/build_sf_bay_fine_tiles.py`'s hardcoded `sfbay_dem_m.tif` path replaced with a
  lookup of the catalog's own `sf_bay_lidar_dem` entry (was drifting from the catalog even
  before this session). Rebuilt Alameda: 187 candidates, 55 no-coverage, 77 above the 5 m
  max-waterlevel ceiling, **54 tiles kept (hop=0: 40, hop>=1: 13, 1 unreachable)** - vs. the
  old 49 tiles (hop=0 36/hop>=1 13, 0 unreachable). Different split is expected (seamless
  nationwide coverage vs. the old survey-limited footprint), not investigated further.

**Stopped here - a real blocker found before any preprocess/simulate run:** sampled 5 of the
40 new hop=0 tiles directly (clipped DEM window + `derive_local_water_mask`, the mechanism
built in this session's earlier mask-derivation fix). **All five have `frac_nodata = 0.0` -
zero nodata anywhere in the tile - so every pixel is classified `land`, zero `ocean`/`lake`/
`river` cells, in all five.** Checked why directly: the raw `output_USGS10m.tif` has **no
nodata at all** anywhere in its whole downloaded extent (checked via a decimated full-raster
read) - USGS 3DEP's seamless product fills open water with a valid elevation (confirmed: the
Bay Bridge/open-bay area reads exactly `0.0 m`, not nodata) rather than leaving a nodata gap
the way the old survey-limited NOAA LIDAR did.

This breaks the core assumption the nodata-edge-detection mask mechanism (this session's
earlier fix, `src/rasters.derive_local_water_mask`) depends on - "a nodata component touching
the tile edge is open water the source didn't survey" does not apply to a seamless DEM with no
nodata at all. Consequence if run as-is: every local-input SF Bay tile would classify entirely
as land, including the open Bay itself, so hop=0 tiles would have no `ocean`-coded cell for
`coastline_mask` to anchor direct boundary-station seeding on - a different, new failure mode
from anything diagnosed so far for England/Wales/the old NOAA DEM.

**Not resolved, no code changed beyond what's listed above - needs the user's direction.**
Candidate approaches, none started:
(a) For this source specifically, skip `derive_mask_from_dem` and fall back to the original
    global-mask-based `extract_dem_mask` path (same as `deltadtm` tiles use today) - simplest,
    but reintroduces the global mask's known land/water misclassification risk inside this
    DEM's footprint (the diked-pond-style problem this session's fix was built to avoid),
    just without the mitigating "trust real local elevation as land" step this source's
    elevation values would otherwise have earned it.
(b) Classify open water by elevation/flatness instead of nodata for this source (e.g. a
    near-zero, large, edge-connected flat region) - more heuristic, no precedent in this
    codebase, risk of misclassifying real low-lying flat land as water.
(c) Some hybrid: use the global mask only to seed direct (hop=0) boundary forcing (its one
    remaining, coarser job per this session's earlier plan), but still trust 3DEP's own
    elevation unconditionally as the land surface everywhere - effectively (a) restricted to
    the seeding step only, not the whole per-pixel classification.
Stopping here - no preprocess/simulate/score run attempted with the new DEM yet, so nothing
downstream (merged chunks, scored metrics) has been touched or needs reverting.

**Resolved (2026-11-08): user's direction was option (b), generalized** - classify open water
by a *flat-fill elevation* signal (not nodata) for this source, not by elevation/flatness
heuristics across the board. `derive_local_water_mask` gained `flat_fill_value: float | None`
(default `None`, so every other source's behaviour is unchanged): the water-candidate test
becomes `(dem_raw == nodata) | (dem_raw == flat_fill_value)`, same edge-component/majority-vote
classification logic as before either way. `sf_bay_lidar_dem`'s catalog meta sets
`flat_fill_value: 0.0`. Threaded through `extract_dem`/`extract_local_dem_mask` and the
`extract_dem`/`extract_dem_mask` Snakemake rules (new `flat_fill_value` param, sourced from the
catalog same as `native_datum`/`pixel_size_m`). One correctness fix needed inside `extract_dem`:
the vertical-datum offset is added to every valid cell *before* `derive_local_water_mask` was
being called, which would have shifted a flat-fill 0.0m cell away from exactly 0.0 - fixed by
capturing the DEM's raw (pre-offset) values specifically for the mask classification call.
4 new unit tests (edge-touching flat-fill -> ocean by vote; interior flat-fill -> land;
`flat_fill_value=None` leaves a real 0.0m cell untouched). `pytest -m "not needs_data"`: 83
passed (was 80).

Verified empirically on 8 sampled hop=0 Alameda tiles before running anything for real: 7/8
found >=1 edge-connected 0.0m component correctly classified as ocean (the 8th simply has no
literal 0.0m water within its own padded footprint - not every hop=0 tile needs to).

**Also found and fixed during the real run (both pre-existing, not caused by the DEM swap):**
- `sf_bay_local_input.yml` was missing the scenario-scope override every other local-input
  config already has (`wales_local_input.yml`, `england_local_input.yml`,
  `scotland_local_input.yml`, `calibration_500_tiles.yml`) - `boundary_conditions.return_periods`/
  `waterlevel_names` restricted to just what the benchmark scores (RP100/SLR_0,
  `validate_sf_bay.py`'s own `OUTPUT_TAG`), plus `adaptation.slr_intensities: []` (the
  Snakefile's `WATERLEVEL_NAMES` is the union of the two). Preprocess job count 2747 -> 327;
  `simulate` ~2430 -> 54. Saved as a standing feedback memory so this gets checked automatically
  on every future local-input config.
- A hop>=1 tile's `run_eikonal` can be scheduled by Snakemake before its hop<k neighbour's own
  job, even at `--cores 1` (no declared Snakemake edge for the implicit neighbour-seed read -
  same already-documented limitation as this file's "Known behaviour" section). Worked around
  by running `simulate` wave-by-wave in ascending `hop_distance` (same pattern as
  `run_delta_floodmaps.py`'s `run_simulate_stage`, reused for one-off local scenarios) rather
  than a flat `snakemake simulate` call.
- The standard `postprocess`/`merge_chunk` rule assumes every tile already shares the global
  EPSG:4326 chunk grid - SF Bay's tiles are native EPSG:26910, same as England's. Used
  `preparation/merge_england_local_tiles.py` (generic despite its name - takes any local-input
  `--config`) instead, matching England's own established pattern.

**Three-way comparison, full Alameda sample (`validate_sf_bay.py`, grid A = coarsen-to-model,
RP100/SLR000, 0.01 m threshold):**

| | HT | FAR | CSI | bias | RMSE_flooded (m) | MAE_flooded (m) | model flooded px (tp+fp) |
|---|---|---|---|---|---|---|---|
| NOAA 1m DEM, before mask fix | 0.9998 | 0.9337 | 0.0663 | 15.07 | 11.80 | 11.79 | 357,630,408 |
| NOAA 1m DEM, after mask fix | 0.999999 | 0.9626 | 0.0374 | 26.75 | 11.73 | 11.73 | 634,736,009 |
| **USGS 3DEP 10m DEM, flat-fill mask** | **0.6305** | **0.8633** | **0.1266** | **4.611** | **0.896** | **0.835** | **1,130,896** |

The pathological implausible-depth mechanism (11+ m RMSE among flooded cells, both NOAA-DEM
rows) is gone with the USGS10m source - RMSE/MAE among flooded cells drop to under 1 m, and
bias drops from 15-27 to 4.6 (still over-predicting, but not by an order of magnitude). The
trade-off: hit rate drops sharply (1.00 -> 0.63) - the coarser 10m DEM misses much more of the
benchmark's wet area than the 1m NOAA LIDAR did, consistent with this being a real resolution
loss, not a new artifact. FAR stays high (0.86) but improves over both NOAA-DEM rows. Not
investigated further this session (the resolution/HT trade-off itself, or whether a 0.05/0.1 m
threshold sweep - already planned below - changes this picture) - open for a follow-up pass.

## Git history rewrite (after the merge)

History holds ~1.2 GB of old files that are no longer tracked (209 MB packed): `sfincs_tiles/*.npy`
(tile306 / tile1669 arrays, up to 102 MB each), `batch_logs/`, `batch_logs_simulate_regions/`,
`run_timings.csv`, `run_postprocess.log`, `preparation/logs/`, `adaptation_validation_output/`,
`tests/*/output/`, `tests/*/plots/`. Rewriting changes every commit hash and needs a force push.

1. Merge `cleanup` into `main`, push, and make sure no clone has unpushed work (local repo,
   cluster `<code_root>`).
2. Install the tool: `pip install git-filter-repo`.
3. Work on a fresh mirror, not your working repo:
   ```
   git clone --mirror <repo_url> gfm-mirror.git
   cd gfm-mirror.git
   git filter-repo --invert-paths \
     --path-glob 'sfincs_tiles/*.npy' --path batch_logs/ --path batch_logs_simulate_regions/ \
     --path run_timings.csv --path run_postprocess.log --path preparation/logs/ \
     --path adaptation_validation_output/ --path-glob 'tests/*/output/*' --path-glob 'tests/*/plots/*'
   git count-objects -vH        # check size-pack dropped
   git log --oneline -5 main    # spot-check
   git push --force --mirror
   ```
4. Re-clone every working copy (copy the gitignored files across: `config_local.yml`,
   `*_materialized.yml`, `.cleanup_baseline/`), or `git fetch && git reset --hard origin/<branch>`.
5. Old hashes in docs and messages change; `.git/filter-repo/commit-map` in the mirror maps old to
   new (e.g. for AUDIT.md's `git show 89e130e:AUDIT.md`).
