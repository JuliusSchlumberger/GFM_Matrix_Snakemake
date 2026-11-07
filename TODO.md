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
3. **On the cluster:** `DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part eikonal` once, to check the
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
  Bangkok, deltas and calibration_500_tiles pin `attenuated` and B off in their `*_pins.yml` and
  set `preprocessing_inputs_dir: null` (own inputs).
- **Protection rule:** `protection.rule: flood_fraction | water_level` (implemented, default
  flood_fraction; `minimum_rp` 2 applies to the water-level rule only).
- **Permanent inundation:** to be counted in EAI by anchoring the exceedance curve at p = 1 with
  the permanently inundated exposure, and reported as a separate metric (not implemented yet, see
  Open).

Open:
- **Bathtub baselines: review all implementations and make them consistent (to do before any
  bathtub vs eikonal comparison goes into the paper).** Each was written for a different study, and
  each makes its own choices, so their scores are not directly comparable:
  - HC-bathtub: `sfincs_tiles/run_calibration_models.compute_bathtub_depth`. One flat level, the
    tile's highest station value, not interpolated. The depth file is unpruned; it is pruned only at
    scoring (`sfincs_tiles/compute_calibration_result.py` via
    `flood_agreement.prune_to_ocean_connected`).
  - Straight-line attenuated bathtub: `review_critiques/calibration_experiments.run_tile_distance`.
    Level = the nearest production seed (IDW of 15 stations) minus rate x Euclidean distance.
    Pruned with `flood_model.prune_to_coast_connected` on the seed cells, on the effective DEM.
  - Static inundation: `flood_model.static_inundation`, a connected flat-level bathtub at
    S = MDT + SLR inside the two-component solve. Land cells only; rivers and lakes are not
    conduits.
  - Spill level: `event_volume.spill_level`, a priority-flood bathtub (minimax elevation from the
    sea) used by correction B.
  - Naming: the eikonal solver is labelled "EA-bathtub" in `sfincs_tiles/` (calibration_results,
    plots, docs/methods_04a) and in `analysis/summarize_run_timings.py`. Literature references:
    Vafeidis et al. 2019 and the Flow-tub rate (0.2 m/km) in docs/methods_02.

  They differ in:
  - the water-level source (tile maximum, nearest seed, or IDW);
  - the connectivity rule: two pruning functions, with different coastline definitions and
    river cells as conduits or not;
  - the DEM (raw or effective, no-data handling);
  - whether pruning is applied to the depth raster or only to the score;
  - the names used for them.

  Goal: one shared implementation, e.g. `src/bathtub.py`, with:
  - a level field: flat, nearest-seed or IDW, optionally minus rate x distance;
  - one connectivity rule;
  - the same DEM, seeds and wet threshold as the eikonal solver.

  Then define one set of names (HC-bathtub, attenuated bathtub, eikonal / eikonal + B), use them
  in `sfincs_tiles/`, `review_critiques/` and `validation/`, and re-score every baseline on the
  same tiles. `sfincs_tiles/` changes need your go-ahead, and none while an HPC batch runs.
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
  Done on `<data_root>` (2026-11-07): IBTrACS in `inputs/IBTrACS/`; `run_preparation.py
  storm_tide_product boundary_conditions` (76 + 3 min; the ~30 GB cache sits on this machine via
  `config_local.yml` `storm_tide_product.cache_dir`, P: has ~35 GB free); `WL_scenarios` now holds
  the product's levels (marker `storm_tide_source.txt`; level + MDT + 2020 offset checked to 1e-15
  m at all 22,670 stations); scenario configs re-materialized here. Left:
  1. re-materialize on Hydrax and set `storm_tide_product.cache_dir` in `config_hpc.yml` before
     any product run there (default: under processed_inputs on the same storage);
  2. Snakemake: the station caches re-run (their scenario NetCDFs are inputs) and with them every
     tile; then the SFINCS forcing, the fsf recalibration on the 498 tiles
     (`sfincs_tiles/generate_calibration_jobs.py stage1|stage2` on Hydrax), test 8 and the tile rerun.
  One processed_inputs tree serves every config of a data root: once it holds product levels,
  configs pinned to `coast_rp` (bangkok_chao_phraya, calibration_500_tiles, deltas_*) stop at
  `boundaries.check_scenario_source` until `boundary_conditions` is rebuilt for them.
- **Pinned scenarios and the new keys:** the scenario pins keep `storm_tide_source: coast_rp`,
  `knn: 15`, `idw_power: 2`; the dry run of bangkok_chao_phraya (re-materialized) has 599 jobs
  against the 480 baseline, from changed params (new keys in the `boundary_conditions` and
  `flooding` dicts, `hydrograph_path`) and the station cache's new input; results are identical.
  `--rerun-triggers mtime` skips them. Re-materialize the scenario configs on both machines
  (`snakemake_workflow/scripts/materialize_configs.py`; the checked-in bangkok one also lacks
  `slr_baseline_offset_m`).
- **Storm-tide product: decisions taken as estimates (open questions; tested in
  `boundary_test/INPUT_ANALYSIS.md` §8.6, fig19-fig23).**
  - TC mask (decide): at the gauges the mask makes TC coasts worse (TC-dominated RP100 median
    -0.17 m with it, -0.03 m without; MAE 0.39 vs 0.36 m; stations with a TC part -0.07 vs
    -0.02 m). Without it a cyclone counts twice (ERA5 record + COAST-RP TC part), which makes up
    for both sources being too low. Proposal: drop the mask (`TC_MASK_KM` = 0, or skip
    `tc_hours`) until test 10 replaces the TC part; a 150 km radius is the halfway option.
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
  - GPD shape cap 0.35 (`SHAPE_CAP`, Fathom RFA's cap; binds at 149 stations): kept. The gauges
    cannot tell caps apart (RP100 MAE 0.224-0.227 m from none to 0.2); at RP1000 no cap adds up
    to 9.7 m at heavy-tailed stations, 0.2 removes up to 3.1 m. Above RP100 nothing checks it.
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

- Outputs at `friction_scale_factor` 9.0 (deltas, bangkok_chao_phraya) and 30.0
  (calibration_500_tiles) are kept as computed; each config says so. Production is 7.5 (with B).
- A hop>=1 triple whose neighbour output is missing fails without writing output. Local multi-hop
  `snakemake simulate` / `run_eikonal_dag.py` can therefore fail when Snakemake schedules a hop-1
  tile before its neighbours (the rule does not declare neighbour inputs); rerun, or run wave by
  wave as `run_delta_floodmaps.py` does.

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
