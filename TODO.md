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

Decided, to implement:
- **SLR propagation (critique 1a):** two-component solve. Static level S = MDT + SLR spreads by
  coast connectivity without attenuation and acts as sea; the total level (storm tide + MDT + SLR)
  is attenuated from the new shoreline. Switch `flooding.slr_propagation: attenuated | two_component`;
  default stays `attenuated` until validated.
- **Protection rule:** water-level based, switch `protection.rule: flood_fraction | water_level`
  (implemented; `minimum_rp` applies to the water-level rule only).
  An event is protected where storm_tide(RP) + SLR <= storm_tide(RP_design) at the nearest COAST-RP
  station (per-cell effective protection RP per SLR level). The adaptation Protect measure raises
  the design level to storm_tide(RP_design) + SLR_intensity.
- **Permanent inundation:** counted in EAI by anchoring the exceedance curve at p = 1 with the
  permanently inundated exposure, and reported as a separate metric.
- **Minimum protection RP2:** explicit setting, also for cells without a geogunit; sub-sea-level land
  is protected at SLR_0 and loses protection once MDT + SLR exceeds the RP2 design level.

Open:
- **Tide in the static level:** decide later whether S = MDT + SLR (current choice) should also
  include a tidal high water (MHHW). Needs a tide dataset (GTSM tide statistics or FES2014); COAST-RP
  cannot separate tide from surge.
- **`protection.default_rp: 5`** for geogunits without FLOPROS values vs the RP2 minimum: keep 5?
- **Event volume / duration correction (critique 3):** compare (A) a weir-inflow volume cap per
  enclosed basin (spill level by morphological reconstruction from the coast, single level pool per
  basin, no fill-spill-merge, per tile with edge-touching basins flagged, weir C = 1.7 fixed plus a
  sensitivity check) against (B) friction scaled by time above ground from the COAST-HG hydrograph;
  B is preferred if it performs as well (simpler). Friction must be recalibrated with the
  correction on, and RP10 / RP1000 transferability tested with it. Overtopping volume over defence
  crests is deferred (would make hazard depend on protection). Open: hydrograph at the sill taken as
  the coastal one shifted down by the eikonal attenuation loss (default, not yet confirmed).
- **Flood-fraction rule never exposes saturated cells:** where the design event already floods the
  whole cell (ff = 1), no larger event or SLR can exceed the design flood fraction, so the cell stays
  protected forever. Bangkok case: 33% of the population in the design floodplain sits in such cells.
  Production uses this rule; the water-level rule does not have the flaw.
- **Water-level rule + binary breach is harsh under SLR:** over all COAST-RP stations the median RP100
  standard drops to "fails every year" by ~0.25 m SLR (RP1000 by ~0.4 m), and a defence then counts
  as fully breached. Bangkok: protect at the +1 m design equals no adaptation from +1.5 m. Needs the
  deferred overtopping-volume treatment (or freeboard) before the water-level rule is adopted.
- **Production wiring of two-component propagation:** outputs would overwrite the attenuated rasters
  under the same paths; needs its own scenario / output location and a `flooding` config switch
  (changing `flooding` re-triggers every run_eikonal job via params).
- **Permanent inundation p = 1 anchor:** needs the static-flood extent from the two-component solve;
  implement together with it.
- **Materialized configs on the cluster:** re-materialize after pulling (new keys `protection.rule`,
  `protection.minimum_rp`).
- **Cluster runs:** SFINCS at +1 m / +2 m SLR (benchmark for 1a) and at RP10 / RP1000 on ~100 tiles
  (transferability, critique 1b; RP100 COAST-HG shape shifted to each RP peak). Tile count and SLR
  levels to fix; check that no calibration batch is running before touching `sfincs_tiles/`.
- **Land-cover friction (critique 2):** uniform-attenuation runs at equal mean (and 0.2 m/km) for
  Table A3; keep or drop land-cover friction depending on the result; rename to land-cover-indexed
  attenuation rate (no Manning) in code and docs.

## Known behaviour

- Outputs at `friction_scale_factor` 9.0 (deltas, bangkok_chao_phraya) and 30.0
  (calibration_500_tiles) are kept as computed; each config says so. Production is 12.0.
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
