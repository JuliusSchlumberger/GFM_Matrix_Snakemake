# TODO: real changes found during the cleanup

The cleanup phases (AUDIT.md §8) are behaviour-preserving. Anything that needs
a real (behaviour- or result-changing) fix is logged here instead of being done
in-phase.

## Open

- **`tests/sfincs_mdt_sign_validation/validate_mdt_sign.py` fails (found in Phase 0, 2026-10-17).**
  `test_elevation_mdt_is_added_not_subtracted` expects the ocean cell to be
  GEBCO + MDT (-19.5664 m), but `sfincs_tiles/build_elevation.build_combined_elevation`
  returns 5.0 m, the land value. This is neither the added nor the subtracted value.
  It already fails at baseline (`.cleanup_baseline/validate_mdt_sign.log`), in hydromt-sfincs-dev.
  *Probable cause, not verified:* `_fill_coastal_transition_elevation` (added in
  283b07c, 2026-09-24) blends the coastal ocean band towards land elevation. On the
  test's 4×4 grid with a single ocean column, every ocean cell lies inside that band.
  If so, the fixture needs a wider ocean region, or the MDT sign should be asserted
  on `info`/`gebco_shifted` before the transition fill. That is a test fix, not a
  sign bug, but confirm that before changing anything.
- **FZ2 band figures disagree between the tracked CSVs (found in Phase 2b).**
  The note in `validation/comparison_vs_published.csv` says bias 2.4x and HR 0.066;
  `validation/overview_extent_metrics.csv` says bias 1.99 and HR 0.060. One of them is
  stale. Regenerate both (`build_overview_table.py`, `compare_against_published.py`)
  after the next England FZ2 run and check which is right. methods_04b currently cites
  only the overview CSV's bias.
- **England catalog comment is out of date.** The header comment of
  `england_ea_floodzone3_sea` / `_floodzone2_sea` in `data_catalog_validation.yml` says
  the data were filtered to `flood_source="sea"` only. `fetch_england_flood_zones.py`
  later added `river and sea`. Confirm which filter the current `fz*_sea_modelled.gpkg`
  were built with, then fix the comment.
- **`validation/validate_sf_bay.py` docstring** refers to `validate_country_depth_raster()`,
  which doesn't exist. `_rescore_england_fz3.py`'s docstring implies it uses
  `postprocessing.block_size`, but it hard-codes 8192 px. Both are docstring-only fixes
  (Phase 4/4b).
- **`fetch_england_flood_zones.py` and `build_overview_table.py` have no CLI** (no
  argparse). Any `--help` runs them for real, and fetch writes to P:\. Add argparse in
  Phase 6, when the hard-coded P:\ paths move to config.
- **Two console messages disappeared with the partial path.** "flood_totals missing →
  model_total_wet_km2 NaN" and "No coastal benchmark found for X" were only printed by
  the deleted `validate_country()`. The national path now prints just "No extent rows
  produced." Re-add them to the national path if you want them.
- **Snakemake "code has changed" on locally produced simulation outputs** (Phase 5b): `run_aqueduct.py`
  was rewritten into a thin wrapper, so e.g. bangkok_chao_phraya's 405 run_aqueduct outputs dry-run as
  "Code has changed since last execution" (with `--rerun-triggers mtime params input software-env`
  there is nothing to do, so inputs and outputs are untouched). Clear it once after the cleanup (item D,
  with the Phase 7a renames) via `snakemake --touch` - it writes Snakemake metadata next to P:\ outputs.
  The earlier "local path lacks the hop-0 fix" item was wrong (465dd93 only staged the file), so the
  deltas do not need a re-run.
- **SF Bay validation reads one hard-coded model chunk.** `validation/validate_sf_bay.py:100`
  points at `merged_results/chunks/waterdepth_N35W125_RP100_SLR_0.tif`. If the Bay domain
  spans more than that chunk, or the chunk grid changes, part of the Bay is silently not
  scored. Phase 6 moves the path into config; picking up overlapping chunks automatically
  (as `validate_country.py` does with `_chunks_overlapping_bbox`) would be a behaviour
  change, so it's logged here. Bringing SF Bay into `run_validation.py` is deferred (AUDIT.md §9.4 Q2).
- **`openpyxl` in environment.yml has no user** (found in Phase 4). Its only consumer,
  a calibration xlsx report script, was removed earlier; nothing imports openpyxl or
  calls `to_excel`/`ExcelWriter`. Dropping the pin is an env change, so it's logged here.
- **`src/flood_model.prune_to_coast_connected` docstring describes a "pre-solve candidate
  mask" use** (found in Phase 4b), but every caller uses it post-solve on the flood mask,
  and `flood_depth_dense` says the dense solver has no pre-solve candidate step. This is
  probably a leftover from an older (sparse) solver. Confirm, then trim the docstring.
- **`src/aqueduct_runner.py` module docstring** names `scripts/run_aqueduct.py` /
  `run_aqueduct_cli.py`; the real paths are under `snakemake_workflow/scripts/`. Fix it with
  the Phase 7a renames (these files become `run_eikonal*.py`).
- **`generate_aqueduct_jobs.generate_wave_dispatch` has 3 unused parameters** (found in
  Phase 4b): `model_outputs`, `hop_by_tile` and `raster_config` are accepted and passed by
  both callers (`rules/hpc_dispatch.smk`, `generate_hpc_simulation_jobs.py:178`) but never
  read. The docstring now says "unused". Drop them with the Phase 5c helper consolidation
  or the Phase 7a renames (signature change).
- **`generate_hpc_postprocess_job.py --help` crashes** (found in Phase 5a, pre-existing at HEAD):
  argparse raises `TypeError: %d format: a real number is required, not dict` because a
  `help=` string contains an unescaped `%`. Only `--help` is affected; normal runs parse fine.
  Fix: escape it as `%%`.
