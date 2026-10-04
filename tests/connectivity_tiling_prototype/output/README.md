# Connectivity-tiling prototype - validated output archive

The prototype scripts that produced these files (`connectivity_tiling.py`,
`world_phase012.py`, `world_full_run.py`, `test_tuned_budget.py`,
`compare_strategies.py`, `run_test.py`) have been removed (2026-10) - their
logic is now the real production implementation at
`src/connectivity_tiling.py`, orchestrated by
`preparation/build_tile_manifest.py`. See
`docs/methods_01_tile_processing_and_waterlevels.md` section 3 for the full
method description and validation numbers.

These files are kept as the real evidence behind that validation:

- `europe_0_baseline.gpkg` / `_1_merge.gpkg` / `_2_balanced.gpkg` / `_3_smoothed.gpkg`
  and the `southeast_asia_*` equivalents - the 3-iteration strategy
  comparison (post-hoc merge vs. balance-aware splitting vs. morphological
  smoothing) that picked the tiered-merge approach now in production.
- `southeast_asia_tuned_*.gpkg` - the budget/merge-ceiling tuning passes
  that settled on the 20M/30M/100M trigger/preferred-ceiling/hard-ceiling
  values now in `config.yml`'s `tile_generation` section.
- `world_phase1_edges.json`, `world_phase2_components.gpkg`,
  `world_phase3_checkpoint.jsonl`, `world_domains.gpkg` - the real
  full-world run (7,417 raw tiles, 4,504 final domains) that was promoted
  to production (`domain_tiles_global.gpkg`) via
  `preparation/promote_world_domains.py`.
- `largest_component_before_after_fix.gpkg` - the real tile-level evidence
  for the cross-component tile-leak bug fix (member_coords restriction in
  `_mosaic_nearest_coarse`).
