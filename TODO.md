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
- **Local DAG path lacks the hop-0 boundary-read change (465dd93).** Fixed by
  Phase 5b; then re-run the deltas (AUDIT.md §8 "After the cleanup").
