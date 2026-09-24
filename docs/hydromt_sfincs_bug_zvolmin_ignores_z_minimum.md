## Subgrid volume-table construction hardcodes a -20 m elevation floor that silently overrides the documented `z_minimum` parameter

**Component:** `SfincsModel.subgrid.create()` (`hydromt_sfincs.components.grid.subgrid`), specifically its internal call chain into `process_tile_regular()` / `subgrid_v_table()`.

**Version:** `hydromt_sfincs` 2.0.0-rc4dev.

**Summary:** `subgrid.create()` exposes a public `z_minimum` parameter, documented as "Minimum depth in the subgrid tables" (default `-99999.0`). It is applied once, to the raw elevation array. But the same array is then passed into the function that actually builds each coarse cell's hypsometric volume/water-level lookup table, which independently re-floors it at a second, hardcoded value of exactly `-20.0` - silently discarding whatever `z_minimum` was set to for any cell below that. The parameter's own docstring gives no indication this happens.

**Where:**
- `hydromt_sfincs/components/grid/subgrid.py`, in `create()` (~line 855): `da_dep = np.maximum(da_dep, z_minimum)` - `z_minimum` correctly applied here, to the raw elevation array.
- Same file, `process_tile_regular()` (~line 1112): `zvmin = -20.0` - a local, hardcoded constant, immediately passed into `subgrid_v_table()` at line 1113-1115 and applied a second time: `hydromt_sfincs/workflows/subgrid.py` line 59, `ele_sort = np.sort(np.maximum(elevation, zvolmin).flatten())`.

Net effect: `z_minimum` only changes the final volume table for values it raises to somewhere *above* -20 m. Anything at or below -20 m - including the parameter's own default of -99999.0, which promises no flooring at all - is clamped to exactly -20.0 m in the table SFINCS's solver actually uses for dry-state and volume bookkeeping, regardless of what the caller asked for.

**Expected:** `z_minimum` (or the absence of any floor, at its default) should determine the minimum elevation used throughout subgrid table construction, including the volume table.

**Actual:** the volume table silently re-clamps to a separate, hardcoded -20 m regardless of `z_minimum`.

**Impact on us:** this clamp is invisible everywhere else in the model output. We separately track the true (unclamped) elevation for our own postprocessing, which is the natural thing to do given `z_minimum`'s own documentation gives no reason to expect further clamping. Any land cell genuinely below roughly -18 m (a real terrain feature - closed/endorheic basins, below-sea-level depressions) then shows a large, entirely spurious "flood" purely from the mismatch between the solver's internal -20 m dry-state reference and the true elevation used elsewhere - with zero real inflow or connectivity involved.

Confirmed live on a real below-sea-level basin (surrounding land genuinely down to -39.32 m in our source DEM): the model's own `zsmax` output was exactly -20.00 m, constant across every one of 141 timesteps, for every affected cell - not a result that varies with the storm at all, just the clamp value itself. An independent, subgrid-free reference computed from the same source data reported a much deeper true level at the same location (39.99 m vs. the model's reported 19.32 m of "depth"), ruling out a source-data error and confirming the clamp as the cause.

**Suggested fix:** thread `z_minimum` through to `process_tile_regular()`/`subgrid_v_table()` as the actual `zvolmin`, replacing the hardcoded `-20.0`. If there's a numerical-stability reason -20 m specifically needs to stay as an absolute floor (the code comment says "needed with single precision"), at minimum document that prominently on `z_minimum` itself, so callers aren't misled into believing they've disabled a floor that in fact silently persists.

**Our current workaround:** floor all land elevation at -15 m (5 m of margin above the internal clamp) before ever calling `create()`, so the clamp is never triggered. This is lossy - any cell whose true elevation is below -15 m gets artificially raised to -15 m, so we lose real depth information for the (uncommon but real) below-sea-level terrain we're trying to represent correctly, purely to stay clear of a clamp we have no way to configure.
