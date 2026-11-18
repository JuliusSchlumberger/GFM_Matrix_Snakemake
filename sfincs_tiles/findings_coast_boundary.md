# Coast boundary test: SFINCS forced at the coastline (C1) vs the station line (L0)

Plan: `<data_root>/inputs/coast_forcing_test_plan.md`. One section per stage; a stage starts only
after the previous section exists.

## Amendments to the plan (2026-11-18)

- **L0 is a fresh run.** The 1,000-tile run that was to be the control is deleted. L0 and C1 are built
  from the same inputs and code per test tile; they differ only in boundary placement, levels and
  zsini (below). The eikonal base variant and the friction sweep are rerun on each test tile.
- **Order.** The 50-tile test (stage 2) runs before the 1,500-tile calibration. `hydrax_rerun.yml`
  has a gate, `coast_boundary_decision`, before `reset_sfincs`. Stage 2 waits for
  `production_preprocess`, so both variants read the rebuilt storm-tide product and tile inputs.
- **Stage 0b, hydrograph smoothness along the coast (before stage 1).** C1 gives each boundary cell
  the nearest ocean-connected hydrograph station's series, shifted to the cell's seed peak. The peaks
  vary smoothly (IDW seeds); the shapes can jump where the assigned station switches. Measure per
  pair of adjacent coastline cells the shape difference after peak alignment,
  `max_t |(h_i(t) - max h_i) - (h_j(t) - max h_j)|`, and the timing difference of the peaks; map
  where they are large. Large jumps between neighbours would drive spurious along-shore flow.
- **zsini (C1).** Each boundary cell's start value is the first value of its own truncated series.
  zsini on the other active cells is interpolated from those start values over ocean and over river
  cells connected to the boundary through water only; land and unconnected water start dry
  (SFINCS takes `max(bed, zsini)`). L0 keeps its current zsini (IDW of the line points' first values
  on ocean cells; rivers dry), so the zsini change is part of the C1 difference.
- **Forcing points.** One point per boundary cell, at the exact cell centre, in netCDF
  (`netbndbzsbzifile`). No thinning (stage 0, point 1).
- **Stage 1 tiles.** 3758 (control-quality tile, Cook Islands), 891 (Minas Basin, macrotidal) and
  1677 (Dutch Wadden: barrier islands, tidal inlets, back-barrier basin; line suitable, so L0 runs).

## Stage 0: SFINCS boundary behaviour (2026-11-18)

Source: SFINCS v2.4.0_Galibier_release Fortran (the HPC version) and the online manual. The local
Windows build is v2.3.0; its boundary routines were not compared.

1. **Interpolation** (`sfincs_boundaries.f90`, `find_boundary_indices`): each msk = 2 cell takes the
   two nearest bnd points by straight-line distance in model coordinates, weight
   `dst2 / (dst1 + dst2)`. No maximum distance, no land or connectivity check: with fewer points
   than cells, a level is pulled across a spit or an island. With one point exactly at each cell
   centre, every cell gets its own series. Decides: one point per cell, no thinning, exact
   centres (netCDF; hydromt's ASCII writer rounds x/y to 0.1 m).
2. **Placement** (`sfincs_domain.f90`, `initialize_boundaries`): any msk = 2 cell is a boundary cell
   wherever it is; no msk = 0 neighbour or grid edge needed. Every face to an msk = 1 cell is a
   boundary face; faces between two msk = 2 cells and faces to msk = 0 are closed. Hence:
   - a channel that is all boundary cells carries no flow along itself; it feeds its banks at the
     imposed levels (consistent with the eikonal seeding);
   - a boundary cell without an active neighbour imposes nothing and keeps its initial level. The
     stage 1 forcing check counts only boundary cells with an active neighbour.
   hydromt's `create_boundary` picks only the outer ring of the active area, so the mask is set
   directly (plan step 3).
3. **Imposition** (`update_boundary_fluxes`): `bndtype = 1` (default, as in L0), weakly
   reflective Riemann-type face flux, boundary cell level clamped to the forced level, face
   velocity capped at 4 m/s, forced level floored at the cell's bed (subgrid zmin). A boundary cell
   whose bed is above its seed imposes nothing; counted per tile.
4. **Size**: no hard limit on bnd points; start-up search costs scale with boundary cells x points
   (thousands fine). netCDF forcing: dimensions `time`, `stations`; variables `x`, `y`, `zs`
   (`time`, `stations`), `time` as `minutes since tref`; it overrides bndfile/bzsfile.
5. **zsini**: initial level `max(bed, zsini)` on every cell including boundary cells; `tspinup`
   default 0, so there is no ramp from zsini to the forcing.

Open: v2.3.0 vs v2.4.0 boundary code (the HPC result of 3758 is deleted; stage 2 includes 3758, so
its HPC L0 and C1 runs are compared with the local stage 1 runs);
stability of a boundary cell enclosed by active cells (stage 1 forcing check).

Decides: C1 is buildable as planned with the mask set directly and one netCDF point per boundary
cell. Next: stage 0b once the storm-tide product rebuild has written
`processed_inputs/storm_tide_product/GFM_storm_tide_hydrographs.nc`.

## Stage 0b, preliminary run with COAST-HG (2026-11-18)

Not the gate: the forcing source is the storm-tide product, which is being rebuilt. This run used
COAST-HG RP100 (`inputs/COAST_HG/COAST-HG_RP100.nc`) to test the check and get a first idea.
`build_coast_forcing.py smoothness`, native mask coarsened 4x (~120 m), coastline and seeds as the
eikonal, nearest hydrograph station in the same ocean body. Outputs: `.cleanup_baseline/coast_boundary/stage0b_coast_hg/`.

| tile | boundary cells | hydrograph stations | station switches | shape jump max / p90 over station pairs (m) | median fall from peak (m) |
|---|---|---|---|---|---|
| 3758 Rarotonga | 390 | 4 | 7 | 0.03 / 0.03 | 0.85 |
| 891 Minas Basin | 26,863 | 22 | 165 | 4.86 / 0.37 | 1.99 |
| 1677 Dutch Wadden | 16,740 | 42 | 186 | 1.75 / 1.38 | 4.77 |

- COAST-HG peaks are all at 74.5 h, so the peak timing never differs between neighbours; the jumps
  are tide-phase and amplitude differences away from the peak. Most boundary cells sit next to a
  cell with the same station (zero jump); the jumps are concentrated at the switches, which are
  frequent where stations are dense (the Wadden islands, 891's south coast).
- 891: 197 cells have no hydrograph station in their ocean body and fall back to the nearest one
  overall; the p95 station distance is 0.88 deg.
- **1677: the IJsselmeer and Markermeer are ocean-coded and connected to the North Sea in the native
  mask**, so they get boundary cells at storm-tide seeds (as the eikonal coastline does). C1 vs L0
  on this tile will be dominated by that, not by the boundary placement. Logged in TODO.md.

To decide when the product run is in: the threshold for an acceptable jump between neighbours,
and whether the jump is measured over the whole 40-110 h window or only around the peak.
