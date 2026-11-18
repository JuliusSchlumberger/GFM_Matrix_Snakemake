# Coast boundary test: SFINCS forced at the coastline (C1) vs the station line (L0)

Plan: `<data_root>/inputs/coast_forcing_test_plan.md`. One section per stage; a stage starts only
after the previous section exists.

## Decision (2026-11-18): keep the station line (L0); C1 not pursued

The storm-tide product has hydrographs only at the COAST-RP stations; the coastline seed cells get
levels, not hydrographs (stage 0b). A coastline boundary would need per-cell series built from the
station series anyway, which is what the station line already forces: its points sit at the
COAST-RP stations, and SFINCS blends the two nearest points by distance (stage 0, point 1), so the
forcing varies continuously along the line. Stages 1-2 are not run; the SFINCS calibration keeps
`build_station_boundary_lines.py` / `build_line_forcing.py`. The stage-0b check
(`build_coast_forcing.py`, removed; in git at e182e65) showed what per-cell nearest-station series
would do: metre-scale level steps between neighbouring cells in macrotidal water.

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

Not the gate: the forcing source is the storm-tide product, which is being rebuilt. The product has
hydrographs only at the 23,226 COAST-RP stations (the same positions as COAST-HG); its coastline
nodes carry levels, not hydrographs (`src/storm_tide_product.py`). So a boundary cell's series has
to be built from station series either way, and this run with COAST-HG RP100 has the right station
geometry; only the shapes will change.

Measure (instability concern): per pair of adjacent boundary cells, the largest difference of their
**forced levels, unshifted**, over the 40-110 h window, `max_t |z_i(t) - z_j(t)|`, at ~120 m spacing
(native mask coarsened 4x). Two forcings, both peaking at the cell's seed:
- **nearest**: the nearest hydrograph station's series (plan step 5; the eikonal's duration
  correction does the same);
- **blend**: the inverse-distance blend of the knn = 5 nearest stations' series, weights
  distance^-3, as the seeds.
Stations are matched within the cell's ocean body on the tile grid, a station joining the body of
its nearest cell, as the seeds. (Matching on the global 30 m mask instead split Cobequid Bay at a
causeway the coarse grid does not resolve, giving 5 m jumps and 197 unconnected cells.)
Outputs: `.cleanup_baseline/coast_boundary/stage0b_coast_hg/`.

| tile | boundary cells | adjacent pairs with a station switch | seeds alone: max | nearest: max / pairs > 25 cm | blend: max / pairs > 25 cm |
|---|---|---|---|---|---|
| 3758 Rarotonga | 390 | 7 | 0.001 m | 0.03 m / 0 | 0.002 m / 0 |
| 891 Minas Basin | 26,863 | 159 | 0.22 m | **6.61 m** / 17 | 0.31 m / 6 |
| 1677 Dutch Wadden | 16,740 | 233 | 0.16 m | **2.05 m** / 95 | 0.32 m / 74 |

- Nearest-station forcing puts metre-scale level steps between neighbouring cells wherever the
  station switches in macrotidal water: neighbours share the peak (seed) but not the low water.
- Blending removes them. What remains: 891's 6 pairs sit where the seeds themselves step by 0.22 m;
  all of 1677's pairs above 10 cm are in the IJsselmeer and Markermeer.
- **1677: the IJsselmeer and Markermeer are ocean-coded and connected to the North Sea in the native
  mask**, so they get boundary cells at storm-tide seeds (as the eikonal coastline does). C1 vs L0
  on this tile will be dominated by that. Logged in TODO.md.

Leaning: blend (the per-cell series is then as smooth as the seeds). To decide on the product run:
the acceptable jump between neighbours.
