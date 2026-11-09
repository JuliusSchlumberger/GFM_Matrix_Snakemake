# Findings: exposure and depth reliability of the eikonal against SFINCS

Assessment of `handoff_exposure_depth_assessment.md` on the existing 1000-tile SFINCS calibration
(RP100, earlier storm-tide forcing, COAST-RP seeding with knn 15 / power 2). Friction factors are the
effective factors of those runs (12 = 0.4 per metre x 30 m). No damage curves: the depth question is
how reliably the eikonal depth matches SFINCS within 10, 20 and 50 cm, per depth class, which bounds
the resolution of a depth-damage curve the model can support (user, 2026-11-09).

Terms:
- **both-wet cells:** 30 m subgrid cells on SFINCS-active land that both models flood (> 0.10 m,
  `exposure.exceedance_threshold_m`); the Voronoi core of each tile (dedup), so overlaps count once.
- **within tau:** share of both-wet cells with |eikonal depth - SFINCS depth| <= tau.
- **same-bin share:** share of both-wet cells whose two depths fall in the same depth bin of a given
  width; it depends on where the bin edges fall, which the tolerance form does not.
- **depth class:** the SFINCS depth of the cell, the reference.

## Stage 0: from the result files only

`exposure_depth.py stage0`; tables and figures in `.cleanup_baseline/exposure_depth/stage0/`.
Depths are stored as a 0.1 m joint histogram, so a cell pair's difference is known only to within a
bin. The estimate assumes depths uniform within their bins (checked against Monte Carlo); the
brackets are the bounds the binning allows. Exact values come from stage 1.

Pooled, friction factor 12 (189 M both-wet cells, 969 tiles with any):

| tolerance | within (estimate) | bounds |
|---|---|---|
| 10 cm | 0.31 | 0.18-0.43 |
| 20 cm | 0.52 | 0.43-0.60 |
| 30 cm | 0.66 | 0.60-0.72 |
| 50 cm | 0.83 | 0.80-0.86 |

Bias (eikonal - SFINCS) +0.03 m, RMSE 0.52 m (exact, from the sufficient statistics). Same-bin
shares: 0.18 (0.1 m bins), 0.48 (0.2 m classes), 0.40 (0.3 m), 0.55 (0.5 m bins).

By SFINCS depth class (within 20 / 50 cm; bias and MAE from bin centres):

| SFINCS depth | share of cells | bias | MAE | within 20 cm | within 50 cm |
|---|---|---|---|---|---|
| 0.1-0.3 m | 8 % | +0.34 m | 0.35 m | 0.50 | 0.75 |
| 0.3-0.5 m | 10 % | +0.20 m | 0.29 m | 0.56 | 0.83 |
| 0.5-1 m | 27 % | +0.05 m | 0.28 m | 0.51 | 0.84 |
| 1-1.5 m | 22 % | -0.07 m | 0.28 m | 0.51 | 0.82 |
| 1.5-2 m | 15 % | -0.13 m | 0.27 m | 0.51 | 0.83 |
| 2-3 m | 14 % | -0.19 m | 0.26 m | 0.51 | 0.85 |
| > 3 m | 4 % | not resolved by the histogram | | | |

Reading:
- The error is about 0.27 m MAE in every class: an absolute error, not one proportional to depth.
  Relative to depth it is largest in the shallow classes, which matter most for damage curves.
- The bias turns with depth: the eikonal is too deep where SFINCS is shallow (+0.34 m at 0.1-0.3 m)
  and too shallow where SFINCS is deep (-0.19 m at 2-3 m), a compression of the depth range.
- Tile medians are better than the pooled curve (tile median within 20 cm ~0.72 against 0.52
  pooled): the large, heavily flooded tiles agree less (X1).
- The tolerance shares peak at friction factor 9, not 12 (within 20 cm 0.53 vs 0.52; within 50 cm
  0.83 vs 0.83): the depth-optimal factor is close to the extent-optimal one. Not re-tuned.

Gate 0: pooled within 50 cm is 0.83 (> 0.6), so the depth question stays open and stage 1 runs.
Dropped for the next stage: nothing. Not answerable from the result files: the 10 cm share
(the bins are as wide as the tolerance) and where the disagreement sits (fringe vs deep, handoff
0.4: the files hold depths only for both-wet cells). Both come from the stage-1 raster pass.

## Stage 1: population-weighted agreement, one raster pass (friction factor 12)

`exposure_depth_pass.py select | stage1`, `exposure_depth.py report`; outputs in
`.cleanup_baseline/exposure_depth/stage1/`. Subset: 92 tiles with SFINCS flooding, 25 per tile
density class (people per km2 of tile land), all 14 tiles above 1000/km2, plus two SF Bay
tiles (728, 817) and one Bay of Fundy tile (891); no calibration tile covers the Severn. Population per subgrid pixel = WorldPop count x pixel
area / cell area (a full cell sums to its count within 1 %, test). Density classes below are those
of the WorldPop cell each pixel lies in. Depth differences are exact (1 cm bins).

Extent, pooled over the subset (HR = hit rate, FAR = false-alarm ratio, exposure bias = eikonal /
SFINCS flooded area or population):

| cell density | CSI area | CSI population | HR population | FAR population | exposure bias population |
|---|---|---|---|---|---|
| < 10 /km2 | 0.80 | 0.77 | 0.88 | 0.13 | 1.01 |
| 10-100 | 0.74 | 0.74 | 0.85 | 0.14 | 0.98 |
| 100-1000 | 0.73 | 0.73 | 0.78 | 0.09 | 0.85 |
| > 1000 | 0.71 | 0.60 | 0.74 | 0.24 | 0.97 |
| all | 0.77 | 0.64 | 0.75 | 0.19 | 0.93 |

- Population weights agree with area weights up to 1000 people/km2. In the densest class the
  population-weighted CSI drops to 0.60 (area 0.71): the eikonal floods densely populated cells
  SFINCS keeps dry (FAR 0.24) and misses others (HR 0.74); the two errors nearly cancel in the
  total (exposure bias 0.97), so totals hide them.
- Exposed population per tile is within a factor 2 of SFINCS for 84 % of the 83 tiles with any
  exposure (X5); the misses are mostly overestimates on tiles with little exposure, the largest
  tiles lie close to 1:1.
- Disagreement is not fringe-dominated: a third of the eikonal-only area (33 %; 25 % of its
  population) has eikonal depth < 0.3 m, and 36 % of the SFINCS-only area (34 %) SFINCS depth
  < 0.3 m. The 100 m population check (handoff 1.4) is dropped.

Depth on both-wet cells, exact (area-weighted / population-weighted):

| SFINCS depth | bias | MAE | within 10 cm | within 20 cm | within 50 cm | P90 abs |
|---|---|---|---|---|---|---|
| 0.1-0.3 m | +0.29 / +0.32 m | 0.31 / 0.35 m | 0.37 / 0.44 | 0.55 / 0.62 | 0.76 / 0.80 | 0.88 / 1.08 m |
| 0.3-0.5 m | +0.13 / +0.15 | 0.25 / 0.29 | 0.36 / 0.37 | 0.61 / 0.63 | 0.85 / 0.88 | 0.72 / 0.77 |
| 0.5-1 m | -0.03 / -0.08 | 0.24 / 0.30 | 0.33 / 0.27 | 0.54 / 0.45 | 0.88 / 0.85 | 0.54 / 0.60 |
| 1-1.5 m | -0.16 / -0.28 | 0.29 / 0.37 | 0.30 / 0.20 | 0.49 / 0.34 | 0.81 / 0.69 | 0.70 / 0.78 |
| 1.5-2 m | -0.21 / -0.28 | 0.29 / 0.35 | 0.31 / 0.23 | 0.49 / 0.39 | 0.81 / 0.74 | 0.69 / 0.76 |
| 2-3 m | -0.22 / -0.25 | 0.29 / 0.30 | 0.31 / 0.24 | 0.52 / 0.47 | 0.82 / 0.80 | 0.69 / 0.67 |
| > 3 m | -0.34 / -0.23 | 0.41 / 0.26 | 0.23 / 0.20 | 0.38 / 0.45 | 0.68 / 0.91 | 0.96 / 0.49 |
| all | -0.10 / -0.15 | 0.28 / 0.33 | 0.32 / 0.26 | 0.52 / 0.44 | 0.82 / 0.79 | 0.69 / 0.72 |

80 % of both-wet cells agree within 0.47 m (population: 0.53 m), 90 % within 0.69 m (0.72 m).

Reading for depth-damage curves:
- 10 cm: not supported; a quarter to a third of cells agree that closely, in every class.
- 20 cm: about half the cells (44 % of the population), and only a third of the population at
  1-2 m SFINCS depth.
- 50 cm: about 80 % of cells and population in every class from 0.3 m up; the 0.1-0.3 m class and
  the 1-2 m class of populated cells are lower (0.69-0.80).
- The errors are systematic, not just noise: the eikonal is too deep in shallow water (+0.3 m at
  0.1-0.3 m) and too shallow in deep water (-0.2 to -0.3 m from 1 m up), worse where people live.
  Depth classes with steps of about 0.5 m (e.g. < 0.5, 0.5-1, 1-1.5, 1.5-2, > 2 m) are what the
  model supports; curves resolved at 10-20 cm are not.
- This is accuracy relative to SFINCS on the same DEM and forcing; DEM error against reality adds
  to it. The SFINCS set is RP100 only.

Gate 1: Q1 answered on the subset. Population- and area-weighted metrics agree within noise up to
1000 people/km2; the > 1000/km2 class is the weakest (CSI 0.60, FAR 0.24) and every tile of that
class is already in the subset, so the full 1000-tile pass (1b) would not add to it and stays
queued for the paper figure only. 84 % of tiles are within a factor 2 (handoff target 90 %).
Dropped: the 100 m population check (1.4), stage 2 damage curves (user, 2026-11-09).

### Which cells count: the denominator of "within tau"

The shares above count only cells both models flood. Against the question "of all cells where
the reference gives a depth, what share does the eikonal match within tau", a cell the reference
floods and the eikonal leaves dry counts as eikonal depth 0 (reference-wet), and a cell only the
eikonal floods can be counted against reference depth 0 (either-wet). One-sided cells are known
from the stage-1 pass in 0.1 m depth bins against the other side's sub-threshold depth (0-0.10 m),
so their shares are bounds; SF Bay comes from the full difference raster and is exact.
Figure X1c (`exposure_depth.py cdf`): cumulative curves, area solid, population dashed.

| within | SFINCS, both-wet | SFINCS, SFINCS-wet | SFINCS, either-wet | CoSMoS, both-wet | CoSMoS, CoSMoS-wet | CoSMoS, either-wet |
|---|---|---|---|---|---|---|
| 10 cm | 0.32 / 0.26 | 0.27-0.30 / 0.19-0.24 | 0.24-0.29 / 0.16-0.22 | 0.07 / 0.15 | 0.05 / 0.07 | 0.03 / 0.03 |
| 20 cm | 0.52 / 0.44 | 0.47-0.49 / 0.38-0.42 | 0.44-0.48 / 0.34-0.39 | 0.14 / 0.27 | 0.11 / 0.15 | 0.08 / 0.12 |
| 30 cm | 0.66 / 0.59 | 0.61-0.63 / 0.53-0.56 | 0.59-0.62 / 0.48-0.53 | 0.23 / 0.38 | 0.18 / 0.24 | 0.14 / 0.21 |
| 50 cm | 0.82 / 0.79 | 0.79-0.80 / 0.74-0.77 | 0.77-0.79 / 0.70-0.73 | 0.46 / 0.56 | 0.41 / 0.42 | 0.29 / 0.43 |

(area / population.) Against SFINCS, counting the eikonal's misses lowers the shares by 3-6
points (area) and 4-7 points (population); the conclusion holds: about half of SFINCS's flooded
area within 20 cm, three quarters within 50 cm. Against CoSMoS in SF Bay, with DEM and forcing
differences included, under half of CoSMoS's flooded area is within 50 cm.
