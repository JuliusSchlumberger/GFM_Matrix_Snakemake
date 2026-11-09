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
