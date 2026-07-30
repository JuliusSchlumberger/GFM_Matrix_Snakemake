# Method: Friction-Weighted Eikonal Propagation for Coastal Flood Depth Estimation

*Draft methods-section text. Placeholders and items needing verification before
submission are marked **[VERIFY]**.*

## 1. Overview

Coastal flood hazard (extent and depth) is estimated for each raster tile and
scenario (a return period of extreme still water level, optionally combined
with a sea-level-rise increment) from three static raster inputs — a digital
elevation model (DEM), a land/ocean/lake mask, and a spatially-varying
hydraulic friction (resistance) surface — together with a set of discrete
boundary water-level points describing the offshore forcing for that
scenario. Rather than solving the full time-dependent shallow-water equations
at every grid cell, the model treats inland flood propagation as a
**static, boundary-value problem governed by the eikonal equation**, with the
land-cover-dependent friction field acting as a spatially varying propagation
resistance. This formulation is solved numerically with the Fast Sweeping
Method (Zhao, 2005 **[VERIFY citation]**), giving a flood water level (and
hence depth) at every cell in a single, non-iterative-in-time pass per
scenario. The approach follows the method implemented in the Deltares
*Aqueduct Coastal Flooding* model **[VERIFY citation / reference]**, of which
this pipeline is an extension.

## 2. Conceptual basis

### 2.1 Why not solve the shallow-water equations directly

The target output is coastal flood extent and depth for a very large number
of (tile × return period × sea-level-rise) combinations, at fine (≈30 m)
resolution, at continental-to-global scale. Solving the full 2-D shallow
water equations (mass and momentum conservation, explicit time-stepping,
wetting/drying) for every one of these combinations is computationally
infeasible at this scale, and arguably unnecessary: the boundary forcing
itself is a static extreme-value quantity (a design still-water level for a
given return period, not a time-varying hydrograph or storm track), so the
quantity of interest is the *steady-state inundation extent and depth*
associated with that water level having been sustained long enough to
propagate inland — not the transient arrival dynamics of a particular storm.

### 2.2 The eikonal-propagation analogy

Instead, the model reframes inland flood propagation as an anisotropic
**cost-distance / geodesic-distance problem**, formally identical to the
eikonal equation used to describe wavefront propagation in geometric optics
and seismic first-arrival travel-time computation:

$$
|\nabla T(x)| = v(x), \qquad x \in \Omega
$$

with $T$ prescribed on a seed set $\Gamma$ (the coastline). Here $v(x) > 0$
is a spatially varying "slowness" field — in this application, the local
hydraulic friction/resistance derived from land cover — and $T(x)$ is
interpreted as the (negative) water level rather than a physical travel
time (see §4.4). The physical intuition is direct: floodwater advancing
inland preferentially follows the path of least resistance, and the
water level attained at any point is controlled by the *cumulative
frictional head loss* along the least-resistive connected path from the
coast, not by straight-line (Euclidean) distance or by elevation alone.

### 2.3 Friction as a land-cover-informed resistance surface

The friction field $v(x)$ is derived from a global land-cover
classification (ESA WorldCover), reclassified to a Manning's roughness
coefficient $n(x)$ per land-cover class via a land-cover-to-roughness
mapping table (processed with the HydroMT/HydroMT-SFINCS toolchain), then
converted to a per-cell friction/resistance value used directly as the
eikonal equation's slowness field. Dense vegetation and built-up land are
assigned substantially higher roughness/resistance than open water, bare
ground, or grassland, so floodwater is attenuated faster crossing rough
terrain than open terrain — matching the qualitative expectation that
(e.g.) a mangrove fringe or urban block impedes inland flood propagation
more than a flat, open floodplain of the same elevation.

### 2.4 Why this is an acceptable approximation

1. **Matches the nature of the forcing.** The boundary condition is itself
   a static extreme-value statistic (a return-period still-water level,
   possibly with an SLR offset), not a transient hydrograph — so a
   steady-state propagation model answers the question actually being
   asked ("how far/how deep does floodwater reach for this design water
   level") without needing to resolve transient dynamics that the input
   forcing does not itself represent.
2. **Captures the dominant first-order controls.** For slowly-varying,
   surge/tide/sea-level-driven coastal inundation (as opposed to
   fast, momentum-dominated flash-flood or dam-break waves), the two
   controls generally accepted as dominant for the inundated extent are
   (a) topography and (b) resistance to overland flow — both of which
   this formulation represents explicitly, unlike a naive elevation-only
   ("bathtub") fill.
3. **Improves on simpler large-scale alternatives.** Purely
   elevation-threshold ("bathtub") inundation mapping is a common
   large-scale alternative but ignores flow resistance entirely and can
   flood elevation-connected but hydraulically implausible areas. This
   method corrects that in two ways: (i) the friction-weighted propagation
   cost suppresses inland penetration through resistant terrain even where
   elevation alone would permit it, and (ii) an explicit connectivity
   filter (§4.5) removes flooded cells that are not reachable via a
   continuously-flooded path back to the coast, eliminating
   elevation-only artefacts (isolated low-lying depressions with no real
   hydraulic connection to the sea).
4. **Computationally tractable at the required scale.** The Fast Sweeping
   solution has a small constant number of grid passes per scenario, in
   practice running in seconds to tens of seconds even for tiles of
   $10^8$–$10^9$ cells (§5), making global-scale, multi-scenario production
   runs feasible on standard compute infrastructure, unlike a
   spatially-resolved unsteady hydrodynamic model at the same resolution
   and extent.
5. **Known, explicit limitations.** The method does not resolve flood-wave
   arrival timing, momentum-driven wave run-up/overtopping beyond the
   prescribed boundary water level itself, or structural/hydraulic
   backwater effects not represented in the friction or elevation surfaces.
   It should be read as a steady-state end-state estimate appropriate for
   hazard and exposure mapping at the design water level supplied, not as a
   flood-forecasting or wave-dynamics tool.

## 3. Data inputs

| Symbol | Description | Source |
|---|---|---|
| $z(x)$ | Land surface elevation | DEM, land cells only (see §4.1) |
| $\mathrm{mask}(x)$ | Land / ocean / lake classification | Land-water mask raster |
| $v(x)$ | Local friction / hydraulic resistance | ESA WorldCover land cover, reclassified to Manning's roughness and converted to resistance |
| $\{(p_i, H_i)\}$ | Discrete boundary water levels at offshore/coastal points $p_i$, for the scenario (return period × SLR) being run | Offshore/coastal boundary-condition model output **[VERIFY exact source model]** |

## 4. Mathematical formulation

### 4.1 Pre-processing

- **Effective elevation**: elevation at non-land cells (ocean, permanent
  water) is set to $0$ (a common reference datum), so that flood/no-flood
  comparisons (§4.5) are well-defined everywhere.
- **Coastline seed set** $\Gamma$: ocean cells directly adjacent (3×3
  dilation of the land mask) to land — i.e. the immediate offshore fringe
  from which inland propagation is seeded.
- **Friction floor**: $v(x)$ is floored at a small positive value to avoid
  degenerate (zero-cost, infinite-speed) propagation at any cell.

### 4.2 Boundary condition: inverse-distance-weighted interpolation

Each coastline seed cell $x_c \in \Gamma$ is assigned an initial water
level by inverse-distance-squared weighting of the $k$ nearest boundary
points (great-circle/haversine distance on the geographic coordinates of
the boundary points and grid cells):

$$
H_0(x_c) = \frac{\sum_{i=1}^{k} w_i H_i}{\sum_{i=1}^{k} w_i}, \qquad
w_i = d(x_c, p_i)^{-2}
$$

where $d(\cdot,\cdot)$ is the haversine distance and $k$ is a configurable
number of nearest boundary stations (default **[VERIFY default value used
in production, e.g. $k=15$]**).

### 4.3 Governing equation

Define the state variable $T(x) = -H(x)$, the negative of the (to be
determined) flood water level. $T$ satisfies the eikonal equation

$$
|\nabla T(x)| = v(x), \qquad x \in \Omega \setminus \Gamma
$$

$$
T(x_c) = -H_0(x_c), \qquad x_c \in \Gamma
$$

solved over the **entire** raster domain $\Omega$ (land, ocean, and lake
cells alike — see §4.6 on why ocean is not excluded).

The viscosity solution of this equation is the friction-weighted geodesic
(cost) distance transform from the seed set:

$$
T(x) = \min_{\gamma:\, \Gamma \to x} \left[ T\big(\gamma(0)\big) +
\int_{\gamma} v(s)\, ds \right]
$$

i.e., in terms of the physical water level,

$$
H(x) = \max_{\gamma:\, \Gamma \to x} \left[ H_0\big(\gamma(0)\big) -
\int_{\gamma} v(s)\, ds \right]
$$

**the propagated water level at any point equals the boundary water level
at the best-connected coastline seed, minus the minimal cumulative
frictional head loss along the least-resistive path connecting them.**
This is the precise sense in which floodwater is modelled as following the
path of least resistance, and in which land-cover-driven resistance
attenuates inland water levels with (friction-weighted) distance from the
coast.

### 4.4 Numerical solution: the Fast Sweeping Method

The eikonal equation is solved on the raster grid using the Fast Sweeping
Method (Zhao, 2005 **[VERIFY]**), a Gauss–Seidel-type scheme that avoids
the need for a priority queue (as in Fast Marching) at the cost of
repeated grid sweeps.

**Grid staggering.** $T$ is defined on the grid *vertices*, one larger in
each dimension than the friction/elevation *cell* grid (an $(m+1)\times
(n+1)$ vertex array for an $m \times n$ cell array) — each vertex update
draws on exactly one corner cell's friction value and its two orthogonal
vertex neighbours.

**Local update.** At an interior vertex with two "upwind" neighbouring
values $t_a, t_b$ (already updated, from one specific sweep direction) and
local friction $v$, the first-order upwind discretisation of the eikonal
equation reduces to the quadratic

$$
2t^2 - 2(t_a + t_b)\,t + \left(t_a^2 + t_b^2 - v^2\right) = 0
$$

with causal solution

$$
t = \frac{(t_a + t_b) + \sqrt{2v^2 - (t_a - t_b)^2}}{2}
$$

accepted only when the discriminant is non-negative and the causality
condition ($t$ must be $\le$ both $t_a,t_b$ in this sign convention) holds;
otherwise the scheme falls back to the one-dimensional update
$t = \min(t_a, t_b) + v$. The vertex value is updated only if the computed
candidate improves on (is smaller than) its current value — i.e., a
standard Gauss–Seidel relaxation toward the eikonal solution of §4.3.

**Sweep ordering.** Each full iteration visits the grid in all four
combinations of ascending/descending row and column traversal order ("the
four orthants" of the 2-D Gray-code sweep sequence), so that information
can propagate in every direction regardless of raster storage order —
each direction uses the correspondingly-oriented pair of neighbours and
corner cell.

**Convergence.** Sweeps repeat until the maximum change across all
vertices in one full directional pass falls below a small tolerance
$\varepsilon$ (set from the minimum friction value and grid resolution).
**[VERIFY / describe for the paper as appropriate to the audience]**: in
this application's operating regime, the reference implementation's own
convergence criterion is satisfied after a small, fixed number of
directional sweeps rather than requiring many iterations to full
numerical convergence — a property of this specific formulation's sign
convention and parameter ranges, confirmed empirically across production
tiles, rather than a general property of Fast Sweeping methods.

### 4.5 Flood classification and depth

A cell is classified as flooded if the propagated water level exceeds the
local ground elevation, excluding permanent open water:

$$
\mathrm{flood}(x) = \big[ H(x) > z(x) \big] \ \wedge\ \big[\mathrm{mask}(x) \ne \text{ocean}\big]
$$

An additional **hydraulic connectivity filter** is then applied: only
8-connected components of $\mathrm{flood}(x)$ that touch the (dilated)
coastline are retained; any flooded region not reachable via a
continuously-flooded path back to the sea is discarded. This removes
elevation/propagation artefacts — cells that numerically satisfy the
threshold in isolation but have no physically connected inundation pathway
to the coast (e.g. below-threshold interior depressions).

Final flood depth is

$$
d(x) = \begin{cases} H(x) - z(x), & \mathrm{flood}(x) = \text{true} \\ 0, & \text{otherwise} \end{cases}
$$

### 4.6 Domain completeness

The eikonal equation is solved over the entire tile — land, ocean, and
lake cells together — rather than a restricted candidate subset. Ocean
cells (uniformly low friction) provide legitimate hydraulic shortcuts
connecting otherwise-separated stretches of coastline (e.g. around a
headland or across a bay); excluding them from the solve domain would
sever these connections and understate inland water levels near such
features.

## 5. Implementation and validation **[optional section — include if a reproducibility/provenance statement is wanted]**

The solver described above is implemented independently in Python (this
pipeline) alongside the original Julia reference implementation, to enable
deployment on HPC infrastructure where the reference toolchain is
unavailable. The Python port was validated against the reference
implementation's actual output on a sample of production tiles spanning a
wide range of sizes (up to $\sim 1.4\times10^8$ cells) and geographic
settings, achieving an exact, cell-for-cell match in both flood extent
(Jaccard index $100.000\%$, zero cells flooded by one implementation and
not the other) and flood depth (root-mean-square difference, mean
signed difference, 90th-percentile absolute difference, and maximum
absolute difference all $0.0\,\mathrm{m}$) across every tile tested. This
gives high confidence that the Python implementation faithfully reproduces
the reference model's numerical behaviour rather than merely
approximating it.

---

*Items marked **[VERIFY]** should be checked against the original Zhao
(2005) citation details, the Aqueduct Coastal Flooding project's own
documentation/publications, the exact default value of $k$ used in
production, and the land-cover-to-friction coefficient table's original
source, before this text is used in a submission.*
