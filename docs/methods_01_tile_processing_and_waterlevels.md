# Model preparation pipeline

Draft methods description of the full data-preparation pipeline that runs
before flood simulation: (1) acquiring the source elevation and coastal
water-mask data, (2) mosaicking it for efficient access, (3) dividing the
global coastline into individual simulation domains, and (4) deriving and
correcting the water levels used to force those domains. Written to inform
the Methods section of a paper; reviewed against the current
implementation but framed at the level of what is done and why, not code
structure.

## 1. Source elevation and coastal-mask data acquisition

The elevation data underlying the model is DeltaDTM v1.1 (Pronk et al.,
2024, 4TU.ResearchData), a global, ~30m resolution coastal digital terrain
model built specifically for near-coast flood applications, together with
its accompanying land/ocean/lake/river classification mask. Both are
distributed as a global set of 1°×1° tiles and are downloaded once,
per continent, from their public repository, then unpacked into a local
tile archive that the rest of the pipeline reads from directly.

## 2. Mosaic construction

The individual 1°×1° elevation and mask tiles are combined into two
virtual, globally-seamless mosaics, so that any later processing step can
read an arbitrary geographic window directly, without needing to know in
advance which individual tile(s) it falls in or to open more files than
necessary. This is a lightweight indexing step: it does not resample or
duplicate the underlying data, it only records where each tile sits within
the global mosaic.

## 3. Simulation domains

The model does not simulate the whole global coastline in one run. It is
divided into a large number of separate rectangular simulation domains,
each covering a stretch of coast or an inland floodplain, which are run
individually (in parallel where possible) and later combined.

Dividing the coastline this way is not a purely geometric problem. The
flood solver (an Eikonal bathtub-type model) requires that water be able
to propagate, within a single simulated domain, along any physically
connected low-lying path from its forcing boundary at the coast inland.
If a domain boundary bisects such a path without compensation, flood
extent on either side of that boundary can be artificially truncated -
wrong for a reason that has nothing to do with the physical problem, only
with how the domain was cut. A valid partitioning scheme must therefore
satisfy four requirements at once: every domain stays under a fixed
computational cell budget; every domain is expressible as a rectangle (the
solver's own grid requirement); no physically connected flood path is
silently severed by a domain boundary; and the full set of domains, and
the order they must run in, is computable up front, with no iterative
re-simulation.

**Base grid.** The starting unit is the same global 1°×1° tile grid used
by the elevation data (section 1) - no geometric merging or trimming
happens at this stage, it is simply the finest unit the rest of the method
operates on.

**Hydraulic-connectivity graph.** A graph is built with one node per base
tile. An edge is added between two laterally adjacent tiles only if real,
physically floodable land is present and touching across their shared
border, checked by sampling a thin strip of cells on each side of the
border. Open ocean is deliberately never treated as a connector: two tiles
sharing only open sea at their border are not linked, even though they are
geographically adjacent. A tile's exposure to the open ocean is already
fully represented through its own direct coastal boundary forcing (below);
two stretches of coastline happening to be adjacent is not evidence that
flooding in one could propagate overland into the other, and treating
ocean adjacency as a connector would inflate domain size with no
corresponding physical justification.

**Connected components.** A standard connected-components pass over that
graph partitions the full set of base tiles into disjoint clusters, giving
the method its central structural guarantee: every tile within one
component is reachable from every other tile in the *same* component via
an unbroken, physically floodable overland path, and tiles in *different*
components are never connected by any physically plausible overland flood
path. This is read directly from the terrain's own floodability structure,
not an approximation - once components are identified, no later step ever
needs to reconsider connectivity *between* components, only *within* one.

**Building domains.** A connected component may still be too large to
simulate as one domain (an entire continuous deltaic floodplain, for
example). Each component is reduced to one or more rectangular domains,
each within the computational budget, in a strict preference order. If the
component's bounding box - trimmed to its real floodable extent - already
fits the budget, it becomes a single domain with no internal seam. The
trimming rule is deliberately conservative: a row or column of the
bounding box is only ever dropped if it is *entirely* nodata or water
across every sample taken along it, never on the basis of elevation alone
and never partially - a binary, content-complete criterion, so two domains
derived from the same underlying data can never disagree about where real
content ends (the root cause of the overlap-erosion problem the previous
version of this pipeline had - see "Superseded design", below). If the
budget is exceeded, the method next searches for a genuine internal gap: a
contiguous band of entirely non-floodable cells running fully across the
component's extent, and splits there - free, since a cut through content-
free terrain cannot sever any real connectivity. Only when a component is
large and continuously floodable enough that no such gap exists anywhere
does the method fall back to a forced geometric cut along the component's
longer axis - the one case a synthetic domain boundary is introduced
through real floodable content, and it is compensated explicitly: a
bounded overlap margin (a fixed target width, not unbounded) is duplicated
on both sides of the cut, so that flood connectivity which would, in
reality, cross that line is still represented in both resulting domains.
Each domain's record of which of these three cases produced it is kept
alongside it, so the provenance of every domain boundary is traceable.
Undersized domains are subsequently merged with a same-component neighbour
where that keeps the result near the target band, to keep the total number
of domains manageable - merging, like the forced-split overlap above, only
ever happens within one connected component, never across one.

**Why this is a valid partitioning scheme.** Every domain boundary this
method produces falls into exactly one of three physically justified
categories, and no other kind of boundary is ever introduced: a boundary
between two different connected components, across which no physically
plausible connectivity exists at all; a boundary at a natural gap within
one component, across which there is no floodable content to lose; or a
forced boundary through genuinely continuous floodable terrain, the only
case where real connectivity is cut, and the only case compensated with an
explicit, bounded, duplicated margin sized to represent exactly the
connectivity that would otherwise be lost. No boundary is ever placed by
independent, local, per-tile judgment without reference to its neighbour -
the failure mode of the previous design (below). The decomposition is also
entirely data-driven (the same mechanism applies uniformly to simple
coastlines, estuaries, and complex multi-channel deltas without region-
specific tuning), bounded by construction, and single-pass/deterministic
(the full run order is computed once, up front, before any simulation).

**Simulation order and boundary forcing.** Once domains are finalised,
each is assigned a "distance from the ocean" in terms of the number of
domains that must be crossed to reach open water (hop distance: 0 for a
domain with its own direct ocean edge, 1 for a domain adjacent to one of
those, and so on) - found by a breadth-first search over domain adjacency,
run strictly within one connected component (by the connected-components
guarantee above, searching across components would be physically
meaningless). Domains are then simulated in that order: ocean-fronting
(hop 0) domains first, using the water levels described in section 4;
inland ("hinterland") domains afterward, each using the flood levels
already computed for its lower-hop neighbour(s) along their shared edge as
its own boundary condition (see section 4's final paragraph) - this lets
flooding propagate inland from the coast through a sequence of domain-
level simulations without one single simulation spanning the whole domain
chain at once. A domain with no path at all back to an ocean-fronting
domain within its own component - i.e. one that could never receive
boundary forcing of any kind - is dropped from the final domain set.

**Coastal buffer.** Every hop-0 domain's bbox is padded outward by a fixed
margin (order of several km) after the hop-distance assignment above. The
trimming step earlier already gives an *implicit* buffer around a coastal
nose or headland for free (water within the land's own row/column bounding
extent survives, since domains are rectangles, not per-pixel masks), but
not an *explicit* minimum-ocean-margin guarantee - this pad restores that
guarantee specifically for hop-0 domains, the ones self-forced directly
from the open coast, where a headland cutting too close to the domain edge
would otherwise matter. Hinterland (hop >= 1) domains are forced from an
already-simulated neighbour's wave rather than from direct coastline
geometry, so they have no analogous need and are left untouched.

**Exposure filtering - removed.** The previous version of this pipeline
dropped any domain with zero population (WorldPop) anywhere in it. This
method has no equivalent step: it can produce domains over genuinely
uninhabited floodable land (e.g. Arctic/Antarctic tundra) that the
previous pipeline would have dropped. Not yet revisited; noted here as a
real scope difference rather than decided silently.

**Validation.** The method was first tested on two structurally different
coastal regions - Southeast Asia (complex multi-delta coastline, 127 base
tiles) and Europe (a more linear coastline with fewer deltaic features, 96
base tiles) - then on the full global set of 7,417 base tiles, all using a
20-30 million cell target domain size (hard ceiling 100 million):

| | Southeast Asia | Europe | Global |
|---|---|---|---|
| Base tiles | 127 | 96 | 7,417 |
| Connected components | 18 | 11 | 1,002 |
| Final domains | 102 | 63 | 4,504 |
| hop = 0 (direct ocean forcing) | 77 (75%) | 55 (87%) | 3,761 (81%) |
| hop 1-3 | 22 (22%) | 7 (11%) | 658 (15%) |
| hop >= 4 | 0 | 0 | 85 (2%) |
| Maximum hop-chain length | 2 | 1 | 10 |
| Dropped as unreachable | 3 | 1 | 114 (3%) |
| Median domain size | 25.0 M cells | 26.0 M cells | 24.8 M cells |
| p25 / p75 domain size | - | - | 21.2 / 30.7 M cells |
| Domains exceeding the hard ceiling | 0% | 0% | 0% |

A meaningful fraction of domains in every test require relayed (hop >= 1)
rather than direct ocean forcing, confirming the hop-distance mechanism is
actively exercised by real terrain rather than a redundant safeguard - the
global run in particular surfaced a 10-hop relay chain, far deeper than
either regional test alone suggested. Every domain dropped as
"unreachable" that was checked was independently confirmed to be a
genuinely isolated, sub-km²-scale fragment of floodable land (e.g. a
single small islet) with no real path to open water within its own
connected component - exactly the case the method is designed to identify
and exclude rather than simulate with no meaningful boundary condition.
Not every domain lands within the 20-30 million cell target band (the
remainder stay under the hard ceiling but are larger than ideal) - a
size-tuning characteristic of the merge step, not a connectivity concern.

**Superseded design.** An earlier version of this pipeline built domains
by a greedy covering procedure (grow outward from an unprocessed tile up to
a 4°x4° cap, with a one-tile overlap along shared edges) and then cropped
each domain *independently* to its own floodable extent plus a small
buffer. Because that cropping decision was made per-domain, from each
domain's own local content only, two neighbouring domains could legitimately
make different cropping decisions along their shared border, eroding the
intended overlap - confirmed on the real production tile grid at the time
(2578 tiles, 2026-10-01): ~24% of neighbour pairs had overlap eroded below
~222m (functionally meaningless for boundary continuity), and ~0.8% had a
genuine gap of real, unmodelled daylight between domains, up to ~33km. That
pipeline also handled deltas/estuaries as a special case (flagging a
coarse ocean-river mixing signature and using it as a preferred seed point
for domain growth) rather than letting the general mechanism handle them -
the connectivity-first method needs no such special-casing, since the
hydraulic-connectivity graph already generalises correctly to any coastline
shape, deltas included, without region-specific tuning.

## 4. Water-level boundary conditions

**Extreme water levels.** The water levels used to force the coastal
domains come from COAST-RP, a global dataset of storm-tide extreme water
levels at coastal stations, available for a range of return periods (2
to 1000 years). Antarctic stations are excluded, since the underlying
storm-tide estimates are considered unreliable there and the coastline is
not populated.

**Vertical reference correction.** The elevation data and the water-level
data are not natively expressed relative to the same reference surface,
and this is corrected before the two are combined:

- The elevation data is referenced to the EGM2008 geoid on release. It is
  converted to the GOCO06s geoid via a smooth, globally-computed
  correction surface, applied once and reused for every location.
- The water-level data is referenced to local mean sea level. To bring it
  onto the same GOCO06s reference as the elevation data, the mean dynamic
  topography (MDT) is added at each station - a satellite-altimetry-based
  estimate (the AVISO CNES-CLS22 product) of the persistent difference
  between the ocean surface and the reference geoid caused by ocean
  currents and water-density variations. Adding this correction converts
  a storm-tide level measured relative to the local sea surface into the
  same absolute reference frame the elevation model uses, so the two can
  be compared and combined directly.

**Sea-level rise.** Regional sea-level-rise projections (IPCC AR6,
SSP2-4.5 scenario, medium confidence, median estimate for the year 2100)
provide a spatial pattern of expected relative sea-level change that
varies by location, reflecting regional differences from ocean dynamics,
ice-sheet gravitational effects, and vertical land motion. This spatial
pattern is scaled to a set of prescribed global-mean sea-level-rise
levels (0, 0.5, 1.0, 1.5 and 2.0m), so each scenario keeps the same
regional pattern while representing a different amount of global-mean
rise. Where no reliable regional projection is available near a station,
the global-mean value is applied uniformly instead, so every station
always has a usable estimate.

**Combined water level.** For each return period and sea-level-rise
scenario, the water level used at a station is the sum of the storm-tide
extreme value, the vertical reference (MDT) correction, and the
sea-level-rise increment for that scenario.

**Assigning stations to a domain.** For each ocean-fronting simulation
domain, candidate stations are drawn from within the domain's own
footprint plus a surrounding search margin (with a minimum search area
enforced for small domains, so a very small domain is never left with too
few candidate stations). Candidates are then checked for ocean
connectivity to the domain: a station that falls within the search area
by straight-line distance but is separated from the domain by land (for
example, on the far shore of a narrow strait or isthmus) is excluded.
The water levels of the remaining, ocean-connected stations are
interpolated onto the domain's own coastal cells to produce its boundary
forcing. An ocean-fronting domain that finds no ocean-connected station
at all within its search area (a short or sparsely-gauged stretch of
coast) is not dropped from the simulation set - it is instead assigned an
explicitly empty boundary forcing, which is a known limitation for the
small number of domains this affects.

**Hinterland domains.** Domains with no ocean edge of their own do not
draw on COAST-RP stations at all. Instead, once their lower-distance
neighbour domain(s) (section 3) have already been simulated for a given
scenario, the flood water levels those neighbours produced along the
shared domain edge are used directly as the boundary condition for the
next domain inland, for that same scenario. This lets a storm surge or
sea-level-rise signal propagate progressively further inland, one domain
at a time, following the "distance from ocean" simulation order.

## Scope note

This covers the full preparation pipeline that runs before simulation:
data acquisition, mosaicking, domain generation, and water-level boundary
preparation. Land-cover/friction data and the flood-solver methodology
itself are separate topics not covered here.
