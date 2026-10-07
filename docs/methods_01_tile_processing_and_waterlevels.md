# Model preparation pipeline

Description of the full data-preparation pipeline that runs
before flood simulation: (1) acquiring the source elevation and coastal
water-mask data, (2) mosaicking it for efficient access, (3) dividing the
global coastline into individual simulation domains, and (4) deriving and
correcting the water levels used to force those domains.

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
individually and later combined.

Dividing the coastline this way is not a purely geometric problem. The
flood propagates inland along any physically
connected low-lying path from its forcing boundary at the coast.
If a domain boundary bisects such a path without compensation, flood
extent on either side of that boundary can be artificially truncated. A valid partitioning scheme must therefore
satisfy three requirements at once: every domain stays under a fixed
computational cell budget; every domain is expressible as a rectangle (the
solver's own grid requirement); no physically connected flood path is
silently severed by a domain boundary.

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
fully represented through its own direct coastal boundary forcing.

**Connected components.** A standard connected-components pass over that
graph partitions the full set of base tiles into disjoint clusters, giving
the method its central structural guarantee: every tile within one
component is reachable from every other tile in the *same* component via
an unbroken, physically floodable overland path, and tiles in *different*
components are never connected by any physically plausible overland flood
path. 

**Building domains.** A connected component may still be too large to
simulate as one domain. Each component is reduced to one or more rectangular domains,
each within the computational budget, in a strict preference order. If the
component's bounding box - trimmed to its real floodable extent - already
fits the budget, it becomes a single domain with no internal seam. The
trimming rule is deliberately conservative: a row or column of the
bounding box is only ever dropped if it is *entirely* nodata or water
across every sample taken along it. If the
budget is exceeded, the method next searches for a genuine internal gap: a
contiguous band of entirely non-floodable cells running fully across the
component's extent, and splits there. Only when a component is
large and continuously floodable that no such gap exists anywhere
does the method fall back to a forced geometric cut along the component's
longer axis - the one case a synthetic domain boundary is introduced
through real floodable content, and it is compensated by a
bounded overlap margin, so that flood connectivity which would, in
reality, cross that line is still represented in both resulting domains.
Each domain's record of which of these three cases produced it is kept
alongside it, so the provenance of every domain boundary is traceable.
Undersized domains are subsequently merged with a same-component neighbour
where that keeps the result near the target band, to keep the total number
of domains manageable - merging, like the forced-split overlap above, only
ever happens within one connected component, never across one.

**Simulation order and boundary forcing.** Once domains are finalised,
each is assigned a "distance from the ocean" in terms of the number of
domains that must be crossed to reach open water (hop distance: 0 for a
domain with its own direct ocean edge, 1 for a domain adjacent to one of
those, and so on). Domains are then simulated in that order: ocean-fronting
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
ocean forcing. Every domain dropped as
"unreachable" that was checked was independently confirmed to be a
genuinely isolated, sub-km²-scale fragment of floodable land (e.g. a
single small islet).


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