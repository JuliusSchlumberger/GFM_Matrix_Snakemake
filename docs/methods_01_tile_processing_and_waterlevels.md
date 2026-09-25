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

**Base grid.** The starting unit is the same global 1°×1° tile grid used
by the elevation data (section 1). A tile is kept for further processing 
only if it contains at least
one cell that is both on land and below a
plausible flood elevation (30m); tiles that are entirely open ocean,
entirely above that elevation, or have no valid elevation data at all are
dropped.

**Building domains.** Remaining tiles are merged into larger rectangular
simulation domains by a greedy covering procedure: starting from an
unprocessed tile, the algorithm grows the domain outward in each
direction as far as the underlying tiles remain present, up to a maximum
size of 4°×4°. This is repeated until every retained tile belongs to at
least one domain. Neighbouring domains are deliberately built with a
one-tile overlap along shared edges, so that boundary conditions and
flood results remain consistent across a domain boundary rather than
producing a visible discontinuity. Overlap is subsequently reduced where
it is redundant, and domains that add no unique coverage are dropped, to
keep the total number of domains and their total simulated area
manageable.

**Exposure filtering.** After domains are built, each is checked against
present-day population data (WorldPop, ~1km resolution, year 2020): a
domain with no population anywhere within its footprint is dropped from
the simulation set entirely, since there is nothing to assess flood
impact for there. This check is applied to the domains' final, already-
trimmed footprints (not their earlier, larger candidate extent), so that
a domain is not kept purely because some other, unrelated part of a
larger initial candidate area happened to be populated.

**Coastline and river-mouth handling.** Tiles are flagged as a river
mouth if a coarse check shows a substantial mixture of both open-ocean
and river water within the same area - a signature of a delta or estuary,
rather than open coast or an inland river reach on its own. This
flag is re-checked against a wider surrounding area before being
accepted, because small inland water bodies (ponds, aquaculture basins,
lakes) are occasionally misclassified as ocean in the source elevation
data at a single-tile scale, and would otherwise be mistaken for
coastline. Confirmed river-mouth tiles are used as preferred starting
points for domain growth, so that domains at deltas are built to hug the
true coastline rather than growing inland by chance.

**Simulation order.** Once domains are finalised, each is assigned a
"distance from the ocean" in terms of the number of domains that must be
crossed to reach open water (0 for a domain with its own ocean edge, 1
for a domain adjacent to one of those, and so on). Domains are then
simulated in that order: ocean-fronting domains first, using the water
levels described in section 4; inland ("hinterland") domains afterward,
each using the flood levels already computed for its lower-distance
neighbour(s) along their shared edge as its own boundary condition (see
section 4's final paragraph). This lets flooding propagate inland from
the coast through a sequence of domain-level simulations, without
requiring one single simulation spanning the whole domain chain at once.
A domain with no chain of neighbouring domains leading back to any
ocean-fronting domain at all - i.e. one that could never receive
boundary forcing of any kind - is dropped from the final domain set at
this stage.

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
