"""Assigns each tile in the tile grid to a coarse geographic region (a
continent, or a longitude-bisected half of one for continents whose tile
count alone would make for an oversized batch), for grouping `simulate`
runs by geography instead of an arbitrary index range - see the Snakefile's
`simulate_region` rule and run_simulate_regions.sh.

Region assignment here is ORCHESTRATION-ONLY: it exists purely to decide
which run_aqueduct output files to ask Snakemake to build together in one
invocation, so a real geographic area finishes as a unit and becomes usable
for preliminary analysis before the rest of the world does. It has NO
effect on where or how any output is actually stored - every waterdepth
raster still lands at exactly
model_outputs/{tile_id}/results/waterdepth_{return_period}_{waterlevel_name}.tif
regardless of which region built it, and postprocessing (merge_chunk,
build_mosaic_vrt, ...) reads those same paths with no awareness that this
grouping exists at all. It's also unrelated to the country/continent
aggregation analysis/compute_exposure_analysis.py and friends already do
downstream from the population/geogunit rasters at global-grid resolution -
that machinery groups pixels by real ISO country codes for the actual
exposure results and is completely untouched by this module; this one only
ever groups whole tiles, for scheduling.
"""

import geopandas as gpd
from shapely.geometry import Point


def load_continent_polygons() -> gpd.GeoDataFrame:
    """The ~7-row Natural Earth continents layer bundled with geopandas -
    same source already used by plot_overlap_continent_diagnostics.py for
    grouping merge chunks; reused here for consistent continent naming."""
    return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))[["continent", "geometry"]]


def continent_for_point(x: float, y: float, continents: gpd.GeoDataFrame) -> str:
    """Point-in-polygon lookup against the continents layer; falls back to
    the nearest continent by centroid distance for points that land just
    offshore of every land polygon (common for coastal tile centroids)."""
    pt = Point(x, y)
    hit = continents[continents.contains(pt)]
    if hit.empty:
        hit = continents.iloc[[continents.distance(pt).idxmin()]]
    return str(hit.iloc[0]["continent"])


def assign_regions(
    tile_gdf: gpd.GeoDataFrame,
    n_scenarios_per_tile: int,
    max_jobs_per_region: int = 8000,
) -> dict[int, str]:
    """Return {tile_id: region_name}.

    region_name is a continent name (e.g. "Africa"), except for a continent
    whose tile_count * n_scenarios_per_tile would exceed max_jobs_per_region
    on its own - that one is bisected into "{continent}_West"/
    "{continent}_East" at the MEDIAN longitude of its OWN tile centroids
    (not a fixed global meridian), so both halves come out roughly
    equal-sized regardless of how that continent's tiles happen to be
    distributed in longitude.

    Args:
        tile_gdf: Tile grid, as returned by tiles.load_tile_grid - must have
            a `tile_id` column and tile polygon geometries.
        n_scenarios_per_tile: len(RETURN_PERIODS) * len(WATERLEVEL_NAMES) -
            each tile contributes this many simulate jobs.
        max_jobs_per_region: Soft cap used only to decide whether a
            continent needs bisecting; the resulting halves are not
            guaranteed to be under this themselves if a continent is more
            than 2x oversized (bisection is one pass, not recursive).
    """
    continents = load_continent_polygons()
    tile_continent: dict[int, str] = {}
    tile_lon: dict[int, float] = {}
    for tile_id, geom in zip(tile_gdf["tile_id"].astype(int), tile_gdf.geometry):
        centroid = geom.centroid
        tile_continent[tile_id] = continent_for_point(centroid.x, centroid.y, continents)
        tile_lon[tile_id] = centroid.x

    counts: dict[str, int] = {}
    for name in tile_continent.values():
        counts[name] = counts.get(name, 0) + 1
    oversized = {name for name, n in counts.items() if n * n_scenarios_per_tile > max_jobs_per_region}

    median_lon: dict[str, float] = {}
    for name in oversized:
        lons = sorted(tile_lon[tid] for tid, c in tile_continent.items() if c == name)
        median_lon[name] = lons[len(lons) // 2]

    regions: dict[int, str] = {}
    for tile_id, name in tile_continent.items():
        if name in oversized:
            suffix = "_West" if tile_lon[tile_id] <= median_lon[name] else "_East"
            regions[tile_id] = f"{name}{suffix}"
        else:
            regions[tile_id] = name
    return regions
