"""Extract water level boundary points for a single tile, return period and SLR scenario."""

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boundaries import (  # noqa: E402
    filter_stations_by_ocean_connectivity,
    save_boundary_points,
    select_stations_for_tile,
)
from config_utils import retry_transient_io  # noqa: E402

bc_cfg = snakemake.params.bc_cfg  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

tile = retry_transient_io(gpd.read_file, snakemake.input.tile_geometry)  # noqa: F821

# Pre-parsed once per (return_period, waterlevel_name) scenario by
# cache_waterlevel_stations - NOT re-read from the raw NetCDF per tile here
# anymore (that NetCDF has no tile dimension at all, so every tile was
# redundantly re-parsing the exact same global station set - see that rule's
# own docstring, conversation 2026-08-10).
stations = retry_transient_io(gpd.read_file, snakemake.input.stations_cache)  # noqa: F821

selected = select_stations_for_tile(
    stations, tile,
    buffer_deg=bc_cfg["station_search_buffer_deg"],
    min_search_size_deg=bc_cfg["station_search_min_size_deg"],
)  # noqa: F821

# Coarse, long-range ocean-connectivity filter - select_stations_for_tile above
# is purely distance-based and can't tell a station across a thin land
# barrier from a genuinely nearby one; see filter_stations_by_ocean_connectivity's
# own docstring for why this has to be a separate, coarser pass rather than
# folded into the native-resolution IDW connectivity fix in flood_model.py.
# mask_dir is resolved once at Snakefile-parse time (preprocessing.smk's
# _deltadtm_mask_dir), not rebuilt from a fresh HydroMT DataCatalog on every
# one of the ~45 (return_period, waterlevel_name) jobs per tile this rule
# runs - that per-job rebuild (~5s each, pure catalog-parsing overhead for a
# path that never changes) was the dominant cost in a tile's whole
# preprocessing time - see conversation 2026-08-10.
mask_dir = Path(snakemake.params.mask_dir)  # noqa: F821
selected = filter_stations_by_ocean_connectivity(
    selected,
    tile,
    mask_dir=mask_dir,
    target_resolution_m=bc_cfg["connectivity_resolution_m"],
    water_fraction_threshold=bc_cfg["connectivity_water_fraction_threshold"],
)

# A tile that finds no real COAST-RP station at all (small/hinterland
# chunks, or genuinely isolated ones) used to be dropped from tile_grid.path
# entirely by the now-retired connectivity_map step. That step is gone, so
# this is now an explicit, intentional EMPTY placeholder instead - `selected`
# already has the right schema/crs even with zero rows, so save_boundary_points
# writes it as-is. Revisit once neighbour-derived boundary forcing (see
# src/tile_chunking.py's compute_run_order / the "NOT IMPLEMENTED" note in
# build_tile_manifest.py's Stage 13) exists to actually fill this in.
if selected.empty:
    print(f"  no COAST-RP station found for this tile/scenario - writing an empty "  # noqa: T201
          f"placeholder boundaries file (tile_id={snakemake.wildcards.tile_id})")  # noqa: F821

save_boundary_points(selected, snakemake.output.boundaries, column_name=waterlevel_name)  # noqa: F821
