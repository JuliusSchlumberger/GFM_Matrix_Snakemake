"""Extract a single tile's geometry from the tile grid and save it to a GeoPackage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiles import get_tile_geometry, load_tile_grid, save_tile_geometry  # noqa: E402

tile_grid = load_tile_grid(snakemake.config["tile_grid"]["path"])  # noqa: F821
tile = get_tile_geometry(tile_grid, int(snakemake.wildcards.tile_id))  # noqa: F821

save_tile_geometry(tile, snakemake.output.tile_geometry)  # noqa: F821
