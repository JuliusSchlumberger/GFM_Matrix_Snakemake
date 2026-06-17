"""Merge tiles with too little ocean or land coverage into a neighboring tile.

One-off script (run manually, not part of the Snakemake DAG): for each tile
in a tile grid, computes the ocean and land area fractions from
`land_use` (see `config/data_catalog_gfm.yml` and `src/tiles.compute_land_ocean_fractions`).
Tiles where either fraction is below `tile_merging.min_fraction` are merged
into a neighboring tile that meets both thresholds (see
`src/tiles.merge_undersized_tiles`), so that every tile has a reasonable
amount of both coastline and open water for the flood model's boundary
conditions.

Run this after select_tiles.py, on its `_filtered.gpkg` output, and point
`paths.tile_grid` at this script's output.

Usage (from `snakemake_workflow/`):
    python merge_tiles.py [--input PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config_utils import get_preprocessing_catalog  # noqa: E402
from tiles import compute_land_ocean_fractions, load_tile_grid, merge_undersized_tiles  # noqa: E402


def main() -> None:
    """Merge undersized tiles in a tile grid and write the result to a new GeoPackage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Tile grid to merge (default: paths.tile_grid from config/config.yml)")
    parser.add_argument("--output", help="Output path for the merged tile grid (default: <input>_merged.gpkg)")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent / "config" / "config.yml") as f:
        config = yaml.safe_load(f)

    input_path = Path(args.input or config["tile_grid"]["path"])
    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_merged")

    catalog = get_preprocessing_catalog()
    tile_grid = load_tile_grid(input_path)

    tg_cfg = config["tile_grid"]
    land_use_path = catalog["land_use"].path
    tile_grid = compute_land_ocean_fractions(
        tile_grid,
        land_use_path,
        ocean_code=tg_cfg["ocean_landcover_code"],
    )
    merged = merge_undersized_tiles(
        tile_grid,
        land_use_path,
        ocean_code=tg_cfg["ocean_landcover_code"],
        min_fraction=tg_cfg["min_coast_fraction"],
    )

    merged.to_file(output_path, driver="GPKG")
    print(f"Wrote {len(merged)}/{len(tile_grid)} tiles (after merging undersized tiles) to {output_path}")


if __name__ == "__main__":
    main()
