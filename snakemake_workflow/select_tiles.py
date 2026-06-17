"""Filter the overlapping tile grid to tiles with DEM data coverage.

One-off script (run manually, not part of the Snakemake DAG): for each tile
in `paths.tile_grid` (see `config/config.yml`), checks whether
`deltadtm` (see `config/data_catalog_gfm.yml`) has any valid data within the tile's bounding box (see
`src/tiles.has_dem_coverage`). Tiles without coverage are logged and dropped.

The result is written to a new GeoPackage. Point `paths.tile_grid` at this
output so the workflow's static `TILE_IDS` (see `Snakefile`) only include
tiles with DEM coverage.

Usage (from `snakemake_workflow/`):
    python select_tiles.py [--input PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config_utils import get_preprocessing_catalog  # noqa: E402
from tiles import filter_tiles_with_dem_coverage, load_tile_grid  # noqa: E402


def main() -> None:
    """Filter `paths.tile_grid` to tiles with DEM coverage and write the result to a new GeoPackage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Tile grid to filter (default: paths.tile_grid from config/config.yml)")
    parser.add_argument("--output", help="Output path for the filtered tile grid (default: <input>_filtered.gpkg)")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent / "config" / "config.yml") as f:
        config = yaml.safe_load(f)

    input_path = Path(args.input or config["tile_grid"]["path"])
    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_filtered")

    catalog = get_preprocessing_catalog()
    tile_grid = load_tile_grid(input_path)
    filtered = filter_tiles_with_dem_coverage(
        tile_grid,
        catalog["deltadtm"].path,
        sample_size=config["tile_grid"]["dem_coverage_sample_size"],
    )
    filtered.to_file(output_path, driver="GPKG")
    print(f"Wrote {len(filtered)}/{len(tile_grid)} tiles with DEM coverage to {output_path}")


if __name__ == "__main__":
    main()
