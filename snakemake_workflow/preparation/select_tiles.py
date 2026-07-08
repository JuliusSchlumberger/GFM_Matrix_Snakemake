"""Filter the overlapping tile grid to tiles with DeltaDTM coverage and population exposure.

One-off script (run manually, not part of the Snakemake DAG): for each tile in
`tile_grid.path` (see `config/config.yml`), applies two sequential checks:

  1. DeltaDTM coverage: checks whether any 1°×1° DeltaDTM mask tile that
     overlaps the tile's bounding box contains land (0), lake (2), or river
     (3) pixels.  Tiles with only ocean (1) or nodata (255) are discarded.
  2. Population exposure: of the tiles surviving check 1, checks whether
     `exposure.population_source` (WorldPop population count) has any
     positive population within the tile's bounding box.  Tiles with zero
     population everywhere are discarded - nobody there for a flood to expose.

Outputs:
  <output>.gpkg               — filtered tile grid (default: <input>_filtered.gpkg)
  tiles_without_dem.gpkg      — tiles discarded by check 1, alongside <output>
  tiles_without_exposure.gpkg — tiles discarded by check 2, alongside <output>
  Both discard files retain full tile geometry, for visual confirmation (e.g. in QGIS).

Usage (from `snakemake_workflow/preparation/`):
    python select_tiles.py [--config PATH]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog  # noqa: E402
from tiles import filter_tiles_by_dem_mask, filter_tiles_by_exposure, load_tile_grid  # noqa: E402


def _expand(s: str, root: str) -> str:
    """Substitute the `{root}` placeholder used throughout config.yml paths."""
    return str(s).replace("{root}", root)


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"Path to config.yml (default: {_DEFAULT_CONFIG}).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root = config["paths"]["root"]
    input_path = Path(_expand(config['one_off_edits']['smaller_tiles'], root))
    output_path = Path(_expand(config['one_off_edits']['smaller_tiles_clean'], root))

    tiles_without_dem_path = output_path.parent / "tiles_without_dem.gpkg"
    tiles_without_exposure_path = output_path.parent / "tiles_without_exposure.gpkg"

    # The catalog entry for deltadtm_mask points to a VRT; the individual
    # 1°×1° tile files live in the same directory.
    repo_root = Path(__file__).resolve().parent.parent.parent
    catalog = get_data_catalog(repo_root / config["paths"]["hydromt_data_catalog"])
    mask_vrt_path = Path(catalog["deltadtm_mask"].path)
    mask_dir = mask_vrt_path.parent

    tile_grid = load_tile_grid(input_path)
    n_total = len(tile_grid)

    print("Check 1/2: DeltaDTM coverage")
    after_dem = filter_tiles_by_dem_mask(
        tile_grid,
        mask_dir=mask_dir,
        discarded_tiles_path=tiles_without_dem_path,
    )
    n_dropped_dem = n_total - len(after_dem)

    print("\nCheck 2/2: population exposure")
    filtered = filter_tiles_by_exposure(
        after_dem,
        data_catalog=catalog,
        population_source=config["exposure"]["population_source"],
        discarded_tiles_path=tiles_without_exposure_path,
    )
    n_dropped_exposure = len(after_dem) - len(filtered)

    filtered.to_file(output_path, driver="GPKG")
    print(
        f"\nWrote {len(filtered)}/{n_total} tiles to {output_path}\n"
        f"  {n_dropped_dem}/{n_total} excluded: no DeltaDTM coverage\n"
        f"  {n_dropped_exposure}/{len(after_dem)} (of the remaining {len(after_dem)}) excluded: no population exposure\n"
        f"  {len(filtered)} tiles kept"
    )


if __name__ == "__main__":
    main()
