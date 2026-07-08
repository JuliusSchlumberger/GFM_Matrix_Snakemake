"""Merge tiles with too little ocean or land coverage into a neighbour.

One-off script (run manually, not part of the Snakemake DAG): for each tile in
a tile grid, computes fractions from the DeltaDTM mask VRT:
  - ocean_fraction   : share of covered pixels that are ocean (mask == 1)
  - land_fraction    : share of covered pixels that are land/lake/river (mask in {0,2,3})
  - mask_fraction    : share of the tile's total pixels covered by the DeltaDTM mask
  - nodata_fraction  : 1 - mask_fraction (diagnostic only)

Every tile is then trimmed to its land-containing extent (+ a coastal buffer)
before any merging - see src/tiles.compute_trimmed_bbox. Tiles that still fail
ocean_fraction or land_fraction >= tile_grid.min_coast_fraction (evaluated
against each tile's ORIGINAL, untrimmed fractions) are merged in two phases:
water-deficient tiles unionize with their highest-water cardinal neighbour,
then land-deficient tiles unionize with their highest-land cardinal neighbour
(see src/tiles.merge_undersized_tiles). A final dedup pass consolidates any
tiles that end up covering the same physical feature.

Run this after select_tiles.py and point tile_grid.path at this script's output.

Outputs:
  <output>.gpkg            — merged tile grid
  <output>_fractions.csv   — ocean/land/mask/nodata fractions for all input tiles

Usage (from `snakemake_workflow/preparation/`):
    python merge_tiles.py [--config PATH] [--input PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog  # noqa: E402
from tiles import (  # noqa: E402
    compute_tile_fractions,
    compute_trimmed_geometries,
    deduplicate_overlapping_tiles,
    load_tile_grid,
    merge_undersized_tiles,
)


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
    parser.add_argument(
        "--input",
        help="Tile grid to merge (default: one_off_edits.smaller_tiles_clean from config)",
    )
    parser.add_argument(
        "--output",
        help="Output path for the merged tile grid (default: tile_grid.path from config)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root = config["paths"]["root"]
    input_path  = Path(args.input  or _expand(config["one_off_edits"]["smaller_tiles_clean"], root))
    output_path = Path(args.output or _expand(config["tile_grid"]["path"], root))
    fractions_csv = output_path.with_stem(output_path.stem + "_fractions").with_suffix(".csv")

    repo_root = Path(__file__).resolve().parent.parent.parent
    catalog = get_data_catalog(repo_root / config["paths"]["hydromt_data_catalog"])
    # The catalog entry points to the VRT; individual 1°×1° tile files live
    # in the same directory and are read directly to avoid VRT gap ambiguity.
    mask_dir = Path(catalog["deltadtm_mask"].path).parent

    tg_cfg = config["tile_grid"]

    tile_grid = load_tile_grid(input_path)

    if fractions_csv.exists():
        print(f"Found existing fractions CSV at {fractions_csv}, reusing it (skipping fraction computation)")
        fractions = pd.read_csv(fractions_csv)
        tile_grid = tile_grid.merge(fractions, on="tile_id", how="left")
        missing = tile_grid["mask_fraction"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} tile(s) in {input_path} are missing from {fractions_csv}; "
                "delete the CSV to recompute fractions"
            )
    else:
        tile_grid = compute_tile_fractions(tile_grid, mask_dir)
        # Save fractions for all input tiles before any merging
        tile_grid[["tile_id", "ocean_fraction", "land_fraction", "mask_fraction", "nodata_fraction"]].to_csv(
            fractions_csv, index=False
        )
        print(f"Wrote tile fractions to {fractions_csv}")

    trim_buffer_arcsec = tg_cfg["trim_buffer_arcsec"]

    # Trim every tile to its land-containing extent (+ buffer) BEFORE merging.
    # This becomes each tile's real working geometry from here on - the
    # ocean_fraction/land_fraction columns above (on the ORIGINAL, untrimmed
    # footprint) are what drive merge_undersized_tiles' decisions; trimming
    # first would trivially inflate a tile's own fraction and defeat the
    # thresholds' purpose.
    tile_grid["geometry"] = compute_trimmed_geometries(tile_grid, mask_dir, trim_buffer_arcsec)

    merged = merge_undersized_tiles(
        tile_grid,
        min_coast_fraction=tg_cfg["min_coast_fraction"],
        max_merge_count=tg_cfg["max_merge_count"],
    )

    # Re-tighten any diagonal slack left by bbox-union merges.
    merged["geometry"] = compute_trimmed_geometries(merged, mask_dir, trim_buffer_arcsec)

    # Dedup pass: merge_undersized_tiles only ever compares a deficient tile
    # against its neighbours, so two tiles that both individually pass the
    # fraction thresholds but happen to trim down to the same physical
    # feature (e.g. two overlapping-grid tiles sharing one small island)
    # are never compared to each other there. Catch those here, then
    # re-trim once more so the consolidated geometry stays tight.
    merged = deduplicate_overlapping_tiles(
        merged,
        iou_threshold=tg_cfg["dedup_iou_threshold"],
        max_merge_count=tg_cfg["max_merge_count"],
    )
    merged["geometry"] = compute_trimmed_geometries(merged, mask_dir, trim_buffer_arcsec)

    merged.to_file(output_path, driver="GPKG")
    print(
        f"Wrote {len(merged)}/{len(tile_grid)} tiles "
        f"(after merging undersized tiles) to {output_path}"
    )


if __name__ == "__main__":
    main()
