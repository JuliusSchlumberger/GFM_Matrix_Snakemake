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

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py merge_tiles`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog, retry_transient_io  # noqa: E402
from tiles import (  # noqa: E402
    compute_tile_fractions,
    compute_trimmed_geometries,
    deduplicate_overlapping_tiles,
    load_tile_grid,
    merge_undersized_tiles,
)


def run(config: dict) -> None:
    input_path  = Path(config["one_off_edits"]["smaller_tiles_clean"])
    output_path = Path(config["tile_grid"]["path"])
    fractions_csv = output_path.with_stem(output_path.stem + "_fractions").with_suffix(".csv")

    repo_root = Path(__file__).resolve().parent.parent.parent
    catalog = get_data_catalog(
        repo_root / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    # The catalog entry points to the VRT; individual 1°×1° tile files live
    # in the same directory and are read directly to avoid VRT gap ambiguity.
    mask_dir = Path(catalog["deltadtm_mask"].path).parent

    tg_cfg = config["tile_grid"]

    tile_grid = load_tile_grid(input_path)

    # fractions_csv doubles as both the incremental checkpoint (resumed from
    # if a run is interrupted partway) and the long-lived complete cache (so
    # a later clean re-run with different merge thresholds, same tile
    # grid/masks, can skip fraction computation entirely) - see
    # compute_tile_fractions' checkpoint_path docstring.
    tile_grid = compute_tile_fractions(tile_grid, mask_dir, checkpoint_path=fractions_csv)
    print(f"Tile fractions ready at {fractions_csv}")

    trim_buffer_arcsec = tg_cfg["trim_buffer_arcsec"]

    # Trim every tile to its land-containing extent (+ buffer) BEFORE merging.
    # This becomes each tile's real working geometry from here on - the
    # ocean_fraction/land_fraction columns above (on the ORIGINAL, untrimmed
    # footprint) are what drive merge_undersized_tiles' decisions; trimming
    # first would trivially inflate a tile's own fraction and defeat the
    # thresholds' purpose.
    #
    # This is the one call against the FULL (pre-merge) tile count - the only
    # one of the three compute_trimmed_geometries calls in this file long
    # enough to be worth checkpointing (the post-merge/post-dedup re-tighten
    # passes below run over far fewer, already-merged tiles). checkpoint_path
    # persists per-tile results as they're computed so an interrupted run
    # (network drop, Ctrl-C) resumes from here instead of tile 0 - see that
    # parameter's docstring in tiles.py.
    trim_checkpoint = output_path.with_stem(output_path.stem + "_trim_checkpoint").with_suffix(".csv")
    tile_grid["geometry"] = compute_trimmed_geometries(
        tile_grid, mask_dir, trim_buffer_arcsec, checkpoint_path=trim_checkpoint
    )

    merged = merge_undersized_tiles(
        tile_grid,
        min_coast_fraction=tg_cfg["min_coast_fraction"],
        max_merge_count=tg_cfg["max_merge_count"],
        cardinal_neighbor_overlap_threshold=tg_cfg["cardinal_neighbor_overlap_threshold"],
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

    retry_transient_io(merged.to_file, output_path, driver="GPKG")
    print(
        f"Wrote {len(merged)}/{len(tile_grid)} tiles "
        f"(after merging undersized tiles) to {output_path}"
    )


if __name__ == "__main__":
    sys.exit(
        "merge_tiles.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py merge_tiles\n"
        "See run_preparation.py --help for the full list of steps."
    )
