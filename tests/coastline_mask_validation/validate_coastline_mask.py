"""Validate `flood_model.coastline_mask`'s 2026-08 fix - restricting the
coastline to ocean cells that are (a) part of a connected-component that
actually touches the tile's own array edge, and (b) adjacent to land OR
(when `river_code` is given) river.

The bug being fixed: the original formula (`dilate(landmask) & (mask ==
ocean_code)`) flags ANY ocean-coded cell touching land as coastline,
including isolated inland "ocean" speckle that DeltaDTM regularly
miscodes (ponds/aquaculture/thermokarst) - such a patch would otherwise
get IDW-seeded from real ocean stations and start flooding from a location
with no real path to the sea.

Pure array-logic unit tests - no raster I/O.

Usage:
    python validate_coastline_mask.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import flood_model as fm  # noqa: E402

LAND, OCEAN, RIVER, OTHER = 0, 1, 3, 9


def test_edge_connected_ocean_flagged() -> None:
    print("=== real, edge-connected ocean ring around a land island -> entire ring is coastline ===")
    mask = np.array([
        [OCEAN, OCEAN, OCEAN, OCEAN, OCEAN],
        [OCEAN, LAND,  LAND,  LAND,  OCEAN],
        [OCEAN, LAND,  LAND,  LAND,  OCEAN],
        [OCEAN, LAND,  LAND,  LAND,  OCEAN],
        [OCEAN, OCEAN, OCEAN, OCEAN, OCEAN],
    ])
    result = fm.coastline_mask(mask, ocean_code=OCEAN)
    expected = mask == OCEAN  # every ocean cell here is edge-connected AND 8-adjacent to the land block
    assert np.array_equal(result, expected), result.astype(int)
    print("PASS: entire ocean ring correctly flagged as coastline")
    print()


def test_isolated_interior_speckle_excluded() -> None:
    print("=== isolated interior ocean-coded speckle, NOT touching the array edge -> excluded (the bug fix) ===")
    mask = np.array([
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, OCEAN, LAND, LAND],  # isolated speckle at (2,2), surrounded by land, no edge contact
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, LAND, LAND, LAND],
    ])
    result = fm.coastline_mask(mask, ocean_code=OCEAN)
    assert not result.any(), result.astype(int)
    print("PASS: isolated interior speckle correctly excluded (old formula would have flagged it - it's ocean adjacent to land)")

    # sanity: confirm the OLD (buggy) formula really would have flagged it,
    # so this test is actually exercising the fix and not a vacuous case
    old_formula = np.zeros_like(mask, dtype=bool)
    from scipy import ndimage
    dilated_land = ndimage.binary_dilation(mask == LAND, structure=np.ones((3, 3), dtype=bool))
    old_formula = dilated_land & (mask == OCEAN)
    assert old_formula[2, 2], "test setup error: old formula should flag (2,2)"
    print("PASS: confirmed old land-only-dilation formula would have incorrectly flagged (2,2) as coastline")
    print()


def test_river_adjacency_only_with_river_code() -> None:
    print("=== edge-connected ocean adjacent only to river (no land at all) -> included iff river_code given ===")
    mask = np.array([
        [OTHER, OTHER, OTHER, OTHER, OTHER],
        [OCEAN, OCEAN, OCEAN, OCEAN, OCEAN],  # edge-connected via col 0 and col 4
        [RIVER, RIVER, RIVER, RIVER, RIVER],
        [OTHER, OTHER, OTHER, OTHER, OTHER],
        [OTHER, OTHER, OTHER, OTHER, OTHER],
    ])
    without_river = fm.coastline_mask(mask, ocean_code=OCEAN, river_code=None)
    assert not without_river.any(), without_river.astype(int)
    print("PASS: without river_code, ocean-adjacent-only-to-river is correctly excluded (no land anywhere)")

    with_river = fm.coastline_mask(mask, ocean_code=OCEAN, river_code=RIVER)
    expected = mask == OCEAN
    assert np.array_equal(with_river, expected), with_river.astype(int)
    print("PASS: with river_code given, the same ocean row is correctly included as coastline")
    print()


def main() -> None:
    test_edge_connected_ocean_flagged()
    test_isolated_interior_speckle_excluded()
    test_river_adjacency_only_with_river_code()
    print("All coastline_mask validation checks passed.")


if __name__ == "__main__":
    main()
