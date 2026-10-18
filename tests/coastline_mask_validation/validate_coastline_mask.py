"""Test flood_model.coastline_mask: edge-connected ocean adjacent to land or river.

Isolated inland "ocean" speckle (DeltaDTM miscodes ponds/aquaculture/thermokarst)
must not become coastline, or it gets IDW-seeded with no path to the sea.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    expected = mask == OCEAN
    assert np.array_equal(result, expected), result.astype(int)
    print("PASS: entire ocean ring correctly flagged as coastline")
    print()


def test_isolated_interior_speckle_excluded() -> None:
    print("=== isolated interior ocean-coded speckle, NOT touching the array edge -> excluded ===")
    mask = np.array([
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, OCEAN, LAND, LAND],
        [LAND, LAND, LAND, LAND, LAND],
        [LAND, LAND, LAND, LAND, LAND],
    ])
    result = fm.coastline_mask(mask, ocean_code=OCEAN)
    assert not result.any(), result.astype(int)
    print("PASS: isolated interior speckle correctly excluded (a plain land-dilation formula would flag it - it's ocean adjacent to land)")

    # A plain land-dilation formula must flag (2,2), so the case is not vacuous.
    old_formula = np.zeros_like(mask, dtype=bool)
    from scipy import ndimage
    dilated_land = ndimage.binary_dilation(mask == LAND, structure=np.ones((3, 3), dtype=bool))
    old_formula = dilated_land & (mask == OCEAN)
    assert old_formula[2, 2], "test setup error: plain land-dilation formula should flag (2,2)"
    print("PASS: confirmed a plain land-only-dilation formula would incorrectly flag (2,2) as coastline")
    print()


def test_river_adjacency_only_with_river_code() -> None:
    print("=== edge-connected ocean adjacent only to river (no land at all) -> included iff river_code given ===")
    mask = np.array([
        [OTHER, OTHER, OTHER, OTHER, OTHER],
        [OCEAN, OCEAN, OCEAN, OCEAN, OCEAN],
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
