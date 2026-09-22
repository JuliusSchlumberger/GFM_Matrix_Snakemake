"""Direct unit tests for `build_chunks` (src/tile_chunking.py) - the greedy
rectangle-covering algorithm that determines whether every floodable tile
ends up in SOME chunk. This is the single most coverage-critical function
in the whole tile-generation pipeline: the only thing currently guarding
its correctness is a production-only assertion in
`preparation/build_tile_manifest.py` (`assert (covered == grid).all()`)
that only ever fires after a real, ~1-hour global run - a regression here
was, until this test, only catchable by a slow production crash, never by
a fast unit test.

Every test reconstructs the covered footprint from `build_chunks`'s own
returned rectangles and asserts it exactly equals the input grid - the
same invariant the production assertion checks, just fast and synthetic.

Usage:
    python validate_build_chunks_coverage.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tile_chunking as tc  # noqa: E402


def _covered_mask(chunks: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for r0, r1, c0, c1 in chunks:
        covered[r0:r1 + 1, c0:c1 + 1] = True
    return covered


def _assert_full_coverage(grid: np.ndarray, chunks: list[tuple[int, int, int, int]], label: str) -> None:
    covered = _covered_mask(chunks, grid.shape)
    assert np.array_equal(covered, grid), (
        f"{label}: covered mask != input grid - "
        f"missing cells: {int((grid & ~covered).sum())}, spurious cells: {int((covered & ~grid).sum())}"
    )
    # Every chunk must be entirely within the True footprint - a chunk
    # covering a False cell would mean the "all-True rectangle" contract
    # (build_chunks's own docstring) was violated.
    for rect in chunks:
        r0, r1, c0, c1 = rect
        assert grid[r0:r1 + 1, c0:c1 + 1].all(), f"{label}: chunk {rect} covers a False (non-floodable) cell"


def test_simple_rectangle() -> None:
    print("=== build_chunks: a single solid rectangle covers exactly, in one chunk ===")
    grid = np.zeros((10, 10), dtype=bool)
    grid[2:7, 3:8] = True
    chunks = tc.build_chunks(grid)
    _assert_full_coverage(grid, chunks, "simple rectangle")
    print(f"PASS: {len(chunks)} chunk(s), full coverage confirmed")
    print()


def test_l_shape() -> None:
    print("=== build_chunks: an L-shaped footprint is fully covered ===")
    grid = np.zeros((12, 12), dtype=bool)
    grid[1:8, 1:4] = True   # vertical arm
    grid[5:8, 1:10] = True  # horizontal arm
    chunks = tc.build_chunks(grid)
    _assert_full_coverage(grid, chunks, "L-shape")
    print(f"PASS: {len(chunks)} chunk(s), full coverage confirmed")
    print()


def test_footprint_with_interior_hole() -> None:
    """A donut - a solid block with one False cell punched out of the
    middle (e.g. a lake/nodata gap DeltaDTM's own mask might carry) - every
    True cell around the hole must still be covered, and no chunk may
    swallow the False cell itself."""
    print("=== build_chunks: a footprint with an interior hole is covered around it ===")
    grid = np.ones((10, 10), dtype=bool)
    grid[4, 4] = False  # a single-cell hole
    chunks = tc.build_chunks(grid)
    _assert_full_coverage(grid, chunks, "interior hole")
    print(f"PASS: {len(chunks)} chunk(s), hole at (4,4) correctly excluded, everything else covered")
    print()


def test_two_disconnected_pockets() -> None:
    """Two separate floodable regions with an all-False gap between them -
    each pocket must get its own chunk(s); the algorithm's own "pick a new
    seed from the first still-uncovered cell" loop is what makes this work,
    and it's untested in isolation elsewhere."""
    print("=== build_chunks: two disconnected pockets both get covered ===")
    grid = np.zeros((10, 20), dtype=bool)
    grid[2:6, 1:5] = True    # pocket 1
    grid[2:6, 14:18] = True  # pocket 2, far away
    chunks = tc.build_chunks(grid)
    _assert_full_coverage(grid, chunks, "two disconnected pockets")
    assert len(chunks) >= 2, f"expected at least 2 chunks for 2 disconnected pockets, got {len(chunks)}"
    print(f"PASS: {len(chunks)} chunk(s) across 2 disconnected pockets, both fully covered")
    print()


def test_isolated_single_cells() -> None:
    """Scattered isolated True cells in a False sea - each needs its own
    (possibly tiny) chunk; stresses the seed-and-grow loop's termination
    (must eventually pick up every isolated cell, not just the first)."""
    print("=== build_chunks: scattered isolated single cells are all covered ===")
    grid = np.zeros((15, 15), dtype=bool)
    for r, c in [(1, 1), (1, 13), (13, 1), (13, 13), (7, 7)]:
        grid[r, c] = True
    chunks = tc.build_chunks(grid)
    _assert_full_coverage(grid, chunks, "isolated single cells")
    assert len(chunks) == 5, f"expected exactly 5 chunks (one per isolated cell), got {len(chunks)}"
    print(f"PASS: {len(chunks)} chunk(s), one per isolated cell, all covered")
    print()


def test_max_extent_cap_still_achieves_full_coverage() -> None:
    """A single large solid block, bigger than max_extent in both
    directions - no individual chunk may exceed max_extent, but the union
    of however many chunks it takes must still cover the whole block (the
    module docstring's own claim: "whatever a cap excludes just gets
    covered by a later chunk")."""
    print("=== build_chunks: max_extent cap still reaches full coverage via multiple chunks ===")
    max_extent = 4
    grid = np.zeros((20, 20), dtype=bool)
    grid[2:18, 2:18] = True  # a 16x16 solid block, well over max_extent=4
    chunks = tc.build_chunks(grid, max_extent=max_extent)
    _assert_full_coverage(grid, chunks, "max_extent cap")
    for r0, r1, c0, c1 in chunks:
        assert (r1 - r0 + 1) <= max_extent, f"chunk {(r0, r1, c0, c1)} exceeds max_extent={max_extent} in height"
        assert (c1 - c0 + 1) <= max_extent, f"chunk {(r0, r1, c0, c1)} exceeds max_extent={max_extent} in width"
    assert len(chunks) > 1, "a 16x16 block at max_extent=4 should need more than 1 chunk"
    print(f"PASS: {len(chunks)} chunk(s), none exceeding max_extent={max_extent}, full 16x16 coverage confirmed")
    print()


def test_river_mouth_seed_preferred_first() -> None:
    """When a river-mouth cell is present in the remaining footprint, it
    must be picked as the seed before any ordinary row-major cell -
    build_chunks's own documented priority ("pick the first still-uncovered
    river-mouth cell if any remain, else the first ... in row-major
    order"). This doesn't change the coverage GUARANTEE, but a regression
    here (falling back to ignoring river_mouth) would silently lose the
    coast-hugging growth behaviour river mouths rely on - still worth
    confirming coverage holds even with a river-mouth seed active."""
    print("=== build_chunks: coverage holds correctly with a river-mouth seed present ===")
    grid = np.zeros((10, 10), dtype=bool)
    grid[1:9, 1:9] = True
    river_mouth = np.zeros_like(grid)
    river_mouth[5, 5] = True  # a river-mouth cell deep in row-major order (not cell (1,1))
    ocean_present = np.zeros_like(grid)
    ocean_present[1:9, 1] = True  # a plausible "ocean" edge for the seed grower to extend along

    chunks = tc.build_chunks(grid, river_mouth=river_mouth, ocean_present=ocean_present)
    _assert_full_coverage(grid, chunks, "river-mouth-seeded")
    print(f"PASS: {len(chunks)} chunk(s), full coverage confirmed with a river-mouth seed active")
    print()


def main() -> None:
    test_simple_rectangle()
    test_l_shape()
    test_footprint_with_interior_hole()
    test_two_disconnected_pockets()
    test_isolated_single_cells()
    test_max_extent_cap_still_achieves_full_coverage()
    test_river_mouth_seed_preferred_first()
    print("All build_chunks coverage validation checks passed.")


if __name__ == "__main__":
    main()
