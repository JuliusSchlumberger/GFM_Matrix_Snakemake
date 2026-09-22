"""Direct unit tests for the overlap-management stages of src/tile_chunking.py
that follow build_chunks: `reduce_overlap`/`_peel_with_splits` (Stage 4),
`add_minimum_overlap`/`_grow_together` (Stage 5), and `add_connector_chunks`
(Stage 6) - none of which had any test coverage before this. These three
functions are exactly where an "unintended gap" or "broken overlap buffer"
bug would show up, since they're what turns build_chunks's raw, messily-
overlapping rectangles into the final, cleanly-adjacent chunk set.

Usage:
    python validate_overlap_repair.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tile_chunking as tc  # noqa: E402


# ---------------------------------------------------------------------------
# reduce_overlap / _peel_with_splits
# ---------------------------------------------------------------------------

def test_peel_with_splits_resolves_an_l_shaped_fat_overlap() -> None:
    """A rect whose only unique territory wraps a settled region's corner
    (an "L": the whole top row + the whole left column) - _peel_to_core
    alone can't remove ANY edge here (every edge has at least one not-yet-
    settled cell blocking a full-edge peel), so this forces the split-and-
    retry fallback the module docstring explicitly calls out as a real,
    anticipated failure mode of plain peeling."""
    print("=== _peel_with_splits: an L-shaped unique-territory case converges to non-fat pieces ===")
    rect = (0, 5, 0, 5)  # 6x6
    settled = np.zeros((6, 6), dtype=bool)
    settled[1:6, 1:6] = True  # everything except row 0 and column 0 is already settled

    pieces = tc._peel_with_splits(rect, settled)

    assert pieces, "expected at least one piece back, got none"
    for p in pieces:
        assert not tc._has_fat_overlap(p, settled), (
            f"piece {p} is still fat against settled - _peel_with_splits failed to converge "
            f"(the L-shape split-and-retry fallback should always resolve this within max_depth)"
        )
    # Every genuinely unique cell (row 0 or column 0, within rect) must be
    # covered by at least one returned piece - no unique territory may be
    # silently lost during the split.
    unique_cells = np.zeros((6, 6), dtype=bool)
    unique_cells[0, :] = True
    unique_cells[:, 0] = True
    covered = np.zeros((6, 6), dtype=bool)
    for r0, r1, c0, c1 in pieces:
        covered[r0:r1 + 1, c0:c1 + 1] = True
    assert (covered | settled)[unique_cells].all(), "some unique (non-settled) L-shape cell was lost"
    print(f"PASS: {len(pieces)} piece(s), none still fat, all unique L-shape territory preserved: {pieces}")
    print()


def test_peel_with_splits_fully_redundant_rect_returns_empty() -> None:
    print("=== _peel_with_splits: a fully-redundant rect (entirely settled) returns nothing ===")
    rect = (1, 3, 1, 3)
    settled = np.ones((6, 6), dtype=bool)
    pieces = tc._peel_with_splits(rect, settled)
    assert pieces == [], pieces
    print("PASS: fully-settled rect peels away to nothing, as expected")
    print()


def test_reduce_overlap_largest_first_shrinks_small_chunk_around_big_one() -> None:
    """The module docstring's own claim: largest-first processing makes
    SMALL, mostly-redundant chunks the ones that peel down, not the other
    way around. A big chunk and a small chunk fully inside it (minus a
    tiny unique sliver) - the small one must survive only at its own
    unique sliver (shrunk), the big one must survive untouched."""
    print("=== reduce_overlap: largest-first - small chunk shrinks around the big one, not vice versa ===")
    big = (0, 9, 0, 9)  # 10x10, area 100
    small = (0, 2, 0, 11)  # 3 rows, but extends 2 cols past the big chunk's own right edge - real unique sliver
    chunks = [small, big]  # deliberately unsorted - reduce_overlap must sort by area itself
    result = tc.reduce_overlap(chunks, grid_shape=(12, 12))

    assert big in result, f"the big chunk should survive untouched: {result}"
    small_survivors = [r for r in result if r != big]
    assert len(small_survivors) == 1, f"expected exactly one shrunk remnant of the small chunk: {result}"
    r0, r1, c0, c1 = small_survivors[0]
    assert c1 >= 10, f"the small chunk's real unique sliver (cols 10-11) must survive: {small_survivors[0]}"
    print(f"PASS: big chunk survives untouched ({big}), small chunk shrinks to its own unique sliver "
          f"({small_survivors[0]})")
    print()


# ---------------------------------------------------------------------------
# add_minimum_overlap / _grow_together
# ---------------------------------------------------------------------------

def test_grow_together_succeeds_with_no_unrelated_conflict() -> None:
    print("=== _grow_together: grows the smaller side by 1 tile when nothing blocks it ===")
    grid = np.ones((5, 10), dtype=bool)
    near = (0, 2, 0, 2)
    far = (0, 2, 3, 5)
    coverage = np.zeros((5, 10), dtype=np.int16)
    coverage[0:3, 0:3] += 1  # near
    coverage[0:3, 3:6] += 1  # far

    new_near, new_far, unresolved = tc._grow_together(near, far, "col", grid, coverage, max_extent=None)
    assert not unresolved, "expected a clean resolution with no unrelated conflict present"
    assert new_near[3] == 3 or new_far[2] == 2, f"neither side grew to create overlap: {new_near}, {new_far}"
    row_overlap = min(new_near[1], new_far[1]) - max(new_near[0], new_far[0]) + 1
    col_overlap = min(new_near[3], new_far[3]) - max(new_near[2], new_far[2]) + 1
    assert row_overlap > 0 and col_overlap > 0, f"near/far still don't overlap after growth: {new_near}, {new_far}"
    print(f"PASS: near={new_near}, far={new_far} now genuinely overlap, unresolved={unresolved}")
    print()


def test_grow_together_refuses_to_create_fat_overlap_with_unrelated_chunk() -> None:
    """A third, unrelated chunk C sits exactly on the near/far boundary
    seam (rows 1-2, cols 2-3) - growing EITHER side into that seam would
    turn C's own legitimate 1-wide overlap with near/far into a fat (>1x>1)
    one. `_grow_together` must refuse both directions and report
    unresolved, rather than silently corrupting C's own overlap buffer -
    exactly the subtle guard (`_creates_fat_overlap`) that had no test at
    all before this."""
    print("=== _grow_together: refuses growth that would create a fat overlap with an unrelated chunk ===")
    grid = np.ones((5, 10), dtype=bool)
    near = (0, 2, 0, 2)
    far = (0, 2, 3, 5)
    coverage = np.zeros((5, 10), dtype=np.int16)
    coverage[0:3, 0:3] += 1  # near's own footprint
    coverage[0:3, 3:6] += 1  # far's own footprint
    coverage[1:3, 2:4] += 1  # unrelated chunk C, straddling the near/far seam (rows 1-2, cols 2-3)

    new_near, new_far, unresolved = tc._grow_together(near, far, "col", grid, coverage, max_extent=None)
    assert unresolved, (
        f"expected growth to be refused (both directions would create a fat overlap with C), "
        f"but got new_near={new_near}, new_far={new_far}, unresolved={unresolved}"
    )
    assert new_near == near and new_far == far, "chunks should be returned UNCHANGED when unresolved"
    print(f"PASS: growth correctly refused on both sides (unresolved={unresolved}), "
          f"near/far left unchanged - C's own overlap buffer wasn't corrupted")
    print()


# ---------------------------------------------------------------------------
# add_connector_chunks
# ---------------------------------------------------------------------------

def test_add_connector_chunks_bridges_a_touching_zero_overlap_pair() -> None:
    print("=== add_connector_chunks: bridges a touching-but-zero-overlap pair with a real connector ===")
    a = (0, 2, 0, 2)   # rows 0-2, cols 0-2
    b = (0, 2, 3, 5)   # rows 0-2, cols 3-5 - touches a's right edge (col2+1==3), zero shared cells
    chunks = [a, b]

    result, pairs_bridged = tc.add_connector_chunks(chunks)
    assert pairs_bridged == 1, pairs_bridged
    assert a in result and b in result, result
    connectors = [c for c in result if c not in (a, b)]
    assert len(connectors) == 1, f"expected exactly one connector chunk, got {connectors}"
    connector = connectors[0]
    assert tc._overlaps(connector, a) and tc._overlaps(connector, b), (
        f"connector {connector} doesn't actually bridge both {a} and {b} - violates add_connector_chunks's "
        f"own internal assertion"
    )
    print(f"PASS: connector {connector} added, genuinely overlaps both {a} and {b}")
    print()


def test_add_connector_chunks_no_op_for_already_overlapping_pair() -> None:
    print("=== add_connector_chunks: no connector added when chunks already overlap ===")
    a = (0, 2, 0, 3)
    b = (0, 2, 2, 5)  # overlaps a at cols 2-3, real shared cells
    result, pairs_bridged = tc.add_connector_chunks([a, b])
    assert pairs_bridged == 0, pairs_bridged
    assert result == [a, b], result
    print("PASS: no connector added for a pair that already genuinely overlaps")
    print()


def main() -> None:
    test_peel_with_splits_resolves_an_l_shaped_fat_overlap()
    test_peel_with_splits_fully_redundant_rect_returns_empty()
    test_reduce_overlap_largest_first_shrinks_small_chunk_around_big_one()
    test_grow_together_succeeds_with_no_unrelated_conflict()
    test_grow_together_refuses_to_create_fat_overlap_with_unrelated_chunk()
    test_add_connector_chunks_bridges_a_touching_zero_overlap_pair()
    test_add_connector_chunks_no_op_for_already_overlapping_pair()
    print("All overlap-repair validation checks passed.")


if __name__ == "__main__":
    main()
