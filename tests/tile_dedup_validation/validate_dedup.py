"""Validate `drop_redundant_chunks` (src/tile_chunking.py, 2026-08) - the
loosened, union-of-overlapping-chunks dedup pass that replaced the old
exact single-chunk `drop_fully_contained_chunks`. Prompted by real data
(tile_id 536/850 in a 2026-08 production run): 536 was only ~99.47%
covered by its single largest neighbour (a shave-step sliver misalignment
of a few hundred metres), and 850 was covered 100% only by the UNION of 5
neighbours, no single one covering more than ~60% of it alone - both
missed by the old exact single-chunk check.

Pure geometry-logic unit tests - no raster I/O, mirrors the style of
tests/tile_run_order_validation/validate_run_order.py.

Usage:
    python validate_dedup.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import tile_chunking as tc  # noqa: E402


def test_exact_duplicate_pair() -> None:
    print("=== exact duplicate pair - exactly one survives ===")
    bboxes = [(0, 0, 1, 1), (0, 0, 1, 1)]
    result = tc.drop_redundant_chunks(bboxes, coverage_threshold=0.998)
    assert len(result) == 1, result
    assert result[0] == (0, 0, 1, 1), result
    print(f"PASS: {len(bboxes)} identical chunks -> {len(result)} survives")
    print()


def test_no_single_neighbour_sufficient_union_does() -> None:
    print("=== union of 3 non-sufficient neighbours fully covers a 4th chunk ===")
    # T is covered exactly by the union of C1/C2/C3 restricted to T's own
    # extent, but no single Ci covers more than 1/3 of T alone, and none
    # of C1/C2/C3 is itself contained in T (each sticks out above T's top
    # edge, y=1..1.5, so they keep real unique territory of their own) -
    # mirrors the real tile_id 850 case (5 neighbours, none >~60% alone).
    T = (0, 0, 3, 1)
    C1 = (0, 0, 1, 1.5)
    C2 = (1, 0, 2, 1.5)
    C3 = (2, 0, 3, 1.5)
    bboxes = [T, C1, C2, C3]
    result = tc.drop_redundant_chunks(bboxes, coverage_threshold=0.998)
    assert T not in result, result
    assert set(result) == {C1, C2, C3}, result
    print(f"PASS: T dropped (100% covered by union of C1/C2/C3, none alone >1/3); C1/C2/C3 kept "
          f"(each retains real unique territory above y=1)")
    print()


def test_threshold_boundary() -> None:
    print("=== loosened threshold: sliver just inside vs just outside tolerance ===")
    # B is a proper subset of A missing a thin sliver at A's bottom edge -
    # B is always 100% covered by A (classic single-chunk containment),
    # but A's own coverage by B depends on the sliver's height.
    threshold = 0.998

    # sliver height 0.001 -> A's coverage by B = 0.999 >= threshold -> both drop-candidates
    # initially, but only B (smaller, genuinely 100% redundant) actually
    # gets dropped - once B is gone, A's sliver is uncovered again, so A
    # must survive (the mutual-dependency invalidation case).
    A = (0.0, 0.0, 1.0, 1.0)
    B = (0.0, 0.001, 1.0, 1.0)
    result = tc.drop_redundant_chunks([A, B], coverage_threshold=threshold)
    assert set(result) == {A}, result
    print(f"PASS: sliver=0.001 (within tolerance) -> B (smaller, fully redundant) dropped, "
          f"A (has the uncovered sliver once B is gone) survives: {result}")

    # sliver height 0.003 -> A's coverage by B = 0.997 < threshold -> A
    # never qualifies at all; B is still always 100% contained in A, so B
    # alone gets dropped.
    A2 = (0.0, 0.0, 1.0, 1.0)
    B2 = (0.0, 0.003, 1.0, 1.0)
    result2 = tc.drop_redundant_chunks([A2, B2], coverage_threshold=threshold)
    assert set(result2) == {A2}, result2
    print(f"PASS: sliver=0.003 (outside tolerance) -> same outcome (B fully contained, "
          f"A's own coverage never even attempted since only B qualified): {result2}")
    print()


def test_genuine_gap_kept() -> None:
    print("=== genuine coverage gap - well below threshold, both kept ===")
    # partial 50% overlap, neither a subset of the other - each has real
    # unique territory the other doesn't cover.
    A = (0.0, 0.0, 1.0, 1.0)
    B = (0.5, 0.0, 1.5, 1.0)
    result = tc.drop_redundant_chunks([A, B], coverage_threshold=0.998)
    assert set(result) == {A, B}, result
    print(f"PASS: A/B overlap only 50% each, neither a subset of the other -> neither dropped: {result}")
    print()


def main() -> None:
    test_exact_duplicate_pair()
    test_no_single_neighbour_sufficient_union_does()
    test_threshold_boundary()
    test_genuine_gap_kept()
    print("All dedup validation checks passed.")


if __name__ == "__main__":
    main()
