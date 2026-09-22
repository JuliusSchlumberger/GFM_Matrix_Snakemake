"""Direct unit tests for two later-stage src/tile_chunking.py functions
with no prior test coverage:

  - `split_oversized_chunks`: when EVERY candidate split axis would either
    strand a dry piece or cut through a river mouth, the chunk is
    deliberately left oversized rather than accepting either - a known,
    accepted production behaviour (config.yml's own comment: "real runs
    have left ~1-2% of chunks over this cap for exactly that reason").
    Untested before this: whether that "leave oversized" path actually
    fires correctly (vs. silently dropping the chunk, or looping forever).
  - `cap_overlap_density`: its own docstring explains why "safely
    redundant" requires >=2 OTHER covering chunks, not just >=1 - a naive
    ">=1 other" check would strip a neighbour's own last remaining overlap
    buffer. Untested before this.

Usage:
    python validate_split_and_cap.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tile_chunking as tc  # noqa: E402


# ---------------------------------------------------------------------------
# split_oversized_chunks
# ---------------------------------------------------------------------------

def test_split_oversized_chunks_splits_when_a_valid_axis_exists() -> None:
    print("=== split_oversized_chunks: splits an oversized chunk when a valid (wet, no-river-mouth) axis exists ===")
    bbox = (0.0, 0.0, 10.0, 2.0)  # 10 wide x 2 tall, width exceeds max_extent=4
    with patch.object(tc, "_has_wet_edge", return_value=True), \
         patch.object(tc, "_axis_hits_river_mouth", return_value=False):
        result = tc.split_oversized_chunks(
            [bbox], mask_index={}, ocean_code=1, river_code=3,
            ocean_frac_min=0.02, river_frac_min=0.02, river_mouth_band_cells=1.0,
            coarse_resolution_m=500.0, max_extent=4,
        )
    assert len(result) > 1, f"expected the oversized chunk to actually split, got {result}"
    for minx, miny, maxx, maxy in result:
        assert round(maxx - minx) <= 4, f"piece {(minx, miny, maxx, maxy)} still exceeds max_extent=4"
    print(f"PASS: {len(result)} piece(s), none exceeding max_extent=4: {result}")
    print()


def test_split_oversized_chunks_leaves_chunk_oversized_when_no_valid_axis() -> None:
    """Every candidate axis fails the wet-edge check (simulating "every
    split would strand a dry piece") - the chunk must be returned
    UNCHANGED and OVERSIZED, not dropped, not infinite-looped."""
    print("=== split_oversized_chunks: leaves a chunk oversized when no axis is both wet and river-mouth-safe ===")
    bbox = (0.0, 0.0, 10.0, 2.0)  # still oversized (width 10 > max_extent 4)
    with patch.object(tc, "_has_wet_edge", return_value=False):
        result = tc.split_oversized_chunks(
            [bbox], mask_index={}, ocean_code=1, river_code=3,
            ocean_frac_min=0.02, river_frac_min=0.02, river_mouth_band_cells=1.0,
            coarse_resolution_m=500.0, max_extent=4,
        )
    assert result == [bbox], (
        f"expected the chunk to be left unchanged and oversized (no valid split axis), got {result}"
    )
    print(f"PASS: chunk left oversized and unchanged (no axis both wet and river-mouth-safe): {result}")
    print()


def test_split_oversized_chunks_river_mouth_cut_also_blocks_that_axis() -> None:
    """Wet-edge passes on both axes, but EVERY axis's cut band shows a
    river-mouth signature - must also fall through to leave-oversized,
    not just the dry-piece case."""
    print("=== split_oversized_chunks: a river-mouth-cutting axis is also rejected, leaving the chunk oversized ===")
    bbox = (0.0, 0.0, 10.0, 2.0)
    with patch.object(tc, "_has_wet_edge", return_value=True), \
         patch.object(tc, "_axis_hits_river_mouth", return_value=True):
        result = tc.split_oversized_chunks(
            [bbox], mask_index={}, ocean_code=1, river_code=3,
            ocean_frac_min=0.02, river_frac_min=0.02, river_mouth_band_cells=1.0,
            coarse_resolution_m=500.0, max_extent=4,
        )
    assert result == [bbox], result
    print(f"PASS: chunk left oversized (every axis would cut a river mouth): {result}")
    print()


def test_split_oversized_chunks_already_within_max_extent_is_untouched() -> None:
    print("=== split_oversized_chunks: a chunk already within max_extent is returned untouched ===")
    bbox = (0.0, 0.0, 3.0, 3.0)  # 3x3, within max_extent=4
    result = tc.split_oversized_chunks(
        [bbox], mask_index={}, ocean_code=1, river_code=3,
        ocean_frac_min=0.02, river_frac_min=0.02, river_mouth_band_cells=1.0,
        coarse_resolution_m=500.0, max_extent=4,
    )
    assert result == [bbox], result
    print("PASS: already-compliant chunk passed through unchanged, no wet-edge/river-mouth checks needed")
    print()


# ---------------------------------------------------------------------------
# cap_overlap_density
# ---------------------------------------------------------------------------

def test_cap_overlap_density_refuses_to_strip_a_neighbours_last_overlap() -> None:
    """Three chunks all cover one shared cell (overlap=3, over a cap of 2).
    Two of them (A, B) are covered elsewhere by exactly one OTHER chunk
    each (not two) - dropping either would leave its remaining neighbour
    at ZERO overlap there, so neither A nor B should be "safely redundant"
    (this is the exact scenario the function's own docstring uses to
    justify requiring >=2 OTHER covering chunks, not just >=1)."""
    print("=== cap_overlap_density: never drops a chunk that would zero out a neighbour's own overlap ===")
    # A, B, C all share cell (0,0); A also uniquely shares a DIFFERENT cell
    # with C only (1,0), and B uniquely shares a different cell with C only
    # (0,1) - so A's only "other" covering neighbour elsewhere is C alone,
    # same for B. Neither A nor B has 2 OTHER covering chunks anywhere.
    A = (0.0, 0.0, 1.0, 2.0)  # cols[0,1), rows[0,2) -> cells (0,0),(1,0)
    B = (0.0, 0.0, 2.0, 1.0)  # cells (0,0),(0,1)
    C = (0.0, 0.0, 2.0, 2.0)  # cells (0,0),(1,0),(0,1),(1,1)
    result = tc.cap_overlap_density([A, B, C], max_overlap=2)
    # cell (0,0) is covered by all 3 (over cap=2) - but A's only removable
    # basis is C (1 other, not 2), same for B; C is the largest so isn't
    # the smallest-first candidate either way. No chunk should have been
    # droppable here without breaking A's or B's own last overlap.
    assert set(result) == {A, B, C}, (
        f"expected all 3 chunks to survive (none is genuinely safely-redundant), got {result}"
    )
    print(f"PASS: all 3 chunks survive - cell (0,0) stays over the nominal cap rather than "
          f"breaking A's or B's own overlap buffer: {result}")
    print()


def test_cap_overlap_density_drops_a_genuinely_safely_redundant_chunk() -> None:
    """Four chunks all cover one shared cell (overlap=4, over a cap of 2).
    C and D are both small, identical-footprint duplicate chunks there,
    ALSO covered by both A and B independently - genuinely safely
    redundant (A+B alone already provide count=2, exactly the cap, with
    or without C/D) - so BOTH should be dropped (removing one doesn't
    disqualify the other, since A+B's own coverage there never depended on
    C or D), while A and B (each other's real, load-bearing coverage) must
    not be."""
    print("=== cap_overlap_density: drops chunks that genuinely have >=2 other covering chunks everywhere ===")
    A = (0.0, 0.0, 3.0, 1.0)  # cols[0,3), row[0,1) -> cells (0,0),(1,0),(2,0)
    B = (0.0, 0.0, 3.0, 1.0)  # identical footprint to A - always covers the same cells
    C = (0.0, 0.0, 1.0, 1.0)  # cell (0,0) only
    D = (0.0, 0.0, 1.0, 1.0)  # cell (0,0) only - identical to C
    result = tc.cap_overlap_density([A, B, C, D], max_overlap=2)
    assert set(result) == {A, B}, (
        f"expected both C and D dropped (each safely redundant given A+B's own coverage alone "
        f"already meets the cap there) and A/B to survive, got {result}"
    )
    print(f"PASS: both C and D dropped, A/B survive (A+B's own overlap at the shared cell already "
          f"satisfies max_overlap=2 without needing C or D): {result}")
    print()


def main() -> None:
    test_split_oversized_chunks_splits_when_a_valid_axis_exists()
    test_split_oversized_chunks_leaves_chunk_oversized_when_no_valid_axis()
    test_split_oversized_chunks_river_mouth_cut_also_blocks_that_axis()
    test_split_oversized_chunks_already_within_max_extent_is_untouched()
    test_cap_overlap_density_refuses_to_strip_a_neighbours_last_overlap()
    test_cap_overlap_density_drops_a_genuinely_safely_redundant_chunk()
    print("All split/cap validation checks passed.")


if __name__ == "__main__":
    main()
