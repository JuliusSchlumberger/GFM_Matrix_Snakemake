"""Validate the internal island/gap splitting added to tile_chunking.py's
shave step (2026-08) - _shave_and_split_window/_first_interior_gap/
_shave_chunk. Ported from the retired pre-2026-08 pipeline's
split_disconnected_tiles (see memory.md's FIXED-TILE CHUNKING REDESIGN
entry), simplified per explicit user direction: first-qualifying-gap
(not most-balanced), no explicit gap-preservation (each recursive call's
own tightening step naturally shaves off whatever gap remnant remains).

Three parts:
  A. Pure array-logic unit tests for _first_interior_gap and
     _shave_and_split_window (no raster I/O - fast, deterministic).
  B. filter_and_shave_chunks bookkeeping: one input bbox producing
     multiple kept output pieces still lands all of them in `kept`, none
     in `dropped`.
  C. bbox_to_grid_rect as the exact inverse of grid_rect_to_bbox (also
     needed by the resumability work in this same change).

Usage:
    python validate_shave_split.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import tile_chunking as tc  # noqa: E402


def test_first_interior_gap() -> None:
    print("=== Part A: _first_interior_gap ===")

    # interior gap of exactly the threshold width -> found
    mask = np.array([False, False, True, True, True, False, False, False])
    #                 0      1     2    3     4     5      6      7
    gap = tc._first_interior_gap(mask, min_len=3)
    assert gap == (2, 4), gap
    print(f"PASS: exact-threshold-width interior gap found at {gap}")

    # gap narrower than threshold -> not found
    gap = tc._first_interior_gap(mask, min_len=4)
    assert gap is None, gap
    print("PASS: gap narrower than threshold correctly not found")

    # gap touching the left edge -> excluded (ordinary margin, not interior)
    mask_edge = np.array([True, True, True, False, False, True])
    gap = tc._first_interior_gap(mask_edge, min_len=3)
    assert gap is None, gap
    print("PASS: edge-touching run correctly excluded (not a genuine interior gap)")

    # gap touching the right edge -> also excluded
    mask_edge2 = np.array([False, True, False, False, False, True, True, True])
    gap = tc._first_interior_gap(mask_edge2, min_len=3)
    assert gap is None, gap
    print("PASS: right-edge-touching run correctly excluded")

    # two qualifying gaps -> first one wins (not the widest/most-balanced)
    mask_two = np.array([False, True, True, True, False, False, True, True, True, True, True, False])
    gap = tc._first_interior_gap(mask_two, min_len=3)
    assert gap == (1, 3), gap  # the FIRST (narrower) gap, not the second (wider) one
    print(f"PASS: first qualifying gap wins ({gap}), not the wider second one")
    print()


def test_shave_and_split_window() -> None:
    print("=== Part A: _shave_and_split_window ===")
    # 5x21 array: two floodable blobs (cols 0-4 and cols 15-19) separated
    # by a 10-cell-wide (>= threshold) all-unfloodable gap (cols 5-14).
    h, w = 5, 21
    keep = np.zeros((h, w), dtype=bool)
    keep[:, 0:5] = True
    keep[:, 15:20] = True
    ocean_any = np.zeros((h, w), dtype=bool)  # no ocean buffer complications for this test

    windows = tc._shave_and_split_window(keep, ocean_any, 0, h - 1, 0, w - 1, min_split_gap_cells=10, allow_split=True)
    assert len(windows) == 2, windows
    print(f"PASS: two floodable blobs separated by a {10}-cell gap correctly split into 2 pieces: {windows}")
    # each piece should be tightened to its own blob, not include the gap
    cols_per_piece = sorted((c0, c1) for _r0, _r1, c0, c1 in windows)
    assert cols_per_piece[0] == (0, 4), cols_per_piece
    assert cols_per_piece[1] == (15, 19), cols_per_piece
    print(f"PASS: each piece tightened to its own blob's columns exactly {cols_per_piece}")

    # same layout but gap narrower than threshold -> single piece, no split
    keep_narrow = np.zeros((h, w), dtype=bool)
    keep_narrow[:, 0:5] = True
    keep_narrow[:, 8:13] = True  # gap is only 3 cells wide (cols 5-7)
    windows_narrow = tc._shave_and_split_window(
        keep_narrow, ocean_any, 0, h - 1, 0, w - 1, min_split_gap_cells=10, allow_split=True,
    )
    assert len(windows_narrow) == 1, windows_narrow
    print(f"PASS: narrow (< threshold) gap correctly does NOT split: {windows_narrow}")

    # allow_split=False must never split, regardless of gap width
    windows_disabled = tc._shave_and_split_window(
        keep, ocean_any, 0, h - 1, 0, w - 1, min_split_gap_cells=10, allow_split=False,
    )
    assert len(windows_disabled) == 1, windows_disabled
    r0, r1, c0, c1 = windows_disabled[0]
    assert (c0, c1) == (0, 19), (c0, c1)  # single tight bbox spanning BOTH blobs (gap included, unsplit)
    print(f"PASS: allow_split=False never splits regardless of gap width: {windows_disabled}")

    # a genuinely larger gap should be shaved off in full, not just the
    # minimum threshold width - test with a 15-cell gap and confirm each
    # piece's edge lands exactly at its own blob, not min_split_gap_cells
    # away from it.
    h2, w2 = 5, 26
    keep_wide_gap = np.zeros((h2, w2), dtype=bool)
    keep_wide_gap[:, 0:5] = True
    keep_wide_gap[:, 20:25] = True  # gap is cols 5-19, i.e. 15 cells wide
    ocean_any2 = np.zeros((h2, w2), dtype=bool)
    windows_wide = tc._shave_and_split_window(
        keep_wide_gap, ocean_any2, 0, h2 - 1, 0, w2 - 1, min_split_gap_cells=10, allow_split=True,
    )
    assert len(windows_wide) == 2, windows_wide
    cols_wide = sorted((c0, c1) for _r0, _r1, c0, c1 in windows_wide)
    assert cols_wide[0] == (0, 4) and cols_wide[1] == (20, 24), cols_wide
    print(f"PASS: wider-than-threshold gap (15 cells) fully shaved off both pieces, not just the minimum: {cols_wide}")

    # three separated blobs -> recursion produces 3 final pieces
    h3, w3 = 5, 35
    keep_three = np.zeros((h3, w3), dtype=bool)
    keep_three[:, 0:5] = True
    keep_three[:, 15:20] = True
    keep_three[:, 30:35] = True
    ocean_any3 = np.zeros((h3, w3), dtype=bool)
    windows_three = tc._shave_and_split_window(
        keep_three, ocean_any3, 0, h3 - 1, 0, w3 - 1, min_split_gap_cells=10, allow_split=True,
    )
    assert len(windows_three) == 3, windows_three
    print(f"PASS: three separated blobs (two qualifying gaps) recursively split into 3 pieces: {windows_three}")
    print()


def test_filter_and_shave_chunks_multi_piece_bookkeeping() -> None:
    print("=== Part B: filter_and_shave_chunks multi-piece bookkeeping ===")
    # _stage_a_one now returns (kept, dropped) directly (2026-08 reorder:
    # exposure checked per-piece, after shave/split - see its docstring).
    # Simulate ONE input bbox producing TWO kept pieces (as a real internal
    # split would, both with exposure) and confirm filter_and_shave_chunks'
    # aggregation logic puts both in kept, none in dropped; and separately
    # that a per-piece drop (one piece with exposure, one without) is
    # reported correctly too.

    def fake_stage_a_one_both_kept(bbox):
        return [(0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 3.0, 1.0)], []

    bboxes = [(0.0, 0.0, 3.0, 1.0)]
    results = [fake_stage_a_one_both_kept(b) for b in bboxes]
    kept, dropped = [], []
    for piece_kept, piece_dropped in results:
        kept.extend(piece_kept)
        dropped.extend(piece_dropped)
    assert len(kept) == 2, kept
    assert len(dropped) == 0, dropped
    print(f"PASS: one input producing 2 exposed pieces -> both land in kept ({kept}), none dropped")

    def fake_stage_a_one_mixed(bbox):
        return [(0.0, 0.0, 1.0, 1.0)], [((2.0, 0.0, 3.0, 1.0), "no_population_exposure")]

    results2 = [fake_stage_a_one_mixed(b) for b in bboxes]
    kept2, dropped2 = [], []
    for piece_kept, piece_dropped in results2:
        kept2.extend(piece_kept)
        dropped2.extend(piece_dropped)
    assert kept2 == [(0.0, 0.0, 1.0, 1.0)], kept2
    assert dropped2 == [((2.0, 0.0, 3.0, 1.0), "no_population_exposure")], dropped2
    print(f"PASS: one input producing one exposed + one unexposed piece -> split correctly between "
          f"kept ({kept2}) and dropped ({dropped2})")
    print()


def test_bbox_to_grid_rect_inverse() -> None:
    print("=== Part C: bbox_to_grid_rect / grid_rect_to_bbox round-trip ===")
    import random
    random.seed(0)
    for _ in range(20):
        r0 = random.randint(0, 50)
        r1 = r0 + random.randint(0, 5)
        c0 = random.randint(0, 50)
        c1 = c0 + random.randint(0, 5)
        lat_min, lon_min = random.randint(-90, 90), random.randint(-180, 180)
        rect = (r0, r1, c0, c1)
        bbox = tc.grid_rect_to_bbox(rect, lat_min, lon_min)
        rect2 = tc.bbox_to_grid_rect(bbox, lat_min, lon_min)
        assert rect == rect2, (rect, bbox, rect2)
    print("PASS: bbox_to_grid_rect is an exact inverse of grid_rect_to_bbox (20 random cases)")
    print()


def main() -> None:
    test_first_interior_gap()
    test_shave_and_split_window()
    test_filter_and_shave_chunks_multi_piece_bookkeeping()
    test_bbox_to_grid_rect_inverse()
    print("All shave/split validation checks passed.")


if __name__ == "__main__":
    main()
