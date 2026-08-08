"""Validate `compute_run_order` (src/tile_chunking.py, 2026-08) - the
hop-distance-from-ocean BFS, provenance-set union-find grouping, and
tie-break ordering added to bake simulation run order into `tile_id` at
tile-generation time (replacing the retired `merge_dry_chunks` - see
tile_chunking.py's module docstring). Companion to
tests/tile_shave_split_validation/validate_shave_split.py.

Pure graph-logic unit tests - no raster I/O. `_classify_wet` (wet/dry) and
`_bbox_ocean_fraction` (wave-0 ordering signal) are the only two functions
`compute_run_order` calls that touch real raster data, so both are
monkeypatched per-test to a fixed lookup table built from hand-designed
bbox layouts - this exercises the REAL adjacency-graph/BFS/union-find/
sort-key code in `compute_run_order` itself, just fed synthetic wet/dry
and ocean-fraction inputs instead of reading DeltaDTM tiles.

Bbox layouts are deliberately built as 1D chains (or a chain sharing one
wide box's two distinct edges) rather than 2x2 grids - `_bboxes_touch_or_
overlap`'s simple range-overlap check treats a shared CORNER as touching
too (an intentional 8-connectivity choice elsewhere in this module), which
would silently add unwanted edges to any layout with two boxes diagonal to
a shared corner.

Usage:
    python validate_run_order.py
"""

import sys
from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import tile_chunking as tc  # noqa: E402

_EMPTY_SEEDS = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def _patch_wet_and_fraction(wet_map: dict[int, bool], frac_map: dict[int, float] | None = None) -> None:
    """Monkeypatch the two raster-touching helpers `compute_run_order`
    calls, with a fixed per-index lookup - no test in this file reads a
    real mask.
    """
    frac_map = frac_map or {}

    def _fake_classify_wet(grid_map, mask_index, ocean_code, max_workers=None):
        return {i: wet_map.get(i, False) for i in grid_map}

    def _fake_ocean_fraction(bbox, mask_index, ocean_code, coarse_resolution_m):
        # identify by position in the call - tests pass bboxes in index
        # order, so the caller's own index isn't available here; instead
        # look the bbox up by value against the test's own bbox list via a
        # closure-captured index map set right before the call (see below).
        return frac_map.get(_fake_ocean_fraction.bbox_to_index.get(bbox, -1), 0.0)

    _fake_ocean_fraction.bbox_to_index = {}
    tc._classify_wet = _fake_classify_wet
    tc._bbox_ocean_fraction = _fake_ocean_fraction


def _run(bboxes, wet_map, frac_map=None, river_mouth_seeds=None):
    _patch_wet_and_fraction(wet_map, frac_map)
    tc._bbox_ocean_fraction.bbox_to_index = {b: i for i, b in enumerate(bboxes)}
    seeds = river_mouth_seeds if river_mouth_seeds is not None else _EMPTY_SEEDS
    return tc.compute_run_order(bboxes, mask_index={}, ocean_code=1, river_mouth_seeds=seeds, coarse_resolution_m=500)


def test_hop_distance_and_unreachable() -> None:
    print("=== hop-distance BFS + unreachable detection ===")
    # O(wet) - H1 - H2, plus a fully isolated 4th box with no adjacency.
    bboxes = [
        (0, 0, 1, 1),      # 0: O, wet, hop 0
        (0, 1, 1, 2),      # 1: H1, touches O, hop 1
        (0, 2, 1, 3),      # 2: H2, touches H1, hop 2
        (100, 100, 101, 101),  # 3: isolated, unreachable
    ]
    wet_map = {0: True}
    order, unreachable, diag = _run(bboxes, wet_map)
    assert diag["hop_distance"] == [0, 1, 2, -1], diag["hop_distance"]
    assert unreachable == [3], unreachable
    assert order == [0, 1, 2], order
    print(f"PASS: hop_distance={diag['hop_distance']}, unreachable={unreachable}, order={order}")
    print()


def test_provenance_grouping_bridge() -> None:
    print("=== provenance-set grouping (union-find) - bridging chunk merges two groups ===")
    # O1(wet) - H1 - H3 - H2 - O2(wet): H3 touches both H1 (descended from
    # O1) and H2 (descended from O2), so H3 must merge O1's and O2's
    # otherwise-separate hinterland groups into one.
    bboxes = [
        (0, 0, 1, 1),  # 0: O1, wet
        (0, 1, 1, 2),  # 1: H1, touches O1
        (0, 2, 1, 3),  # 2: H3, touches H1 and H2 (the bridge)
        (0, 3, 1, 4),  # 3: H2, touches H3 and O2
        (0, 4, 1, 5),  # 4: O2, wet
    ]
    wet_map = {0: True, 4: True}
    order, unreachable, diag = _run(bboxes, wet_map)
    assert unreachable == [], unreachable
    assert diag["hop_distance"] == [0, 1, 2, 1, 0], diag["hop_distance"]
    g = diag["group_id"]
    assert g[1] == g[2] == g[3], g  # H1, H3, H2 all merged into one group via the bridge
    assert g[0] == -1 and g[4] == -1, g  # wave-0 rows carry the sentinel, not a real group id
    print(f"PASS: hop_distance={diag['hop_distance']}, group_id={g} - H1/H3/H2 share one group")
    print()


def test_wave0_sorts_by_ocean_fraction() -> None:
    print("=== wave-0 chunks order by descending ocean fraction ===")
    bboxes = [
        (0, 0, 1, 1),      # 0: low ocean fraction
        (10, 10, 11, 11),  # 1: high ocean fraction, far away (no adjacency either way)
    ]
    wet_map = {0: True, 1: True}
    frac_map = {0: 0.3, 1: 0.8}
    order, unreachable, diag = _run(bboxes, wet_map, frac_map)
    assert unreachable == [], unreachable
    assert order == [1, 0], order  # higher ocean fraction (idx 1) first
    print(f"PASS: order={order} (idx 1, ocean_fraction=0.8, before idx 0, ocean_fraction=0.3)")
    print()


def test_river_mouth_priority_tiebreak() -> None:
    print("=== hinterland tie-break: river-mouth-seed chunk first (equal degree/group) ===")
    # O wide enough to touch RM on its right edge and NRM on (half of) its
    # top edge, without RM and NRM touching each other - both hop 1, same
    # group (both descend from O), same degree (1) - river-mouth flag is
    # the only differentiator.
    O = (0, 0, 2, 1)
    RM = (2, 0, 3, 1)
    NRM = (0, 1, 1, 2)
    bboxes = [O, RM, NRM]
    wet_map = {0: True}
    seeds = gpd.GeoDataFrame({"geometry": [tc.box(*RM)]}, geometry="geometry", crs="EPSG:4326")
    order, unreachable, diag = _run(bboxes, wet_map, river_mouth_seeds=seeds)
    assert unreachable == [], unreachable
    assert diag["degree"][1] == diag["degree"][2] == 1, diag["degree"]
    assert diag["group_id"][1] == diag["group_id"][2], diag["group_id"]
    # only RM/NRM's flags matter here (both hop>=1, tie-break candidates) -
    # O itself may also intersect the seed box at their shared edge
    # (shapely `.intersects()` counts touching boundaries), which is
    # harmless since wave-0 rows never consult `is_river_mouth_seed` for
    # their own ordering (they sort purely by ocean_fraction).
    assert diag["is_river_mouth_seed"][1] is True, diag["is_river_mouth_seed"]
    assert diag["is_river_mouth_seed"][2] is False, diag["is_river_mouth_seed"]
    assert order.index(1) < order.index(2), order  # RM (idx 1) before NRM (idx 2)
    print(f"PASS: order={order} - river-mouth chunk (idx 1) precedes non-river-mouth (idx 2)")
    print()


def test_degree_tiebreak() -> None:
    print("=== hinterland tie-break: higher static degree first (equal group, no river mouth) ===")
    # O wide enough to touch A (right edge) and B (half of top edge)
    # without A/B touching each other; D touches B only, giving B degree 2
    # vs A's degree 1 - both hop 1, same group, neither is a river mouth.
    O = (0, 0, 2, 1)
    A = (2, 0, 3, 1)
    B = (0, 1, 1, 2)
    D = (0, 2, 1, 3)
    bboxes = [O, A, B, D]
    wet_map = {0: True}
    order, unreachable, diag = _run(bboxes, wet_map)
    assert unreachable == [], unreachable
    assert diag["hop_distance"] == [0, 1, 1, 2], diag["hop_distance"]
    assert diag["degree"][1] == 1, diag["degree"]  # A: only touches O
    assert diag["degree"][2] == 2, diag["degree"]  # B: touches O and D
    assert diag["group_id"][1] == diag["group_id"][2], diag["group_id"]
    assert order.index(2) < order.index(1), order  # B (degree 2) before A (degree 1)
    print(f"PASS: order={order} - higher-degree chunk (idx 2, degree 2) precedes idx 1 (degree 1)")
    print()


def main() -> None:
    test_hop_distance_and_unreachable()
    test_provenance_grouping_bridge()
    test_wave0_sorts_by_ocean_fraction()
    test_river_mouth_priority_tiebreak()
    test_degree_tiebreak()
    print("All run-order validation checks passed.")


if __name__ == "__main__":
    main()
