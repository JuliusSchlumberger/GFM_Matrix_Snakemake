"""Validate build_sfincs_tile.py's antimeridian-crossing error classifier
(`_classify_water_level_create_error`) - turns hydromt_sfincs's own raw
GEOS `TopologyException`/"side location conflict" (from a shapely
union_all() over a buffered search geometry that straddles +-180 deg
longitude) into a clear, actionable `RuntimeError` naming the tile, instead
of letting a raw GEOS traceback propagate.

Real, confirmed failures this guards against: tiles 1852, 2029, 2077 (all
near +-180 deg longitude) hit exactly this failure live during the first
258-tile HPC test batch. Without this classifier, a batch run's failure log
would show an opaque GEOS error instead of "this tile is antimeridian-
crossing, drop it" - the only signal telling a human (or a batch script's
own `--skip-tile-ids` list) which tiles are permanently unforceable versus
which hit a real, fixable bug.

This test does NOT reproduce a real antimeridian geometry through the full
hydromt_sfincs pipeline (expensive, and hydromt's own internal error text
isn't a stable contract to build a synthetic case against) - it directly
validates the classifier's own string-matching contract against realistic
exception messages/types, which is the actual logic this repo depends on.

Usage:
    python validate_antimeridian_handling.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_sfincs_tile import _classify_water_level_create_error  # noqa: E402


class _FakeTopologyException(Exception):
    """Stand-in for shapely's real TopologyException - only the class NAME
    matters to the classifier (it matches on str(e), not isinstance), which
    itself is worth locking down as a test: if hydromt/shapely's error text
    ever changes, this test - not a live antimeridian tile failure - is
    where that would be caught."""


def test_classifies_topology_exception_by_name() -> None:
    print("=== _classify_water_level_create_error: matches on 'TopologyException' in the message ===")
    e = _FakeTopologyException("TopologyException: side location conflict at 179.998 65.432")
    result = _classify_water_level_create_error(e, tile_id="2029")
    assert result is not None, "expected a wrapped RuntimeError, got None (fell through to re-raise original)"
    assert isinstance(result, RuntimeError)
    assert "2029" in str(result)
    assert "antimeridian" in str(result).lower()
    assert "side location conflict at 179.998 65.432" in str(result), "original error text should be preserved"
    print(f"PASS: wrapped as RuntimeError naming the tile: {result}")
    print()


def test_classifies_side_location_conflict_without_topology_exception_name() -> None:
    """The real GEOS message sometimes surfaces as a bare RuntimeError/
    ValueError whose text includes 'side location conflict' without the
    class name 'TopologyException' literally in the string - the
    classifier matches on EITHER substring, not just one."""
    print("=== _classify_water_level_create_error: matches on 'side location conflict' alone ===")
    e = ValueError("GEOS error: side location conflict")
    result = _classify_water_level_create_error(e, tile_id="2077")
    assert result is not None
    assert isinstance(result, RuntimeError)
    assert "2077" in str(result)
    print(f"PASS: wrapped even without the literal 'TopologyException' substring: {result}")
    print()


def test_unrelated_error_is_not_classified() -> None:
    """A genuinely different failure (e.g. a real missing-file bug) must
    NOT be silently reclassified as antimeridian - the caller re-raises the
    original exception unchanged when this returns None, so masking a real
    bug as antimeridian would hide it from a batch's own failure triage."""
    print("=== _classify_water_level_create_error: leaves unrelated errors alone ===")
    e = FileNotFoundError("matched_boundary_points.gpkg not found")
    result = _classify_water_level_create_error(e, tile_id="1573")
    assert result is None, f"unrelated error was wrongly classified as antimeridian: {result}"
    print("PASS: unrelated error returns None (caller re-raises it unchanged)")
    print()


def main() -> None:
    test_classifies_topology_exception_by_name()
    test_classifies_side_location_conflict_without_topology_exception_name()
    test_unrelated_error_is_not_classified()
    print("All antimeridian error-classification checks passed.")


if __name__ == "__main__":
    main()
