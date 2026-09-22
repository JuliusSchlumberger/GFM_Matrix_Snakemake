"""Direct unit tests for two real, documented 2026-08 production incidents
in src/tile_chunking.py's Stage 7 (exposure filter + shave), neither of
which had a test before this:

  - `_has_exposure`: for a handful of bboxes, `get_rasterdataset(...,
    bbox=...)`'s clip silently fell back to the FULL global WorldPop
    raster (43200x18720, ~3GB float32) instead of a small windowed read -
    this crashed a real ~50min production run with an ArrayMemoryError.
    Fixed with two guards (oversized-bounds check before materializing,
    and a try/except around the actual read) - on EITHER guard tripping,
    the chunk is KEPT (never silently dropped), per this codebase's
    "uncertain means keep, not drop" principle.
  - `_stage_a_one`: a single transient raster-read failure (a momentary
    file-lock that outlasted 3 retries over 15s) crashed an entire
    ~78-minute global run. Fixed by keeping the chunk UNSHAVED AND
    UNSPLIT on any `_shave_chunk` failure, skipping the exposure check
    entirely, rather than dropping it or propagating the crash.

Both are exactly the "silently drop real coastline" failure mode this
task cares about - a regression in either fix would remove a real
simulation domain from the final tile manifest without any error.

Usage:
    python validate_exposure_guards.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tile_chunking as tc  # noqa: E402


class _FakeRasterWrapper:
    """Mimics the object get_rasterdataset returns: `.raster.bounds`,
    `.raster.nodata`, `.values`."""
    def __init__(self, values: np.ndarray, bounds: tuple, nodata=None):
        self._values = values
        self._bounds = bounds
        self._nodata = nodata

    @property
    def values(self):
        return self._values

    class _RasterAccessor:
        def __init__(self, outer):
            self._outer = outer

        @property
        def bounds(self):
            return self._outer._bounds

        @property
        def nodata(self):
            return self._outer._nodata

    @property
    def raster(self):
        return self._RasterAccessor(self)


def test_has_exposure_keeps_real_population() -> None:
    print("=== _has_exposure: bbox with real population -> True (kept) ===")
    bbox = (10.0, 50.0, 10.1, 50.1)
    da = _FakeRasterWrapper(np.array([[0.0, 5.0], [0.0, 0.0]]), bounds=bbox, nodata=-9999.0)
    catalog = MagicMock()
    catalog.get_rasterdataset.return_value = da
    result = tc._has_exposure(bbox, catalog, "population")
    assert result is True, result
    print("PASS: real positive population -> kept")
    print()


def test_has_exposure_drops_zero_population() -> None:
    print("=== _has_exposure: bbox with zero real population -> False (dropped) ===")
    bbox = (10.0, 50.0, 10.1, 50.1)
    da = _FakeRasterWrapper(np.array([[0.0, 0.0], [0.0, 0.0]]), bounds=bbox, nodata=-9999.0)
    catalog = MagicMock()
    catalog.get_rasterdataset.return_value = da
    result = tc._has_exposure(bbox, catalog, "population")
    assert result is False, result
    print("PASS: genuinely zero population, correctly-sized read -> dropped")
    print()


def test_has_exposure_kept_when_clip_returns_oversized_raster() -> None:
    """The exact documented bug: get_rasterdataset's clip silently returns
    something far larger than the requested bbox (here: ~40x larger, well
    over the 20x guard threshold) - must be kept, never materialized."""
    print("=== _has_exposure: oversized clip result (the real 2026-08 bug) -> True (kept), no materialize ===")
    bbox = (10.0, 50.0, 10.1, 50.1)  # 0.1 x 0.1 deg requested
    huge_bounds = (-180.0, -90.0, 180.0, 90.0)  # the whole globe - what the real bug returned

    class _BoomOnMaterialize(_FakeRasterWrapper):
        # If the oversized-bounds guard fails to trip, the code falls
        # through to `.values` - make that raise, so a false "kept anyway
        # by accident" is caught as a test failure (materializing a real
        # 3GB array here would both be wrong AND slow the test down for real).
        @property
        def values(self):
            raise MemoryError("would-be 3GB materialize - the guard should prevent this from ever being called")

    da = _BoomOnMaterialize(np.zeros((1, 1)), bounds=huge_bounds, nodata=-9999.0)
    catalog = MagicMock()
    catalog.get_rasterdataset.return_value = da

    result = tc._has_exposure(bbox, catalog, "population")
    assert result is True, result
    print("PASS: oversized clip result detected before materializing - chunk kept, no MemoryError")
    print()


def test_has_exposure_kept_on_read_exception() -> None:
    print("=== _has_exposure: get_rasterdataset itself raises -> True (kept), not propagated ===")
    bbox = (10.0, 50.0, 10.1, 50.1)
    catalog = MagicMock()
    catalog.get_rasterdataset.side_effect = RuntimeError("transient I/O failure")
    result = tc._has_exposure(bbox, catalog, "population")
    assert result is True, result
    print("PASS: read exception caught, chunk kept rather than crashing the whole run")
    print()


def test_stage_a_one_kept_unshaved_on_shave_failure() -> None:
    """The real 2026-08 incident: a transient _shave_chunk read failure
    must leave the chunk UNCHANGED (original bbox, single piece, never
    split) and skip the exposure check entirely - not crash, not drop."""
    print("=== _stage_a_one: _shave_chunk failure -> kept unshaved/unsplit, exposure check skipped ===")
    bbox = (10.0, 50.0, 10.1, 50.1)
    with patch.object(tc, "_shave_chunk", side_effect=RuntimeError("transient file-lock")), \
         patch.object(tc, "_has_exposure") as mock_has_exposure:
        kept, dropped = tc._stage_a_one(bbox)
    assert kept == [bbox], kept
    assert dropped == [], dropped
    mock_has_exposure.assert_not_called()
    print(f"PASS: kept=[{bbox}] unchanged, dropped=[], _has_exposure never called (skipped entirely)")
    print()


def test_stage_a_one_normal_path_still_applies_exposure_per_piece() -> None:
    """Sanity check the normal (non-failure) path still works: shave
    produces 2 pieces, each independently exposure-checked - one kept,
    one dropped."""
    print("=== _stage_a_one: normal path applies exposure independently per shaved piece ===")
    bbox = (10.0, 50.0, 10.2, 50.1)
    piece_a = (10.0, 50.0, 10.1, 50.1)
    piece_b = (10.1, 50.0, 10.2, 50.1)
    with patch.object(tc, "_shave_chunk", return_value=[piece_a, piece_b]), \
         patch.object(tc, "_has_exposure", side_effect=lambda p, *_: p == piece_a):
        kept, dropped = tc._stage_a_one(bbox)
    assert kept == [piece_a], kept
    assert dropped == [(piece_b, "no_population_exposure")], dropped
    print(f"PASS: kept={kept}, dropped={dropped} - each piece checked independently")
    print()


def main() -> None:
    test_has_exposure_keeps_real_population()
    test_has_exposure_drops_zero_population()
    test_has_exposure_kept_when_clip_returns_oversized_raster()
    test_has_exposure_kept_on_read_exception()
    test_stage_a_one_kept_unshaved_on_shave_failure()
    test_stage_a_one_normal_path_still_applies_exposure_per_piece()
    print("All exposure-guard validation checks passed.")


if __name__ == "__main__":
    main()
