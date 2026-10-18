"""Test tiles._clamp_window against per-source window overshoot of the mosaic.

Per-tile from_bounds rounding is independent of the mosaic's own, so on a
non-pixel-aligned bbox a window can overshoot by one pixel ((3525,233) vs (3525,232)).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tiles import _clamp_window  # noqa: E402


def test_clamp_window_reproduces_the_real_documented_overshoot() -> None:
    print("=== _clamp_window: the exact real documented overshoot (3525,233) into (3525,232) ===")
    h, w = _clamp_window(r0=0, c0=0, h=3525, w=233, out_height=3525, out_width=232)
    assert (h, w) == (3525, 232), (h, w)
    print("PASS: an oversized (3525,233) window clamps to (3525,232), fitting the mosaic exactly")
    print()


def test_clamp_window_no_op_when_window_already_fits() -> None:
    print("=== _clamp_window: no-op when the window already fits ===")
    h, w = _clamp_window(r0=10, c0=20, h=100, w=150, out_height=500, out_width=500)
    assert (h, w) == (100, 150), (h, w)
    print("PASS: a window that already fits is returned unchanged")
    print()


def test_clamp_window_accounts_for_nonzero_offset() -> None:
    print("=== _clamp_window: clamps relative to the window's own offset, not just raw size ===")
    h, w = _clamp_window(r0=480, c0=480, h=50, w=50, out_height=500, out_width=500)
    assert (h, w) == (20, 20), (h, w)
    print("PASS: a 50x50 window at offset (480,480) in a 500x500 mosaic clamps to (20,20)")
    print()


def test_clamp_window_offset_already_past_the_edge() -> None:
    """Callers skip h <= 0 or w <= 0 (_mosaic_mask_for_trim), so <= 0 is enough."""
    print("=== _clamp_window: offset already past the mosaic edge clamps to <= 0 ===")
    h, w = _clamp_window(r0=500, c0=0, h=10, w=10, out_height=500, out_width=500)
    assert h <= 0, h
    print(f"PASS: r0=500 in a 500-row mosaic clamps h to {h} (<=0, caller's own guard handles this)")
    print()


def main() -> None:
    test_clamp_window_reproduces_the_real_documented_overshoot()
    test_clamp_window_no_op_when_window_already_fits()
    test_clamp_window_accounts_for_nonzero_offset()
    test_clamp_window_offset_already_past_the_edge()
    print("All _clamp_window validation checks passed.")


if __name__ == "__main__":
    main()
