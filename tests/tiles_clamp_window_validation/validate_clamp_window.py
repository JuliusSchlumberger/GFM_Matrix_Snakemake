"""Direct unit test for tiles.py's `_clamp_window` - a real, confirmed
off-by-one mosaic bug: `from_bounds(...).round_lengths().round_offsets()`
rounds each source tile's window independently of the overall output
array's own rounding, so on a bbox whose edges aren't aligned to a whole
number of pixels, individual per-source-tile windows can together overshoot
the shared mosaic array by a pixel.

Hit live: `ValueError: could not broadcast input array from shape
(3525,233) into shape (3525,232)`, once Stage 3e started allowing oversized
merges through to `classify_mosaic` instead of rejecting them outright.

Usage:
    python validate_clamp_window.py
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
    print(f"PASS: an oversized (3525,233) window clamps to (3525,232), fitting the mosaic exactly")
    print()


def test_clamp_window_no_op_when_window_already_fits() -> None:
    print("=== _clamp_window: no-op when the window already fits ===")
    h, w = _clamp_window(r0=10, c0=20, h=100, w=150, out_height=500, out_width=500)
    assert (h, w) == (100, 150), (h, w)
    print("PASS: a window that already fits is returned unchanged")
    print()


def test_clamp_window_accounts_for_nonzero_offset() -> None:
    """The clamp must subtract the window's own (r0, c0) offset from the
    output bounds, not just compare (h, w) against (out_height, out_width)
    directly - a window starting near the mosaic's own far edge overshoots
    at a SMALLER (h, w) than one starting at the origin would."""
    print("=== _clamp_window: clamps relative to the window's own offset, not just raw size ===")
    # A 50x50 window starting at (480, 480) in a 500x500 mosaic only has
    # room for 20x20, even though 50x50 alone would easily fit an origin-anchored window.
    h, w = _clamp_window(r0=480, c0=480, h=50, w=50, out_height=500, out_width=500)
    assert (h, w) == (20, 20), (h, w)
    print(f"PASS: a 50x50 window at offset (480,480) in a 500x500 mosaic clamps to (20,20)")
    print()


def test_clamp_window_offset_already_past_the_edge() -> None:
    """A window whose own offset is already >= the mosaic bound clamps to
    zero (or negative-turned-into-a-non-positive value) - callers already
    guard `if h <= 0 or w <= 0: continue` (see _mosaic_mask_for_trim), so
    this just confirms the clamp itself doesn't crash or return something
    misleadingly positive in that case."""
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
