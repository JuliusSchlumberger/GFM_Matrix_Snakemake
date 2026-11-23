"""Test the calibration tile selection's antimeridian flag."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
from select_calibration_tiles import touches_antimeridian  # noqa: E402


def test_antimeridian_flag_catches_tiles_stored_past_180_and_wrapped_ones() -> None:
    bounds = pd.DataFrame({"minx": [170.0, 178.3, -180.0, -179.0, -179.5],
                           "maxx": [172.0, 180.2, -178.0, 179.0, -178.0]})
    assert touches_antimeridian(bounds).tolist() == [False, True, True, True, False]
