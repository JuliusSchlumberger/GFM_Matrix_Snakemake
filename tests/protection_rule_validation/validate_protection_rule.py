"""Test the water-level protection rule against the flood-fraction rule on synthetic cells.

Two cells share one station: a coastal cell flooded by the design event and an inland
cell the design event never reaches (design flood fraction 0) that floods only under SLR.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exposure_analysis import (  # noqa: E402
    compute_avoid,
    design_return_period_grid,
    protect_exposure_grid,
    water_level_exceedance,
)

# Storm tide + MDT at one station (m); SLR adds on top.
STORM_TIDE = {2: 1.0, 100: 2.0, 1000: 2.5}
SLR_M = {"SLR_0": 0.0, "SLR_500": 0.5, "SLR_1500": 1.5}
LEVELS = {(rp, slr): np.array([STORM_TIDE[rp] + dz]) for rp in STORM_TIDE for slr, dz in SLR_M.items()}
KEYS = list(LEVELS)


def _exceeded(design_rp: int, design_slr: str = "SLR_0") -> dict:
    station_idx = np.zeros((1, 2), dtype="int64")
    return water_level_exceedance(LEVELS, station_idx, np.full((1, 2), design_rp), design_slr, KEYS)


def test_design_rp_grid_applies_minimum_and_missing_geogunits() -> None:
    geo = np.array([[0, 1, -1, 7]])
    out = design_return_period_grid(geo, {0: 100, 1: 1}, minimum_rp=2)
    assert out.tolist() == [[100, 2, 2, 2]], out


def test_event_at_or_below_design_level_is_protected() -> None:
    exc = _exceeded(design_rp=100)
    assert not exc[(2, "SLR_0")].any()
    assert not exc[(100, "SLR_0")].any()
    assert exc[(1000, "SLR_0")].all()
    assert not exc[(2, "SLR_500")].any()      # 1.5 m <= 2.0 m design
    assert exc[(100, "SLR_500")].all()        # 2.5 m > 2.0 m
    assert exc[(2, "SLR_1500")].all()         # 2.5 m > 2.0 m: SLR has eroded the standard


def test_inland_cell_keeps_protection_under_water_level_rule() -> None:
    pop = np.array([[100.0, 100.0]])
    design_ff = np.array([[0.4, 0.0]])        # design event never reaches the inland cell
    ff_slr = np.array([[0.4, 0.3]])           # RP2 at +0.5 m wets the inland cell
    ff_rule = protect_exposure_grid(ff_slr, design_ff, pop)
    wl_rule = protect_exposure_grid(ff_slr, design_ff, pop, _exceeded(100)[(2, "SLR_500")])
    assert ff_rule.tolist() == [[0.0, 30.0]], ff_rule
    assert wl_rule.tolist() == [[0.0, 0.0]], wl_rule


def test_water_level_rule_exposes_overtopped_cell_with_unchanged_extent() -> None:
    # Steep cell: RP1000 floods the same share as the design event. The flood-fraction
    # rule calls it protected; the water level overtops the defence.
    pop = np.array([[100.0, 100.0]])
    design_ff = np.array([[0.4, 0.4]])
    ff_rp1000 = np.array([[0.4, 0.4]])
    assert protect_exposure_grid(ff_rp1000, design_ff, pop).sum() == 0.0
    wl_rule = protect_exposure_grid(ff_rp1000, design_ff, pop, _exceeded(100)[(1000, "SLR_0")])
    assert wl_rule.tolist() == [[40.0, 40.0]], wl_rule


def test_avoid_existing_population_uses_water_level_exceedance() -> None:
    pop = np.array([[100.0, 100.0]])
    apf = np.array([[0.4, 0.0]])
    ff = np.array([[0.4, 0.3]])
    exceeded = _exceeded(100)[(2, "SLR_500")]
    out = compute_avoid(ff, apf, pop, np.zeros_like(pop), 1.0, existing_prot_frac=apf, existing_exceeded=exceeded)
    assert out.tolist() == [[0.0, 0.0]], out
