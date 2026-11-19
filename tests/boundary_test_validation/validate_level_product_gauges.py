"""Synthetic checks of the gauge scores of boundary_test/level_product_gauges."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "boundary_test"))

from level_product_gauges import GAUGE_RPS, MIN_REGION_GAUGES, breakdown_groups, scores  # noqa: E402


def _gauges() -> pd.DataFrame:
    n_big = MIN_REGION_GAUGES
    res = pd.DataFrame({"region": ["big"] * n_big + ["small"] * 3,
                        "spring_range_m": np.r_[np.full(n_big, 0.5), [1.5, 3.0, 6.0]],
                        "station_class": "etc", "has_tc": False})
    for rp in GAUGE_RPS:
        res[f"gauge_rp{rp}_m"] = 1.0 + 0.1 * rp
        res[f"new_rp{rp}_m"] = res[f"gauge_rp{rp}_m"] + 0.2
    return res


def test_breakdown_scores_small_regions_as_other_and_every_range_class() -> None:
    res = _gauges()
    groups = breakdown_groups(res)
    assert {g for s, g in groups if s == "region"} == {"big", "other"}
    assert sum(groups[("region", "other")]) == 3
    ranges = [m for (s, _), m in groups.items() if s != "region"]
    assert len(ranges) == 4 and all(sum(m) >= 1 for m in ranges)
    out = scores(res, {"new": "new"}, groups)
    assert set(out.split) == {"region", "spring tidal range (m)"}
    assert np.allclose(out["median"], 0.2) and np.allclose(out.mae, 0.2)


def test_default_scores_keep_the_tc_groups() -> None:
    out = scores(_gauges(), {"new": "new"})
    assert "split" not in out and out.gauges.iloc[0] == "all"
    assert out.n[out.gauges == "TC-dominated gauges"].eq(0).all()
