"""Illustrative continuation of the SSP population projections beyond their last year (2100).

The IIASA-WiC SSP projections (population_growth.load_ssp_growth_factors) end in 2100; the AR6
sea-level trajectories run to 2150. To follow the scenario-neutral growth matrices past 2100,
each country's SSP growth factor is continued from its 2100 value along an assumed variant
(config population_extension.variants). These are assumptions, not projections:

    constant   hold the 2100 level
    trend      continue the country's own average annual rate over the last `window` years,
               decaying with a half-life of `half_life` years (0 / omitted = no decay)
    rate       a fixed annual rate `annual` (e.g. -0.005 = -0.5 %/yr) from the last year on

Usage:
    ext = extend_growth_factors(growth_df, {"mode": "trend", "window": 25, "half_life": 50}, to_year=2150)
    factor = interpolate_growth_factor(ext, "SSP2", "NGA", 2130)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _continuation(f_end: np.ndarray, f_ref: np.ndarray, window: float, dt: np.ndarray, variant: dict) -> np.ndarray:
    """Growth factors (rows = countries, cols = dt years after the last year) for one variant."""
    mode = variant.get("mode", "constant")
    if mode == "constant":
        return np.repeat(f_end[:, None], len(dt), axis=1)
    if mode == "rate":
        return f_end[:, None] * (1.0 + float(variant["annual"])) ** dt[None, :]
    if mode == "trend":
        with np.errstate(divide="ignore", invalid="ignore"):
            r0 = np.where((f_end > 0) & (f_ref > 0), np.log(f_end / f_ref) / window, 0.0)  # per year (log)
        h = float(variant.get("half_life") or 0.0)
        # integral of r0 * 0.5**(t/h) from 0 to dt  (= r0 * dt without decay)
        cum = dt if h <= 0 else h / np.log(2.0) * (1.0 - 0.5 ** (dt / h))
        return f_end[:, None] * np.exp(r0[:, None] * cum[None, :])
    raise ValueError(f"unknown population_extension mode {mode!r} (constant / trend / rate)")


def extend_growth_factors(growth_df: pd.DataFrame, variant: dict, to_year: int = 2150,
                          step: int = 5) -> pd.DataFrame:
    """Return `growth_df` (MultiIndex (scenario, ISO), integer year columns) with columns
    added every `step` years from its last year up to `to_year`, continued per `variant`."""
    years = sorted(int(c) for c in growth_df.columns)
    last = years[-1]
    window = int(variant.get("window", 25))
    ref_year = max((y for y in years if y <= last - window), default=years[0])
    new_years = np.arange(last + step, to_year + 1, step)
    if len(new_years) == 0:
        return growth_df.copy()
    f_end = growth_df[last].to_numpy(float)
    f_ref = growth_df[ref_year].to_numpy(float)
    ext = _continuation(f_end, f_ref, float(last - ref_year), (new_years - last).astype(float), variant)
    out = pd.concat([growth_df, pd.DataFrame(ext, index=growth_df.index, columns=new_years.tolist())], axis=1)
    out.columns = [int(c) for c in out.columns]
    return out
