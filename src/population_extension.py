"""Continue SSP growth factors past 2100 (AR6 SLR runs to 2150) along assumed variants.

Assumptions, not projections:
    constant   hold the 2100 level
    trend      continue the average annual rate over the last `window` years,
               decaying with half-life `half_life` years (0 / omitted = no decay)
    rate       fixed annual rate `annual` (e.g. -0.005 = -0.5 %/yr)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _continuation(f_end: np.ndarray, f_ref: np.ndarray, window: float, dt: np.ndarray, variant: dict) -> np.ndarray:
    mode = variant.get("mode", "constant")
    if mode == "constant":
        return np.repeat(f_end[:, None], len(dt), axis=1)
    if mode == "rate":
        return f_end[:, None] * (1.0 + float(variant["annual"])) ** dt[None, :]
    if mode == "trend":
        with np.errstate(divide="ignore", invalid="ignore"):
            r0 = np.where((f_end > 0) & (f_ref > 0), np.log(f_end / f_ref) / window, 0.0)  # log rate per year
        h = float(variant.get("half_life") or 0.0)
        # integral of r0 * 0.5**(t/h) from 0 to dt  (= r0 * dt without decay)
        cum = dt if h <= 0 else h / np.log(2.0) * (1.0 - 0.5 ** (dt / h))
        return f_end[:, None] * np.exp(r0[:, None] * cum[None, :])
    raise ValueError(f"unknown population_extension mode {mode!r} (constant / trend / rate)")


def extend_growth_factors(growth_df: pd.DataFrame, variant: dict, to_year: int = 2150,
                          step: int = 5) -> pd.DataFrame:
    """Append year columns every `step` years up to `to_year`, continued per `variant`."""
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
