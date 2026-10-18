"""Helpers shared by several analysis/ scripts.

Each had identical copies in two or three scripts (compute_exposure_analysis,
compute_flood_totals, compute_bangkok_case_exposure, compute_adaptation_pathways,
compute_tipping_points_summary); this is now the single definition.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import retry_transient_io  # noqa: E402


def _round_val(val: float) -> float | int:
    """Same magnitude-conditional rounding rule as validate_country.py's _round_row -
    whole integer, except abs(val) < 1 rounds to 2 decimals so a small real
    nonzero value doesn't silently read as "none"."""
    if val != val:  # NaN
        return val
    return round(val, 2) if abs(val) < 1 else int(round(val))


def _growth_matrix_path(exp_dir: Path, strategy: str, design: str | None) -> Path:
    label = "baseline" if strategy == "baseline" else f"{strategy}_{design}"
    return exp_dir / f"exposure_{label}_growth_matrix.csv"


def _growth_path(growth_df: pd.DataFrame, ssp: str, iso: str, years: np.ndarray, default: float = 1.0) -> np.ndarray:
    """Vectorized equivalent of population_growth.interpolate_growth_factor,
    called once per (ssp, iso) over the WHOLE `years` array instead of once
    per year. That function re-does the (ssp, iso) DataFrame lookup and
    re-sorts/re-extracts its own row EVERY call - fine for a handful of
    countries (plot_tipping_points.py/plot_extended_pathways.py's own use),
    but this script calls it for every member country of every continent,
    which multiplies the per-call overhead by orders of magnitude (confirmed
    live: the original per-year-call version was still running after 55
    minutes). Same clipping behaviour at the range edges as the original
    (np.interp's default is constant extrapolation, matching it exactly).
    """
    try:
        row = growth_df.loc[(ssp, iso)]
    except KeyError:
        return np.full(len(years), default)
    yrs = np.array(sorted(row.index.astype(int)))
    vals = row[yrs].to_numpy(dtype=float)
    return np.interp(years.astype(float), yrs.astype(float), vals)


def _eai_from_interp(interp, slr_path_mm: np.ndarray, growth_path: np.ndarray) -> np.ndarray:
    """Same as pathways.eai_along_pathway, but takes an ALREADY-BUILT
    interpolator instead of (re-)constructing a fresh RegularGridInterpolator
    on every call - eai_along_pathway's own per-call construction is cheap in
    isolation, but this script calls it (country/continent) x strategy x
    design x ssp x variant x percentile times; for a continent with ~50
    member countries that's tens of thousands of rebuilds of the SAME grid
    per (country, strategy, design) - confirmed live as a real secondary
    bottleneck (the _growth_path fix alone only brought this down to ~1 CPU-
    second per wall-clock minute of real progress, still far too slow for a
    6-continent sweep)."""
    pts = np.column_stack([np.asarray(growth_path, float), np.asarray(slr_path_mm, float)])
    return np.clip(interp(pts), 0.0, None)


def _read_band_and_transform(path: Path, dtype: str, fill) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Read band 1 (file nodata -> `fill`, cast to `dtype`) plus the file's transform.

    The file's OWN nodata is replaced via `masked=True` - population's sentinel (-3.4e38,
    float32-min) is finite, so a plain `src.read(1)` plus an `isfinite` cleanup would not catch
    it. Open + read are retried as ONE unit (`retry_transient_io`): a transient P:\ hiccup under
    concurrent load surfaces inside `.read()` (seen live 2026-10-09, `GetBlockRef failed`), not
    only in `rasterio.open`.
    """
    def _do():
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
            return arr.filled(fill).astype(dtype), src.transform
    return retry_transient_io(_do)


def _read_band(path: Path, dtype: str, fill) -> np.ndarray:
    """Read band 1 with the file's nodata replaced by `fill` (see `_read_band_and_transform`)."""
    return _read_band_and_transform(path, dtype, fill)[0]


def _first_crossing(eai: np.ndarray, threshold: float, years: np.ndarray) -> float | None:
    """First (fractional) year in `years` where `eai` reaches `threshold`, or None."""
    above = np.nonzero(eai >= threshold)[0]
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(years[0])
    e0, e1 = eai[k - 1], eai[k]
    return float(years[k - 1] + (threshold - e0) / (e1 - e0))


def _people(v: float, _=None) -> str:
    """Axis/label formatter for people counts: 1.2 M, 340 k, 12 (sign kept for negatives)."""
    a = abs(v)
    s = f"{a / 1e6:g} M" if a >= 1e6 else (f"{a / 1e3:g} k" if a >= 1e3 else f"{a:g}")
    return ("-" if v < 0 else "") + s
