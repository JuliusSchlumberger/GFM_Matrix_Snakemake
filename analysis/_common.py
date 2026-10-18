"""Helpers shared by several analysis/ scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import retry_transient_io  # noqa: E402


def _round_val(val: float) -> float | int:
    """Round to int, but keep 2 decimals for abs(val) < 1 so small values don't read as 0."""
    if val != val:
        return val
    return round(val, 2) if abs(val) < 1 else int(round(val))


def _growth_matrix_path(exp_dir: Path, strategy: str, design: str | None) -> Path:
    label = "baseline" if strategy == "baseline" else f"{strategy}_{design}"
    return exp_dir / f"exposure_{label}_growth_matrix.csv"


def _growth_path(growth_df: pd.DataFrame, ssp: str, iso: str, years: np.ndarray, default: float = 1.0) -> np.ndarray:
    """Vectorized population_growth.interpolate_growth_factor over all `years`.

    Per-year calls are orders of magnitude too slow for continent sweeps. Edges are
    clamped (np.interp constant extrapolation), matching the original.
    """
    try:
        row = growth_df.loc[(ssp, iso)]
    except KeyError:
        return np.full(len(years), default)
    yrs = np.array(sorted(row.index.astype(int)))
    vals = row[yrs].to_numpy(dtype=float)
    return np.interp(years.astype(float), yrs.astype(float), vals)


def _eai_from_interp(interp, slr_path_mm: np.ndarray, growth_path: np.ndarray) -> np.ndarray:
    """pathways.eai_along_pathway with a prebuilt interpolator (avoids per-call rebuilds)."""
    pts = np.column_stack([np.asarray(growth_path, float), np.asarray(slr_path_mm, float)])
    return np.clip(interp(pts), 0.0, None)


def _chunk_is_populated(chunks_dir: Path, cid: str) -> bool:
    # Zero-byte placeholders (ocean chunks) make rasterio.open raise; HPC callers
    # may pass an unfiltered list planned before postprocessing ran.
    p = chunks_dir / f"exposure_population_grid_{cid}.tif"
    return p.exists() and p.stat().st_size > 0


def discover_chunk_ids(flood_frac_dir: Path, chunks_dir: Path) -> tuple[list[str], list[str]]:
    """Return ``(populated_ids, all_ids)`` for the chunks with flood-fraction output.

    `all_ids` (sorted) are parsed from ``flood_fraction_<cid>_RP*.tif`` names;
    `populated_ids` drops chunks whose population grid is missing or a zero-byte
    placeholder (ocean/unpopulated). Both are returned so callers can tell "no
    flood fraction yet" from "nothing populated" with a single directory glob.
    """
    all_ids = sorted(set(
        p.stem.split("_RP")[0].replace("flood_fraction_", "")
        for p in flood_frac_dir.glob("flood_fraction_*.tif")
    ))
    return [cid for cid in all_ids if _chunk_is_populated(chunks_dir, cid)], all_ids


def _read_band_and_transform(path: Path, dtype: str, fill) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Read band 1 (file nodata -> `fill`, cast to `dtype`) plus the file's transform.

    `masked=True` is required: population's nodata (-3.4e38, float32-min) is finite.
    Open + read are retried as one unit since transient P:\ errors (`GetBlockRef failed`)
    surface inside `.read()`.
    """
    def _do():
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
            return arr.filled(fill).astype(dtype), src.transform
    return retry_transient_io(_do)


def _read_band(path: Path, dtype: str, fill) -> np.ndarray:
    return _read_band_and_transform(path, dtype, fill)[0]


def _first_crossing(eai: np.ndarray, threshold: float, years: np.ndarray) -> float | None:
    """First year (linearly interpolated) where `eai` reaches `threshold`, or None."""
    above = np.nonzero(eai >= threshold)[0]
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(years[0])
    e0, e1 = eai[k - 1], eai[k]
    return float(years[k - 1] + (threshold - e0) / (e1 - e0))


def _people(v: float, _=None) -> str:
    a = abs(v)
    s = f"{a / 1e6:g} M" if a >= 1e6 else (f"{a / 1e3:g} k" if a >= 1e3 else f"{a:g}")
    return ("-" if v < 0 else "") + s
