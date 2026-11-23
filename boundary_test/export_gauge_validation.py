"""Copy the gauge-validation tables of INPUT_ANALYSIS.md §8.7 into input_analysis_data/gauge_validation/.

Reads the --dest-dir outputs of the §10 step 24 scripts: level_product_gauges.py (--levels-dir),
open_guesses.py ingredients (--ingredients-dir, its ingredients/ folder), review_diagnostics.py
(--review-dir, one folder per stage) and its shift_bootstrap stage on production's shift-fit
gauges (--bootstrap-dir); and production's levels, tide gauges and COAST-RP_TC shifts
(--product-dir). The stations' tide factor with all gauges within 25 km is recomputed here from
production's tide gauges, beside production's through-water factor. Writes CSV only; read them
with keep_default_na=False ("NA" is the North Atlantic basin).

Usage:
    python boundary_test/export_gauge_validation.py --levels-dir <dir> --ingredients-dir <dir>
        --review-dir <dir> --bootstrap-dir <dir> --product-dir <storm_tide_product_dir>
        [--dest boundary_test/input_analysis_data/gauge_validation]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import tide_correction as tdc  # noqa: E402

DEFAULT_DEST = REPO / "boundary_test" / "input_analysis_data" / "gauge_validation"

# exported name -> (source folder argument, path in it)
COPIES = {
    "levels_by_gauge.csv": ("levels_dir", "gauge_check.parquet"),
    "levels_summary.csv": ("levels_dir", "gauge_check_summary.csv"),
    "levels_breakdown.csv": ("levels_dir", "gauge_check_breakdown.csv"),
    "ingredients_scores.csv": ("ingredients_dir", "scores.csv"),
    "tc_raise_cv_scores.csv": ("review_dir", "basin_fit_cv/scores.csv"),
    "tc_raise_cv_fits.csv": ("review_dir", "basin_fit_cv/fits.csv"),
    "tc_shift_bootstrap.csv": ("bootstrap_dir", "shift_bootstrap/shifts.csv"),
    "tc_mask_et_scores.csv": ("review_dir", "tc_mask_et/scores.csv"),
    "tc_mask_et_maxima.csv": ("review_dir", "tc_mask_et/maxima_loss.csv"),
    "tide_neighbours_amplitude_scores.csv": ("review_dir", "tide_neighbours/amplitude_scores.csv"),
    "tide_neighbours_level_scores.csv": ("review_dir", "tide_neighbours/level_scores.csv"),
    "r_by_rp.csv": ("review_dir", "gauge_series/r_by_rp.csv"),
    "r_fallback_scores.csv": ("review_dir", "gauge_series/r_fallback_scores.csv"),
    "surge_tide_dependence.csv": ("review_dir", "gauge_series/tide_dependence.csv"),
    "seasonal_gap.csv": ("review_dir", "gauge_series/seasonal_gap.csv"),
    "despike_near_tracks.csv": ("review_dir", "despike_tracks/stations.parquet"),
    "tc_shifts_production.csv": ("product_dir", "tc_shifts.csv"),
}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.reset_index() if df.index.name else df
    return pd.read_csv(path, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("levels-dir", "ingredients-dir", "review-dir", "bootstrap-dir", "product-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    sources = {name: Path(getattr(args, arg)) / rel for name, (arg, rel) in COPIES.items()}
    product, dest = Path(args.product_dir), Path(args.dest)
    tide_gauges = product / "gauges" / "tide_gauges.parquet"
    missing = [str(p) for p in [*sources.values(), product / "levels.parquet", tide_gauges] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n  " + "\n  ".join(missing))
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        _read(src).to_csv(dest / name, index=False, float_format="%.6g")
    lev = pd.read_parquet(product / "levels.parquet", columns=["station_id", "lon", "lat", "atlas", "tide_factor",
                                                               "tide_gauges"])
    gt = pd.read_parquet(tide_gauges)
    lonlat, g_lonlat = lev[["lon", "lat"]].to_numpy(dtype=float), gt[["lon", "lat"]].to_numpy(dtype=float)
    within_25km = np.ones(len(lev))
    for atlas in ("fes2022", "fes2014"):
        sel = (lev.atlas == atlas).to_numpy()
        within_25km[sel] = tdc.correction_factors(lonlat[sel], g_lonlat, gt[f"log_ratio_{atlas}"].to_numpy())[0]
    lev.rename(columns={"tide_factor": "tide_factor_through_water"}).assign(tide_factor_25km=within_25km) \
        .reset_index().to_csv(dest / "tide_factor_stations.csv", index=False, float_format="%.6g")
    print(f"wrote {len(sources) + 1} tables to {dest}")


if __name__ == "__main__":
    main()
