"""Step 2d: compare forms of correction B on the synthetic strips with fsf refitted per form.

`duration_analysis.py` holds fsf at the tile-calibrated 7.5 / 12, which favours whichever form
happens to match that level. Here each form gets its own fsf (grid search), and the forms are
compared on what fsf cannot fix: the spread of the eikonal / SFINCS extent ratio across
hydrographs and its dependence on hydrograph duration. With a two-sided multiplier a constant
t_ref is absorbed into fsf, so only per-event or per-head references and the clip are tested.
Forms (multiplier m = clip((t_ref / T)^alpha, lo, 10), T = `time_within_peak` at the cell head):
  no_B             m = 1
  tref12_clip      t_ref 12 h, lo 1 (production)
  tshore_clip      t_ref = the hydrograph's own T at the shore crest (longest exposure), lo 1
  tcalib_clip      t_ref = T(head) of the median COAST-HG spring hydrograph, lo 1
  tcalib_2s        as tcalib_clip, lo 0.1
  invT_2s          t_ref constant, lo 0.1 (m proportional to 1 / T^alpha)
each B form at alpha 1 and 0.5. Runs on flat strips (SFINCS floods the whole domain) and runs
where SFINCS reaches the domain end are left out of the scores.

Writes <out-dir>/duration_experiment/{refit_grid.parquet, refit_summary.csv}.

Usage:
    python boundary_test/duration_refit.py --out-dir .cleanup_baseline/level_sources --workers 6
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
import duration_analysis as da  # noqa: E402
import duration_experiment as de  # noqa: E402
from event_volume import time_within_peak  # noqa: E402

FSF_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)
FORMS = (("tref12_clip", 1.0), ("tshore_clip", 1.0), ("tcalib_clip", 1.0), ("tcalib_2s", 0.1), ("invT_2s", 0.1))
ALPHAS = (1.0, 0.5)
SATURATED_M = 14900.0
_H_CALIB = None


def _multiplier(first, z, h, form, lo, alpha):
    wet = np.isfinite(first) & (z > -1.0)
    head = np.where(wet, first - z, 0.0)[wet]
    t = np.maximum(time_within_peak(h, de.DT_S, head), de.DT_S)
    if form == "tshore_clip":
        ref = time_within_peak(h, de.DT_S, np.array([de.PEAK_M - de.SHORE_Z_M]))[0]
    elif form.startswith("tcalib"):
        ref = np.maximum(time_within_peak(_H_CALIB, de.DT_S, head), de.DT_S)
    else:
        ref = da.T_REF_S
    m = np.ones(z.shape)
    m[wet] = np.clip((ref / t) ** alpha, lo, da.MAX_MULTIPLIER)
    return m


def _run_task(args):
    name, slope, manning, h, h_calib = args
    global _H_CALIB
    _H_CALIB = h_calib
    x_all, _, transform = da.strip()
    land = x_all > 0
    z_all = np.where(x_all < 0, -5.0, de.SHORE_Z_M + slope * x_all)
    z = np.tile(z_all, (de.N_ACROSS, 1))
    base = np.full(z.shape, manning / 100.0)
    rows = []
    for fsf in FSF_GRID:
        first = da.eikonal_level(z, base * fsf, transform)
        rows.append({"form": "no_B", "alpha": np.nan, "fsf": fsf, "extent_m": _extent(x_all, land, z_all, first)})
        for form, lo in FORMS:
            for alpha in ALPHAS:
                lev = da.eikonal_level(z, base * fsf * _multiplier(first, z, h, form, lo, alpha), transform)
                rows.append({"form": form, "alpha": alpha, "fsf": fsf, "extent_m": _extent(x_all, land, z_all, lev)})
    for r in rows:
        r.update(name=name, slope=slope, manning=manning)
    return rows


def _extent(x_all, land, z_all, level):
    lev = level[1, land]
    wet = np.isfinite(lev) & (lev - z_all[land] > de.WET_DEPTH_M)
    return float(x_all[land][wet].max()) if wet.any() else 0.0


def summarise(grid: pd.DataFrame, sfincs: pd.DataFrame) -> pd.DataFrame:
    g = grid.merge(sfincs, on=["name", "slope", "manning"])
    g = g[(g.slope > 0) & (g.sfincs_extent_m < SATURATED_M)]
    g["log_ratio"] = np.log(np.maximum(g.extent_m, de.DX_M) / g.sfincs_extent_m)
    rows = []
    for (form, alpha), f in g.groupby(["form", g.alpha.fillna(0)]):
        rmse = f.groupby("fsf").log_ratio.apply(lambda v: np.sqrt(np.mean(v ** 2)))
        best = f[f.fsf == rmse.idxmin()]
        within = [np.std(b.log_ratio) for _, b in best.groupby(["slope", "manning"])]
        rho = [spearmanr(b.dur_hr, b.log_ratio)[0] for _, b in best.groupby(["slope", "manning"])]
        rows.append({"form": form, "alpha": alpha or np.nan, "best_fsf": rmse.idxmin(), "log_rmse": rmse.min(),
                     "median_ratio": np.exp(best.log_ratio.median()), "within_terrain_spread": np.median(within),
                     "rho_duration": np.median(rho),
                     "ratio_dur_lt6h": np.exp(best[best.dur_hr < 6].log_ratio.median()),
                     "ratio_dur_6_24h": np.exp(best[best.dur_hr.between(6, 24)].log_ratio.median()),
                     "ratio_dur_gt24h": np.exp(best[best.dur_hr > 24].log_ratio.median()), "n_runs": len(best)})
    return pd.DataFrame(rows).sort_values("log_rmse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "duration_experiment"
    runs = pd.read_parquet(dest / "runs.parquet")
    runs = runs[runs.slope > 0]
    hydrographs = dict(np.load(dest / "hydrographs.npz"))
    h_calib = da.calibration_hydrograph()
    tasks = [(r.name, r.slope, r.manning, hydrographs[r.name], h_calib) for r in runs.itertuples()]
    with ProcessPoolExecutor(args.workers) as pool:
        grid = pd.DataFrame([row for rows in pool.map(_run_task, tasks, chunksize=4) for row in rows])
    grid.to_parquet(dest / "refit_grid.parquet")
    sfincs = runs[["name", "slope", "manning", "extent_m", "t_above_shore_run_hr"]].rename(
        columns={"extent_m": "sfincs_extent_m", "t_above_shore_run_hr": "dur_hr"})
    summary = summarise(grid, sfincs)
    summary.to_csv(dest / "refit_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
