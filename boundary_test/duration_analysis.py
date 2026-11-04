"""Step 2c: eikonal vs SFINCS on the synthetic floodplains of `duration_experiment.py`.

The production eikonal solver (`flood_model.flood_depth_dense`, calibration settings) runs on the
same strips with friction = Manning n / 100 per metre (the land-cover rule, docs/methods_02 §2.2)
and the coastline cells seeded at PEAK_M. Variants:
  no_B         fsf 12, no duration correction (the optimum without it)
  B_tref12     fsf 7.5, B with t_ref = 12 h (production)
  B_tref_shore fsf 7.5, B with t_ref = the time the hydrograph stays above the shore crest
               (the longest exposure any land cell can have)
  B_tref_calib fsf 7.5, B with t_ref = the same T(head) of the median calibration hydrograph
               (COAST-HG spring tide, scaled to PEAK_M), so the correction is relative to the
               durations the friction was calibrated on
B multiplier: clip((t_ref / T)^alpha, 1, 10), T = `event_volume.time_within_peak` at the cell's
head over its spill level (the ground on these rising strips), alpha 1.
Scores per run: flood extent (m) and the RMSE of the max water level over cells wet in either.

Writes <out-dir>/duration_experiment/{eikonal_runs.parquet, *.png}.

Usage:
    python boundary_test/duration_analysis.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio.transform  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
import duration_experiment as de  # noqa: E402
from event_volume import time_within_peak  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402

COAST_HG = Path("P:/11212688-004-global-floodmaps/modelling/inputs/COAST_HG/COAST-HG_RP100.nc")
SOLVER = dict(ocean_code=1, river_code=3, obstacle_coupling=True, max_outer_iterations=4,
              outer_convergence_pct=0.01, max_rounds=40, waterlevel_epsilon_m=0.03)
FSF_NO_B, FSF_B = 12.0, 7.5
T_REF_S = 12 * 3600.0
ALPHA = 1.0
MAX_MULTIPLIER = 10.0
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COLORS = {"SFINCS": "#0b0b0b", "no_B": "#2a78d6", "B_tref12": "#eb6834", "B_tref_shore": "#1baf7a",
          "B_tref_calib": "#6250d6"}


def strip() -> tuple[np.ndarray, np.ndarray, object]:
    m = de.OCEAN_CELLS + int(de.LAND_KM * 1000 / de.DX_M)
    x = (np.arange(m) - de.OCEAN_CELLS + 0.5) * de.DX_M
    transform = rasterio.transform.from_origin(0.0, de.N_ACROSS * de.DX_M, de.DX_M, de.DX_M)
    return x, np.tile(x, (de.N_ACROSS, 1)), transform


def calibration_hydrograph() -> np.ndarray:
    """Median-shaped COAST-HG spring-tide hydrograph (10 min), scaled to peak PEAK_M."""
    with netCDF4.Dataset(COAST_HG) as ds:
        h = np.ma.filled(ds["hydrograph_spring_tide_signal"][::10], np.nan)
    h = h / np.nanmax(h, axis=1, keepdims=True)
    return np.nanmedian(h, axis=0) * de.PEAK_M


def eikonal_level(z: np.ndarray, friction: np.ndarray, transform) -> np.ndarray:
    mask = np.where(z < -1.0, 1, 0).astype(np.int8)
    coast = np.zeros(z.shape, dtype=bool)
    coast[:, de.OCEAN_CELLS - 1] = True
    rows, cols = np.nonzero(coast)
    depth, _ = flood_depth_dense(z, mask, friction.astype(np.float32), transform, seed_rows=rows, seed_cols=cols,
                                 seed_values=np.full(rows.size, de.PEAK_M), resolution=de.DX_M, **SOLVER)
    return np.where(depth > 0, z + depth, np.nan)


def b_multiplier(level: np.ndarray, z: np.ndarray, h: np.ndarray, t_ref_s) -> np.ndarray:
    wet = np.isfinite(level) & (z > -1.0)
    head = np.where(wet, level - z, 0.0)
    t = np.maximum(time_within_peak(h, de.DT_S, head[wet]), de.DT_S)
    ref = t_ref_s(head[wet]) if callable(t_ref_s) else t_ref_s
    mult = np.ones(z.shape)
    mult[wet] = np.clip((ref / t) ** ALPHA, 1.0, MAX_MULTIPLIER)
    return mult


def solve_variants(z: np.ndarray, n: float, h: np.ndarray, h_calib: np.ndarray, transform) -> dict:
    base = np.full(z.shape, n / 100.0)
    out = {"no_B": eikonal_level(z, base * FSF_NO_B, transform)}
    first = eikonal_level(z, base * FSF_B, transform)
    t_shore = time_within_peak(h, de.DT_S, np.array([de.PEAK_M - de.SHORE_Z_M]))[0]
    refs = {"B_tref12": T_REF_S, "B_tref_shore": t_shore,
            "B_tref_calib": lambda head: np.maximum(time_within_peak(h_calib, de.DT_S, head), de.DT_S)}
    for name, ref in refs.items():
        out[name] = eikonal_level(z, base * FSF_B * b_multiplier(first, z, h, ref), transform)
    return out


def score(x: np.ndarray, z_land: np.ndarray, level: np.ndarray, zs: np.ndarray) -> dict:
    wet_e = np.isfinite(level) & (level - z_land > de.WET_DEPTH_M)
    wet_s = zs - z_land > de.WET_DEPTH_M
    both = wet_e | wet_s
    e = np.where(wet_e, level, z_land)
    s = np.where(wet_s, zs, z_land)
    return {"extent_m": float(x[wet_e].max()) if wet_e.any() else 0.0,
            "level_rmse_m": float(np.sqrt(np.mean((e[both] - s[both]) ** 2))) if both.any() else 0.0}


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "duration_experiment"
    runs = pd.read_parquet(dest / "runs.parquet")
    cells = pd.read_parquet(dest / "cells.parquet")
    hydrographs = dict(np.load(dest / "hydrographs.npz"))
    h_calib = calibration_hydrograph()
    x_all, x2d, transform = strip()
    land = x_all > 0
    rows, profiles = [], {}
    for r in runs.itertuples():
        z_all = np.where(x_all < 0, -5.0, de.SHORE_Z_M + r.slope * x_all)
        z = np.tile(z_all, (de.N_ACROSS, 1))
        levels = solve_variants(z, r.manning, hydrographs[r.name], h_calib, transform)
        c = cells[(cells.name == r.name) & (cells.slope == r.slope) & (cells.manning == r.manning)]
        zs = c.zsmax_m.to_numpy()
        rows.append({"name": r.name, "slope": r.slope, "manning": r.manning, "variant": "SFINCS",
                     "extent_m": r.extent_m, "level_rmse_m": 0.0, "t_above_shore_run_hr": r.t_above_shore_run_hr,
                     "t_above_shore_cum_hr": r.t_above_shore_cum_hr})
        for v, lev in levels.items():
            lev_land = lev[1, land]
            rows.append({"name": r.name, "slope": r.slope, "manning": r.manning, "variant": v,
                         **score(x_all[land], z_all[land], lev_land, zs), "t_above_shore_run_hr": r.t_above_shore_run_hr,
                         "t_above_shore_cum_hr": r.t_above_shore_cum_hr})
            profiles[(r.name, r.slope, r.manning, v)] = lev_land
        profiles[(r.name, r.slope, r.manning, "SFINCS")] = zs
    table = pd.DataFrame(rows)
    table.to_parquet(dest / "eikonal_runs.parquet")
    summary = table[table.variant != "SFINCS"].groupby(["variant", "slope"]).level_rmse_m.median().unstack().round(3)
    print("median max-level RMSE vs SFINCS (m), by slope\n" + summary.to_string())
    ext = table.pivot_table(index=["name", "slope", "manning"], columns="variant", values="extent_m")
    for v in ("no_B", "B_tref12", "B_tref_shore", "B_tref_calib"):
        ext[f"{v}_ratio"] = ext[v] / ext["SFINCS"].where(ext["SFINCS"] > 0)
    print("\nmedian extent ratio eikonal / SFINCS by slope\n"
          + ext.groupby(level="slope")[[c for c in ext.columns if c.endswith("_ratio")]].median().round(3).to_string())
    ext.reset_index().to_csv(dest / "extent_table.csv", index=False)

    picks = [n for n in ("steady", "syn_A1_D6", "syn_A0_D48", "syn_A1.5_D2") if n in hydrographs]
    fig, axes = plt.subplots(len(picks), len(de.SLOPES), figsize=(16, 3.2 * len(picks)), squeeze=False)
    for i, name in enumerate(picks):
        for j, s in enumerate(de.SLOPES):
            ax = axes[i, j]
            for v, color in COLORS.items():
                y = profiles.get((name, s, 0.06, v))
                if y is not None:
                    ax.plot(x_all[land] / 1000, y, color=color, linewidth=2 if v == "SFINCS" else 1.3, label=v)
            ax.plot(x_all[land] / 1000, de.SHORE_Z_M + s * x_all[land], color=MUTED, linewidth=1, linestyle=":")
            ax.set(ylim=(0, de.PEAK_M + 0.2), title=f"{name}, slope {s:g}, n 0.06", xlabel="km inland" if i == len(picks) - 1 else None)
            ax.title.set_fontsize(9)
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Max water level: SFINCS (black) vs eikonal variants; dotted = ground", color=INK)
    fig.tight_layout()
    fig.savefig(dest / "profiles.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
