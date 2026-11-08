"""Step 2c: how storm-tide duration sets inland flood levels, on synthetic 1-D floodplains (SFINCS).

A strip of OCEAN_CELLS sea cells (bed -5 m), then land rising from the shore (MSL + SHORE_Z_M) with
slope S, cells DX_M wide, 3 cells across. Boundary level: hydrographs with the same peak PEAK_M
above MSL but different shapes:
  steady       PEAK_M held for STEADY_DAYS (the level the floodplain approaches, no attenuation)
  syn_A<a>_D<d> semidiurnal tide of amplitude a + Gaussian surge peaking on high water, surge time
               above half its peak d hours, total peak PEAK_M
  obs_<i>      observed GESLA annual-maximum events (`shapes_events.py` profiles), scaled to PEAK_M
SFINCS (regular grid, ASCII input, no subgrid) gives zsmax(x). Per run and land cell the
duration measures of the forcing at that cell's head (PEAK_M - zsmax(x)) are written beside it,
so the analysis can test which measure predicts the level loss.

Writes <out-dir>/duration_experiment/{runs.parquet, cells.parquet, hydrographs.npz}.

Usage:
    python boundary_test/duration_experiment.py --out-dir .cleanup_baseline/level_sources \
        --sfincs-exe <sfincs.exe> [--workers 3] [--n-observed 24]
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DX_M = 30.0  # the eikonal grid step of DeltaDTM
N_ACROSS = 3
OCEAN_CELLS = 10
LAND_KM = 15.0
SHORE_Z_M = 0.5  # shore crest above MSL, so tides alone do not keep the land wet
PEAK_M = 2.0
SLOPES = (0.0, 1e-4, 3e-4, 1e-3)
MANNING = (0.03, 0.06, 0.12)
TIDE_AMPLITUDES_M = (0.0, 0.5, 1.0, 1.5)
SURGE_DUR50_HR = (2.0, 6.0, 12.0, 24.0, 48.0)
M2_PERIOD_S = 44714.0
WINDOW_HR = 72
TAIL_HR = 24  # simulated after the window so the floodplain can drain back and zsmax settles
STEADY_DAYS = 10
DT_S = 600.0
WET_DEPTH_M = 0.05


def synthetic(amplitude: float, dur50_hr: float) -> np.ndarray:
    t = np.arange(-WINDOW_HR * 3600, WINDOW_HR * 3600 + 1, DT_S)
    tide = amplitude * np.cos(2 * np.pi * t / M2_PERIOD_S)
    sigma = dur50_hr * 3600 / (2 * np.sqrt(2 * np.log(2)))
    return tide + (PEAK_M - amplitude) * np.exp(-0.5 * (t / sigma) ** 2)


def observed(profiles: np.ndarray, n: int, events: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    """n GESLA events spread over the time-above-MSL distribution, 10-min, scaled to PEAK_M."""
    ok = (events.source == "gesla").to_numpy() & np.isfinite(profiles).all(axis=1)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(events.d_above_msl.to_numpy()[idx], kind="stable")]
    pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
    hours = np.arange(-WINDOW_HR, WINDOW_HR + 1)
    t = np.arange(-WINDOW_HR * 3600, WINDOW_HR * 3600 + 1, DT_S) / 3600.0
    return ([f"obs_{events.file_name.iloc[i]}_{int(events.year.iloc[i])}" for i in pick],
            [np.interp(t, hours, profiles[i]) * PEAK_M for i in pick])


def write_case(run_dir: Path, slope: float, manning: float, h: np.ndarray, tstop_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    m = OCEAN_CELLS + int(LAND_KM * 1000 / DX_M)
    x = (np.arange(m) - OCEAN_CELLS + 0.5) * DX_M
    z = np.where(x < 0, -5.0, SHORE_Z_M + slope * x)
    np.savetxt(run_dir / "sfincs.dep", np.tile(z, (N_ACROSS, 1)), fmt="%.4f")
    msk = np.ones((N_ACROSS, m), dtype=int)
    msk[:, 0] = 2
    np.savetxt(run_dir / "sfincs.msk", msk, fmt="%d")
    np.savetxt(run_dir / "sfincs.bnd", [[0.5 * DX_M, 0.5 * N_ACROSS * DX_M]], fmt="%.2f")
    t = np.arange(len(h)) * DT_S
    t_full = np.append(t, tstop_s)
    h_full = np.append(h, h[-1])
    np.savetxt(run_dir / "sfincs.bzs", np.column_stack([t_full, h_full]), fmt="%.1f %.4f")
    stop = (np.datetime64("2000-01-01T00:00:00") + np.timedelta64(int(tstop_s), "s")).astype(str)
    stop = stop.replace("-", "").replace("T", " ").replace(":", "")
    (run_dir / "sfincs.inp").write_text("\n".join([
        f"mmax = {m}", f"nmax = {N_ACROSS}", f"dx = {DX_M:g}", f"dy = {DX_M:g}", "x0 = 0", "y0 = 0", "rotation = 0",
        "tref = 20000101 000000", "tstart = 20000101 000000", f"tstop = {stop}", "inputformat = asc",
        "outputformat = asc", "depfile = sfincs.dep", "mskfile = sfincs.msk", "bndfile = sfincs.bnd",
        "bzsfile = sfincs.bzs", f"manning = {manning:g}", "dtmapout = 0", f"dtmaxout = {tstop_s + 1:.0f}",
        "alpha = 0.5", "huthresh = 0.01", "advection = 0", ""]))


def read_zsmax(run_dir: Path) -> np.ndarray:
    raw = np.fromfile(run_dir / "zsmax.dat", dtype=np.float32)
    values = raw[1:-1]  # one Fortran record: 4-byte length markers around the data
    m = len(values) // N_ACROSS
    return values.reshape(m, N_ACROSS).mean(axis=1)  # active points are stored across-strip first


def duration_measures(h: np.ndarray, head: np.ndarray) -> dict:
    """Per head (m below the peak): B's T, contiguous T around the peak, and the excess volume."""
    peak = h.max()
    k = int(np.argmax(h))
    thr = peak - head[:, None]
    above = h[None, :] >= thr
    t_cum = above.sum(axis=1) * DT_S / 3600.0
    t_run = np.empty(len(head))
    for j in range(len(head)):
        a = above[j]
        lo = k - np.argmin(a[k::-1]) + 1 if not a[k::-1].all() else 0
        hi = k + np.argmin(a[k:]) if not a[k:].all() else len(a)
        t_run[j] = (hi - lo) * DT_S / 3600.0
    excess = np.clip(h[None, :] - thr, 0.0, None).sum(axis=1) * DT_S / 3600.0
    return {"t_cum_hr": t_cum, "t_run_hr": t_run, "excess_m_hr": excess}


def _run(args) -> dict:
    exe, run_dir, name, slope, manning, h, tstop_s = args
    run_dir = Path(run_dir)
    if not (run_dir / "zsmax.dat").exists():
        write_case(run_dir, slope, manning, h, tstop_s)
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        subprocess.run([exe], cwd=run_dir, env=env, stdout=open(run_dir / "run.log", "w"), stderr=subprocess.STDOUT,
                       check=True, timeout=3 * 3600)
    zs = read_zsmax(run_dir)
    for f in ("sfincs.dep", "sfincs.msk"):
        (run_dir / f).unlink(missing_ok=True)
    return {"name": name, "slope": slope, "manning": manning, "zsmax": zs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sfincs-exe", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n-observed", type=int, default=24)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "duration_experiment"
    dest.mkdir(exist_ok=True)

    hydrographs = {"steady": np.full(int(STEADY_DAYS * 86400 / DT_S) + 1, PEAK_M)}
    for a in TIDE_AMPLITUDES_M:
        for d in SURGE_DUR50_HR:
            hydrographs[f"syn_A{a:g}_D{d:g}"] = synthetic(a, d)
    if (out / "events.parquet").exists():
        names, series = observed(np.load(out / "event_profiles.npz")["profiles"], args.n_observed,
                                 pd.read_parquet(out / "events.parquet"))
        hydrographs.update(zip(names, series))
    np.savez(dest / "hydrographs.npz", **hydrographs)

    tasks = []
    for name, h in hydrographs.items():
        tstop = len(h) * DT_S + (0 if name == "steady" else TAIL_HR * 3600)
        h_run = h if name == "steady" else np.append(h, np.full(int(TAIL_HR * 3600 / DT_S), h[-1]))
        for s in SLOPES:
            for n in MANNING:
                tasks.append((args.sfincs_exe, str(dest / "runs" / f"{name}__S{s:g}__n{n:g}"), name, s, n, h_run, tstop))
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(_run, tasks)):
            rows.append(r)
            if k % 25 == 0:
                print(f"  {k + 1}/{len(tasks)} runs", flush=True)

    m = OCEAN_CELLS + int(LAND_KM * 1000 / DX_M)
    x = (np.arange(m) - OCEAN_CELLS + 0.5) * DX_M
    land = x > 0
    cells, run_rows = [], []
    for r in rows:
        h = hydrographs[r["name"]]
        z = SHORE_Z_M + r["slope"] * x[land]
        zs = r["zsmax"][land]
        wet = zs - z > WET_DEPTH_M
        head = np.clip(PEAK_M - zs, 0.0, PEAK_M)
        meas = duration_measures(h, head)
        whole = duration_measures(h, np.array([PEAK_M - SHORE_Z_M]))
        extent = float(x[land][wet].max()) if wet.any() else 0.0
        run_rows.append({"name": r["name"], "slope": r["slope"], "manning": r["manning"], "extent_m": extent,
                         "t_above_shore_cum_hr": float(whole["t_cum_hr"][0]),
                         "t_above_shore_run_hr": float(whole["t_run_hr"][0]),
                         "excess_above_shore_m_hr": float(whole["excess_m_hr"][0])})
        cells.append(pd.DataFrame({"name": r["name"], "slope": r["slope"], "manning": r["manning"], "x_m": x[land],
                                   "ground_m": z, "zsmax_m": zs, "wet": wet, "head_m": head, **meas}))
    pd.DataFrame(run_rows).to_parquet(dest / "runs.parquet")
    pd.concat(cells).to_parquet(dest / "cells.parquet")
    print(pd.DataFrame(run_rows).pivot_table(index="name", columns=["slope"], values="extent_m", aggfunc="median")
          .round(0).head(30).to_string())


if __name__ == "__main__":
    main()
