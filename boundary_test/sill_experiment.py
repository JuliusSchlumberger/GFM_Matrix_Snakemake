"""Test 11: how far may a boundary hydrograph deviate before flood extent and depth change (SFINCS)?

1-D strips as in `duration_experiment.py`: OCEAN_CELLS sea cells (bed -5 m), a sill of SILL_CELLS
cells with its crest HEAD_FRACTIONS x peak below the peak, then a hinterland at MSL: a closed
basin of 2 or 5 km, or an open slope rising SLOPE inland over SLOPE_KM. Levels relative to MSL.

Part A, controlled deviations. Per tidal class (gauge spring range) and RP in A_RPS the base
hydrograph is the one gauge event (GESLA annual maximum of the RP window, `event_profiles.npz`)
whose T(0.25), T(0.5) and nb24 lie closest to the class medians of the gauges' events (a median
profile over sites or events is narrower than any event), scaled to the class median COAST-RP
level. Deviations,
one at a time: the whole hydrograph stretched in time around the peak by STRETCH (every duration,
T(0.25) and T(0.5) included, scales by the factor; not a physical tide, a probe of duration
error); a level offset of OFFSET x peak away from the peak (0 at the peak, full after ~12 h;
changes nb24 and the levels between high waters).

Part B, real deviations. Per tidal class and RP in TARGET_RPS, B_SITES random quality sites: the
generator's hydrograph (`generator_profiles.npz`, generator_smooth) and up to B_EVENTS of the
gauge's own events of that RP window, all scaled to the site's COAST-RP level so only the shape
differs. The spread of flooding over the real events is the yardstick for the generator.

Every hydrograph is followed by a 6-h fall to MSL and 18 h at MSL so the strip can settle.
Per run: extent (wet distance behind the sill), flooded volume per metre width, maximum and mean
depth; per hydrograph its T(0.25), T(0.5) and nb24 (`generator_validation.profile_measures`).

Writes <out-dir>/sill_experiment/{hydrographs_<part>.parquet, hydrographs_<part>.npz,
runs_<part>.parquet, runs/}.

Usage:
    python boundary_test/sill_experiment.py --out-dir .cleanup_baseline/level_sources \
        --sfincs-exe <sfincs.exe> [--part A|B|both] [--workers 3] [--trial N]
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from duration_experiment import DT_S, DX_M, N_ACROSS, OCEAN_CELLS, read_zsmax  # noqa: E402
from generator_validation import RANGE_BINS, TARGET_RPS, hourly_profile, profile_measures  # noqa: E402
from storm_tide_hydrograph import STEPS_PER_HR, rp_event_selection  # noqa: E402

SILL_CELLS = 3
HEAD_FRACTIONS = (0.1, 0.25, 0.5)
BASINS_KM = (2.0, 5.0)
SLOPE, SLOPE_KM = 2e-4, 20.0
A_MANNING = (0.03, 0.06)
B_MANNING = (0.03,)
B_TERRAINS = ("basin5", "slope")
A_RPS = (2, 10, 100)
STRETCH = (0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3)
OFFSET = (-0.1, -0.05, 0.05, 0.1)
OFFSET_RAMP_HR = 6.0
WINDOW_HR = 72
FALL_HR, SETTLE_HR = 6, 18
B_SITES = 6
B_EVENTS = 6
WET_DEPTH_M = 0.05
SEED = 0


def terrain(name: str, crest_m: float) -> np.ndarray:
    """Bed level per cell along the strip: sea, sill, hinterland."""
    if name.startswith("basin"):
        land = np.zeros(int(float(name[5:]) * 1000 / DX_M))
    else:
        land = SLOPE * (np.arange(int(SLOPE_KM * 1000 / DX_M)) + 0.5) * DX_M
    return np.r_[np.full(OCEAN_CELLS, -5.0), np.full(SILL_CELLS, crest_m), land]


def with_settling(h: np.ndarray) -> np.ndarray:
    fall = np.linspace(h[-1], 0.0, FALL_HR * STEPS_PER_HR + 1)[1:]
    return np.r_[h, fall, np.zeros(SETTLE_HR * STEPS_PER_HR)]


def write_case(run_dir: Path, z: np.ndarray, manning: float, h: np.ndarray) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    m = len(z)
    np.savetxt(run_dir / "sfincs.dep", np.tile(z, (N_ACROSS, 1)), fmt="%.4f")
    msk = np.ones((N_ACROSS, m), dtype=int)
    msk[:, 0] = 2
    np.savetxt(run_dir / "sfincs.msk", msk, fmt="%d")
    np.savetxt(run_dir / "sfincs.bnd", [[0.5 * DX_M, 0.5 * N_ACROSS * DX_M]], fmt="%.2f")
    t = np.arange(len(h)) * DT_S
    np.savetxt(run_dir / "sfincs.bzs", np.column_stack([t, h]), fmt="%.1f %.4f")
    tstop_s = t[-1]
    stop = (np.datetime64("2000-01-01T00:00:00") + np.timedelta64(int(tstop_s), "s")).astype(str)
    stop = stop.replace("-", "").replace("T", " ").replace(":", "")
    (run_dir / "sfincs.inp").write_text("\n".join([
        f"mmax = {m}", f"nmax = {N_ACROSS}", f"dx = {DX_M:g}", f"dy = {DX_M:g}", "x0 = 0", "y0 = 0", "rotation = 0",
        "tref = 20000101 000000", "tstart = 20000101 000000", f"tstop = {stop}", "inputformat = asc",
        "outputformat = asc", "depfile = sfincs.dep", "mskfile = sfincs.msk", "bndfile = sfincs.bnd",
        "bzsfile = sfincs.bzs", f"manning = {manning:g}", "dtmapout = 0", f"dtmaxout = {tstop_s + 1:.0f}",
        "alpha = 0.5", "huthresh = 0.01", "advection = 0", ""]))


def flood_measures(z: np.ndarray, zsmax: np.ndarray) -> dict:
    land = slice(OCEAN_CELLS + SILL_CELLS, None)
    depth = np.clip(zsmax[land] - z[land], 0.0, None)
    wet = depth > WET_DEPTH_M
    return {"extent_m": float((np.flatnonzero(wet).max() + 1) * DX_M) if wet.any() else 0.0,
            "volume_m2": float(depth[wet].sum() * DX_M), "max_depth_m": float(depth.max()),
            "mean_depth_m": float(depth[wet].mean()) if wet.any() else 0.0}


def _run(task: dict) -> dict:
    run_dir = Path(task["run_dir"])
    z = terrain(task["terrain"], task["crest_m"])
    if not (run_dir / "zsmax.dat").exists():
        write_case(run_dir, z, task["manning"], task["h"])
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        with open(run_dir / "run.log", "w") as log:
            subprocess.run([task["exe"]], cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=3 * 3600)
        for f in ("sfincs.dep", "sfincs.msk", "sfincs.bzs"):
            (run_dir / f).unlink(missing_ok=True)
    meas = flood_measures(z, read_zsmax(run_dir))
    return {k: v for k, v in task.items() if k not in ("h", "exe", "run_dir")} | meas


def _filled(p: np.ndarray) -> np.ndarray:
    hours = np.arange(len(p)) - len(p) // 2
    ok = np.isfinite(p)
    return np.interp(hours, hours[ok], p[ok])


def _to_10min(p: np.ndarray, stretch: float = 1.0) -> np.ndarray:
    hours = np.arange(-WINDOW_HR, WINDOW_HR + 1)
    t_hr = np.arange(-WINDOW_HR * STEPS_PER_HR, WINDOW_HR * STEPS_PER_HR + 1) / STEPS_PER_HR
    return np.interp(t_hr / stretch, hours, p)


def gauge_events(out: Path, sites: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Complete GESLA annual-maximum events at the quality sites with their profile measures."""
    ev = pd.read_parquet(out / "events.parquet").reset_index(drop=True)
    prof = np.load(out / "event_profiles.npz")["profiles"]
    inner = np.abs(np.arange(prof.shape[1]) - prof.shape[1] // 2) <= 48
    keep = (ev.source == "gesla").to_numpy() & ev.file_name.isin(sites.index).to_numpy()
    keep &= np.isfinite(prof[:, inner]).all(axis=1)
    ev = ev[keep].copy()
    ev["range_class"] = ev.file_name.map(sites.range_class)
    meas = pd.DataFrame([profile_measures(p) for p in prof[ev.index]], index=ev.index)
    return pd.concat([ev, meas], axis=1), prof


def part_a(ev: pd.DataFrame, prof: np.ndarray, sites: pd.DataFrame, coast: pd.DataFrame) -> dict[str, dict]:
    t_hr = np.arange(-WINDOW_HR * STEPS_PER_HR, WINDOW_HR * STEPS_PER_HR + 1) / STEPS_PER_HR
    ramp = 1 - np.exp(-(t_hr / OFFSET_RAMP_HR) ** 2)
    cols = ["t025_hr", "t05_hr", "nb24"]
    hydro = {}
    for rc in sites.range_class.cat.categories:
        for rp in A_RPS:
            e = ev[ev.range_class == rc]
            e = e.iloc[rp_event_selection(e.rp_empirical_yr.to_numpy(), rp)]
            target = e.groupby("file_name")[cols].median().median()
            pick = (e[cols] / target - 1).abs().sum(axis=1).idxmin()
            base = _filled(prof[pick].astype(float))
            peak = float(coast.loc[coast.index.intersection(e.file_name.unique()), f"coast_rp_rp{rp}_m"].median())
            case = f"c{rc.left:g}-{rc.right:g}_rp{rp}"
            meta = {"part": "A", "range_class": str(rc), "rp": rp, "peak_m": peak,
                    "site": ev.file_name[pick], "event_year": ev.year[pick],
                    **{f"class_{c}": target[c] for c in cols}}
            for st in (1.0, *STRETCH):
                hydro[f"A_{case}_stretch{st:g}"] = {**meta, "kind": "stretch", "value": st,
                                                    "h": peak * _to_10min(base, st)}
            h0 = hydro[f"A_{case}_stretch1"]["h"]
            for d in OFFSET:
                hydro[f"A_{case}_offset{d:+g}"] = {**meta, "kind": "offset", "value": d, "h": h0 + d * peak * ramp}
    return hydro


def part_b(out: Path, ev: pd.DataFrame, prof: np.ndarray, sites: pd.DataFrame,
           coast: pd.DataFrame) -> dict[str, dict]:
    gp = np.load(out / "analysis" / "generator_profiles.npz")
    gen = pd.DataFrame({"file_name": gp["file_name"], "rp": gp["rp"], "variant": gp["variant"]})
    gen = gen[(gen.variant == "generator_smooth") & gen.file_name.isin(coast.index)]
    rng = np.random.default_rng(SEED)
    hydro = {}
    for rc in sites.range_class.cat.categories:
        for rp in TARGET_RPS:
            cand = []
            for f in np.sort(gen.file_name[gen.rp == rp].unique()):
                if sites.range_class.get(f) != rc or not np.isfinite(coast.loc[f, f"coast_rp_rp{rp}_m"]):
                    continue
                e = ev[ev.file_name == f]
                sel = e.index[rp_event_selection(e.rp_empirical_yr.to_numpy(), rp)]
                if len(sel):
                    cand.append((f, sel))
            for i in rng.choice(len(cand), min(B_SITES, len(cand)), replace=False):
                f, sel = cand[i]
                peak = float(coast.loc[f, f"coast_rp_rp{rp}_m"])
                meta = {"part": "B", "range_class": str(rc), "rp": rp, "peak_m": peak, "site": f}
                k = gen.index[(gen.file_name == f) & (gen.rp == rp)][0]
                hydro[f"B_{f}_rp{rp}_generator"] = {**meta, "kind": "generator",
                                                    "h": peak * _to_10min(_filled(gp["profile"][k].astype(float)))}
                for j in rng.choice(sel, min(B_EVENTS, len(sel)), replace=False):
                    hydro[f"B_{f}_rp{rp}_gauge{int(ev.year[j])}"] = {
                        **meta, "kind": "gauge", "event_year": ev.year[j],
                        "h": peak * _to_10min(_filled(prof[j].astype(float)))}
    return hydro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sfincs-exe", required=True)
    parser.add_argument("--part", choices=("A", "B", "both"), default="both")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--trial", type=int, default=None, help="run only the first N runs (timing)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "sill_experiment"
    dest.mkdir(exist_ok=True)
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    sites = sites[sites.quality]
    sites["range_class"] = pd.cut(sites.gesla_harm__spring_range_m, RANGE_BINS)
    coast = pd.read_parquet(out / "extremes.parquet")
    coast = coast[coast.source == "gesla"].set_index("file_name")

    ev, prof = gauge_events(out, sites)
    hydro = {}
    if args.part in ("A", "both"):
        hydro |= part_a(ev, prof, sites, coast)
    if args.part in ("B", "both"):
        hydro |= part_b(out, ev, prof, sites, coast)
    meta = []
    for name, d in hydro.items():
        m = profile_measures(hourly_profile(d["h"]))
        meta.append({"hydrograph": name, **{k: v for k, v in d.items() if k != "h"}, **m})
    meta = pd.DataFrame(meta)
    meta.to_parquet(dest / f"hydrographs_{args.part}.parquet")
    np.savez(dest / f"hydrographs_{args.part}.npz", **{name: d["h"] for name, d in hydro.items()})

    tasks = []
    for name, d in hydro.items():
        terrains = ([f"basin{b:g}" for b in BASINS_KM] + ["slope"]) if d["part"] == "A" else B_TERRAINS
        for head in HEAD_FRACTIONS:
            for ter in terrains:
                for n in (A_MANNING if d["part"] == "A" else B_MANNING):
                    domain = f"h{head:g}_{ter}_n{n:g}"
                    tasks.append({"hydrograph": name, "head_fraction": head, "terrain": ter, "manning": n,
                                  "crest_m": d["peak_m"] * (1 - head), "h": with_settling(d["h"]),
                                  "exe": args.sfincs_exe, "run_dir": str(dest / "runs" / f"{name}__{domain}")})
    tasks = tasks[:args.trial] if args.trial else tasks
    print(f"{len(hydro)} hydrographs, {len(tasks)} runs", flush=True)
    rows, start = [], time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(_run, tasks, chunksize=1)):
            rows.append(r)
            if (k + 1) % 50 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1}/{len(tasks)} runs, {(time.time() - start) / 60:.0f} min", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_parquet(dest / f"runs_{args.part}{'_trial' if args.trial else ''}.parquet")
    print(runs.groupby(["terrain", "head_fraction"])[["extent_m", "volume_m2", "max_depth_m"]].median().round(2))


if __name__ == "__main__":
    main()
