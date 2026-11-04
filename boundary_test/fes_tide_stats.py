"""Plan §5.A: FES2022 tide statistics (MHWS, MHW, HW99, HAT) at the test tiles' boundary cells.

Boundary cells are the production seeds of each tile on its SFINCS subgrid (as the calibration
harness), grouped by FES grid node (1/30 deg). Each node gets one nodal cycle at 10 min
(hourly steps miss shallow-water crests, `tide_statistics.peak_values`), predicted as pyTMD
does (FES nodal corrections, inferred minor constituents, extrapolation to the nearest valid
ocean node within CUTOFF_KM) but as one matrix product: prediction is linear in the harmonic
constants, so the time x constituent basis is built once (`prediction_basis`, checked against
`pyTMD.compute.tide_elevations` to 1e-8 m). Each constituent grid is decompressed once and
interpolated per node group on a crop (`extract_constants`); pyTMD re-reads the global grids
on every call (~4 min). FES files are decompressed from P: (.nc.xz, read-only) into a local
cache once.

Writes <out-dir>/fes_nodes.parquet + fes_node_windows.npy (all nodes, incremental) and per
tile fes_node_stats/, fes_cell_node_map/, fes_spring_windows/; then tide_stats_fes2022.parquet.

Usage:
    python boundary_test/fes_tide_stats.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_representative.csv [--workers 4]
"""

import argparse
import lzma
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.arguments
import pyTMD.io
import pyTMD.io.FES
import pyTMD.predict
import rasterio.transform
import timescale.time
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "review_critiques"))
from storm_tide_hydrograph import STEPS_PER_HR  # noqa: E402
from tide_statistics import high_water_statistics, spring_high_waters  # noqa: E402

FES_SOURCE = Path("P:/11212688-004-global-floodmaps/modelling/inputs/fes2022b/ocean_tide_extrapolated")
FES_MODEL = "FES2022_extrapolated"
NODE_DEG = 1.0 / 30.0
CUTOFF_KM = 20.0
NODAL_CYCLE_YR = 18.61
START = np.datetime64("1996-01-01T00:00")
STEP_MIN = 60 // STEPS_PER_HR  # 10 min, the step of the synthetic hydrographs
WINDOW_HR = 72  # plan §5.E: 6-day window centred on a spring high water
NODES_PER_TASK = 40
KEY_DECIMALS = 6
CROP_BUFFER_DEG = 0.5  # > CUTOFF_KM, so extrapolation sees every valid cell within the cutoff


def ensure_local_fes(cache: Path) -> Path:
    """Decompress the FES2022 .nc.xz files into cache/fes2022b/ocean_tide_extrapolated once."""
    dest = cache / "fes2022b" / "ocean_tide_extrapolated"
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(FES_SOURCE.iterdir()):
        name = src.name[:-3] if src.name.endswith(".xz") else src.name
        target = dest / name
        if target.exists() or not name.endswith(".nc"):
            continue
        tmp = target.with_suffix(".nc.part")
        opener = lzma.open if src.name.endswith(".xz") else open
        with opener(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 << 20)
        tmp.replace(target)
    return cache


def tile_seeds(tile_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the tile's production seed (lon, lat) and RP100 seed level (m), in seed order."""
    import calibration_experiments as ce
    t = ce.load_tile(tile_id)
    if t["seeds"] is None:
        return np.zeros((0, 2)), np.zeros(0)
    rows, cols, values = t["seeds"]
    xs, ys = rasterio.transform.xy(t["transform"], rows, cols)
    lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    return np.column_stack([lon, lat]), np.asarray(values, dtype=float)


def node_of(lonlat: np.ndarray) -> np.ndarray:
    return np.round(np.round(lonlat / NODE_DEG) * NODE_DEG, KEY_DECIMALS)


def extract_constants(model, groups: list[np.ndarray]) -> np.ndarray:
    """Complex harmonic constants (node, constituent) for the (lon, lat) node groups, stacked.

    As `model.extract_constants(crop=True, extrapolate=True)` per group, with each constituent
    grid read once: `pyTMD.io.FES.read_netcdf_file` is swapped for a one-file cache meanwhile.
    """
    fes, original = pyTMD.io.FES, pyTMD.io.FES.read_netcdf_file
    cache: dict = {}

    def cached(path, **kwargs):
        if str(path) not in cache:
            cache.clear()
            cache[str(path)] = original(path, **kwargs)
        return cache[str(path)]

    # 0-360 longitudes (the FES grid's) avoid shifting the global grid per crop, except for
    # groups that straddle 0 deg.
    groups = [np.column_stack([np.where(g[:, 0] < 0, g[:, 0] + 360, g[:, 0]), g[:, 1]])
              if np.ptp(np.where(g[:, 0] < 0, g[:, 0] + 360, g[:, 0])) < 180 else g for g in groups]
    hc = [np.full((len(g), len(model.model_file)), np.nan + 1j * np.nan) for g in groups]
    fes.read_netcdf_file = cached
    try:
        for k, model_file in enumerate(model.model_file):
            for g, out in zip(groups, hc):
                amp, ph = fes.extract_constants(g[:, 0], g[:, 1], [model_file], version=model.version,
                                                compressed=model.compressed, type=model.type, scale=model.scale,
                                                crop=True, buffer=CROP_BUFFER_DEG, method="spline",
                                                extrapolate=True, cutoff=CUTOFF_KM)
                out[:, k] = np.ma.filled(amp[:, 0] * np.exp(-1j * ph[:, 0] * np.pi / 180.0), np.nan)
            print(f"  constituent {k + 1}/{len(model.model_file)}", flush=True)
    finally:
        fes.read_netcdf_file = original
    return np.vstack(hc)


def prediction_basis(constituents: list, model) -> tuple[np.ndarray, np.ndarray]:
    """Return (B_re, B_im), (time, constituent): tide = B_re @ hc.real + B_im @ hc.imag.

    Major part as `pyTMD.predict.time_series`; minor part from `pyTMD.predict.infer_minor`
    evaluated on unit constants, which is exact because the inference is linear in them.
    """
    steps = np.arange(int(NODAL_CYCLE_YR * 365.25 * 24 * 60 / STEP_MIN))
    ts = timescale.time.Timescale().from_datetime(START + (steps * STEP_MIN).astype("timedelta64[m]"))
    pu, pf, G = pyTMD.arguments.arguments(ts.tide + pyTMD.predict._mjd_tide, constituents,
                                          deltat=ts.tt_ut1, corrections=model.corrections)
    th = G * np.pi / 180.0 + pu
    b_re, b_im = pf * np.cos(th), -pf * np.sin(th)
    del pu, pf, G, th
    nc = len(constituents)
    for k in range(nc):
        for unit, dst in ((1.0, b_re), (1j, b_im)):
            z = np.ma.zeros((1, nc), dtype=np.complex128)
            z[0, k] = unit
            dst[:, k] += np.ma.filled(pyTMD.predict.infer_minor(ts.tide, z, constituents, deltat=ts.tt_ut1,
                                                                corrections=model.corrections, minor=model.minor), 0.0)
    return b_re, b_im


def spring_window(series: np.ndarray, dt_s: float, mhws_m: float) -> np.ndarray:
    """±WINDOW_HR tide around the spring high water closest to MHWS."""
    half = int(round(WINDOW_HR * 3600 / dt_s))
    spring = spring_high_waters(series, dt_s)
    spring = spring[(spring - half >= 0) & (spring + half < len(series))]
    centre = int(spring[np.argmin(np.abs(series[spring] - mhws_m))])
    return series[centre - half: centre + half + 1]


def _node_task(args) -> tuple[list[dict], np.ndarray]:
    basis_dir, hc_re, hc_im = args
    b_re = np.load(Path(basis_dir) / "basis_re.npy", mmap_mode="r")
    b_im = np.load(Path(basis_dir) / "basis_im.npy", mmap_mode="r")
    dt_s = STEP_MIN * 60.0
    n_window = int(2 * WINDOW_HR * 60 / STEP_MIN) + 1
    tide = b_re @ np.nan_to_num(hc_re).T + b_im @ np.nan_to_num(hc_im).T
    rows, windows = [], []
    for j in range(hc_re.shape[0]):
        if not (np.isfinite(hc_re[j]).all() and np.isfinite(hc_im[j]).all()):
            rows.append({"valid": False})
            windows.append(np.full(n_window, np.nan))
            continue
        s = high_water_statistics(tide[:, j], dt_s)
        rows.append({"valid": True, **{k: v for k, v in s.items() if k != "high_waters_m"}})
        windows.append(spring_window(tide[:, j], dt_s, s["mhws_m"]))
    return rows, np.vstack(windows).astype(np.float32)


def node_statistics(groups: list[np.ndarray], fes_dir: Path, cache: Path,
                    workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Statistics and spring windows of the nodes of all groups, in stacked group order."""
    model = pyTMD.io.model(str(fes_dir)).elevation(FES_MODEL)
    nodes = np.vstack(groups)
    print(f"extracting constants at {len(nodes)} nodes in {len(groups)} groups", flush=True)
    hc = extract_constants(model, groups)
    if not (cache / "basis_re.npy").exists():
        print("building the prediction basis", flush=True)
        b_re, b_im = prediction_basis(list(model.constituents), model)
        np.save(cache / "basis_im.npy", b_im)
        np.save(cache / "basis_re.npy", b_re)
        del b_re, b_im
    tasks = [(str(cache), hc.real[i:i + NODES_PER_TASK], hc.imag[i:i + NODES_PER_TASK])
             for i in range(0, len(nodes), NODES_PER_TASK)]
    rows, windows = [], []
    with ProcessPoolExecutor(workers) as pool:
        for r, w in pool.map(_node_task, tasks):
            rows += r
            windows.append(w)
            print(f"  nodes {len(rows)}/{len(nodes)}", flush=True)
    stats = pd.DataFrame(rows).assign(node_lon=nodes[:, 0], node_lat=nodes[:, 1])
    return stats, np.vstack(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiles-csv", required=True)
    parser.add_argument("--fes-cache", default=".cleanup_baseline/fes2022_cache")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out, cache = Path(args.out_dir), Path(args.fes_cache)
    fes_dir = ensure_local_fes(cache)
    tiles = list(pd.read_csv(args.tiles_csv, dtype={"tile_id": str}).tile_id)
    stations = pd.read_csv(out / "test_station_tiles.csv", dtype={"tile_id": str})
    for d in ("fes_cell_node_map", "fes_spring_windows", "fes_node_stats"):
        (out / d).mkdir(exist_ok=True)

    todo = [t for t in tiles if not (out / "fes_cell_node_map" / f"{t}.parquet").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for tile_id, (ll, seed_level) in zip(todo, pool.map(tile_seeds, todo)):
            nodes = node_of(ll)
            pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "node_lon": nodes[:, 0], "node_lat": nodes[:, 1],
                          "seed_prod_rp100_m": seed_level}).to_parquet(out / "fes_cell_node_map" / f"{tile_id}.parquet")
            print(f"seeds {tile_id}: {len(ll)}", flush=True)

    tile_nodes = {}
    for tile_id in tiles:
        cells = pd.read_parquet(out / "fes_cell_node_map" / f"{tile_id}.parquet")
        st = node_of(stations.loc[stations.tile_id == tile_id, ["lon", "lat"]].to_numpy())
        tile_nodes[tile_id] = np.unique(np.concatenate([cells[["node_lon", "node_lat"]].to_numpy(), st]), axis=0)

    nodes_path, windows_path = out / "fes_nodes.parquet", out / "fes_node_windows.npy"
    known = (pd.read_parquet(nodes_path) if nodes_path.exists()
             else pd.DataFrame({"node_lon": pd.Series(dtype=float), "node_lat": pd.Series(dtype=float)}))
    known_windows = np.load(windows_path) if windows_path.exists() else np.zeros((0, 2 * WINDOW_HR * STEPS_PER_HR + 1))
    have = set(zip(known.node_lon.round(KEY_DECIMALS), known.node_lat.round(KEY_DECIMALS)))
    groups, seen = [], set(have)
    for nodes in tile_nodes.values():
        g = [n for n in nodes if (n[0], n[1]) not in seen]
        seen.update((n[0], n[1]) for n in g)
        if g:
            groups.append(np.array(g))
    if groups:
        new, new_windows = node_statistics(groups, fes_dir, cache, args.workers)
        known = pd.concat([known, new], ignore_index=True)
        known_windows = np.vstack([known_windows, new_windows]).astype(np.float32)
        known.to_parquet(nodes_path)
        np.save(windows_path, known_windows)

    index = {(round(a, KEY_DECIMALS), round(b, KEY_DECIMALS)): i
             for i, (a, b) in enumerate(zip(known.node_lon, known.node_lat))}
    time_hr = np.arange(-WINDOW_HR * STEPS_PER_HR, WINDOW_HR * STEPS_PER_HR + 1) / STEPS_PER_HR
    for tile_id, nodes in tile_nodes.items():
        idx = [index[(round(a, KEY_DECIMALS), round(b, KEY_DECIMALS))] for a, b in nodes]
        stats = known.iloc[idx].reset_index(drop=True)
        stats.assign(tile_id=tile_id).to_parquet(out / "fes_node_stats" / f"{tile_id}.parquet")
        np.savez_compressed(out / "fes_spring_windows" / f"{tile_id}.npz", node_lon=stats.node_lon.to_numpy(),
                            node_lat=stats.node_lat.to_numpy(), tide=known_windows[idx], time_hr=time_hr)
    stats = pd.concat([pd.read_parquet(f) for f in sorted((out / "fes_node_stats").glob("*.parquet"))],
                      ignore_index=True)
    stats.to_parquet(out / "tide_stats_fes2022.parquet")
    print(f"{len(known)} nodes, {known.valid.mean():.1%} valid; {stats.tile_id.nunique()} tiles")


if __name__ == "__main__":
    main()
