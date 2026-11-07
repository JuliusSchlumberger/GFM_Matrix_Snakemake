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
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.io
import rasterio.transform
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "review_critiques"))
from storm_tide_hydrograph import STEPS_PER_HR  # noqa: E402
from tide_statistics import high_water_statistics, spring_high_waters  # noqa: E402
from tide_prediction import extract_constants, nodal_times, tide_basis  # noqa: E402
from tide_prediction import ensure_local_fes as _ensure_local_fes  # noqa: E402

FES_SOURCE = Path("P:/11212688-004-global-floodmaps/modelling/inputs/fes2022b/ocean_tide_extrapolated")
FES_MODEL = "FES2022_extrapolated"
NODE_DEG = 1.0 / 30.0
STEP_MIN = 60 // STEPS_PER_HR  # 10 min, the step of the synthetic hydrographs
def ensure_local_fes(cache: Path) -> Path:
    return _ensure_local_fes(FES_SOURCE, cache)


def prediction_basis(constituents: list, model) -> tuple[np.ndarray, np.ndarray]:
    return tide_basis(model, nodal_times())


WINDOW_HR = 72  # plan §5.E: 6-day window centred on a spring high water
NODES_PER_TASK = 40
KEY_DECIMALS = 6


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


def _tile_seeds_or_none(tile_id: str):
    try:
        return tile_seeds(tile_id)
    except Exception as exc:  # e.g. a SFINCS grid that does not nest into its subgrid; tile is skipped
        print(f"seeds {tile_id}: FAILED {exc!r}", flush=True)
        return None


def node_of(lonlat: np.ndarray) -> np.ndarray:
    return np.round(np.round(lonlat / NODE_DEG) * NODE_DEG, KEY_DECIMALS)


def spring_window(series: np.ndarray, dt_s: float, mhws_m: float) -> np.ndarray:
    """±WINDOW_HR tide around the spring high water closest to MHWS."""
    half = int(round(WINDOW_HR * 3600 / dt_s))
    spring = spring_high_waters(series, dt_s)
    spring = spring[(spring - half >= 0) & (spring + half < len(series))]
    if spring.size == 0 or not np.isfinite(mhws_m):  # amphidromic or tideless node
        return np.full(2 * half + 1, np.nan)
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
        for tile_id, seeds in zip(todo, pool.map(_tile_seeds_or_none, todo)):
            if seeds is None:
                continue
            ll, seed_level = seeds
            nodes = node_of(ll)
            pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "node_lon": nodes[:, 0], "node_lat": nodes[:, 1],
                          "seed_prod_rp100_m": seed_level}).to_parquet(out / "fes_cell_node_map" / f"{tile_id}.parquet")
            print(f"seeds {tile_id}: {len(ll)}", flush=True)

    tiles = [t for t in tiles if (out / "fes_cell_node_map" / f"{t}.parquet").exists()]
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
