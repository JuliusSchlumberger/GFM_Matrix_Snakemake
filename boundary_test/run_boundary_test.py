"""Plan §5.G: eikonal runs on the test tiles with the new boundary levels (H1) and hydrographs (H2).

Per tile, on the SFINCS subgrid with the production solver and fsf (calibration harness):
  baseline  production seeds, B from COAST-HG RP100 (production)
  H1        seeds = L_RP100 (build_boundary_product), B from COAST-HG
  H2        production seeds, B from the synthetic hydrographs scaled to the production level
  H1H2      seeds = L_RP100, B from the synthetic hydrographs scaled to L_RP100
H2 and H1H2 are run for every alpha in ALPHAS. Synthetic hydrograph per FES node: the node's
spring-tide window + the RP100 surge shape of the nearest GTSM station (per-RP event selection;
Gaussian at TC-dominated stations), scaled so the peak is the node's level (local MSL).
Scores are against the EXISTING SFINCS runs, forced with COAST-HG shifted to COAST-RP: a
stand-in that shares the old boundary levels, so H1 and H1H2 scores are not a test of H1.

Writes <out-dir>/runs/results.jsonl.

Usage:
    python boundary_test/run_boundary_test.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_provisional.csv [--workers 3]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.transform
import xarray as xr
from pyproj import Transformer
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "review_critiques"))
import calibration_experiments as ce  # noqa: E402
from event_volume import DurationCorrection, load_coast_hg  # noqa: E402
from flood_model import effective_dem  # noqa: E402
from storm_tide_hydrograph import cell_hydrograph, fill_short_gaps, remove_running_mean, storm_events, to_10min  # noqa: E402

FSF = 7.5  # production friction_scale_factor with B
T_REF_S = 12 * 3600.0
ALPHAS = (0.5, 1.0, 1.5, 2.0)
HALF_WINDOW_HR = 72.0
TC_HALF_WIDTH_HR = 8.0
SHAPE_MODE = "per_rp"  # F6a: per-RP selection; IMPLEMENTATION_NOTES §2.12


def node_hydrographs(tile_id: str, out: Path, level_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic RP100 hydrograph per FES node (lon/lat, series at 10 min), peak = node level."""
    cells = pd.read_parquet(out / "boundary_product" / f"{tile_id}.parquet")
    win = np.load(out / "fes_spring_windows" / f"{tile_id}.npz")
    nodes = cells.groupby(["node_lon", "node_lat"]).agg(level=(level_col, "median"), mdt=("mdt_m", "median"),
                                                        mhws=("mhws_fes_m", "median")).reset_index()
    win_key = {(round(float(a), 6), round(float(b), 6)): i for i, (a, b) in enumerate(zip(win["node_lon"], win["node_lat"]))}
    surge = xr.open_dataset(out / "gtsm_surge_test_stations.nc")
    stations = pd.read_parquet(out / "stations.parquet")
    st_tree = BallTree(np.radians(np.column_stack([surge.lat.values, surge.lon.values])), metric="haversine")
    inv_tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    events_cache: dict[int, dict] = {}
    lonlat, series = [], []
    for _, n in nodes.iterrows():
        i = win_key.get((round(n.node_lon, 6), round(n.node_lat, 6)))
        if i is None or not np.isfinite(win["tide"][i]).all() or not np.isfinite(n.level):
            continue
        j = int(st_tree.query(np.radians([[n.node_lat, n.node_lon]]), k=1)[1][0, 0])
        if j not in events_cache:
            s = fill_short_gaps(pd.Series(surge["surge"].values[:, j].astype(float),
                                          index=pd.DatetimeIndex(surge.time.values)))
            events_cache[j] = storm_events(to_10min(remove_running_mean(s)), HALF_WINDOW_HR) if s is not None else None
        events = events_cache[j]
        if events is None:
            continue
        tc = bool(stations.iloc[inv_tree.query(np.radians([[float(surge.lat[j]), float(surge.lon[j])]]), k=1)[1][0, 0]]
                  .tc_dominated)
        h = cell_hydrograph(win["tide"][i].astype(float), events, n.level - n.mdt, n.mhws, HALF_WINDOW_HR,
                            "gaussian" if tc else SHAPE_MODE, tc_half_width_hr=TC_HALF_WIDTH_HR)
        lonlat.append((n.node_lon, n.node_lat))
        series.append(h["level"] + n.mdt)
    return np.asarray(lonlat), np.vstack(series)


def run_tile(tile_id: str, out_dir: str) -> list[dict]:
    t0 = time.perf_counter()
    out = Path(out_dir)
    t = ce.load_tile(tile_id)
    if t["seeds"] is None:
        return [{"tile_id": tile_id, "variant": "no_seeds"}]
    rows, cols, prod_values = t["seeds"]
    cells = pd.read_parquet(out / "boundary_product" / f"{tile_id}.parquet")
    h1_values = cells.l_rp100_m.to_numpy()
    if len(h1_values) != len(prod_values):
        raise ValueError(f"tile {tile_id}: {len(h1_values)} product cells vs {len(prod_values)} seeds")
    h1_values = np.where(np.isfinite(h1_values), h1_values, prod_values)

    shape = t["dem"].shape
    to_ll = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True)
    xs, ys = to_ll.transform(*[np.array(v) for v in rasterio.transform.xy(
        t["transform"], [0, 0, shape[0] - 1, shape[0] - 1], [0, shape[1] - 1, 0, shape[1] - 1])])
    hg_ll, hg_series, hg_dt = load_coast_hg(ce.COAST_HG_NC, (min(xs), min(ys), max(xs), max(ys)))
    syn_prod = node_hydrographs(tile_id, out, "seed_prod_rp100_m")
    syn_h1 = node_hydrographs(tile_id, out, "l_rp100_m")
    sources = t["mask"] == ce.rcm.OCEAN_CODE
    sources[rows, cols] = True
    dem_eff = effective_dem(t["dem"], t["mask"])
    friction = t["friction"] * FSF

    def solve_seeds(values, fr):
        t["seeds"] = (rows, cols, values)
        return ce.solve(t, fr)

    def with_b(values, lonlat, series, dt_s, alpha):
        base = solve_seeds(values, friction)
        corr = DurationCorrection(lonlat, series, dt_s, T_REF_S, alpha, crs=t["crs"])
        mult = corr.multiplier(base, dem_eff, sources, t["transform"])
        return solve_seeds(values, friction * mult), float(mult[base > 0].mean()) if (base > 0).any() else 1.0

    records = []

    def add(variant, alpha, depth, mean_mult):
        records.append({"tile_id": tile_id, "variant": variant, "alpha": alpha, **ce.score(t, depth),
                        "mean_multiplier": mean_mult, "sfincs": "existing (COAST-HG + COAST-RP), stand-in"})

    add("baseline", 1.0, *with_b(prod_values, hg_ll, hg_series, hg_dt, 1.0))
    add("H1", 1.0, *with_b(h1_values, hg_ll, hg_series, hg_dt, 1.0))
    for alpha in ALPHAS:
        add("H2", alpha, *with_b(prod_values, *syn_prod, 600.0, alpha))
        add("H1H2", alpha, *with_b(h1_values, *syn_h1, 600.0, alpha))
    seeds_shift = h1_values - prod_values
    for r in records:
        r.update(elapsed_s_tile=round(time.perf_counter() - t0, 1), seed_shift_median_m=float(np.median(seeds_shift)),
                 seed_shift_p95_m=float(np.quantile(seeds_shift, 0.95)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiles-csv", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "runs").mkdir(exist_ok=True)
    results = out / "runs" / "results.jsonl"
    done = {json.loads(line)["tile_id"] for line in results.read_text().splitlines()} if results.exists() else set()
    tiles = [t for t in pd.read_csv(args.tiles_csv, dtype={"tile_id": str}).tile_id if t not in done]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_tile, t, str(out)): t for t in tiles}
        for future in as_completed(futures):
            try:
                recs = future.result()
            except Exception as exc:  # one bad tile must not stop the batch; recorded for the report
                recs = [{"tile_id": futures[future], "variant": "error", "error": repr(exc)}]
            with open(results, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"done {futures[future]}", flush=True)


if __name__ == "__main__":
    main()
