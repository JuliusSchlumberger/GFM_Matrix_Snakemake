"""Plan §5.A: FES2022 tide statistics (MHWS, MHW, HW99, HAT) at the test tiles' boundary cells.

Boundary cells are the production seeds of each tile on its SFINCS subgrid (as the calibration
harness). Cells are grouped by FES grid node (1/30 deg); pyTMD predicts hourly over one nodal
cycle at each node (nodal corrections, extrapolation to the nearest valid ocean node within
CUTOFF_KM). FES files are decompressed from P: (.nc.xz, read-only) into a local cache once.
Also evaluates the COAST-RP stations of the tiles, for F2/F4.

Writes <out-dir>/tide_stats_fes2022.parquet (per node) and fes_cell_node_map/<tile>.parquet.

Usage:
    python boundary_test/fes_tide_stats.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_provisional.csv
"""

import argparse
import lzma
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.compute
import rasterio.transform
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "review_critiques"))
from scipy.interpolate import CubicSpline  # noqa: E402
from storm_tide_hydrograph import STEPS_PER_HR  # noqa: E402
from tide_statistics import high_water_statistics, spring_high_waters  # noqa: E402

FES_SOURCE = Path("P:/11212688-004-global-floodmaps/modelling/inputs/fes2022b/ocean_tide_extrapolated")
FES_MODEL = "FES2022_extrapolated"
NODE_DEG = 1.0 / 30.0
CUTOFF_KM = 20.0
NODAL_CYCLE_YR = 18.61
START = np.datetime64("1996-01-01T00:00")
NODES_PER_CHUNK = 50
WINDOW_HR = 72  # plan §5.E: 6-day window centred on a spring high water


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
    return np.round(lonlat / NODE_DEG) * NODE_DEG


def spring_window_10min(series_hourly: np.ndarray, mhws_m: float) -> np.ndarray:
    """±WINDOW_HR tide at 10 min around the spring high water closest to MHWS (cubic from hourly)."""
    spring = spring_high_waters(series_hourly, 3600.0)
    pad = int(WINDOW_HR) + 2
    spring = spring[(spring - pad >= 0) & (spring + pad < len(series_hourly))]
    centre = int(spring[np.argmin(np.abs(series_hourly[spring] - mhws_m))])
    hrs = np.arange(centre - pad, centre + pad + 1)
    fine = CubicSpline(hrs, series_hourly[hrs])(np.arange(hrs[0], hrs[-1] + 1e-9, 1 / STEPS_PER_HR))
    peak = int(np.argmax(fine[(pad - 1) * STEPS_PER_HR:(pad + 1) * STEPS_PER_HR])) + (pad - 1) * STEPS_PER_HR
    half = int(WINDOW_HR * STEPS_PER_HR)
    return fine[peak - half: peak + half + 1]


def predict_stats(nodes: np.ndarray, fes_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    hours = np.arange(int(NODAL_CYCLE_YR * 365.25 * 24))
    times = START + hours.astype("timedelta64[h]")
    rows, windows = [], []
    for i in range(0, len(nodes), NODES_PER_CHUNK):
        chunk = nodes[i:i + NODES_PER_CHUNK]
        # CROP: read only the constituent grids around these nodes, not the global 1/30 deg grid.
        tide = pyTMD.compute.tide_elevations(chunk[:, 0], chunk[:, 1], times, DIRECTORY=str(fes_dir),
                                             MODEL=FES_MODEL, EPSG=4326, TYPE="time series", TIME="datetime",
                                             EXTRAPOLATE=True, CUTOFF=CUTOFF_KM, CROP=True)
        tide = np.ma.filled(np.atleast_2d(tide), np.nan)
        for (lon, lat), series in zip(chunk, tide):
            if not np.isfinite(series).all():
                rows.append({"node_lon": lon, "node_lat": lat, "valid": False})
                windows.append(np.full(int(2 * WINDOW_HR * STEPS_PER_HR) + 1, np.nan))
                continue
            s = high_water_statistics(series, 3600.0)
            rows.append({"node_lon": lon, "node_lat": lat, "valid": True,
                         **{k: v for k, v in s.items() if k != "high_waters_m"}})
            windows.append(spring_window_10min(series, s["mhws_m"]))
        print(f"  nodes {i + len(chunk)}/{len(nodes)}", flush=True)
    return pd.DataFrame(rows), np.vstack(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiles-csv", required=True)
    parser.add_argument("--fes-cache", default=".cleanup_baseline/fes2022_cache")
    args = parser.parse_args()
    out = Path(args.out_dir)
    fes_dir = ensure_local_fes(Path(args.fes_cache))
    tiles = pd.read_csv(args.tiles_csv, dtype={"tile_id": str}).tile_id
    stations = pd.read_csv(out / "test_station_tiles.csv", dtype={"tile_id": str})
    (out / "fes_cell_node_map").mkdir(exist_ok=True)
    (out / "fes_spring_windows").mkdir(exist_ok=True)
    parts = []
    for tile_id in tiles:
        ll, seed_level = tile_seeds(tile_id)
        cell_nodes = node_of(ll)
        pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "node_lon": cell_nodes[:, 0], "node_lat": cell_nodes[:, 1],
                      "seed_prod_rp100_m": seed_level}) \
            .to_parquet(out / "fes_cell_node_map" / f"{tile_id}.parquet")
        station_nodes = node_of(stations.loc[stations.tile_id == tile_id, ["lon", "lat"]].to_numpy())
        nodes = np.unique(np.round(np.concatenate([cell_nodes, station_nodes]), 6), axis=0)
        print(f"tile {tile_id}: {len(ll)} seeds, {len(nodes)} FES nodes", flush=True)
        stats, windows = predict_stats(nodes, fes_dir)
        np.savez_compressed(out / "fes_spring_windows" / f"{tile_id}.npz", node_lon=stats.node_lon.to_numpy(),
                            node_lat=stats.node_lat.to_numpy(), tide=windows.astype(np.float32),
                            time_hr=np.arange(-WINDOW_HR * STEPS_PER_HR, WINDOW_HR * STEPS_PER_HR + 1) / STEPS_PER_HR)
        parts.append(stats.assign(tile_id=tile_id))
    stats = pd.concat(parts, ignore_index=True)
    stats.to_parquet(out / "tide_stats_fes2022.parquet")
    print(f"{len(stats)} nodes, {stats.valid.mean():.1%} valid")


if __name__ == "__main__":
    main()
