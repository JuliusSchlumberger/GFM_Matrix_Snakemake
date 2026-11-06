"""Test 6: MHWS of FES2022 and FES2014 at every boundary node of the test tiles.

Same prediction and statistics as at the gauges (`atlas_check.gauge_table`): constants
extracted with `fes_tide_stats.extract_constants` (spline, extrapolation to the nearest valid
cell within CUTOFF_KM), one 18.61-y nodal cycle at 10 min without Sa / Ssa, MHWS from
`tide_statistics.high_water_statistics` (the high-water half of `levels_tides.levels`). Nodes are all rows of `fes_nodes.parquet`, including those the earlier
extraction left invalid.

Writes <out-dir>/analysis/atlas_node_mhws.parquet (node_lon, node_lat, fes2022_mhws_m,
fes2014_mhws_m).

Usage:
    python boundary_test/atlas_node_mhws.py --out-dir .cleanup_baseline/level_sources [--workers 3]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.io

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_check import FES_NODES, SEASONAL, atlas_basis, grouped_constants, reference_model  # noqa: E402
from fes_tide_stats import FES_MODEL, STEP_MIN, ensure_local_fes  # noqa: E402
from levels_tides import FES_CACHE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tide_statistics import high_water_statistics  # noqa: E402

NODES_PER_TASK = 50


def _mhws_task(args) -> list[float]:
    re_path, im_path, hc = args
    b_re = np.load(re_path, mmap_mode="r")
    b_im = np.load(im_path, mmap_mode="r")
    ok = np.isfinite(hc).all(axis=1)
    tide = np.asarray(b_re) @ np.nan_to_num(hc.real).T + np.asarray(b_im) @ np.nan_to_num(hc.imag).T
    return [high_water_statistics(tide[:, j], STEP_MIN * 60.0)["mhws_m"] if ok[j] else np.nan for j in range(len(hc))]


def atlas_mhws(model, name: str, lonlat: np.ndarray, workers: int) -> pd.DataFrame:
    hc = grouped_constants(model, lonlat, "spline")
    cons = list(model.constituents)
    hc[:, [cons.index(c) for c in SEASONAL if c in cons]] = 0.0
    re_path, im_path = atlas_basis(model, name)
    tasks = [(re_path, im_path, hc[i:i + NODES_PER_TASK]) for i in range(0, len(hc), NODES_PER_TASK)]
    stats = []
    with ProcessPoolExecutor(workers) as pool:
        for k, chunk in enumerate(pool.map(_mhws_task, tasks)):
            stats += chunk
            if k % 100 == 0:
                print(f"  {name}: {len(stats)}/{len(hc)} nodes", flush=True)
    return pd.DataFrame({f"{name}_mhws_m": stats})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    dest = Path(args.out_dir) / "analysis"
    nodes = pd.read_parquet(FES_NODES)[["node_lon", "node_lat"]].reset_index(drop=True)
    lonlat = nodes.to_numpy(dtype=float)
    models = {"fes2022": pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL),
              "fes2014": reference_model("fes2014")}
    out = nodes.copy()
    for name, model in models.items():
        out = pd.concat([out, atlas_mhws(model, name, lonlat, args.workers)], axis=1)
        out.to_parquet(dest / "atlas_node_mhws.parquet")
        print(f"{name}: {out[f'{name}_mhws_m'].notna().sum()} of {len(out)} nodes with MHWS", flush=True)


if __name__ == "__main__":
    main()
