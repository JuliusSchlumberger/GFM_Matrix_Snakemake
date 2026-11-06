"""Test 6: FES2014 as the reference tide atlas at the gauges (input of the FES version rule).

FES2014 is extracted at each of the 1,154 gauges with `fes_tide_stats.extract_constants` (spline,
as FES2022; extrapolation to the nearest valid cell within CUTOFF_KM), predicted over the same
18.61-y 10-min epoch with its own constituents (Sa / Ssa dropped, as the `no_seasonal` variant) and
reduced with `tide_statistics` to MHWS, next to GESLA, GTSM and FES2022 (`levels_sites.csv`).
`atlas_node_mhws.py` and `atlas_rule_sensitivity.py` build on these helpers and this table.

The atlas is read from the folder that holds its constituent files (catalog entry
`fes2014_ocean_tide`).

Writes <out-dir>/analysis/atlas_gauges.csv.

Usage:
    python boundary_test/atlas_check.py --out-dir .cleanup_baseline/level_sources [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from fes_tide_stats import STEP_MIN, extract_constants, prediction_basis  # noqa: E402
from levels_tides import GROUP_DEG, levels  # noqa: E402

INPUTS = Path("P:/11212688-004-global-floodmaps/modelling/inputs")
REFERENCE_ATLASES = {"fes2014": ("FES2014", INPUTS / "fes2014" / "fes2014b_elevations_extrapolated"
                                 / "ocean_tide_extrapolated", "FES2014_extrapolated", "spline")}
FES_NODES = REPO / ".cleanup_baseline" / "boundary_test" / "fes_nodes.parquet"
SEASONAL = ("sa", "ssa")
POINTS_PER_TASK = 25
BASINS = {"Minas Basin / Cobequid Bay": (-65.2, -63.2, 44.9, 45.6),
          "Strait of Magellan / Bahia Lomas": (-71.5, -68.0, -53.6, -52.0),
          "Upper Gulf of California": (-115.3, -113.0, 30.3, 32.2),
          "Glacier Bay / Lynn Canal": (-137.2, -134.6, 58.2, 59.6)}


def reference_model(name: str):
    _, files_dir, model_name, _ = REFERENCE_ATLASES[name]
    # pyTMD expects its own folder layout; the inputs keep the provider's, so the file paths are remapped.
    model = pyTMD.io.model(verify=False).elevation(model_name)
    model.model_file = [files_dir / Path(f).name for f in model.model_file]
    return model


def grouped_constants(model, lonlat: np.ndarray, method: str = "spline") -> np.ndarray:
    cell = np.floor(lonlat / GROUP_DEG).astype(int)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    hc = extract_constants(model, [lonlat[g] for g in groups], method)
    out = np.empty_like(hc)
    out[np.concatenate(groups)] = hc
    return out


def atlas_basis(model, name: str) -> tuple[Path, Path]:
    cache = REPO / ".cleanup_baseline" / f"{name}_cache"
    cache.mkdir(parents=True, exist_ok=True)
    re_path, im_path = cache / "basis_re.npy", cache / "basis_im.npy"
    if not re_path.exists():
        b_re, b_im = prediction_basis(list(model.constituents), model)
        np.save(re_path, b_re.astype(np.float32))
        np.save(im_path, b_im.astype(np.float32))
    return re_path, im_path


def _levels_task(args) -> list[dict]:
    re_path, im_path, hc = args
    b_re = np.load(re_path, mmap_mode="r")
    b_im = np.load(im_path, mmap_mode="r")
    ok = np.isfinite(hc).all(axis=1)
    tide = np.asarray(b_re) @ np.nan_to_num(hc.real).T + np.asarray(b_im) @ np.nan_to_num(hc.imag).T
    return [levels(tide[:, j], STEP_MIN * 60.0) if ok[j] else {} for j in range(len(hc))]


def gauge_table(out: Path, workers: int) -> pd.DataFrame:
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    pairs = pd.read_parquet(out / "pairs.parquet").set_index("file_name").reindex(sites.index)
    lonlat = pairs[["lon", "lat"]].to_numpy(dtype=float)
    t = sites[["site_name", "country", "region", "lon", "lat", "quality", "gesla_harm__mhws_m", "gtsm_harm__mhws_m",
               "fes_gauge__mhws_m", "gesla_harm__spring_range_m"]].copy()
    for name, (_, _, _, method) in REFERENCE_ATLASES.items():
        model = reference_model(name)
        hc = grouped_constants(model, lonlat, method)
        cons = list(model.constituents)
        hc[:, [cons.index(c) for c in SEASONAL if c in cons]] = 0.0
        re_path, im_path = atlas_basis(model, name)
        tasks = [(re_path, im_path, hc[i:i + POINTS_PER_TASK]) for i in range(0, len(hc), POINTS_PER_TASK)]
        with ProcessPoolExecutor(workers) as pool:
            stats = [r for chunk in pool.map(_levels_task, tasks) for r in chunk]
        t[f"{name}__mhws_m"] = [r.get("mhws_m", np.nan) for r in stats]
        t[f"{name}__spring_range_m"] = [r.get("spring_range_m", np.nan) for r in stats]
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out_dir)
    gauges = gauge_table(out, args.workers)
    gauges.round(4).to_csv(out / "analysis" / "atlas_gauges.csv")
    q = gauges[gauges.quality]
    for col, label in (("fes_gauge__mhws_m", "FES2022"), ("fes2014__mhws_m", "FES2014")):
        d = (q[col] - q.gesla_harm__mhws_m).dropna()
        print(f"{label} vs gauge MHWS: median ratio {(q[col] / q.gesla_harm__mhws_m).median():.3f}, "
              f"MAE {d.abs().mean():.3f} m, within 0.1 m {(d.abs() <= 0.1).mean():.2f}, "
              f"> 0.25 m {(d.abs() > 0.25).sum()} (n={len(d)})")
    miss = q[(q.fes_gauge__mhws_m - q.gesla_harm__mhws_m).abs() > 0.25]
    print("\nat the FES-miss sites\n" + miss[["site_name", "gesla_harm__mhws_m", "fes_gauge__mhws_m", "fes2014__mhws_m",
                                               "gtsm_harm__mhws_m"]].round(2).to_string())


if __name__ == "__main__":
    main()
