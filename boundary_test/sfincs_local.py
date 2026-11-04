"""Plan §5.F/G: re-run the existing SFINCS tile models locally with new boundary forcing (H1).

Per tile the model inputs are copied once from P: (read-only) to <out-dir>/sfincs/<tile>/model;
each variant runs in its own folder with only sfincs.bzs rewritten:
  control  the original sfincs.bzs; reproduces the HPC run (v2.4.0) exactly with the local
           v2.3.0 build on tile 3758, so the HPC hmax is the control and this is optional
  H1       sfincs.bzs + (L_RP100 - production level) of the nearest boundary seed, the same
           per-seed shift as the eikonal H1 run (hydrograph shape unchanged)
Then hmax_subgrid.tif is written by sfincs_tiles' own postprocessing (imported, unchanged).
Runs in the hydromt-sfincs-dev env.

Usage:
    python boundary_test/sfincs_local.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/sfincs_tiles_local.csv --sfincs-exe <sfincs.exe> [--threads 2]
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sfincs_tiles"))
from run_sfincs_tile import compute_max_inundation  # noqa: E402
from sfincs_run import run_sfincs_subprocess  # noqa: E402

BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
MODEL_FILES = ("sfincs.inp", "sfincs.bnd", "sfincs.bzs", "sfincs.ind", "sfincs.ini", "sfincs.msk",
               "sfincs_subgrid.nc", "subgrid/dep_subgrid.tif", "gis/bnd.geojson", "gis/mask.tif",
               "gis/region.geojson")
TIMEOUT_S = 6 * 3600.0


class _Log:
    def info(self, msg):
        print(msg, flush=True)

    warning = info


def stage_model(tile_id: str, dest: Path) -> Path:
    model = dest / "model"
    if (model / "sfincs.inp").exists():
        return model
    tmp = dest / "model.part"
    for name in MODEL_FILES:
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / tile_id / "sfincs_model" / name, tmp / name)
    tmp.replace(model)
    return model


def read_inp(path: Path) -> dict:
    return {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)}


def h1_shift_per_bnd_point(tile_id: str, model: Path, out: Path) -> np.ndarray:
    """L_RP100 - production level at the boundary seed nearest each sfincs.bnd point (m)."""
    cells = pd.read_parquet(out / "boundary_product" / f"{tile_id}.parquet")
    shift = (cells.l_rp100_m - cells.seed_prod_rp100_m).to_numpy()
    ok = np.isfinite(shift)
    epsg = int(read_inp(model / "sfincs.inp")["epsg"])
    x, y = Transformer.from_crs("EPSG:4326", epsg, always_xy=True).transform(cells.lon[ok].to_numpy(),
                                                                          cells.lat[ok].to_numpy())
    bnd = np.loadtxt(model / "sfincs.bnd", ndmin=2)
    return shift[ok][cKDTree(np.column_stack([x, y])).query(bnd[:, :2])[1]]


def write_variant(model: Path, run_dir: Path, bzs_shift: np.ndarray | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = run_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != "sfincs.bzs" and not target.exists():
            os.link(model / name, target)
    bzs = np.loadtxt(model / "sfincs.bzs", ndmin=2)
    if bzs_shift is not None:
        bzs[:, 1:] += bzs_shift[None, :]
    np.savetxt(run_dir / "sfincs.bzs", bzs, fmt=["%8.1f"] + ["%7.3f"] * (bzs.shape[1] - 1))


def run_variant(tile_id: str, run_dir: Path, exe: Path, threads: int) -> float:
    t0 = time.perf_counter()
    if not (run_dir / "hmax_subgrid.tif").exists():
        # A map without hmax is from an interrupted run, possibly truncated.
        (run_dir / "sfincs_map.nc").unlink(missing_ok=True)
        run_sfincs_subprocess(exe, run_dir, TIMEOUT_S, _Log(), label=f"SFINCS {tile_id} {run_dir.name}",
                              n_threads=threads)
        compute_max_inundation(run_dir, BASE / tile_id / "inputs" / "mask.tif")
        (run_dir / "sfincs_map.nc").unlink()  # 0.1-1 GB per run; zsmax lives on in hmax_subgrid.tif
    return time.perf_counter() - t0


def release_inputs(dest: Path, variants: list[str]) -> None:
    """Delete the staged model and the run inputs (~0.5 GB per tile); keep hmax, bzs and logs."""
    shutil.rmtree(dest / "model", ignore_errors=True)
    for variant in variants:
        for name in MODEL_FILES:
            if name not in ("sfincs.bzs", "sfincs.inp"):
                (dest / variant / name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiles-csv", required=True)
    parser.add_argument("--sfincs-exe", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["H1"])
    args = parser.parse_args()
    out = Path(args.out_dir)
    log = out / "sfincs" / "runs.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    for tile_id in pd.read_csv(args.tiles_csv, dtype={"tile_id": str}).tile_id:
        dest = out / "sfincs" / tile_id
        try:
            model = stage_model(tile_id, dest)
            for variant in args.variants:
                shift = h1_shift_per_bnd_point(tile_id, model, out) if variant == "H1" else None
                write_variant(model, dest / variant, shift)
                elapsed = run_variant(tile_id, dest / variant, Path(args.sfincs_exe), args.threads)
                row = {"tile_id": tile_id, "variant": variant, "elapsed_s": round(elapsed, 1),
                       "bzs_shift_median_m": float(np.median(shift)) if shift is not None else 0.0}
                pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
                print(row, flush=True)
            release_inputs(dest, args.variants)
        except Exception as exc:  # one bad tile must not stop the batch; recorded for the report
            pd.DataFrame([{"tile_id": tile_id, "variant": "error", "error": repr(exc)}]) \
                .to_csv(log, mode="a", header=not log.exists(), index=False)
            print(f"tile {tile_id}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
