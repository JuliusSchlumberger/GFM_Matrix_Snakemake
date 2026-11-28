"""The calibration's model runs give the same rasters solved in a worker pool as one after another."""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))
sys.path.insert(0, str(REPO_ROOT / "src"))
import run_calibration_models as rcm  # noqa: E402
from calibration_paths import run_path  # noqa: E402
from config_utils import materialize_config  # noqa: E402
from process_pool import process_pool  # noqa: E402

RUNS = [("eikonal_landcover", 0.35, None), ("eikonal_uniform", 0.125, None), ("bathtub", None, None)]


def _tile(out_dir: Path) -> dict:
    dem = np.tile(np.linspace(0.0, 4.0, 30, dtype=np.float32), (20, 1))
    dem[:, 12:14] = 6.0  # a ridge the coupled solve has to block
    mask = np.full(dem.shape, rcm.LAND_CODE, dtype=np.int8)
    mask[:, 0] = rcm.OCEAN_CODE
    rows = np.arange(20, dtype=np.int64)
    flooding = materialize_config(REPO_ROOT / "snakemake_workflow" / "config" / "config.yml", [])["simulation"]["flooding"]
    return {"dem": dem, "mask": mask, "base_friction": np.full(dem.shape, 0.0005, dtype=np.float32),
            "transform": from_origin(500000.0, 4000000.0, 30.0, 30.0), "crs": "EPSG:32631",
            "seed_rows": rows, "seed_cols": np.ones(20, dtype=np.int64), "seed_values": np.full(20, 3.0, np.float32),
            "cell_dx_m": 30.0, "cell_dy_m": 30.0, "flooding": flooding, "river_code": 3,
            "sfincs_dir": out_dir, "out_dir": out_dir}


def _depths(out_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for name, parameter, alpha in RUNS:
        key = rcm.run_key(name, parameter, alpha)
        with rasterio.open(run_path(out_dir, key)) as src:
            out[key] = src.read(1)
    return out


def test_pool_and_sequential_runs_write_the_same_rasters(tmp_path) -> None:
    seq, par = tmp_path / "seq", tmp_path / "par"
    rcm._set_tile(_tile(seq))
    seq_records = {key: rec for run in RUNS for key, rec, _ in rcm._solve(*run, [run[0]])}
    with process_pool(2, initializer=rcm._set_tile, initargs=(_tile(par),)) as pool:
        par_records = {key: rec for f in [pool.submit(rcm._solve, *run, [run[0]]) for run in RUNS]
                       for key, rec, _ in f.result()}
    a, b = _depths(seq), _depths(par)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert any((d > 0).any() for d in a.values())  # something floods
    eik = seq_records[rcm.run_key("eikonal_landcover", 0.35, None)]
    assert eik["solver"] == par_records[rcm.run_key("eikonal_landcover", 0.35, None)]["solver"]
    assert len(eik["solver"]["inner_rounds_per_outer"]) == eik["solver"]["outer_iterations_used"]
