"""Branch coverage for `aqueduct_runner.run_tile_scenario`, the per-tile logic shared by
both production entry points (snakemake_workflow/scripts/run_aqueduct.py and
run_aqueduct_cli.py).

Fully synthetic: tiny dem/mask/friction rasters in the real int16 on-disk conventions plus
boundaries GeoPackages and a tile grid, covering every outcome: hop-0 solve, hop-0 with no
stations, hop-1 seeded from an overlapping wave-0 tile, OOM pre-marked, and a MemoryError
during the solve. Also checks that `stage_boundaries=True` cleans up its scratch copy.

Plain assert-based script (this repo's tests/ convention): runs standalone with
`python tests/aqueduct_runner_validation/validate_run_tile_scenario.py`; pytest collects it.
"""

import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import aqueduct_runner  # noqa: E402
from aqueduct_runner import (  # noqa: E402
    NO_STATIONS_REASON,
    OOM_PREMARKED_REASON,
    mark_tile_oom,
    run_tile_scenario,
)
from rasters import WATERDEPTH_NODATA_INT16, encode_dem_cm, encode_friction_int16, encode_waterlevel_cm  # noqa: E402

D, M, N = 0.0003, 10, 12
SCN, SLR = "RP100_SLR_0", "SLR_0"
RASTER_CONFIG = {"driver": "GTiff", "compression": "deflate", "predictor": 3, "nodata": -9999}
FLOODING = {"resolution": 30.0, "knn": 5, "max_rounds": 40, "waterlevel_epsilon_m": 0.03,
            "obstacle_coupling": {"enabled": False}}


def _write_tile(root: Path, tid: int, x0: float, dem_m, mask, stations) -> box:
    d = root / "inputs" / str(tid) / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    tr = Affine(D, 0, x0, 0, -D, 50.0)
    for name, arr in (("dem.tif", encode_dem_cm(dem_m)), ("mask.tif", mask),
                      ("friction.tif", encode_friction_int16(np.full((M, N), 0.0005)))):
        with rasterio.open(d / name, "w", driver="GTiff", dtype="int16", count=1, height=M, width=N,
                           transform=tr, crs="EPSG:4326") as dst:
            dst.write(np.asarray(arr).astype(np.int16), 1)
    gpd.GeoDataFrame({SLR: encode_waterlevel_cm(np.array([s[2] for s in stations], dtype=float))},
                     geometry=[Point(s[0], s[1]) for s in stations], crs="EPSG:4326",
                     ).to_file(d / f"boundaries_{SCN}.gpkg", driver="GPKG")
    return box(x0, 50.0 - M * D, x0 + N * D, 50.0)


def _setup(root: Path) -> None:
    ramp = np.tile(np.linspace(0.3, 3.0, N), (M, 1))
    mask = np.zeros((M, N), np.uint8)
    mask[:, :3] = 1
    dem = np.where(mask == 1, 0.0, ramp)
    geoms = {
        1: _write_tile(root, 1, 10.0, dem, mask, [(10.0002, 49.9985, 2.5)]),
        2: _write_tile(root, 2, 20.0, dem, mask, []),
        3: _write_tile(root, 3, 10.0 + 6 * D, np.tile(np.linspace(0.6, 3.0, N), (M, 1)),
                       np.zeros((M, N), np.uint8), []),
        4: _write_tile(root, 4, 40.0, dem, mask, [(40.0002, 49.9985, 2.5)]),
    }
    gpd.GeoDataFrame({"tile_id": list(geoms), "hop_distance": [0, 0, 1, 0]}, geometry=list(geoms.values()),
                     crs="EPSG:4326").to_file(root / "tile_grid.gpkg", driver="GPKG")


def _run(root: Path, tid: int, **kw) -> tuple[dict, np.ndarray]:
    d = root / "inputs" / str(tid) / "inputs"
    out = root / "mo" / str(tid) / "results" / f"waterdepth_{SCN}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = run_tile_scenario(
        str(tid), "RP100", SLR,
        dem_path=str(d / "dem.tif"), mask_path=str(d / "mask.tif"), friction_path=str(d / "friction.tif"),
        boundaries_path=str(d / f"boundaries_{SCN}.gpkg"), output_path=str(out),
        model_outputs=str(root / "mo"), preprocessing_inputs_dir=str(root / "inputs"),
        tile_grid_path=str(root / "tile_grid.gpkg"), raster_config=RASTER_CONFIG, flooding_config=FLOODING,
        ocean_code=1, river_code=3, **kw,
    )
    with rasterio.open(out) as src:
        return res, src.read(1)


def test_all_outcomes() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _setup(root)
        scratch = root / "scratch"
        old_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(scratch)  # _stage_file_locally copies into $TMPDIR
        try:
            res, arr = _run(root, 1, stage_boundaries=True)
        finally:
            if old_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = old_tmpdir
        assert res["outcome"] == "solved" and "diagnostics" in res and res["solve_elapsed_s"] >= 0, res
        assert (arr > 0).sum() > 0
        assert not any(p.is_file() for p in scratch.rglob("*")), "staged boundaries copy not removed"
        print("PASS: hop-0 tile with a station solves; staged boundaries copy is removed")

        res, arr = _run(root, 2)
        assert res == {"outcome": "skip_no_inundation", "skip_reason": NO_STATIONS_REASON}, res
        assert (arr == 0).all()
        print("PASS: hop-0 tile without stations -> zero raster + skip reason")

        res, arr = _run(root, 3)
        assert res["outcome"] == "solved", res
        assert (arr > 0).sum() > 0, "hop-1 tile should be seeded from tile 1's flooding"
        print("PASS: hop-1 tile is seeded from the overlapping wave-0 tile's result")

        mark_tile_oom(root / "mo" / "oom_tiles", "2", reason="test")
        res, arr = _run(root, 2)
        assert res == {"outcome": "oom_premarked"}, res
        assert (arr == WATERDEPTH_NODATA_INT16).all()
        assert (root / "mo" / "skipped_tiles" / f"2_{SCN}.txt").read_text() == OOM_PREMARKED_REASON
        print("PASS: OOM-premarked tile -> nodata placeholder + skip reason")

        original = aqueduct_runner.run_aqueduct_python
        def _oom(*a, **k):
            raise MemoryError
        aqueduct_runner.run_aqueduct_python = _oom
        try:
            res, arr = _run(root, 4)
        finally:
            aqueduct_runner.run_aqueduct_python = original
        assert res == {"outcome": "oom"}, res
        assert (arr == WATERDEPTH_NODATA_INT16).all()
        assert (root / "mo" / "oom_tiles" / "4.txt").exists(), "MemoryError must mark the tile OOM"
        print("PASS: MemoryError during the solve -> tile marked OOM + nodata placeholder")


def main() -> None:
    test_all_outcomes()
    print("\nAll run_tile_scenario checks passed.")


if __name__ == "__main__":
    main()
