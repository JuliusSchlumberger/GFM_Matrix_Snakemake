"""Native-CRS chunked merge of local-input tiles: chunk plan, batches and the VRT over them."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "preparation"))
import merge_england_local_tiles as driver  # noqa: E402
from merge import native_chunk_plan, open_depth  # noqa: E402
from rasters import save_waterdepth_raster  # noqa: E402

PX = 10.0  # m
CRS = "EPSG:27700"
# tile_id: (minx, maxy, cols, rows) on one 10 m grid; tiles 1/2 overlap by 50 cells, 3 stands apart.
TILES = {1: (400_000.0, 200_000.0, 300, 300), 2: (402_500.0, 200_000.0, 300, 300), 3: (410_000.0, 196_000.0, 120, 80)}
RASTER = {"driver": "GTiff", "compression": "zstd", "predictor": 3, "nodata": -9999}


def _depth(tid: int, rows: int, cols: int) -> np.ndarray:
    rng = np.random.default_rng(tid)
    return np.round(rng.uniform(0.0, 3.0, (rows, cols)) * (rng.random((rows, cols)) > 0.5), 2).astype("float32")


def _write_tiles(root: Path) -> dict[int, tuple[Affine, np.ndarray]]:
    out = {}
    for tid, (x0, y1, cols, rows) in TILES.items():
        d = root / "mo" / str(tid)
        (d / "inputs").mkdir(parents=True)
        (d / "results").mkdir()
        transform = Affine(PX, 0, x0, 0, -PX, y1)
        with rasterio.open(d / "inputs" / "dem.tif", "w", driver="GTiff", width=cols, height=rows, count=1,
                           dtype="float32", crs=CRS, transform=transform) as dst:
            dst.write(np.zeros((1, rows, cols), "float32"))
        depth = _depth(tid, rows, cols)
        save_waterdepth_raster(d / "inputs" / "dem.tif", depth, d / "results" / "waterdepth_RP100_SLR_0.tif")
        out[tid] = (transform, depth)
    return out


def _config(root: Path, chunk_m: float) -> dict:
    return {"simulation": {"model_outputs": str(root / "mo")}, "raster_format": RASTER,
            "postprocessing": {"merged_outputs": str(root / "merged"), "native_chunk_m": chunk_m, "block_size": 64}}


def test_chunk_plan_tiles_cover_their_chunks_without_overlap() -> None:
    bounds = {Path("a"): (0.0, 0.0, 2500.0, 1500.0), Path("b"): (2000.0, 0.0, 3000.0, 1000.0)}
    plan = native_chunk_plan(bounds, 1000.0)
    assert list(plan) == ["x0_y0", "x0_y1", "x1_y0", "x1_y1", "x2_y0", "x2_y1"]
    assert plan["x2_y0"] == ((2000.0, 0.0, 3000.0, 1000.0), [Path("a"), Path("b")])
    assert plan["x1_y1"][0] == (1000.0, 1000.0, 2000.0, 1500.0)  # clipped to the tiles' union
    assert plan["x2_y1"] == ((2000.0, 1000.0, 2500.0, 1500.0), [Path("a")])
    with pytest.raises(ValueError):
        native_chunk_plan(bounds, 0.0)


def test_batches_partition_the_chunks() -> None:
    plan = {f"c{i}": ((0, 0, 1, 1), [Path(str(k)) for k in range(i + 1)]) for i in range(7)}
    batches = [driver.batch_chunks(plan, k, 3) for k in range(3)]
    assert sorted(c for b in batches for c in b) == sorted(plan)
    with pytest.raises(ValueError):
        driver.batch_chunks(plan, 3, 3)


def test_vrt_over_chunks_is_the_per_cell_maximum_of_the_tiles(tmp_path) -> None:
    tiles = _write_tiles(tmp_path)
    config = _config(tmp_path, 2000.0)
    with pytest.raises(FileNotFoundError, match="chunk"):
        driver.run(config, ["RP100"], "SLR_0", vrt=True)
    written = [p for k in range(2) for p in driver.run(config, ["RP100"], "SLR_0", batch=k, n_batches=2)]
    assert len(written) == len(set(written)) > 2
    vrt = driver.run(config, ["RP100"], "SLR_0", vrt=True)[0]

    with rasterio.open(vrt) as raw:
        assert raw.dtypes[0] == "int16" and raw.scales == (0.01,)
    with open_depth(vrt) as src:
        assert src.crs.to_string() == CRS and src.res == (PX, PX)
        merged = src.read(1, masked=True)
        to_rc = ~src.transform
    expected = np.ma.masked_all(merged.shape, dtype="float64")
    for transform, depth in tiles.values():
        c0, r0 = (round(v) for v in to_rc * (transform.c, transform.f))
        window = expected[r0:r0 + depth.shape[0], c0:c0 + depth.shape[1]]
        expected[r0:r0 + depth.shape[0], c0:c0 + depth.shape[1]] = np.ma.where(
            window.mask, depth, np.maximum(window.filled(-np.inf), depth))
    assert (merged.mask == expected.mask).all(), "coverage differs from the tiles' union"
    np.testing.assert_allclose(merged.compressed(), expected.compressed(), atol=0.006)
    with rasterio.open(vrt) as src:
        assert len(src.files) - 1 == len(written)  # the VRT itself + one file per chunk


def test_missing_tile_output_fails(tmp_path) -> None:
    _write_tiles(tmp_path)
    (tmp_path / "mo" / "2" / "results" / "waterdepth_RP100_SLR_0.tif").unlink()
    with pytest.raises(FileNotFoundError, match="missing waterdepth"):
        driver.run(_config(tmp_path, 2000.0), ["RP100"], "SLR_0")
