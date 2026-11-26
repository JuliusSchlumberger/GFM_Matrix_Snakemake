"""Merged water depth as int16 cm: what merge writes, and readers seeing metres for either encoding."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))
from _chunks import _mosaic_read  # noqa: E402
from merge import MERGED_NODATA_M, WATERDEPTH_NODATA_INT16, merge_tile_rasters_chunk, open_depth  # noqa: E402
from pipeline_plots import compute_flood_area_km2  # noqa: E402
from rasters import save_waterdepth_raster  # noqa: E402

RES = 0.001  # deg
RASTER = {"driver": "GTiff", "compression": "zstd", "predictor": 3, "nodata": -9999}
# tile_id: (west, north, cols, rows) on one EPSG:4326 grid; 1 and 2 overlap by 20 columns.
TILES = {1: (10.0, 50.1, 60, 50), 2: (10.04, 50.1, 60, 50)}
CHUNK = (10.0, 50.0, 10.12, 50.12)  # wider and taller than the tiles: uncovered rows and columns


def _tiles(root: Path) -> dict[int, tuple[Affine, np.ndarray, Path]]:
    out = {}
    for tid, (west, north, cols, rows) in TILES.items():
        d = root / "mo" / str(tid) / "results"  # the merge takes the tile_id from this layout
        d.mkdir(parents=True)
        transform = Affine(RES, 0, west, 0, -RES, north)
        with rasterio.open(d / "dem.tif", "w", driver="GTiff", width=cols, height=rows, count=1, dtype="float32",
                           crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.zeros((1, rows, cols), "float32"))
        rng = np.random.default_rng(tid)
        depth = (np.round(rng.uniform(0, 4, (rows, cols)), 2) * (rng.random((rows, cols)) > 0.4)).astype("float32")
        path = d / "waterdepth_RP100_SLR_0.tif"
        save_waterdepth_raster(d / "dem.tif", depth, path)
        out[tid] = (transform, depth, path)
    return out


def _merge(root: Path) -> tuple[Path, Path, dict]:
    tiles = _tiles(root)
    wd, prov = root / "waterdepth_chunk.tif", root / "provenance_chunk.tif"
    merge_tile_rasters_chunk([p for _, _, p in tiles.values()], CHUNK, wd, prov, 32, RASTER)
    return wd, prov, tiles


def test_chunk_merge_writes_int16_cm_that_decodes_to_the_tile_maximum(tmp_path) -> None:
    wd, prov, tiles = _merge(tmp_path)
    with rasterio.open(wd) as raw, rasterio.open(prov) as p:
        assert (raw.dtypes[0], raw.nodata, raw.scales, raw.units) == ("int16", WATERDEPTH_NODATA_INT16, (0.01,), ("m",))
        assert p.dtypes[0] == "int16"
        provenance = p.read(1)
    with open_depth(wd) as src:
        assert (src.nodata, src.dtypes[0], src.profile["dtype"]) == (MERGED_NODATA_M, "float32", "float32")
        merged = src.read(1)
        to_rc = ~src.transform
    expected = np.full(merged.shape, MERGED_NODATA_M, dtype="float32")
    for transform, depth, _ in tiles.values():
        c0, r0 = (round(v) for v in to_rc * (transform.c, transform.f))
        window = expected[r0:r0 + depth.shape[0], c0:c0 + depth.shape[1]]
        window[:] = np.where(window == MERGED_NODATA_M, depth, np.maximum(window, depth))
    np.testing.assert_array_equal(merged, expected)
    assert ((provenance == -1) == (merged == MERGED_NODATA_M)).all()


def test_depth_raster_passes_float_files_through(tmp_path) -> None:
    data = np.array([[0.0, 1.25], [-9999.0, 3.5]], dtype="float32")
    path = tmp_path / "f32.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32", nodata=-9999.0,
                       crs="EPSG:4326", transform=Affine(RES, 0, 0, 0, -RES, 0)) as dst:
        dst.write(data, 1)
    with open_depth(path) as src:
        assert src.nodata == -9999.0
        np.testing.assert_array_equal(src.read(1), data)


def test_boundless_and_masked_reads_decode(tmp_path) -> None:
    wd, _, _ = _merge(tmp_path)
    with open_depth(wd) as src:
        window = rasterio.windows.Window(-5, -5, 20, 20)
        out = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        assert (out[:5, :] == MERGED_NODATA_M).all()
        masked = src.read(1, masked=True)
        assert masked.mask.sum() == (src.read(1) == MERGED_NODATA_M).sum()


def test_mosaic_and_flood_area_agree_with_the_float32_equivalent(tmp_path) -> None:
    wd, _, _ = _merge(tmp_path)
    with open_depth(wd) as src:
        depth, profile = src.read(1), src.profile
    f32 = tmp_path / "waterdepth_f32.tif"
    with rasterio.open(f32, "w", **{k: v for k, v in profile.items() if k != "predictor"}) as dst:
        dst.write(depth, 1)
    for threshold in (0.0, 0.1, 1.0):
        assert compute_flood_area_km2(wd, threshold) == pytest.approx(compute_flood_area_km2(f32, threshold), rel=1e-12)
    m_int, t_int = _mosaic_read([wd], [10.0, 50.0, 10.12, 50.12])
    m_f32, t_f32 = _mosaic_read([f32], [10.0, 50.0, 10.12, 50.12])
    assert t_int == t_f32
    np.testing.assert_array_equal(m_int, m_f32)
    with pytest.raises(ValueError, match="mixed"):
        _mosaic_read([wd, f32], [10.0, 50.0, 10.12, 50.12])
