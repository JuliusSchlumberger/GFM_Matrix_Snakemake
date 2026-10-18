"""validation/validate_sf_bay._read_model_bay: one chunk -> windowed read, several -> mosaic.

Synthetic: two adjacent 1-degree "chunk" rasters with different constant depths and a bbox
straddling their shared edge, plus the single-chunk case. Plain assert-based script (this
repo's tests/ convention); pytest collects it.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "validation")]
import validate_sf_bay as sfb  # noqa: E402


def _chunk(path: Path, west: float, value: float) -> None:
    res = 0.01  # 100 x 100 px per 1-degree chunk
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=1, height=100, width=100,
                       transform=Affine(res, 0, west, 0, -res, 38.0), crs="EPSG:4326", nodata=-9999.0) as dst:
        dst.write(np.full((100, 100), value, dtype="float32"), 1)


def test_single_and_multi_chunk_reads() -> None:
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.tif", Path(td) / "b.tif"
        _chunk(a, -123.0, 1.0)
        _chunk(b, -122.0, 2.0)

        depth, transform, _crs, nodata = sfb._read_model_bay([a], (-122.8, 37.5, -122.2, 37.9))
        assert depth.shape == (40, 60) and (depth == 1.0).all() and nodata == -9999.0, (depth.shape, nodata)
        assert np.isclose(transform.c, -122.8) and np.isclose(transform.f, 37.9)
        print("PASS: one chunk -> windowed read of exactly the bbox")

        depth, transform, _crs, nodata = sfb._read_model_bay([a, b], (-122.3, 37.5, -121.7, 37.9))
        assert depth.shape == (40, 60), depth.shape
        assert (depth[:, :30] == 1.0).all() and (depth[:, 30:] == 2.0).all(), "mosaic must keep each chunk's values"
        assert np.isclose(transform.c, -122.3) and nodata == -9999.0
        print("PASS: two chunks -> mosaic across the shared edge, each side from its own chunk")

        try:
            sfb._read_model_bay([], (-122.3, 37.5, -121.7, 37.9))
        except FileNotFoundError:
            print("PASS: no overlapping chunk -> clear FileNotFoundError")
        else:
            raise AssertionError("expected FileNotFoundError for no chunks")


def main() -> None:
    test_single_and_multi_chunk_reads()
    print("\nAll SF Bay model-read checks passed.")


if __name__ == "__main__":
    main()
