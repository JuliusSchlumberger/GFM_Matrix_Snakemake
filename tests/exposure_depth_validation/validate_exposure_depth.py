"""Population weights and the depth-tolerance estimator of the exposure/depth assessment."""

import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exposure_depth import within_tolerance_share  # noqa: E402
from population_grid import population_per_pixel  # noqa: E402

RES = 30 / 3600


def test_population_weights_sum_to_the_cell_count(tmp_path) -> None:
    arr = np.full((24, 24), -99999, np.float32)
    arr[12, 12] = 1000
    path = tmp_path / "pop.tif"
    with rasterio.open(path, "w", driver="GTiff", height=24, width=24, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(4.9, 50.1, RES, RES), nodata=-99999) as dst:
        dst.write(arr, 1)
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform(
        4.9 + 12 * RES - 0.01, 50.1 - 12 * RES + 0.01)
    people, density = population_per_pixel(path, (120, 120), from_origin(x0, y0, 30, 30), "EPSG:32631")
    assert abs(people.sum() - 1000) < 20  # pixel-centre discretisation of one 0.55 km2 cell
    assert density.max() > 1000


def test_within_tolerance_share_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    for k in (0, 1, 2, 3):
        d = np.abs(k + rng.random(200_000) - rng.random(200_000)) * 0.1
        for tau in (0.1, 0.2, 0.3):
            assert abs(float(within_tolerance_share(np.array(float(k)), tau)) - (d <= tau).mean()) < 0.01
