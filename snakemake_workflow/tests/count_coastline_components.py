import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flood_model import coastline_mask  # noqa: E402

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

tiles = ["1963", "1981", "1990", "3053", "2331", "2340", "2871", "2670", "2660", "26729"]
for t in tiles:
    p = Path(f"D:/GFM/model_outputs/{t}/inputs/mask.tif")
    with rasterio.open(p) as src:
        mask = src.read(1).astype(np.int8)
    coastline = coastline_mask(mask, ocean_code=1)
    labels, n = ndimage.label(coastline, structure=_STRUCTURE_8)
    river = int((mask == 3).sum())
    lake = int((mask == 2).sum())
    print(f"tile {t}: shape={mask.shape} cells={mask.size:,} "
          f"coastline_components={n:,} coastline_cells={int(coastline.sum()):,} "
          f"river_cells={river:,} ({100*river/mask.size:.2f}%) "
          f"lake_cells={lake:,} ({100*lake/mask.size:.2f}%)")
