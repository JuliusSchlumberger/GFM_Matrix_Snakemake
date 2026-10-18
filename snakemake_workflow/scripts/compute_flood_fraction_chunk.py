"""Compute the coarse-resolution flood fraction for one (chunk, RP, SLR) scenario.

Fraction of fine pixels with depth > exceedance_threshold_m per ~1 km population
cell; float32 in [0, 1], nodata = -1.0.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import retry_transient_io  # noqa: E402
from rasters import average_pool_to_grid  # noqa: E402

_NODATA_FINE = -1.0
_NODATA_COARSE = -1.0


def compute_flood_fraction(
    waterdepth_path: str | Path,
    population_path: str | Path,
    threshold_m: float,
    output_path: str | Path,
    block_size: int,
) -> None:
    """Write a coarse flood-fraction raster from a fine waterdepth chunk.

    ff = sum(flooded fine pixels) / n_total_fine_pixels_in_cell on
    `population_path`'s grid (rasters.average_pool_to_grid). A zero-byte
    `population_path` (chunk without population coverage) yields a zero-byte
    placeholder output. `threshold_m` is in metres.
    """
    if Path(population_path).stat().st_size == 0:
        retry_transient_io(Path(output_path).touch)
        return

    with retry_transient_io(rasterio.open, waterdepth_path) as wd:
        src_transform, src_crs = wd.transform, wd.crs
        exc_arr = np.empty((wd.height, wd.width), dtype="float32")  # binary exceedance, nodata preserved
        dom_arr = np.empty((wd.height, wd.width), dtype="float32")  # domain mask (0/1, no nodata)
        for row_off in range(0, wd.height, block_size):
            bh = min(block_size, wd.height - row_off)
            for col_off in range(0, wd.width, block_size):
                bw = min(block_size, wd.width - col_off)
                window = Window(col_off, row_off, bw, bh)
                depth = wd.read(1, window=window)
                valid = (
                    np.isfinite(depth) if wd.nodata is None
                    else (depth != wd.nodata)
                )
                # Valid-but-dry pixels MUST be 0.0, not nodata: average_pool_to_grid
                # treats nodata as "outside the domain", which would drop dry
                # pixels and inflate ff (e.g. a true 0.25 came out as 0.75).
                exc_arr[row_off:row_off + bh, col_off:col_off + bw] = np.where(
                    valid, np.where(depth > threshold_m, 1.0, 0.0), _NODATA_FINE
                ).astype("float32")
                dom_arr[row_off:row_off + bh, col_off:col_off + bw] = np.where(valid, 1.0, 0.0).astype("float32")

    # Plain rasterio, not xarray: backend-plugin discovery costs ~4 s per process.
    with rasterio.open(population_path) as pop_src:
        out_h, out_w = pop_src.height, pop_src.width
        dst_transform, dst_crs = pop_src.transform, pop_src.crs

    # ff = sum(flooded) / n_total; see average_pool_to_grid for why its two
    # reproject() calls must stay separate.
    frac = average_pool_to_grid(
        numerator=exc_arr, domain=dom_arr,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_shape=(out_h, out_w),
        numerator_nodata=_NODATA_FINE,
    )

    out_frac = np.where(np.isnan(frac), _NODATA_COARSE, frac).astype("float32")
    coarse_profile = {
        "driver": "GTiff", "crs": dst_crs,
        "transform": dst_transform,
        "width": out_w, "height": out_h,
        "count": 1, "dtype": "float32", "nodata": _NODATA_COARSE,
        "compress": "lzw",
    }
    with retry_transient_io(rasterio.open, output_path, "w", **coarse_profile) as dst:
        dst.write(out_frac, 1)


compute_flood_fraction(
    waterdepth_path=snakemake.input.waterdepth,          # noqa: F821
    population_path=snakemake.input.population,          # noqa: F821
    threshold_m=snakemake.params.threshold_m,            # noqa: F821
    output_path=snakemake.output.flood_fraction,         # noqa: F821
    block_size=snakemake.params.block_size,              # noqa: F821
)
