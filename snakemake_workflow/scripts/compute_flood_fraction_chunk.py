"""Compute the coarse-resolution flood fraction for one (chunk, RP, SLR) scenario.

Reads the fine-resolution merged waterdepth chunk, applies the binary flood
threshold (depth > exceedance_threshold_m), and immediately average-pools the
result to the population raster's native ~1 km resolution.

Output: a tiny float32 raster (one value per ~1 km² population cell) where
each value is the fraction of fine Aqueduct pixels in that cell that are
flooded.  Values are in [0, 1]; nodata = -1.0.

This is the only spatial output needed downstream — the fine waterdepth raster
is no longer required after this step (marked temp() in postprocessing.smk and
deleted automatically by Snakemake).
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

    Processes the fine raster block-by-block (memory-safe) into an
    exceedance array (nodata outside the model domain) and a domain mask
    (0/1, no nodata), then hands both to `rasters.average_pool_to_grid` for
    the actual area-weighted reprojection - see that function's docstring
    for the full A×B derivation and why it MUST run as two separate
    `reproject()` calls (an earlier combined-call version silently
    corrupted ~360,000 real coarse cells - reverted, correctness over that
    specific speedup).

    ff = sum(flooded fine pixels) / n_total_fine_pixels_in_cell. Coarse
    cells inside the tile but entirely outside the domain end up NaN →
    written as this module's own `_NODATA_COARSE`.
    """
    # Step 1: build the exceedance and domain arrays in memory, block by block
    # (single read pass). Previously each block was written to an LZW GeoTIFF
    # in a temp dir and both files were then read back WHOLE for step 2 - no
    # memory saved, just an extra compress/decompress of ~2 full-size float32
    # rasters per job. Filling the arrays directly gives average_pool_to_grid
    # the exact same values (LZW is lossless; 1/0/_NODATA_FINE are exact in
    # float32) with the exact same transform/CRS (the temp files copied both
    # from the waterdepth chunk's own profile) - bit-identical output,
    # verified on all 90 thailand_bangkok (chunk, RP, SLR) scenarios.
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
                # valid-but-dry pixels MUST be 0.0, never _NODATA_FINE -
                # average_pool_to_grid's numerator contract (see its
                # own docstring in src/rasters.py) requires nodata to
                # mean ONLY "outside the domain"; marking dry pixels
                # as nodata too silently drops them from Pass A's
                # average instead of counting them as non-flooded,
                # inflating ff for any partially-flooded coarse cell
                # (confirmed on a real synthetic case, 2026-09: a
                # true fraction of 0.25 came out as 0.75 - not a
                # small error, and it scales with the domain/flood
                # ratio, so it's worse for less-flooded cells).
                exc_arr[row_off:row_off + bh, col_off:col_off + bw] = np.where(
                    valid, np.where(depth > threshold_m, 1.0, 0.0), _NODATA_FINE
                ).astype("float32")
                dom_arr[row_off:row_off + bh, col_off:col_off + bw] = np.where(valid, 1.0, 0.0).astype("float32")

    # Step 2: reproject both rasters to the coarse population grid.
    # Population grid metadata read directly via rasterio (no
    # xarray/rioxarray) since only height/width/transform/crs are
    # needed; profiling showed the xarray-backend-plugin-discovery
    # machinery (`load_raster`'s `xr.open_dataarray`, even with
    # `engine=` given explicitly) costs ~4s of pure import/reflection
    # overhead per process, unrelated to this raster's actual (tiny)
    # size - a real cost repeated on every one of this rule's many
    # fresh-process invocations. This part IS verified safe (metadata
    # only, no resampling/masking logic touched).
    with rasterio.open(population_path) as pop_src:
        out_h, out_w = pop_src.height, pop_src.width
        dst_transform, dst_crs = pop_src.transform, pop_src.crs

    # ff = A × B = sum(flooded) / n_total (average_pool_to_grid's own
    # "numerator"/"domain" split, extracted from this function's
    # original inline two-pass logic - src/rasters.py::average_pool_to_grid
    # for the shared implementation and why the two reproject() calls
    # must stay separate).
    frac = average_pool_to_grid(
        numerator=exc_arr, domain=dom_arr,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_shape=(out_h, out_w),
        numerator_nodata=_NODATA_FINE,
    )

    # Step 3: write coarse output
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


# ── Snakemake entry point ─────────────────────────────────────────────────────
compute_flood_fraction(
    waterdepth_path=snakemake.input.waterdepth,          # noqa: F821
    population_path=snakemake.input.population,          # noqa: F821
    threshold_m=snakemake.params.threshold_m,            # noqa: F821
    output_path=snakemake.output.flood_fraction,         # noqa: F821
    block_size=snakemake.params.block_size,              # noqa: F821
)
