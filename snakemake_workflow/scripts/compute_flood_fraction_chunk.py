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
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import retry_transient_io  # noqa: E402
from rasters import load_raster  # noqa: E402

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

    Processes the fine raster block-by-block (memory-safe), writes two
    intermediate rasters to a temp directory, then reprojects both to the
    population grid using AVERAGE downsampling.

    ff = sum(flooded fine pixels) / n_total_fine_pixels_in_cell

    This is achieved via two passes so that out-of-domain (nodata) fine pixels
    contribute 0 to the numerator and are counted in the denominator, without
    needing to map nodata to 0 in the exceedance raster:

        Pass A — binary exceedance, nodata preserved:
            reproject(exclude nodata) → A = sum(flooded) / n_in_domain

        Pass B — binary domain mask (1 = in domain, 0 = outside, no nodata):
            reproject(all values count) → B = n_in_domain / n_total

        ff = A × B = sum(flooded) / n_total

    Coarse cells inside the tile but entirely outside the domain give
    A = NaN, B = 0 → ff = NaN → nodata.
    """
    with TemporaryDirectory() as tmpdir:
        exc_path = Path(tmpdir) / "exc.tif"   # binary exceedance, nodata preserved
        dom_path = Path(tmpdir) / "dom.tif"   # domain mask (0/1, no nodata)

        # Step 1: write exceedance and domain mask simultaneously (single read pass)
        with retry_transient_io(rasterio.open, waterdepth_path) as wd:
            exc_profile = wd.profile.copy()
            exc_profile.update(dtype="float32", count=1, nodata=_NODATA_FINE,
                               compress="lzw", tiled=True, bigtiff="YES")
            dom_profile = wd.profile.copy()
            dom_profile.update(dtype="float32", count=1, compress="lzw",
                               tiled=True, bigtiff="YES")
            dom_profile.pop("nodata", None)  # no nodata: 0 = outside domain, 1 = inside

            with rasterio.open(exc_path, "w", **exc_profile) as exc_dst, \
                 rasterio.open(dom_path, "w", **dom_profile) as dom_dst:
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
                        exc = np.where(
                            valid & (depth > threshold_m), 1.0, _NODATA_FINE
                        ).astype("float32")
                        dom = np.where(valid, 1.0, 0.0).astype("float32")
                        exc_dst.write(exc, 1, window=window)
                        dom_dst.write(dom, 1, window=window)

        # Step 2: reproject both rasters to the coarse population grid
        pop_da = load_raster(population_path)
        out_h, out_w = pop_da.raster.height, pop_da.raster.width
        dst_kwargs = dict(
            dst_transform=pop_da.raster.transform, dst_crs=pop_da.raster.crs,
            dst_nodata=np.nan, resampling=Resampling.average,
        )

        # Pass A: sum(flooded) / n_in_domain  (nodata excluded)
        frac_a = np.full((out_h, out_w), np.nan, dtype="float32")
        with rasterio.open(exc_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=frac_a,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=_NODATA_FINE, **dst_kwargs,
            )

        # Pass B: n_in_domain / n_total  (0 = outside domain, counts in denominator)
        frac_b = np.full((out_h, out_w), np.nan, dtype="float32")
        with rasterio.open(dom_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=frac_b,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=None, **dst_kwargs,
            )

        # ff = A × B = sum(flooded) / n_total
        # NaN × anything = NaN → cells outside domain or tile remain nodata
        frac = frac_a * frac_b

    # Step 3: write coarse output
    out_frac = np.where(np.isnan(frac), _NODATA_COARSE, frac).astype("float32")
    coarse_profile = {
        "driver": "GTiff", "crs": pop_da.raster.crs,
        "transform": pop_da.raster.transform,
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
