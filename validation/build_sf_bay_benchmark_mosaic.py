"""One-off: mosaic the 9 Bay Area counties' CoSMoS flood-depth rasters
(San Francisco, Alameda, Contra Costa, Marin, Napa, San Mateo, Santa Clara,
Solano, Sonoma - all RP100/no-SLR, v2.1, EPSG:26910, 2m, int32 centimeters,
nodata -9999) into one Bay-wide depth raster in METERS (float32), for
validate_sf_bay.py.

CoSMoS flood-depth values are documented in centimeters (confirmed via the
depth distribution: median ~1.8m, max ~19m is only physically plausible as
cm, not m, for a raster that also includes open Bay water depth, not just
new overland flooding - this benchmark was never meant to be depth-above-
ground only, see validate_sf_bay.py's own permanent-water exclusion for how
that's handled before scoring).

Usage:
    python validation/build_sf_bay_benchmark_mosaic.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

COUNTIES = [
    "san_francisco", "alameda", "contra_costa", "marin", "napa",
    "san_mateo", "santa_clara", "solano", "sonoma",
]
SRC_DIR = Path("P:/11212688-004-global-floodmaps/modelling/inputs/validation/California/flooding/v2.1")
OUT_PATH = Path("P:/11212688-004-global-floodmaps/modelling/inputs/validation/California/flooding/sf_bay_flddepth_slr000_w100_mosaic.tif")


def main() -> None:
    paths = [SRC_DIR / f"county_{c}_flddepth_slr000_w100.tif" for c in COUNTIES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing county rasters: {missing}")

    # Crop to the actual SF Bay extent (not each county's full extent, which
    # reaches far inland) - EPSG:26910 bounds of v2.1's sfb_fldhazd_slr100_w100.shp,
    # padded ~1km, confirmed this matches the real Bay footprint when checked earlier.
    bay_bounds = (530000.0, 4139000.0, 599500.0, 4238000.0)

    srcs = [rasterio.open(p) for p in paths]
    mosaic_cm, out_transform = merge(
        srcs, bounds=bay_bounds, nodata=-9999, resampling=rasterio.enums.Resampling.nearest,
    )
    crs = srcs[0].crs
    for s in srcs:
        s.close()

    mosaic_cm = mosaic_cm[0]  # single band
    valid = mosaic_cm != -9999
    mosaic_m = np.full(mosaic_cm.shape, np.nan, dtype="float32")
    mosaic_m[valid] = mosaic_cm[valid].astype("float32") / 100.0

    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": mosaic_m.shape[1], "height": mosaic_m.shape[0],
        "count": 1, "crs": crs, "transform": out_transform,
        "compress": "deflate",
    }
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(mosaic_m, 1)

    n_valid = int(valid.sum())
    print(f"wrote {OUT_PATH}")
    print(f"shape: {mosaic_m.shape}, crs: {crs}, n_valid_px: {n_valid}")
    wet = mosaic_m[valid][mosaic_m[valid] > 0]
    print(f"depth (m) over valid: min={np.nanmin(mosaic_m[valid]):.3f} max={np.nanmax(mosaic_m[valid]):.3f}")
    print(f"wet px (>0m): {wet.size} ({100 * wet.size / n_valid:.1f}% of valid)")


if __name__ == "__main__":
    main()
