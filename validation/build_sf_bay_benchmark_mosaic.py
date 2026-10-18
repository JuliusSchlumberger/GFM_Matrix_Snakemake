"""Mosaic the 9 Bay Area counties' CoSMoS depth rasters into one raster in meters.

Input: CoSMoS v2.1, RP100/no-SLR, EPSG:26910, 2m, int32 centimeters, nodata -9999.
Output: float32 meters, NaN nodata. Includes open Bay water depth, not only overland
flooding; validate_sf_bay.py excludes permanent water before scoring.

Usage:
    python validation/build_sf_bay_benchmark_mosaic.py [--config path/to/config.yml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import load_config  # noqa: E402

COUNTIES = [
    "san_francisco", "alameda", "contra_costa", "marin", "napa",
    "san_mateo", "santa_clara", "solano", "sonoma",
]


def main(src_dir: Path | None = None, out_path: Path | None = None, config_path: str | Path | None = None) -> None:
    """Mosaic the county rasters; paths default to ``validation.benchmark_root``."""
    if src_dir is None or out_path is None:
        cfg = load_config(config_path or _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
        benchmark_root = Path(cfg["validation"]["benchmark_root"])
        if src_dir is None:
            src_dir = benchmark_root / "California" / "flooding" / "v2.1"
        if out_path is None:
            out_path = benchmark_root / "California" / "flooding" / "sf_bay_flddepth_slr000_w100_mosaic.tif"

    paths = [src_dir / f"county_{c}_flddepth_slr000_w100.tif" for c in COUNTIES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing county rasters: {missing}")

    # Counties reach far inland; crop to EPSG:26910 bounds of v2.1's
    # sfb_fldhazd_slr100_w100.shp, padded ~1km.
    bay_bounds = (530000.0, 4139000.0, 599500.0, 4238000.0)

    srcs = [rasterio.open(p) for p in paths]
    mosaic_cm, out_transform = merge(
        srcs, bounds=bay_bounds, nodata=-9999, resampling=rasterio.enums.Resampling.nearest,
    )
    crs = srcs[0].crs
    for s in srcs:
        s.close()

    mosaic_cm = mosaic_cm[0]
    valid = mosaic_cm != -9999
    mosaic_m = np.full(mosaic_cm.shape, np.nan, dtype="float32")
    mosaic_m[valid] = mosaic_cm[valid].astype("float32") / 100.0

    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": mosaic_m.shape[1], "height": mosaic_m.shape[0],
        "count": 1, "crs": crs, "transform": out_transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic_m, 1)

    n_valid = int(valid.sum())
    print(f"wrote {out_path}")
    print(f"shape: {mosaic_m.shape}, crs: {crs}, n_valid_px: {n_valid}")
    wet = mosaic_m[valid][mosaic_m[valid] > 0]
    print(f"depth (m) over valid: min={np.nanmin(mosaic_m[valid]):.3f} max={np.nanmax(mosaic_m[valid]):.3f}")
    print(f"wet px (>0m): {wet.size} ({100 * wet.size / n_valid:.1f}% of valid)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()
    main(config_path=args.config)
