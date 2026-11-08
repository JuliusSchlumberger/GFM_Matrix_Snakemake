#!/usr/bin/env python3
"""Reproject the downloaded USGS 3DEP 10 m DEM to SF Bay's working grid.

Source: USGS 3D Elevation Program 1/3 arc-second (10 m) DEM, distributed by
OpenTopography (DOI 10.5069/G98K778D) - downloaded by hand (requires a personal
OpenTopography API key), not fetched by this script. Input is NAD83 geographic
(EPSG:4269), nodata -999999.0; output matches the repo's existing SF Bay grid
convention (EPSG:26910, metres) used by build_sf_bay_fine_tiles.py and the
local-input catalog.

Usage (repo pipeline, into inputs/validation/California/new_inputs/lidar_dem):
  python run_preparation.py sf_bay_usgs10m_dem --config snakemake_workflow/config/config.yml

Usage (standalone CLI):
  python sf_bay_usgs10m_dem.py --in output_USGS10m.tif --out sfbay_dem_usgs10m.tif
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import retry_transient_io  # noqa: E402

_SRC_NODATA = -999999.0
_DST_EPSG = 26910
_DST_RES_M = 10.0


def reproject(src_path: Path, dst_path: Path, epsg: int = _DST_EPSG, res: float = _DST_RES_M) -> None:
    """Warp the raw USGS10m download (NAD83 geographic) to EPSG:`epsg` at `res` m."""
    from osgeo import gdal

    gdal.UseExceptions()
    print(f"warping {src_path.name} -> EPSG:{epsg} @ {res} m ...", file=sys.stderr, flush=True)
    retry_transient_io(dst_path.parent.mkdir, parents=True, exist_ok=True)
    gdal.Warp(
        str(dst_path), str(src_path), dstSRS=f"EPSG:{epsg}", xRes=res, yRes=res,
        resampleAlg="bilinear", srcNodata=_SRC_NODATA, dstNodata=_SRC_NODATA,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
        callback=gdal.TermProgress_nocb,
    )
    print(f"-> {dst_path}  (EPSG:{epsg}, {res} m, nodata {_SRC_NODATA})", flush=True)


def run(config: dict) -> None:
    """Reproject `output_USGS10m.tif` (already downloaded) to `sfbay_dem_usgs10m.tif`.

    Both paths live in inputs/validation/California/new_inputs/lidar_dem, alongside
    the old NOAA-derived sfbay_dem_m.tif (left untouched - the catalog entry is what
    switches production over to this file, not an overwrite in place).
    """
    root = Path(config["validation"]["benchmark_root"])
    lidar_dir = root / "California" / "new_inputs" / "lidar_dem"
    src_path = lidar_dir / "output_USGS10m.tif"
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} not found - download the USGS 3DEP 10m DEM for the SF Bay region "
            "from OpenTopography (DOI 10.5069/G98K778D) first; this step only reprojects "
            "what is already on disk."
        )
    dst_path = lidar_dir / "sfbay_dem_usgs10m.tif"
    reproject(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", required=True, help="raw USGS10m download (NAD83 geographic)")
    parser.add_argument("--out", required=True, help="reprojected output path")
    parser.add_argument("--epsg", type=int, default=_DST_EPSG)
    parser.add_argument("--res", type=float, default=_DST_RES_M)
    args = parser.parse_args()
    reproject(Path(args.src), Path(args.out), args.epsg, args.res)


if __name__ == "__main__":
    main()
