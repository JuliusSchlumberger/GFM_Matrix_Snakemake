"""Build gebco.vrt from the 8 GEBCO 2026 quadrant GeoTIFFs in
inputs/gebco_2026_geotiff/, with a relative (portable) source reference -
same "verify what actually got included, don't trust gdal.BuildVRT()
silently" principle as preparation/build_deltadtm_vrt.py, but without that
script's elaborate round-based-retry machinery: GEBCO ships as 8 large
whole-globe-quadrant files, not ~7500 small tiles, so a single
retry_transient_io-wrapped gdal.BuildVRT() call is proportionate here.

Not part of the main production preparation pipeline (preparation/
run_preparation.py) - GEBCO is only used by this SFINCS first-pass
(sfincs_tiles/), never by the eikonal model - so this stays a standalone
script here rather than a registered preparation step.

Safe/idempotent to re-run any time after more GEBCO tiles are added.

Usage:
    python sfincs_tiles/build_gebco_vrt.py [--config <gfm config.yml>]
"""

import argparse
import sys
from pathlib import Path

from osgeo import gdal

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402


def build_gebco_vrt(tile_dir: Path, out_vrt: Path) -> None:
    tif_paths = sorted(tile_dir.glob("gebco_*_geotiff.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No GEBCO quadrant tiles found in {tile_dir}")
    print(f"Building {out_vrt.name} from {len(tif_paths)} tile(s) in {tile_dir}...")

    warnings: list[str] = []

    def _collect_warning(err_class, err_num, msg):
        if err_class == gdal.CE_Warning:
            warnings.append(msg)

    gdal.PushErrorHandler(_collect_warning)
    try:
        # GEBCO quadrants already tile a regular global grid at one uniform
        # native resolution (15 arcsec, confirmed identical on every tile
        # checked) - no outputBounds/xRes/yRes pinning needed the way
        # DeltaDTM's wildly-varying-resolution tiles required.
        ds = retry_transient_io(
            gdal.BuildVRT,
            str(out_vrt), [str(p) for p in tif_paths],
            options=gdal.BuildVRTOptions(resampleAlg="nearest"),
        )
        if ds is not None:
            ds.FlushCache()
    finally:
        gdal.PopErrorHandler()
    ds = None

    if warnings:
        print(f"  GDAL warning(s) ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"    {w}")

    if not out_vrt.exists():
        raise RuntimeError(f"gdal.BuildVRT produced no output at {out_vrt}")

    # Verify every source tile actually made it in - gdal.BuildVRT silently
    # drops a source it transiently fails to open instead of raising (same
    # failure mode documented in build_deltadtm_vrt.py's own module
    # docstring) - a real guarantee, not an assumption.
    import xml.etree.ElementTree as ET

    tree = retry_transient_io(ET.parse, out_vrt)
    included = {
        Path(el.find("SourceFilename").text.replace("\\", "/")).name
        for el in tree.getroot().iter()
        if el.tag in ("ComplexSource", "SimpleSource")
    }
    expected = {p.name for p in tif_paths}
    missing = sorted(expected - included)
    if missing:
        raise RuntimeError(
            f"{out_vrt} is missing {len(missing)}/{len(tif_paths)} source tile(s) after "
            f"building - gdal.BuildVRT silently dropped: {missing}. Do not use this VRT "
            "until fixed (re-run this script - transient P: share failures are the usual cause)."
        )

    # Rewrite every <SourceFilename> to a bare relative filename with
    # relativeToVRT="1" ourselves, same reasoning as build_deltadtm_vrt.py's
    # own _rewrite_sources_relative - do not trust gdal.BuildVRT's own
    # relative-path autodetection on this P: share.
    for el in tree.getroot().iter("SourceFilename"):
        el.text = Path(el.text.replace("\\", "/")).name
        el.set("relativeToVRT", "1")
    from config_utils import atomic_write
    atomic_write(out_vrt, lambda f: tree.write(f, encoding="unicode"), mode="w", encoding="utf-8")

    print(f"Wrote {out_vrt} ({len(tif_paths)}/{len(tif_paths)} tiles, all sources verified present + relative)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_dir = Path(cfg["paths"]["root"]) / "inputs" / "gebco_2026_geotiff"
    out_vrt = tile_dir / "gebco.vrt"
    build_gebco_vrt(tile_dir, out_vrt)


if __name__ == "__main__":
    main()
