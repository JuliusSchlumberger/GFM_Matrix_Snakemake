"""Build the deltadtm/deltadtm_mask VRT mosaics from locally-extracted DEM/
mask tiles, with RELATIVE (portable) source-file references.

Separate from sync_deltadtm.py (which only downloads/extracts .tif tiles)
so the VRT-build step can be re-run any time - cheap (seconds, no network)
- without re-downloading anything, and so it's the ONE place both VRTs get
built, instead of sync_deltadtm.py doing the DEM one its own (previously
buggy) way and the mask one a different, already-correct way.

Why this exists / what it replaces: sync_deltadtm.py used to download
4TU's own pre-built DEM VRT and patch its <SourceFilename> entries to this
machine's LOCAL ABSOLUTE tile paths (relativeToVRT="0", e.g.
"P:\\...\\DeltaDTM\\DeltaDTM_v1_1_N36E120.tif" on the Windows preprocessing
machine). That only resolves on the exact machine it was built on - it
broke every DeltaDTM read on the Linux HPC side, where "P:\\..." isn't a
valid path at all (confirmed directly: 27 of 31 HPC preprocessing jobs on
2026-08-08 failed with RasterioIOError on a DeltaDTM_v1_1_*.tif "No such
file or directory", one different tile per job, all via that same
absolute-Windows-path shape). The mask VRT never had this problem because
build_mask_vrt (moved here unchanged) always used plain gdal.BuildVRT()
without forcing absolute paths, which resolves to relative references by
default whenever the sources live under (or alongside) the VRT's own
directory - this script now does the DEM VRT the exact same, already-proven
way, and downloading/patching 4TU's own VRT is retired entirely (a locally
plain-built VRT from tiles that are downloaded either way carries the same
information, portably).

`_assert_all_relative` re-parses the written VRT and hard-fails if any
entry ended up non-relative, so a broken VRT is caught here - loudly, at
build time, on whichever machine ran this - rather than surfacing later as
a confusing per-tile file-not-found on a DIFFERENT machine.

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py build_deltadtm_vrt`). Safe/
idempotent to re-run any time after sync_deltadtm has extracted tiles -
always rebuilds both VRTs from whatever .tif tiles are currently present
(unlike sync_deltadtm's tile-download idempotency, a VRT rebuild is cheap
enough that there's no reason to skip it just because a - possibly stale -
one already exists).
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _assert_all_relative(vrt_path: Path) -> None:
    """Hard-fail if any <SourceFilename> in `vrt_path` is not relativeToVRT="1".

    A real guarantee this step actually produced a portable VRT, not an
    assumption about gdal.BuildVRT's default behaviour - that default has
    already been silently wrong once for this exact file (see module
    docstring), so it's checked explicitly every time rather than trusted.
    """
    tree = ET.parse(vrt_path)
    bad = [
        src.text for src in tree.getroot().iter("SourceFilename")
        if src.get("relativeToVRT") != "1"
    ]
    if bad:
        raise RuntimeError(
            f"{vrt_path} has {len(bad)} non-relative SourceFilename entr"
            f"{'y' if len(bad) == 1 else 'ies'} after building (first: {bad[0]!r}) - "
            "gdal.BuildVRT did not produce a portable VRT here. This is exactly the "
            "cross-platform breakage this script exists to prevent - do not use this "
            "VRT on another machine until fixed."
        )


def _build_vrt(
    tile_dir: Path, dest_path: Path, resample_alg: str, vrt_nodata: float | None = None,
) -> None:
    tif_paths = sorted(str(p) for p in tile_dir.glob("*.tif"))
    if not tif_paths:
        print(f"  No tiles found in {tile_dir} - skipping {dest_path.name}")
        return

    print(f"  Building {dest_path.name} from {len(tif_paths)} tile(s)...")
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg=resample_alg,
        **({"VRTNodata": vrt_nodata} if vrt_nodata is not None else {}),
    )
    ds = gdal.BuildVRT(str(dest_path), tif_paths, options=vrt_options)
    if ds is None:
        raise RuntimeError(
            f"gdal.BuildVRT returned None for {dest_path} - see GDAL warnings above "
            "for which source tile(s) it could not open."
        )
    ds.FlushCache()
    ds = None

    _assert_all_relative(dest_path)
    print(f"  Wrote {dest_path} (verified: all source paths relative)")


def run(config: dict) -> None:
    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    dem_vrt_path = Path(catalog.get_source("deltadtm").path)
    mask_vrt_path = Path(catalog.get_source("deltadtm_mask").path)

    print("=== Building DeltaDTM VRT mosaics (relative source paths) ===")
    print(f"DEM tiles:  {dem_vrt_path.parent}")
    print(f"Mask tiles: {mask_vrt_path.parent}")

    # DEM tiles carry their own embedded nodata (-9999.0, confirmed against
    # a real tile) - no VRTNodata override needed, gdal.BuildVRT inherits
    # each source's own value by default.
    _build_vrt(dem_vrt_path.parent, dem_vrt_path, resample_alg="bilinear")

    # Mask values are categorical (0=land, 1=ocean, 2=lake, 3=river,
    # 255=nodata) - nearest-neighbour resampling, explicit nodata (matches
    # the previous build_mask_vrt behaviour in sync_deltadtm.py exactly).
    _build_vrt(mask_vrt_path.parent, mask_vrt_path, resample_alg="nearest", vrt_nodata=255)

    print("\nDone.")
    print(f"DEM VRT:  {dem_vrt_path.resolve() if dem_vrt_path.exists() else '(no DEM tiles found)'}")
    print(f"Mask VRT: {mask_vrt_path.resolve() if mask_vrt_path.exists() else '(no mask tiles found)'}")
