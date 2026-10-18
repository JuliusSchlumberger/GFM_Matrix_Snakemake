"""Build the DeltaDTM DEM/mask VRT mosaics with relative (portable) source paths.

Entry point: run(config), via `python run_preparation.py build_deltadtm_vrt`.
gdal.BuildVRT silently skips sources it transiently fails to open on the shared
data root (warning only, random per run) and mixes absolute/relative paths. So the
grid is pinned from filenames, missing tiles are retried in shrinking rounds plus
single-tile retries, the build hard-fails on any unrecovered tile, and sources
are rewritten relative and verified.
"""

import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, get_data_catalog, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

_NATIVE_RESOLUTION_DEG = 1.0 / 3600.0  # finest native DeltaDTM resolution (1 arcsec)

# {N|S}{lat}{E|W}{lon} is the SOUTH-WEST corner of a 1x1 degree cell (checked
# against N00E006, S69W092, N80E014 GeoTransforms).
_TILE_NAME_RE = re.compile(r"^DeltaDTM_v1_1_([NS])(\d{2})([EW])(\d{3})\.tif$", re.IGNORECASE)


def _tile_bounds(path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) degrees parsed from a DeltaDTM tile filename.

    Raises ValueError on a non-matching name rather than guessing bounds.
    """
    m = _TILE_NAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"{path.name} doesn't match the expected DeltaDTM tile naming convention "
            f"({_TILE_NAME_RE.pattern}) - refusing to guess its bounds."
        )
    ns, lat_digits, ew, lon_digits = m.groups()
    lat_south = int(lat_digits) * (1 if ns.upper() == "N" else -1)
    lon_west = int(lon_digits) * (1 if ew.upper() == "E" else -1)
    return (float(lon_west), float(lat_south), float(lon_west + 1), float(lat_south + 1))


# Real tiles sit half a native pixel outside their filename-implied edges (5 arcsec
# x-res at high latitude); pad so BuildVRT's outputBounds crop never clips a tile.
# Cost: every source gets a sub-pixel resample onto the shared grid.
_BOUNDS_PADDING_DEG = 0.01  # 36 arcsec - well over 2x any tile's native pixel size


def _mosaic_output_bounds(tif_paths: list[Path]) -> tuple[float, float, float, float]:
    """Return padded (west, south, east, north) degrees from filenames, no file opens."""
    boxes = [_tile_bounds(p) for p in tif_paths]
    return (
        min(b[0] for b in boxes) - _BOUNDS_PADDING_DEG,
        min(b[1] for b in boxes) - _BOUNDS_PADDING_DEG,
        max(b[2] for b in boxes) + _BOUNDS_PADDING_DEG,
        max(b[3] for b in boxes) + _BOUNDS_PADDING_DEG,
    )


def _assert_all_relative(vrt_path: Path) -> None:
    """Raise if any <SourceFilename> in vrt_path is not relativeToVRT="1"."""
    tree = retry_transient_io(ET.parse, vrt_path)
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


def _rewrite_sources_relative(tree: ET.ElementTree) -> None:
    """Set every <SourceFilename> to a bare filename with relativeToVRT="1".

    BuildVRT's own relative-path detection mixes "0"/"1" on a flaky network share;
    valid because tile_dir == dest_path.parent in run().
    """
    for src in tree.getroot().iter("SourceFilename"):
        src.text = Path(src.text.replace("\\", "/")).name
        src.set("relativeToVRT", "1")


def _run_build_vrt(
    paths: list[Path], vrt_kwargs: dict, out_vrt: Path,
) -> tuple[ET.ElementTree | None, list[ET.Element], set[str]]:
    """Run one gdal.BuildVRT into scratch out_vrt; return (tree, sources, basenames).

    tree is None if nothing was written. Source elements can be spliced into
    another VRT band because every call shares the pinned grid in vrt_kwargs.
    """
    warnings: list[str] = []

    def _collect_warning(err_class, err_num, msg):
        if err_class == gdal.CE_Warning:
            warnings.append(msg)

    gdal.PushErrorHandler(_collect_warning)
    try:
        ds = gdal.BuildVRT(
            str(out_vrt), [str(p) for p in paths], options=gdal.BuildVRTOptions(**vrt_kwargs)
        )
        if ds is not None:
            ds.FlushCache()
    finally:
        gdal.PopErrorHandler()
    ds = None

    if warnings:
        shown = warnings[:10]
        print(f"    GDAL warning(s) this call ({len(warnings)} total):")
        for w in shown:
            print(f"      {w}")
        if len(warnings) > len(shown):
            print(f"      ... and {len(warnings) - len(shown)} more")

    if not out_vrt.exists():
        return None, [], set()

    tree = retry_transient_io(ET.parse, out_vrt)
    band = tree.getroot().find(".//VRTRasterBand")
    src_elems = [el for el in list(band) if el.tag in ("ComplexSource", "SimpleSource")]
    names = set()
    for el in src_elems:
        fn_el = el.find("SourceFilename")
        names.add(Path(fn_el.text.replace("\\", "/")).name)
    return tree, src_elems, names


def _build_vrt(
    tile_dir: Path,
    dest_path: Path,
    resample_alg: str,
    vrt_nodata: float | None = None,
    max_rounds: int = 8,
    straggler_attempts: int = 4,
    retry_delay_s: float = 5.0,
) -> None:
    tif_paths = sorted(tile_dir.glob("*.tif"))
    if not tif_paths:
        print(f"  No tiles found in {tile_dir} - skipping {dest_path.name}")
        return
    expected_names = {p.name for p in tif_paths}

    print(f"  Building {dest_path.name} from {len(tif_paths)} tile(s)...")
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)

    output_bounds = _mosaic_output_bounds(tif_paths)
    vrt_kwargs = dict(
        resampleAlg=resample_alg,
        # Pin the grid (bounds from filenames, 1 arcsec) instead of
        # resolution="highest", which opens every file. DeltaDTM x-resolution
        # coarsens near the poles (3 arcsec at 76-77N, 5 at 80-81N); those
        # tiles are upsampled. A shared grid is what makes splicing sources
        # from separate BuildVRT calls safe.
        resolution="user",
        outputBounds=output_bounds,
        xRes=_NATIVE_RESOLUTION_DEG,
        yRes=_NATIVE_RESOLUTION_DEG,
        **({"VRTNodata": vrt_nodata} if vrt_nodata is not None else {}),
    )

    main_tree: ET.ElementTree | None = None
    main_band: ET.Element | None = None
    included: set[str] = set()
    remaining = list(tif_paths)

    with tempfile.TemporaryDirectory(prefix="deltadtm_vrt_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)

        # Each round retries only missing tiles: converges geometrically at any
        # transient failure rate, unlike retrying the whole list.
        for round_num in range(1, max_rounds + 1):
            if not remaining:
                break
            tree, src_elems, names = _run_build_vrt(
                remaining, vrt_kwargs, tmpdir / f"round{round_num}.vrt"
            )
            if main_tree is None and tree is not None:
                main_tree = tree
                main_band = main_tree.getroot().find(".//VRTRasterBand")
            elif tree is not None:
                for el in src_elems:
                    main_band.append(el)
            included |= names
            remaining = [p for p in remaining if p.name not in names]
            print(
                f"  [round {round_num}/{max_rounds}] {dest_path.name}: +{len(names)} tile(s) "
                f"this round -> {len(included)}/{len(tif_paths)} total, {len(remaining)} still missing."
            )

        # Stragglers: individual single-tile retries.
        if remaining:
            print(
                f"  {len(remaining)} tile(s) still missing after {max_rounds} round(s) - "
                f"retrying individually (up to {straggler_attempts} attempts each)..."
            )
            for p in remaining:
                for attempt in range(1, straggler_attempts + 1):
                    tree, src_elems, names = _run_build_vrt(
                        [p], vrt_kwargs, tmpdir / f"straggler_{p.stem}_{attempt}.vrt"
                    )
                    if names:
                        if main_tree is None:
                            main_tree = tree
                            main_band = main_tree.getroot().find(".//VRTRasterBand")
                        else:
                            main_band.append(src_elems[0])
                        included.add(p.name)
                        print(f"    recovered {p.name} on individual attempt {attempt}/{straggler_attempts}.")
                        break
                    if attempt < straggler_attempts:
                        time.sleep(retry_delay_s)
                else:
                    print(f"    {p.name}: could not be recovered after {straggler_attempts} individual attempts.")

    final_missing = sorted(expected_names - included)
    if final_missing:
        raise RuntimeError(
            f"{dest_path} is missing {len(final_missing)} of {len(tif_paths)} source "
            f"tile(s) even after {max_rounds} shrinking round(s) plus individual retries - "
            "gdal.BuildVRT silently drops any source tile it transiently fails to open "
            "instead of raising (see module docstring), and these tile(s) could not be "
            "recovered. This is a real, unrecoverable gap - do not use this VRT until "
            "it's fixed. Missing tile(s):\n" + "\n".join(f"  {n}" for n in final_missing)
        )

    assert main_tree is not None and main_band is not None  # final_missing empty => something succeeded
    _rewrite_sources_relative(main_tree)
    atomic_write(dest_path, lambda f: main_tree.write(f, encoding="unicode"), mode="w", encoding="utf-8")
    _assert_all_relative(dest_path)
    print(f"  Wrote {dest_path} ({len(tif_paths)}/{len(tif_paths)} tiles, verified: all source paths relative)")


def run(config: dict) -> None:
    """Rebuild the DeltaDTM DEM and mask VRTs with relative source paths.

    DEM: bilinear, embedded nodata. Mask: nearest, nodata=255.
    """
    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    dem_vrt_path = Path(catalog.get_source("deltadtm").path)
    mask_vrt_path = Path(catalog.get_source("deltadtm_mask").path)

    print("=== Building DeltaDTM VRT mosaics (relative source paths) ===")
    print(f"DEM tiles:  {dem_vrt_path.parent}")
    print(f"Mask tiles: {mask_vrt_path.parent}")

    # DEM tiles embed nodata -9999.0; BuildVRT inherits it.
    _build_vrt(dem_vrt_path.parent, dem_vrt_path, resample_alg="bilinear")

    # Categorical mask (0=land, 1=ocean, 2=lake, 3=river, 255=nodata): nearest.
    _build_vrt(mask_vrt_path.parent, mask_vrt_path, resample_alg="nearest", vrt_nodata=255)

    print("\nDone.")
    print(f"DEM VRT:  {dem_vrt_path.resolve() if dem_vrt_path.exists() else '(no DEM tiles found)'}")
    print(f"Mask VRT: {mask_vrt_path.resolve() if mask_vrt_path.exists() else '(no mask tiles found)'}")
