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

Second failure mode found 2026-09-11, on a REAL production run of this exact
script: `gdal.BuildVRT()` over the full ~7417-tile DeltaDTM set silently
dropped 4447 of them (`deltadtm.vrt` ended up with only 2970
<SourceFilename> entries) and, independently, wrote a mix of
relativeToVRT="0" (absolute) and ="1" (relative) entries for the tiles it
did include - `_assert_all_relative` (below) caught the second symptom and
hard-failed, which is what surfaced this investigation. Root-caused with
real evidence, not by inspection alone: an instrumented rebuild (GDAL error
handler installed, same 7417 real tiles, written to a scratch VRT) captured
596 explicit GDAL warnings of the form `Can't open P:\...\DeltaDTM_v1_1_
S22W176.tif. Skipping it` - i.e. gdal.BuildVRT() treats a source it
transiently fails to open as "skip it and keep going", NOT as a hard error:
no exception, `ds` is NOT None, the call reports success, and the missing
source is just... absent from the output VRT, with only a warning that
nothing in this script previously looked at. A separate, standalone
open()-per-tile test of the 4447 tiles missing from that first production
run's VRT got `No such file or directory` on 494 of them even though every
one of those files is confirmed present via a plain directory listing -
classic transient-SMB-blip shape (this codebase already has
`retry_transient_io`, in `config_utils.py`, specifically for this failure
class elsewhere). Decisively ruling out "corrupt files" or "a hard
file-count/size limit": rerunning the exact same 7417-tile build a second
time (to scratch) dropped a *different* count of tiles (594, not 4447) at a
*completely different* cutoff point (`S22W175`/`S22W176`, not `N57W158`/
`N57W159`) - if any specific tile or a fixed limit were the cause, the same
tiles would fail the same way every time; instead it's random per run,
which is the signature of flaky P:\\ network I/O, not bad data. The
relative/absolute path split is presumed the same underlying cause wearing
a different hat (GDAL's relative-path-resolution logic, run per-source
during the same build loop, presumably also transiently failing/falling
back to an absolute path independently of the open-and-include decision) -
not independently proven to the same standard, which is exactly why this
script no longer trusts gdal.BuildVRT()'s own relative-path autodetection
at all (see `_rewrite_sources_relative` below) rather than trying to fully
understand or fix that GDAL-internal behaviour.

Fix, in `_build_vrt` below - NOT a simple "retry the whole 7417-tile call up
to N times": at whatever per-tile transient failure rate this share shows on
a given day (observed anywhere from ~8% to ~60% in real testing on
2026-09-11), there is no guarantee all 7417 tiles ever open successfully
*simultaneously* in one pass no matter how many times the whole thing is
retried - that would be gambling on an all-or-nothing event, not converging.
Instead this uses a SHRINKING round-based build: (1) an explicit output grid
(bounds + 1 arcsec resolution) is pinned up front, computed PURELY by
parsing all tile filenames (`DeltaDTM_v1_1_{N|S}{lat:02d}{E|W}{lon:03d}.tif`
names the south-west corner of an exact 1x1 degree cell - verified against 3
real tiles spanning the globe, see `_TILE_NAME_RE`/`_tile_bounds` below) -
zero file opens needed, so this step alone can't be affected by the flaky
share at all, and it means every `gdal.BuildVRT()` call below, no matter
which subset of tiles it's given, lands its sources on the exact same pixel
grid; (2) round 1 calls `gdal.BuildVRT()` over ALL tiles; whatever's missing
after that becomes the (much smaller) input to round 2, and so on for up to
`max_rounds` rounds - each round's <ComplexSource> XML elements are spliced
directly into one accumulating VRT (safe, because of the pinned grid from
(1): no coordinate translation is needed, the elements already describe
positions in the shared pixel space); this converges geometrically
regardless of the per-tile failure rate (e.g. even a pessimistic 60%-per-
round failure rate shrinks 7417 tiles to roughly 4450 -> 2670 -> 1600 -> ...
-> ~120 after 8 rounds); (3) any tiny handful of stragglers still missing
after `max_rounds` get individual single-tile `gdal.BuildVRT()` retries
(cheap at that point - single digits to low tens of files); (4) only if
tiles are STILL missing after that does this hard-fail, loudly, listing
every unrecovered tile by filename - a real, unrecoverable gap must never
again be silent or vague. Once a build is confirmed complete, every
<SourceFilename> is deterministically rewritten to a bare relative filename
with relativeToVRT="1" ourselves via ElementTree (`_rewrite_sources_relative`)
rather than trusting GDAL's own relative-path autodetection (shown
unreliable above) - unconditionally correct here because
`tile_dir == dest_path.parent` always holds by construction (see `run()`
below). `_assert_all_relative` is kept as a final verifying assertion AFTER
that rewrite - a real guarantee, not an assumption.

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py build_deltadtm_vrt`). Safe/
idempotent to re-run any time after sync_deltadtm has extracted tiles -
always rebuilds both VRTs from whatever .tif tiles are currently present
(unlike sync_deltadtm's tile-download idempotency, a VRT rebuild is cheap
enough that there's no reason to skip it just because a - possibly stale -
one already exists).
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

# The finest native DeltaDTM resolution (1 arcsec) - see the long comment in
# `_build_vrt` on why the whole global grid is pinned to this instead of
# gdal.BuildVRT's own resolution="highest" autodetection.
_NATIVE_RESOLUTION_DEG = 1.0 / 3600.0

# DeltaDTM_v1_1_{N|S}{lat:02d}{E|W}{lon:03d}.tif names the SOUTH-WEST corner
# of an exact 1x1 degree cell. Verified against 3 real tiles spanning the
# globe (their own GeoTransform origins, read directly off disk, 2026-09-11):
#   DeltaDTM_v1_1_N00E006.tif -> origin lon=5.9999/lat=1.0001 -> W=6,  N=1
#   DeltaDTM_v1_1_S69W092.tif -> origin lon=-92.0003/lat=-67.9999 -> W=-92, N=-68
#   DeltaDTM_v1_1_N80E014.tif -> origin lon=13.9993/lat=81.0001 -> W=14, N=81
# All three match "the two-letter+number pair is the south-west corner"
# exactly, including the high-latitude tile whose NATIVE x-resolution is
# coarser than 1 arcsec (that only affects pixel spacing within the cell,
# never the cell's own 1x1 degree footprint).
_TILE_NAME_RE = re.compile(r"^DeltaDTM_v1_1_([NS])(\d{2})([EW])(\d{3})\.tif$", re.IGNORECASE)


def _tile_bounds(path: Path) -> tuple[float, float, float, float]:
    """(west, south, east, north) degrees for one DeltaDTM tile, parsed
    purely from its filename - see the module-level naming-convention note
    above. Raises ValueError on any filename that doesn't match, rather
    than silently mis-parsing (a wrong bounds/grid assumption here would
    silently misplace tiles, which is worse than the bug this script
    exists to fix)."""
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


# Real tiles are NOT pixel-edge-aligned to the exact integer-degree grid
# `_tile_bounds` computes from filenames - confirmed directly (2026-09-11,
# quick 3-tile sanity check, real GeoTransforms): DeltaDTM_v1_1_N00E006.tif's
# own origin is (5.999861..., 1.000139...), i.e. its true west/north edges
# sit HALF A NATIVE PIXEL outside the "nominal" 6.0/1.0 filename-implied
# edges - and the same half-native-pixel outward offset was confirmed on
# S69W092 and N80E014 too (the latter's native x-resolution is coarser, 5
# arcsec, so ITS half-pixel offset is proportionally bigger in x). Padding
# `outputBounds` outward by more than any tile's largest possible
# half-pixel offset guarantees `gdal.BuildVRT`'s hard outputBounds crop
# never clips a real tile's true edge - the alternative (computing each
# tile's exact native resolution to correct for this precisely) would
# require opening every file, defeating the whole point of computing bounds
# from filenames alone. The cost is purely cosmetic: every source now needs
# a small sub-pixel resample against the shared grid instead of a handful of
# high-latitude ones - never a placement or data-loss issue, since tiles
# stay correctly adjacent to each other regardless (the offset is the same
# for all of them).
_BOUNDS_PADDING_DEG = 0.01  # 36 arcsec - well over 2x any tile's native pixel size


def _mosaic_output_bounds(tif_paths: list[Path]) -> tuple[float, float, float, float]:
    """(west, south, east, north) degrees covering every tile in `tif_paths`
    with a safety margin (see `_BOUNDS_PADDING_DEG`) - computed purely from
    filenames, zero file opens, so this step alone can never be affected by
    the flaky P:\\ share."""
    boxes = [_tile_bounds(p) for p in tif_paths]
    return (
        min(b[0] for b in boxes) - _BOUNDS_PADDING_DEG,
        min(b[1] for b in boxes) - _BOUNDS_PADDING_DEG,
        max(b[2] for b in boxes) + _BOUNDS_PADDING_DEG,
        max(b[3] for b in boxes) + _BOUNDS_PADDING_DEG,
    )


def _assert_all_relative(vrt_path: Path) -> None:
    """Hard-fail if any <SourceFilename> in `vrt_path` is not relativeToVRT="1".

    A real guarantee this step actually produced a portable VRT, not an
    assumption about gdal.BuildVRT's default behaviour - that default has
    already been silently wrong once for this exact file (see module
    docstring), so it's checked explicitly every time rather than trusted.
    Runs AFTER `_rewrite_sources_relative` has already deterministically
    rewritten every entry, so this should always pass in practice - kept as
    a belt-and-suspenders verification, not the primary correctness
    mechanism anymore.
    """
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
    """Mutate `tree` in place so every <SourceFilename> becomes a bare
    relative filename with relativeToVRT="1", regardless of what
    gdal.BuildVRT wrote for it (or which scratch directory it was actually
    built in - see `_build_vrt`).

    gdal.BuildVRT's own relative-path autodetection has been shown
    unreliable on this P:\\ share under transient network conditions (see
    module docstring: a real production build emitted a mix of
    relativeToVRT="0" and ="1" entries with no correlation to tile
    identity). Since `tile_dir == dest_path.parent` is guaranteed by
    construction in `run()` below, every source is unconditionally
    expressible as a bare relative filename - so instead of trusting
    GDAL's own heuristic (again), this rewrites it ourselves,
    deterministically, once, right before the final write.
    """
    for src in tree.getroot().iter("SourceFilename"):
        src.text = Path(src.text.replace("\\", "/")).name
        src.set("relativeToVRT", "1")


def _run_build_vrt(
    paths: list[Path], vrt_kwargs: dict, out_vrt: Path,
) -> tuple[ET.ElementTree | None, list[ET.Element], set[str]]:
    """Run one gdal.BuildVRT() call over `paths` into scratch file `out_vrt`
    (never `dest_path` itself - see `_build_vrt`), and report what actually
    made it in.

    Returns `(tree, source_elements, included_basenames)`: `tree` is None
    if gdal.BuildVRT produced no usable output at all this call (e.g. every
    source in `paths` failed to open this time); `source_elements` are the
    <ComplexSource>/<SimpleSource> XML elements gdal.BuildVRT DID include,
    ready to be spliced into another VRT's <VRTRasterBand> - safe to do
    blindly because every call in `_build_vrt` uses the SAME pinned
    outputBounds/resolution (`vrt_kwargs`), so every call's sources already
    sit in the identical pixel grid/coordinate space no matter which subset
    of tiles, or how many separate calls, produced them - no coordinate
    translation needed, just XML element re-parenting.
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
        # Pin the exact output grid ourselves instead of resolution="highest"
        # (which requires gdal.BuildVRT to open every file just to auto-detect
        # it) - outputBounds comes purely from filenames (see module
        # docstring), xRes/yRes is the known-constant finest native DeltaDTM
        # resolution (1 arcsec - DeltaDTM tiles do NOT all share one native
        # resolution: y stays a constant 1 arcsec, but x deliberately
        # coarsens near the poles to compensate longitude convergence, e.g. 3
        # arcsec at 76-77N, 5 arcsec at 80-81N - see src/tiles.py's own
        # docstring on this same fact - those coarser tiles just get
        # upsampled onto this grid same as before). Pinning this explicitly
        # (rather than auto-detecting) is what makes it safe to build
        # different SUBSETS of tiles across separate gdal.BuildVRT() calls
        # (the round-based retry below) and splice their sources together
        # afterwards: every call lands on the identical pixel grid.
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

        # Shrinking round-based build: each round only re-attempts what's
        # still missing, not all 7417 tiles again - this converges
        # geometrically regardless of the per-tile transient failure rate
        # (see module docstring), unlike retrying the whole list from
        # scratch every time (which gambles on all tiles opening
        # simultaneously in one pass).
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

        # Any tiny handful of stragglers left after the shrinking rounds get
        # individual single-tile retries - cheap at this point.
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
