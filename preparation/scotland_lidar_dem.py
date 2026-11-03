#!/usr/bin/env python3
"""
Download Scottish public-sector LiDAR DTM tiles from the public S3 mirror.

No AWS account/credentials needed (unsigned S3 REST access, same approach as
noaa_sfbay_lidar.py - no aws-cli/boto3 dependency). Five survey phases, two
tile conventions on the same OS National Grid (EPSG:27700):
  phase-1, phase-2: 1 m DTM, 10 km tiles,  "{grid}_1M_DTM_PHASE{n}.tif"        (e.g. HY20)
  phase-3..phase-5: 0.5 m DTM, 5 km tiles, "{grid}{quad}_50CM_DTM_PHASE{n}.tif" (e.g. NO10NE)
Outer Hebrides (separate prefix/convention, not handled here - geographically
irrelevant to this repo's Firth of Forth use).

Usage (standalone CLI):
  python scotland_lidar_dem.py list --phase phase-4
  python scotland_lidar_dem.py download --bbox-27700 280000 695000 300000 710000 --outdir forth_dtm
  python scotland_lidar_dem.py mosaic --indir forth_dtm --out forth_dtm.vrt

Usage (repo pipeline, Firth of Forth only - the region already defined in
scotland_sepa_coastal_m's own catalog entry - into
inputs/validation/Scotland/new_inputs/lidar_dem):
  python run_preparation.py scotland_lidar_dem --config snakemake_workflow/config/config.yml

Requires: requests; GDAL python bindings for `mosaic`.
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402
import validation as v  # noqa: E402

BUCKET = "https://srsp-open-data.s3.eu-west-2.amazonaws.com"
PHASES = ["phase-1", "phase-2", "phase-3", "phase-4", "phase-5"]
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # OS grid letters, matches ea_coastal_dtm.py (no I)
_GRID_RE = re.compile(r"^([A-Za-z]{2})(\d)(\d)([NS][EW])?_")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_BENCHMARK_KEY = "scotland_sepa_coastal_m"
_REGION = "firth_of_forth"


# ---------------------------------------------------------------- OS grid --

def grid_ref_bbox(filename: str) -> tuple[float, float, float, float] | None:
    """Decode a tile filename's OS grid reference into its EPSG:27700 bbox.

    Inverse of ea_coastal_dtm.py's os_tile_name: 10 km tile with no quadrant
    suffix (phase-1/2), or 5 km tile with an NE/NW/SE/SW suffix (phase-3..5).
    None if the filename doesn't start with a grid reference.
    """
    m = _GRID_RE.match(filename)
    if not m:
        return None
    l1, l2 = m.group(1)[0].upper(), m.group(1)[1].upper()
    e10, n10, quad = int(m.group(2)), int(m.group(3)), m.group(4)
    idx1, idx2 = LETTERS.index(l1), LETTERS.index(l2)
    e = (idx1 % 5) * 500_000 + (idx2 % 5) * 100_000 + e10 * 10_000 - 1_000_000
    n = (4 - idx1 // 5) * 500_000 + (4 - idx2 // 5) * 100_000 + n10 * 10_000 - 500_000
    size = 10_000
    if quad:
        size = 5_000
        if quad[0] == "N":
            n += 5_000
        if quad[1] == "E":
            e += 5_000
    return (e, n, e + size, n + size)


# -------------------------------------------------------------- S3 listing --

def list_keys(prefix: str, session: requests.Session | None = None) -> list[tuple[str, int]]:
    """All (key, size) under a prefix via unsigned ListObjectsV2."""
    session = session or requests.Session()
    keys, token = [], None
    print(f"listing s3://{prefix} ...", file=sys.stderr, flush=True)
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", NS):
            keys.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            print(f"  {len(keys)} keys", file=sys.stderr, flush=True)
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def select_by_bbox(keys: list[tuple[str, int]], bbox_27700: tuple[float, float, float, float]) -> list[tuple[str, int]]:
    """Keep tiles whose own grid-reference bbox (not an external tile index - none exists
    for this bucket) intersects bbox_27700."""
    from shapely.geometry import box

    aoi = box(*bbox_27700)
    sel = []
    for k, s in keys:
        gb = grid_ref_bbox(k.rsplit("/", 1)[-1])
        if gb is not None and box(*gb).intersects(aoi):
            sel.append((k, s))
    return sel


# --------------------------------------------------------------- download --

def fetch(session: requests.Session, key: str, dst: Path, size: int | None = None) -> str:
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return "skip"
    with session.get(f"{BUCKET}/{key}", stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
        tmp.rename(dst)
    return "ok"


def download(sel: list[tuple[str, int]], outdir: Path, workers: int = 6) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    total = sum(s for _, s in sel) / 1e9
    print(f"{len(sel)} files, {total:.2f} GB -> {outdir}", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, requests.Session(), k, outdir / Path(k).name, s): k for k, s in sel}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(sel)}] {f.result()} {Path(futs[f]).name}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(sel)}] FAIL {futs[f]}: {e}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------- mosaic --

def mosaic(indir: Path, out: Path) -> None:
    """One VRT over every downloaded tile; natural phase-1..5 filename sort means a
    later (higher-numbered, higher-resolution) phase's tile paints on top and wins
    on overlap, GDAL's BuildVRT default stacking order. resolution="highest" picks
    the finer 0.5 m (phase-3..5) pixel size where 1 m (phase-1/2) tiles also exist.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    tifs = sorted(str(p) for p in indir.rglob("*.tif"))
    print(f"building VRT from {len(tifs)} tiles -> {out}", file=sys.stderr, flush=True)
    gdal.BuildVRT(str(out), tifs, resolution="highest")
    print(f"-> {out}", flush=True)


# ------------------------------------------------------------- run(config) --

def run(config: dict) -> None:
    """Download Firth of Forth LiDAR DTM tiles (all 5 phases), mosaic to a VRT.

    AOI is the firth_of_forth region already defined in scotland_sepa_coastal_m's
    own catalog entry (data_catalog_validation.yml) - the same benchmark region
    this repo's Scotland validation already scores, not a hand-picked bbox.
    No tile index exists for this bucket (unlike NOAA's SF Bay one): each tile's
    footprint is instead decoded directly from its own OS grid reference filename
    (grid_ref_bbox), which is exact, not a buffer/approximation.
    Output: inputs/validation/Scotland/new_inputs/lidar_dem/ (tiles flat,
    firth_of_forth_dtm.vrt the final mosaic). Resumable: fetch() skips a file
    already on disk at its expected size.
    """
    val_cfg = config["validation"]
    bench_catalog = get_data_catalog(
        Path(__file__).resolve().parents[1] / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"],
    )
    v.fix_catalog_meta_encoding(bench_catalog, Path(__file__).resolve().parents[1] / val_cfg["benchmark_catalog"])
    spec = v.load_benchmark_spec(bench_catalog, _BENCHMARK_KEY)
    bbox_4326 = spec.regions[_REGION]

    from rasterio.warp import transform_bounds
    bbox_27700 = transform_bounds("EPSG:4326", "EPSG:27700", *bbox_4326)
    print(f"{_REGION} AOI: {bbox_4326} (EPSG:4326) -> {bbox_27700} (EPSG:27700)", file=sys.stderr, flush=True)

    out_root = Path(val_cfg["benchmark_root"]) / "Scotland" / "new_inputs" / "lidar_dem"
    retry_transient_io(out_root.mkdir, parents=True, exist_ok=True)

    session = requests.Session()
    for phase in PHASES:
        print(f"=== {phase} ===", file=sys.stderr, flush=True)
        keys = list_keys(f"lidar/{phase}/dtm/27700/gridded/", session)
        sel = select_by_bbox(keys, bbox_27700)
        print(f"  {len(sel)}/{len(keys)} tiles intersect {_REGION}", file=sys.stderr, flush=True)
        download(sel, out_root, workers=6)

    mosaic(out_root, out_root / "firth_of_forth_dtm.vrt")


# ------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list")
    l.add_argument("--phase", choices=PHASES, default="phase-1")

    d = sub.add_parser("download")
    d.add_argument("--phase", choices=PHASES, nargs="+", default=PHASES)
    d.add_argument("--bbox-27700", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="EPSG:27700 bbox; omit to download a phase in full")
    d.add_argument("--outdir", default="scotland_dtm")
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--dry", action="store_true", help="list selection only")

    m = sub.add_parser("mosaic")
    m.add_argument("--indir", default="scotland_dtm")
    m.add_argument("--out", default="scotland_dtm.vrt")

    args = ap.parse_args()
    if args.cmd == "mosaic":
        mosaic(Path(args.indir), Path(args.out))
        return

    session = requests.Session()
    if args.cmd == "list":
        keys = list_keys(f"lidar/{args.phase}/dtm/27700/gridded/", session)
        for k, s in keys:
            print(f"{s / 1e6:9.1f} MB  {k}")
        print(f"\n{len(keys)} tiles, {sum(s for _, s in keys) / 1e9:.2f} GB", file=sys.stderr)
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sel_all: list[tuple[str, int]] = []
    for phase in args.phase:
        keys = list_keys(f"lidar/{phase}/dtm/27700/gridded/", session)
        sel = select_by_bbox(keys, tuple(args.bbox_27700)) if args.bbox_27700 else keys
        print(f"  {phase}: {len(sel)}/{len(keys)} tiles selected", file=sys.stderr)
        sel_all += sel
    if args.dry:
        for k, s in sel_all:
            print(f"{s / 1e6:9.1f} MB  {k}")
        return
    download(sel_all, outdir, args.workers)


if __name__ == "__main__":
    main()
