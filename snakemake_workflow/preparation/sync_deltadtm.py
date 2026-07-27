"""
Download DeltaDTM v1.1 tiles from 4TU.ResearchData and sort them into the
DEM/mask tile directories the rest of the pipeline expects — the parent
directories of the `deltadtm` / `deltadtm_mask` sources in
data_catalog_gfm.yml (e.g. merge_tiles.py, select_tiles.py and
extract_dem.py all read individual tile files from those same directories,
next to the `deltadtm.vrt` / `deltadtm_mask.vrt` mosaics).

Also downloads 4TU's own pre-built global DEM VRT mosaic and rewrites it to
point at this machine's local tile directory (see download_and_patch_vrt),
and builds a mask VRT mosaic locally from the extracted mask tiles (see
build_mask_vrt - no pre-built mask VRT exists from 4TU, unlike the DEM).
Both are saved directly to the exact paths `deltadtm.path`/`deltadtm_mask.
path` (data_catalog_gfm.yml) already expect, so nothing downstream needs to
know this script ran.

HOW TO GET THE URLS:
Go to https://data.4tu.nl/datasets/1da2e70f-6c4d-4b03-86bd-b53e789cc629
For each file below, right-click its download button/link -> "Copy Link Address"
and paste it into the dicts below. They look like:
https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/<file-uuid>

Only fill in the continents you actually need -- leave others as None
and they'll be skipped. Same for DEM_VRT_URL: leave it as None to skip
downloading/patching the VRT mosaic (deltadtm.path must then already exist
some other way - e.g. hand-built with gdalbuildvrt - before the rest of the
pipeline can read the `deltadtm` catalog source).

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py sync_deltadtm`).
"""

import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# 1) FILL IN YOUR COPIED LINKS HERE
# ---------------------------------------------------------------------------

DEM_ZIP_URLS = {
    "Africa": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/22ffa027-184b-4f67-9979-c182f3dfb1ab",
    "Antarctica": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/ca957a40-34fa-41eb-b101-e45d1ccbd890",
    "Asia": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/672eba4c-1334-44c6-8119-8879ded25912",
    "Europe": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/cb0b8ee3-b018-4828-a74e-2fb05020b1b6",
    "North_America": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/037664c6-1494-4889-9689-a56570728320",
    "Oceania": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/de972de1-26bd-4303-afdf-21a90a232cff",
    "South_America": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/db980f00-63cd-4a07-a4df-55ab06510594",
    "Seven_seas_(open_ocean)": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/fe986ba6-3db9-40e2-8a49-0fcdb341244a",
}

MASK_ZIP_URL = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/bfe0fbc1-fdf3-40d0-a62c-adc58bfd9478"  # mask_tiles.zip

# 4TU's own pre-built global VRT mosaic over every DEM tile (all continents).
# Its <SourceFilename> entries are bare filenames (e.g. "DeltaDTM_v1_1_
# N00E006.tif") - download_and_patch_vrt rewrites these to this machine's
# actual local tile paths, so it does not matter which continents (if any)
# were already extracted below.
DEM_VRT_URL = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/1892b825-3e68-4337-9b8c-03fcffe4588b"

# ---------------------------------------------------------------------------
# 2) CORE FUNCTIONS -- shouldn't need to touch below this line
# ---------------------------------------------------------------------------


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream a large file to disk with progress reporting, resuming if partially present."""
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)

    # Skip if already fully downloaded (best-effort check via Content-Length)
    head = requests.head(url, allow_redirects=True, timeout=30)
    expected_size = int(head.headers.get("Content-Length", 0))
    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        print(f"  Already downloaded: {dest_path.name} ({expected_size / 1e9:.2f} GB)")
        return

    print(f"  Downloading {dest_path.name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)", end="")
        print()  # newline after progress


def extract_tifs(zip_path: Path, out_dir: Path) -> int:
    """Extract every .tif file inside a zip (ignoring subfolder structure) into out_dir."""
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".tif"):
                continue
            filename = Path(member).name  # flatten any subfolder structure
            target = out_dir / filename
            if target.exists():
                continue  # skip if already extracted (e.g. re-running the script)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def extraction_marker_path(out_dir: Path, name: str) -> Path:
    """Path to the "already extracted" marker for zip `name` in `out_dir`.

    Needed because `download_file`'s own already-downloaded check compares
    against the local .zip, which no longer exists once
    delete_zip_after_extract has removed it - without a separate marker,
    every re-run would re-download a continent's multi-GB zip purely to
    re-extract .tif files that are already all there (extract_tifs already
    skips per-file, but that doesn't avoid the download itself).
    """
    return out_dir / f"{name}.extracted"


def _try_delete(path: Path, attempts: int = 5, delay_s: float = 2.0) -> bool:
    """Best-effort delete, retrying briefly on PermissionError.

    Windows can transiently hold a lock on a just-closed file (e.g. Defender/
    endpoint AV real-time-scanning a freshly downloaded/extracted zip) for a
    second or two after the `zipfile.ZipFile` context manager has released
    it. Returns False (rather than raising) once attempts are exhausted, so
    a lock that outlasts the retries costs disk space, not the whole
    preparation run - extraction has already succeeded and the `.extracted`
    marker is already written by the time this is called.
    """
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            return True
        except PermissionError:
            if attempt == attempts:
                return False
            time.sleep(delay_s)
    return False


def process_zip(
    name: str, url: str, zip_download_dir: Path, out_dir: Path, delete_zip_after_extract: bool,
) -> None:
    if not url:
        print(f"Skipping {name} (no URL provided)")
        return

    print(f"\n=== {name} ===")
    marker = extraction_marker_path(out_dir, name)
    if marker.exists():
        print(f"  Already extracted ({marker.name} found) - skipping download "
              f"(delete {marker} to re-download/re-extract)")
        return

    zip_path = zip_download_dir / f"{name}.zip"
    download_file(url, zip_path)

    print(f"  Extracting .tif files to {out_dir} ...")
    n = extract_tifs(zip_path, out_dir)
    print(f"  Extracted {n} file(s).")
    marker.write_text(f"{n} file(s) extracted to {out_dir}")

    if delete_zip_after_extract:
        if _try_delete(zip_path):
            print(f"  Deleted {zip_path.name} to save space.")
        else:
            print(f"  WARNING: {zip_path} is still locked by another process (e.g. antivirus) "
                  "- left on disk, safe to delete manually later.")


def download_and_patch_vrt(url: str, dest_path: Path, tile_dir: Path, zip_download_dir: Path) -> None:
    """Download 4TU's pre-built global DEM VRT mosaic and rewrite it for this machine.

    The downloaded VRT's <SourceFilename> entries are bare filenames (e.g.
    "DeltaDTM_v1_1_N00E006.tif", relativeToVRT="1") with no path information
    of their own. Every one is rewritten here to the tile's actual absolute
    path in `tile_dir` (relativeToVRT="0"), so the VRT resolves correctly
    regardless of where the .vrt file itself is saved - independent of
    relative-path resolution, and independent of which continents (if any)
    were actually extracted into `tile_dir`: GDAL only opens a given
    <SimpleSource> lazily when a read window actually touches its extent, so
    a reference to a tile you never downloaded is harmless until something
    actually queries that region.

    Args:
        url: Download link for the raw VRT (DEM_VRT_URL). Skipped (not
            failed) if empty/None, matching DEM_ZIP_URLS' per-continent
            skip behaviour.
        dest_path: Final path for the patched VRT - the exact path
            `deltadtm.path` already points at in data_catalog_gfm.yml, so
            every other script keeps reading from the same place.
        tile_dir: Directory the DEM .tif tiles are (or will be) extracted
            into (dem_out_dir) - the absolute paths are built against this.
        zip_download_dir: Scratch directory for the raw, unpatched download.
    """
    if not url:
        print("Skipping DEM VRT (no URL provided)")
        return

    if dest_path.exists():
        print(f"DEM VRT already exists at {dest_path}, skipping (delete it to re-download/re-patch)")
        return

    raw_path = zip_download_dir / "deltadtm_raw.vrt"
    print("\n=== DeltaDTM VRT mosaic ===")
    download_file(url, raw_path)

    tree = ET.parse(raw_path)
    root = tree.getroot()

    n_total = 0
    n_missing = 0
    for src_fn in root.iter("SourceFilename"):
        n_total += 1
        filename = Path(src_fn.text.strip()).name  # drop whatever path the source VRT itself carries
        local_path = tile_dir / filename
        src_fn.text = str(local_path)
        src_fn.set("relativeToVRT", "0")
        if not local_path.exists():
            n_missing += 1

    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)
    tree.write(dest_path, encoding="utf-8", xml_declaration=True)
    print(f"  Patched {n_total} tile reference(s) to absolute paths under {tile_dir}")
    if n_missing:
        print(
            f"  NOTE: {n_missing} referenced tile(s) not found locally yet (e.g. a continent "
            "you haven't downloaded) - harmless unless a later read touches that region."
        )
    print(f"  Wrote {dest_path}")


def build_mask_vrt(tile_dir: Path, dest_path: Path) -> None:
    """Build a VRT mosaic over the extracted DeltaDTM mask tiles.

    Unlike the DEM, 4TU doesn't host a pre-built mask VRT, so this builds
    one locally with GDAL's BuildVRT over whatever mask .tif tiles are
    present in `tile_dir` - handles the same per-latitude-band native
    resolution variation as the DEM VRT automatically (mosaics at the
    finest resolution found among the inputs). Values are categorical
    (0=land, 1=ocean, 2=lake, 3=river, 255=nodata), so resampling is
    nearest-neighbour.

    Args:
        tile_dir: Directory containing the extracted mask .tif tiles
            (mask_out_dir).
        dest_path: Final path for the VRT - the exact path
            `deltadtm_mask.path` already points at in data_catalog_gfm.yml.
    """
    if dest_path.exists():
        print(f"Mask VRT already exists at {dest_path}, skipping (delete it to rebuild)")
        return

    tif_paths = sorted(str(p) for p in tile_dir.glob("*.tif"))
    if not tif_paths:
        print(f"No mask tiles found in {tile_dir} - skipping mask VRT build")
        return

    print("\n=== DeltaDTM mask VRT mosaic ===")
    print(f"  Building VRT from {len(tif_paths)} mask tile(s)...")
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)
    vrt_options = gdal.BuildVRTOptions(VRTNodata=255, resampleAlg="nearest")
    ds = gdal.BuildVRT(str(dest_path), tif_paths, options=vrt_options)
    if ds is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {dest_path}")
    ds.FlushCache()
    ds = None
    print(f"  Wrote {dest_path}")


def run(config: dict) -> None:
    sync_cfg = config["sync_deltadtm"]

    # DEM/mask tiles land next to the `deltadtm`/`deltadtm_mask` VRT mosaics
    # in data_catalog_gfm.yml — the same directories merge_tiles.py,
    # select_tiles.py and tile_mask_creation.py already read individual tile
    # files from, so nothing downstream needs to know this script ran.
    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    dem_vrt_path = Path(catalog.get_source("deltadtm").path)
    dem_out_dir = dem_vrt_path.parent
    mask_vrt_path = Path(catalog.get_source("deltadtm_mask").path)
    mask_out_dir = mask_vrt_path.parent

    zip_download_dir = Path(sync_cfg["zip_download_dir"])
    delete_zips = bool(sync_cfg.get("delete_zips_after_extract", False))

    retry_transient_io(dem_out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(mask_out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(zip_download_dir.mkdir, parents=True, exist_ok=True)

    print(f"DEM tiles will be extracted to:  {dem_out_dir}")
    print(f"Mask tiles will be extracted to: {mask_out_dir}")

    print("\nDownloading DEM tiles (per continent) ...")
    for continent, url in DEM_ZIP_URLS.items():
        process_zip(continent, url, zip_download_dir, dem_out_dir, delete_zips)

    download_and_patch_vrt(DEM_VRT_URL, dem_vrt_path, dem_out_dir, zip_download_dir)

    print("\nDownloading mask tiles ...")
    process_zip("mask_tiles", MASK_ZIP_URL, zip_download_dir, mask_out_dir, delete_zips)

    build_mask_vrt(mask_out_dir, mask_vrt_path)

    print("\nDone.")
    print(f"DEM tiles:  {dem_out_dir.resolve()}")
    print(f"DEM VRT:    {dem_vrt_path.resolve() if dem_vrt_path.exists() else '(not downloaded)'}")
    print(f"Mask tiles: {mask_out_dir.resolve()}")
    print(f"Mask VRT:   {mask_vrt_path.resolve() if mask_vrt_path.exists() else '(not built)'}")
